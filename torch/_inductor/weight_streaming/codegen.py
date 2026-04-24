from __future__ import annotations

import bisect
import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from torch._inductor.codegen.wrapper import WrapperLine
from torch._inductor.weight_streaming.plan import (
    ColdStartOp,
    EvictDramOp,
    EvictVramOp,
    H2DPrefetchOp,
    IOSchedule,
    PrefetchOp,
)

if TYPE_CHECKING:
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.utils import IndentedBuffer

# Pre-kernel ops: SSD prefetch, H2D prefetch
PreKernelOp = Union[PrefetchOp, H2DPrefetchOp]
# Post-kernel ops: VRAM evict, DRAM evict
PostKernelOp = Union[EvictVramOp, EvictDramOp]

# GPU node name patterns: CUDA kernels, memory copies, NCCL collectives
_GPU_NAME_PREFIXES = (
    "triton_",
    "cublas",
    "Memcpy",
    "cudaMemcpy",
    "nccl",
    "cublasLt",
    "cusparse",
    "cudnn",
    "cutlass",
)

# Authoritative resource_kind values that indicate GPU execution
_GPU_RESOURCE_KINDS = frozenset({"gpu_stream", "gpu"})


def _is_gpu_node(node: dict) -> bool:
    """Classify a trace node as GPU or CPU."""
    rk = node.get("resource_kind", "")
    if rk:
        return rk in _GPU_RESOURCE_KINDS
    # Fallback heuristic based on node name
    name = node.get("name", "")
    return name.startswith(_GPU_NAME_PREFIXES)


class ScheduleAdapter:
    """Converts scheduler trace indices to wrapper kernel indices.

    The scheduler's trace has interleaved CPU+GPU nodes (e.g. 11,130 total).
    The Inductor wrapper only emits GPU kernel calls. This adapter builds the
    mapping and re-indexes prefetch/evict operations accordingly.

    For prefetches targeting CPU nodes, we snap to the next GPU kernel (earlier
    prefetch = conservative). For evictions targeting CPU nodes, we snap to the
    previous GPU kernel (later eviction = conservative).
    """

    def __init__(
        self, schedule: IOSchedule, current_graph_id: int = -1,
    ) -> None:
        self._schedule = schedule
        self._current_graph_id = current_graph_id
        self._gpu_trace_indices: list[int] = []
        self._trace_to_wrapper: dict[int, int] = {}
        self._all_indices: set[int] = set()

        for node in schedule.nodes:
            idx = node.get("idx", -1)
            self._all_indices.add(idx)
            if _is_gpu_node(node):
                self._gpu_trace_indices.append(idx)

        self._gpu_trace_indices.sort()
        for wrapper_idx, trace_idx in enumerate(self._gpu_trace_indices):
            self._trace_to_wrapper[trace_idx] = wrapper_idx

        # Pre-kernel: SSD prefetches + H2D prefetches (both keyed by before_node).
        # Only include ops WITHOUT launch IDs; ops with launch IDs are served
        # by before_launch and placed by exact compiled_launch_id.
        self._before_kernel: dict[int, list[PreKernelOp]] = defaultdict(list)
        for op in schedule.prefetches:
            if op.before_launch_id >= 0:
                continue
            widx = self._snap_to_gpu_next(op.before_node)
            if widx is not None:
                self._before_kernel[widx].append(op)
        for op in schedule.h2d_prefetches:
            if op.before_launch_id >= 0:
                continue
            widx = self._snap_to_gpu_next(op.before_node)
            if widx is not None:
                self._before_kernel[widx].append(op)

        # Post-kernel: VRAM evictions + DRAM evictions (both keyed by after_node).
        # Same filter: only ops WITHOUT launch IDs.
        self._after_kernel: dict[int, list[PostKernelOp]] = defaultdict(list)
        for op in schedule.evict_vram:
            if op.after_launch_id >= 0:
                continue
            widx = self._snap_to_gpu_prev(op.after_node)
            if widx is not None:
                self._after_kernel[widx].append(op)
        for op in schedule.evict_dram:
            if op.after_launch_id >= 0:
                continue
            widx = self._snap_to_gpu_prev(op.evict_after_node)
            if widx is not None:
                self._after_kernel[widx].append(op)

        self._startup_ops = list(schedule.cold_starts)

    def _snap_to_gpu_next(self, trace_idx: int) -> int | None:
        """Map a trace index to wrapper kernel index, snapping CPU nodes forward."""
        if trace_idx in self._trace_to_wrapper:
            return self._trace_to_wrapper[trace_idx]
        pos = bisect.bisect_left(self._gpu_trace_indices, trace_idx)
        if pos < len(self._gpu_trace_indices):
            return pos
        return None

    def _snap_to_gpu_prev(self, trace_idx: int) -> int | None:
        """Map a trace index to wrapper kernel index, snapping CPU nodes backward."""
        if trace_idx in self._trace_to_wrapper:
            return self._trace_to_wrapper[trace_idx]
        pos = bisect.bisect_right(self._gpu_trace_indices, trace_idx) - 1
        if pos >= 0:
            return pos
        return None

    def rescale_before_kernel(
        self, n_wrapper_kernels: int
    ) -> dict[int, list[PreKernelOp]]:
        """Re-index before_kernel ops to match the actual wrapper kernel count.

        The scheduler trace may have many more GPU nodes than Inductor wrapper
        kernels (due to fusion). This rescales the adapter's GPU-node indices
        to the [0, n_wrapper_kernels) range via linear interpolation.
        """
        if not self._gpu_trace_indices or n_wrapper_kernels <= 0:
            return {}
        n_gpu = len(self._gpu_trace_indices)
        result: dict[int, list[PreKernelOp]] = defaultdict(list)
        for gpu_idx, ops in self._before_kernel.items():
            wrapper_idx = min(
                gpu_idx * n_wrapper_kernels // n_gpu, n_wrapper_kernels - 1
            )
            result[wrapper_idx].extend(ops)
        return result

    def rescale_after_kernel(
        self, n_wrapper_kernels: int
    ) -> dict[int, list[PostKernelOp]]:
        """Re-index after_kernel ops to match the actual wrapper kernel count."""
        if not self._gpu_trace_indices or n_wrapper_kernels <= 0:
            return {}
        n_gpu = len(self._gpu_trace_indices)
        result: dict[int, list[PostKernelOp]] = defaultdict(list)
        for gpu_idx, ops in self._after_kernel.items():
            wrapper_idx = min(
                gpu_idx * n_wrapper_kernels // n_gpu, n_wrapper_kernels - 1
            )
            result[wrapper_idx].extend(ops)
        return result

    @property
    def before_kernel(self) -> dict[int, list[PreKernelOp]]:
        """wrapper_kernel_idx -> ops to execute before that kernel."""
        return self._before_kernel

    @property
    def after_kernel(self) -> dict[int, list[PostKernelOp]]:
        """wrapper_kernel_idx -> ops to execute after that kernel."""
        return self._after_kernel

    @property
    def startup_ops(self) -> list[ColdStartOp]:
        """Ops to execute before any kernel (cold start prefetches)."""
        return self._startup_ops

    def _is_consumer_here(self, op) -> bool:
        """Current wrapper is the op's CONSUMER graph (emits wait or sync)."""
        cgid = getattr(op, "compiled_graph_id", -1)
        if self._current_graph_id < 0 or cgid < 0:
            return True
        return cgid == self._current_graph_id

    def _is_issuer_here(self, op) -> bool:
        """Current wrapper is the op's ISSUE graph (emits async start).

        Falls back to ``compiled_graph_id`` when ``issue_compiled_graph_id``
        is unset so same-graph async ops (legacy schedules) keep working.
        """
        cgid = getattr(op, "compiled_graph_id", -1)
        issue_gid = getattr(op, "issue_compiled_graph_id", -1)
        if issue_gid < 0:
            issue_gid = cgid
        if self._current_graph_id < 0 or issue_gid < 0:
            return True
        return issue_gid == self._current_graph_id

    @property
    def before_launch(self) -> dict[int, list[PreKernelOp]]:
        """compiled_launch_id -> ops to execute before that launch.

        Sync H2D prefetches (no separate async start anchor) are emitted in
        the CONSUMER graph only. SSD prefetches follow the same rule.
        """
        result: dict[int, list[PreKernelOp]] = defaultdict(list)
        for op in self._schedule.prefetches:
            if op.before_launch_id >= 0 and self._is_consumer_here(op):
                result[op.before_launch_id].append(op)
        for op in self._schedule.h2d_prefetches:
            if op.cross_iter:
                continue
            if not self._is_consumer_here(op):
                continue
            if (
                op.before_launch_id >= 0
                and (op.after_launch_id < 0 or op.after_launch_id == op.before_launch_id)
            ):
                result[op.before_launch_id].append(op)
        return dict(result)

    @property
    def h2d_after_launch(self) -> dict[int, list[H2DPrefetchOp]]:
        """compiled_launch_id -> async H2D starts after that launch.

        Only emit in the op's ISSUE graph. For cross-graph async the issue
        graph differs from the consumer graph; same-graph async uses
        ``issue_compiled_graph_id = compiled_graph_id`` (default).
        """
        result: dict[int, list[H2DPrefetchOp]] = defaultdict(list)
        for op in self._schedule.h2d_prefetches:
            if op.after_launch_id < 0:
                continue
            if op.cross_iter or op.after_launch_id != op.before_launch_id:
                if not self._is_issuer_here(op):
                    continue
                result[op.after_launch_id].append(op)
        return dict(result)

    @property
    def h2d_wait_launch(self) -> dict[int, list[H2DPrefetchOp]]:
        """compiled_launch_id -> H2D ops whose async copy must complete.

        Only emit in the op's CONSUMER graph.
        """
        result: dict[int, list[H2DPrefetchOp]] = defaultdict(list)
        for op in self._schedule.h2d_prefetches:
            if op.after_launch_id < 0 or op.before_launch_id < 0:
                continue
            if op.cross_iter or op.after_launch_id != op.before_launch_id:
                if not self._is_consumer_here(op):
                    continue
                result[op.before_launch_id].append(op)
        return dict(result)

    @property
    def after_launch(self) -> dict[int, list[PostKernelOp]]:
        """compiled_launch_id -> ops to execute after that launch."""
        result: dict[int, list[PostKernelOp]] = defaultdict(list)
        for op in self._schedule.evict_vram:
            if op.after_launch_id >= 0:
                result[op.after_launch_id].append(op)
        for op in self._schedule.evict_dram:
            if op.after_launch_id >= 0:
                result[op.after_launch_id].append(op)
        return dict(result)


def _op_to_call(op: Union[PreKernelOp, PostKernelOp]) -> str:
    """Convert a SSD/DRAM IO op to the runtime method call string."""
    if isinstance(op, PrefetchOp):
        return f"_ws_rt.ssd_prefetch({op.tensor_name!r})"
    elif isinstance(op, EvictDramOp):
        return f"_ws_rt.evict_dram({op.tensor_name!r})"
    elif isinstance(op, H2DPrefetchOp):
        return f"_ws_rt.h2d_prefetch({op.tensor_name!r})"
    elif isinstance(op, EvictVramOp):
        return f"_ws_rt.evict_vram({op.tensor_name!r})"
    else:
        raise TypeError(f"Unknown IO op type: {type(op)}")


def _vram_op_to_call(
    op: Union[H2DPrefetchOp, EvictVramOp],
    tensor_id_to_input: dict[int, str],
) -> str:
    """Emit a variable-reference VRAM call for a resolved op.

    The generated code passes the graph input variable directly (not a string),
    routing to the tensor-object runtime path.

    H2D ops with after_launch_id use the async path (h2d_prefetch_async);
    a separate wait call is emitted at the consumer launch.
    """
    if op.compiled_tensor_id < 0:
        raise ValueError(
            f"{type(op).__name__} for '{op.tensor_name}' missing "
            f"compiled_tensor_id — schedule must include tensor crosswalk"
        )
    var = tensor_id_to_input.get(op.compiled_tensor_id)
    if var is None:
        raise ValueError(
            f"compiled_tensor_id {op.compiled_tensor_id} for "
            f"'{op.tensor_name}' not found in graph inputs — "
            f"compilation_hash mismatch?"
        )
    if isinstance(op, H2DPrefetchOp):
        if op.after_launch_id >= 0 and op.after_launch_id != op.before_launch_id:
            return f"_ws_rt.h2d_prefetch_async({var})"
        return f"_ws_rt.h2d_prefetch({var})"
    return f"_ws_rt.evict_vram({var})"


def _vram_wait_to_call(
    op: H2DPrefetchOp,
    tensor_id_to_input: dict[int, str],
) -> str:
    """Emit a wait call for an async H2D prefetch at the consumer launch."""
    var = tensor_id_to_input.get(op.compiled_tensor_id)
    if var is None:
        raise ValueError(
            f"compiled_tensor_id {op.compiled_tensor_id} for "
            f"'{op.tensor_name}' not found in graph inputs — "
            f"compilation_hash mismatch?"
        )
    return f"_ws_rt.h2d_wait({var})"


def _py_tuple(items: list[str], quote: bool = False) -> str:
    """Build a Python tuple-literal string from a list of items.

    If quote=True, items are wrapped in repr() (for string names).
    Otherwise items are passed through verbatim (for variable references).
    Single-item tuples get a trailing comma.
    """
    if quote:
        rendered = [repr(x) for x in items]
    else:
        rendered = list(items)
    body = ", ".join(rendered)
    if len(rendered) == 1:
        body = body + ","
    return f"({body})"


def build_ws_ops_call(
    *,
    waits: list[str] = (),
    sync_h2d: list[str] = (),
    ssd: list[str] = (),
    cold_start: list[str] = (),
    cold_start_tensors: list[str] = (),
    cross_graph_evict: bool = False,
    evict_vram: list[str] = (),
    evict_vram_names: list[str] = (),
    async_h2d: list[str] = (),
    evict_dram: list[str] = (),
    flush: bool = False,
) -> str | None:
    """Build one `_ws_rt.ws_ops(...)` call string for all ops at a boundary.

    Returns None if every category is empty and flush=False. Tuples are used
    (not lists) because they're cheaper to construct at call sites.

    Tensor-reference kwargs (waits, sync_h2d, cold_start_tensors, evict_vram,
    async_h2d) contain variable names that the runtime receives as
    torch.Tensor objects. Name kwargs (ssd, cold_start, evict_vram_names,
    evict_dram) contain strings.
    """
    parts: list[str] = []
    if waits:
        parts.append(f"waits={_py_tuple(waits)}")
    if sync_h2d:
        parts.append(f"sync_h2d={_py_tuple(sync_h2d)}")
    if ssd:
        parts.append(f"ssd={_py_tuple(ssd, quote=True)}")
    if cold_start:
        parts.append(f"cold_start={_py_tuple(cold_start, quote=True)}")
    if cold_start_tensors:
        parts.append(f"cold_start_tensors={_py_tuple(cold_start_tensors)}")
    if cross_graph_evict:
        parts.append("cross_graph_evict=True")
    if evict_vram:
        parts.append(f"evict_vram={_py_tuple(evict_vram)}")
    if evict_vram_names:
        parts.append(f"evict_vram_names={_py_tuple(evict_vram_names, quote=True)}")
    if async_h2d:
        parts.append(f"async_h2d={_py_tuple(async_h2d)}")
    if evict_dram:
        parts.append(f"evict_dram={_py_tuple(evict_dram, quote=True)}")
    if flush:
        parts.append("flush=True")
    if not parts:
        return None
    return f"_ws_rt.ws_ops({', '.join(parts)})"


@dataclasses.dataclass
class WeightStreamingLine(WrapperLine):
    """WrapperLine that emits weight streaming IO calls in generated code."""

    wrapper: PythonWrapperCodegen
    calls: list[str]

    def codegen(self, code: IndentedBuffer) -> None:
        for line in self.calls:
            code.writeline(line)
