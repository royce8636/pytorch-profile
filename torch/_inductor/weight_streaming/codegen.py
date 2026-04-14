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

    def __init__(self, schedule: IOSchedule) -> None:
        self._schedule = schedule
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

    @property
    def before_launch(self) -> dict[int, list[PreKernelOp]]:
        """compiled_launch_id -> ops to execute before that launch."""
        result: dict[int, list[PreKernelOp]] = defaultdict(list)
        for op in self._schedule.prefetches:
            if op.before_launch_id >= 0:
                result[op.before_launch_id].append(op)
        for op in self._schedule.h2d_prefetches:
            if op.before_launch_id >= 0:
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
        return f"_ws_rt.h2d_prefetch({var})"
    return f"_ws_rt.evict_vram({var})"


@dataclasses.dataclass
class WeightStreamingLine(WrapperLine):
    """WrapperLine that emits weight streaming IO calls in generated code."""

    wrapper: PythonWrapperCodegen
    calls: list[str]

    def codegen(self, code: IndentedBuffer) -> None:
        for line in self.calls:
            code.writeline(line)
