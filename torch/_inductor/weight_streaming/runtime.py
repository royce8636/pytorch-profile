from __future__ import annotations

import logging
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ClassVar

import torch
from torch._inductor.weight_streaming.plan import IOSchedule, TensorEntry
from torch.profiler import record_function

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class WeightStreamRuntime:
    """Singleton that tracks tensor locations and drives SSD->DRAM->VRAM IO."""

    _instance: ClassVar[WeightStreamRuntime | None] = None

    def __init__(self, schedule: IOSchedule, device: torch.device) -> None:
        self._schedule = schedule
        self._device = device

        # SSD -> DRAM: thread pool for async reads
        self._ssd_pool = ThreadPoolExecutor(max_workers=4)
        self._inflight_reads: dict[str, Future] = {}

        # DRAM buffers (pinned CPU tensors)
        self._dram_buffers: dict[str, torch.Tensor] = {}

        # H2D: K dedicated CUDA streams. K defaults to 1 for backwards
        # compatibility; set TORCH_WS_H2D_STREAMS=K to get K streams. Multi-
        # stream reduces head-of-line blocking without changing PCIe peak BW.
        # Round-robin per async dispatch keeps load balanced.
        num_h2d_streams = max(1, int(os.environ.get("TORCH_WS_H2D_STREAMS", "1")))
        self._h2d_streams: list[torch.cuda.Stream] = [
            torch.cuda.Stream(device=device) for _ in range(num_h2d_streams)
        ]
        self._h2d_stream = self._h2d_streams[0]  # alias for legacy paths
        self._h2d_next_stream_idx = 0
        self._h2d_events: dict[str, torch.cuda.Event] = {}

        # VRAM tensors (GPU)
        self._vram_tensors: dict[str, torch.Tensor] = {}

        # "ssd" | "dram" | "vram"
        self._location: dict[str, str] = {}
        for key in schedule.tensors:
            self._location[key] = "ssd"

        # Registered GPU weight tensors keyed by id(tensor).
        # Stores CPU backup and original storage size for evict/restore.
        # _registered_weights keeps a strong reference so cross-graph eviction
        # can iterate and resize storages at phase boundaries (at wrapper
        # entry for graph N, other graphs' weights can be evicted).
        self._weight_backups: dict[int, torch.Tensor] = {}
        self._weight_storage_nbytes: dict[int, int] = {}
        self._registered_weights: dict[int, torch.Tensor] = {}

        # VRAM evictions are deferred until the CUDA stream reaches the point
        # where eviction was requested. This prevents host-side storage resize
        # from racing outstanding kernels that still read the tensor.
        self._pending_vram_evictions: dict[
            int, tuple[torch.Tensor, torch.cuda.Event]
        ] = {}
        self._pending_vram_name_evictions: dict[
            str, tuple[torch.Tensor, torch.cuda.Event]
        ] = {}

        # Pool of recycled torch.cuda.Event objects. Event construction costs
        # ~3 µs of CUDA API time; with 232 prefetches/iter and 232 evictions,
        # reusing events meaningfully reduces per-op overhead.
        self._event_pool: deque[torch.cuda.Event] = deque()
        self._event_pool_cap = 2048

        # Unhidden-H2D stall instrumentation. When WS_MEASURE_H2D_STALL=1, each
        # h2d_wait bracketing records (pre, post) timing events on the compute
        # stream so we can later sum elapsed(pre, post) = time the compute
        # stream blocked waiting for H2D. Off by default: event.record() costs
        # ~3 µs/call and perturbs measurements at high rates.
        self._measure_h2d_stall = bool(
            int(os.environ.get("WS_MEASURE_H2D_STALL", "0"))
        )
        self._h2d_wait_hit_count = 0   # event was already done → hidden
        self._h2d_wait_miss_count = 0  # event not done → at least partially unhidden
        self._h2d_stall_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def _pick_h2d_stream(self) -> torch.cuda.Stream:
        """Round-robin pick across the H2D stream pool.

        Balances dispatch load. For K=1 (default) always returns the single
        stream; for K>1 rotates, so the driver can overlap transfers whose
        sizes and timings differ.
        """
        if len(self._h2d_streams) == 1:
            return self._h2d_streams[0]
        idx = self._h2d_next_stream_idx
        self._h2d_next_stream_idx = (idx + 1) % len(self._h2d_streams)
        return self._h2d_streams[idx]

    @classmethod
    def instance(cls) -> WeightStreamRuntime:
        assert cls._instance is not None, "WeightStreamRuntime not initialized"
        return cls._instance

    @classmethod
    def initialize(
        cls, schedule: IOSchedule, device: torch.device
    ) -> WeightStreamRuntime:
        cls._instance = cls(schedule, device)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance._ssd_pool.shutdown(wait=False)
            cls._instance = None

    def register_dram_tensor(self, tensor_name: str, tensor: torch.Tensor) -> None:
        """Register a pre-loaded DRAM tensor (for when tensors are already in CPU memory)."""
        self._dram_buffers[tensor_name] = tensor
        self._location[tensor_name] = "dram"

    def register_weight(self, gpu_tensor: torch.Tensor) -> None:
        """Register a GPU weight tensor for VRAM eviction/restore.

        Creates a pinned CPU backup copy. The generated code will pass this
        tensor object directly to evict_vram/h2d_prefetch (not a string name).
        """
        key = id(gpu_tensor)
        self._weight_backups[key] = gpu_tensor.data.cpu().pin_memory()
        self._weight_storage_nbytes[key] = gpu_tensor.untyped_storage().nbytes()
        self._registered_weights[key] = gpu_tensor
        self._pending_vram_evictions.pop(key, None)

    def evict_cross_graph(self, cold_start_tensors: tuple) -> None:
        """Evict every registered weight not in cold_start_tensors.

        Called at the top of each wrapper invocation. The cold_start list for
        graph N contains graph N's pinned tensors; anything else registered
        (other graphs' weights) isn't used by graph N's compute and can be
        freed while N runs. This is what delivers cross-graph peak savings —
        each graph's peak is bounded by (its pinned + bandwidth-limited
        cyclable in flight), not by the union of all graphs' weights.

        Eviction is synchronous (direct storage.resize_(0)): at wrapper entry
        no kernels are running on the default stream that could be reading
        these tensors, so it's safe to free immediately without an event
        wait. `_pending_vram_evictions` entries are drained first in case a
        prior eviction hadn't flushed yet.
        """
        if not self._registered_weights:
            return
        keep_ids: set[int] = {id(t) for t in cold_start_tensors}
        with record_function("ws_rt::evict_cross_graph"):
            for key, tensor in self._registered_weights.items():
                if key in keep_ids:
                    continue
                pending = self._pending_vram_evictions.pop(key, None)
                if pending is not None:
                    pending_tensor, evict_event = pending
                    if not evict_event.query():
                        evict_event.synchronize()
                    self._release_event(evict_event)
                if tensor.untyped_storage().nbytes() > 0:
                    tensor.untyped_storage().resize_(0)

    def _acquire_event(self) -> torch.cuda.Event:
        """Get a reusable CUDA event (recycled if pool non-empty)."""
        pool = self._event_pool
        if pool:
            return pool.pop()
        return torch.cuda.Event()

    def _release_event(self, event: torch.cuda.Event) -> None:
        """Return an event to the pool for reuse. Caller must guarantee the
        event is no longer in use (fired and all consumers have observed it).
        """
        pool = self._event_pool
        if len(pool) < self._event_pool_cap:
            pool.append(event)

    def ssd_prefetch(self, tensor_name: str) -> None:
        """Submit async SSD->pinned-DRAM read."""
        with record_function("ws_rt::ssd_prefetch"):
            self._ssd_prefetch_inline(tensor_name)

    def _ssd_prefetch_inline(self, tensor_name: str) -> None:
        if self._location.get(tensor_name) != "ssd":
            return
        if tensor_name in self._inflight_reads:
            return
        entry = self._schedule.tensors.get(tensor_name)
        if entry is None or not entry.file:
            return
        future = self._ssd_pool.submit(self._read_from_file, entry)
        self._inflight_reads[tensor_name] = future

    def cold_start_prefetch(self, tensor_name: str) -> None:
        """Load a tensor at startup (before any kernels run).

        If the tensor is already in DRAM (pre-loaded by user), this is a no-op.
        Otherwise, initiates an SSD prefetch and waits for completion.
        """
        with record_function("ws_rt::cold_start_prefetch"):
            self._cold_start_prefetch_inline(tensor_name)

    def _cold_start_prefetch_inline(self, tensor_name: str) -> None:
        loc = self._location.get(tensor_name, "ssd")
        if loc in ("dram", "vram"):
            return
        self._ssd_prefetch_inline(tensor_name)
        if tensor_name in self._inflight_reads:
            cpu_tensor = self._inflight_reads.pop(tensor_name).result()
            self._dram_buffers[tensor_name] = cpu_tensor
            self._location[tensor_name] = "dram"

    def cold_start_prefetch_tensor(self, gpu_tensor: torch.Tensor) -> None:
        """Load a registered WS weight at iter start (tensor-object path).

        The scheduler emits a large cold_start list assuming it refills all
        pinned tensors before any compute runs. For WS-registered weights the
        string-name `_cold_start_prefetch_inline` path is a no-op (no SSD
        entry), so the scheduler's plan silently fails and every use falls
        through the wrapper safety net as a sync H2D.

        Routing WS cold-starts through `_h2d_prefetch_tensor` (sync) restores
        the contract: post cold-start, storage is resized and data is copied
        on the default stream, so the safety-net's start-of-iter residency
        matches the scheduler's assumption.

        Note: sync cold-start serializes the bulk reload against compute.
        Async cold-start + per-tensor waits-before-first-use would overlap
        the copy with compute; doing that requires the wrapper to emit
        waits at each tensor's first use, which is a follow-up.
        """
        with record_function("ws_rt::cold_start_prefetch_tensor"):
            self._h2d_prefetch_tensor(gpu_tensor)

    def h2d_prefetch(self, tensor_name_or_tensor: str | torch.Tensor) -> torch.Tensor | None:
        """Restore a tensor's GPU storage from CPU backup, or do SSD->DRAM->VRAM.

        When called with a torch.Tensor (from generated code with numel matching),
        restores the tensor's storage from the CPU backup created by register_weight().

        When called with a string name (legacy path), does SSD->DRAM->VRAM copy.
        """
        with record_function("ws_rt::h2d_prefetch"):
            if isinstance(tensor_name_or_tensor, torch.Tensor):
                return self._h2d_prefetch_tensor(tensor_name_or_tensor)
            return self._h2d_prefetch_by_name(tensor_name_or_tensor)

    def _h2d_prefetch_tensor(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Restore a GPU tensor's storage from its CPU backup (synchronous)."""
        key = id(gpu_tensor)
        backup = self._weight_backups.get(key)
        if backup is None:
            return gpu_tensor

        pending = self._pending_vram_evictions.get(key)
        if pending is not None:
            _, event = pending
            if event.query():
                pending_tensor, pending_evt = self._pending_vram_evictions.pop(key)
                self._free_tensor_storage(pending_tensor)
                self._release_event(pending_evt)
            elif gpu_tensor.untyped_storage().nbytes() > 0:
                del self._pending_vram_evictions[key]
                return gpu_tensor

        self._flush_ready_vram_evictions_inline()
        if gpu_tensor.untyped_storage().nbytes() == 0:
            nbytes = self._weight_storage_nbytes[key]
            gpu_tensor.untyped_storage().resize_(nbytes)
            with torch.no_grad():
                gpu_tensor.copy_(backup)
        return gpu_tensor

    def h2d_prefetch_async(
        self, tensor_name_or_tensor: str | torch.Tensor
    ) -> torch.Tensor | None:
        """Start an async H2D copy on the dedicated stream."""
        with record_function("ws_rt::h2d_prefetch_async"):
            if isinstance(tensor_name_or_tensor, torch.Tensor):
                return self._h2d_prefetch_tensor_async(tensor_name_or_tensor)
            return self._h2d_prefetch_by_name(tensor_name_or_tensor)

    def _h2d_prefetch_tensor_async(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Restore a GPU tensor's storage from its CPU backup (async).

        Allocation is tied to the default stream so freeing (via resize_(0))
        recycles cleanly into the same allocator pool. The H2D copy itself
        runs on _h2d_stream for compute overlap. Cross-stream coordination:
          - host-block on any pending eviction event (typically already fired
            for cross-iter ops since the evict was many launches ago).
          - record_stream so the allocator knows _h2d_stream is using storage
            allocated on the default stream.
        """
        key = id(gpu_tensor)
        backup = self._weight_backups.get(key)
        if backup is None:
            return gpu_tensor

        if key in self._h2d_events:
            return gpu_tensor

        default_stream = torch.cuda.current_stream(self._device)

        pending = self._pending_vram_evictions.get(key)
        if pending is not None:
            pending_tensor, evict_event = pending
            # Cross-iter case: evict was recorded far back in prev iter, has
            # almost certainly fired by now — query() avoids the ~25 µs CPU
            # cost of a kernel-mode synchronize. Fall back to synchronize()
            # only when the event genuinely hasn't fired (intra-iter racing
            # case where evict happened just before this prefetch).
            if not evict_event.query():
                evict_event.synchronize()
            pending_tensor.untyped_storage().resize_(0)
            self._pending_vram_evictions.pop(key)
            self._release_event(evict_event)

        h2d_stream = self._pick_h2d_stream()
        if gpu_tensor.untyped_storage().nbytes() == 0:
            nbytes = self._weight_storage_nbytes[key]
            gpu_tensor.untyped_storage().resize_(nbytes)
            gpu_tensor.record_stream(h2d_stream)
            h2d_stream.wait_stream(default_stream)
            with torch.cuda.stream(h2d_stream):
                with torch.no_grad():
                    gpu_tensor.copy_(backup, non_blocking=True)
        # Always record an event so the cross-iter wait can pop one even when
        # the storage was already resident (no copy needed). The event fires
        # immediately if no work was queued on h2d_stream.
        event = self._acquire_event()
        h2d_stream.record_event(event)
        self._h2d_events[key] = event
        return gpu_tensor

    def h2d_wait(self, tensor_name_or_tensor: str | torch.Tensor) -> torch.Tensor | None:
        """Wait for an async H2D copy to complete before using the tensor."""
        with record_function("ws_rt::h2d_wait"):
            if isinstance(tensor_name_or_tensor, torch.Tensor):
                return self._h2d_wait_tensor(tensor_name_or_tensor)
            return self.wait_h2d(tensor_name_or_tensor)

    def _h2d_wait_tensor(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Wait for the async H2D event recorded by _h2d_prefetch_tensor_async.

        Also calls record_stream so the caching allocator knows the current
        (default) stream is using storage that was allocated on _h2d_stream —
        otherwise the allocator may recycle the memory before this stream's
        kernels finish reading it.
        """
        key = id(gpu_tensor)
        event = self._h2d_events.pop(key, None)
        if event is not None:
            current = torch.cuda.current_stream(self._device)
            if self._measure_h2d_stall:
                if event.query():
                    self._h2d_wait_hit_count += 1
                else:
                    self._h2d_wait_miss_count += 1
                pre = torch.cuda.Event(enable_timing=True)
                pre.record(current)
                current.wait_event(event)
                post = torch.cuda.Event(enable_timing=True)
                post.record(current)
                self._h2d_stall_events.append((pre, post))
            else:
                current.wait_event(event)
            gpu_tensor.record_stream(current)
            # Don't recycle the event here: stream.wait_event() is GPU-side
            # and doesn't observe the event. Re-recording on it before the
            # GPU processes the wait corrupts the wait's captured fence
            # (cuda illegal memory access).
        return gpu_tensor

    def drain_h2d_stall_stats(self) -> dict:
        """Sum elapsed(pre, post) across all recorded wait brackets.

        Requires WS_MEASURE_H2D_STALL=1. Synchronizes the device so timing
        events are queryable. Clears accumulated events after reading.
        """
        if not self._measure_h2d_stall:
            return {"enabled": False}
        torch.cuda.synchronize(self._device)
        total_ms = 0.0
        for pre, post in self._h2d_stall_events:
            total_ms += pre.elapsed_time(post)
        stats = {
            "enabled": True,
            "total_stall_ms": total_ms,
            "wait_count": len(self._h2d_stall_events),
            "hit_count": self._h2d_wait_hit_count,
            "miss_count": self._h2d_wait_miss_count,
        }
        self._h2d_stall_events.clear()
        self._h2d_wait_hit_count = 0
        self._h2d_wait_miss_count = 0
        return stats

    def _h2d_prefetch_by_name(self, tensor_name: str) -> torch.Tensor | None:
        """Legacy string-name path: SSD->DRAM->VRAM copy."""
        pending = self._pending_vram_name_evictions.get(tensor_name)
        if pending is not None:
            gpu_tensor, _ = pending
            event = pending[1]
            if event.query():
                del self._pending_vram_name_evictions[tensor_name]
                self._free_tensor_storage(gpu_tensor)
                self._vram_tensors.pop(tensor_name, None)
                if self._location.get(tensor_name) == "vram":
                    self._location[tensor_name] = "dram"
            elif gpu_tensor.untyped_storage().nbytes() > 0:
                del self._pending_vram_name_evictions[tensor_name]
                self._vram_tensors[tensor_name] = gpu_tensor
                self._location[tensor_name] = "vram"
                return gpu_tensor

        gpu_tensor = self._vram_tensors.get(tensor_name)
        if gpu_tensor is not None and gpu_tensor.untyped_storage().nbytes() > 0:
            return gpu_tensor

        self.flush_ready_vram_evictions()

        # Complete any pending SSD read
        if tensor_name in self._inflight_reads:
            cpu_tensor = self._inflight_reads.pop(tensor_name).result()
            self._dram_buffers[tensor_name] = cpu_tensor
            self._location[tensor_name] = "dram"
        elif self._location.get(tensor_name) == "ssd":
            entry = self._schedule.tensors.get(tensor_name)
            if entry and entry.file:
                self._dram_buffers[tensor_name] = self._read_from_file(entry)
                self._location[tensor_name] = "dram"

        cpu_tensor = self._dram_buffers.get(tensor_name)
        if cpu_tensor is None:
            return None

        # Async copy on a chosen H2D stream (round-robin when K>1).
        h2d_stream = self._pick_h2d_stream()
        with torch.cuda.stream(h2d_stream):
            gpu_tensor = cpu_tensor.to(self._device, non_blocking=True)

        event = torch.cuda.Event()
        h2d_stream.record_event(event)
        self._h2d_events[tensor_name] = event
        self._vram_tensors[tensor_name] = gpu_tensor
        self._location[tensor_name] = "vram"
        return gpu_tensor

    def wait_h2d(self, tensor_name: str) -> torch.Tensor | None:
        """Synchronize on the H2D copy event, return the GPU tensor."""
        event = self._h2d_events.get(tensor_name)
        if event is not None:
            torch.cuda.current_stream(self._device).wait_event(event)
            del self._h2d_events[tensor_name]
        return self._vram_tensors.get(tensor_name)

    def _record_vram_eviction_event(self) -> torch.cuda.Event:
        event = self._acquire_event()
        torch.cuda.current_stream(self._device).record_event(event)
        return event

    def _free_tensor_storage(self, gpu_tensor: torch.Tensor) -> None:
        if gpu_tensor.untyped_storage().nbytes() > 0:
            gpu_tensor.untyped_storage().resize_(0)

    def flush_ready_vram_evictions(self) -> None:
        """Free pending VRAM evictions whose CUDA completion events are ready."""
        with record_function("ws_rt::flush_ready_vram_evictions"):
            self._flush_ready_vram_evictions_inline()

    def _flush_ready_vram_evictions_inline(self) -> None:
        for key, (gpu_tensor, event) in list(self._pending_vram_evictions.items()):
            if event.query():
                self._free_tensor_storage(gpu_tensor)
                del self._pending_vram_evictions[key]
                self._release_event(event)

        for tensor_name, (gpu_tensor, event) in list(
            self._pending_vram_name_evictions.items()
        ):
            if event.query():
                self._free_tensor_storage(gpu_tensor)
                del self._pending_vram_name_evictions[tensor_name]
                self._vram_tensors.pop(tensor_name, None)
                if self._location.get(tensor_name) == "vram":
                    self._location[tensor_name] = "dram"
                self._release_event(event)

    def flush_all_vram_evictions(self) -> None:
        """Block until all pending VRAM evictions are safe, then free them."""
        for key, (gpu_tensor, event) in list(self._pending_vram_evictions.items()):
            event.synchronize()
            self._free_tensor_storage(gpu_tensor)
            del self._pending_vram_evictions[key]
            self._release_event(event)

        for tensor_name, (gpu_tensor, event) in list(
            self._pending_vram_name_evictions.items()
        ):
            event.synchronize()
            self._free_tensor_storage(gpu_tensor)
            del self._pending_vram_name_evictions[tensor_name]
            self._vram_tensors.pop(tensor_name, None)
            if self._location.get(tensor_name) == "vram":
                self._location[tensor_name] = "dram"
            self._release_event(event)

    def evict_vram(self, tensor_name_or_tensor: str | torch.Tensor) -> None:
        """Free GPU memory for this tensor.

        When called with a torch.Tensor (from generated code with numel matching),
        records an event and defers storage resize until the event completes.

        When called with a string name (legacy path), pops from _vram_tensors
        after the deferred eviction event completes.
        """
        with record_function("ws_rt::evict_vram"):
            if isinstance(tensor_name_or_tensor, torch.Tensor):
                self._evict_vram_tensor(tensor_name_or_tensor)
            else:
                self._evict_vram_by_name(tensor_name_or_tensor)

    def _evict_vram_tensor(self, gpu_tensor: torch.Tensor) -> None:
        """Queue eviction of a registered GPU tensor after current-stream work."""
        # Kill-switch: TORCH_WS_DISABLE_EVICT=1 turns all evictions into no-ops
        # so we can verify H2D/wait plumbing works independently of eviction
        # correctness. Useful when schedule's tensor IDs may be misaligned with
        # the current compilation (compilation_hash mismatch).
        if os.environ.get("TORCH_WS_DISABLE_EVICT"):
            return
        key = id(gpu_tensor)
        if key not in self._weight_backups:
            return
        if key in self._pending_vram_evictions:
            return
        if gpu_tensor.untyped_storage().nbytes() > 0:
            self._pending_vram_evictions[key] = (
                gpu_tensor,
                self._record_vram_eviction_event(),
            )

    def _evict_vram_by_name(self, tensor_name: str) -> None:
        """Legacy string-name path: queue eviction after current-stream work."""
        if tensor_name in self._pending_vram_name_evictions:
            return
        gpu_tensor = self._vram_tensors.get(tensor_name)
        if gpu_tensor is not None and gpu_tensor.untyped_storage().nbytes() > 0:
            self._pending_vram_name_evictions[tensor_name] = (
                gpu_tensor,
                self._record_vram_eviction_event(),
            )

    def evict_dram(self, tensor_name: str) -> None:
        """Free pinned CPU memory for this tensor."""
        self._evict_dram_inline(tensor_name)

    def _evict_dram_inline(self, tensor_name: str) -> None:
        self._dram_buffers.pop(tensor_name, None)
        if self._location.get(tensor_name) in ("dram", "vram"):
            self._location[tensor_name] = "ssd"

    def ws_ops(
        self,
        *,
        waits: tuple = (),
        sync_h2d: tuple = (),
        ssd: tuple = (),
        cold_start: tuple = (),
        cold_start_tensors: tuple = (),
        cross_graph_evict: bool = False,
        evict_vram: tuple = (),
        evict_vram_names: tuple = (),
        async_h2d: tuple = (),
        evict_dram: tuple = (),
        flush: bool = False,
    ) -> None:
        """Batched IO operations for one launch boundary.

        All ops of a given category are applied in one call under a single
        `record_function` span — this amortizes the per-op record_function
        and dispatch overhead. Ordering within the call matches the previous
        per-op codegen emission order:
          waits -> sync_h2d -> ssd -> cold_start (pre-launch)
          evict_vram -> async_h2d -> evict_dram -> flush (post-launch)
        The caller (codegen) supplies only the kwargs it needs for this site.
        """
        with record_function("ws_rt::ws_ops"):
            for t in waits:
                self._h2d_wait_tensor(t)
            if sync_h2d and self._measure_h2d_stall:
                current = torch.cuda.current_stream(self._device)
                pre = torch.cuda.Event(enable_timing=True)
                pre.record(current)
                for t in sync_h2d:
                    self._h2d_prefetch_tensor(t)
                post = torch.cuda.Event(enable_timing=True)
                post.record(current)
                self._h2d_stall_events.append((pre, post))
                self._h2d_wait_miss_count += len(sync_h2d)
            else:
                for t in sync_h2d:
                    self._h2d_prefetch_tensor(t)
            for n in ssd:
                self._ssd_prefetch_inline(n)
            for n in cold_start:
                self._cold_start_prefetch_inline(n)
            if cross_graph_evict and cold_start_tensors:
                # Run cross-graph eviction BEFORE loading this graph's cold
                # start — otherwise the new loads double-count against the
                # momentary peak. The cross-graph evict keeps cold_start_tensors
                # as the resident set survivors.
                self.evict_cross_graph(cold_start_tensors)
            for t in cold_start_tensors:
                self._h2d_prefetch_tensor(t)
            for t in evict_vram:
                self._evict_vram_tensor(t)
            for n in evict_vram_names:
                self._evict_vram_by_name(n)
            for t in async_h2d:
                self._h2d_prefetch_tensor_async(t)
            for n in evict_dram:
                self._evict_dram_inline(n)
            if flush:
                self._flush_ready_vram_evictions_inline()

    def evict(self, tensor_name: str) -> None:
        """Evict a tensor from both VRAM and DRAM."""
        self.evict_vram(tensor_name)
        self.evict_dram(tensor_name)

    def _read_from_file(self, entry: TensorEntry) -> torch.Tensor:
        """Blocking read from disk, returns a pinned CPU tensor."""
        dtype = _DTYPE_MAP[entry.dtype]
        numel = 1
        for s in entry.shape:
            numel *= s
        nbytes = numel * torch._utils._element_size(dtype)

        mmap_storage = torch.UntypedStorage.from_file(
            entry.file, False, entry.file_offset + nbytes
        )
        sliced = mmap_storage[entry.file_offset : entry.file_offset + nbytes]
        tensor = torch.tensor([], dtype=dtype).set_(
            torch.TypedStorage(
                wrap_storage=sliced, dtype=dtype, _internal=True
            )
        ).reshape(entry.shape)

        pinned = tensor.pin_memory()
        del tensor, sliced, mmap_storage
        return pinned
