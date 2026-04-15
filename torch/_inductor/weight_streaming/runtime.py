from __future__ import annotations

import logging
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

        # H2D: dedicated CUDA stream + events
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._h2d_events: dict[str, torch.cuda.Event] = {}

        # VRAM tensors (GPU)
        self._vram_tensors: dict[str, torch.Tensor] = {}

        # "ssd" | "dram" | "vram"
        self._location: dict[str, str] = {}
        for key in schedule.tensors:
            self._location[key] = "ssd"

        # Registered GPU weight tensors keyed by id(tensor).
        # Stores CPU backup and original storage size for evict/restore.
        self._weight_backups: dict[int, torch.Tensor] = {}
        self._weight_storage_nbytes: dict[int, int] = {}

        # VRAM evictions are deferred until the CUDA stream reaches the point
        # where eviction was requested. This prevents host-side storage resize
        # from racing outstanding kernels that still read the tensor.
        self._pending_vram_evictions: dict[
            int, tuple[torch.Tensor, torch.cuda.Event]
        ] = {}
        self._pending_vram_name_evictions: dict[
            str, tuple[torch.Tensor, torch.cuda.Event]
        ] = {}

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
        self._pending_vram_evictions.pop(key, None)

    def ssd_prefetch(self, tensor_name: str) -> None:
        """Submit async SSD->pinned-DRAM read."""
        with record_function("ws_rt::ssd_prefetch"):
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
            loc = self._location.get(tensor_name, "ssd")
            if loc in ("dram", "vram"):
                return
            self.ssd_prefetch(tensor_name)
            # Wait for SSD read to complete so it's ready before first kernel
            if tensor_name in self._inflight_reads:
                cpu_tensor = self._inflight_reads.pop(tensor_name).result()
                self._dram_buffers[tensor_name] = cpu_tensor
                self._location[tensor_name] = "dram"

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
                pending_tensor, _ = self._pending_vram_evictions.pop(key)
                self._free_tensor_storage(pending_tensor)
            elif gpu_tensor.untyped_storage().nbytes() > 0:
                del self._pending_vram_evictions[key]
                return gpu_tensor

        self.flush_ready_vram_evictions()
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
            evict_event.synchronize()
            pending_tensor.untyped_storage().resize_(0)
            self._pending_vram_evictions.pop(key)

        if gpu_tensor.untyped_storage().nbytes() == 0:
            nbytes = self._weight_storage_nbytes[key]
            gpu_tensor.untyped_storage().resize_(nbytes)
            gpu_tensor.record_stream(self._h2d_stream)
            self._h2d_stream.wait_stream(default_stream)
            with torch.cuda.stream(self._h2d_stream):
                with torch.no_grad():
                    gpu_tensor.copy_(backup, non_blocking=True)
        # Always record an event so the cross-iter wait can pop one even when
        # the storage was already resident (no copy needed). The event fires
        # immediately if no work was queued on h2d_stream.
        event = torch.cuda.Event()
        self._h2d_stream.record_event(event)
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
            current.wait_event(event)
            gpu_tensor.record_stream(current)
        return gpu_tensor

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

        # Async copy on dedicated H2D stream
        with torch.cuda.stream(self._h2d_stream):
            gpu_tensor = cpu_tensor.to(self._device, non_blocking=True)

        event = torch.cuda.Event()
        self._h2d_stream.record_event(event)
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
        event = torch.cuda.Event()
        torch.cuda.current_stream(self._device).record_event(event)
        return event

    def _free_tensor_storage(self, gpu_tensor: torch.Tensor) -> None:
        if gpu_tensor.untyped_storage().nbytes() > 0:
            gpu_tensor.untyped_storage().resize_(0)

    def flush_ready_vram_evictions(self) -> None:
        """Free pending VRAM evictions whose CUDA completion events are ready."""
        with record_function("ws_rt::flush_ready_vram_evictions"):
            for key, (gpu_tensor, event) in list(self._pending_vram_evictions.items()):
                if event.query():
                    self._free_tensor_storage(gpu_tensor)
                    del self._pending_vram_evictions[key]

            for tensor_name, (gpu_tensor, event) in list(
                self._pending_vram_name_evictions.items()
            ):
                if event.query():
                    self._free_tensor_storage(gpu_tensor)
                    del self._pending_vram_name_evictions[tensor_name]
                    self._vram_tensors.pop(tensor_name, None)
                    if self._location.get(tensor_name) == "vram":
                        self._location[tensor_name] = "dram"

    def flush_all_vram_evictions(self) -> None:
        """Block until all pending VRAM evictions are safe, then free them."""
        for key, (gpu_tensor, event) in list(self._pending_vram_evictions.items()):
            event.synchronize()
            self._free_tensor_storage(gpu_tensor)
            del self._pending_vram_evictions[key]

        for tensor_name, (gpu_tensor, event) in list(
            self._pending_vram_name_evictions.items()
        ):
            event.synchronize()
            self._free_tensor_storage(gpu_tensor)
            del self._pending_vram_name_evictions[tensor_name]
            self._vram_tensors.pop(tensor_name, None)
            if self._location.get(tensor_name) == "vram":
                self._location[tensor_name] = "dram"

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
        self._dram_buffers.pop(tensor_name, None)
        if self._location.get(tensor_name) in ("dram", "vram"):
            self._location[tensor_name] = "ssd"

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
