// Weight streaming runtime hot-path, implemented in C++ to eliminate
// Python per-op dispatch cost.
//
// Schedulers emit ``torch.ops.ws_rt.ws_ops(...)`` at each kernel boundary;
// this file implements the native version. State is a process-singleton
// indexed by ``TensorImpl*`` (stable for the life of the tensor).
//
// The Python ``WeightStreamRuntime`` in torch/_inductor/weight_streaming/
// now becomes a thin facade: it populates this C++ state via
// ``torch.ops.ws_rt.register_weight`` and forwards hot-path calls here.
// Set ``TORCH_WS_USE_PYTHON=1`` in the environment to bypass the native
// path and fall back to the pure-Python implementation (for differential
// testing). The Python facade handles that switch; the native op only
// implements one path.

#include <torch/library.h>

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#ifdef USE_CUDA
#include <ATen/native/cuda/Resize.h>
#endif

#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch::inductor::weight_streaming {

using Key = uintptr_t;

// ---------------------------------------------------------------------------
// Singleton state
// ---------------------------------------------------------------------------

struct WSState {
  // All maps keyed by TensorImpl*. The Tensor wrappers are held by value
  // (via at::Tensor) so their refcount keeps the storage alive for the
  // life of an entry — critical because the user's Python holds the
  // tensor alongside us.
  std::unordered_map<Key, at::Tensor> weight_backups;   // pinned CPU backup
  std::unordered_map<Key, int64_t>    weight_storage_nbytes;
  std::unordered_map<Key, at::Tensor> registered_weights;

  // Deferred eviction: (GPU tensor to free, event recorded on compute
  // stream after the last use). ``_flush_ready_vram_evictions_inline``
  // drains these once the event has fired.
  std::unordered_map<Key, std::pair<at::Tensor, at::cuda::CUDAEvent>>
      pending_evictions;

  // In-flight async H2D: event recorded on h2d_stream. ``h2d_wait``
  // blocks the compute stream on this event and pops it.
  std::unordered_map<Key, at::cuda::CUDAEvent> h2d_events;

  // H2D stream(s). Single stream matches default runtime config
  // (``TORCH_WS_H2D_STREAMS=1``); multi-stream support can be bolted on
  // later.
  std::vector<at::cuda::CUDAStream> h2d_streams;
  size_t h2d_next_stream_idx = 0;

  // Device index (cached; all tensors must be on this device).
  c10::DeviceIndex device_index = -1;

  // Event pool — avoids CUDAEvent alloc on the hot path. The pool owns
  // the events; acquire moves them out, release moves them back.
  std::vector<at::cuda::CUDAEvent> event_pool;
  size_t event_pool_cap = 2048;

  std::mutex mu;  // protects ONLY the maps / pools; most hot-path work
                  // is expected to be single-threaded (main inference),
                  // but guard against accidental multi-thread use.
};

static WSState& state() {
  static WSState s;
  return s;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline Key key_of(const at::Tensor& t) {
  return reinterpret_cast<Key>(t.unsafeGetTensorImpl());
}

static void ensure_streams(WSState& s, c10::DeviceIndex di) {
  if (!s.h2d_streams.empty()) {
    return;
  }
  s.device_index = di;
  s.h2d_streams.push_back(
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false, di));
}

static at::cuda::CUDAEvent acquire_event(WSState& s) {
  if (!s.event_pool.empty()) {
    auto ev = std::move(s.event_pool.back());
    s.event_pool.pop_back();
    return ev;
  }
  return at::cuda::CUDAEvent();
}

static void release_event(WSState& s, at::cuda::CUDAEvent&& ev) {
  if (s.event_pool.size() < s.event_pool_cap) {
    s.event_pool.push_back(std::move(ev));
  }
  // Otherwise just let the temporary destruct → cuEventDestroy.
}

static void resize_storage_bytes(const at::Tensor& t, int64_t nbytes) {
#ifdef USE_CUDA
  at::native::resize_bytes_cuda(t.storage().unsafeGetStorageImpl(), nbytes);
#else
  TORCH_CHECK(false, "weight streaming requires CUDA build");
#endif
}

static void free_tensor_storage(const at::Tensor& t) {
  if (t.storage().nbytes() > 0) {
    resize_storage_bytes(t, 0);
  }
}

static c10::cuda::CUDAStream& pick_h2d_stream(WSState& s) {
  // Single-stream case is the common one.
  if (s.h2d_streams.size() == 1) {
    return s.h2d_streams[0];
  }
  auto& st = s.h2d_streams[s.h2d_next_stream_idx];
  s.h2d_next_stream_idx = (s.h2d_next_stream_idx + 1) % s.h2d_streams.size();
  return st;
}

// ---------------------------------------------------------------------------
// Registration (cold path)
// ---------------------------------------------------------------------------

static void register_weight_impl(WSState& s, const at::Tensor& gpu_tensor) {
  std::lock_guard<std::mutex> lk(s.mu);
  Key k = key_of(gpu_tensor);
  ensure_streams(s, gpu_tensor.device().index());
  // Pinned CPU backup: t.data.cpu().pin_memory() in Python. Mirrors that.
  auto cpu_backup = gpu_tensor.detach().to(
      at::kCPU,
      /*non_blocking=*/false,
      /*copy=*/true,
      /*optional_memory_format=*/std::nullopt);
  cpu_backup = cpu_backup.pin_memory();
  s.weight_backups[k] = std::move(cpu_backup);
  s.weight_storage_nbytes[k] = gpu_tensor.storage().nbytes();
  s.registered_weights[k] = gpu_tensor;
  s.pending_evictions.erase(k);
}

static void reset_impl(WSState& s) {
  std::lock_guard<std::mutex> lk(s.mu);
  // Drain pending events (they own CUDA resources).
  for (auto& [_, pair] : s.pending_evictions) {
    pair.second.synchronize();
  }
  s.pending_evictions.clear();
  for (auto& [_, ev] : s.h2d_events) {
    ev.synchronize();
  }
  s.h2d_events.clear();
  s.weight_backups.clear();
  s.weight_storage_nbytes.clear();
  s.registered_weights.clear();
  s.event_pool.clear();
  s.h2d_streams.clear();
  s.device_index = -1;
  s.h2d_next_stream_idx = 0;
}

// ---------------------------------------------------------------------------
// Hot-path primitives (not exposed as individual ops — callers should use
// ``ws_ops`` for batching)
// ---------------------------------------------------------------------------

// H2D async: start the copy on h2d_stream, record an event for wait.
// Handles pending eviction (wait for evict event → resize(0)) before
// starting the new copy.
static void h2d_prefetch_async(WSState& s, const at::Tensor& gpu_tensor) {
  Key k = key_of(gpu_tensor);
  auto backup_it = s.weight_backups.find(k);
  if (backup_it == s.weight_backups.end()) {
    return;  // not registered — noop
  }
  if (s.h2d_events.count(k)) {
    return;  // already in flight; wait will drain it
  }

  // Drain any pending evict first.
  auto pend_it = s.pending_evictions.find(k);
  if (pend_it != s.pending_evictions.end()) {
    auto& [pending_tensor, evict_event] = pend_it->second;
    if (!evict_event.query()) {
      evict_event.synchronize();
    }
    free_tensor_storage(pending_tensor);
    release_event(s, std::move(evict_event));
    s.pending_evictions.erase(pend_it);
  }

  auto& h2d_stream = pick_h2d_stream(s);
  if (gpu_tensor.storage().nbytes() == 0) {
    int64_t nbytes = s.weight_storage_nbytes[k];
    resize_storage_bytes(gpu_tensor, nbytes);
    // record_stream tells the caching allocator the h2d_stream is using
    // storage allocated on the default stream.
    c10::cuda::CUDACachingAllocator::recordStream(
        gpu_tensor.storage().data_ptr(), h2d_stream);
    auto default_stream =
        at::cuda::getCurrentCUDAStream(s.device_index);
    h2d_stream.stream();  // materialise
    // wait_stream: h2d_stream waits on any compute_stream-side ops so
    // the resize above is visible.
    {
      at::cuda::CUDAEvent fence;
      fence.record(default_stream);
      fence.block(h2d_stream);
    }
    // Do the copy on the h2d_stream.
    c10::cuda::CUDAStreamGuard guard(h2d_stream);
    auto& backup = backup_it->second;
    gpu_tensor.copy_(backup, /*non_blocking=*/true);
  }

  auto ev = acquire_event(s);
  ev.record(h2d_stream);
  s.h2d_events[k] = std::move(ev);
}

// H2D sync: block the compute stream until the copy is ready.
static void h2d_prefetch_sync(WSState& s, const at::Tensor& gpu_tensor) {
  Key k = key_of(gpu_tensor);
  auto backup_it = s.weight_backups.find(k);
  if (backup_it == s.weight_backups.end()) {
    return;
  }
  auto pend_it = s.pending_evictions.find(k);
  if (pend_it != s.pending_evictions.end()) {
    auto& [pending_tensor, evict_event] = pend_it->second;
    if (evict_event.query()) {
      free_tensor_storage(pending_tensor);
      release_event(s, std::move(evict_event));
      s.pending_evictions.erase(pend_it);
    } else if (gpu_tensor.storage().nbytes() > 0) {
      s.pending_evictions.erase(pend_it);
      return;  // still resident — fine, evict will drop on the floor
    }
  }
  if (gpu_tensor.storage().nbytes() == 0) {
    int64_t nbytes = s.weight_storage_nbytes[k];
    resize_storage_bytes(gpu_tensor, nbytes);
    auto& backup = backup_it->second;
    gpu_tensor.copy_(backup, /*non_blocking=*/true);
  }
}

// Wait for a previously-started async H2D.
static void h2d_wait(WSState& s, const at::Tensor& gpu_tensor) {
  Key k = key_of(gpu_tensor);
  auto it = s.h2d_events.find(k);
  if (it == s.h2d_events.end()) {
    return;
  }
  auto ev = std::move(it->second);
  s.h2d_events.erase(it);
  auto current = at::cuda::getCurrentCUDAStream(s.device_index);
  ev.block(current);
  c10::cuda::CUDACachingAllocator::recordStream(
      gpu_tensor.storage().data_ptr(), current);
  // NOTE: do NOT recycle the event here. ev.block() is GPU-side; the
  // event's cudaEvent_t is still being observed by the stream's fence.
  // Re-recording would corrupt the captured fence. Let it destruct.
}

// Queue an eviction. Actual resize(0) happens when the eviction event
// has fired (drained by ``flush_ready`` or ``h2d_prefetch_async`` above).
static void evict_vram(WSState& s, const at::Tensor& gpu_tensor) {
  Key k = key_of(gpu_tensor);
  if (!s.weight_backups.count(k)) {
    return;
  }
  if (s.pending_evictions.count(k)) {
    return;
  }
  if (gpu_tensor.storage().nbytes() == 0) {
    return;
  }
  auto ev = acquire_event(s);
  auto current = at::cuda::getCurrentCUDAStream(s.device_index);
  ev.record(current);
  s.pending_evictions.emplace(
      k, std::make_pair(gpu_tensor, std::move(ev)));
}

// Drain any evictions whose events have fired (non-blocking).
static void flush_ready_vram_evictions(WSState& s) {
  for (auto it = s.pending_evictions.begin();
       it != s.pending_evictions.end();) {
    auto& [gpu_tensor, ev] = it->second;
    if (ev.query()) {
      free_tensor_storage(gpu_tensor);
      release_event(s, std::move(ev));
      it = s.pending_evictions.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
// Registered torch ops
// ---------------------------------------------------------------------------

// Batched dispatch matching the Python ``ws_ops`` signature. Emitted by
// Inductor wrappers at each launch boundary.
static void ws_ops_impl(
    at::TensorList waits,
    at::TensorList sync_h2d,
    at::TensorList evict_vram_list,
    at::TensorList async_h2d,
    bool flush) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);

  // Order matches Python's WeightStreamRuntime.ws_ops:
  //   waits -> sync_h2d -> evict_vram -> async_h2d -> flush.
  // Waits drain any previously-issued async H2D; must happen before
  // subsequent ops that may free its storage.
  for (const auto& t : waits) {
    h2d_wait(s, t);
  }
  for (const auto& t : sync_h2d) {
    h2d_prefetch_sync(s, t);
  }
  for (const auto& t : evict_vram_list) {
    evict_vram(s, t);
  }
  for (const auto& t : async_h2d) {
    h2d_prefetch_async(s, t);
  }
  if (flush) {
    flush_ready_vram_evictions(s);
  }
}

static void register_weight_op(const at::Tensor& t) {
  register_weight_impl(state(), t);
}

static void reset_op() {
  reset_impl(state());
}

static void flush_ready_op() {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  flush_ready_vram_evictions(s);
}

// ---------------------------------------------------------------------------
// TORCH_LIBRARY registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(ws_rt, m) {
  m.def(
      "ws_ops(Tensor[] waits, Tensor[] sync_h2d, Tensor[] evict_vram, "
      "Tensor[] async_h2d, bool flush) -> ()",
      ws_ops_impl);
  m.def(
      "register_weight(Tensor t) -> ()",
      register_weight_op);
  m.def("reset() -> ()", reset_op);
  m.def("flush_ready_vram_evictions() -> ()", flush_ready_op);
}

}  // namespace torch::inductor::weight_streaming
