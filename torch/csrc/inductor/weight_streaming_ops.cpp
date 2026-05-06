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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch::inductor::weight_streaming {

using Key = uintptr_t;

static std::string make_named_key(int64_t gid, c10::string_view name) {
  std::string r;
  r.reserve(name.size() + 12);
  r.append(std::to_string(gid));
  r.push_back(':');
  r.append(name.data(), name.size());
  return r;
}

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

  // By-key registration for cross-graph async. Populated from
  // ``register_weight_named(gid, name, tensor)`` — lets graph A's
  // wrapper issue an H2D on a tensor it has no local Python reference
  // for (the tensor is only a graph input in graph B). The consumer
  // graph still waits via the normal TensorImpl*-keyed path.
  std::unordered_map<std::string, Key> named_to_key;

  // Tensors the scheduler has promised to reload (any graph's prefetch
  // list references them). ``evict_cross_graph`` only considers tensors
  // in this set — prevents zeroing storage of a tensor that no graph's
  // schedule will subsequently reload (which would crash the consumer
  // kernel). Safely degrades to a no-op if the set is empty.
  std::unordered_set<Key> reloadable_keys;

  // Per-graph iter counter. Advanced by ``begin_graph_iter(gid)``, read
  // by ws_ops to filter per-iter masked ops. -1 means "graph not yet
  // entered" (initial state).
  std::unordered_map<int64_t, int64_t> graph_iter_idx;

  // Per-tensor iter masks for partial schedules. If a Key is present in
  // the map, the corresponding op only fires when current iter index of
  // the tensor's graph is in the bitset. Empty map = legacy semantics
  // (every tensor fires on every iter).
  // - h2d_iter_mask: applies to async_h2d / sync_h2d / waits for that
  //   tensor (if absent in current iter, the op is skipped entirely;
  //   tensor must already be resident).
  // - evict_iter_mask: applies to evict_vram for that tensor.
  // For typical multi-iter partial: a tensor evicted in N of M-1 gaps has
  // N entries in evict_iter_mask (iter k where evict fires) and N
  // entries in h2d_iter_mask (iter k+1 where reload fires).
  std::unordered_map<Key, std::vector<uint8_t>> h2d_iter_mask;   // bit per iter
  std::unordered_map<Key, std::vector<uint8_t>> evict_iter_mask;

  // Most-recently-entered graph (set by begin_graph_iter). ws_ops uses
  // this to look up the current iter index for masking.
  int64_t current_graph_id = -1;

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

  // Diagnostic counters (read via ``get_wait_stats``).
  uint64_t wait_hit = 0;        // event was pending → async actually consumed
  uint64_t wait_miss_synced = 0;// no event, fell back to sync copy
  uint64_t wait_miss_resident = 0; // no event, storage already non-empty
  uint64_t fire_async = 0;      // h2d_prefetch_async actually queued copy
  uint64_t fire_skip_inflight = 0;
  uint64_t fire_skip_resident = 0;
  uint64_t xg_dispatched = 0;   // xg_async loop ran for this (gid,name)
  uint64_t xg_named_miss = 0;   // (gid,name) not in named_to_key
  uint64_t xg_rw_miss = 0;      // key not in registered_weights
  uint64_t xg_consumed = 0;     // fired and event later popped by wait

  // When TORCH_WS_MEASURE_STALL=1 in env, ``h2d_wait`` brackets each
  // event-block with timing CUDA events on the compute stream so we
  // can later sum elapsed(pre, post) = actual time the compute stream
  // stalled waiting for in-flight async H2D. Off by default (timing
  // events cost ~3 µs/call and perturb high-rate measurements).
  bool measure_stall = false;
  std::vector<std::pair<at::cuda::CUDAEvent, at::cuda::CUDAEvent>>
      stall_brackets;
};

static WSState& state() {
  static WSState s;
  static bool inited = false;
  if (!inited) {
    s.measure_stall = (std::getenv("TORCH_WS_MEASURE_STALL") != nullptr);
    inited = true;
  }
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
  // Multi-stream pool. ``TORCH_WS_H2D_STREAMS`` env var controls count
  // (default 1). With N > 1 streams, fires round-robin across streams
  // — multiple copy engines (typical 2 on consumer Ada, 4-8 on data
  // center) can DMA in parallel, reducing head-of-line blocking and
  // amortizing per-fire launch overhead. Total PCIe link bandwidth is
  // still shared, so this helps when fires bunch (FIFO drain time
  // exceeds compute window) but not when bandwidth-bound.
  int n = 1;
  if (const char* env = std::getenv("TORCH_WS_H2D_STREAMS")) {
    int parsed = std::atoi(env);
    if (parsed >= 1 && parsed <= 16) {
      n = parsed;
    }
  }
  for (int i = 0; i < n; ++i) {
    s.h2d_streams.push_back(
        c10::cuda::getStreamFromPool(/*isHighPriority=*/false, di));
  }
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

// Core registration (caller holds s.mu).
static void register_weight_locked(WSState& s, const at::Tensor& gpu_tensor) {
  Key k = key_of(gpu_tensor);
  if (s.weight_backups.count(k)) {
    return;  // already registered — idempotent
  }
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

static void register_weight_impl(WSState& s, const at::Tensor& gpu_tensor) {
  std::lock_guard<std::mutex> lk(s.mu);
  register_weight_locked(s, gpu_tensor);
}

// Record a (gid, name) -> TensorImpl* mapping for a tensor that was
// previously registered via ``register_weight``. No-op (silently) if
// the tensor isn't a registered WS weight — the wrapper emits this
// call for every graph input but only weights should enter the named
// map; activations must not be pin-cloned.
static void register_weight_named_impl(
    WSState& s,
    int64_t gid,
    c10::string_view name,
    const at::Tensor& gpu_tensor) {
  std::lock_guard<std::mutex> lk(s.mu);
  Key k = key_of(gpu_tensor);
  if (!s.weight_backups.count(k)) {
    return;
  }
  s.named_to_key[make_named_key(gid, name)] = k;
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
  s.named_to_key.clear();
  s.reloadable_keys.clear();
  s.event_pool.clear();
  s.h2d_streams.clear();
  s.device_index = -1;
  s.h2d_next_stream_idx = 0;
  s.graph_iter_idx.clear();
  s.h2d_iter_mask.clear();
  s.evict_iter_mask.clear();
  s.current_graph_id = -1;
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
    ++s.fire_skip_inflight;
    return;  // already in flight; wait will drain it
  }
  if (gpu_tensor.storage().nbytes() != 0
      && s.pending_evictions.find(k) == s.pending_evictions.end()) {
    // Tensor is already resident with no pending evict — issuing an
    // async copy here is wasteful (it'd re-copy already-resident data).
    // Counter for visibility; we still record an event so the wait
    // unblocks, but the bytes saved here are zero.
    ++s.fire_skip_resident;
  } else {
    ++s.fire_async;
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
    // NOTE: previously a stream-wide fence on default_stream was
    // recorded here ("h2d_stream waits on any compute_stream-side ops
    // so the resize is visible"). That fence was redundant: the
    // pending_evictions drain above already synchronizes with the prior
    // eviction's event (which was recorded AFTER the kernel that used
    // the old storage finished), and resize_storage_bytes is host-
    // synchronous so the new storage is visible immediately to all
    // streams. The fence was costing async overlap — every H2D after
    // an eviction serialized with the entire compute stream's prior
    // ops. Removed.
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

// Wait for a previously-started async H2D. If no event is pending AND
// the tensor's storage is empty, fall back to a sync copy — handles
// the bootstrap case where a cross-graph async was scheduled but the
// issuer graph's wrapper couldn't fire it (e.g., iter 0 before the
// consumer has registered its graph input under (gid, name)).
static void h2d_wait(WSState& s, const at::Tensor& gpu_tensor) {
  Key k = key_of(gpu_tensor);
  auto it = s.h2d_events.find(k);
  if (it == s.h2d_events.end()) {
    if (s.weight_backups.count(k) && gpu_tensor.storage().nbytes() == 0) {
      if (std::getenv("TORCH_WS_TRACE_MISS")) {
        fprintf(stderr,
                "[ws_rt miss_synced] key=%p storage_nbytes=%ld "
                "registered=1 inflight_total=%zu\n",
                (void*)k, (long)gpu_tensor.storage().nbytes(),
                s.h2d_events.size());
      }
      h2d_prefetch_sync(s, gpu_tensor);
      ++s.wait_miss_synced;
    } else {
      ++s.wait_miss_resident;
    }
    return;
  }
  ++s.wait_hit;
  auto ev = std::move(it->second);
  s.h2d_events.erase(it);
  auto current = at::cuda::getCurrentCUDAStream(s.device_index);
  if (s.measure_stall) {
    // cudaEventDefault (= 0) enables timing; the default CUDAEvent
    // constructor uses cudaEventDisableTiming for performance.
    at::cuda::CUDAEvent pre(cudaEventDefault);
    at::cuda::CUDAEvent post(cudaEventDefault);
    pre.record(current);
    ev.block(current);
    post.record(current);
    s.stall_brackets.emplace_back(std::move(pre), std::move(post));
  } else {
    ev.block(current);
  }
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

// Forward declarations for per-iter mask helpers (defined further
// below; ws_ops_impl uses them).
static inline bool fires_in_iter(
    const std::unordered_map<Key, std::vector<uint8_t>>& mask_map,
    Key key,
    int64_t iter);
static int64_t current_iter_of(WSState& s, int64_t gid);

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
//
// ``xg_async_gids`` / ``xg_async_names`` are parallel arrays describing
// cross-graph async H2D issues. The issuing graph has no local Python
// reference to the target tensor; it was previously registered by the
// consumer graph via ``register_weight_named``.
static void ws_ops_impl(
    at::TensorList waits,
    at::TensorList sync_h2d,
    at::TensorList evict_vram_list,
    at::TensorList async_h2d,
    at::ArrayRef<int64_t> xg_async_gids,
    const c10::List<std::string>& xg_async_names,
    bool flush) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);

  TORCH_CHECK(
      xg_async_gids.size() == xg_async_names.size(),
      "ws_ops: xg_async_gids and xg_async_names must have the same length");

  // Current iter for masking. -1 if no graph has been entered (e.g.,
  // initial cold-start path). All masked ops with cur_iter=-1 fire by
  // default (their masks aren't consulted unless cur_iter is in range).
  const int64_t cur_iter = current_iter_of(s, s.current_graph_id);
  const bool have_h2d_masks = !s.h2d_iter_mask.empty();
  const bool have_evict_masks = !s.evict_iter_mask.empty();

  // Order matches Python's WeightStreamRuntime.ws_ops:
  //   waits -> sync_h2d -> evict_vram -> async_h2d -> xg_async -> flush.
  // Waits drain any previously-issued async H2D; must happen before
  // subsequent ops that may free its storage.
  for (const auto& t : waits) {
    if (have_h2d_masks &&
        !fires_in_iter(s.h2d_iter_mask, key_of(t), cur_iter)) {
      continue;  // partial schedule: tensor was kept resident this iter
    }
    h2d_wait(s, t);
  }
  // For a large sync_h2d batch (e.g., cold-start of a whole graph),
  // intersperse flush_ready_vram_evictions between loads so freed
  // storage returns to the caching allocator in time to be reused
  // by the next allocation. Matches the pre-refactor Python path
  // which called ``_flush_ready_vram_evictions_inline`` on every
  // _h2d_prefetch_tensor. Without this, peak grows by the sum of
  // cold-start size (tensors accumulate before any freeing).
  const bool interleave_flush = sync_h2d.size() > 16;
  for (const auto& t : sync_h2d) {
    if (interleave_flush && !s.pending_evictions.empty()) {
      flush_ready_vram_evictions(s);
    }
    h2d_prefetch_sync(s, t);
  }
  for (const auto& t : evict_vram_list) {
    if (have_evict_masks &&
        !fires_in_iter(s.evict_iter_mask, key_of(t), cur_iter)) {
      continue;  // partial schedule: not evicted in this iter's gap
    }
    evict_vram(s, t);
  }
  for (const auto& t : async_h2d) {
    if (have_h2d_masks &&
        !fires_in_iter(s.h2d_iter_mask, key_of(t), cur_iter)) {
      continue;
    }
    h2d_prefetch_async(s, t);
  }
  for (size_t i = 0; i < xg_async_gids.size(); ++i) {
    ++s.xg_dispatched;
    auto it = s.named_to_key.find(
        make_named_key(xg_async_gids[i], xg_async_names.get(i)));
    if (it == s.named_to_key.end()) {
      ++s.xg_named_miss;
      continue;  // not yet registered (e.g. iter 0 before consumer runs)
    }
    auto rw = s.registered_weights.find(it->second);
    if (rw == s.registered_weights.end()) {
      ++s.xg_rw_miss;
      continue;
    }
    h2d_prefetch_async(s, rw->second);
  }
  if (flush) {
    flush_ready_vram_evictions(s);
  }
}

static void register_weight_op(const at::Tensor& t) {
  register_weight_impl(state(), t);
}

static void register_weight_named_op(
    int64_t gid, c10::string_view name, const at::Tensor& t) {
  register_weight_named_impl(state(), gid, name, t);
}

// Drain the stall-bracket queue: synchronize all events and sum
// elapsed(pre, post) on the compute stream → total time the stream
// stalled waiting for in-flight async H2D. Returns total_stall_ms ×
// 1000 (so the int64 return preserves microsecond precision).
// Resets the bracket list. Returns 0 if stall measurement disabled.
static int64_t drain_stall_us_op() {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  if (!s.measure_stall || s.stall_brackets.empty()) {
    return 0;
  }
  // Synchronize so all post events are queryable.
  s.stall_brackets.back().second.synchronize();
  double total_ms = 0.0;
  for (auto& pr : s.stall_brackets) {
    float ms = pr.first.elapsed_time(pr.second);
    if (ms > 0) total_ms += ms;
  }
  s.stall_brackets.clear();
  return (int64_t)(total_ms * 1000.0);  // microseconds
}

// Read & reset the diagnostic counters. Returns
// [wait_hit, wait_miss_synced, wait_miss_resident,
//  fire_async, fire_skip_inflight, fire_skip_resident,
//  xg_dispatched, xg_named_miss, xg_rw_miss].
static std::vector<int64_t> get_wait_stats_op() {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  std::vector<int64_t> out{
      (int64_t)s.wait_hit, (int64_t)s.wait_miss_synced,
      (int64_t)s.wait_miss_resident, (int64_t)s.fire_async,
      (int64_t)s.fire_skip_inflight, (int64_t)s.fire_skip_resident,
      (int64_t)s.xg_dispatched, (int64_t)s.xg_named_miss,
      (int64_t)s.xg_rw_miss};
  s.wait_hit = s.wait_miss_synced = s.wait_miss_resident = 0;
  s.fire_async = s.fire_skip_inflight = s.fire_skip_resident = 0;
  s.xg_dispatched = s.xg_named_miss = s.xg_rw_miss = 0;
  return out;
}

static void reset_op() {
  reset_impl(state());
}

// ---------------------------------------------------------------------------
// Per-iter masking for partial schedules.
//
// ``begin_graph_iter(gid)`` advances the per-graph iter counter (starting
// at 0 on first call) and marks ``gid`` as the current graph. ws_ops
// then uses ``s.graph_iter_idx[s.current_graph_id]`` to decide whether
// each masked op fires this iter.
//
// ``set_iter_mask(gid, name, h2d_iters, evict_iters, total_iters)``
// records, per (gid,name) → registered Key, which iter indices the
// corresponding op fires on. Empty masks (or missing key) → fires every
// iter (legacy semantic).
// ---------------------------------------------------------------------------

static void begin_graph_iter_op(int64_t gid) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  auto it = s.graph_iter_idx.find(gid);
  if (it == s.graph_iter_idx.end()) {
    s.graph_iter_idx[gid] = 0;
  } else {
    it->second += 1;
  }
  s.current_graph_id = gid;
}

static void set_iter_mask_op(
    int64_t gid,
    c10::string_view name,
    at::ArrayRef<int64_t> h2d_iters,
    at::ArrayRef<int64_t> evict_iters,
    int64_t total_iters) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  auto nit = s.named_to_key.find(make_named_key(gid, std::string(name)));
  if (nit == s.named_to_key.end()) {
    return;  // tensor not registered yet; caller's bug — silently ignore
  }
  Key key = nit->second;
  auto convert = [&](at::ArrayRef<int64_t> iters) {
    std::vector<uint8_t> mask((size_t)std::max<int64_t>(total_iters, 0), 0);
    for (auto k : iters) {
      if (k >= 0 && k < total_iters) mask[(size_t)k] = 1;
    }
    return mask;
  };
  s.h2d_iter_mask[key] = convert(h2d_iters);
  s.evict_iter_mask[key] = convert(evict_iters);
}

// Returns true iff the op should fire this iter. Empty/missing mask =
// "fires every iter" (legacy, full-evict, or unmasked single-iter op).
static inline bool fires_in_iter(
    const std::unordered_map<Key, std::vector<uint8_t>>& mask_map,
    Key key,
    int64_t iter) {
  auto it = mask_map.find(key);
  if (it == mask_map.end()) return true;
  const auto& mask = it->second;
  if (iter < 0 || iter >= (int64_t)mask.size()) return true;
  return mask[(size_t)iter] != 0;
}

static int64_t current_iter_of(WSState& s, int64_t gid) {
  auto it = s.graph_iter_idx.find(gid);
  return it == s.graph_iter_idx.end() ? -1 : it->second;
}

static void flush_ready_op() {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  flush_ready_vram_evictions(s);
}

// True if an async H2D for this tensor is still pending a wait — used
// by ``evict_cross_graph`` to avoid resizing a storage that's about to
// be consumed by another graph's kernels.
static bool has_inflight_h2d_op(const at::Tensor& t) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  return s.h2d_events.count(key_of(t)) > 0;
}

// At wrapper entry for graph N, evict every registered weight that is
// NOT in ``keep_tensors`` and NOT the target of an in-flight async H2D.
// Synchronous resize(0) — at wrapper entry no kernels are using these
// tensors on the default stream. Skipping in-flight keys preserves
// cross-graph async correctness: an earlier graph may have fired an
// H2D whose consumer kernel in a later graph hasn't yet run.
static void evict_cross_graph_op(at::TensorList keep_tensors) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  if (s.registered_weights.empty() || s.reloadable_keys.empty()) {
    // No reloadable set → can't prove eviction is safe → skip.
    // This is the defensive posture; schedulers that want the peak
    // reduction must call ``mark_reloadable`` for every tensor they
    // plan to prefetch.
    return;
  }
  std::unordered_set<Key> keep_keys;
  keep_keys.reserve(keep_tensors.size());
  for (const auto& t : keep_tensors) {
    keep_keys.insert(key_of(t));
  }
  for (const Key& k : s.reloadable_keys) {
    if (keep_keys.count(k)) continue;
    if (s.h2d_events.count(k)) continue;  // async in flight; consumer waits
    auto rw = s.registered_weights.find(k);
    if (rw == s.registered_weights.end()) continue;
    auto pend_it = s.pending_evictions.find(k);
    if (pend_it != s.pending_evictions.end()) {
      auto& [pending_tensor, evict_event] = pend_it->second;
      if (!evict_event.query()) {
        evict_event.synchronize();
      }
      release_event(s, std::move(evict_event));
      s.pending_evictions.erase(pend_it);
    }
    if (rw->second.storage().nbytes() > 0) {
      resize_storage_bytes(rw->second, 0);
    }
  }
}

// Populate the scheduler-promised reload set. Idempotent — can be
// called from every wrapper entry without blowing up state. Only adds;
// cleared by ``reset``.
static void mark_reloadable_op(at::TensorList tensors) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  for (const auto& t : tensors) {
    s.reloadable_keys.insert(key_of(t));
  }
}

// ---------------------------------------------------------------------------
// TORCH_LIBRARY registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(ws_rt, m) {
  m.def(
      "ws_ops(Tensor[] waits, Tensor[] sync_h2d, Tensor[] evict_vram, "
      "Tensor[] async_h2d, int[] xg_async_gids, str[] xg_async_names, "
      "bool flush) -> ()",
      ws_ops_impl);
  m.def(
      "register_weight(Tensor t) -> ()",
      register_weight_op);
  m.def(
      "register_weight_named(int gid, str name, Tensor t) -> ()",
      register_weight_named_op);
  m.def("reset() -> ()", reset_op);
  m.def("flush_ready_vram_evictions() -> ()", flush_ready_op);
  m.def("has_inflight_h2d(Tensor t) -> bool", has_inflight_h2d_op);
  m.def("evict_cross_graph(Tensor[] keep_tensors) -> ()",
        evict_cross_graph_op);
  m.def("mark_reloadable(Tensor[] tensors) -> ()", mark_reloadable_op);
  m.def("get_wait_stats() -> int[]", get_wait_stats_op);
  m.def("drain_stall_us() -> int", drain_stall_us_op);
  m.def("begin_graph_iter(int gid) -> ()", begin_graph_iter_op);
  m.def(
      "set_iter_mask(int gid, str name, int[] h2d_iters, "
      "int[] evict_iters, int total_iters) -> ()",
      set_iter_mask_op);
}

}  // namespace torch::inductor::weight_streaming
