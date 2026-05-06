#!/usr/bin/env python3
"""Unified profile + schedule + run pipeline for SD 3.5 weight streaming.

Port of streaming_e2e.py for StableDiffusion3Pipeline. Same single-process
profile + scheduler + run flow so the kernel layout used to build the schedule
matches the layout used at run time.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
_REPO_ROOT = str(THIS_DIR.parent)
sys.path.insert(0, _REPO_ROOT)
sys.path = [
    p for p in sys.path
    if "torchvision" not in p and not (".local" in p and Path(p).joinpath("torchvision").is_dir())
]

LLAMASIM_ROOT = "/data/llamasim"
CG_SIM_ROOT = "/data/cg-sim"
_wants_cgsim = any(a.startswith("cgsim_") for a in sys.argv)
if _wants_cgsim:
    if CG_SIM_ROOT not in sys.path:
        sys.path.insert(0, CG_SIM_ROOT)
else:
    if LLAMASIM_ROOT not in sys.path:
        sys.path.insert(0, LLAMASIM_ROOT)

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch._inductor.config as iconfig  # noqa: E402
from torch._inductor.weight_streaming.plan import load_io_schedule  # noqa: E402
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime  # noqa: E402
from profile_sd3_common import (  # noqa: E402
    DTYPE_BY_NAME,
    decode_latents_to_pil,
    load_pipeline,
)
from profile_sdxl_turbo_common import (  # noqa: E402
    configure_llamasim_inductor_markers,
    write_llamasim_runtime_bundle,
)


VARIANT_MODULES = {
    "baseline": "graph_modifiers.jit_sim_prune_compiled_tensor.scheduler",
    "ct_maxbytes": "graph_modifiers.jit_sim_prune_ct_maxbytes.scheduler",
    "ct_largest": "graph_modifiers.jit_sim_prune_ct_largest.scheduler",
    "ct_multichan": "graph_modifiers.jit_sim_prune_ct_multichan.scheduler",
    "ct_belady_pcie": "graph_modifiers.ct_belady_pcie.scheduler",
    "ct_belady_pcie_strict": "graph_modifiers.ct_belady_pcie_strict.scheduler",
}

CGSIM_VARIANT_MODULES = {
    "cgsim_belady": "graph_modifiers.ct_belady_pcie.scheduler",
    "cgsim_maxbytes": "graph_modifiers.ct_maxbytes.scheduler",
    "cgsim_belady_multigraph": "graph_modifiers.ct_belady_multigraph.scheduler",
    "cgsim_rcpsp": "graph_modifiers.ct_rcpsp.scheduler",
    "cgsim_mincost_flow": "graph_modifiers.ct_mincost_flow.scheduler",
    "cgsim_milp_aggregate": "graph_modifiers.ct_milp_aggregate.scheduler",
    "cgsim_milp_aggregate_duplex": "graph_modifiers.ct_milp_aggregate.scheduler",
    "cgsim_milp_oracle": "graph_modifiers.ct_milp_oracle.scheduler",
    "cgsim_milp_oracle_duplex": "graph_modifiers.ct_milp_oracle.scheduler",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/llamasim/models/sd-3.5-med")
    p.add_argument("--hw", required=True)
    p.add_argument(
        "--variant", default="ct_belady_pcie",
        choices=list(VARIANT_MODULES.keys())
        + list(CGSIM_VARIANT_MODULES.keys())
        + ["baseline_nows"],
    )
    p.add_argument("--work-dir", default="/tmp/sd3_e2e_out")
    p.add_argument("--prompt", default="a photo of a cat")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max-sequence-length", type=int, default=256)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--dtype", choices=tuple(DTYPE_BY_NAME.keys()), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable-evict", action="store_true")
    p.add_argument("--disable-safety-net", action="store_true", default=False)
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--num-channels", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    return p.parse_args()


def run_pipeline(pipe, args):
    return pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        max_sequence_length=args.max_sequence_length,
        output_type="latent",
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stage_profile(pipe, device, args, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    trace_path = bundle_dir.parent / "profile_trace.json"
    exec_trace_path = bundle_dir.parent / "profile_execution_trace.json"
    csv_path = bundle_dir.parent / "profile_trace.csv"

    from torch.profiler import profile as torch_profile
    from torch.profiler import ProfilerActivity

    execution_trace_observer = torch.profiler.ExecutionTraceObserver()
    execution_trace_observer.register_callback(str(exec_trace_path))
    print("→ running kineto-profiled inference for bundle generation ...")
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        execution_trace_observer=execution_trace_observer,
    ) as prof:
        with torch.autograd.profiler.record_function("sd3_run"):
            _ = run_pipeline(pipe, args)
        sync(device)
    execution_trace_observer.unregister_callback()

    prof.export_chrome_trace(str(trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(csv_path))

    print(f"→ exporting llamasim bundle to {bundle_dir}")
    write_llamasim_runtime_bundle(
        prof, exec_trace_path, bundle_dir, trace_json_path=trace_path,
    )

    hw_src = Path(args.hw)
    hw_dst = bundle_dir / hw_src.name
    if not hw_dst.exists():
        shutil.copy2(hw_src, hw_dst)


def stage_schedule(args, bundle_dir: Path) -> Path:
    if args.variant in CGSIM_VARIANT_MODULES:
        return _stage_schedule_cgsim(args, bundle_dir)

    from graph_modifiers.common.parser import load_problem, load_hw_params_full
    import glob

    module_path = VARIANT_MODULES[args.variant]
    mod = importlib.import_module(module_path)

    hw = load_hw_params_full(args.hw)
    print(f"→ loading problem from {bundle_dir} for scheduler {args.variant}")
    prob = load_problem(
        str(bundle_dir),
        dram_cap_mb=200.0, step=1, multi_graph=True, context_seq_len=2048,
        skip_first_file=False,
        ssd_read_bw=hw.ssd_read_bw, ssd_write_bw=hw.ssd_write_bw,
        ssd_read_latency_ns=hw.ssd_read_latency_ns,
        ssd_write_latency_ns=hw.ssd_write_latency_ns,
        h2d_bw=hw.h2d_bw, d2h_bw=hw.d2h_bw,
        h2d_latency_ns=hw.h2d_latency_ns, d2h_latency_ns=hw.d2h_latency_ns,
        cpu_per_launch_ns=hw.cpu_per_launch_ns,
    )
    # Load EVERY compile graph's sidecar (not just graph 0). The previous
    # code passed only graph[0] to solve(), leaving the scheduler blind to
    # every other compile (text encoders / transformer / vae). Here we
    # iterate and call solve() once per graph, masking non-target graph
    # GPU nodes as CPU so each solve sees only its own graph's timeline.
    import re
    pat = re.compile(r"compiled_tensor_map_graph(\d+)\.json$")
    tensor_paths = sorted(
        glob.glob(str(bundle_dir / "compiled_tensor_map_graph*.json"))
    )
    all_graphs: list[tuple[int, dict, dict]] = []
    for tp in tensor_paths:
        m = pat.search(tp)
        if not m:
            continue
        gid = int(m.group(1))
        lp = bundle_dir / f"compiled_launch_map_graph{gid}.json"
        if not lp.exists():
            continue
        all_graphs.append((gid, json.load(open(lp)), json.load(open(tp))))
    if not all_graphs:
        raise RuntimeError(f"No compile sidecars found in {bundle_dir}")
    print(
        f"→ found {len(all_graphs)} compile graphs: tensor counts="
        f"{[len(tm.get('tensors', [])) for _, _, tm in all_graphs]}"
    )

    # SD3 transformer input embedding: pos_embed.proj.weight / context_embedder.
    # Keep the common embedding-ish names locked so the scheduler doesn't evict them.
    kwargs = {"locked_tensor_names": {
        "pos_embed.proj.weight",
        "context_embedder.weight",
        "token_embd.weight",  # legacy, harmless
    }}
    if args.variant == "ct_multichan":
        kwargs["num_channels"] = args.num_channels

    GPU_KINDS = frozenset({"gpu_stream", "gpu"})
    merged_io_ops: list[dict] = []
    merged_cold: list[dict] = []
    summary_agg: dict = {
        "io_model": args.variant,
        "per_graph_summaries": [],
        "total_weight_bytes": 0,
        "peak_vram_bytes": 0,
        "vram_h2d_bytes": 0,
        "vram_d2h_bytes": 0,
        "vram_h2d_prefetches": 0,
        "vram_d2h_evictions": 0,
        "pinned_tensor_count": 0,
        "pinned_tensor_bytes": 0,
        "cyclable_tensor_count": 0,
        "h2d_ops": 0,
        "d2h_ops": 0,
        "compiled_tensor_count": 0,
    }
    compilation_hash = ""

    for gid, lm, tm in all_graphs:
        orig_kinds: dict[int, str] = {}
        for i, node in enumerate(prob.nodes):
            if (
                node.compiled_graph_id != gid
                and node.resource_kind in GPU_KINDS
            ):
                orig_kinds[i] = node.resource_kind
                node.resource_kind = "cpu_thread"
        try:
            sched_g = mod.solve(prob, launch_map=lm, tensor_map=tm, **kwargs)
        finally:
            for i, rk in orig_kinds.items():
                prob.nodes[i].resource_kind = rk

        for op in sched_g.get("io_operations", []):
            op["compiled_graph_id"] = gid
            merged_io_ops.append(op)
        for cop in sched_g.get("cold_start_prefetches", []):
            cop["compiled_graph_id"] = gid
            merged_cold.append(cop)

        s = sched_g.get("summary", {})
        summary_agg["per_graph_summaries"].append({"graph_id": gid, **s})
        for k in (
            "total_weight_bytes", "vram_h2d_bytes", "vram_d2h_bytes",
            "vram_h2d_prefetches", "vram_d2h_evictions",
            "pinned_tensor_count", "pinned_tensor_bytes",
            "cyclable_tensor_count", "h2d_ops", "d2h_ops",
            "compiled_tensor_count",
        ):
            summary_agg[k] += int(s.get(k, 0))
        summary_agg["peak_vram_bytes"] = max(
            summary_agg["peak_vram_bytes"], int(s.get("peak_vram_bytes", 0))
        )
        if not compilation_hash:
            compilation_hash = sched_g.get("compilation_hash", "")
        print(
            f"→ graph {gid}: {s.get('compiled_tensor_count', 0)} tensors, "
            f"{s.get('total_weight_bytes', 0) / 1e6:.1f} MB, "
            f"H2D={s.get('h2d_ops', 0)} D2H={s.get('d2h_ops', 0)}"
        )

    merged = {
        "summary": summary_agg,
        "nodes": [],  # not consumed by the runtime
        "io_operations": merged_io_ops,
        "spill_decisions": [],
        "cold_start_prefetches": merged_cold,
        "steady_state_resident": [],
    }
    if compilation_hash:
        merged["compilation_hash"] = compilation_hash

    out_path = bundle_dir / "jit_sim_prune_schedule.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(
        f"→ merged multi-graph schedule written to {out_path}: "
        f"{len(merged_io_ops)} io_ops, {len(merged_cold)} cold_starts, "
        f"total_weights={summary_agg['total_weight_bytes']/1e6:.1f} MB"
    )
    return out_path


_MULTIGRAPH_SD3_VARIANTS = (
    "cgsim_belady_multigraph", "cgsim_rcpsp", "cgsim_mincost_flow",
    "cgsim_milp_aggregate", "cgsim_milp_aggregate_duplex",
    "cgsim_milp_oracle", "cgsim_milp_oracle_duplex",
)


def _stage_schedule_cgsim(args, bundle_dir: Path) -> Path:
    import importlib as _imp
    from graph_modifiers.common import (
        load_trace_from_bundle, load_sidecars, load_multi_graph_sidecars,
        load_hw_params, write_schedule_json,
    )
    module_path = CGSIM_VARIANT_MODULES[args.variant]
    mod = _imp.import_module(module_path)
    trace = load_trace_from_bundle(str(bundle_dir))
    hw = load_hw_params(args.hw)
    if args.variant in _MULTIGRAPH_SD3_VARIANTS:
        sidecars = load_multi_graph_sidecars(str(bundle_dir))
        if not sidecars.launch_maps:
            raise RuntimeError(f"no compile sidecars in {bundle_dir}")
        kwargs: dict = {}
        if args.variant == "cgsim_mincost_flow":
            kwargs["alpha"] = args.alpha
        if args.variant in ("cgsim_milp_aggregate_duplex", "cgsim_milp_oracle_duplex"):
            kwargs["duplex"] = True
        # Per-graph multiplicity for SD3-med pipeline (compile order
        # observed: text encoders + transformer + VAE = 4 graphs).
        # The transformer (gid=2) runs 2*steps times due to CFG; others
        # once each. Lets ct_milp_oracle's mult-gated cross-graph
        # check filter out unsafe issuer placements.
        if args.variant in ("cgsim_milp_oracle", "cgsim_milp_oracle_duplex"):
            kwargs["graph_multiplicity"] = {
                0: 1, 1: 1, 2: 2 * args.steps, 3: 1
            }
        result = mod.solve(
            trace, sidecars=sidecars, hw=hw,
            locked_graph_input_names={"pos_embed.proj.weight", "context_embedder.weight"},
            **kwargs,
        )
    else:
        sidecars_one = load_sidecars(str(bundle_dir))
        kwargs = {}
        if args.variant == "cgsim_belady":
            kwargs["cap_bytes"] = None
        result = mod.solve(
            trace,
            launch_map=sidecars_one.launch_map,
            tensor_map=sidecars_one.tensor_map,
            hw=hw,
            locked_graph_input_names={"pos_embed.proj.weight", "context_embedder.weight"},
            **kwargs,
        )
    if hasattr(mod, "print_summary"):
        mod.print_summary(result)
    out_path = bundle_dir / "jit_sim_prune_schedule.json"
    write_schedule_json(
        out_path,
        trace=trace,
        node_starts=result.node_starts,
        node_ends=result.node_ends,
        io_operations=result.io_operations,
        cold_start_prefetches=result.cold_start_prefetches,
        summary=result.summary,
        compilation_hash=result.compilation_hash,
    )
    print(f"→ schedule written to {out_path}")
    return out_path


def stage_run(pipe, device, args, schedule_path: Path, bundle_dir: Path):
    import os
    if args.disable_evict:
        os.environ["TORCH_WS_DISABLE_EVICT"] = "1"
    if args.disable_safety_net:
        os.environ["TORCH_WS_DISABLE_SAFETY_NET"] = "1"

    iconfig.weight_streaming_plan = str(schedule_path)
    nodes_csv = bundle_dir / "runtime_nodes.csv"
    tensor_csv = bundle_dir / "pytorch_runtime_tensors.csv"
    if nodes_csv.exists():
        iconfig.weight_streaming_nodes_csv = str(nodes_csv)
    if tensor_csv.exists():
        iconfig.weight_streaming_tensor_csv = str(tensor_csv)

    schedule = load_io_schedule(
        str(schedule_path),
        nodes_csv=str(nodes_csv) if nodes_csv.exists() else "",
        tensor_csv=str(tensor_csv) if tensor_csv.exists() else "",
    )
    WeightStreamRuntime.reset()
    rt = WeightStreamRuntime.initialize(schedule, device)

    # Register weights across every SD3 sub-module.
    # Register BOTH the Parameter object AND its underlying tensor (param.data).
    # Inductor can bind either one as the graph input depending on the
    # compilation path, so registering both keeps id()-keyed lookups stable.
    def _register(module):
        if module is None:
            return
        under = module._orig_mod if hasattr(module, "_orig_mod") else module
        for param in under.parameters():
            rt.register_weight(param)
            rt.register_weight(param.data)
        for buf in under.buffers():
            rt.register_weight(buf)
            rt.register_weight(buf.data)

    _register(pipe.transformer)
    _register(getattr(pipe, "text_encoder", None))
    _register(getattr(pipe, "text_encoder_2", None))
    _register(getattr(pipe, "text_encoder_3", None))
    _register(getattr(pipe, "vae", None))

    torch._dynamo.reset()
    import torch._inductor.codegen.wrapper as _wrapper_mod
    _wrapper_mod._ws_compile_pos = 0

    def _recompile(attr_name, sub_attr=None):
        m = getattr(pipe, attr_name, None)
        if m is None:
            return
        if sub_attr is not None:
            m = getattr(m, sub_attr, None)
            if m is None:
                return
        under = m._orig_mod if hasattr(m, "_orig_mod") else m
        recompiled = torch.compile(under, backend="inductor", mode="default")
        if sub_attr is not None:
            setattr(getattr(pipe, attr_name), sub_attr, recompiled)
        else:
            setattr(pipe, attr_name, recompiled)

    _recompile("transformer")
    _recompile("text_encoder")
    _recompile("text_encoder_2")
    _recompile("text_encoder_3")
    _recompile("vae", "decode")

    print("→ warmup recompile run ...")
    _ = run_pipeline(pipe, args)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if hasattr(torch.ops.ws_rt, "get_wait_stats"):
            torch.ops.ws_rt.get_wait_stats()
    print("→ timed inference run ...")
    t0 = time.perf_counter()
    _ = run_pipeline(pipe, args)
    sync(device)
    elapsed = time.perf_counter() - t0

    alloc_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    wait_stats = None
    if device.type == "cuda" and hasattr(torch.ops.ws_rt, "get_wait_stats"):
        wait_stats = list(torch.ops.ws_rt.get_wait_stats())
    return {"allocated_mb": alloc_mb, "peak_mb": peak_mb, "inference_s": elapsed,
            "wait_stats": wait_stats}


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = DTYPE_BY_NAME[args.dtype]

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    bundle_dir = work / "llama_bundle"

    iconfig.weight_streaming_plan = ""
    iconfig.force_disable_caches = True
    configure_llamasim_inductor_markers(bundle_dir, include_diagnostic_markers=True)

    print(f"→ loading pipeline from {args.model}")
    pipe = load_pipeline(args.model, dtype, device, vae_tiling=args.vae_tiling)
    pipe.set_progress_bar_config(disable=True)
    # Disable the SDXL-style fp32 upcast dance around VAE for the same reason
    # the SDXL e2e does it.
    if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "config"):
        pipe.vae.config.force_upcast = False

    pipe.transformer = torch.compile(pipe.transformer, backend="inductor", mode="default")
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            setattr(pipe, attr, torch.compile(mod, backend="inductor", mode="default"))
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.decode = torch.compile(pipe.vae.decode, backend="inductor", mode="default")

    print("→ warmup run 1 (triggers first compile) ...")
    _ = run_pipeline(pipe, args)
    sync(device)

    if args.variant == "baseline_nows":
        print("→ baseline warmup 2 ...")
        _ = run_pipeline(pipe, args)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print("→ baseline timed run ...")
        t0 = time.perf_counter()
        _ = run_pipeline(pipe, args)
        sync(device)
        elapsed = time.perf_counter() - t0
        alloc_mb = torch.cuda.memory_allocated(device) / 1e6
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print("\n=== RESULTS ===")
        print(f"variant        : baseline_nows")
        print(f"model          : {args.model}")
        print(f"steps          : {args.steps}")
        print(f"resolution     : {args.height}x{args.width}")
        print(f"allocated_mb   : {alloc_mb:.1f}")
        print(f"peak_mb        : {peak_mb:.1f}")
        print(f"inference_s    : {elapsed:.4f}")
        return

    stage_profile(pipe, device, args, bundle_dir)
    schedule_path = stage_schedule(args, bundle_dir)
    results = stage_run(pipe, device, args, schedule_path, bundle_dir)

    print("\n=== RESULTS ===")
    print(f"variant        : {args.variant}")
    print(f"model          : {args.model}")
    print(f"steps          : {args.steps}")
    print(f"resolution     : {args.height}x{args.width}")
    print(f"allocated_mb   : {results['allocated_mb']:.1f}")
    print(f"peak_mb        : {results['peak_mb']:.1f}")
    print(f"inference_s    : {results['inference_s']:.4f}")
    import os as _os
    if _os.environ.get("TORCH_WS_LOG_DISPATCH"):
        s = results.get("wait_stats")
        if s is not None:
            print(
                f"ws_wait_stats  : hit={s[0]} miss_synced={s[1]} miss_resident={s[2]} "
                f"fire_async={s[3]} fire_skip_inflight={s[4]} fire_skip_resident={s[5]}"
            )
            if len(s) >= 9:
                print(f"ws_xg_stats    : dispatched={s[6]} named_miss={s[7]} rw_miss={s[8]}")
    WeightStreamRuntime.reset()


if __name__ == "__main__":
    main()
