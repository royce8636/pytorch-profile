#!/usr/bin/env python3
"""Unified profile + schedule + run pipeline for weight streaming.

Same-process execution so the profile compile and the run compile share
deterministic kernel ordering. This avoids the compile-schedule structural
mismatch that crashes when profile and run are in separate processes.

Stages:
  1. Load SDXL pipeline.
  2. Configure Inductor for profile: emit_ids=True, plan=empty, markers off.
  3. Compile unet; run warmup.
  4. Run once under kineto + execution-trace, export llamasim bundle.
  5. Invoke a scheduler variant on the bundle → schedule.json.
  6. Set weight_streaming_plan=schedule, dynamo.reset(), recompile.
  7. Run timed inference, measure peak VRAM.

Usage:
  python3 scripts/streaming_e2e.py \\
      --model /data/llamasim/models/sdxl-1.0-base \\
      --hw <...>/hw_pcie4.json \\
      --variant ct_belady_pcie \\
      --work-dir /tmp/streaming_e2e_out
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import os
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

# /data/llamasim has both a ``graph_modifiers/`` package AND a ``sim.py``
# module. /data/cg-sim has ``sim/`` as a namespace package. Putting llamasim
# first shadows cg-sim's ``sim`` with llamasim's module. We detect which
# backend the user asked for and insert paths in the right order.
LLAMASIM_ROOT = "/data/llamasim"
CG_SIM_ROOT = "/data/cg-sim"
_wants_cgsim = any(a.startswith("cgsim_") for a in sys.argv)
if _wants_cgsim:
    if CG_SIM_ROOT not in sys.path:
        sys.path.insert(0, CG_SIM_ROOT)
    # Don't insert llamasim at all (its sim.py would shadow cg-sim's sim/).
else:
    if LLAMASIM_ROOT not in sys.path:
        sys.path.insert(0, LLAMASIM_ROOT)

import torch
import torch._dynamo
import torch._inductor.config as iconfig
from torch._inductor.weight_streaming.plan import IOSchedule, load_io_schedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime
from profile_sdxl_turbo_common import (
    configure_llamasim_inductor_markers,
    decode_latents_to_pil,
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

# Variants that use the cg-sim backend instead of llamasim's custom parser.
# These take a cg-sim ``Trace`` built from ``sim.load.pytorch_profile`` plus
# the compile sidecars (launch_map, tensor_map), and emit a schedule in the
# same JSON schema as the legacy variants.
CGSIM_VARIANT_MODULES = {
    "cgsim_belady": "graph_modifiers.ct_belady_pcie.scheduler",
    "cgsim_maxbytes": "graph_modifiers.ct_maxbytes.scheduler",
    "cgsim_steady_state": "graph_modifiers.ct_steady_state.scheduler",
    "cgsim_spread_cycle": "graph_modifiers.ct_spread_cycle.scheduler",
    "cgsim_multigraph": "graph_modifiers.ct_multigraph.scheduler",
    "cgsim_graph_swap": "graph_modifiers.ct_graph_swap.scheduler",
    "cgsim_graph_swap_noevict": "graph_modifiers.ct_graph_swap.scheduler",
    # Cross-graph algorithmic schedulers (multi-graph, non-heuristic).
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
    p.add_argument(
        "--model", default="/data/llamasim/models/sdxl-1.0-base",
    )
    p.add_argument("--hw", required=True)
    p.add_argument("--variant", default="ct_belady_pcie",
                   choices=list(VARIANT_MODULES.keys())
                   + list(CGSIM_VARIANT_MODULES.keys())
                   + ["baseline_nows"])
    p.add_argument(
        "--work-dir", default="/tmp/streaming_e2e_out",
        help="Where bundle + schedule + outputs go.",
    )
    p.add_argument("--prompt", default="a lovely cat")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--disable-evict", action="store_true",
                   help="Set TORCH_WS_DISABLE_EVICT equivalent to test h2d only.")
    p.add_argument("--disable-safety-net", action="store_true", default=False)
    p.add_argument(
        "--num-channels", type=int, default=4,
        help="For ct_multichan: number of h2d channels.",
    )
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help="For cgsim_mincost_flow: 0..1 peak-vs-stall Pareto weight.",
    )
    return p.parse_args()


def load_pipeline(model: str, dtype: torch.dtype, device: torch.device):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model, torch_dtype=dtype, use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_pipeline(pipe, args):
    return pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        height=args.height,
        width=args.width,
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stage_profile(
    pipe, device, args, bundle_dir: Path
) -> None:
    """Run once under kineto and emit the llamasim bundle."""
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
        with torch.autograd.profiler.record_function("sdxl_run"):
            output = run_pipeline(pipe, args)
        sync(device)
    execution_trace_observer.unregister_callback()

    prof.export_chrome_trace(str(trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(csv_path))

    print(f"→ exporting llamasim bundle to {bundle_dir}")
    write_llamasim_runtime_bundle(
        prof, exec_trace_path, bundle_dir, trace_json_path=trace_path,
    )

    # Copy HW config into bundle_dir (scheduler expects it there OR via --hw arg)
    hw_src = Path(args.hw)
    hw_dst = bundle_dir / hw_src.name
    if not hw_dst.exists():
        shutil.copy2(hw_src, hw_dst)

    del output


def stage_schedule(
    args, bundle_dir: Path
) -> Path:
    """Invoke the selected scheduler variant on the bundle; return schedule path."""
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
        dram_cap_mb=200.0, step=1, multi_graph=True,
        context_seq_len=2048, skip_first_file=False,
        ssd_read_bw=hw.ssd_read_bw, ssd_write_bw=hw.ssd_write_bw,
        ssd_read_latency_ns=hw.ssd_read_latency_ns,
        ssd_write_latency_ns=hw.ssd_write_latency_ns,
        h2d_bw=hw.h2d_bw, d2h_bw=hw.d2h_bw,
        h2d_latency_ns=hw.h2d_latency_ns, d2h_latency_ns=hw.d2h_latency_ns,
        cpu_per_launch_ns=hw.cpu_per_launch_ns,
    )
    launch_map_paths = sorted(
        glob.glob(str(bundle_dir / "compiled_launch_map_graph*.json"))
    )
    tensor_map_paths = sorted(
        glob.glob(str(bundle_dir / "compiled_tensor_map_graph*.json"))
    )
    launch_map = json.load(open(launch_map_paths[0])) if launch_map_paths else None
    tensor_map = json.load(open(tensor_map_paths[0])) if tensor_map_paths else None

    kwargs = {"locked_tensor_names": {"token_embd.weight"}}
    if args.variant == "ct_multichan":
        kwargs["num_channels"] = args.num_channels
    schedule = mod.solve(
        prob, launch_map=launch_map, tensor_map=tensor_map, **kwargs,
    )
    if hasattr(mod, "print_summary"):
        mod.print_summary(schedule)

    out_path = bundle_dir / "jit_sim_prune_schedule.json"
    with open(out_path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"→ schedule written to {out_path}")
    return out_path


def _stage_schedule_cgsim(args, bundle_dir: Path) -> Path:
    """Path for the cg-sim-backed scheduler variants."""
    import importlib as _imp
    import glob
    import json
    from graph_modifiers.common import (
        load_trace_from_bundle, load_sidecars, load_multi_graph_sidecars,
        load_hw_params, write_schedule_json,
    )

    module_path = CGSIM_VARIANT_MODULES[args.variant]
    mod = _imp.import_module(module_path)

    print(f"→ [cg-sim] loading trace from {bundle_dir}")
    trace = load_trace_from_bundle(str(bundle_dir))
    hw = load_hw_params(args.hw)

    # Multi-graph variants take a MultiGraphSidecars; others take a single pair.
    _MULTIGRAPH_VARIANTS = (
        "cgsim_multigraph", "cgsim_graph_swap", "cgsim_graph_swap_noevict",
        "cgsim_belady_multigraph", "cgsim_rcpsp", "cgsim_mincost_flow",
        "cgsim_milp_aggregate", "cgsim_milp_aggregate_duplex",
        "cgsim_milp_oracle", "cgsim_milp_oracle_duplex",
    )
    if args.variant in _MULTIGRAPH_VARIANTS:
        sidecars = load_multi_graph_sidecars(str(bundle_dir))
        if not sidecars.launch_maps:
            raise RuntimeError(
                f"[cg-sim variant] bundle {bundle_dir} has no compile sidecars."
            )
        kwargs = {}
        if args.variant == "cgsim_graph_swap_noevict":
            kwargs["include_last_evict"] = False
        if args.variant == "cgsim_mincost_flow":
            kwargs["alpha"] = args.alpha
        if args.variant in ("cgsim_milp_aggregate_duplex", "cgsim_milp_oracle_duplex"):
            kwargs["duplex"] = True
        result = mod.solve(
            trace,
            sidecars=sidecars,
            hw=hw,
            locked_graph_input_names={"token_embd.weight"},
            **kwargs,
        )
    else:
        sidecars_one = load_sidecars(str(bundle_dir))
        if sidecars_one.launch_map is None or sidecars_one.tensor_map is None:
            raise RuntimeError(
                f"[cg-sim variant] bundle {bundle_dir} missing compile sidecars."
            )
        kwargs = {}
        if args.variant == "cgsim_belady":
            kwargs["cap_bytes"] = None
        result = mod.solve(
            trace,
            launch_map=sidecars_one.launch_map,
            tensor_map=sidecars_one.tensor_map,
            hw=hw,
            locked_graph_input_names={"token_embd.weight"},
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


def stage_run(
    pipe, device, args, schedule_path: Path, bundle_dir: Path
):
    """Recompile with schedule applied and run timed inference."""
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

    # Load the schedule to init runtime.
    schedule = load_io_schedule(
        str(schedule_path),
        nodes_csv=str(nodes_csv) if nodes_csv.exists() else "",
        tensor_csv=str(tensor_csv) if tensor_csv.exists() else "",
    )
    # Reset any prior runtime instance (we compiled before).
    WeightStreamRuntime.reset()
    rt = WeightStreamRuntime.initialize(schedule, device)

    # Register GPU weight tensors across ALL pipeline components so a
    # multi-graph schedule can touch any of them.
    def _register(module):
        if module is None:
            return
        under = module._orig_mod if hasattr(module, "_orig_mod") else module
        for param in under.parameters():
            rt.register_weight(param)
        for buf in under.buffers():
            rt.register_weight(buf)

    _register(pipe.unet)
    _register(getattr(pipe, "text_encoder", None))
    _register(getattr(pipe, "text_encoder_2", None))
    _register(getattr(pipe, "vae", None))

    # Force recompile on all components with schedule injected.
    torch._dynamo.reset()
    # Reset the multi-graph compile-position counter so our scheduler's
    # compiled_graph_id tags (0, 1, 2, ...) map to the fresh compile
    # sequence (TE first, TE2 second, UNet third, ...).
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

    _recompile("unet")
    _recompile("text_encoder")
    _recompile("text_encoder_2")
    _recompile("vae", "decode")

    # Warmup to trigger compile.
    print("→ warmup recompile run ...")
    _ = run_pipeline(pipe, args)
    sync(device)

    # (Pre-evict of non-current-graph weights was here — removed because
    # the compiled wrappers don't emit reload ops to bring them back.)

    # Timed run with peak tracking.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print("→ timed inference run ...")
    t0 = time.perf_counter()
    output = run_pipeline(pipe, args)
    sync(device)
    elapsed = time.perf_counter() - t0

    alloc_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    return {
        "allocated_mb": alloc_mb, "peak_mb": peak_mb, "inference_s": elapsed,
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    bundle_dir = work / "llama_bundle"

    # Stage 0: Configure Inductor for PROFILE mode.
    iconfig.weight_streaming_plan = ""
    # Force fresh compile so wrapper.py runs and emits the sidecars
    # (compiled_launch_map, compiled_tensor_map). A cache hit would skip
    # codegen entirely and leave the bundle dir without sidecars.
    iconfig.force_disable_caches = True

    # Phase 1 reconciliation fix: freeze kernel selection so the profile
    # compile and the post-dynamo.reset run compile produce byte-identical
    # generated code + sidecars. Without this, autotune / coordinate-
    # descent tuning picks different kernels across compiles, drifting
    # `used_by_launch_ids` per tensor — which makes the schedule unsafe
    # for aggressive evictions (the run compile may reference a tensor
    # past the schedule's last_use claim → CUDA illegal access).
    iconfig.max_autotune = False
    iconfig.coordinate_descent_tuning = False
    iconfig.freezing = False           # constant folding varies across compiles
    import torch._dynamo.config as _dcfg
    _dcfg.automatic_dynamic_shapes = False
    _dcfg.assume_static_by_default = True
    torch.manual_seed(0)

    configure_llamasim_inductor_markers(bundle_dir, include_diagnostic_markers=True)

    # Load + compile ALL major components so the profile captures every
    # graph (text encoder, text encoder 2, unet, vae). Matches what
    # profile_sdxl_base_gpu.py's maybe_compile does.
    print(f"→ loading pipeline from {args.model}")
    pipe = load_pipeline(args.model, dtype, device)
    # Disable the SDXL fp32-upcast dance around VAE.decode. When enabled the
    # pipeline calls self.vae.to(dtype=torch.float16) AFTER VAE.decode, which
    # reads every VAE param — but weight streaming has already evicted them
    # (resize_(0)) at VAE's tail, so the downcast hits freed storage.
    if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "config"):
        pipe.vae.config.force_upcast = False
    pipe.unet = torch.compile(
        pipe.unet, backend="inductor", mode="default",
        fullgraph=args.fullgraph if hasattr(args, "fullgraph") else False,
    )
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = torch.compile(
            pipe.text_encoder, backend="inductor", mode="default",
        )
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2 = torch.compile(
            pipe.text_encoder_2, backend="inductor", mode="default",
        )
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.decode = torch.compile(
            pipe.vae.decode, backend="inductor", mode="default",
        )

    # Warmup compiles
    print("→ warmup run 1 (triggers first compile) ...")
    _ = run_pipeline(pipe, args)
    sync(device)

    if args.variant == "baseline_nows":
        # No weight streaming: warmup once more, then timed run.
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
        print(f"allocated_mb   : {alloc_mb:.1f}")
        print(f"peak_mb        : {peak_mb:.1f}")
        print(f"inference_s    : {elapsed:.4f}")
        return

    # Stage 1: profile (one more run, captured under kineto)
    stage_profile(pipe, device, args, bundle_dir)

    # Stage 2: schedule
    schedule_path = stage_schedule(args, bundle_dir)

    # Stage 3: run with schedule
    results = stage_run(pipe, device, args, schedule_path, bundle_dir)

    print("\n=== RESULTS ===")
    print(f"variant        : {args.variant}")
    print(f"allocated_mb   : {results['allocated_mb']:.1f}")
    print(f"peak_mb        : {results['peak_mb']:.1f}")
    print(f"inference_s    : {results['inference_s']:.4f}")
    if os.environ.get("TORCH_WS_LOG_DISPATCH"):
        import torch._inductor.weight_streaming.runtime as _rt
        cpp = getattr(_rt, "_ws_cpp_hits", 0)
        py = getattr(_rt, "_ws_py_hits", 0)
        print(f"ws_dispatch    : cpp={cpp} python={py}")
    WeightStreamRuntime.reset()


if __name__ == "__main__":
    main()
