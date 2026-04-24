#!/usr/bin/env python3
"""Unified profile + schedule + run pipeline for Llama weight streaming.

Mirrors scripts/streaming_e2e.py but drives a single HuggingFace causal-LM
(`AutoModelForCausalLM`) instead of SDXL's multi-component pipeline. Same
rationale: run profile + run in one process so the kernel ordering used to
produce the schedule matches the ordering used at run time.

Stages:
  1. Load Llama pipeline.
  2. Configure Inductor for profile (emit_ids=True, plan=empty, markers on).
  3. Compile model; warmup.
  4. Run once under kineto + execution-trace, export llamasim bundle.
  5. Invoke a scheduler variant on the bundle → jit_sim_prune_schedule.json.
  6. Point Inductor at the schedule, dynamo.reset(), recompile.
  7. Timed inference; record inference time and peak VRAM.

Usage:
  python3 scripts/streaming_e2e_llama.py \\
      --model meta-llama/Llama-3.2-3B \\
      --hw /data/pytorch-source/sdxl-turbo-profile-0414/llama_bundle/hw_pcie4.json \\
      --variant ct_belady_pcie \\
      --work-dir /tmp/llama_e2e_out
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
from profile_llama_common import (  # noqa: E402
    DTYPE_BY_NAME,
    encode_prompt,
    greedy_generate,
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
    "cgsim_steady_state": "graph_modifiers.ct_steady_state.scheduler",
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
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--hw", required=True)
    p.add_argument(
        "--variant",
        default="ct_belady_pcie",
        choices=list(VARIANT_MODULES.keys())
        + list(CGSIM_VARIANT_MODULES.keys())
        + ["baseline_nows"],
    )
    p.add_argument("--work-dir", default="/tmp/llama_e2e_out")
    p.add_argument("--prompt", default="The quick brown fox")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--dtype", choices=tuple(DTYPE_BY_NAME.keys()), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable-evict", action="store_true")
    p.add_argument("--disable-safety-net", action="store_true", default=False)
    p.add_argument("--num-channels", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=True,
        help="Compile with dynamic shapes (default).",
    )
    p.add_argument(
        "--no-dynamic", dest="dynamic", action="store_false",
    )
    return p.parse_args()


def run_pipeline(pipe, args):
    input_ids = encode_prompt(pipe, args)
    with torch.no_grad():
        sequences = greedy_generate(
            pipe.model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=pipe.tokenizer.eos_token_id,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
    return sequences


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
        with torch.autograd.profiler.record_function("llama_run"):
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
        dram_cap_mb=200.0,
        step=1,
        multi_graph=True,
        context_seq_len=2048,
        skip_first_file=False,
        ssd_read_bw=hw.ssd_read_bw,
        ssd_write_bw=hw.ssd_write_bw,
        ssd_read_latency_ns=hw.ssd_read_latency_ns,
        ssd_write_latency_ns=hw.ssd_write_latency_ns,
        h2d_bw=hw.h2d_bw,
        d2h_bw=hw.d2h_bw,
        h2d_latency_ns=hw.h2d_latency_ns,
        d2h_latency_ns=hw.d2h_latency_ns,
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

    # For HF Llama, the embedding weight graph-input name is typically
    # "model.embed_tokens.weight"; locking it prevents the scheduler from
    # evicting it (embedding lookups don't show up as kernels we can gate).
    kwargs = {"locked_tensor_names": {
        "model.embed_tokens.weight",
        "token_embd.weight",  # legacy llamasim name, harmless to include
    }}
    if args.variant == "ct_multichan":
        kwargs["num_channels"] = args.num_channels
    schedule = mod.solve(prob, launch_map=launch_map, tensor_map=tensor_map, **kwargs)
    if hasattr(mod, "print_summary"):
        mod.print_summary(schedule)

    out_path = bundle_dir / "jit_sim_prune_schedule.json"
    with open(out_path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"→ schedule written to {out_path}")
    return out_path


_MULTIGRAPH_LLAMA_VARIANTS = (
    "cgsim_belady_multigraph", "cgsim_rcpsp", "cgsim_mincost_flow",
    "cgsim_milp_aggregate", "cgsim_milp_aggregate_duplex",
    "cgsim_milp_oracle", "cgsim_milp_oracle_duplex",
)


def _stage_schedule_cgsim(args, bundle_dir: Path) -> Path:
    import importlib as _imp
    from graph_modifiers.common import (
        load_trace_from_bundle,
        load_sidecars,
        load_multi_graph_sidecars,
        load_hw_params,
        write_schedule_json,
    )

    module_path = CGSIM_VARIANT_MODULES[args.variant]
    mod = _imp.import_module(module_path)

    print(f"→ [cg-sim] loading trace from {bundle_dir}")
    trace = load_trace_from_bundle(str(bundle_dir))
    hw = load_hw_params(args.hw)

    if args.variant in _MULTIGRAPH_LLAMA_VARIANTS:
        sidecars = load_multi_graph_sidecars(str(bundle_dir))
        if not sidecars.launch_maps:
            raise RuntimeError(
                f"[cg-sim variant] bundle {bundle_dir} has no compile sidecars."
            )
        kwargs: dict = {}
        if args.variant == "cgsim_mincost_flow":
            kwargs["alpha"] = args.alpha
        if args.variant in ("cgsim_milp_aggregate_duplex", "cgsim_milp_oracle_duplex"):
            kwargs["duplex"] = True
        result = mod.solve(
            trace,
            sidecars=sidecars,
            hw=hw,
            locked_graph_input_names={"model.embed_tokens.weight"},
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
            locked_graph_input_names={"model.embed_tokens.weight"},
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

    under = pipe.model._orig_mod if hasattr(pipe.model, "_orig_mod") else pipe.model
    for param in under.parameters():
        rt.register_weight(param)
    for buf in under.buffers():
        rt.register_weight(buf)

    torch._dynamo.reset()
    import torch._inductor.codegen.wrapper as _wrapper_mod
    _wrapper_mod._ws_compile_pos = 0

    pipe.model = torch.compile(
        under, backend="inductor", mode="default", dynamic=args.dynamic,
    )

    print("→ warmup recompile run ...")
    _ = run_pipeline(pipe, args)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print("→ timed inference run ...")
    t0 = time.perf_counter()
    _ = run_pipeline(pipe, args)
    sync(device)
    elapsed = time.perf_counter() - t0

    alloc_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    return {"allocated_mb": alloc_mb, "peak_mb": peak_mb, "inference_s": elapsed}


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
    pipe = load_pipeline(args.model, dtype, device)
    pipe.model = torch.compile(
        pipe.model, backend="inductor", mode="default", dynamic=args.dynamic,
    )

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
        print(f"max_new_tokens : {args.max_new_tokens}")
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
    print(f"max_new_tokens : {args.max_new_tokens}")
    print(f"allocated_mb   : {results['allocated_mb']:.1f}")
    print(f"peak_mb        : {results['peak_mb']:.1f}")
    print(f"inference_s    : {results['inference_s']:.4f}")
    WeightStreamRuntime.reset()


if __name__ == "__main__":
    main()
