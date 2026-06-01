#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_llama_common as llama_common  # noqa: E402
import run_llama_decoder_layer_cpu_offload as offload_common  # noqa: E402
from profile_sdxl_turbo_common import (  # noqa: E402
    build_module_catalog,
    build_module_id_to_path_from_prof,
    build_pipeline_module_index_from_prof,
    print_llamasim_runtime_summary,
    profile_activities,
    profiler_synchronize_device,
    write_llamasim_runtime_bundle,
)


@dataclass(frozen=True)
class OutputPaths:
    trace_path: Path
    csv_path: Path
    text_path: Path
    execution_trace_path: Path
    llamasim_output_dir: Path


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace", default=None)
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--execution-trace", default=None)
    parser.add_argument("--llamasim-output-dir", default=None)
    parser.add_argument(
        "--record-shapes",
        dest="record_shapes",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-record-shapes", dest="record_shapes", action="store_false")
    parser.add_argument(
        "--profile-memory",
        dest="profile_memory",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-profile-memory", dest="profile_memory", action="store_false")
    parser.add_argument(
        "--with-stack",
        dest="with_stack",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-with-stack", dest="with_stack", action="store_false")
    parser.add_argument(
        "--with-modules",
        dest="with_modules",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-with-modules", dest="with_modules", action="store_false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Llama with HF Accelerate decoder-layer CPU offload. "
            "Profiling starts after model load and warmup."
        )
    )
    offload_common.add_decoder_layer_args(
        parser,
        default_model="/data/llamasim/models/Llama-3.1-8b",
        default_dtype="bfloat16",
        default_output_prefix=None,
        default_warmup_runs=1,
        default_max_new_tokens=15,
    )
    add_profile_args(parser)
    return parser.parse_args()


def output_stem(args: argparse.Namespace) -> str:
    output_prefix = getattr(args, "output_prefix", None)
    if output_prefix is not None:
        return llama_common.output_stem(output_prefix, args.fusion)
    return f"{offload_common.output_stem_base(args)}_profile"


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    stem = output_stem(args)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        trace_path = output_dir / f"{stem}_trace.json"
        csv_path = output_dir / f"{stem}_trace.csv"
        text_path = (
            Path(args.text_output)
            if args.text_output is not None
            else output_dir / f"{stem}_output.txt"
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace is not None
            else output_dir / f"{stem}_execution_trace.json"
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir is not None
            else output_dir / "llama_bundle"
        )
    else:
        trace_path = (
            Path(args.trace)
            if args.trace is not None
            else Path("/tmp") / f"{stem}_trace.json"
        )
        csv_path = (
            Path(args.trace_csv)
            if args.trace_csv is not None
            else trace_path.with_suffix(".csv")
        )
        text_path = (
            Path(args.text_output)
            if args.text_output is not None
            else trace_path.with_name(f"{stem}_output.txt")
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace is not None
            else trace_path.with_name(f"{stem}_execution_trace.json")
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir is not None
            else trace_path.with_name(f"{stem}_llamasim_runtime")
        )
    for path in (trace_path, csv_path, text_path, execution_trace_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    llamasim_output_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        trace_path=trace_path,
        csv_path=csv_path,
        text_path=text_path,
        execution_trace_path=execution_trace_path,
        llamasim_output_dir=llamasim_output_dir,
    )


def record_function_name(args: argparse.Namespace) -> str:
    del args
    return "llama_decoder_layer_cpu_offload_run"


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"model_family=llama8b device={args.device} dtype={args.dtype} "
        f"offload_mode={args.offload_mode} fusion={args.fusion} "
        f"preload_module_classes={args.preload_module_classes}"
    )


def main() -> None:
    args = parse_args()
    device = offload_common.validate_run_device(args.device)
    offload_common.validate_fusion_runtime(args, device)
    accelerate_version = offload_common.ensure_accelerate_available()
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
    torch_dtype = llama_common.DTYPE_BY_NAME[args.dtype]
    output_paths = resolve_output_paths(args)
    classes = offload_common.preload_module_classes(args)

    pipe = offload_common.load_pipeline(
        args.model,
        torch_dtype,
        device,
        offload_mode=args.offload_mode,
    )
    llama_common.configure_llamasim_inductor_markers(output_paths.llamasim_output_dir)
    offload_common.maybe_compile(pipe, args)
    offload_common.apply_decoder_layer_cpu_offload(pipe, device, classes=classes)
    for _ in range(args.warmup_runs):
        warmup_output = llama_common.run_pipeline(pipe, args)
        llama_common.synchronize_device(device)
        del warmup_output

    execution_trace_observer = torch.profiler.ExecutionTraceObserver()
    execution_trace_observer.register_callback(str(output_paths.execution_trace_path))
    try:
        with torch.profiler.profile(
            activities=profile_activities(device),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            with_modules=args.with_modules,
            execution_trace_observer=execution_trace_observer,
        ) as prof:
            with torch.autograd.profiler.record_function(
                record_function_name(args),
                metadata_for_scope(args),
            ):
                output = llama_common.run_pipeline(pipe, args)
            profiler_synchronize_device(device)
    finally:
        execution_trace_observer.unregister_callback()

    metadata_json = None
    for event in prof.events():
        if event.name == record_function_name(args):
            metadata_json = event.metadata_json
            break

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    decoded = llama_common.decode_outputs(pipe, output)
    offload_common.free_cpu_offload_hooks(pipe)
    output_paths.text_path.write_text("\n---\n".join(decoded), encoding="utf-8")

    model = llama_common.underlying_model(pipe.model)
    pipeline_catalog, pipeline_id_to_path = build_pipeline_module_index_from_prof(
        prof,
        model,
    )
    write_llamasim_runtime_bundle(
        prof,
        output_paths.execution_trace_path,
        output_paths.llamasim_output_dir,
        trace_json_path=output_paths.trace_path,
        module_catalog=build_module_catalog(model),
        module_id_to_path=build_module_id_to_path_from_prof(prof, model),
        pipeline_module_catalog=pipeline_catalog,
        pipeline_module_id_to_path=pipeline_id_to_path,
    )

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    if args.fusion != "none":
        from torch._dynamo.utils import counters as _dynamo_counters

        unique_graphs = _dynamo_counters.get("stats", {}).get("unique_graphs", 0)
        print(f"inductor_unique_graphs: {unique_graphs}")
    print("model:", args.model)
    print("requested_device:", device)
    print("offload_mode:", args.offload_mode)
    print("preload_module_classes:", ",".join(classes))
    print("offload_device:", "cpu")
    print("fusion:", args.fusion)
    print("accelerate_version:", accelerate_version)
    print("dtype:", args.dtype)
    print("seed:", args.seed)
    print("enable_eos_stop:", args.enable_eos_stop)
    print("profile_memory:", args.profile_memory)
    print("record_shapes:", args.record_shapes)
    print("with_stack:", args.with_stack)
    print("with_modules:", args.with_modules)
    print("warmup_runs:", args.warmup_runs)
    print("trace_path:", output_paths.trace_path)
    if hasattr(prof, "export_csv"):
        print("trace_csv_path:", output_paths.csv_path)
    print("text_output_path:", output_paths.text_path)
    print("execution_trace_path:", output_paths.execution_trace_path)
    print("llamasim_output_dir:", output_paths.llamasim_output_dir)
    print_llamasim_runtime_summary(output_paths.llamasim_output_dir)
    print("sequence_shape:", tuple(output.sequences.shape))
    print("generated_text_preview:", decoded[0][:200])
    print("metadata_json:", metadata_json)
    if metadata_json:
        print("metadata_parsed:", json.dumps(json.loads(metadata_json), sort_keys=True))


if __name__ == "__main__":
    main()
