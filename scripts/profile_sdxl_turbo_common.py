#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import inspect
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any

import torch


DTYPE_BY_NAME = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def format_missing_diffusers_runtime_error(
    *,
    component: str,
    exc: ModuleNotFoundError,
) -> str:
    lines = [
        f"Failed to import `diffusers` required by {component}.",
        f"sys.executable = {sys.executable!r}",
        f"torch.__file__ = {Path(torch.__file__).resolve()}",
        f"Original error: {exc}",
    ]
    venv_python = Path("/tmp/ptvenv/bin/python")
    if venv_python.exists():
        rerun_command = shlex.join([str(venv_python), *sys.argv])
        lines.extend(
            [
                "This interpreter does not have `diffusers` installed.",
                "Re-run with the profiler/runtime environment:",
                f"PYTHONNOUSERSITE=1 {rerun_command}",
            ]
        )
    else:
        lines.append(
            "Use a Python environment that has `diffusers` installed."
        )
    return "\n".join(lines)


def is_missing_diffusers_error(exc: ModuleNotFoundError) -> bool:
    return exc.name == "diffusers" or str(exc) == "No module named 'diffusers'"


@dataclass(frozen=True)
class OutputPaths:
    trace_path: Path
    csv_path: Path
    image_path: Path
    fx_dot_path: Path | None
    execution_dot_path: Path | None
    execution_trace_path: Path | None
    memory_dot_path: Path | None
    kineto_dot_path: Path | None
    kineto_map_path: Path | None
    hybrid_dot_path: Path | None
    runtime_io_dot_path: Path | None
    llamasim_output_dir: Path | None


def parse_args(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_fusion: str = "none",
    default_dot_level: str,
    default_profile_memory: bool,
    default_record_shapes: bool,
    default_with_stack: bool,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDXL Turbo profiler driver for the patched PyTorch build."
    )
    parser.add_argument(
        "--model",
        default="/data/llamasim/models/sdxl-turbo",
        help="Path to the SDXL Turbo model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="a lovely cat",
        help="Prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Output height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Output width.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Execution device, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BY_NAME.keys()),
        default=default_dtype,
        help="Torch dtype used to load the pipeline.",
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default=default_fusion,
        help=(
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the UNet "
            "with torch.compile; on CUDA this typically produces fused Triton kernels."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode when --fusion=inductor.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help=(
            "Warmup iterations run before profiling. Defaults to 1 for "
            "--fusion=inductor and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for profiler outputs. If set, writes trace JSON, trace CSV, "
            "PNG, and optional DOT outputs into this directory."
        ),
    )
    parser.add_argument(
        "--trace",
        default=f"/tmp/{default_output_prefix}_trace.json",
        help="Path for the exported Chrome trace JSON.",
    )
    parser.add_argument(
        "--trace-csv",
        default=None,
        help=(
            "Path for the exported profiler CSV. Defaults to the trace path with "
            "a .csv suffix, or a filename under --output-dir."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Path for the generated PNG. Defaults to the trace directory or "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--dot-level",
        choices=(
            "none",
            "fx",
            "execution",
            "memory",
            "kineto",
            "hybrid",
            "runtime-io",
            "llamasim-runtime",
            "both",
            "all",
        ),
        default=default_dot_level,
        help=(
            "Optional DOT graph outputs. 'fx' exports the logical UNet FX graph, "
            "'execution' exports a DOT converted from ExecutionTraceObserver output, "
            "'memory' exports a tensor-flow DOT from the profiler memory graph, "
            "'kineto' exports the post-fusion runtime event graph from Kineto, "
            "'hybrid' overlays matched tensor-flow onto Kineto runtime nodes, "
            "'runtime-io' overlays exact ExecutionTraceObserver I/O onto Kineto "
            "runtime nodes, 'llamasim-runtime' emits a step_0_compute_graph.dot "
            "bundle for /data/llamasim with CPU leaf nodes plus GPU runtime nodes, "
            "'both' writes fx+execution, and 'all' "
            "writes fx+execution+memory+kineto+hybrid+runtime-io+llamasim-runtime."
        ),
    )
    parser.add_argument(
        "--fx-dot",
        default=None,
        help=(
            "Path for the exported UNet FX DOT graph. Requires --dot-level fx, both, or all."
        ),
    )
    parser.add_argument(
        "--execution-dot",
        default=None,
        help=(
            "Path for the execution-trace DOT graph. Requires --dot-level execution, both, or all."
        ),
    )
    parser.add_argument(
        "--execution-trace",
        default=None,
        help=(
            "Path for the raw execution trace JSON used to build --execution-dot. "
            "Requires --dot-level execution, runtime-io, llamasim-runtime, both, or all."
        ),
    )
    parser.add_argument(
        "--memory-dot",
        default=None,
        help=(
            "Path for the memory/data-flow DOT graph. Requires --dot-level memory or all."
        ),
    )
    parser.add_argument(
        "--kineto-dot",
        default=None,
        help=(
            "Path for the Kineto runtime DOT graph. Requires --dot-level kineto or all."
        ),
    )
    parser.add_argument(
        "--kineto-map",
        default=None,
        help=(
            "Path for the Kineto-to-trace.csv mapping file. Requires --dot-level kineto or all."
        ),
    )
    parser.add_argument(
        "--hybrid-dot",
        default=None,
        help=(
            "Path for the hybrid runtime+tensors DOT graph. Requires --dot-level hybrid or all."
        ),
    )
    parser.add_argument(
        "--runtime-io-dot",
        default=None,
        help=(
            "Path for the exact runtime+I/O DOT graph. Requires --dot-level runtime-io or all."
        ),
    )
    parser.add_argument(
        "--llamasim-output-dir",
        default=None,
        help=(
            "Directory for a llamasim runtime bundle containing step_0_compute_graph.dot, "
            "ggml_profile_node_records.csv, runtime node/edge metadata, tensor metadata, "
            "and a manifest. "
            "Requires --dot-level llamasim-runtime or all."
        ),
    )
    parser.add_argument(
        "--component",
        default="sdxl_turbo_pipeline",
        help="Component label stored in profiler metadata.",
    )
    parser.add_argument(
        "--phase",
        default="text_to_image",
        help="Phase label stored in profiler metadata.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output.",
    )
    parser.add_argument(
        "--profile-memory",
        dest="profile_memory",
        action="store_true",
        default=default_profile_memory,
        help="Include profiler memory events in the Chrome trace JSON.",
    )
    parser.add_argument(
        "--no-profile-memory",
        dest="profile_memory",
        action="store_false",
        help="Disable profiler memory events in the Chrome trace JSON.",
    )
    parser.add_argument(
        "--record-shapes",
        dest="record_shapes",
        action="store_true",
        default=default_record_shapes,
        help="Record operator input shapes in the profiler results.",
    )
    parser.add_argument(
        "--no-record-shapes",
        dest="record_shapes",
        action="store_false",
        help="Disable operator input-shape recording in the profiler results.",
    )
    parser.add_argument(
        "--with-stack",
        dest="with_stack",
        action="store_true",
        default=default_with_stack,
        help="Record Python source locations for profiler events.",
    )
    parser.add_argument(
        "--no-with-stack",
        dest="with_stack",
        action="store_false",
        help="Disable Python source-location capture in the profiler results.",
    )
    return parser.parse_args()


def output_stem(default_output_prefix: str, fusion: str) -> str:
    if fusion == "none":
        return default_output_prefix
    return f"{default_output_prefix}_{fusion}"


def dot_level_enabled(dot_level: str, kind: str) -> bool:
    if dot_level == "all":
        return True
    if dot_level == "both":
        return kind in {"fx", "execution"}
    return dot_level == kind


def validate_dot_args(args: argparse.Namespace) -> None:
    fx_enabled = dot_level_enabled(args.dot_level, "fx")
    execution_enabled = dot_level_enabled(args.dot_level, "execution")
    memory_enabled = dot_level_enabled(args.dot_level, "memory")
    kineto_enabled = dot_level_enabled(args.dot_level, "kineto")
    hybrid_enabled = dot_level_enabled(args.dot_level, "hybrid")
    runtime_io_enabled = dot_level_enabled(args.dot_level, "runtime-io")
    llamasim_enabled = dot_level_enabled(args.dot_level, "llamasim-runtime")
    if args.fx_dot is not None and not fx_enabled:
        raise RuntimeError("--fx-dot requires --dot-level fx, both, or all")
    if args.execution_dot is not None and not execution_enabled:
        raise RuntimeError(
            "--execution-dot requires --dot-level execution, both, or all"
        )
    if args.execution_trace is not None and not (
        execution_enabled or runtime_io_enabled or llamasim_enabled
    ):
        raise RuntimeError(
            "--execution-trace requires --dot-level execution, runtime-io, llamasim-runtime, both, or all"
        )
    if args.memory_dot is not None and not memory_enabled:
        raise RuntimeError("--memory-dot requires --dot-level memory or all")
    if args.kineto_dot is not None and not kineto_enabled:
        raise RuntimeError("--kineto-dot requires --dot-level kineto or all")
    if args.kineto_map is not None and not kineto_enabled:
        raise RuntimeError("--kineto-map requires --dot-level kineto or all")
    if args.hybrid_dot is not None and not hybrid_enabled:
        raise RuntimeError("--hybrid-dot requires --dot-level hybrid or all")
    if args.runtime_io_dot is not None and not runtime_io_enabled:
        raise RuntimeError(
            "--runtime-io-dot requires --dot-level runtime-io or all"
        )
    if args.llamasim_output_dir is not None and not llamasim_enabled:
        raise RuntimeError(
            "--llamasim-output-dir requires --dot-level llamasim-runtime or all"
        )
    if llamasim_enabled and args.device.startswith("cpu"):
        raise RuntimeError(
            "llamasim-runtime export is intended for post-fusion GPU kernel traces"
        )
    if memory_enabled:
        missing = []
        if not args.profile_memory:
            missing.append("--profile-memory")
        if not args.record_shapes:
            missing.append("--record-shapes")
        if not args.with_stack:
            missing.append("--with-stack")
        if missing:
            raise RuntimeError(
                "Memory DOT export requires "
                + ", ".join(missing)
                + "."
            )
    if hybrid_enabled:
        missing = []
        if not args.profile_memory:
            missing.append("--profile-memory")
        if not args.record_shapes:
            missing.append("--record-shapes")
        if not args.with_stack:
            missing.append("--with-stack")
        if missing:
            raise RuntimeError(
                "Hybrid DOT export requires "
                + ", ".join(missing)
                + "."
            )


def resolve_output_paths(
    args: argparse.Namespace,
    default_output_prefix: str,
    default_llamasim_output_dirname: str | None = None,
) -> OutputPaths:
    stem = output_stem(default_output_prefix, args.fusion)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        trace_path = output_dir / f"{stem}_trace.json"
        csv_path = output_dir / f"{stem}_trace.csv"
        image_path = output_dir / f"{stem}_output.png"
        fx_dot_path = (
            Path(args.fx_dot)
            if args.fx_dot is not None
            else output_dir / f"{stem}_fx.dot"
        )
        execution_dot_path = (
            Path(args.execution_dot)
            if args.execution_dot is not None
            else output_dir / f"{stem}_execution.dot"
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace is not None
            else output_dir / f"{stem}_execution_trace.json"
        )
        memory_dot_path = (
            Path(args.memory_dot)
            if args.memory_dot is not None
            else output_dir / f"{stem}_memory.dot"
        )
        kineto_dot_path = (
            Path(args.kineto_dot)
            if args.kineto_dot is not None
            else output_dir / f"{stem}_kineto.dot"
        )
        kineto_map_path = (
            Path(args.kineto_map)
            if args.kineto_map is not None
            else output_dir / f"{stem}_kineto_map.csv"
        )
        hybrid_dot_path = (
            Path(args.hybrid_dot)
            if args.hybrid_dot is not None
            else output_dir / f"{stem}_hybrid.dot"
        )
        runtime_io_dot_path = (
            Path(args.runtime_io_dot)
            if args.runtime_io_dot is not None
            else output_dir / f"{stem}_runtime_io.dot"
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir is not None
            else (
                output_dir / default_llamasim_output_dirname
                if default_llamasim_output_dirname is not None
                else output_dir / f"{stem}_llamasim_runtime"
            )
        )
    else:
        trace_path = Path(args.trace)
        csv_path = (
            Path(args.trace_csv)
            if args.trace_csv is not None
            else trace_path.with_suffix(".csv")
        )
        image_path = (
            Path(args.image)
            if args.image is not None
            else trace_path.with_name(f"{stem}_output.png")
        )
        fx_dot_path = (
            Path(args.fx_dot)
            if args.fx_dot is not None
            else trace_path.with_name(f"{stem}_fx.dot")
        )
        execution_dot_path = (
            Path(args.execution_dot)
            if args.execution_dot is not None
            else trace_path.with_name(f"{stem}_execution.dot")
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace is not None
            else trace_path.with_name(f"{stem}_execution_trace.json")
        )
        memory_dot_path = (
            Path(args.memory_dot)
            if args.memory_dot is not None
            else trace_path.with_name(f"{stem}_memory.dot")
        )
        kineto_dot_path = (
            Path(args.kineto_dot)
            if args.kineto_dot is not None
            else trace_path.with_name(f"{stem}_kineto.dot")
        )
        kineto_map_path = (
            Path(args.kineto_map)
            if args.kineto_map is not None
            else trace_path.with_name(f"{stem}_kineto_map.csv")
        )
        hybrid_dot_path = (
            Path(args.hybrid_dot)
            if args.hybrid_dot is not None
            else trace_path.with_name(f"{stem}_hybrid.dot")
        )
        runtime_io_dot_path = (
            Path(args.runtime_io_dot)
            if args.runtime_io_dot is not None
            else trace_path.with_name(f"{stem}_runtime_io.dot")
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir is not None
            else trace_path.with_name(f"{stem}_llamasim_runtime")
        )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not dot_level_enabled(args.dot_level, "fx"):
        fx_dot_path = None
    if not dot_level_enabled(args.dot_level, "execution"):
        execution_dot_path = None
    if not (
        dot_level_enabled(args.dot_level, "execution")
        or dot_level_enabled(args.dot_level, "runtime-io")
        or dot_level_enabled(args.dot_level, "llamasim-runtime")
    ):
        execution_trace_path = None
    if not dot_level_enabled(args.dot_level, "memory"):
        memory_dot_path = None
    if not dot_level_enabled(args.dot_level, "kineto"):
        kineto_dot_path = None
        kineto_map_path = None
    if not dot_level_enabled(args.dot_level, "hybrid"):
        hybrid_dot_path = None
    if not dot_level_enabled(args.dot_level, "runtime-io"):
        runtime_io_dot_path = None
    if not dot_level_enabled(args.dot_level, "llamasim-runtime"):
        llamasim_output_dir = None
    if fx_dot_path is not None:
        fx_dot_path.parent.mkdir(parents=True, exist_ok=True)
    if execution_dot_path is not None:
        execution_dot_path.parent.mkdir(parents=True, exist_ok=True)
    if execution_trace_path is not None:
        execution_trace_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_dot_path is not None:
        memory_dot_path.parent.mkdir(parents=True, exist_ok=True)
    if kineto_dot_path is not None:
        kineto_dot_path.parent.mkdir(parents=True, exist_ok=True)
    if kineto_map_path is not None:
        kineto_map_path.parent.mkdir(parents=True, exist_ok=True)
    if hybrid_dot_path is not None:
        hybrid_dot_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_io_dot_path is not None:
        runtime_io_dot_path.parent.mkdir(parents=True, exist_ok=True)
    if llamasim_output_dir is not None:
        llamasim_output_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        trace_path=trace_path,
        csv_path=csv_path,
        image_path=image_path,
        fx_dot_path=fx_dot_path,
        execution_dot_path=execution_dot_path,
        execution_trace_path=execution_trace_path,
        memory_dot_path=memory_dot_path,
        kineto_dot_path=kineto_dot_path,
        kineto_map_path=kineto_map_path,
        hybrid_dot_path=hybrid_dot_path,
        runtime_io_dot_path=runtime_io_dot_path,
        llamasim_output_dir=llamasim_output_dir,
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profiler_synchronize_device(device: torch.device) -> None:
    with torch.autograd.profiler.record_function("profiler_sync:device"):
        synchronize_device(device)


def profile_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def load_pipeline(model: str, torch_dtype: torch.dtype, device: torch.device) -> Any:
    try:
        from diffusers import StableDiffusionXLPipeline
    except ModuleNotFoundError as exc:
        if is_missing_diffusers_error(exc):
            raise RuntimeError(
                format_missing_diffusers_runtime_error(
                    component="the SDXL Turbo scripts",
                    exc=exc,
                )
            ) from exc
        raise

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)
    return pipe


def configure_llamasim_inductor_markers(output_dir: Path | str | None) -> None:
    """Enable compile-time provenance needed for exact LLaMASim runtime edges."""
    if output_dir is None:
        return

    output_dir_str = str(output_dir)
    desired = {
        "weight_streaming_emit_ids": True,
        "weight_streaming_emit_launch_markers": True,
        "weight_streaming_emit_sync_markers": True,
        "weight_streaming_output_code": output_dir_str,
    }
    config = torch._inductor.config
    changed = any(getattr(config, name) != value for name, value in desired.items())
    for name, value in desired.items():
        setattr(config, name, value)

    if changed:
        try:
            import torch._dynamo as dynamo
        except ImportError:
            dynamo = None
        if dynamo is not None:
            dynamo.reset()


def maybe_compile(
    pipe: Any,
    args: argparse.Namespace,
    output_paths: OutputPaths | None = None,
) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")

    if output_paths is not None and output_paths.llamasim_output_dir is not None:
        configure_llamasim_inductor_markers(output_paths.llamasim_output_dir)

    pipe.unet = torch.compile(
        pipe.unet,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )


def snapshot_example_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(snapshot_example_value(v) for v in value)
    if isinstance(value, list):
        return [snapshot_example_value(v) for v in value]
    if isinstance(value, dict):
        return {k: snapshot_example_value(v) for k, v in value.items()}
    return value


def capture_example_inputs(module: torch.nn.Module) -> tuple[dict[str, Any], Any]:
    captured: dict[str, Any] = {}
    hook_handle = None

    def hook(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal hook_handle
        if captured:
            return
        captured["args"] = snapshot_example_value(args)
        captured["kwargs"] = snapshot_example_value(kwargs)
        if hook_handle is not None:
            hook_handle.remove()

    hook_handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, hook_handle


def underlying_unet_module(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_orig_mod", module)


def module_accepts_kwarg(module: torch.nn.Module, kwarg_name: str) -> bool:
    try:
        parameters = inspect.signature(module.forward).parameters.values()
    except (TypeError, ValueError):
        return True
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == kwarg_name:
            return True
    return False


def export_unet_graph_module(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> torch.fx.GraphModule:
    export_kwargs = dict(kwargs)
    if module_accepts_kwarg(module, "return_dict"):
        export_kwargs.setdefault("return_dict", False)

    export_errors: list[str] = []
    try:
        exported = torch.export.export(module, args, export_kwargs)
        return exported.graph_module
    except Exception as exc:
        export_errors.append(f"torch.export.export failed: {exc}")

    try:
        exported = torch._dynamo.export(module)(*args, **export_kwargs)
        return exported.graph_module
    except Exception as exc:
        export_errors.append(f"torch._dynamo.export failed: {exc}")

    raise RuntimeError("Failed to export FX graph for UNet.\n" + "\n".join(export_errors))


def dot_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def fx_target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    return getattr(target, "__name__", str(target))


def fx_shape_summary(node: torch.fx.Node) -> str | None:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is not None and hasattr(tensor_meta, "shape"):
        return str(tuple(tensor_meta.shape))
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    return None


def fx_node_fillcolor(node: torch.fx.Node) -> str:
    if node.op == "placeholder":
        return "lightyellow"
    if node.op == "output":
        return "lightblue"
    if node.op == "get_attr":
        return "mistyrose"
    return "white"


def write_fx_graph_dot(
    graph_module: torch.fx.GraphModule, path: Path, graph_name: str
) -> None:
    with path.open("w") as f:
        f.write(f'digraph "{dot_escape(graph_name)}" {{\n')
        f.write("  rankdir=TB;\n")
        for node in graph_module.graph.nodes:
            label_lines = [node.name, f"op={node.op}"]
            if node.op != "output":
                label_lines.append(f"target={fx_target_name(node.target)}")
            shape_summary = fx_shape_summary(node)
            if shape_summary is not None:
                label_lines.append(f"shape={shape_summary}")
            label = "\\n".join(dot_escape(line) for line in label_lines)
            f.write(
                f'  "{dot_escape(node.name)}" '
                f'[shape=box, style=filled, fillcolor="{fx_node_fillcolor(node)}", '
                f'label="{label}"];\n'
            )

        for node in graph_module.graph.nodes:
            for user in node.users:
                f.write(
                    f'  "{dot_escape(node.name)}" -> "{dot_escape(user.name)}";\n'
                )
        f.write("}\n")


def open_json_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def execution_trace_attr(node: dict[str, Any], name: str) -> Any:
    for attr in node.get("attrs", []):
        if attr.get("name") == name:
            return attr.get("value")
    return None


def execution_node_fillcolor(name: str) -> str:
    if name.startswith("triton_"):
        return "moccasin"
    if "Call CompiledFxGraph" in name:
        return "lightblue"
    if name.startswith("aten::"):
        return "white"
    return "lightyellow"


def is_execution_meta_node(name: str) -> bool:
    return name in {
        "[pytorch|profiler|execution_trace|process]",
        "[pytorch|profiler|execution_trace|thread]",
    }


def direct_tensor_value_key(type_name: Any, value: Any) -> str | None:
    if not isinstance(type_name, str) or not type_name.startswith("Tensor"):
        return None
    if not isinstance(value, list):
        return None
    return json.dumps(value, separators=(",", ":"))


def tensor_shape_summary(shape: Any) -> str:
    if isinstance(shape, list) and all(not isinstance(dim, list) for dim in shape):
        return str(tuple(shape))
    return str(shape)


def tensor_value_summary(value: Any, max_chars: int = 60) -> str:
    text = json.dumps(value, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_tensor_label(
    tensor_name: str,
    kind: str,
    type_name: str,
    shape: Any,
    value: Any,
) -> str:
    label_lines = [tensor_name, kind, type_name, f"shape={tensor_shape_summary(shape)}"]
    label_lines.append(f"value={tensor_value_summary(value)}")
    return "\\n".join(dot_escape(line) for line in label_lines)


def write_execution_trace_dot(
    execution_trace_path: Path, dot_path: Path, graph_name: str
) -> None:
    with open_json_maybe_gz(execution_trace_path) as f:
        payload = json.load(f)

    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise RuntimeError(
            f"Execution trace at {execution_trace_path} does not contain a 'nodes' list"
        )

    nodes = {
        int(node["id"]): node
        for node in raw_nodes
        if "id" in node
        and "name" in node
        and not is_execution_meta_node(str(node["name"]))
    }

    control_edges: set[tuple[int, int]] = set()
    tensor_nodes: list[dict[str, Any]] = []
    op_to_tensor_edges: list[tuple[int, str, int]] = []
    tensor_to_op_edges: list[tuple[str, int, int]] = []
    latest_tensor_node_by_key: dict[str, str] = {}
    leaf_tensor_node_by_key: dict[str, str] = {}

    def make_tensor_node(
        *,
        key: str,
        type_name: str,
        shape: Any,
        value: Any,
        kind: str,
    ) -> str:
        tensor_dot_id = f"t{len(tensor_nodes)}"
        tensor_nodes.append(
            {
                "dot_id": tensor_dot_id,
                "label": build_tensor_label(
                    tensor_name=f"tensor_{len(tensor_nodes)}",
                    kind=kind,
                    type_name=type_name,
                    shape=shape,
                    value=value,
                ),
                "key": key,
            }
        )
        return tensor_dot_id

    for node_id, node in sorted(nodes.items()):
        ctrl_dep = node.get("ctrl_deps")
        if isinstance(ctrl_dep, int) and ctrl_dep in nodes:
            control_edges.add((ctrl_dep, node_id))

        inputs = node.get("inputs", {})
        values = inputs.get("values", [])
        types = inputs.get("types", [])
        shapes = inputs.get("shapes", [])
        for input_index, (type_name, value, shape) in enumerate(
            zip(types, values, shapes)
        ):
            key = direct_tensor_value_key(type_name, value)
            if key is None:
                continue

            tensor_dot_id = latest_tensor_node_by_key.get(key)
            if tensor_dot_id is None:
                tensor_dot_id = leaf_tensor_node_by_key.get(key)
            if tensor_dot_id is None:
                tensor_dot_id = make_tensor_node(
                    key=key,
                    type_name=type_name,
                    shape=shape,
                    value=value,
                    kind="input",
                )
                leaf_tensor_node_by_key[key] = tensor_dot_id
                latest_tensor_node_by_key[key] = tensor_dot_id
            tensor_to_op_edges.append((tensor_dot_id, node_id, input_index))

        outputs = node.get("outputs", {})
        values = outputs.get("values", [])
        types = outputs.get("types", [])
        shapes = outputs.get("shapes", [])
        for output_index, (type_name, value, shape) in enumerate(
            zip(types, values, shapes)
        ):
            key = direct_tensor_value_key(type_name, value)
            if key is None:
                continue
            tensor_dot_id = make_tensor_node(
                key=key,
                type_name=type_name,
                shape=shape,
                value=value,
                kind=f"output {output_index}",
            )
            latest_tensor_node_by_key[key] = tensor_dot_id
            op_to_tensor_edges.append((node_id, tensor_dot_id, output_index))

    with dot_path.open("w") as f:
        f.write(f'digraph "{dot_escape(graph_name)}" {{\n')
        f.write("  newrank=true;\n")
        f.write("  rankdir=TB;\n")
        for node_id, node in nodes.items():
            name = str(node["name"])
            label_lines = [name, f"id={node_id}"]
            rf_id = execution_trace_attr(node, "rf_id")
            if rf_id not in (None, 0):
                label_lines.append(f"rf_id={rf_id}")
            kernel_file = execution_trace_attr(node, "kernel_file")
            if isinstance(kernel_file, str) and kernel_file:
                label_lines.append(f"kernel={Path(kernel_file).name}")
            outputs = node.get("outputs", {})
            shapes = outputs.get("shapes", [])
            if shapes:
                label_lines.append(f"outputs={len(shapes)}")
            label = "\\n".join(dot_escape(line) for line in label_lines)
            f.write(
                f'  "n{node_id}" [shape=box, style=filled, '
                f'fillcolor="{execution_node_fillcolor(name)}", label="{label}"];\n'
            )

        for tensor_node in tensor_nodes:
            f.write(
                f'  "{tensor_node["dot_id"]}" [shape=record, style=filled, '
                f'fillcolor="pink", label="{tensor_node["label"]}"];\n'
            )

        for src, dst in sorted(control_edges):
            f.write(
                f'  "n{src}" -> "n{dst}" [style=dashed, color="gray50", label="ctrl"];\n'
            )
        for src, tensor_dot_id, output_index in op_to_tensor_edges:
            f.write(
                f'  "n{src}" -> "{tensor_dot_id}" [color="black", label="out {output_index}"];\n'
            )
        for tensor_dot_id, dst, input_index in tensor_to_op_edges:
            f.write(
                f'  "{tensor_dot_id}" -> "n{dst}" [color="black", label="in {input_index}"];\n'
            )
        f.write("}\n")


def load_execution_trace_nodes(execution_trace_path: Path) -> dict[int, dict[str, Any]]:
    with open_json_maybe_gz(execution_trace_path) as f:
        payload = json.load(f)

    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise RuntimeError(
            f"Execution trace at {execution_trace_path} does not contain a 'nodes' list"
        )

    return {
        int(node["id"]): node
        for node in raw_nodes
        if "id" in node
        and "name" in node
        and not is_execution_meta_node(str(node["name"]))
    }


def execution_rf_id(node: dict[str, Any]) -> int | None:
    rf_id = execution_trace_attr(node, "rf_id")
    if isinstance(rf_id, int) and rf_id > 0:
        return rf_id
    return None


def execution_tensor_tuple(value: Any) -> tuple[int, int, int, int, int, str] | None:
    if not isinstance(value, list) or len(value) != 6:
        return None
    tensor_id, storage_id, offset, numel, itemsize, device = value
    if not all(
        isinstance(v, int)
        for v in (tensor_id, storage_id, offset, numel, itemsize)
    ):
        return None
    if not isinstance(device, str):
        return None
    return (
        tensor_id,
        storage_id,
        offset,
        numel,
        itemsize,
        device,
    )


def execution_tensor_key(value: Any) -> str | None:
    tensor_tuple = execution_tensor_tuple(value)
    if tensor_tuple is None:
        return None
    return json.dumps(tensor_tuple, separators=(",", ":"))


def execution_tensor_label(
    tensor_tuple: tuple[int, int, int, int, int, str],
    *,
    type_name: str | None,
    shape: Any,
) -> str:
    tensor_id, storage_id, offset, numel, itemsize, device = tensor_tuple
    label_lines = [
        f"tensor_id={tensor_id}",
        f"storage_id={storage_id}",
        f"offset={offset}",
        f"numel={numel}",
        f"itemsize={itemsize}",
        f"device={device}",
    ]
    if isinstance(type_name, str) and type_name:
        label_lines.append(f"type={type_name}")
    if shape not in (None, [], ""):
        label_lines.append(f"shape={tensor_shape_summary(shape)}")
    return "\\n".join(dot_escape(line) for line in label_lines)


def execution_value_contains_tensor(value: Any) -> bool:
    if execution_tensor_tuple(value) is not None:
        return True
    if isinstance(value, list):
        return any(execution_value_contains_tensor(v) for v in value)
    return False


def flatten_execution_tensor_entries(
    value: Any,
    *,
    path: str,
    shape: Any,
    type_name: Any,
) -> list[dict[str, Any]]:
    tensor_tuple = execution_tensor_tuple(value)
    if tensor_tuple is not None:
        display_type = type_name if isinstance(type_name, str) else "Tensor"
        return [
            {
                "path": path,
                "tensor_tuple": tensor_tuple,
                "shape": shape,
                "type_name": display_type,
            }
        ]

    if not isinstance(value, list):
        return []

    entries: list[dict[str, Any]] = []
    for child_index, child_value in enumerate(value):
        child_shape = (
            shape[child_index]
            if isinstance(shape, list) and child_index < len(shape)
            else None
        )
        child_path = f"{path}.{child_index}"
        entries.extend(
            flatten_execution_tensor_entries(
                child_value,
                path=child_path,
                shape=child_shape,
                type_name=None,
            )
        )
    return entries


def summarize_execution_scalar_args(
    types: Any,
    values: Any,
    *,
    prefix: str,
) -> list[str]:
    if not isinstance(types, list) or not isinstance(values, list):
        return []

    scalar_labels: list[str] = []
    for arg_index, (type_name, value) in enumerate(zip(types, values)):
        if execution_value_contains_tensor(value):
            continue
        label = f"{prefix}{arg_index}="
        if isinstance(type_name, str):
            label += f"{type_name}:"
        label += tensor_value_summary(value)
        scalar_labels.append(label)
    return scalar_labels


def index_execution_trace_nodes_by_rf_id(
    execution_nodes: dict[int, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    execution_nodes_by_rf_id: dict[int, list[dict[str, Any]]] = {}
    for node in execution_nodes.values():
        rf_id = execution_rf_id(node)
        if rf_id is None:
            continue
        execution_nodes_by_rf_id.setdefault(rf_id, []).append(node)
    return execution_nodes_by_rf_id


def memory_tensor_label(
    key: Any,
    version: int,
    category_name: str | None,
    num_bytes: int | None,
    kind: str | None = None,
) -> str:
    label_lines = [f"T{key.id} v{version}", f"device={key.device}"]
    label_lines.append(f"alloc={key.storage.allocation_id}")
    label_lines.append(f"ptr={hex(key.storage.ptr)}")
    if num_bytes is not None:
        label_lines.append(f"bytes={num_bytes}")
    if category_name is not None:
        label_lines.append(f"category={category_name}")
    if kind is not None:
        label_lines.append(f"kind={kind}")
    return "\\n".join(dot_escape(line) for line in label_lines)


def write_memory_profile_dot(prof: Any, dot_path: Path, graph_name: str) -> None:
    memory_profile = prof._memory_profile()
    graph = memory_profile._data_flow_graph
    flow_nodes = tuple(
        node
        for node in graph.flow_nodes
        if node._event.name not in {"[memory]", "[OutOfMemory]"}
    )
    op_dot_ids = {id(node): f"n{index}" for index, node in enumerate(flow_nodes)}
    tensor_dot_ids: dict[tuple[Any, int], str] = {}
    tensor_labels: dict[tuple[Any, int], str] = {}
    input_edges: list[tuple[str, str, str]] = []
    output_edges: list[tuple[str, str, str, str]] = []
    intermediate_edges: list[tuple[str, str]] = []

    def tensor_sort_key(item: tuple[Any, Any]) -> tuple[int, int]:
        key, value = item
        if isinstance(value, tuple):
            return (key.id, value[1])
        return (key.id, value)

    def ensure_tensor_node(
        key: Any,
        version: int,
        *,
        kind: str | None = None,
    ) -> str:
        pair = (key, version)
        tensor_dot_id = tensor_dot_ids.get(pair)
        if tensor_dot_id is not None:
            return tensor_dot_id

        tensor_dot_id = f"t{len(tensor_dot_ids)}"
        tensor_dot_ids[pair] = tensor_dot_id
        category = memory_profile._categories.get(key, version)
        num_bytes = None
        try:
            num_bytes = memory_profile._size_map[key]
        except KeyError:
            num_bytes = None
        tensor_labels[pair] = memory_tensor_label(
            key,
            version,
            None if category is None else category.name,
            num_bytes,
            kind=kind,
        )
        return tensor_dot_id

    with dot_path.open("w") as f:
        f.write(f'digraph "{dot_escape(graph_name)}" {{\n')
        f.write("  newrank=true;\n")
        f.write("  rankdir=TB;\n")

        for node in flow_nodes:
            name = node._event.name.replace("torch::autograd::", "")
            label_lines = [name, f"start_ns={node.start_time}"]
            label_lines.append(f"inputs={len(node.inputs)}")
            label_lines.append(f"outputs={len(node.outputs)}")
            if node.intermediates:
                label_lines.append(f"temps={len(node.intermediates)}")
            label = "\\n".join(dot_escape(line) for line in label_lines)
            f.write(
                f'  "{op_dot_ids[id(node)]}" [shape=box, style=filled, '
                f'fillcolor="{execution_node_fillcolor(name)}", label="{label}"];\n'
            )

            for key, (mutated, version) in sorted(
                node.inputs.items(), key=tensor_sort_key
            ):
                tensor_dot_id = ensure_tensor_node(
                    key,
                    version,
                    kind="input" if not mutated else "input/output",
                )
                input_label = "inout" if mutated else "in"
                input_edges.append((tensor_dot_id, op_dot_ids[id(node)], input_label))

            for key, version in sorted(node.outputs.items(), key=tensor_sort_key):
                tensor_dot_id = ensure_tensor_node(
                    key,
                    version,
                    kind="output",
                )
                output_edges.append(
                    (op_dot_ids[id(node)], tensor_dot_id, "out", "black")
                )

            for key in sorted(node.intermediates):
                tensor_dot_id = ensure_tensor_node(key, 0, kind="temporary")
                intermediate_edges.append((op_dot_ids[id(node)], tensor_dot_id))

        for pair, tensor_dot_id in tensor_dot_ids.items():
            label = tensor_labels[pair]
            f.write(
                f'  "{tensor_dot_id}" [shape=record, style=filled, '
                f'fillcolor="pink", label="{label}"];\n'
            )

        for src, dst, label in input_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="black", label="{label}"];\n'
            )
        for src, dst, label, color in output_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="{color}", label="{label}"];\n'
            )
        for src, dst in intermediate_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dashed, color="gray50", label="tmp"];\n'
            )
        f.write("}\n")


def kineto_node_fillcolor(device_type: Any, name: str) -> str:
    device_name = getattr(device_type, "name", str(device_type))
    if device_name != "CPU":
        if name.startswith("triton_"):
            return "moccasin"
        return "peachpuff"
    return execution_node_fillcolor(name)


def get_raw_kineto_events(prof: Any) -> tuple[int, list[Any]]:
    from torch.autograd.profiler_util import _filter_name

    if getattr(prof, "profiler", None) is None or prof.profiler.kineto_results is None:
        raise RuntimeError("Kineto DOT export requires profiler.kineto_results")

    result = prof.profiler.kineto_results
    trace_start_ns = result.trace_start_ns()
    raw_events = []
    for event in result.events():
        if _filter_name(event.name()):
            continue
        if getattr(event, "is_hidden_event", lambda: False)():
            continue
        raw_events.append(event)

    raw_events.sort(
        key=lambda event: (
            event.start_ns(),
            event.end_ns(),
            event.correlation_id(),
            str(event.device_type()),
            event.start_thread_id(),
            event.device_index(),
            event.device_resource_id(),
            event.name(),
        )
    )
    return trace_start_ns, raw_events


def build_kineto_runtime_graph(raw_events: list[Any]) -> tuple[
    dict[int, str],
    list[tuple[str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str]],
]:
    from torch.autograd import DeviceType

    node_ids = {id(event): f"k{index}" for index, event in enumerate(raw_events)}

    cpu_parent_edges: list[tuple[str, str]] = []
    cpu_sync_events_by_thread: dict[int, list[Any]] = {}
    for event in raw_events:
        device_type = event.device_type()
        is_async = event.is_async() or (
            event.start_thread_id() != event.end_thread_id()
        )
        if device_type == DeviceType.CPU and not is_async:
            cpu_sync_events_by_thread.setdefault(event.start_thread_id(), []).append(
                event
            )

    for thread_events in cpu_sync_events_by_thread.values():
        thread_events.sort(key=lambda event: (event.start_ns(), -event.end_ns()))
        current_events: list[Any] = []
        for event in thread_events:
            while current_events:
                parent = current_events[-1]
                if (
                    event.start_ns() >= parent.end_ns()
                    or event.end_ns() > parent.end_ns()
                ):
                    current_events.pop()
                else:
                    cpu_parent_edges.append((node_ids[id(parent)], node_ids[id(event)]))
                    break
            current_events.append(event)

    corr_to_cpu_node: dict[int, str] = {}
    for event in raw_events:
        if event.device_type() == DeviceType.CPU:
            corr_to_cpu_node[event.correlation_id()] = node_ids[id(event)]

    linked_edges: list[tuple[str, str, str]] = []
    for event in raw_events:
        linked_corr_id = event.linked_correlation_id()
        if linked_corr_id <= 0:
            continue
        src = corr_to_cpu_node.get(linked_corr_id)
        if src is None:
            continue
        label = "launch" if event.device_type() != DeviceType.CPU else "linked"
        linked_edges.append((src, node_ids[id(event)], label))

    stream_edges: list[tuple[str, str]] = []
    device_events_by_resource: dict[tuple[Any, int, int], list[Any]] = {}
    for event in raw_events:
        if event.device_type() == DeviceType.CPU:
            continue
        key = (
            event.device_type(),
            event.device_index(),
            event.device_resource_id(),
        )
        device_events_by_resource.setdefault(key, []).append(event)

    for device_events in device_events_by_resource.values():
        device_events.sort(key=lambda event: (event.start_ns(), event.end_ns()))
        for current, nxt in zip(device_events, device_events[1:]):
            stream_edges.append((node_ids[id(current)], node_ids[id(nxt)]))

    return node_ids, cpu_parent_edges, linked_edges, stream_edges


def scalarize_trace_us(value: float) -> int:
    return int(round(float(value) * 1000.0))


def csv_event_depth(event: Any) -> int:
    depth = 0
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        depth += 1
        parent = getattr(parent, "cpu_parent", None)
    return depth


def function_event_match_key(event: Any) -> tuple[Any, ...]:
    device_type = getattr(event, "device_type", None)
    return (
        getattr(event, "id", None),
        getattr(event, "trace_name", None),
        getattr(device_type, "name", str(device_type)),
        getattr(event, "device_index", None),
        getattr(event, "device_resource_id", None),
        getattr(event, "thread", None),
        scalarize_trace_us(getattr(event.time_range, "start", 0.0)),
        scalarize_trace_us(event.time_range.elapsed_us()),
        getattr(event, "is_user_annotation", None),
    )


def kineto_event_match_key(event: Any, trace_start_ns: int) -> tuple[Any, ...]:
    from torch.autograd.profiler_util import _rewrite_name

    device_type = event.device_type()
    start_us = (event.start_ns() - trace_start_ns) / 1000.0
    duration_us = event.duration_ns() / 1000.0
    return (
        event.correlation_id(),
        _rewrite_name(name=event.name(), with_wildcard=False),
        getattr(device_type, "name", str(device_type)),
        event.device_index(),
        event.device_resource_id(),
        event.start_thread_id(),
        scalarize_trace_us(start_us),
        scalarize_trace_us(duration_us),
        event.is_user_annotation(),
    )


def build_kineto_csv_matches(
    prof: Any, raw_events: list[Any], trace_start_ns: int
) -> dict[int, list[tuple[int, Any]]]:
    csv_events = list(prof.events())
    csv_events_by_key: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
    for event_index, event in enumerate(csv_events):
        if getattr(event, "trace_name", None) is None:
            continue
        csv_events_by_key.setdefault(function_event_match_key(event), []).append(
            (event_index, event)
        )
    return {
        id(event): csv_events_by_key.get(kineto_event_match_key(event, trace_start_ns), [])
        for event in raw_events
    }


def kineto_runtime_node_label_lines(
    event: Any,
    *,
    trace_start_ns: int,
    csv_matches: dict[int, list[tuple[int, Any]]],
) -> list[str]:
    from torch.autograd import DeviceType
    from torch.autograd.profiler_util import _rewrite_name

    name = event.name()
    device_type = event.device_type()
    device_name = getattr(device_type, "name", str(device_type))
    rel_start_us = (event.start_ns() - trace_start_ns) / 1000.0
    duration_us = event.duration_ns() / 1000.0
    label_lines = [name, f"cid={event.correlation_id()}"]
    label_lines.append(f"trace={_rewrite_name(name=name, with_wildcard=False)}")
    if event.linked_correlation_id() > 0:
        label_lines.append(f"linked={event.linked_correlation_id()}")
    label_lines.append(f"device={device_name}:{event.device_index()}")
    if device_type == DeviceType.CPU:
        label_lines.append(f"thread={event.start_thread_id()}")
    else:
        label_lines.append(f"stream={event.device_resource_id()}")
    label_lines.append(f"start_us={rel_start_us:.3f}")
    label_lines.append(f"dur_us={duration_us:.3f}")
    matches = csv_matches.get(id(event), [])
    if len(matches) == 1:
        csv_event_index, csv_event = matches[0]
        label_lines.append(f"csv_index={csv_event_index}")
        label_lines.append(f"csv_id={getattr(csv_event, 'id', None)}")
    elif len(matches) > 1:
        label_lines.append(f"csv_matches={len(matches)}")
    if event.is_user_annotation():
        label_lines.append("user_annotation=1")
    return label_lines


def match_execution_nodes_to_kineto_runtime(
    raw_events: list[Any],
    execution_nodes_by_rf_id: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    from torch.autograd import DeviceType

    matched_execution_node_by_event: dict[int, dict[str, Any]] = {}
    ambiguous_rf_ids: set[int] = set()

    raw_events_by_rf_id: dict[int, list[Any]] = {}
    for event in raw_events:
        if event.device_type() != DeviceType.CPU:
            continue
        rf_id = event.correlation_id()
        if rf_id <= 0:
            continue
        raw_events_by_rf_id.setdefault(rf_id, []).append(event)

    for rf_id, execution_nodes in execution_nodes_by_rf_id.items():
        runtime_candidates = raw_events_by_rf_id.get(rf_id, [])
        if len(execution_nodes) != 1 or not runtime_candidates:
            ambiguous_rf_ids.add(rf_id)
            continue

        execution_node = execution_nodes[0]
        name = str(execution_node["name"])
        same_name = [event for event in runtime_candidates if event.name() == name]
        if same_name:
            runtime_candidates = same_name

        if len(runtime_candidates) != 1:
            ambiguous_rf_ids.add(rf_id)
            continue

        matched_execution_node_by_event[id(runtime_candidates[0])] = execution_node

    return matched_execution_node_by_event, ambiguous_rf_ids


def write_kineto_event_mapping(
    raw_events: list[Any],
    node_ids: dict[int, str],
    csv_matches: dict[int, list[tuple[int, Any]]],
    trace_start_ns: int,
    path: Path,
) -> None:
    fieldnames = [
        "kineto_node_id",
        "kineto_correlation_id",
        "kineto_linked_correlation_id",
        "kineto_name",
        "kineto_device_type",
        "kineto_device_index",
        "kineto_device_resource_id",
        "kineto_start_us",
        "kineto_duration_us",
        "csv_match_count",
        "csv_event_indexes",
        "csv_ids",
        "csv_trace_names",
        "csv_names",
        "csv_device_types",
        "csv_start_us",
        "csv_duration_us",
        "csv_depths",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in raw_events:
            matches = csv_matches.get(id(event), [])
            writer.writerow(
                {
                    "kineto_node_id": node_ids[id(event)],
                    "kineto_correlation_id": event.correlation_id(),
                    "kineto_linked_correlation_id": event.linked_correlation_id(),
                    "kineto_name": event.name(),
                    "kineto_device_type": getattr(
                        event.device_type(), "name", str(event.device_type())
                    ),
                    "kineto_device_index": event.device_index(),
                    "kineto_device_resource_id": event.device_resource_id(),
                    "kineto_start_us": (event.start_ns() - trace_start_ns) / 1000.0,
                    "kineto_duration_us": event.duration_ns() / 1000.0,
                    "csv_match_count": len(matches),
                    "csv_event_indexes": json.dumps(
                        [event_index for event_index, _ in matches],
                        separators=(",", ":"),
                    ),
                    "csv_ids": json.dumps(
                        [getattr(csv_event, "id", None) for _, csv_event in matches],
                        separators=(",", ":"),
                    ),
                    "csv_trace_names": json.dumps(
                        [
                            getattr(csv_event, "trace_name", None)
                            for _, csv_event in matches
                        ],
                        separators=(",", ":"),
                    ),
                    "csv_names": json.dumps(
                        [getattr(csv_event, "name", None) for _, csv_event in matches],
                        separators=(",", ":"),
                    ),
                    "csv_device_types": json.dumps(
                        [
                            getattr(
                                getattr(csv_event, "device_type", None),
                                "name",
                                str(getattr(csv_event, "device_type", None)),
                            )
                            for _, csv_event in matches
                        ],
                        separators=(",", ":"),
                    ),
                    "csv_start_us": json.dumps(
                        [
                            getattr(csv_event.time_range, "start", None)
                            for _, csv_event in matches
                        ],
                        separators=(",", ":"),
                    ),
                    "csv_duration_us": json.dumps(
                        [
                            csv_event.time_range.elapsed_us()
                            for _, csv_event in matches
                        ],
                        separators=(",", ":"),
                    ),
                    "csv_depths": json.dumps(
                        [csv_event_depth(csv_event) for _, csv_event in matches],
                        separators=(",", ":"),
                    ),
                }
            )


def write_kineto_profile_dot(
    prof: Any,
    dot_path: Path,
    graph_name: str,
    mapping_path: Path | None = None,
) -> None:
    from torch.autograd import DeviceType
    from torch.autograd.profiler_util import _rewrite_name

    trace_start_ns, raw_events = get_raw_kineto_events(prof)
    node_ids, cpu_parent_edges, linked_edges, stream_edges = build_kineto_runtime_graph(
        raw_events
    )
    csv_matches = build_kineto_csv_matches(prof, raw_events, trace_start_ns)

    with dot_path.open("w") as f:
        f.write(f'digraph "{dot_escape(graph_name)}" {{\n')
        f.write("  newrank=true;\n")
        f.write("  rankdir=TB;\n")

        for event in raw_events:
            name = event.name()
            device_type = event.device_type()
            device_name = getattr(device_type, "name", str(device_type))
            rel_start_us = (event.start_ns() - trace_start_ns) / 1000.0
            duration_us = event.duration_ns() / 1000.0
            label_lines = [name, f"cid={event.correlation_id()}"]
            label_lines.append(
                f"trace={_rewrite_name(name=name, with_wildcard=False)}"
            )
            if event.linked_correlation_id() > 0:
                label_lines.append(f"linked={event.linked_correlation_id()}")
            label_lines.append(f"device={device_name}:{event.device_index()}")
            if device_type == DeviceType.CPU:
                label_lines.append(f"thread={event.start_thread_id()}")
            else:
                label_lines.append(f"stream={event.device_resource_id()}")
            label_lines.append(f"start_us={rel_start_us:.3f}")
            label_lines.append(f"dur_us={duration_us:.3f}")
            matches = csv_matches.get(id(event), [])
            if len(matches) == 1:
                csv_event_index, csv_event = matches[0]
                label_lines.append(f"csv_index={csv_event_index}")
                label_lines.append(f"csv_id={getattr(csv_event, 'id', None)}")
            elif len(matches) > 1:
                label_lines.append(f"csv_matches={len(matches)}")
            if event.is_user_annotation():
                label_lines.append("user_annotation=1")
            label = "\\n".join(dot_escape(line) for line in label_lines)
            f.write(
                f'  "{node_ids[id(event)]}" [shape=box, style=filled, '
                f'fillcolor="{kineto_node_fillcolor(device_type, name)}", '
                f'label="{label}"];\n'
            )

        for src, dst in cpu_parent_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dashed, color="gray50", label="cpu"];\n'
            )
        for src, dst, label in linked_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="orangered3", penwidth=1.4, label="{label}"];\n'
            )
        for src, dst in stream_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dotted, color="steelblue4", label="stream"];\n'
            )
        f.write("}\n")

    if mapping_path is not None:
        write_kineto_event_mapping(
            raw_events,
            node_ids,
            csv_matches,
            trace_start_ns,
            mapping_path,
        )


def write_runtime_io_profile_dot(
    prof: Any,
    execution_trace_path: Path,
    dot_path: Path,
    graph_name: str,
) -> None:
    from torch.autograd import DeviceType

    trace_start_ns, raw_events = get_raw_kineto_events(prof)
    node_ids, cpu_parent_edges, linked_edges, stream_edges = build_kineto_runtime_graph(
        raw_events
    )
    csv_matches = build_kineto_csv_matches(prof, raw_events, trace_start_ns)
    execution_nodes = load_execution_trace_nodes(execution_trace_path)
    execution_nodes_by_rf_id = index_execution_trace_nodes_by_rf_id(execution_nodes)
    matched_execution_node_by_event, ambiguous_rf_ids = match_execution_nodes_to_kineto_runtime(
        raw_events,
        execution_nodes_by_rf_id,
    )

    tensor_dot_ids: dict[str, str] = {}
    tensor_labels: dict[str, str] = {}
    input_edges: list[tuple[str, str, str]] = []
    output_edges: list[tuple[str, str, str]] = []
    scalar_labels_by_event: dict[int, list[str]] = {}
    io_match_by_event: dict[int, str] = {}

    def ensure_tensor_node(
        tensor_tuple: tuple[int, int, int, int, int, str],
        *,
        type_name: str | None,
        shape: Any,
    ) -> str:
        tensor_key = execution_tensor_key(list(tensor_tuple))
        if tensor_key is None:
            raise RuntimeError("Failed to serialize execution-trace tensor tuple")
        tensor_dot_id = tensor_dot_ids.get(tensor_key)
        if tensor_dot_id is not None:
            return tensor_dot_id

        tensor_dot_id = f"t{len(tensor_dot_ids)}"
        tensor_dot_ids[tensor_key] = tensor_dot_id
        tensor_labels[tensor_key] = execution_tensor_label(
            tensor_tuple,
            type_name=type_name,
            shape=shape,
        )
        return tensor_dot_id

    for event in raw_events:
        event_id = id(event)
        if event.device_type() == DeviceType.CPU and event.correlation_id() > 0:
            rf_id = event.correlation_id()
            if event_id in matched_execution_node_by_event:
                io_match_by_event[event_id] = "ok"
            elif rf_id in ambiguous_rf_ids:
                io_match_by_event[event_id] = "ambiguous"
            else:
                io_match_by_event[event_id] = "missing"

        execution_node = matched_execution_node_by_event.get(event_id)
        if execution_node is None:
            continue

        inputs = execution_node.get("inputs", {})
        input_values = inputs.get("values", [])
        input_types = inputs.get("types", [])
        input_shapes = inputs.get("shapes", [])
        scalar_labels_by_event[event_id] = summarize_execution_scalar_args(
            input_types,
            input_values,
            prefix="arg",
        )
        for input_index, (type_name, shape, value) in enumerate(
            zip(input_types, input_shapes, input_values)
        ):
            for entry in flatten_execution_tensor_entries(
                value,
                path=str(input_index),
                shape=shape,
                type_name=type_name,
            ):
                tensor_dot_id = ensure_tensor_node(
                    entry["tensor_tuple"],
                    type_name=entry["type_name"],
                    shape=entry["shape"],
                )
                input_edges.append(
                    (tensor_dot_id, node_ids[event_id], f"in{entry['path']}")
                )

        outputs = execution_node.get("outputs", {})
        for output_index, (type_name, shape, value) in enumerate(
            zip(
                outputs.get("types", []),
                outputs.get("shapes", []),
                outputs.get("values", []),
            )
        ):
            for entry in flatten_execution_tensor_entries(
                value,
                path=str(output_index),
                shape=shape,
                type_name=type_name,
            ):
                tensor_dot_id = ensure_tensor_node(
                    entry["tensor_tuple"],
                    type_name=entry["type_name"],
                    shape=entry["shape"],
                )
                output_edges.append(
                    (node_ids[event_id], tensor_dot_id, f"out{entry['path']}")
                )

    with dot_path.open("w") as f:
        f.write(f'digraph "{dot_escape(graph_name)}" {{\n')
        f.write("  newrank=true;\n")
        f.write("  rankdir=TB;\n")

        for event in raw_events:
            label_lines = kineto_runtime_node_label_lines(
                event,
                trace_start_ns=trace_start_ns,
                csv_matches=csv_matches,
            )
            execution_node = matched_execution_node_by_event.get(id(event))
            if execution_node is not None:
                rf_id = execution_rf_id(execution_node)
                if rf_id is not None:
                    label_lines.append(f"rf_id={rf_id}")
                kernel_file = execution_trace_attr(execution_node, "kernel_file")
                if isinstance(kernel_file, str) and kernel_file:
                    label_lines.append(f"kernel={Path(kernel_file).name}")
            io_match = io_match_by_event.get(id(event))
            if io_match is not None:
                label_lines.append(f"io_match={io_match}")
            label_lines.extend(scalar_labels_by_event.get(id(event), []))
            label = "\\n".join(dot_escape(line) for line in label_lines)
            f.write(
                f'  "{node_ids[id(event)]}" [shape=box, style=filled, '
                f'fillcolor="{kineto_node_fillcolor(event.device_type(), event.name())}", '
                f'label="{label}"];\n'
            )

        for tensor_key, tensor_dot_id in tensor_dot_ids.items():
            label = tensor_labels[tensor_key]
            f.write(
                f'  "{tensor_dot_id}" [shape=record, style=filled, '
                f'fillcolor="pink", label="{label}"];\n'
            )

        for src, dst in cpu_parent_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dashed, color="gray50", label="cpu"];\n'
            )
        for src, dst, label in linked_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="orangered3", penwidth=1.4, label="{label}"];\n'
            )
        for src, dst in stream_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dotted, color="steelblue4", label="stream"];\n'
            )
        for src, dst, label in input_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="black", label="{label}"];\n'
            )
        for src, dst, label in output_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="black", label="{label}"];\n'
            )
        f.write("}\n")


def execution_node_stream_hint(node: dict[str, Any]) -> int | None:
    # The execution trace observer (execution_trace_observer.cpp) appends
    # the CUDA stream as the last Int input for Triton kernels only.
    if execution_trace_attr(node, "kernel_backend") != "triton":
        return None
    inputs = node.get("inputs", {})
    input_values = inputs.get("values", [])
    input_types = inputs.get("types", [])
    if not isinstance(input_values, list) or not isinstance(input_types, list):
        return None
    if not input_values or not input_types:
        return None
    if len(input_values) != len(input_types):
        return None
    value = input_values[-1]
    type_name = input_types[-1]
    if isinstance(value, int) and type_name == "Int":
        return value
    return None


def execution_is_llamasim_runtime_node(node: dict[str, Any]) -> bool:
    kernel_backend = execution_trace_attr(node, "kernel_backend")
    if isinstance(kernel_backend, str) and kernel_backend:
        return True
    name = str(node.get("name", ""))
    return name.startswith("triton_")


def match_execution_nodes_to_gpu_kernels(
    raw_events: list[Any],
    execution_nodes: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[str], list[str]]:
    from torch.autograd import DeviceType

    gpu_events_by_linked_rf_id: dict[int, list[Any]] = {}
    for event in raw_events:
        if event.device_type() == DeviceType.CPU:
            continue
        linked_correlation_id = event.linked_correlation_id()
        if linked_correlation_id <= 0:
            continue
        gpu_events_by_linked_rf_id.setdefault(linked_correlation_id, []).append(event)

    matched_execution_node_by_event: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    matched_event_ids: set[int] = set()

    for node in execution_nodes.values():
        if not execution_is_llamasim_runtime_node(node):
            continue

        rf_id = execution_rf_id(node)
        if rf_id is None:
            warnings.append(
                f"Execution node {node.get('name', '<unknown>')} is missing rf_id"
            )
            continue

        candidates = list(gpu_events_by_linked_rf_id.get(rf_id, ()))
        if not candidates:
            warnings.append(
                f"No linked GPU kernel found for execution node {node.get('name', '<unknown>')} rf_id={rf_id}"
            )
            continue

        same_name = [event for event in candidates if event.name() == node["name"]]
        if same_name:
            candidates = same_name

        stream_hint = execution_node_stream_hint(node)
        if stream_hint is not None:
            same_stream = [
                event
                for event in candidates
                if event.device_resource_id() == stream_hint
            ]
            if same_stream:
                candidates = same_stream

        if len(candidates) != 1:
            candidate_summary = ", ".join(
                f"{event.name()}@stream{event.device_resource_id()} linked={event.linked_correlation_id()}"
                for event in candidates
            )
            errors.append(
                "Ambiguous GPU kernel match for execution node "
                f"{node.get('name', '<unknown>')} rf_id={rf_id}: {candidate_summary}"
            )
            continue

        matched_event = candidates[0]
        event_id = id(matched_event)
        if event_id in matched_event_ids:
            errors.append(
                "GPU kernel matched more than once: "
                f"{matched_event.name()} cid={matched_event.correlation_id()}"
            )
            continue

        matched_event_ids.add(event_id)
        matched_execution_node_by_event[event_id] = node

    return matched_execution_node_by_event, errors, warnings


def execution_type_to_llamasim_dtype(type_name: Any) -> str | None:
    if not isinstance(type_name, str):
        return None
    prefix = "Tensor("
    suffix = ")"
    if type_name.startswith(prefix) and type_name.endswith(suffix):
        inner = type_name[len(prefix) : -len(suffix)]
    else:
        inner = type_name
    return {
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "bfloat16": "bf16",
        "int8": "i8",
        "uint8": "u8",
        "int16": "i16",
        "uint16": "u16",
        "int32": "i32",
        "uint32": "u32",
        "int64": "i64",
        "uint64": "u64",
        "bool": "bool",
    }.get(inner)


def execution_shape_to_tuple(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, list) and all(isinstance(dim, int) for dim in shape):
        return tuple(shape)
    return ()


def llamasim_tensor_name(
    tensor_tuple: tuple[int, int, int, int, int, str],
    *,
    kind: str,
) -> str:
    tensor_id, storage_id, offset, _numel, _itemsize, _device = tensor_tuple
    prefix = "cache" if kind == "CONTEXT" else "param"
    if offset:
        return f"{prefix}_{tensor_id}_{storage_id}_off{offset}"
    return f"{prefix}_{tensor_id}_{storage_id}"


def raw_event_sort_key(event: Any) -> tuple[Any, ...]:
    return (
        event.start_ns(),
        event.end_ns(),
        getattr(event.device_type(), "name", str(event.device_type())),
        event.device_index(),
        event.device_resource_id(),
        event.start_thread_id(),
        event.correlation_id(),
        event.name(),
    )


def is_llamasim_gpu_runtime_event(event: Any) -> bool:
    from torch.autograd import DeviceType

    return event.device_type() != DeviceType.CPU and not event.is_user_annotation()


def select_llamasim_runtime_events(
    raw_events: list[Any],
    *,
    extra_cpu_event_ids: set[int] | None = None,
    exclude_cpu_event_ids: set[int] | None = None,
) -> tuple[list[Any], list[Any], list[Any]]:
    from torch.autograd import DeviceType

    extra_cpu_event_ids = extra_cpu_event_ids or set()
    exclude_cpu_event_ids = exclude_cpu_event_ids or set()
    node_ids, cpu_parent_edges, _linked_edges, _stream_edges = build_kineto_runtime_graph(
        raw_events
    )
    cpu_parent_node_ids = {src for src, _dst in cpu_parent_edges}

    selected_cpu_events = [
        event
        for event in raw_events
        if event.device_type() == DeviceType.CPU
        and id(event) not in exclude_cpu_event_ids
        and (
            node_ids[id(event)] not in cpu_parent_node_ids
            or id(event) in extra_cpu_event_ids
        )
    ]
    selected_gpu_events = [
        event for event in raw_events if is_llamasim_gpu_runtime_event(event)
    ]
    selected_events = sorted(
        [*selected_cpu_events, *selected_gpu_events],
        key=raw_event_sort_key,
    )
    return selected_events, selected_cpu_events, selected_gpu_events


def build_llamasim_thread_order_edges(
    node_ids: dict[int, str], cpu_events: list[Any]
) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    cpu_events_by_thread: dict[int, list[Any]] = {}
    for event in cpu_events:
        cpu_events_by_thread.setdefault(event.start_thread_id(), []).append(event)

    for thread_events in cpu_events_by_thread.values():
        thread_events.sort(key=raw_event_sort_key)
        for current, nxt in zip(thread_events, thread_events[1:]):
            if current.end_ns() > nxt.start_ns():
                continue
            edges.append((node_ids[id(current)], node_ids[id(nxt)], "thread_order"))
    return edges


def build_llamasim_stream_order_edges(
    node_ids: dict[int, str], gpu_events: list[Any]
) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    gpu_events_by_resource: dict[tuple[Any, int, int], list[Any]] = {}
    for event in gpu_events:
        key = (
            event.device_type(),
            event.device_index(),
            event.device_resource_id(),
        )
        gpu_events_by_resource.setdefault(key, []).append(event)

    for device_events in gpu_events_by_resource.values():
        device_events.sort(key=raw_event_sort_key)
        for current, nxt in zip(device_events, device_events[1:]):
            edges.append((node_ids[id(current)], node_ids[id(nxt)], "stream_order"))
    return edges


def build_llamasim_submit_edges(
    node_ids: dict[int, str],
    cpu_events: list[Any],
    gpu_events: list[Any],
    all_raw_events: list[Any],
) -> tuple[list[tuple[str, str, str]], dict[int, str]]:
    from torch.autograd import DeviceType

    # Index ALL CPU events by correlation_id so we can find launch calls
    # even if they were classified as parent (non-leaf) events.
    cpu_events_by_corr_id: dict[int, list[Any]] = {}
    for event in all_raw_events:
        if event.device_type() != DeviceType.CPU:
            continue
        corr_id = event.correlation_id()
        if corr_id > 0:
            cpu_events_by_corr_id.setdefault(corr_id, []).append(event)

    submit_edges: list[tuple[str, str, str]] = []
    runtime_role_by_event_id: dict[int, str] = {}
    seen_pairs: set[tuple[str, str, str]] = set()

    for gpu_event in gpu_events:
        corr_id = gpu_event.correlation_id()
        if corr_id <= 0:
            continue
        candidates = list(cpu_events_by_corr_id.get(corr_id, ()))
        if not candidates:
            continue
        # Prefer candidates that are in the selected node set.
        in_graph = [e for e in candidates if id(e) in node_ids]
        if in_graph:
            candidates = in_graph
        else:
            # The launch CPU event is a parent (not selected); skip since
            # we can't create an edge to a node that isn't in the graph.
            continue
        before_start = [
            event for event in candidates if event.start_ns() <= gpu_event.start_ns()
        ]
        if before_start:
            candidates = before_start
        submit_event = max(
            candidates,
            key=lambda event: (
                event.end_ns(),
                event.start_ns(),
                event.name(),
            ),
        )
        edge = (node_ids[id(submit_event)], node_ids[id(gpu_event)], "submit")
        if edge in seen_pairs:
            continue
        seen_pairs.add(edge)
        submit_edges.append(edge)
        runtime_role_by_event_id[id(submit_event)] = "submit"

    return submit_edges, runtime_role_by_event_id


_AC2G_CPU_ENDPOINT_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
_AC2G_GPU_ENDPOINT_CATEGORIES = frozenset(
    {"kernel", "gpu_memcpy", "gpu_memset"}
)
_AC2G_TRACE_ENDPOINT_TOLERANCE_US = 1e-3
_AC2G_RAW_EVENT_MATCH_TOLERANCE_US = 2.0


@dataclass(frozen=True)
class Ac2gTraceEndpoint:
    correlation_id: int
    category: str
    name: str
    ts_us: float
    relative_ts_us: float
    duration_us: float
    pid: Any
    tid: Any
    device_index: int | None
    resource_id: int | None


@dataclass(frozen=True)
class Ac2gFlowEndpoints:
    paired_correlation_ids: set[int]
    # {correlation_id: ts_us} — CPU-side ac2g start timestamp relative to trace start.
    start_ts_us_by_corr_id: dict[int, float]
    cpu_endpoint_by_corr_id: dict[int, Ac2gTraceEndpoint]
    gpu_endpoint_by_corr_id: dict[int, Ac2gTraceEndpoint]
    trace_start_us: float


@dataclass(frozen=True)
class Ac2gRawEventPair:
    correlation_id: int
    cpu_event: Any
    gpu_event: Any
    cpu_endpoint: Ac2gTraceEndpoint
    gpu_endpoint: Ac2gTraceEndpoint


@dataclass(frozen=True)
class Ac2gRawEventMatches:
    pairs_by_corr_id: dict[int, Ac2gRawEventPair]
    errors: list[str]


def _trace_event_correlation_id(event: dict[str, Any]) -> int | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    for key in ("correlation", "correlation_id"):
        value = args.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _trace_event_int_arg(event: dict[str, Any], key: str) -> int | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    value = args.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_chrome_trace_start_us(trace_events: list[dict[str, Any]]) -> float:
    for event in trace_events:
        if event.get("ph") == "X" and event.get("cat") == "Trace":
            ts = event.get("ts")
            if ts is not None:
                return float(ts)
    for event in trace_events:
        if event.get("name") == "Iteration Start: PyTorch Profiler":
            ts = event.get("ts")
            if ts is not None:
                return float(ts)
    return 0.0


def _select_ac2g_trace_endpoint(
    *,
    correlation_id: int,
    flow_event: dict[str, Any],
    x_events: list[dict[str, Any]],
    categories: frozenset[str],
    trace_start_us: float,
) -> Ac2gTraceEndpoint | None:
    flow_ts = flow_event.get("ts")
    if flow_ts is None:
        return None
    flow_ts_us = float(flow_ts)
    candidates: list[dict[str, Any]] = []
    for event in x_events:
        if event.get("ph") != "X" or event.get("cat") not in categories:
            continue
        ts = event.get("ts")
        dur = event.get("dur")
        name = event.get("name")
        if ts is None or dur is None or not isinstance(name, str):
            continue
        if abs(float(ts) - flow_ts_us) <= _AC2G_TRACE_ENDPOINT_TOLERANCE_US:
            candidates.append(event)
    if len(candidates) != 1:
        return None

    endpoint_event = candidates[0]
    category = str(endpoint_event.get("cat"))
    return Ac2gTraceEndpoint(
        correlation_id=correlation_id,
        category=category,
        name=str(endpoint_event["name"]),
        ts_us=float(endpoint_event["ts"]),
        relative_ts_us=float(endpoint_event["ts"]) - trace_start_us,
        duration_us=float(endpoint_event["dur"]),
        pid=endpoint_event.get("pid"),
        tid=endpoint_event.get("tid"),
        device_index=_trace_event_int_arg(endpoint_event, "device"),
        resource_id=_trace_event_int_arg(endpoint_event, "stream"),
    )


def parse_ac2g_flow_endpoints(
    trace_json_path: Path, trace_start_us: float | None = None
) -> Ac2gFlowEndpoints:
    with trace_json_path.open(encoding="utf-8") as f:
        trace_json = json.load(f)
    trace_events = trace_json.get("traceEvents", [])
    if not isinstance(trace_events, list):
        trace_events = []
    if trace_start_us is None:
        trace_start_us = _infer_chrome_trace_start_us(trace_events)

    starts: dict[int, list[dict[str, Any]]] = {}
    ends: dict[int, list[dict[str, Any]]] = {}
    x_events_by_corr_id: dict[int, list[dict[str, Any]]] = {}
    for event in trace_events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") == "X":
            corr_id = _trace_event_correlation_id(event)
            if corr_id is not None:
                x_events_by_corr_id.setdefault(corr_id, []).append(event)
            continue

        if event.get("cat") != "ac2g":
            continue
        flow_id = event.get("id")
        if flow_id is None:
            continue
        try:
            corr_id = int(flow_id)
        except (TypeError, ValueError):
            continue
        ph = event.get("ph")
        if ph == "s":
            starts.setdefault(corr_id, []).append(event)
        elif ph == "f":
            ends.setdefault(corr_id, []).append(event)

    paired = set(starts) & set(ends)
    start_ts_us: dict[int, float] = {}
    cpu_endpoint_by_corr_id: dict[int, Ac2gTraceEndpoint] = {}
    gpu_endpoint_by_corr_id: dict[int, Ac2gTraceEndpoint] = {}
    for corr_id in paired:
        start_events = sorted(starts[corr_id], key=lambda e: float(e.get("ts", 0.0)))
        end_events = sorted(ends[corr_id], key=lambda e: float(e.get("ts", 0.0)))
        start_event = start_events[0]
        end_event = end_events[0]
        ts = start_event.get("ts")
        if ts is not None:
            start_ts_us[corr_id] = float(ts) - trace_start_us
        x_events = x_events_by_corr_id.get(corr_id, [])
        cpu_endpoint = _select_ac2g_trace_endpoint(
            correlation_id=corr_id,
            flow_event=start_event,
            x_events=x_events,
            categories=_AC2G_CPU_ENDPOINT_CATEGORIES,
            trace_start_us=trace_start_us,
        )
        if cpu_endpoint is not None:
            cpu_endpoint_by_corr_id[corr_id] = cpu_endpoint
        gpu_endpoint = _select_ac2g_trace_endpoint(
            correlation_id=corr_id,
            flow_event=end_event,
            x_events=x_events,
            categories=_AC2G_GPU_ENDPOINT_CATEGORIES,
            trace_start_us=trace_start_us,
        )
        if gpu_endpoint is not None:
            gpu_endpoint_by_corr_id[corr_id] = gpu_endpoint

    return Ac2gFlowEndpoints(
        paired_correlation_ids=paired,
        start_ts_us_by_corr_id={k: v for k, v in start_ts_us.items() if k in paired},
        cpu_endpoint_by_corr_id=cpu_endpoint_by_corr_id,
        gpu_endpoint_by_corr_id=gpu_endpoint_by_corr_id,
        trace_start_us=trace_start_us,
    )


def parse_ac2g_paired_correlation_ids(trace_json_path: Path) -> set[int]:
    return parse_ac2g_flow_endpoints(trace_json_path).paired_correlation_ids


def _event_relative_start_us(event: Any, trace_start_ns: int) -> float:
    return (event.start_ns() - trace_start_ns) / 1000.0


def _event_duration_us(event: Any) -> float:
    return event.duration_ns() / 1000.0


def _match_raw_event_to_ac2g_endpoint(
    raw_events: list[Any],
    endpoint: Ac2gTraceEndpoint,
    *,
    trace_start_ns: int,
) -> tuple[Any | None, str | None]:
    from torch.autograd import DeviceType

    is_cpu_endpoint = endpoint.category in _AC2G_CPU_ENDPOINT_CATEGORIES
    candidates: list[Any] = []
    for event in raw_events:
        if event.correlation_id() != endpoint.correlation_id:
            continue
        if event.name() != endpoint.name:
            continue
        if is_cpu_endpoint:
            if event.device_type() != DeviceType.CPU:
                continue
        else:
            if event.device_type() == DeviceType.CPU or event.is_user_annotation():
                continue
            if (
                endpoint.device_index is not None
                and event.device_index() != endpoint.device_index
            ):
                continue
            if (
                endpoint.resource_id is not None
                and event.device_resource_id() != endpoint.resource_id
            ):
                continue
        if (
            abs(_event_relative_start_us(event, trace_start_ns) - endpoint.relative_ts_us)
            > _AC2G_RAW_EVENT_MATCH_TOLERANCE_US
        ):
            continue
        if (
            abs(_event_duration_us(event) - endpoint.duration_us)
            > _AC2G_RAW_EVENT_MATCH_TOLERANCE_US
        ):
            continue
        candidates.append(event)

    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return (
            None,
            "ambiguous raw Kineto endpoint "
            f"cid={endpoint.correlation_id} name={endpoint.name!r} "
            f"category={endpoint.category!r} candidates={len(candidates)}",
        )

    same_corr = [
        event.name()
        for event in raw_events
        if event.correlation_id() == endpoint.correlation_id
    ]
    return (
        None,
        "missing raw Kineto endpoint "
        f"cid={endpoint.correlation_id} name={endpoint.name!r} "
        f"category={endpoint.category!r} start_us={endpoint.relative_ts_us:.3f} "
        f"dur_us={endpoint.duration_us:.3f} corr_names={sorted(set(same_corr))[:6]}",
    )


def match_ac2g_endpoints_to_raw_events(
    raw_events: list[Any],
    ac2g_endpoints: Ac2gFlowEndpoints,
    *,
    trace_start_ns: int = 0,
) -> Ac2gRawEventMatches:
    raw_events_by_corr_id: dict[int, list[Any]] = {}
    for event in raw_events:
        corr_id = event.correlation_id()
        if corr_id in ac2g_endpoints.paired_correlation_ids:
            raw_events_by_corr_id.setdefault(corr_id, []).append(event)

    pairs_by_corr_id: dict[int, Ac2gRawEventPair] = {}
    errors: list[str] = []
    for corr_id in sorted(ac2g_endpoints.paired_correlation_ids):
        cpu_endpoint = ac2g_endpoints.cpu_endpoint_by_corr_id.get(corr_id)
        gpu_endpoint = ac2g_endpoints.gpu_endpoint_by_corr_id.get(corr_id)
        if cpu_endpoint is None:
            errors.append(f"missing Chrome ac2g CPU endpoint cid={corr_id}")
            continue
        if gpu_endpoint is None:
            errors.append(f"missing Chrome ac2g GPU endpoint cid={corr_id}")
            continue
        raw_candidates = raw_events_by_corr_id.get(corr_id, [])
        cpu_event, cpu_error = _match_raw_event_to_ac2g_endpoint(
            raw_candidates,
            cpu_endpoint,
            trace_start_ns=trace_start_ns,
        )
        gpu_event, gpu_error = _match_raw_event_to_ac2g_endpoint(
            raw_candidates,
            gpu_endpoint,
            trace_start_ns=trace_start_ns,
        )
        if cpu_error is not None:
            errors.append(cpu_error)
        if gpu_error is not None:
            errors.append(gpu_error)
        if cpu_event is None or gpu_event is None:
            continue
        pairs_by_corr_id[corr_id] = Ac2gRawEventPair(
            correlation_id=corr_id,
            cpu_event=cpu_event,
            gpu_event=gpu_event,
            cpu_endpoint=cpu_endpoint,
            gpu_endpoint=gpu_endpoint,
        )

    return Ac2gRawEventMatches(
        pairs_by_corr_id=pairs_by_corr_id,
        errors=errors,
    )


def _format_ac2g_match_errors(errors: list[str], *, limit: int = 20) -> str:
    shown = errors[:limit]
    suffix = "" if len(errors) <= limit else f"\n... {len(errors) - limit} more"
    return "\n".join(shown) + suffix


def build_llamasim_submit_edges_from_ac2g(
    node_ids: dict[int, str],
    cpu_events: list[Any],
    gpu_events: list[Any],
    all_raw_events: list[Any],
    ac2g_correlation_ids: set[int],
    *,
    ac2g_endpoints: Ac2gFlowEndpoints | None = None,
    trace_start_ns: int = 0,
    ac2g_matches: Ac2gRawEventMatches | None = None,
    strict: bool = False,
) -> tuple[list[tuple[str, str, str]], dict[int, str], dict[tuple[str, str], dict[str, str]]]:
    from torch.autograd import DeviceType

    if ac2g_endpoints is not None:
        if ac2g_matches is None:
            ac2g_matches = match_ac2g_endpoints_to_raw_events(
                all_raw_events,
                ac2g_endpoints,
                trace_start_ns=trace_start_ns,
            )
        if strict and ac2g_matches.errors:
            raise RuntimeError(
                "Failed to match all Chrome ac2g endpoints to raw Kineto events:\n"
                + _format_ac2g_match_errors(ac2g_matches.errors)
            )

        submit_edges: list[tuple[str, str, str]] = []
        submit_edge_attrs: dict[tuple[str, str], dict[str, str]] = {}
        runtime_role_by_event_id: dict[int, str] = {}
        seen_pairs: set[tuple[str, str, str]] = set()
        selected_gpu_event_ids = {id(event) for event in gpu_events}
        requested_ac2g_correlation_ids = set(ac2g_correlation_ids)
        missing_node_errors: list[str] = []

        matched_pairs = sorted(
            ac2g_matches.pairs_by_corr_id.values(),
            key=lambda pair: raw_event_sort_key(pair.gpu_event),
        )
        for pair in matched_pairs:
            if pair.correlation_id not in requested_ac2g_correlation_ids:
                continue
            if id(pair.gpu_event) not in selected_gpu_event_ids:
                continue
            if id(pair.cpu_event) not in node_ids:
                missing_node_errors.append(
                    f"exact ac2g submit node is not in runtime graph cid={pair.correlation_id} "
                    f"name={pair.cpu_event.name()!r}"
                )
                continue
            if id(pair.gpu_event) not in node_ids:
                missing_node_errors.append(
                    f"exact ac2g GPU node is not in runtime graph cid={pair.correlation_id} "
                    f"name={pair.gpu_event.name()!r}"
                )
                continue
            edge = (
                node_ids[id(pair.cpu_event)],
                node_ids[id(pair.gpu_event)],
                "submit",
            )
            if edge in seen_pairs:
                continue
            seen_pairs.add(edge)
            submit_edges.append(edge)
            submit_edge_attrs[(edge[0], edge[1])] = {"submit_exact": "true"}
            runtime_role_by_event_id[id(pair.cpu_event)] = "submit"

        if strict:
            if missing_node_errors:
                raise RuntimeError(
                    "Failed to emit exact ac2g submit edges:\n"
                    + _format_ac2g_match_errors(missing_node_errors)
                )
            expected_gpu_event_ids = {
                id(pair.gpu_event)
                for pair in ac2g_matches.pairs_by_corr_id.values()
                if (
                    pair.correlation_id in requested_ac2g_correlation_ids
                    and id(pair.gpu_event) in selected_gpu_event_ids
                )
            }
            submitted_gpu_event_ids = {
                id(pair.gpu_event)
                for pair in ac2g_matches.pairs_by_corr_id.values()
                if (
                    pair.correlation_id in requested_ac2g_correlation_ids
                    and id(pair.gpu_event) in selected_gpu_event_ids
                    and id(pair.cpu_event) in node_ids
                    and (
                        node_ids[id(pair.cpu_event)],
                        node_ids[id(pair.gpu_event)],
                        "submit",
                    )
                    in seen_pairs
                )
            }
            if submitted_gpu_event_ids != expected_gpu_event_ids:
                raise RuntimeError(
                    "Failed to emit one exact submit edge per selected ac2g GPU event "
                    f"(expected={len(expected_gpu_event_ids)}, emitted={len(submitted_gpu_event_ids)})"
                )

        return submit_edges, runtime_role_by_event_id, submit_edge_attrs

    cpu_events_by_corr_id: dict[int, list[Any]] = {}
    for event in all_raw_events:
        if event.device_type() != DeviceType.CPU:
            continue
        corr_id = event.correlation_id()
        if corr_id in ac2g_correlation_ids:
            cpu_events_by_corr_id.setdefault(corr_id, []).append(event)

    submit_edges: list[tuple[str, str, str]] = []
    submit_edge_attrs: dict[tuple[str, str], dict[str, str]] = {}
    runtime_role_by_event_id: dict[int, str] = {}
    seen_pairs: set[tuple[str, str, str]] = set()

    for gpu_event in gpu_events:
        corr_id = gpu_event.correlation_id()
        if corr_id not in ac2g_correlation_ids:
            continue
        candidates = list(cpu_events_by_corr_id.get(corr_id, ()))
        if not candidates:
            continue
        in_graph = [e for e in candidates if id(e) in node_ids]
        if not in_graph:
            continue
        candidates = in_graph

        before_start = [
            event
            for event in candidates
            if event.start_ns() <= gpu_event.start_ns()
        ]
        if before_start:
            candidates = before_start
        submit_event = max(
            candidates,
            key=lambda event: (
                event.end_ns(),
                event.start_ns(),
                event.name(),
            ),
        )

        edge = (node_ids[id(submit_event)], node_ids[id(gpu_event)], "submit")
        if edge in seen_pairs:
            continue
        seen_pairs.add(edge)
        submit_edges.append(edge)
        submit_edge_attrs[(edge[0], edge[1])] = {"submit_exact": "false"}
        runtime_role_by_event_id[id(submit_event)] = "submit"

    return submit_edges, runtime_role_by_event_id, submit_edge_attrs


def build_llamasim_wait_edges(
    node_ids: dict[int, str],
    cpu_events: list[Any],
    gpu_events: list[Any],
    runtime_role_by_event_id: dict[int, str],
) -> tuple[list[tuple[str, str, str]], dict[int, str]]:
    gpu_events_by_family: dict[int, list[Any]] = {}
    for event in gpu_events:
        linked_corr_id = event.linked_correlation_id()
        if linked_corr_id > 0:
            gpu_events_by_family.setdefault(linked_corr_id, []).append(event)

    cpu_events_by_family: dict[int, list[Any]] = {}
    for event in cpu_events:
        linked_corr_id = event.linked_correlation_id()
        if linked_corr_id > 0:
            cpu_events_by_family.setdefault(linked_corr_id, []).append(event)

    wait_edges: list[tuple[str, str, str]] = []
    wait_role_by_event_id = dict(runtime_role_by_event_id)
    seen_pairs: set[tuple[str, str, str]] = set()

    for linked_corr_id, family_gpu_events in gpu_events_by_family.items():
        family_cpu_events = cpu_events_by_family.get(linked_corr_id, [])
        if not family_cpu_events:
            continue

        family_gpu_events = sorted(family_gpu_events, key=raw_event_sort_key)
        family_end_ns = max(event.end_ns() for event in family_gpu_events)
        submit_corr_ids = {event.correlation_id() for event in family_gpu_events}

        for cpu_event in family_cpu_events:
            if id(cpu_event) in wait_role_by_event_id:
                continue
            if cpu_event.correlation_id() in submit_corr_ids:
                continue
            if cpu_event.end_ns() < family_end_ns:
                continue
            if cpu_event.start_ns() > family_end_ns:
                continue

            added_wait_edge = False
            for gpu_event in family_gpu_events:
                edge = (node_ids[id(gpu_event)], node_ids[id(cpu_event)], "wait")
                if edge in seen_pairs:
                    continue
                seen_pairs.add(edge)
                wait_edges.append(edge)
                added_wait_edge = True
            if added_wait_edge:
                wait_role_by_event_id[id(cpu_event)] = "wait"

    return wait_edges, wait_role_by_event_id


_CUDA_SYNC_EVENT_NAMES = frozenset({
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
})

_CUDA_DEVICE_SYNC_EVENT_NAMES = frozenset({
    "cudaDeviceSynchronize",
})
_CUDA_STREAM_SYNC_EVENT_NAMES = frozenset({
    "cudaStreamSynchronize",
})
_CUDA_EVENT_SYNC_EVENT_NAMES = frozenset({
    "cudaEventSynchronize",
})
_CUDA_STREAM_WAIT_EVENT_NAMES = frozenset({
    "cudaStreamWaitEvent",
})
_CUDA_ASYNC_SUBMIT_EVENT_NAMES = frozenset({
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cudaMemcpyAsync",
    "cudaMemcpy2DAsync",
    "cudaMemcpy3DAsync",
    "cudaMemsetAsync",
})
_ATEN_PARENT_SCOPE_EXACT_SYNC_NAMES = frozenset({
    "aten::copy_",
    "aten::_to_copy",
    "aten::nonzero",
    "aten::_local_scalar_dense",
})
_PROFILER_SYNC_RE = re.compile(r"^profiler_sync:(\w+)$")


def heuristic_sync_wait_attrs(event_name: str) -> dict[str, str]:
    attrs = {
        "wait_exact": "false",
        "wait_source": "kineto_sync_frontier",
    }
    if event_name in _CUDA_EVENT_SYNC_EVENT_NAMES:
        attrs["wait_kind"] = "event_sync"
    elif event_name in _CUDA_STREAM_SYNC_EVENT_NAMES:
        attrs["wait_kind"] = "stream_sync"
    elif event_name in _CUDA_DEVICE_SYNC_EVENT_NAMES:
        attrs["wait_kind"] = "device_sync"
    return attrs


def wait_kind_for_sync_event(event_name: str) -> str:
    if event_name in _CUDA_EVENT_SYNC_EVENT_NAMES:
        return "event_sync"
    if event_name in _CUDA_STREAM_SYNC_EVENT_NAMES:
        return "stream_sync"
    if event_name in _CUDA_DEVICE_SYNC_EVENT_NAMES:
        return "device_sync"
    if event_name in _CUDA_STREAM_WAIT_EVENT_NAMES or event_name.startswith(
        "cudaStreamWaitEvent"
    ):
        return "wait_event"
    return ""


def _collect_cpu_descendant_ids(
    parent_id: int,
    parent_to_children: dict[int, list[int]],
) -> list[int]:
    descendants: list[int] = []
    stack = list(parent_to_children.get(parent_id, ()))
    while stack:
        event_id = stack.pop()
        descendants.append(event_id)
        stack.extend(parent_to_children.get(event_id, ()))
    return descendants


def find_profiler_sync_excluded_cpu_event_ids(raw_events: list[Any]) -> set[int]:
    """Return CUDA sync leaves nested under profiler_sync:* markers."""
    from torch.autograd import DeviceType

    _child_to_parent, parent_to_children, event_by_id = _build_cpu_parent_child_maps(
        raw_events
    )
    excluded: set[int] = set()
    for event in raw_events:
        if event.device_type() != DeviceType.CPU:
            continue
        if _PROFILER_SYNC_RE.match(event.name()) is None:
            continue
        for event_id in _collect_cpu_descendant_ids(id(event), parent_to_children):
            descendant = event_by_id.get(event_id)
            if descendant is None:
                continue
            if wait_kind_for_sync_event(descendant.name()):
                excluded.add(event_id)
    return excluded


def _nearest_aten_exact_sync_parent_id(
    event_id: int,
    child_to_parent: dict[int, int],
    event_by_id: dict[int, Any],
) -> int | None:
    current = child_to_parent.get(event_id)
    while current is not None:
        event = event_by_id.get(current)
        if event is not None and event.name() in _ATEN_PARENT_SCOPE_EXACT_SYNC_NAMES:
            return current
        current = child_to_parent.get(current)
    return None


def _gpu_events_linked_to_cpu_submit_descendants(
    *,
    gpu_events: list[Any],
    cpu_descendants: list[Any],
    parent_event: Any,
    sync_event: Any,
    node_ids: dict[int, str],
) -> list[Any]:
    submit_corr_ids: set[int] = set()
    for event in cpu_descendants:
        if event.name() not in _CUDA_ASYNC_SUBMIT_EVENT_NAMES:
            continue
        corr_id = event.correlation_id()
        if corr_id > 0:
            submit_corr_ids.add(corr_id)
    if not submit_corr_ids:
        return []

    candidates: list[Any] = []
    for gpu_event in gpu_events:
        if id(gpu_event) not in node_ids:
            continue
        gpu_corr_ids = {
            cid
            for cid in (
                gpu_event.correlation_id(),
                gpu_event.linked_correlation_id(),
            )
            if cid > 0
        }
        if not gpu_corr_ids & submit_corr_ids:
            continue
        if gpu_event.start_ns() < parent_event.start_ns():
            continue
        if gpu_event.end_ns() > sync_event.end_ns():
            continue
        candidates.append(gpu_event)
    return candidates


def _select_aten_exact_wait_sources(
    sync_event: Any,
    candidate_gpu_events: list[Any],
) -> list[Any]:
    if not candidate_gpu_events:
        return []

    gpu_stream_resource_ids = {
        event.device_resource_id() for event in candidate_gpu_events
    }
    if sync_event.name() == "cudaStreamSynchronize":
        if sync_event.device_resource_id() in gpu_stream_resource_ids:
            candidate_gpu_events = [
                event
                for event in candidate_gpu_events
                if event.device_resource_id() == sync_event.device_resource_id()
            ]
        elif len(gpu_stream_resource_ids) != 1:
            return []
    elif sync_event.name() == "cudaDeviceSynchronize":
        # A device sync waits all streams. Parent-scope evidence alone does not
        # prove unrelated earlier stream work, so do not mark it exact here.
        return []

    latest_by_stream: dict[tuple[Any, int, int], Any] = {}
    for event in candidate_gpu_events:
        key = (
            event.device_type(),
            event.device_index(),
            event.device_resource_id(),
        )
        current = latest_by_stream.get(key)
        if current is None or raw_event_sort_key(current) < raw_event_sort_key(event):
            latest_by_stream[key] = event
    return sorted(latest_by_stream.values(), key=raw_event_sort_key)


def build_llamasim_aten_sync_wait_edges(
    node_ids: dict[int, str],
    cpu_events: list[Any],
    gpu_events: list[Any],
    raw_events: list[Any],
    runtime_role_by_event_id: dict[int, str],
) -> tuple[
    list[tuple[str, str, str]],
    dict[int, str],
    set[int],
    dict[tuple[str, str], dict[str, str]],
]:
    """Build exact wait edges for synchronous ATen runtime waits.

    These are sync leaves inside allowlisted ATen parent scopes where the same
    parent also launched the GPU memcpy/kernel being synchronized.
    """
    child_to_parent, parent_to_children, event_by_id = _build_cpu_parent_child_maps(
        raw_events
    )
    edges: list[tuple[str, str, str]] = []
    updated_roles = dict(runtime_role_by_event_id)
    handled: set[int] = set()
    attrs_by_edge: dict[tuple[str, str], dict[str, str]] = {}
    seen: set[tuple[str, str]] = set()

    for sync_event in cpu_events:
        if id(sync_event) in updated_roles:
            continue
        if id(sync_event) not in node_ids:
            continue
        wait_kind = wait_kind_for_sync_event(sync_event.name())
        if not wait_kind:
            continue
        parent_id = _nearest_aten_exact_sync_parent_id(
            id(sync_event),
            child_to_parent,
            event_by_id,
        )
        if parent_id is None:
            continue
        parent_event = event_by_id[parent_id]
        cpu_descendants = [
            event_by_id[event_id]
            for event_id in _collect_cpu_descendant_ids(
                parent_id,
                parent_to_children,
            )
            if event_id in event_by_id
        ]
        candidates = _gpu_events_linked_to_cpu_submit_descendants(
            gpu_events=gpu_events,
            cpu_descendants=cpu_descendants,
            parent_event=parent_event,
            sync_event=sync_event,
            node_ids=node_ids,
        )
        source_gpu_events = _select_aten_exact_wait_sources(sync_event, candidates)
        if not source_gpu_events:
            continue

        added = False
        dst = node_ids[id(sync_event)]
        source_corr_ids: set[int] = set()
        for source_gpu in source_gpu_events:
            src = node_ids[id(source_gpu)]
            pair = (src, dst)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append((src, dst, "wait"))
            attrs_by_edge[pair] = {
                "wait_exact": "true",
                "wait_source": "aten_parent_scope_exact",
                "wait_kind": wait_kind,
                "wait_source_launch_ids": "",
            }
            for corr_id in (
                source_gpu.correlation_id(),
                source_gpu.linked_correlation_id(),
            ):
                if corr_id > 0:
                    source_corr_ids.add(corr_id)
            added = True
        if added:
            updated_roles[id(sync_event)] = "wait"
            handled.add(id(sync_event))
            if source_corr_ids:
                corr_text = ",".join(str(cid) for cid in sorted(source_corr_ids))
                for source_gpu in source_gpu_events:
                    pair = (node_ids[id(source_gpu)], dst)
                    if pair in attrs_by_edge:
                        attrs_by_edge[pair]["wait_source_correlation_ids"] = corr_text

    return edges, updated_roles, handled, attrs_by_edge


def build_llamasim_sync_wait_edges(
    node_ids: dict[int, str],
    cpu_events: list[Any],
    gpu_events: list[Any],
    runtime_role_by_event_id: dict[int, str],
) -> tuple[list[tuple[str, str, str]], dict[int, str]]:
    from torch.autograd import DeviceType

    gpu_events_by_stream: dict[tuple[Any, int, int], list[Any]] = {}
    for event in gpu_events:
        key = (
            event.device_type(),
            event.device_index(),
            event.device_resource_id(),
        )
        gpu_events_by_stream.setdefault(key, []).append(event)
    for stream_events in gpu_events_by_stream.values():
        stream_events.sort(key=raw_event_sort_key)
    gpu_stream_resource_ids = {key[2] for key in gpu_events_by_stream}

    sync_wait_edges: list[tuple[str, str, str]] = []
    updated_roles = dict(runtime_role_by_event_id)
    seen_pairs: set[tuple[str, str, str]] = set()

    for cpu_event in cpu_events:
        if cpu_event.name() not in _CUDA_SYNC_EVENT_NAMES:
            continue
        if id(cpu_event) in updated_roles:
            continue
        if id(cpu_event) not in node_ids:
            continue

        source_gpu_events: list[Any] = []
        for stream_key, stream_events in gpu_events_by_stream.items():
            if cpu_event.name() == "cudaStreamSynchronize":
                # Kineto does not reliably expose the synchronized CUDA stream
                # on the CPU event.  When it does, restrict to that stream;
                # otherwise fall back to the latest completed event per stream.
                if (
                    cpu_event.device_resource_id() in gpu_stream_resource_ids
                    and stream_key[2] != cpu_event.device_resource_id()
                ):
                    continue
            completed = [
                event for event in stream_events if event.end_ns() <= cpu_event.end_ns()
            ]
            if not completed:
                continue
            source_gpu_events.append(max(completed, key=raw_event_sort_key))

        added = False
        for source_gpu in source_gpu_events:
            edge = (node_ids[id(source_gpu)], node_ids[id(cpu_event)], "wait")
            if edge in seen_pairs:
                continue
            seen_pairs.add(edge)
            sync_wait_edges.append(edge)
            added = True
        if added:
            updated_roles[id(cpu_event)] = "wait"

    return sync_wait_edges, updated_roles


def _propagate_parent_tensor_io(
    raw_events: list[Any],
    matched_execution_node_by_event: dict[int, dict[str, Any]],
    execution_nodes_by_rf_id: dict[int, list[dict[str, Any]]],
) -> dict[str, int]:
    """Inherit tensor I/O for GPU events without a direct execution trace match.

    cuBLAS/cutlass kernels lack kernel_backend in the execution trace, so
    match_execution_nodes_to_gpu_kernels skips them.  Their Kineto
    linked_correlation_id still points to the parent ATen op's rf_id
    (e.g. aten::mm), which carries the tensor I/O we need.

    Returns a stats dict with counts for each outcome category.
    """
    from torch.autograd import DeviceType

    propagated = 0
    no_linked_rf_id = 0
    no_execution_node = 0
    ambiguous_rf_id = 0

    for event in raw_events:
        if event.device_type() == DeviceType.CPU:
            continue
        if id(event) in matched_execution_node_by_event:
            continue
        linked_rf_id = event.linked_correlation_id()
        if linked_rf_id <= 0:
            no_linked_rf_id += 1
            continue
        candidates = execution_nodes_by_rf_id.get(linked_rf_id, [])
        if not candidates:
            no_execution_node += 1
            continue
        if len(candidates) > 1:
            ambiguous_rf_id += 1
            continue
        matched_execution_node_by_event[id(event)] = candidates[0]
        propagated += 1

    return {
        "propagated": propagated,
        "no_linked_rf_id": no_linked_rf_id,
        "no_execution_node": no_execution_node,
        "ambiguous_rf_id": ambiguous_rf_id,
    }


_WS_LAUNCH_RE = re.compile(r"^ws_launch:(\d+):(\d+)$")


def build_ws_launch_index(
    raw_events: list[Any],
) -> dict[int, tuple[int, int]]:
    """Map ``id(event) → (graph_id, launch_id)`` via ws_launch markers.

    ``LaunchMarkerLine`` (wrapper.py) emits ``record_function("ws_launch:{graph_id}:{launch_id}")``
    wrapping each compute line.  In the Kineto trace this creates a CPU
    parent event.  GPU kernels link back via ``linked_correlation_id``.
    We walk these chains to propagate the IDs to every event.
    """
    from torch.autograd import DeviceType

    # 1. Identify ws_launch CPU markers.
    ws_markers: dict[int, tuple[int, int]] = {}
    for event in raw_events:
        m = _WS_LAUNCH_RE.match(event.name())
        if m and event.device_type() == DeviceType.CPU:
            ws_markers[id(event)] = (int(m.group(1)), int(m.group(2)))

    if not ws_markers:
        return {}

    # 2. Build child→parent map using CPU event nesting (same algorithm as
    #    build_kineto_runtime_graph lines 1346-1360).
    cpu_sync_events_by_thread: dict[int, list[Any]] = {}
    for event in raw_events:
        is_async = event.is_async() or (
            event.start_thread_id() != event.end_thread_id()
        )
        if event.device_type() == DeviceType.CPU and not is_async:
            cpu_sync_events_by_thread.setdefault(
                event.start_thread_id(), []
            ).append(event)

    child_to_parent: dict[int, int] = {}
    for thread_events in cpu_sync_events_by_thread.values():
        thread_events.sort(key=lambda e: (e.start_ns(), -e.end_ns()))
        stack: list[Any] = []
        for event in thread_events:
            while stack:
                parent = stack[-1]
                if (
                    event.start_ns() >= parent.end_ns()
                    or event.end_ns() > parent.end_ns()
                ):
                    stack.pop()
                else:
                    child_to_parent[id(event)] = id(parent)
                    break
            stack.append(event)

    # 3. Build correlation_id → id(event) for CPU events.
    corr_to_event_id: dict[int, int] = {}
    for event in raw_events:
        if event.device_type() == DeviceType.CPU:
            cid = event.correlation_id()
            if cid > 0:
                corr_to_event_id[cid] = id(event)

    # 4. For each event, resolve its ws_launch ancestor.
    def _walk_to_ws_marker(event_id: int) -> tuple[int, int] | None:
        cur = event_id
        while cur is not None:
            hit = ws_markers.get(cur)
            if hit is not None:
                return hit
            cur = child_to_parent.get(cur)
        return None

    index: dict[int, tuple[int, int]] = {}
    for event in raw_events:
        eid = id(event)
        if event.device_type() == DeviceType.CPU:
            resolved = _walk_to_ws_marker(eid)
        else:
            # GPU event: find CPU launch parent via linked_correlation_id.
            linked = event.linked_correlation_id()
            cpu_eid = corr_to_event_id.get(linked) if linked > 0 else None
            resolved = _walk_to_ws_marker(cpu_eid) if cpu_eid is not None else None
        if resolved is not None:
            index[eid] = resolved

    return index


_WS_SYNC_RE = re.compile(r"^ws_sync:(\d+):(\d+):(\w+):(.*)$")


@dataclass(frozen=True)
class WsSyncInfo:
    graph_id: int
    sync_id: int
    kind: str  # "device_sync" | "stream_sync" | "wait_event" | "event_sync"
    src_launch_ids: tuple[int, ...]


def _build_cpu_parent_child_maps(
    raw_events: list[Any],
) -> tuple[dict[int, int], dict[int, list[int]], dict[int, Any]]:
    from torch.autograd import DeviceType

    event_by_id = {id(event): event for event in raw_events}
    cpu_sync_events_by_thread: dict[int, list[Any]] = {}
    for event in raw_events:
        is_async = event.is_async() or (
            event.start_thread_id() != event.end_thread_id()
        )
        if event.device_type() == DeviceType.CPU and not is_async:
            cpu_sync_events_by_thread.setdefault(
                event.start_thread_id(), []
            ).append(event)

    child_to_parent: dict[int, int] = {}
    parent_to_children: dict[int, list[int]] = {}
    for thread_events in cpu_sync_events_by_thread.values():
        thread_events.sort(key=lambda e: (e.start_ns(), -e.end_ns()))
        stack: list[Any] = []
        for event in thread_events:
            while stack:
                parent = stack[-1]
                if (
                    event.start_ns() >= parent.end_ns()
                    or event.end_ns() > parent.end_ns()
                ):
                    stack.pop()
                else:
                    child_id = id(event)
                    parent_id = id(parent)
                    child_to_parent[child_id] = parent_id
                    parent_to_children.setdefault(parent_id, []).append(child_id)
                    break
            stack.append(event)

    return child_to_parent, parent_to_children, event_by_id


def _cpu_event_depth(event_id: int, child_to_parent: dict[int, int]) -> int:
    depth = 0
    current = event_id
    while current in child_to_parent:
        depth += 1
        current = child_to_parent[current]
    return depth


def _is_ws_sync_leaf_for_kind(event_name: str, kind: str) -> bool:
    if kind == "device_sync":
        return event_name in _CUDA_DEVICE_SYNC_EVENT_NAMES
    if kind == "stream_sync":
        return event_name in _CUDA_STREAM_SYNC_EVENT_NAMES
    if kind == "wait_event":
        return event_name in _CUDA_STREAM_WAIT_EVENT_NAMES or event_name.startswith(
            "cudaStreamWaitEvent"
        )
    if kind == "event_sync":
        return event_name in _CUDA_EVENT_SYNC_EVENT_NAMES
    return False


def build_ws_sync_index(raw_events: list[Any]) -> dict[int, WsSyncInfo]:
    """Parse ws_sync:* markers and map id(event) -> WsSyncInfo."""
    from torch.autograd import DeviceType

    index: dict[int, WsSyncInfo] = {}
    for event in raw_events:
        if event.device_type() != DeviceType.CPU:
            continue
        m = _WS_SYNC_RE.match(event.name())
        if m is None:
            continue
        ids_str = m.group(4)
        src_ids = tuple(int(x) for x in ids_str.split(",") if x)
        index[id(event)] = WsSyncInfo(
            graph_id=int(m.group(1)),
            sync_id=int(m.group(2)),
            kind=m.group(3),
            src_launch_ids=src_ids,
        )
    return index


def propagate_ws_sync_to_leaf_events(
    raw_events: list[Any],
    ws_sync_marker_index: dict[int, WsSyncInfo],
) -> dict[int, WsSyncInfo]:
    """Attach ws_sync marker metadata to the actual CUDA sync child event.

    ``record_function("ws_sync:*")`` appears as a CPU parent range.  The
    LLaMASim runtime graph keeps CPU leaves, so exact wait edges must point to
    the child Kineto event such as ``cudaDeviceSynchronize`` rather than the
    marker range.
    """
    if not ws_sync_marker_index:
        return {}

    child_to_parent, parent_to_children, event_by_id = _build_cpu_parent_child_maps(
        raw_events
    )
    leaf_index: dict[int, WsSyncInfo] = {}

    for marker_id, sync_info in ws_sync_marker_index.items():
        descendants: list[int] = []
        stack = list(parent_to_children.get(marker_id, ()))
        while stack:
            event_id = stack.pop()
            descendants.append(event_id)
            stack.extend(parent_to_children.get(event_id, ()))

        candidates = [
            event_by_id[event_id]
            for event_id in descendants
            if event_id in event_by_id
            and _is_ws_sync_leaf_for_kind(
                event_by_id[event_id].name(),
                sync_info.kind,
            )
        ]
        if not candidates:
            continue

        leaf = min(
            candidates,
            key=lambda event: (
                -_cpu_event_depth(id(event), child_to_parent),
                event.duration_ns(),
                event.start_ns(),
                event.end_ns(),
                event.name(),
            ),
        )
        leaf_index[id(leaf)] = sync_info

    return leaf_index


def collect_ws_sync_unmatched_marker_samples(
    raw_events: list[Any],
    ws_sync_marker_index: dict[int, WsSyncInfo],
    ws_sync_leaf_index: dict[int, WsSyncInfo],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not ws_sync_marker_index:
        return []

    _child_to_parent, parent_to_children, event_by_id = _build_cpu_parent_child_maps(
        raw_events
    )
    matched_leaf_ids = set(ws_sync_leaf_index)
    samples: list[dict[str, Any]] = []
    for marker_id, sync_info in sorted(
        ws_sync_marker_index.items(),
        key=lambda item: (item[1].graph_id, item[1].sync_id),
    ):
        descendants: list[int] = []
        stack = list(parent_to_children.get(marker_id, ()))
        while stack:
            event_id = stack.pop()
            descendants.append(event_id)
            stack.extend(parent_to_children.get(event_id, ()))
        if any(event_id in matched_leaf_ids for event_id in descendants):
            continue

        child_names: list[str] = []
        for event_id in sorted(
            descendants,
            key=lambda eid: (
                event_by_id[eid].start_ns(),
                event_by_id[eid].end_ns(),
                event_by_id[eid].name(),
            ),
        ):
            name = event_by_id[event_id].name()
            if name not in child_names:
                child_names.append(name)
            if len(child_names) >= 8:
                break
        marker = event_by_id.get(marker_id)
        samples.append({
            "marker": marker.name() if marker is not None else "",
            "kind": sync_info.kind,
            "src_launch_ids": ",".join(
                str(launch_id) for launch_id in sync_info.src_launch_ids
            ),
            "child_cpu_event_names": child_names,
        })
        if len(samples) >= limit:
            break
    return samples


def collect_ws_sync_unresolved_launch_samples(
    gpu_events: list[Any],
    ws_launch_index: dict[int, tuple[int, int]],
    ws_sync_index: dict[int, WsSyncInfo],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    resolved_launches = {
        ws_launch_index[id(event)]
        for event in gpu_events
        if id(event) in ws_launch_index
    }
    samples: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, tuple[int, ...]]] = set()
    for sync_info in sorted(
        ws_sync_index.values(),
        key=lambda info: (info.graph_id, info.sync_id, info.kind),
    ):
        missing = tuple(
            launch_id
            for launch_id in sync_info.src_launch_ids
            if (sync_info.graph_id, launch_id) not in resolved_launches
        )
        if not missing:
            continue
        key = (sync_info.graph_id, sync_info.sync_id, sync_info.kind, missing)
        if key in seen:
            continue
        seen.add(key)
        samples.append({
            "graph_id": sync_info.graph_id,
            "sync_id": sync_info.sync_id,
            "kind": sync_info.kind,
            "missing_launch_ids": ",".join(str(launch_id) for launch_id in missing),
        })
        if len(samples) >= limit:
            break
    return samples


def build_llamasim_exact_wait_edges(
    node_ids: dict[int, str],
    cpu_events: list[Any],
    gpu_events: list[Any],
    ws_launch_index: dict[int, tuple[int, int]],
    ws_sync_index: dict[int, WsSyncInfo],
    runtime_role_by_event_id: dict[int, str],
) -> tuple[list[tuple[str, str, str]], dict[int, str], set[int]]:
    """Build exact wait edges from ws_sync markers and ws_launch tags.

    Returns (edges, updated_roles, handled_cpu_event_ids).
    """
    # Reverse lookup: (graph_id, launch_id) -> list of GPU events
    gpu_by_launch: dict[tuple[int, int], list[Any]] = {}
    for event in gpu_events:
        key = ws_launch_index.get(id(event))
        if key is not None:
            gpu_by_launch.setdefault(key, []).append(event)

    edges: list[tuple[str, str, str]] = []
    updated_roles = dict(runtime_role_by_event_id)
    handled: set[int] = set()
    seen: set[tuple[str, str]] = set()

    for cpu_event in cpu_events:
        sync_info = ws_sync_index.get(id(cpu_event))
        if sync_info is None:
            continue
        if id(cpu_event) not in node_ids:
            continue
        if id(cpu_event) in updated_roles:
            continue

        cpu_node_id = node_ids[id(cpu_event)]
        added = False
        for lid in sync_info.src_launch_ids:
            for gpu_event in gpu_by_launch.get(
                (sync_info.graph_id, lid), []
            ):
                if id(gpu_event) not in node_ids:
                    continue
                gpu_node_id = node_ids[id(gpu_event)]
                pair = (gpu_node_id, cpu_node_id)
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append((gpu_node_id, cpu_node_id, "wait"))
                added = True
        if added:
            updated_roles[id(cpu_event)] = "wait"
            handled.add(id(cpu_event))

    return edges, updated_roles, handled


def write_llamasim_runtime_bundle(
    prof: Any,
    execution_trace_path: Path,
    output_dir: Path,
    *,
    trace_json_path: Path | None = None,
) -> None:
    from torch.autograd import DeviceType
    from torch.autograd.profiler_util import _rewrite_name

    trace_start_ns, raw_events = get_raw_kineto_events(prof)
    ws_launch_index = build_ws_launch_index(raw_events)
    execution_nodes = load_execution_trace_nodes(execution_trace_path)
    matched_execution_node_by_event, match_errors, match_warnings = (
        match_execution_nodes_to_gpu_kernels(raw_events, execution_nodes)
    )
    if match_errors:
        raise RuntimeError(
            "llamasim runtime export has ambiguous GPU kernel matches.\n"
            + "\n".join(match_errors)
        )

    execution_nodes_by_rf_id = index_execution_trace_nodes_by_rf_id(execution_nodes)
    cpu_matched, _ambiguous = match_execution_nodes_to_kineto_runtime(
        raw_events, execution_nodes_by_rf_id,
    )
    matched_execution_node_by_event.update(cpu_matched)

    # Propagate tensor I/O from parent CPU events to GPU kernels that lack
    # a direct execution trace match (e.g. cuBLAS/cutlass kernels whose
    # tensor data lives on the parent aten::mm).
    propagation_stats = _propagate_parent_tensor_io(
        raw_events, matched_execution_node_by_event, execution_nodes_by_rf_id,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    ac2g_endpoints = (
        parse_ac2g_flow_endpoints(trace_json_path)
        if trace_json_path is not None
        else None
    )
    ac2g_ids = ac2g_endpoints.paired_correlation_ids if ac2g_endpoints else set()
    ac2g_matches: Ac2gRawEventMatches | None = None
    exact_submit_cpu_event_ids: set[int] = set()
    if ac2g_endpoints is not None and ac2g_ids:
        ac2g_matches = match_ac2g_endpoints_to_raw_events(
            raw_events,
            ac2g_endpoints,
            trace_start_ns=trace_start_ns,
        )
        if ac2g_matches.errors:
            raise RuntimeError(
                "llamasim runtime export could not exactly resolve Chrome ac2g endpoints.\n"
                + _format_ac2g_match_errors(ac2g_matches.errors)
            )
        exact_submit_cpu_event_ids = {
            id(pair.cpu_event) for pair in ac2g_matches.pairs_by_corr_id.values()
        }

    profiler_sync_excluded_cpu_event_ids = find_profiler_sync_excluded_cpu_event_ids(
        raw_events
    )
    selected_events, selected_cpu_events, selected_gpu_events = (
        select_llamasim_runtime_events(
            raw_events,
            extra_cpu_event_ids=exact_submit_cpu_event_ids,
            exclude_cpu_event_ids=profiler_sync_excluded_cpu_event_ids,
        )
    )

    node_ids = {id(event): f"k{index}" for index, event in enumerate(selected_events)}
    node_indexes = {id(event): index for index, event in enumerate(selected_events)}
    selected_event_ids = set(node_ids.values())

    thread_order_edges = build_llamasim_thread_order_edges(
        node_ids, selected_cpu_events
    )
    stream_order_edges = build_llamasim_stream_order_edges(
        node_ids, selected_gpu_events
    )
    if ac2g_ids:
        submit_edges, runtime_role_by_event_id, submit_edge_attrs = (
            build_llamasim_submit_edges_from_ac2g(
                node_ids,
                selected_cpu_events,
                selected_gpu_events,
                raw_events,
                ac2g_ids,
                ac2g_endpoints=ac2g_endpoints,
                trace_start_ns=trace_start_ns,
                ac2g_matches=ac2g_matches,
                strict=True,
            )
        )
        submit_edge_provenance = "ac2g_endpoint_exact"
    else:
        submit_edges, runtime_role_by_event_id = build_llamasim_submit_edges(
            node_ids,
            selected_cpu_events,
            selected_gpu_events,
            raw_events,
        )
        submit_edge_attrs = {}
        submit_edge_provenance = "correlation_id_heuristic"
    # Exact wait edges from ws_sync markers (takes priority over heuristics).
    # ws_sync record_function events are parents; propagate their metadata to
    # the actual CUDA sync/wait leaf nodes selected in the runtime graph.
    ws_sync_marker_index = build_ws_sync_index(raw_events)
    ws_sync_index = propagate_ws_sync_to_leaf_events(
        raw_events,
        ws_sync_marker_index,
    )
    exact_wait_edges: list[tuple[str, str, str]] = []
    exact_handled_ids: set[int] = set()
    if ws_sync_index:
        exact_wait_edges, runtime_role_by_event_id, exact_handled_ids = (
            build_llamasim_exact_wait_edges(
                node_ids,
                selected_cpu_events,
                selected_gpu_events,
                ws_launch_index,
                ws_sync_index,
                runtime_role_by_event_id,
            )
        )
    ws_sync_unmatched_marker_samples = collect_ws_sync_unmatched_marker_samples(
        raw_events,
        ws_sync_marker_index,
        ws_sync_index,
    )
    ws_sync_unresolved_launch_samples = collect_ws_sync_unresolved_launch_samples(
        selected_gpu_events,
        ws_launch_index,
        ws_sync_index,
    )
    ws_sync_empty_source_count = sum(
        1 for sync_info in ws_sync_index.values() if not sync_info.src_launch_ids
    )
    aten_exact_wait_edges, runtime_role_by_event_id, _aten_handled_ids, aten_wait_attrs = (
        build_llamasim_aten_sync_wait_edges(
            node_ids,
            selected_cpu_events,
            selected_gpu_events,
            raw_events,
            runtime_role_by_event_id,
        )
    )

    # Heuristic wait edges (fallback for events not handled by exact path)
    linked_wait_edges, runtime_role_by_event_id = build_llamasim_wait_edges(
        node_ids,
        selected_cpu_events,
        selected_gpu_events,
        runtime_role_by_event_id,
    )
    sync_wait_edges, runtime_role_by_event_id = build_llamasim_sync_wait_edges(
        node_ids,
        selected_cpu_events,
        selected_gpu_events,
        runtime_role_by_event_id,
    )

    # Provenance: track which edges are exact vs heuristic
    wait_edge_attrs: dict[tuple[str, str], dict[str, str]] = {}
    for src, dst, _ in exact_wait_edges:
        sync_info = None
        for cpu_event in selected_cpu_events:
            if node_ids.get(id(cpu_event)) == dst:
                sync_info = ws_sync_index.get(id(cpu_event))
                break
        attr: dict[str, str] = {
            "wait_exact": "true",
            "wait_source": "wrapper_ws_sync",
        }
        if sync_info is not None:
            attr["wait_kind"] = sync_info.kind
            attr["wait_source_launch_ids"] = ",".join(
                str(lid) for lid in sync_info.src_launch_ids
            )
        wait_edge_attrs[(src, dst)] = attr
    for src, dst, _ in aten_exact_wait_edges:
        attrs = aten_wait_attrs.get((src, dst))
        if attrs is not None:
            wait_edge_attrs[(src, dst)] = attrs
    for src, dst, _ in linked_wait_edges:
        if (src, dst) not in wait_edge_attrs:
            wait_edge_attrs[(src, dst)] = {
                "wait_exact": "false",
                "wait_source": "linked_family_heuristic",
            }
    for src, dst, _ in sync_wait_edges:
        if (src, dst) not in wait_edge_attrs:
            for cpu_event in selected_cpu_events:
                if node_ids.get(id(cpu_event)) == dst:
                    wait_edge_attrs[(src, dst)] = heuristic_sync_wait_attrs(
                        cpu_event.name()
                    )
                    break
            else:
                wait_edge_attrs[(src, dst)] = {
                    "wait_exact": "false",
                    "wait_source": "kineto_sync_frontier",
                }

    # Merge: exact edges first, then heuristic
    exact_wait_edges = exact_wait_edges + aten_exact_wait_edges
    heuristic_wait_edges = linked_wait_edges + sync_wait_edges
    wait_edges = exact_wait_edges + heuristic_wait_edges
    wait_source_counts: dict[str, int] = {}
    wait_kind_counts: dict[str, int] = {}
    for attrs in wait_edge_attrs.values():
        source = attrs.get("wait_source", "")
        if source:
            wait_source_counts[source] = wait_source_counts.get(source, 0) + 1
        kind = attrs.get("wait_kind", "")
        if kind:
            wait_kind_counts[kind] = wait_kind_counts.get(kind, 0) + 1

    tensor_records: dict[str, dict[str, Any]] = {}
    input_edges: list[tuple[str, str]] = []
    output_edges: list[tuple[str, str]] = []
    produced_tensor_keys: set[str] = set()

    def ensure_tensor_record(
        tensor_tuple: tuple[int, int, int, int, int, str],
        *,
        type_name: str | None,
        shape: Any,
    ) -> str:
        tensor_key = execution_tensor_key(list(tensor_tuple))
        if tensor_key is None:
            raise RuntimeError("Failed to serialize execution-trace tensor tuple")
        record = tensor_records.get(tensor_key)
        if record is None:
            tensor_id, storage_id, offset, numel, itemsize, device = tensor_tuple
            shape_tuple = execution_shape_to_tuple(shape)
            tensor_records[tensor_key] = {
                "tensor_tuple": tensor_tuple,
                "type_name": type_name,
                "shape": shape_tuple,
                "num_bytes": numel * itemsize,
                "producers": set(),
                "consumers": set(),
                "kind": "WEIGHT",
                "dtype": execution_type_to_llamasim_dtype(type_name),
                "device": device,
                "tensor_name": llamasim_tensor_name(
                    tensor_tuple,
                    kind="WEIGHT",
                ),
                "tensor_id": tensor_id,
                "storage_id": storage_id,
                "offset": offset,
                "numel": numel,
                "itemsize": itemsize,
            }
            record = tensor_records[tensor_key]
        return tensor_key

    for event in selected_events:
        execution_node = matched_execution_node_by_event.get(id(event))
        if execution_node is None:
            continue
        inputs = execution_node.get("inputs", {})
        for input_index, (type_name, shape, value) in enumerate(
            zip(
                inputs.get("types", []),
                inputs.get("shapes", []),
                inputs.get("values", []),
            )
        ):
            for entry in flatten_execution_tensor_entries(
                value,
                path=str(input_index),
                shape=shape,
                type_name=type_name,
            ):
                tensor_key = ensure_tensor_record(
                    entry["tensor_tuple"],
                    type_name=entry["type_name"],
                    shape=entry["shape"],
                )
                tensor_records[tensor_key]["consumers"].add(node_ids[id(event)])
                input_edges.append((tensor_key, node_ids[id(event)]))

        outputs = execution_node.get("outputs", {})
        for output_index, (type_name, shape, value) in enumerate(
            zip(
                outputs.get("types", []),
                outputs.get("shapes", []),
                outputs.get("values", []),
            )
        ):
            for entry in flatten_execution_tensor_entries(
                value,
                path=str(output_index),
                shape=shape,
                type_name=type_name,
            ):
                tensor_key = ensure_tensor_record(
                    entry["tensor_tuple"],
                    type_name=entry["type_name"],
                    shape=entry["shape"],
                )
                produced_tensor_keys.add(tensor_key)
                tensor_records[tensor_key]["producers"].add(node_ids[id(event)])
                output_edges.append((node_ids[id(event)], tensor_key))

    tensor_keys_in_order = sorted(
        tensor_records.keys(),
        key=lambda key: (
            tensor_records[key]["tensor_id"],
            tensor_records[key]["storage_id"],
            tensor_records[key]["offset"],
            tensor_records[key]["device"],
        ),
    )
    tensor_dot_ids = {tensor_key: f"t{index}" for index, tensor_key in enumerate(tensor_keys_in_order)}

    for tensor_key in tensor_keys_in_order:
        record = tensor_records[tensor_key]
        if tensor_key in produced_tensor_keys:
            record["kind"] = "CONTEXT"
            record["tensor_name"] = llamasim_tensor_name(
                record["tensor_tuple"],
                kind="CONTEXT",
            )

    dot_path = output_dir / "step_0_compute_graph.dot"
    timing_csv_path = output_dir / "ggml_profile_node_records.csv"
    tensor_csv_path = output_dir / "pytorch_runtime_tensors.csv"
    runtime_nodes_csv_path = output_dir / "runtime_nodes.csv"
    runtime_edges_csv_path = output_dir / "runtime_edges.csv"
    manifest_path = output_dir / "manifest.json"

    with dot_path.open("w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        f.write("  newrank = true;\n")
        f.write("  rankdir = TB;\n")

        for event in selected_events:
            node_id = node_ids[id(event)]
            node_index = node_indexes[id(event)]
            name = _rewrite_name(name=event.name(), with_wildcard=False)
            start_ns = event.start_ns() - trace_start_ns
            end_ns = start_ns + event.duration_ns()
            label = dot_escape(
                f"{name} | {node_index} [{start_ns}-{end_ns}] | <x>{event.name()}"
            )
            execution_node = matched_execution_node_by_event.get(id(event))
            kernel_file = (
                execution_trace_attr(execution_node, "kernel_file")
                if execution_node is not None
                else None
            )
            resource_kind = (
                "cpu_thread"
                if event.device_type() == DeviceType.CPU
                else "gpu_stream"
            )
            resource_id = (
                event.start_thread_id()
                if event.device_type() == DeviceType.CPU
                else event.device_resource_id()
            )
            runtime_role = runtime_role_by_event_id.get(
                id(event),
                "cpu_leaf" if event.device_type() == DeviceType.CPU else "gpu_runtime",
            )
            attrs = [
                'style = filled',
                'fillcolor = white',
                'shape = box',
                f'label="{label}"',
                f'device="{getattr(event.device_type(), "name", str(event.device_type()))}:{event.device_index()}"',
                f'resource_kind="{resource_kind}"',
                f'resource_id="{resource_id}"',
                f'runtime_role="{runtime_role}"',
                f'thread="{event.start_thread_id()}"',
                f'stream="{event.device_resource_id()}"',
                f'correlation_id="{event.correlation_id()}"',
                f'linked_correlation_id="{event.linked_correlation_id()}"',
            ]
            rf_id = execution_rf_id(execution_node) if execution_node is not None else None
            if rf_id is not None:
                attrs.append(f'rf_id="{rf_id}"')
            if isinstance(kernel_file, str) and kernel_file:
                attrs.append(f'kernel_file="{dot_escape(Path(kernel_file).name)}"')
            ws = ws_launch_index.get(id(event))
            if ws is not None:
                attrs.append(f'compiled_graph_id="{ws[0]}"')
                attrs.append(f'compiled_launch_id="{ws[1]}"')
            f.write(f'  "{node_id}" [ ' + "; ".join(attrs) + "; ]\n")

        for tensor_index, tensor_key in enumerate(tensor_keys_in_order):
            record = tensor_records[tensor_key]
            dtype = record["dtype"] or "f32"
            shape = record["shape"]
            shape_text = ", ".join(str(dim) for dim in shape) if shape else "1"
            label = dot_escape(
                f'{record["tensor_name"]} ({dtype})|{tensor_index} [{shape_text}] | <x>tensor'
            )
            f.write(
                '  "{node_id}" [ style = filled; fillcolor = pink; shape = record; '
                'label="{label}"; size="{size}"; tensor_kind="{kind}"; tensor_id="{tensor_id}"; '
                'storage_id="{storage_id}"; offset="{offset}"; numel="{numel}"; '
                'itemsize="{itemsize}"; device="{device}"; ];\n'.format(
                    node_id=tensor_dot_ids[tensor_key],
                    label=label,
                    size=record["num_bytes"],
                    kind=record["kind"],
                    tensor_id=record["tensor_id"],
                    storage_id=record["storage_id"],
                    offset=record["offset"],
                    numel=record["numel"],
                    itemsize=record["itemsize"],
                    device=record["device"],
                )
            )

        for tensor_key, node_id in input_edges:
            f.write(f'  "{tensor_dot_ids[tensor_key]}" -> "{node_id}";\n')
        for node_id, tensor_key in output_edges:
            f.write(f'  "{node_id}" -> "{tensor_dot_ids[tensor_key]}";\n')
        for edge_group, color in (
            (thread_order_edges, "gray45"),
            (stream_order_edges, "steelblue4"),
            (submit_edges, "orangered3"),
            (wait_edges, "firebrick3"),
        ):
            for src, dst, edge_kind in edge_group:
                if src in selected_event_ids and dst in selected_event_ids:
                    style = "dotted" if edge_kind.endswith("_order") else "solid"
                    f.write(
                        f'  "{src}" -> "{dst}" [ style = {style}; color = {color}; label = "{edge_kind}"; ];\n'
                    )
        f.write("}\n")

    with timing_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "node_n",
            "node_name",
            "tensor_addr",
            "node_compute_time_ns",
            "node_tensor_size_bytes",
            "start_ns",
            "end_ns",
            "node_id",
            "device_type",
            "device_index",
            "thread_id",
            "stream_id",
            "correlation_id",
            "linked_correlation_id",
            "rf_id",
            "kernel_file",
            "node_kind",
            "resource_kind",
            "resource_id",
            "runtime_role",
            "compiled_graph_id",
            "compiled_launch_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in selected_events:
            execution_node = matched_execution_node_by_event.get(id(event))
            rf_id = execution_rf_id(execution_node) if execution_node is not None else None
            node_kind = (
                "cpu_leaf" if event.device_type() == DeviceType.CPU else "gpu_runtime"
            )
            resource_kind = (
                "cpu_thread"
                if event.device_type() == DeviceType.CPU
                else "gpu_stream"
            )
            resource_id = (
                event.start_thread_id()
                if event.device_type() == DeviceType.CPU
                else event.device_resource_id()
            )
            ws = ws_launch_index.get(id(event))
            writer.writerow(
                {
                    "step": 0,
                    "node_n": node_indexes[id(event)],
                    "node_name": _rewrite_name(
                        name=event.name(), with_wildcard=False
                    ),
                    "tensor_addr": "",
                    "node_compute_time_ns": event.duration_ns(),
                    "node_tensor_size_bytes": 0,
                    "start_ns": event.start_ns() - trace_start_ns,
                    "end_ns": event.end_ns() - trace_start_ns,
                    "node_id": node_ids[id(event)],
                    "device_type": getattr(
                        event.device_type(), "name", str(event.device_type())
                    ),
                    "device_index": event.device_index(),
                    "thread_id": event.start_thread_id(),
                    "stream_id": event.device_resource_id(),
                    "correlation_id": event.correlation_id(),
                    "linked_correlation_id": event.linked_correlation_id(),
                    "rf_id": "" if rf_id is None else rf_id,
                    "kernel_file": (
                        execution_trace_attr(execution_node, "kernel_file") or ""
                        if execution_node is not None
                        else ""
                    ),
                    "node_kind": node_kind,
                    "resource_kind": resource_kind,
                    "resource_id": resource_id,
                    "runtime_role": runtime_role_by_event_id.get(
                        id(event),
                        "cpu_leaf"
                        if event.device_type() == DeviceType.CPU
                        else "gpu_runtime",
                    ),
                    "compiled_graph_id": ws[0] if ws is not None else "",
                    "compiled_launch_id": ws[1] if ws is not None else -1,
                }
            )

    with runtime_nodes_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "node_id",
            "node_n",
            "node_name",
            "op_name",
            "node_kind",
            "resource_kind",
            "resource_id",
            "runtime_role",
            "device_type",
            "device_index",
            "thread_id",
            "stream_id",
            "start_ns",
            "end_ns",
            "duration_ns",
            "correlation_id",
            "linked_correlation_id",
            "rf_id",
            "kernel_file",
            "has_tensor_io",
            "compiled_graph_id",
            "compiled_launch_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in selected_events:
            execution_node = matched_execution_node_by_event.get(id(event))
            rf_id = execution_rf_id(execution_node) if execution_node is not None else None
            ws = ws_launch_index.get(id(event))
            writer.writerow(
                {
                    "step": 0,
                    "node_id": node_ids[id(event)],
                    "node_n": node_indexes[id(event)],
                    "node_name": _rewrite_name(
                        name=event.name(), with_wildcard=False
                    ),
                    "op_name": event.name(),
                    "node_kind": (
                        "cpu_leaf"
                        if event.device_type() == DeviceType.CPU
                        else "gpu_runtime"
                    ),
                    "resource_kind": (
                        "cpu_thread"
                        if event.device_type() == DeviceType.CPU
                        else "gpu_stream"
                    ),
                    "resource_id": (
                        event.start_thread_id()
                        if event.device_type() == DeviceType.CPU
                        else event.device_resource_id()
                    ),
                    "runtime_role": runtime_role_by_event_id.get(
                        id(event),
                        "cpu_leaf"
                        if event.device_type() == DeviceType.CPU
                        else "gpu_runtime",
                    ),
                    "device_type": getattr(
                        event.device_type(), "name", str(event.device_type())
                    ),
                    "device_index": event.device_index(),
                    "thread_id": event.start_thread_id(),
                    "stream_id": event.device_resource_id(),
                    "start_ns": event.start_ns() - trace_start_ns,
                    "end_ns": event.end_ns() - trace_start_ns,
                    "duration_ns": event.duration_ns(),
                    "correlation_id": event.correlation_id(),
                    "linked_correlation_id": event.linked_correlation_id(),
                    "rf_id": "" if rf_id is None else rf_id,
                    "kernel_file": (
                        execution_trace_attr(execution_node, "kernel_file") or ""
                        if execution_node is not None
                        else ""
                    ),
                    "has_tensor_io": int(execution_node is not None),
                    "compiled_graph_id": ws[0] if ws is not None else "",
                    "compiled_launch_id": ws[1] if ws is not None else -1,
                }
            )

    with runtime_edges_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "src_node_id",
            "dst_node_id",
            "edge_kind",
            "submit_exact",
            "wait_kind",
            "wait_exact",
            "wait_source",
            "wait_source_launch_ids",
            "wait_source_correlation_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for src, dst, edge_kind in [
            *thread_order_edges,
            *stream_order_edges,
            *submit_edges,
            *wait_edges,
        ]:
            row: dict[str, Any] = {
                "step": 0,
                "src_node_id": src,
                "dst_node_id": dst,
                "edge_kind": edge_kind,
            }
            s_attrs = submit_edge_attrs.get((src, dst))
            if s_attrs:
                row.update(s_attrs)
            attrs = wait_edge_attrs.get((src, dst))
            if attrs:
                row.update(attrs)
            writer.writerow(row)
        for tensor_key, node_id in input_edges:
            writer.writerow(
                {
                    "step": 0,
                    "src_node_id": tensor_dot_ids[tensor_key],
                    "dst_node_id": node_id,
                    "edge_kind": "data_input",
                }
            )
        for node_id, tensor_key in output_edges:
            writer.writerow(
                {
                    "step": 0,
                    "src_node_id": node_id,
                    "dst_node_id": tensor_dot_ids[tensor_key],
                    "edge_kind": "data_output",
                }
            )

    with tensor_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "tensor_n",
            "tensor_name",
            "tensor_node_id",
            "tensor_kind",
            "tensor_id",
            "storage_id",
            "offset",
            "numel",
            "itemsize",
            "tensor_size_bytes",
            "device",
            "dtype",
            "shape",
            "producer_count",
            "consumer_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tensor_index, tensor_key in enumerate(tensor_keys_in_order):
            record = tensor_records[tensor_key]
            writer.writerow(
                {
                    "step": 0,
                    "tensor_n": tensor_index,
                    "tensor_name": record["tensor_name"],
                    "tensor_node_id": tensor_dot_ids[tensor_key],
                    "tensor_kind": record["kind"],
                    "tensor_id": record["tensor_id"],
                    "storage_id": record["storage_id"],
                    "offset": record["offset"],
                    "numel": record["numel"],
                    "itemsize": record["itemsize"],
                    "tensor_size_bytes": record["num_bytes"],
                    "device": record["device"],
                    "dtype": record["dtype"] or "",
                    "shape": json.dumps(record["shape"], separators=(",", ":")),
                    "producer_count": len(record["producers"]),
                    "consumer_count": len(record["consumers"]),
                }
            )

    gpu_tensor_io_count = sum(
        1
        for event in selected_gpu_events
        if id(event) in matched_execution_node_by_event
    )
    cpu_tensor_io_count = sum(
        1
        for event in selected_cpu_events
        if id(event) in matched_execution_node_by_event
    )
    gpu_node_id_set = {node_ids[id(e)] for e in selected_gpu_events}
    gpu_data_input_edge_count = sum(1 for _, dst in input_edges if dst in gpu_node_id_set)
    gpu_data_output_edge_count = sum(1 for src, _ in output_edges if src in gpu_node_id_set)
    weight_count = sum(
        1 for r in tensor_records.values() if r["kind"] == "WEIGHT"
    )
    context_count = sum(
        1 for r in tensor_records.values() if r["kind"] == "CONTEXT"
    )
    memory_stats: dict[str, int] = {}
    if torch.cuda.is_available():
        memory_stats["vram_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        memory_stats["vram_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        memory_stats["vram_allocated_bytes"] = torch.cuda.memory_allocated()
        memory_stats["vram_reserved_bytes"] = torch.cuda.memory_reserved()
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS.
        memory_stats["dram_peak_rss_bytes"] = rusage.ru_maxrss * 1024
    except (ImportError, AttributeError):
        pass

    manifest = {
        "schema": "pytorch_runtime_v3",
        "compute_node_scope": "cpu_leaf_plus_exact_submit_plus_all_gpu_runtime",
        "tensor_io_scope": "cpu_and_gpu",
        "step_dot_files": ["step_0_compute_graph.dot"],
        "timing_csv": timing_csv_path.name,
        "node_csv": runtime_nodes_csv_path.name,
        "edge_csv": runtime_edges_csv_path.name,
        "tensor_csv": tensor_csv_path.name,
        "trace_start_ns": trace_start_ns,
        "node_count": len(selected_events),
        "gpu_node_count": len(selected_gpu_events),
        "cpu_node_count": len(selected_cpu_events),
        "gpu_tensor_io_count": gpu_tensor_io_count,
        "cpu_tensor_io_count": cpu_tensor_io_count,
        "gpu_no_tensor_io_count": len(selected_gpu_events) - gpu_tensor_io_count,
        "cpu_no_tensor_io_count": len(selected_cpu_events) - cpu_tensor_io_count,
        "tensor_count": len(tensor_keys_in_order),
        "tensor_weight_count": weight_count,
        "tensor_context_count": context_count,
        "data_input_edge_count": len(input_edges),
        "data_output_edge_count": len(output_edges),
        "gpu_data_input_edge_count": gpu_data_input_edge_count,
        "gpu_data_output_edge_count": gpu_data_output_edge_count,
        "cpu_data_input_edge_count": len(input_edges) - gpu_data_input_edge_count,
        "cpu_data_output_edge_count": len(output_edges) - gpu_data_output_edge_count,
        "submit_edge_count": len(submit_edges),
        "exact_submit_edge_count": sum(
            1 for a in submit_edge_attrs.values() if a.get("submit_exact") == "true"
        ),
        "heuristic_submit_edge_count": len(submit_edges) - sum(
            1 for a in submit_edge_attrs.values() if a.get("submit_exact") == "true"
        ),
        "submit_edge_provenance": submit_edge_provenance,
        "ac2g_paired_count": len(ac2g_ids),
        "ac2g_endpoint_matched_count": (
            len(ac2g_matches.pairs_by_corr_id) if ac2g_matches is not None else 0
        ),
        "ac2g_endpoint_match_error_count": (
            len(ac2g_matches.errors) if ac2g_matches is not None else 0
        ),
        "wait_edge_count": len(wait_edges),
        "exact_wait_edge_count": len(exact_wait_edges),
        "wrapper_exact_wait_edge_count": (
            len(exact_wait_edges) - len(aten_exact_wait_edges)
        ),
        "aten_exact_wait_edge_count": len(aten_exact_wait_edges),
        "heuristic_wait_edge_count": len(heuristic_wait_edges),
        "wait_source_counts": wait_source_counts,
        "wait_kind_counts": wait_kind_counts,
        "excluded_wait_source_counts": (
            {"profiler_sync_excluded": len(profiler_sync_excluded_cpu_event_ids)}
            if profiler_sync_excluded_cpu_event_ids
            else {}
        ),
        "profiler_sync_excluded_count": len(profiler_sync_excluded_cpu_event_ids),
        "ws_sync_marker_count": len(ws_sync_marker_index),
        "ws_sync_leaf_count": len(ws_sync_index),
        "ws_sync_unmatched_marker_samples": ws_sync_unmatched_marker_samples,
        "ws_sync_unresolved_launch_samples": ws_sync_unresolved_launch_samples,
        "ws_sync_empty_source_count": ws_sync_empty_source_count,
        "thread_order_edge_count": len(thread_order_edges),
        "stream_order_edge_count": len(stream_order_edges),
        "match_warning_count": len(match_warnings),
        "propagation_matched": propagation_stats["propagated"],
        "propagation_no_linked_rf_id": propagation_stats["no_linked_rf_id"],
        "propagation_no_execution_node": propagation_stats["no_execution_node"],
        "propagation_ambiguous_rf_id": propagation_stats["ambiguous_rf_id"],
        "ws_launch_resolved_count": sum(
            1 for e in selected_events if id(e) in ws_launch_index
        ),
        "ws_launch_unresolved_count": sum(
            1 for e in selected_events if id(e) not in ws_launch_index
        ),
        **memory_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _format_count_fraction(exact: Any, total: Any) -> str:
    exact_int = int(exact or 0)
    total_int = int(total or 0)
    if total_int <= 0:
        return f"{exact_int}/{total_int}"
    return f"{exact_int}/{total_int} ({(exact_int / total_int) * 100.0:.2f}%)"


def _format_count_map(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "{}"
    return ", ".join(
        f"{key}={value[key]}"
        for key in sorted(value)
    )


def print_llamasim_runtime_summary(bundle_dir: Path) -> None:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        print("llamasim_runtime_summary: manifest missing")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wait_source_counts = manifest.get("wait_source_counts")
    wait_kind_counts = manifest.get("wait_kind_counts")
    excluded_wait_source_counts = manifest.get("excluded_wait_source_counts") or {}
    if not wait_source_counts or not wait_kind_counts:
        edge_csv_path = bundle_dir / manifest.get("edge_csv", "runtime_edges.csv")
        if edge_csv_path.exists():
            wait_source_counts = {}
            wait_kind_counts = {}
            with edge_csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("edge_kind") != "wait":
                        continue
                    source = row.get("wait_source", "")
                    if source:
                        wait_source_counts[source] = (
                            wait_source_counts.get(source, 0) + 1
                        )
                    kind = row.get("wait_kind", "")
                    if kind:
                        wait_kind_counts[kind] = wait_kind_counts.get(kind, 0) + 1
    submit_total = manifest.get("submit_edge_count", 0)
    submit_exact = manifest.get("exact_submit_edge_count", 0)
    wait_total = manifest.get("wait_edge_count", 0)
    wait_exact = manifest.get("exact_wait_edge_count", 0)

    print("llamasim_runtime_summary:")
    print(
        "  nodes: "
        f"total={manifest.get('node_count', 0)} "
        f"gpu={manifest.get('gpu_node_count', 0)} "
        f"cpu={manifest.get('cpu_node_count', 0)}"
    )
    print(
        "  submit_edges: "
        f"exact={_format_count_fraction(submit_exact, submit_total)} "
        f"heuristic={manifest.get('heuristic_submit_edge_count', 0)} "
        f"provenance={manifest.get('submit_edge_provenance', '')}"
    )
    print(
        "  ac2g: "
        f"paired={manifest.get('ac2g_paired_count', 0)} "
        f"matched={manifest.get('ac2g_endpoint_matched_count', 0)} "
        f"errors={manifest.get('ac2g_endpoint_match_error_count', 0)}"
    )
    print(
        "  wait_edges: "
        f"exact={_format_count_fraction(wait_exact, wait_total)} "
        f"heuristic={manifest.get('heuristic_wait_edge_count', 0)}"
    )
    print(
        "  ws_sync: "
        f"markers={manifest.get('ws_sync_marker_count', 0)} "
        f"leaves={manifest.get('ws_sync_leaf_count', 0)}"
    )
    print(
        "  wait_sources: "
        f"{_format_count_map(wait_source_counts)}"
    )
    if excluded_wait_source_counts:
        print(
            "  excluded_wait_sources: "
            f"{_format_count_map(excluded_wait_source_counts)}"
        )
    print(
        "  wait_kinds: "
        f"{_format_count_map(wait_kind_counts)}"
    )
    if int(wait_total or 0) > 0 and int(manifest.get("ws_sync_marker_count", 0) or 0) == 0:
        if int(wait_exact or 0) == 0:
            print(
                "  wait_exact_note: wrapper emitted no ws_sync evidence; "
                "wait edges in this run are heuristic fallback edges"
            )
        else:
            print(
                "  wait_exact_note: wrapper emitted no ws_sync evidence; "
                "exact waits came from non-wrapper provenance"
            )
    elif (
        int(manifest.get("ws_sync_marker_count", 0) or 0) > 0
        and int(manifest.get("ws_sync_leaf_count", 0) or 0) == 0
    ):
        print(
            "  wait_exact_note: ws_sync markers were emitted but not matched "
            "to CUDA sync leaf events"
        )
        samples = manifest.get("ws_sync_unmatched_marker_samples") or []
        if samples:
            print(
                "  ws_sync_unmatched_marker_samples: "
                f"{json.dumps(samples[:3], sort_keys=True)}"
            )
    elif (
        int(manifest.get("ws_sync_leaf_count", 0) or 0) > 0
        and int(wait_exact or 0) == 0
    ):
        print(
            "  wait_exact_note: ws_sync leaves were found but no exact wait "
            "edges were emitted"
        )
        unresolved = manifest.get("ws_sync_unresolved_launch_samples") or []
        if unresolved:
            print(
                "  ws_sync_unresolved_launch_samples: "
                f"{json.dumps(unresolved[:3], sort_keys=True)}"
            )
        elif int(manifest.get("ws_sync_empty_source_count", 0) or 0) > 0:
            print(
                "  ws_sync_empty_source_count: "
                f"{manifest.get('ws_sync_empty_source_count', 0)}"
            )


def write_hybrid_profile_dot(prof: Any, dot_path: Path, graph_name: str) -> None:
    from torch.autograd import DeviceType
    from torch.autograd.profiler_util import _rewrite_name

    trace_start_ns, raw_events = get_raw_kineto_events(prof)
    node_ids, cpu_parent_edges, linked_edges, stream_edges = build_kineto_runtime_graph(
        raw_events
    )
    csv_matches = build_kineto_csv_matches(prof, raw_events, trace_start_ns)
    memory_profile = prof._memory_profile()
    flow_nodes = tuple(
        node
        for node in memory_profile._data_flow_graph.flow_nodes
        if node._event.name not in {"[memory]", "[OutOfMemory]"}
    )

    raw_events_by_cid: dict[int, list[Any]] = {}
    for event in raw_events:
        raw_events_by_cid.setdefault(event.correlation_id(), []).append(event)

    matched_runtime_node_by_flow_node: dict[int, str] = {}
    for flow_node in flow_nodes:
        flow_event = flow_node._event
        candidates = raw_events_by_cid.get(flow_event.correlation_id, [])
        if not candidates:
            continue

        same_name = [event for event in candidates if event.name() == flow_event.name]
        if same_name:
            candidates = same_name

        exact_start = [
            event for event in candidates if event.start_ns() == flow_event.start_time_ns
        ]
        if exact_start:
            candidates = exact_start

        best_match = min(
            candidates,
            key=lambda event: abs(event.start_ns() - flow_event.start_time_ns),
        )
        matched_runtime_node_by_flow_node[id(flow_node)] = node_ids[id(best_match)]

    tensor_dot_ids: dict[tuple[Any, int], str] = {}
    tensor_labels: dict[tuple[Any, int], str] = {}
    input_edges: list[tuple[str, str, str]] = []
    output_edges: list[tuple[str, str, str, str]] = []
    intermediate_edges: list[tuple[str, str]] = []

    def tensor_sort_key(item: tuple[Any, Any]) -> tuple[int, int]:
        key, value = item
        if isinstance(value, tuple):
            return (key.id, value[1])
        return (key.id, value)

    def ensure_tensor_node(
        key: Any,
        version: int,
        *,
        kind: str | None = None,
    ) -> str:
        pair = (key, version)
        tensor_dot_id = tensor_dot_ids.get(pair)
        if tensor_dot_id is not None:
            return tensor_dot_id

        tensor_dot_id = f"t{len(tensor_dot_ids)}"
        tensor_dot_ids[pair] = tensor_dot_id
        category = memory_profile._categories.get(key, version)
        num_bytes = None
        try:
            num_bytes = memory_profile._size_map[key]
        except KeyError:
            num_bytes = None
        tensor_labels[pair] = memory_tensor_label(
            key,
            version,
            None if category is None else category.name,
            num_bytes,
            kind=kind,
        )
        return tensor_dot_id

    for flow_node in flow_nodes:
        runtime_node_id = matched_runtime_node_by_flow_node.get(id(flow_node))
        if runtime_node_id is None:
            continue

        for key, (mutated, version) in sorted(
            flow_node.inputs.items(), key=tensor_sort_key
        ):
            tensor_dot_id = ensure_tensor_node(
                key,
                version,
                kind="input" if not mutated else "input/output",
            )
            input_edges.append(
                (tensor_dot_id, runtime_node_id, "inout" if mutated else "in")
            )

        for key, version in sorted(flow_node.outputs.items(), key=tensor_sort_key):
            tensor_dot_id = ensure_tensor_node(key, version, kind="output")
            output_edges.append((runtime_node_id, tensor_dot_id, "out", "black"))

        for key in sorted(flow_node.intermediates):
            tensor_dot_id = ensure_tensor_node(key, 0, kind="temporary")
            intermediate_edges.append((runtime_node_id, tensor_dot_id))

    with dot_path.open("w") as f:
        f.write(f'digraph "{dot_escape(graph_name)}" {{\n')
        f.write("  newrank=true;\n")
        f.write("  rankdir=TB;\n")

        for event in raw_events:
            name = event.name()
            device_type = event.device_type()
            device_name = getattr(device_type, "name", str(device_type))
            rel_start_us = (event.start_ns() - trace_start_ns) / 1000.0
            duration_us = event.duration_ns() / 1000.0
            label_lines = [name, f"cid={event.correlation_id()}"]
            label_lines.append(f"trace={_rewrite_name(name=name, with_wildcard=False)}")
            if event.linked_correlation_id() > 0:
                label_lines.append(f"linked={event.linked_correlation_id()}")
            label_lines.append(f"device={device_name}:{event.device_index()}")
            if device_type == DeviceType.CPU:
                label_lines.append(f"thread={event.start_thread_id()}")
            else:
                label_lines.append(f"stream={event.device_resource_id()}")
            label_lines.append(f"start_us={rel_start_us:.3f}")
            label_lines.append(f"dur_us={duration_us:.3f}")
            matches = csv_matches.get(id(event), [])
            if len(matches) == 1:
                csv_event_index, csv_event = matches[0]
                label_lines.append(f"csv_index={csv_event_index}")
                label_lines.append(f"csv_id={getattr(csv_event, 'id', None)}")
            elif len(matches) > 1:
                label_lines.append(f"csv_matches={len(matches)}")
            if event.is_user_annotation():
                label_lines.append("user_annotation=1")
            label = "\\n".join(dot_escape(line) for line in label_lines)
            f.write(
                f'  "{node_ids[id(event)]}" [shape=box, style=filled, '
                f'fillcolor="{kineto_node_fillcolor(device_type, name)}", '
                f'label="{label}"];\n'
            )

        for pair, tensor_dot_id in tensor_dot_ids.items():
            label = tensor_labels[pair]
            f.write(
                f'  "{tensor_dot_id}" [shape=record, style=filled, '
                f'fillcolor="pink", label="{label}"];\n'
            )

        for src, dst in cpu_parent_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dashed, color="gray50", label="cpu"];\n'
            )
        for src, dst, label in linked_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="orangered3", penwidth=1.4, label="{label}"];\n'
            )
        for src, dst in stream_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dotted, color="steelblue4", label="stream"];\n'
            )
        for src, dst, label in input_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="black", label="{label}"];\n'
            )
        for src, dst, label, color in output_edges:
            f.write(
                f'  "{src}" -> "{dst}" [color="{color}", label="{label}"];\n'
            )
        for src, dst in intermediate_edges:
            f.write(
                f'  "{src}" -> "{dst}" [style=dashed, color="gray50", label="tmp"];\n'
            )
        f.write("}\n")


def triton_install_hint() -> str:
    install_script = Path(__file__).resolve().with_name("install_triton_wheel.sh")
    return (
        "Install Triton into the active Python environment and rerun. "
        f"Active Python: {sys.executable}. "
        f"Repo helper: {install_script}"
    )


def validate_fusion_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")
    if device.type != "cuda":
        return

    from torch.utils._triton import has_triton, has_triton_package

    if not has_triton_package():
        raise RuntimeError(
            "--fusion=inductor on CUDA requires Triton, but no `triton` package is "
            f"installed in this runtime. {triton_install_hint()}"
        )
    if not has_triton():
        raise RuntimeError(
            "--fusion=inductor on CUDA found a `triton` package, but it is not usable "
            f"with this runtime or device. {triton_install_hint()}"
        )


def decode_latents_to_pil(pipe: Any, latents: torch.Tensor) -> list[Any]:
    needs_upcasting = (
        pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    )

    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype and torch.backends.mps.is_available():
        pipe.vae = pipe.vae.to(latents.dtype)

    has_latents_mean = (
        hasattr(pipe.vae.config, "latents_mean")
        and pipe.vae.config.latents_mean is not None
    )
    has_latents_std = (
        hasattr(pipe.vae.config, "latents_std")
        and pipe.vae.config.latents_std is not None
    )
    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1)
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1)
        latents_std = latents_std.to(latents.device, latents.dtype)
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
    if pipe.watermark is not None:
        image = pipe.watermark.apply_watermark(image)
    return pipe.image_processor.postprocess(image, output_type="pil")


def run_pipeline(pipe: Any, args: argparse.Namespace) -> Any:
    return pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        height=args.height,
        width=args.width,
        output_type="latent",
    )


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"sample_step=0 phase={args.phase} component={args.component} "
        f"device={args.device} fusion={args.fusion}"
    )


def cuda_unavailable_error(device: torch.device) -> str:
    imported_from = Path(torch.__file__).resolve()
    lines = [
        f"CUDA profiling requested for {device}, but torch.cuda.is_available() is False.",
        f"Imported torch from: {imported_from}",
        f"torch.version.cuda = {torch.version.cuda!r}",
        f"torch.backends.cuda.is_built() = {torch.backends.cuda.is_built()}",
    ]
    if not torch.backends.cuda.is_built():
        lines.append(
            "This Python process is using a CPU-only PyTorch build, so the GPU profiler script cannot run."
        )
        lines.append(
            "System CUDA tools such as nvcc and nvidia-smi only show that the machine has CUDA and a visible GPU."
        )
        lines.append(
            "Use a CUDA-enabled PyTorch build for this source tree, or rerun with a runtime that imports one."
        )
    else:
        lines.append(
            "This PyTorch build has CUDA support, but no CUDA device is available to this process."
        )
    return "\n".join(lines)


def validate_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(cuda_unavailable_error(device))
    return device


def main(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_fusion: str = "none",
    default_dot_level: str = "none",
    default_llamasim_output_dirname: str | None = None,
    default_profile_memory: bool = False,
    default_record_shapes: bool = False,
    default_with_stack: bool = False,
) -> None:
    args = parse_args(
        default_device=default_device,
        default_dtype=default_dtype,
        default_output_prefix=default_output_prefix,
        default_fusion=default_fusion,
        default_dot_level=default_dot_level,
        default_profile_memory=default_profile_memory,
        default_record_shapes=default_record_shapes,
        default_with_stack=default_with_stack,
    )
    validate_dot_args(args)
    device = validate_device(args)
    validate_fusion_runtime(args, device)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    warmup_runs = (
        args.warmup_runs if args.warmup_runs is not None else int(args.fusion != "none")
    )
    output_paths = resolve_output_paths(
        args,
        default_output_prefix,
        default_llamasim_output_dirname=default_llamasim_output_dirname,
    )

    pipe = load_pipeline(args.model, torch_dtype, device)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    maybe_compile(pipe, args, output_paths)

    captured_unet_inputs: dict[str, Any] = {}
    capture_handle = None
    if output_paths.fx_dot_path is not None:
        captured_unet_inputs, capture_handle = capture_example_inputs(pipe.unet)

    for _ in range(warmup_runs):
        warmup_output = run_pipeline(pipe, args)
        synchronize_device(device)
        del warmup_output

    scope_args = metadata_for_scope(args)
    execution_trace_observer = None
    if output_paths.execution_trace_path is not None:
        execution_trace_observer = torch.profiler.ExecutionTraceObserver()
        execution_trace_observer.register_callback(str(output_paths.execution_trace_path))
    with torch.profiler.profile(
        activities=profile_activities(device),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        execution_trace_observer=execution_trace_observer,
    ) as prof:
        with torch.autograd.profiler.record_function("sdxl_turbo_run", scope_args):
            output = run_pipeline(pipe, args)
        profiler_synchronize_device(device)

    if capture_handle is not None:
        capture_handle.remove()
    if execution_trace_observer is not None:
        execution_trace_observer.unregister_callback()

    metadata_json = None
    for event in prof.events():
        if event.name == "sdxl_turbo_run":
            metadata_json = event.metadata_json
            break

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    with torch.no_grad():
        images = decode_latents_to_pil(pipe, output.images)
    images[0].save(output_paths.image_path)

    if output_paths.fx_dot_path is not None:
        if "args" not in captured_unet_inputs or "kwargs" not in captured_unet_inputs:
            raise RuntimeError("Failed to capture UNet example inputs for FX DOT export")
        fx_graph_module = export_unet_graph_module(
            underlying_unet_module(pipe.unet),
            captured_unet_inputs["args"],
            captured_unet_inputs["kwargs"],
        )
        write_fx_graph_dot(fx_graph_module, output_paths.fx_dot_path, output_paths.fx_dot_path.stem)

    if output_paths.execution_dot_path is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError("Execution DOT export requested without an execution trace path")
        write_execution_trace_dot(
            output_paths.execution_trace_path,
            output_paths.execution_dot_path,
            output_paths.execution_dot_path.stem,
        )
    if output_paths.memory_dot_path is not None:
        write_memory_profile_dot(
            prof,
            output_paths.memory_dot_path,
            output_paths.memory_dot_path.stem,
        )
    if output_paths.kineto_dot_path is not None:
        write_kineto_profile_dot(
            prof,
            output_paths.kineto_dot_path,
            output_paths.kineto_dot_path.stem,
            output_paths.kineto_map_path,
        )
    if output_paths.hybrid_dot_path is not None:
        write_hybrid_profile_dot(
            prof,
            output_paths.hybrid_dot_path,
            output_paths.hybrid_dot_path.stem,
        )
    if output_paths.runtime_io_dot_path is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError("Runtime-I/O DOT export requested without an execution trace path")
        write_runtime_io_profile_dot(
            prof,
            output_paths.execution_trace_path,
            output_paths.runtime_io_dot_path,
            output_paths.runtime_io_dot_path.stem,
        )
    if output_paths.llamasim_output_dir is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError(
                "llamasim-runtime export requested without an execution trace path"
            )
        write_llamasim_runtime_bundle(
            prof,
            output_paths.execution_trace_path,
            output_paths.llamasim_output_dir,
            trace_json_path=output_paths.trace_path,
        )

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("device:", device)
    print("dtype:", args.dtype)
    print("fusion:", args.fusion)
    print("dot_level:", args.dot_level)
    print("profile_memory:", args.profile_memory)
    print("record_shapes:", args.record_shapes)
    print("with_stack:", args.with_stack)
    print("warmup_runs:", warmup_runs)
    print("trace_path:", output_paths.trace_path)
    if hasattr(prof, "export_csv"):
        print("trace_csv_path:", output_paths.csv_path)
    print("image_path:", output_paths.image_path)
    if output_paths.fx_dot_path is not None:
        print("fx_dot_path:", output_paths.fx_dot_path)
    if output_paths.execution_trace_path is not None:
        print("execution_trace_path:", output_paths.execution_trace_path)
    if output_paths.execution_dot_path is not None:
        print("execution_dot_path:", output_paths.execution_dot_path)
    if output_paths.memory_dot_path is not None:
        print("memory_dot_path:", output_paths.memory_dot_path)
    if output_paths.kineto_dot_path is not None:
        print("kineto_dot_path:", output_paths.kineto_dot_path)
    if output_paths.kineto_map_path is not None:
        print("kineto_map_path:", output_paths.kineto_map_path)
    if output_paths.hybrid_dot_path is not None:
        print("hybrid_dot_path:", output_paths.hybrid_dot_path)
    if output_paths.runtime_io_dot_path is not None:
        print("runtime_io_dot_path:", output_paths.runtime_io_dot_path)
    if output_paths.llamasim_output_dir is not None:
        print("llamasim_output_dir:", output_paths.llamasim_output_dir)
        print_llamasim_runtime_summary(output_paths.llamasim_output_dir)
    print("latent_shape:", tuple(output.images.shape))
    print("metadata_json:", metadata_json)
    if metadata_json:
        print("metadata_parsed:", json.dumps(json.loads(metadata_json), sort_keys=True))
