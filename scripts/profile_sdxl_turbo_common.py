#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import torch


DTYPE_BY_NAME = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


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


def parse_args(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
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
        default="none",
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
            "both",
            "all",
        ),
        default="none",
        help=(
            "Optional DOT graph outputs. 'fx' exports the logical UNet FX graph, "
            "'execution' exports a DOT converted from ExecutionTraceObserver output, "
            "'memory' exports a tensor-flow DOT from the profiler memory graph, "
            "'kineto' exports the post-fusion runtime event graph from Kineto, "
            "'hybrid' overlays matched tensor-flow onto Kineto runtime nodes, "
            "'runtime-io' overlays exact ExecutionTraceObserver I/O onto Kineto "
            "runtime nodes, 'both' writes fx+execution, and 'all' writes "
            "fx+execution+memory+kineto+hybrid+runtime-io."
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
            "Requires --dot-level execution, runtime-io, both, or all."
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
    if args.fx_dot is not None and not fx_enabled:
        raise RuntimeError("--fx-dot requires --dot-level fx, both, or all")
    if args.execution_dot is not None and not execution_enabled:
        raise RuntimeError(
            "--execution-dot requires --dot-level execution, both, or all"
        )
    if args.execution_trace is not None and not (
        execution_enabled or runtime_io_enabled
    ):
        raise RuntimeError(
            "--execution-trace requires --dot-level execution, runtime-io, both, or all"
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
    args: argparse.Namespace, default_output_prefix: str
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

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not dot_level_enabled(args.dot_level, "fx"):
        fx_dot_path = None
    if not (
        dot_level_enabled(args.dot_level, "execution")
        or dot_level_enabled(args.dot_level, "runtime-io")
    ):
        execution_dot_path = None
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
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def load_pipeline(model: str, torch_dtype: torch.dtype, device: torch.device) -> Any:
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)
    return pipe


def maybe_compile(pipe: Any, args: argparse.Namespace) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")
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
    default_profile_memory: bool = False,
    default_record_shapes: bool = False,
    default_with_stack: bool = False,
) -> None:
    args = parse_args(
        default_device=default_device,
        default_dtype=default_dtype,
        default_output_prefix=default_output_prefix,
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
        args, default_output_prefix
    )

    pipe = load_pipeline(args.model, torch_dtype, device)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    maybe_compile(pipe, args)

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
            synchronize_device(device)

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
    print("latent_shape:", tuple(output.images.shape))
    print("metadata_json:", metadata_json)
    if metadata_json:
        print("metadata_parsed:", json.dumps(json.loads(metadata_json), sort_keys=True))
