#!/usr/bin/env python3
"""Llama 3B / 8B profiler driver (mirrors profile_sdxl_turbo_common).

Reuses the generic DOT/execution-trace/llamasim-runtime bundle machinery from
profile_sdxl_turbo_common and overrides only the model-specific pieces:

* load_pipeline   — HuggingFace `AutoModelForCausalLM` + `AutoTokenizer`
* maybe_compile   — single `torch.compile(model)` (no UNet/VAE/text-encoder split)
* run_pipeline    — `model.generate(input_ids, max_new_tokens=..., ...)`
* decode_outputs  — tokenizer.batch_decode
* main()          — replaces SDXL's pipe.unet-based FX capture with model-root capture
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_sdxl_turbo_common import (  # noqa: E402
    DTYPE_BY_NAME,
    OutputPaths,
    capture_example_inputs,
    configure_llamasim_inductor_markers,
    dot_level_enabled,
    export_unet_graph_module,
    output_stem,
    print_llamasim_runtime_summary,
    profile_activities,
    profiler_synchronize_device,
    synchronize_device,
    triton_install_hint,
    underlying_unet_module,
    validate_dot_args,
    validate_device,
    validate_fusion_runtime,
    write_execution_trace_dot,
    write_fx_graph_dot,
    write_hybrid_profile_dot,
    write_kineto_profile_dot,
    write_llamasim_runtime_bundle,
    write_memory_profile_dot,
    write_runtime_io_profile_dot,
)

import time  # noqa: E402


@dataclass
class LlamaPipeline:
    model: torch.nn.Module
    tokenizer: Any
    device: torch.device


@dataclass
class LlamaOutput:
    sequences: torch.Tensor
    input_ids: torch.Tensor


def format_missing_transformers_error(exc: ModuleNotFoundError) -> str:
    lines = [
        "Failed to import `transformers` required by the Llama scripts.",
        f"sys.executable = {sys.executable!r}",
        f"torch.__file__ = {Path(torch.__file__).resolve()}",
        f"Original error: {exc}",
    ]
    venv_python = Path("/tmp/ptvenv/bin/python")
    if venv_python.exists():
        rerun_command = shlex.join([str(venv_python), *sys.argv])
        lines.extend(
            [
                "This interpreter does not have `transformers` installed.",
                "Re-run with the profiler/runtime environment:",
                f"PYTHONNOUSERSITE=1 {rerun_command}",
            ]
        )
    else:
        lines.append(
            "Use a Python environment that has `transformers` installed."
        )
    return "\n".join(lines)


def parse_args(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_model: str | None,
    default_component: str,
    default_fusion: str,
    default_dot_level: str,
    default_profile_memory: bool,
    default_record_shapes: bool,
    default_with_stack: bool,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama profiler driver for the patched PyTorch build."
    )
    parser.add_argument(
        "--model",
        required=default_model is None,
        default=default_model,
        help=(
            "HuggingFace hub id or local path to the Llama checkpoint "
            "(e.g. meta-llama/Llama-3.2-3B)."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox",
        help="Prompt fed to the causal LM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of tokens to generate per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompt replicas per forward pass.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default is greedy generation).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --do-sample).",
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
        help="Torch dtype used to load the model.",
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default=default_fusion,
        help=(
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the model "
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
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=True,
        help=(
            "Compile with dynamic shapes (default). For causal LMs the seq_length "
            "grows each generation step; dynamic=False triggers a fresh compile "
            "per token, making the timed run dominated by compilation."
        ),
    )
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="Disable dynamic-shape compilation (per-step recompiles).",
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
            "text output, and optional DOT outputs into this directory."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=default_output_prefix,
        help="Prefix used for trace, text, and optional DOT output filenames.",
    )
    parser.add_argument(
        "--trace",
        default=None,
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
        "--text-output",
        default=None,
        help=(
            "Path for the generated text. Defaults to the trace directory or "
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
        help="Optional DOT graph outputs (same semantics as profile_sdxl_turbo_common).",
    )
    parser.add_argument("--fx-dot", default=None)
    parser.add_argument("--execution-dot", default=None)
    parser.add_argument("--execution-trace", default=None)
    parser.add_argument("--memory-dot", default=None)
    parser.add_argument("--kineto-dot", default=None)
    parser.add_argument("--kineto-map", default=None)
    parser.add_argument("--hybrid-dot", default=None)
    parser.add_argument("--runtime-io-dot", default=None)
    parser.add_argument("--llamasim-output-dir", default=None)
    parser.add_argument(
        "--component",
        default=default_component,
        help="Component label stored in profiler metadata.",
    )
    parser.add_argument(
        "--phase",
        default="text_generation",
        help="Phase label stored in profiler metadata.",
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


def resolve_output_paths(
    args: argparse.Namespace,
    default_output_prefix: str,
    default_llamasim_output_dirname: str | None = None,
) -> OutputPaths:
    """Like profile_sdxl_turbo_common.resolve_output_paths but writes text, not PNG.

    Reuses the OutputPaths dataclass; `image_path` is repurposed to store the
    generated text path.
    """
    output_prefix = getattr(args, "output_prefix", default_output_prefix)
    stem = output_stem(output_prefix, args.fusion)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        trace_path = output_dir / f"{stem}_trace.json"
        csv_path = output_dir / f"{stem}_trace.csv"
        text_path = (
            Path(args.text_output)
            if args.text_output is not None
            else output_dir / f"{stem}_output.txt"
        )
        fx_dot_path = Path(args.fx_dot) if args.fx_dot else output_dir / f"{stem}_fx.dot"
        execution_dot_path = (
            Path(args.execution_dot)
            if args.execution_dot
            else output_dir / f"{stem}_execution.dot"
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace
            else output_dir / f"{stem}_execution_trace.json"
        )
        memory_dot_path = (
            Path(args.memory_dot)
            if args.memory_dot
            else output_dir / f"{stem}_memory.dot"
        )
        kineto_dot_path = (
            Path(args.kineto_dot)
            if args.kineto_dot
            else output_dir / f"{stem}_kineto.dot"
        )
        kineto_map_path = (
            Path(args.kineto_map)
            if args.kineto_map
            else output_dir / f"{stem}_kineto_map.csv"
        )
        hybrid_dot_path = (
            Path(args.hybrid_dot)
            if args.hybrid_dot
            else output_dir / f"{stem}_hybrid.dot"
        )
        runtime_io_dot_path = (
            Path(args.runtime_io_dot)
            if args.runtime_io_dot
            else output_dir / f"{stem}_runtime_io.dot"
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir
            else (
                output_dir / default_llamasim_output_dirname
                if default_llamasim_output_dirname is not None
                else output_dir / f"{stem}_llamasim_runtime"
            )
        )
    else:
        if args.trace is not None:
            trace_path = Path(args.trace)
        elif output_prefix == default_output_prefix:
            trace_path = Path("/tmp") / f"{default_output_prefix}_trace.json"
        else:
            trace_path = Path("/tmp") / f"{stem}_trace.json"
        csv_path = (
            Path(args.trace_csv)
            if args.trace_csv
            else trace_path.with_suffix(".csv")
        )
        text_path = (
            Path(args.text_output)
            if args.text_output
            else trace_path.with_name(f"{stem}_output.txt")
        )
        fx_dot_path = (
            Path(args.fx_dot)
            if args.fx_dot
            else trace_path.with_name(f"{stem}_fx.dot")
        )
        execution_dot_path = (
            Path(args.execution_dot)
            if args.execution_dot
            else trace_path.with_name(f"{stem}_execution.dot")
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace
            else trace_path.with_name(f"{stem}_execution_trace.json")
        )
        memory_dot_path = (
            Path(args.memory_dot)
            if args.memory_dot
            else trace_path.with_name(f"{stem}_memory.dot")
        )
        kineto_dot_path = (
            Path(args.kineto_dot)
            if args.kineto_dot
            else trace_path.with_name(f"{stem}_kineto.dot")
        )
        kineto_map_path = (
            Path(args.kineto_map)
            if args.kineto_map
            else trace_path.with_name(f"{stem}_kineto_map.csv")
        )
        hybrid_dot_path = (
            Path(args.hybrid_dot)
            if args.hybrid_dot
            else trace_path.with_name(f"{stem}_hybrid.dot")
        )
        runtime_io_dot_path = (
            Path(args.runtime_io_dot)
            if args.runtime_io_dot
            else trace_path.with_name(f"{stem}_runtime_io.dot")
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir
            else trace_path.with_name(f"{stem}_llamasim_runtime")
        )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

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
    for p in (
        fx_dot_path,
        execution_dot_path,
        execution_trace_path,
        memory_dot_path,
        kineto_dot_path,
        kineto_map_path,
        hybrid_dot_path,
        runtime_io_dot_path,
    ):
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
    if llamasim_output_dir is not None:
        llamasim_output_dir.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        trace_path=trace_path,
        csv_path=csv_path,
        image_path=text_path,
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


def load_pipeline(
    model: str, torch_dtype: torch.dtype, device: torch.device
) -> LlamaPipeline:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(format_missing_transformers_error(exc)) from exc

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch_dtype
    ).to(device)
    hf_model.eval()
    return LlamaPipeline(model=hf_model, tokenizer=tokenizer, device=device)


def maybe_compile(
    pipe: LlamaPipeline,
    args: argparse.Namespace,
    output_paths: OutputPaths | None = None,
) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")

    if output_paths is not None and output_paths.llamasim_output_dir is not None:
        configure_llamasim_inductor_markers(output_paths.llamasim_output_dir)

    pipe.model = torch.compile(
        pipe.model,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
    )


def encode_prompt(
    pipe: LlamaPipeline, args: argparse.Namespace
) -> torch.Tensor:
    prompts = [args.prompt] * args.batch_size
    encoded = pipe.tokenizer(prompts, return_tensors="pt", padding=True)
    return encoded["input_ids"].to(pipe.device)


def greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Minimal manual generation loop.

    We avoid `HFModel.generate()` because it imports `torch.distributed.fsdp`,
    which fails on PyTorch builds compiled without distributed support. This
    keeps the script usable against the source tree's editable install.
    """
    sequences = input_ids
    for _ in range(max_new_tokens):
        outputs = model(sequences)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_logits = logits[:, -1, :]
        if do_sample:
            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        sequences = torch.cat([sequences, next_token], dim=-1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
    return sequences


def run_pipeline(pipe: LlamaPipeline, args: argparse.Namespace) -> LlamaOutput:
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
    return LlamaOutput(sequences=sequences, input_ids=input_ids)


def decode_outputs(pipe: LlamaPipeline, output: LlamaOutput) -> list[str]:
    return pipe.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"sample_step=0 phase={args.phase} component={args.component} "
        f"device={args.device} fusion={args.fusion}"
    )


def underlying_model(module: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps the module in an OptimizedModule whose unwrapped
    # module is accessible via `_orig_mod`. Same convention as SDXL's
    # underlying_unet_module helper.
    return underlying_unet_module(module)


def main(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_model: str | None,
    default_component: str = "llama",
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
        default_model=default_model,
        default_component=default_component,
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
    maybe_compile(pipe, args, output_paths)

    captured_inputs: dict[str, Any] = {}
    capture_handle = None
    if output_paths.fx_dot_path is not None:
        captured_inputs, capture_handle = capture_example_inputs(pipe.model)

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
        with torch.autograd.profiler.record_function("llama_run", scope_args):
            output = run_pipeline(pipe, args)
        profiler_synchronize_device(device)

    if capture_handle is not None:
        capture_handle.remove()
    if execution_trace_observer is not None:
        execution_trace_observer.unregister_callback()

    metadata_json = None
    for event in prof.events():
        if event.name == "llama_run":
            metadata_json = event.metadata_json
            break

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    decoded = decode_outputs(pipe, output)
    output_paths.image_path.write_text(
        "\n---\n".join(decoded), encoding="utf-8"
    )

    if output_paths.fx_dot_path is not None:
        if "args" not in captured_inputs or "kwargs" not in captured_inputs:
            raise RuntimeError("Failed to capture example inputs for FX DOT export")
        fx_graph_module = export_unet_graph_module(
            underlying_model(pipe.model),
            captured_inputs["args"],
            captured_inputs["kwargs"],
        )
        write_fx_graph_dot(
            fx_graph_module,
            output_paths.fx_dot_path,
            output_paths.fx_dot_path.stem,
        )

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
    if args.fusion != "none":
        from torch._dynamo.utils import counters as _dynamo_counters
        unique_graphs = _dynamo_counters.get("stats", {}).get("unique_graphs", 0)
        print(f"inductor_unique_graphs: {unique_graphs}")
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
    print("text_output_path:", output_paths.image_path)
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
    print("sequence_shape:", tuple(output.sequences.shape))
    print("generated_text_preview:", decoded[0][:200])
    print("metadata_json:", metadata_json)
    if metadata_json:
        print("metadata_parsed:", json.dumps(json.loads(metadata_json), sort_keys=True))


# Re-export triton_install_hint for any callers (matches profile_sdxl_turbo_common).
__all__ = [
    "DTYPE_BY_NAME",
    "LlamaOutput",
    "LlamaPipeline",
    "decode_outputs",
    "encode_prompt",
    "greedy_generate",
    "load_pipeline",
    "main",
    "maybe_compile",
    "metadata_for_scope",
    "resolve_output_paths",
    "run_pipeline",
    "triton_install_hint",
    "underlying_model",
]
