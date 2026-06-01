#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Sequence

import torch

import profile_llama_common as llama_common
import run_llama_accelerate_cpu_offload as offload_common


DEFAULT_PRELOAD_MODULE_CLASSES = "LlamaDecoderLayer"

elapsed_seconds = offload_common.elapsed_seconds
ensure_accelerate_available = offload_common.ensure_accelerate_available
free_cpu_offload_hooks = offload_common.free_cpu_offload_hooks
load_pipeline = offload_common.load_pipeline
maybe_compile = offload_common.maybe_compile
output_stem_base = offload_common.output_stem_base
resolve_text_path = offload_common.resolve_text_path
run_inference = offload_common.run_inference
run_warmups = offload_common.run_warmups
synchronize_device = offload_common.synchronize_device
validate_fusion_runtime = offload_common.validate_fusion_runtime
validate_run_device = offload_common.validate_run_device


def _set_fixed_offload_mode(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if action.dest != "offload_mode":
            continue
        action.choices = ("decoder-layer",)
        action.default = "decoder-layer"
        action.help = (
            "Fixed mode for this script: Accelerate cpu_offload with "
            "preload_module_classes targeting Llama decoder layers."
        )
        return
    raise RuntimeError("Could not find --offload-mode parser action.")


def add_decoder_layer_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_dtype: str,
    default_output_prefix: str | None,
    default_warmup_runs: int,
    default_max_new_tokens: int,
) -> None:
    offload_common.add_common_args(
        parser,
        default_model=default_model,
        default_dtype=default_dtype,
        default_offload_mode="decoder-layer",
        default_output_prefix=default_output_prefix,
        default_warmup_runs=default_warmup_runs,
        default_max_new_tokens=default_max_new_tokens,
    )
    _set_fixed_offload_mode(parser)
    parser.add_argument(
        "--preload-module-classes",
        default=DEFAULT_PRELOAD_MODULE_CLASSES,
        help=(
            "Comma-separated module class names passed to "
            "accelerate.cpu_offload(..., preload_module_classes=...)."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Llama with HF Accelerate CPU offload scoped at decoder "
            "layers via preload_module_classes."
        )
    )
    add_decoder_layer_args(
        parser,
        default_model="/data/llamasim/models/Llama-3.1-8b",
        default_dtype="bfloat16",
        default_output_prefix=None,
        default_warmup_runs=1,
        default_max_new_tokens=15,
    )
    return parser.parse_args()


def preload_module_classes(args: argparse.Namespace) -> list[str]:
    classes = [
        item.strip()
        for item in str(args.preload_module_classes).split(",")
        if item.strip()
    ]
    if not classes:
        raise RuntimeError("--preload-module-classes must not be empty.")
    return classes


def apply_decoder_layer_cpu_offload(
    pipe: llama_common.LlamaPipeline,
    device: torch.device,
    *,
    classes: Sequence[str],
) -> None:
    from accelerate import cpu_offload

    offload_common._prepare_direct_offload(pipe)
    cpu_offload(
        pipe.model,
        execution_device=device,
        offload_buffers=True,
        preload_module_classes=list(classes),
    )


def main() -> None:
    args = parse_args()
    device = validate_run_device(args.device)
    validate_fusion_runtime(args, device)
    accelerate_version = ensure_accelerate_available()
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
    torch_dtype = llama_common.DTYPE_BY_NAME[args.dtype]
    text_path = resolve_text_path(args)
    classes = preload_module_classes(args)

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    pipe = load_pipeline(
        args.model,
        torch_dtype,
        device,
        offload_mode=args.offload_mode,
    )
    maybe_compile(pipe, args)
    apply_decoder_layer_cpu_offload(pipe, device, classes=classes)
    synchronize_device(device)
    load_seconds = elapsed_seconds(load_start)

    warmup_seconds = run_warmups(pipe, args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output, inference_seconds = run_inference(pipe, args, device)
    decoded = llama_common.decode_outputs(pipe, output)
    free_cpu_offload_hooks(pipe)

    save_start = time.perf_counter()
    text_path.write_text("\n---\n".join(decoded), encoding="utf-8")
    save_seconds = elapsed_seconds(save_start)
    e2e_seconds = elapsed_seconds(total_start)

    alloc_mb = None
    peak_mb = None
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

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
    print("max_new_tokens:", args.max_new_tokens)
    print("batch_size:", args.batch_size)
    print("warmup_runs:", args.warmup_runs)
    print("text_path:", text_path)
    print("sequence_shape:", tuple(output.sequences.shape))
    print("generated_text_preview:", decoded[0][:200])
    if alloc_mb is not None and peak_mb is not None:
        print(f"GPU memory: {alloc_mb:.1f} MB allocated, {peak_mb:.1f} MB peak")
    print(f"load_seconds: {load_seconds:.6f}")
    print(f"warmup_seconds: {warmup_seconds:.6f}")
    print(f"inference_seconds: {inference_seconds:.6f}")
    print(f"save_seconds: {save_seconds:.6f}")
    print(f"e2e_seconds: {e2e_seconds:.6f}")


if __name__ == "__main__":
    main()
