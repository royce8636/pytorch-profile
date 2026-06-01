#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_llama_common as llama_common  # noqa: E402
from run_llama_3b_gpu import (  # noqa: E402
    cuda_unavailable_error,
    validate_fusion_runtime,
)


_DIRECT_HF_ACCELERATE_HOOKS_ATTR = "_direct_hf_accelerate_cpu_offload_hooks"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_dtype: str,
    default_offload_mode: str,
    default_output_prefix: str | None,
    default_warmup_runs: int,
    default_max_new_tokens: int,
) -> None:
    parser.add_argument(
        "--model",
        default=default_model,
        help="HuggingFace hub id or local path to the Llama checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox",
        help="Prompt fed to the causal LM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed passed to torch.manual_seed. Set negative to skip seeding.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help="Number of tokens to generate per run.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enable-eos-stop", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=tuple(llama_common.DTYPE_BY_NAME.keys()),
        default=default_dtype,
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default="none",
        help="Execution mode; inductor compiles the HF model before offload hooks.",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.add_argument(
        "--offload-mode",
        choices=("model",),
        default=default_offload_mode,
        help="'model' applies HF Accelerate cpu_offload to the whole model.",
    )
    parser.add_argument("--warmup-runs", type=int, default=default_warmup_runs)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated text file.",
    )
    parser.add_argument(
        "--output-prefix",
        default=default_output_prefix,
        help="Optional prefix used for generated output filenames.",
    )
    parser.add_argument(
        "--text-output",
        default=None,
        help="Path for the generated text. Defaults to /tmp or --output-dir.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Llama with direct HF Accelerate CPU offload modes."
    )
    add_common_args(
        parser,
        default_model="/data/llamasim/models/Llama-3.1-8b",
        default_dtype="bfloat16",
        default_offload_mode="model",
        default_output_prefix=None,
        default_warmup_runs=1,
        default_max_new_tokens=15,
    )
    return parser.parse_args()


def validate_run_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cpu":
        raise RuntimeError(
            "Llama CPU offload requires an accelerator execution device such as cuda:0."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(cuda_unavailable_error(device))
    return device


def ensure_accelerate_available() -> str:
    try:
        import accelerate
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This runner requires `accelerate` for direct Llama CPU offload modes."
        ) from exc
    version = getattr(accelerate, "__version__", None)
    return version if isinstance(version, str) else "unknown"


def output_stem_base(args: argparse.Namespace) -> str:
    output_prefix = getattr(args, "output_prefix", None)
    if output_prefix is None:
        offload_mode = args.offload_mode.replace("-", "_")
        stem = f"llama8b_{offload_mode}_cpu_offload"
        if args.fusion != "none":
            stem = f"{stem}_{args.fusion}"
    else:
        stem = llama_common.output_stem(output_prefix, args.fusion)
    return f"{stem}_tokens{args.max_new_tokens}"


def resolve_text_path(args: argparse.Namespace) -> Path:
    stem = output_stem_base(args)
    if args.text_output is not None:
        path = Path(args.text_output)
    elif args.output_dir is not None:
        path = Path(args.output_dir) / f"{stem}_output.txt"
    else:
        path = Path("/tmp") / f"{stem}_output.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(llama_common.format_missing_transformers_error(exc)) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_pipeline(
    model: str,
    torch_dtype: torch.dtype,
    execution_device: torch.device,
) -> llama_common.LlamaPipeline:
    AutoModelForCausalLM, AutoTokenizer = _load_transformers_classes()
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_device = torch.device("cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
    ).to(load_device)
    hf_model.eval()
    return llama_common.LlamaPipeline(
        model=hf_model,
        tokenizer=tokenizer,
        device=execution_device,
    )


def _prepare_direct_offload(pipe: llama_common.LlamaPipeline) -> None:
    setattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, [])
    pipe.model.to("cpu")


def apply_model_cpu_offload(
    pipe: llama_common.LlamaPipeline,
    device: torch.device,
) -> None:
    from accelerate import cpu_offload

    _prepare_direct_offload(pipe)
    cpu_offload(pipe.model, execution_device=device, offload_buffers=False)


def apply_cpu_offload(
    pipe: llama_common.LlamaPipeline,
    device: torch.device,
) -> None:
    apply_model_cpu_offload(pipe, device)


def free_cpu_offload_hooks(pipe: llama_common.LlamaPipeline) -> None:
    hooks = getattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, ())
    for hook in reversed(hooks or ()):
        offload = getattr(hook, "offload", None)
        if offload is not None:
            offload()
    setattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, [])


def maybe_compile(
    pipe: llama_common.LlamaPipeline,
    args: argparse.Namespace,
) -> None:
    llama_common.maybe_compile(pipe, args)


def run_warmups(
    pipe: llama_common.LlamaPipeline,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    start = time.perf_counter()
    for _ in range(args.warmup_runs):
        warmup_output = llama_common.run_pipeline(pipe, args)
        synchronize_device(device)
        del warmup_output
    return elapsed_seconds(start)


def run_inference(
    pipe: llama_common.LlamaPipeline,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[llama_common.LlamaOutput, float]:
    start = time.perf_counter()
    output = llama_common.run_pipeline(pipe, args)
    synchronize_device(device)
    return output, elapsed_seconds(start)


def main() -> None:
    args = parse_args()
    device = validate_run_device(args.device)
    validate_fusion_runtime(args, device)
    accelerate_version = ensure_accelerate_available()
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
    torch_dtype = llama_common.DTYPE_BY_NAME[args.dtype]
    text_path = resolve_text_path(args)

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    pipe = load_pipeline(
        args.model,
        torch_dtype,
        device,
    )
    maybe_compile(pipe, args)
    apply_cpu_offload(pipe, device)
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
    print("offload_device:", "cpu" if args.offload_mode != "none" else "not_used")
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
