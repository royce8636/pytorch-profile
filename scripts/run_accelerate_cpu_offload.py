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

import profile_qwen_image_common as qwen_common  # noqa: E402
import profile_sdxl_turbo_common as sdxl_common  # noqa: E402


OFFLOAD_MODES = ("none", "model", "sequential", "module", "module-hook")
_DIRECT_HF_ACCELERATE_HOOKS_ATTR = "_direct_hf_accelerate_cpu_offload_hooks"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_dtype: str,
    default_offload_mode: str,
    default_warmup_runs: int,
) -> None:
    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to the model directory.",
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
        default="cuda:0",
        help="Accelerator device used for execution, for example cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(sdxl_common.DTYPE_BY_NAME.keys()),
        default=default_dtype,
        help="Torch dtype used to load the pipeline.",
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default="none",
        help=(
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the hot "
            "model path with torch.compile before any offload hooks are installed."
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
        "--offload-mode",
        choices=OFFLOAD_MODES,
        default=default_offload_mode,
        help=(
            "Execution mode. 'none' keeps the full pipeline on the accelerator without "
            "Accelerate offload hooks, 'model' uses Diffusers enable_model_cpu_offload, "
            "'sequential' uses Diffusers enable_sequential_cpu_offload, 'module' applies "
            "HF Accelerate cpu_offload directly to each pipeline module, and "
            "'module-hook' applies HF Accelerate cpu_offload_with_hook directly to the "
            "pipeline module sequence."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=default_warmup_runs,
        help="Warmup iterations run before timing.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated PNG.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path for the generated PNG. Defaults to /tmp or --output-dir.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SDXL-Turbo or Qwen-Image with HF Accelerate CPU offload so the "
            "pipeline can exceed GPU VRAM."
        )
    )
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    sdxl_parser = subparsers.add_parser(
        "sdxl-turbo",
        help="Run the local SDXL-Turbo pipeline with CPU offload.",
    )
    add_common_args(
        sdxl_parser,
        default_model="/data/llamasim/models/sdxl-turbo",
        default_dtype="float16",
        default_offload_mode="model",
        default_warmup_runs=0,
    )
    sdxl_parser.set_defaults(
        steps=8,
        height=512,
        width=512,
        prompt="a cute cat and a cute dog in a park",
    )

    qwen_parser = subparsers.add_parser(
        "qwen-image",
        help="Run the local Qwen-Image pipeline with CPU offload.",
    )
    add_common_args(
        qwen_parser,
        default_model="/data/llamasim/models/qwen-image-2512",
        default_dtype="bfloat16",
        default_offload_mode="sequential",
        default_warmup_runs=0,
    )
    qwen_parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="Negative prompt passed to the pipeline.",
    )
    qwen_parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="Traditional classifier-free guidance scale used by Qwen-Image.",
    )
    qwen_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Optional guidance-distilled scale.",
    )
    qwen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the generation RNG. Set to a negative value to disable.",
    )
    qwen_parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum prompt sequence length passed to the pipeline.",
    )
    return parser.parse_args()


def output_stem_base(args: argparse.Namespace) -> str:
    pipeline = args.pipeline.replace("-", "_")
    offload_mode = args.offload_mode.replace("-", "_")
    stem = f"{pipeline}_{offload_mode}_cpu_offload"
    if args.fusion != "none":
        stem = f"{stem}_{args.fusion}"
    return f"{stem}_steps{args.steps}"


def resolve_image_path(args: argparse.Namespace) -> Path:
    stem = output_stem_base(args)
    if args.image is not None:
        image_path = Path(args.image)
    elif args.output_dir is not None:
        image_path = Path(args.output_dir) / f"{stem}_output.png"
    else:
        image_path = Path("/tmp") / f"{stem}_output.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def validate_run_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cpu":
        raise RuntimeError(
            "Accelerate CPU offload still requires an accelerator device such as cuda:0. "
            "Use the existing CPU-only scripts if you want to run entirely on CPU."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(sdxl_common.cuda_unavailable_error(device))
    return device


def ensure_accelerate_available() -> str:
    try:
        import accelerate
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This runner requires `accelerate` because it uses HF Accelerate CPU "
            "offload hooks."
        ) from exc
    version = getattr(accelerate, "__version__", None)
    return version if isinstance(version, str) else "unknown"


def maybe_compile(pipe: Any, args: argparse.Namespace) -> None:
    if args.fusion == "none":
        return
    if args.pipeline == "sdxl-turbo":
        sdxl_common.maybe_compile(pipe, args)
        return
    qwen_common.maybe_compile(pipe, args)


def pipeline_module_components(pipe: Any) -> dict[str, torch.nn.Module]:
    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        modules = {
            name: module
            for name, module in components.items()
            if isinstance(module, torch.nn.Module)
        }
        if modules:
            return modules

    modules = {}
    for name in ("text_encoder", "text_encoder_2", "unet", "transformer", "vae"):
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            modules[name] = module
    return modules


def _pipeline_to(pipe: Any, device: str | torch.device) -> None:
    if not hasattr(pipe, "to"):
        return
    try:
        pipe.to(device, silence_dtype_warnings=True)
    except TypeError:
        pipe.to(device)


def _prepare_direct_hf_accelerate_offload(pipe: Any, device: torch.device) -> None:
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    setattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, [])
    _pipeline_to(pipe, "cpu")
    setattr(pipe, "_offload_device", device)
    if device.index is not None:
        setattr(pipe, "_offload_gpu_id", device.index)


def _excluded_cpu_offload_components(pipe: Any) -> set[str]:
    excluded = getattr(pipe, "_exclude_from_cpu_offload", ())
    return set(excluded or ())


def _offload_buffers_for_module(module: torch.nn.Module) -> bool:
    return len(getattr(module, "_parameters", {})) > 0


def apply_accelerate_module_cpu_offload(
    pipe: Any,
    device: torch.device,
) -> None:
    from accelerate import cpu_offload

    _prepare_direct_hf_accelerate_offload(pipe, device)
    components = pipeline_module_components(pipe)
    if not components:
        raise RuntimeError(
            "Could not find any torch.nn.Module pipeline components for "
            "--offload-mode module."
        )

    excluded = _excluded_cpu_offload_components(pipe)
    for name, module in components.items():
        if name in excluded:
            module.to(device)
            continue
        cpu_offload(
            module,
            execution_device=device,
            offload_buffers=_offload_buffers_for_module(module),
        )


def _ordered_hook_offload_components(
    pipe: Any,
    components: dict[str, torch.nn.Module],
) -> list[tuple[str, torch.nn.Module]]:
    remaining = dict(components)
    ordered: list[tuple[str, torch.nn.Module]] = []
    sequence = getattr(pipe, "model_cpu_offload_seq", None)
    if isinstance(sequence, str):
        for name in sequence.split("->"):
            module = remaining.pop(name, None)
            if module is not None:
                ordered.append((name, module))
    ordered.extend(remaining.items())
    return ordered


def apply_accelerate_module_hook_cpu_offload(
    pipe: Any,
    device: torch.device,
) -> None:
    from accelerate import cpu_offload_with_hook

    _prepare_direct_hf_accelerate_offload(pipe, device)
    components = pipeline_module_components(pipe)
    if not components:
        raise RuntimeError(
            "Could not find any torch.nn.Module pipeline components for "
            "--offload-mode module-hook."
        )

    excluded = _excluded_cpu_offload_components(pipe)
    hooks = []
    prev_hook = None
    for name, module in _ordered_hook_offload_components(pipe, components):
        if name in excluded:
            module.to(device)
            continue
        _, prev_hook = cpu_offload_with_hook(
            module,
            execution_device=device,
            prev_module_hook=prev_hook,
        )
        hooks.append(prev_hook)
    setattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, hooks)


def apply_cpu_offload(pipe: Any, args: argparse.Namespace, device: torch.device) -> None:
    if args.offload_mode == "none":
        pipe.to(device)
        return
    if args.offload_mode == "model":
        pipe.enable_model_cpu_offload(device=device)
        return
    if args.offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        return
    if args.offload_mode == "module":
        apply_accelerate_module_cpu_offload(pipe, device)
        return
    apply_accelerate_module_hook_cpu_offload(pipe, device)


def free_cpu_offload_hooks(pipe: Any) -> None:
    hooks = getattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, ())
    for hook in reversed(hooks or ()):
        offload = getattr(hook, "offload", None)
        if offload is not None:
            offload()
    setattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, [])
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()


def format_qwen_model_offload_oom_error(
    *,
    model: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    exc: BaseException,
) -> str:
    return "\n".join(
        [
            "Qwen-Image hit CUDA OOM while using Accelerate model CPU offload.",
            f"Model path: {model}",
            f"device = {device}",
            f"torch_dtype = {torch_dtype}",
            f"Original error: {exc}",
            "In `--offload-mode model`, Diffusers still moves the entire transformer onto the accelerator for each forward pass.",
            "That transformer does not fit on this GPU, even though CPU RAM is available.",
            "Re-run with `--offload-mode sequential` so Accelerate offloads submodules instead of the whole transformer.",
        ]
    )


def load_sdxl_pipeline(model: str, torch_dtype: torch.dtype) -> Any:
    try:
        from diffusers import StableDiffusionXLPipeline
    except ModuleNotFoundError as exc:
        if sdxl_common.is_missing_diffusers_error(exc):
            raise RuntimeError(
                sdxl_common.format_missing_diffusers_runtime_error(
                    component="the Accelerate CPU offload runner",
                    exc=exc,
                )
            ) from exc
        raise

    return StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )


def load_qwen_pipeline(model: str, torch_dtype: torch.dtype) -> Any:
    architecture, model_type = qwen_common._read_qwen_model_requirements(model)
    qwen_common.install_qwen25_transformers_compat()
    try:
        from diffusers import (
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
            QwenImagePipeline,
            QwenImageTransformer2DModel,
        )
        import transformers
    except ModuleNotFoundError as exc:
        if sdxl_common.is_missing_diffusers_error(exc):
            raise RuntimeError(
                sdxl_common.format_missing_diffusers_runtime_error(
                    component="the Accelerate CPU offload runner",
                    exc=exc,
                )
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            qwen_common.format_qwen_runtime_error(
                model=model,
                exc=exc,
                diffusers_version=qwen_common._module_version("diffusers"),
                transformers_version=qwen_common._module_version("transformers"),
                architecture=architecture,
                model_type=model_type,
            )
        ) from exc

    text_encoder_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    tokenizer_cls = getattr(transformers, "Qwen2Tokenizer", None)
    missing_symbols = [
        name
        for name, value in (
            ("Qwen2_5_VLForConditionalGeneration", text_encoder_cls),
            ("Qwen2Tokenizer", tokenizer_cls),
        )
        if value is None
    ]
    if missing_symbols:
        raise RuntimeError(
            "transformers is missing required Qwen symbols after installing "
            f"compatibility hooks: {', '.join(missing_symbols)}"
        )

    model_path = Path(model)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        model_path / "text_encoder",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    tokenizer = tokenizer_cls.from_pretrained(model_path / "tokenizer")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path / "transformer",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path / "vae",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    missing = [
        attr
        for attr in ("transformer", "vae", "image_processor", "_unpack_latents", "vae_scale_factor")
        if not hasattr(pipe, attr)
    ]
    if missing:
        raise RuntimeError(
            f"Loaded pipeline {type(pipe).__name__} from {model} is missing "
            f"required Qwen attributes: {', '.join(missing)}"
        )
    return pipe


def load_pipeline(args: argparse.Namespace, torch_dtype: torch.dtype) -> Any:
    if args.pipeline == "sdxl-turbo":
        return load_sdxl_pipeline(args.model, torch_dtype)
    return load_qwen_pipeline(args.model, torch_dtype)


def run_qwen_pipeline_with_oom_guidance(
    pipe: Any,
    args: argparse.Namespace,
    generator: torch.Generator | None,
) -> Any:
    try:
        return qwen_common.run_pipeline(pipe, args, generator)
    except torch.OutOfMemoryError as exc:
        if args.offload_mode == "model":
            raise RuntimeError(
                format_qwen_model_offload_oom_error(
                    model=args.model,
                    device=torch.device(args.device),
                    torch_dtype=sdxl_common.DTYPE_BY_NAME[args.dtype],
                    exc=exc,
                )
            ) from exc
        raise


def run_warmups(
    pipe: Any,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator | None,
) -> float:
    warmup_start = time.perf_counter()
    for _ in range(args.warmup_runs):
        if args.pipeline == "sdxl-turbo":
            warmup_output = sdxl_common.run_pipeline(pipe, args)
        else:
            warmup_output = run_qwen_pipeline_with_oom_guidance(
                pipe,
                args,
                generator,
            )
        sdxl_common.synchronize_device(device)
        del warmup_output
    return elapsed_seconds(warmup_start)


def run_inference(
    pipe: Any,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[Any, float]:
    inference_start = time.perf_counter()
    if args.pipeline == "sdxl-turbo":
        output = sdxl_common.run_pipeline(pipe, args)
    else:
        output = run_qwen_pipeline_with_oom_guidance(
            pipe,
            args,
            generator,
        )
    sdxl_common.synchronize_device(device)
    return output, elapsed_seconds(inference_start)


def decode_images(pipe: Any, args: argparse.Namespace, output: Any, device: torch.device) -> tuple[list[Any], float]:
    decode_start = time.perf_counter()
    with torch.no_grad():
        if args.pipeline == "sdxl-turbo":
            images = sdxl_common.decode_latents_to_pil(pipe, output.images)
        else:
            images = qwen_common.decode_latents_to_pil(
                pipe,
                output.images,
                height=args.height,
                width=args.width,
            )
    sdxl_common.synchronize_device(device)
    return images, elapsed_seconds(decode_start)


def main() -> None:
    args = parse_args()
    device = validate_run_device(args.device)
    sdxl_common.validate_fusion_runtime(args, device)
    accelerate_version = (
        ensure_accelerate_available()
        if args.offload_mode != "none"
        else "not_used"
    )
    torch_dtype = sdxl_common.DTYPE_BY_NAME[args.dtype]
    image_path = resolve_image_path(args)

    total_start = time.perf_counter()

    load_start = time.perf_counter()
    pipe = load_pipeline(args, torch_dtype)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    maybe_compile(pipe, args)
    apply_cpu_offload(pipe, args, device)
    sdxl_common.synchronize_device(device)
    load_seconds = elapsed_seconds(load_start)

    generator = (
        qwen_common.build_generator(args, device)
        if args.pipeline == "qwen-image"
        else None
    )

    warmup_seconds = run_warmups(pipe, args, device, generator)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output, inference_seconds = run_inference(pipe, args, device, generator)
    images, decode_seconds = decode_images(pipe, args, output, device)

    free_cpu_offload_hooks(pipe)

    save_start = time.perf_counter()
    images[0].save(image_path)
    save_seconds = elapsed_seconds(save_start)

    e2e_seconds = elapsed_seconds(total_start)
    alloc_mb = None
    peak_mb = None
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print("pipeline:", args.pipeline)
    print("requested_device:", device)
    print("execution_device:", getattr(pipe, "_execution_device", "unknown"))
    print("offload_mode:", args.offload_mode)
    print("fusion:", args.fusion)
    print("accelerate_version:", accelerate_version)
    print("dtype:", args.dtype)
    print("steps:", args.steps)
    print("warmup_runs:", args.warmup_runs)
    print("image_path:", image_path)
    print("latent_shape:", tuple(output.images.shape))
    if alloc_mb is not None and peak_mb is not None:
        print(f"GPU memory: {alloc_mb:.1f} MB allocated, {peak_mb:.1f} MB peak")
    print(f"load_seconds: {load_seconds:.6f}")
    print(f"warmup_seconds: {warmup_seconds:.6f}")
    print(f"inference_seconds: {inference_seconds:.6f}")
    print(f"decode_seconds: {decode_seconds:.6f}")
    print(f"save_seconds: {save_seconds:.6f}")
    print(f"e2e_seconds: {e2e_seconds:.6f}")


if __name__ == "__main__":
    main()
