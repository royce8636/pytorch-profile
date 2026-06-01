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

import profile_sdxl_turbo_common as sdxl_common  # noqa: E402


OFFLOAD_MODES = ("none", "model", "sequential", "module", "module-hook", "group")
_DIRECT_HF_ACCELERATE_HOOKS_ATTR = "_direct_hf_accelerate_cpu_offload_hooks"
DEFAULT_SDXL_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_SDXL_PIPELINE_VARIANT = "fp16"
_GROUP_OFFLOAD_COMPONENTS = ("unet", "text_encoder", "text_encoder_2", "vae")


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
        default="A cute dog and a cat in a park",
        help="Prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed passed to torch.manual_seed. Set negative to skip seeding.",
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
        "--vae-model",
        default=DEFAULT_SDXL_VAE_MODEL,
        help=(
            "VAE checkpoint loaded into the SDXL pipeline. Use 'none' to keep the "
            "VAE bundled with --model."
        ),
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_SDXL_PIPELINE_VARIANT,
        help=(
            "Diffusers checkpoint variant loaded for the SDXL pipeline. Use 'none' "
            "or an empty string to omit the variant argument."
        ),
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
            "pipeline module sequence, and 'group' uses Diffusers group offload with "
            "CPU as the offload target."
        ),
    )
    parser.add_argument(
        "--group-offload-type",
        choices=("block_level", "leaf_level"),
        default="block_level",
        help="Diffusers group-offload granularity used by --offload-mode group.",
    )
    parser.add_argument(
        "--group-offload-num-blocks",
        type=int,
        default=1,
        help=(
            "Number of blocks per group for --offload-mode group when "
            "--group-offload-type=block_level."
        ),
    )
    parser.add_argument(
        "--group-offload-use-stream",
        dest="group_offload_use_stream",
        action="store_true",
        default=True,
        help="Use a transfer stream for Diffusers group offload.",
    )
    parser.add_argument(
        "--no-group-offload-use-stream",
        dest="group_offload_use_stream",
        action="store_false",
        help="Disable the transfer stream for Diffusers group offload.",
    )
    parser.add_argument(
        "--group-offload-non-blocking",
        action="store_true",
        help="Use non-blocking transfers for Diffusers group offload.",
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
            "Run SDXL-Turbo with HF Accelerate CPU offload so the pipeline can "
            "exceed GPU VRAM."
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
        default_warmup_runs=4,
    )
    sdxl_parser.set_defaults(
        steps=4,
        height=512,
        width=512,
        prompt="A cute dog and a cat in a park",
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
    sdxl_common.maybe_compile(pipe, args)


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


def apply_diffusers_group_cpu_offload(
    pipe: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    from diffusers.hooks import apply_group_offloading

    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()

    modules = [
        (name, module)
        for name in _GROUP_OFFLOAD_COMPONENTS
        if isinstance((module := getattr(pipe, name, None)), torch.nn.Module)
    ]
    if not modules:
        raise RuntimeError(
            "Could not find any SDXL torch.nn.Module components for "
            "--offload-mode group."
        )

    group_offload_kwargs = dict(
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type=args.group_offload_type,
        num_blocks_per_group=args.group_offload_num_blocks,
        non_blocking=args.group_offload_non_blocking,
        use_stream=args.group_offload_use_stream,
    )
    for _name, module in modules:
        apply_group_offloading(module=module, **group_offload_kwargs)


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
    if args.offload_mode == "module-hook":
        apply_accelerate_module_hook_cpu_offload(pipe, device)
        return
    apply_diffusers_group_cpu_offload(pipe, args, device)


def free_cpu_offload_hooks(pipe: Any) -> None:
    hooks = getattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, ())
    for hook in reversed(hooks or ()):
        offload = getattr(hook, "offload", None)
        if offload is not None:
            offload()
    setattr(pipe, _DIRECT_HF_ACCELERATE_HOOKS_ATTR, [])
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()


def _optional_model_arg(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "" or value.lower() == "none":
        return None
    return value


def _sdxl_text_to_image_pipeline_class() -> type[Any]:
    try:
        from diffusers import AutoPipelineForText2Image

        return AutoPipelineForText2Image
    except ModuleNotFoundError as exc:
        if sdxl_common.is_missing_diffusers_error(exc):
            raise RuntimeError(
                sdxl_common.format_missing_diffusers_runtime_error(
                    component="the Accelerate CPU offload runner",
                    exc=exc,
                )
            ) from exc
        raise
    except (ImportError, RuntimeError):
        from diffusers import StableDiffusionXLPipeline

        return StableDiffusionXLPipeline


def load_sdxl_pipeline(
    model: str,
    torch_dtype: torch.dtype,
    vae_model: str | None = DEFAULT_SDXL_VAE_MODEL,
    variant: str | None = DEFAULT_SDXL_PIPELINE_VARIANT,
) -> Any:
    try:
        from diffusers import AutoencoderKL
    except ModuleNotFoundError as exc:
        if sdxl_common.is_missing_diffusers_error(exc):
            raise RuntimeError(
                sdxl_common.format_missing_diffusers_runtime_error(
                    component="the Accelerate CPU offload runner",
                    exc=exc,
                )
            ) from exc
        raise

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
    }
    selected_variant = _optional_model_arg(variant)
    if selected_variant is not None:
        load_kwargs["variant"] = selected_variant

    selected_vae_model = _optional_model_arg(vae_model)
    if selected_vae_model is not None:
        load_kwargs["vae"] = AutoencoderKL.from_pretrained(
            selected_vae_model,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    pipeline_cls = _sdxl_text_to_image_pipeline_class()
    pipe = pipeline_cls.from_pretrained(model, **load_kwargs)
    vae_config = getattr(getattr(pipe, "vae", None), "config", None)
    if vae_config is not None and hasattr(vae_config, "force_upcast"):
        vae_config.force_upcast = False
    return pipe


def load_pipeline(args: argparse.Namespace, torch_dtype: torch.dtype) -> Any:
    return load_sdxl_pipeline(
        args.model,
        torch_dtype,
        getattr(args, "vae_model", DEFAULT_SDXL_VAE_MODEL),
        getattr(args, "variant", DEFAULT_SDXL_PIPELINE_VARIANT),
    )


def run_warmups(
    pipe: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    warmup_start = time.perf_counter()
    for _ in range(args.warmup_runs):
        warmup_output = sdxl_common.run_pipeline(pipe, args)
        sdxl_common.synchronize_device(device)
        del warmup_output
    return elapsed_seconds(warmup_start)


def run_inference(
    pipe: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, float]:
    inference_start = time.perf_counter()
    output = sdxl_common.run_pipeline(pipe, args)
    sdxl_common.synchronize_device(device)
    return output, elapsed_seconds(inference_start)


def decode_images(pipe: Any, args: argparse.Namespace, output: Any, device: torch.device) -> tuple[list[Any], float]:
    decode_start = time.perf_counter()
    with torch.no_grad():
        images = sdxl_common.decode_latents_to_pil(pipe, output.images)
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
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
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

    warmup_seconds = run_warmups(pipe, args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output, inference_seconds = run_inference(pipe, args, device)
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
    print("vae_model:", args.vae_model)
    print("variant:", args.variant)
    print("seed:", args.seed)
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
