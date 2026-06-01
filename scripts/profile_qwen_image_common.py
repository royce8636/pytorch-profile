#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_sdxl_turbo_common as sdxl_common


DTYPE_BY_NAME = sdxl_common.DTYPE_BY_NAME
RUN_RECORD_FUNCTION_NAME = "qwen_image_run"


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
        description="Qwen-Image profiler driver for the patched PyTorch build."
    )
    parser.add_argument(
        "--model",
        default="/data/llamasim/models/qwen-image-2512",
        help="Path to the Qwen-Image-2512 model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="a lovely cat",
        help="Prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=" ",
        help=(
            "Negative prompt passed to the pipeline. A non-empty value keeps "
            "classifier-free guidance enabled when --true-cfg-scale > 1."
        ),
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="Traditional classifier-free guidance scale used by Qwen-Image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Optional guidance-distilled scale. Leave unset for the current "
            "Qwen-Image-2512 model."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the generation RNG. Set to a negative value to disable.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum prompt sequence length passed to the pipeline.",
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
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the "
            "Qwen transformer forward with torch.compile; on CUDA this "
            "typically produces fused Triton kernels."
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
            "Directory for profiler outputs. If set, writes trace JSON, trace "
            "CSV, PNG, and optional DOT outputs into this directory."
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
            "Path for the exported profiler CSV. Defaults to the trace path "
            "with a .csv suffix, or a filename under --output-dir."
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
        default="none",
        help=(
            "Optional DOT graph outputs. 'fx' exports the logical transformer "
            "FX graph, 'execution' exports a DOT converted from "
            "ExecutionTraceObserver output, 'memory' exports a tensor-flow DOT "
            "from the profiler memory graph, 'kineto' exports the post-fusion "
            "runtime event graph from Kineto, 'hybrid' overlays matched "
            "tensor-flow onto Kineto runtime nodes, 'runtime-io' overlays "
            "exact ExecutionTraceObserver I/O onto Kineto runtime nodes, "
            "'llamasim-runtime' emits a step_0_compute_graph.dot bundle for "
            "/data/llamasim with CPU leaf nodes plus GPU runtime nodes, "
            "'both' writes fx+execution, and 'all' writes "
            "fx+execution+memory+kineto+hybrid+runtime-io+llamasim-runtime."
        ),
    )
    parser.add_argument(
        "--fx-dot",
        default=None,
        help=(
            "Path for the exported transformer FX DOT graph. Requires "
            "--dot-level fx, both, or all."
        ),
    )
    parser.add_argument(
        "--execution-dot",
        default=None,
        help=(
            "Path for the execution-trace DOT graph. Requires --dot-level "
            "execution, both, or all."
        ),
    )
    parser.add_argument(
        "--execution-trace",
        default=None,
        help=(
            "Path for the raw execution trace JSON used to build "
            "--execution-dot. Requires --dot-level execution, runtime-io, "
            "llamasim-runtime, both, or all."
        ),
    )
    parser.add_argument(
        "--memory-dot",
        default=None,
        help=(
            "Path for the memory/data-flow DOT graph. Requires --dot-level "
            "memory or all."
        ),
    )
    parser.add_argument(
        "--kineto-dot",
        default=None,
        help=(
            "Path for the Kineto runtime DOT graph. Requires --dot-level "
            "kineto or all."
        ),
    )
    parser.add_argument(
        "--kineto-map",
        default=None,
        help=(
            "Path for the Kineto-to-trace.csv mapping file. Requires "
            "--dot-level kineto or all."
        ),
    )
    parser.add_argument(
        "--hybrid-dot",
        default=None,
        help=(
            "Path for the hybrid runtime+tensors DOT graph. Requires "
            "--dot-level hybrid or all."
        ),
    )
    parser.add_argument(
        "--runtime-io-dot",
        default=None,
        help=(
            "Path for the exact runtime+I/O DOT graph. Requires "
            "--dot-level runtime-io or all."
        ),
    )
    parser.add_argument(
        "--llamasim-output-dir",
        default=None,
        help=(
            "Directory for a llamasim runtime bundle containing "
            "step_0_compute_graph.dot, runtime node/edge metadata, tensor "
            "metadata, and a manifest. Requires --dot-level llamasim-runtime "
            "or all."
        ),
    )
    parser.add_argument(
        "--component",
        default="qwen_image_pipeline",
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


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) else None


def _read_qwen_model_requirements(model: str) -> tuple[str | None, str | None]:
    config_path = Path(model) / "text_encoder" / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None

    architectures = config.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        model_type = None
    if not isinstance(architecture, str):
        architecture = None
    return architecture, model_type


def format_qwen_runtime_error(
    *,
    model: str,
    exc: BaseException,
    diffusers_version: str | None = None,
    transformers_version: str | None = None,
    architecture: str | None = None,
    model_type: str | None = None,
) -> str:
    lines = [
        "Failed to import Qwen-Image runtime support needed by this profiler.",
        f"Model path: {model}",
    ]
    if diffusers_version is not None:
        lines.append(f"diffusers.__version__ = {diffusers_version!r}")
    if transformers_version is not None:
        lines.append(f"transformers.__version__ = {transformers_version!r}")
    if architecture is not None:
        lines.append(f"Text encoder architecture = {architecture!r}")
    if model_type is not None:
        lines.append(f"Text encoder model_type = {model_type!r}")
    lines.append(f"Original error: {exc}")

    if (
        architecture == "Qwen2_5_VLForConditionalGeneration"
        or model_type == "qwen2_5_vl"
        or "Qwen2_5_VLForConditionalGeneration" in str(exc)
        or "qwen2_5_vl" in str(exc)
    ):
        lines.append(
            "This runtime does not support the Qwen 2.5 VL text encoder "
            "required by Qwen-Image-2512."
        )
        lines.append(
            "Use a Python environment with a transformers build that includes "
            "`Qwen2_5_VLForConditionalGeneration` and `qwen2_5_vl` configs."
        )
    return "\n".join(lines)


def format_qwen_cuda_oom_error(
    *,
    model: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    exc: BaseException,
    accelerate_version: str | None,
) -> str:
    lines = [
        "Failed to place Qwen-Image-2512 on the requested accelerator for profiling.",
        f"Model path: {model}",
        f"device = {device}",
        f"torch_dtype = {torch_dtype}",
        f"Original error: {exc}",
        "The model does not fit on this accelerator in the current runtime.",
    ]
    if accelerate_version is None:
        lines.append(
            "This environment does not have `accelerate`, so diffusers device_map/offload loading is unavailable."
        )
    else:
        lines.append(f"accelerate.__version__ = {accelerate_version!r}")
    lines.append(
        "Use a larger GPU, run with `--device cpu`, or use an environment with `accelerate` so the model can be loaded with device mapping/offload."
    )
    return "\n".join(lines)


def install_qwen25_transformers_compat() -> None:
    try:
        import transformers
        import transformers.models.qwen2_vl as qwen2_vl_module
        import transformers.models.qwen2_vl.configuration_qwen2_vl as qwen2_vl_config_module
        import transformers.models.qwen2_vl.modeling_qwen2_vl as qwen2_vl_model_module
        from transformers import (
            PretrainedConfig,
            Qwen2VLConfig,
            Qwen2VLForConditionalGeneration,
        )
    except Exception:
        return

    if hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        return

    class Qwen25VLTextConfig(PretrainedConfig):
        model_type = "qwen2_5_vl_text"

    class Qwen25VLCompatConfig(Qwen2VLConfig):
        model_type = "qwen2_5_vl"

        def __init__(
            self,
            text_config: dict[str, Any] | PretrainedConfig | None = None,
            vision_config: dict[str, Any] | None = None,
            **kwargs: Any,
        ):
            super().__init__(vision_config=vision_config, **kwargs)
            if isinstance(text_config, dict):
                self.text_config = Qwen25VLTextConfig(**text_config)
            else:
                self.text_config = text_config

    class Qwen25VLCompatForConditionalGeneration(
        Qwen2VLForConditionalGeneration
    ):
        config_class = Qwen25VLCompatConfig

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str | Path,
            *model_args: Any,
            **kwargs: Any,
        ) -> "Qwen25VLCompatForConditionalGeneration":
            config = kwargs.get("config")
            if config is None:
                config = cls.config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    local_files_only=kwargs.get("local_files_only", False),
                )
            elif isinstance(config, Qwen2VLConfig) and not isinstance(
                config, cls.config_class
            ):
                config = cls.config_class.from_dict(config.to_dict())

            if hasattr(config, "text_config"):
                # The older transformers generation config code expects the
                # decoder text config to be a PretrainedConfig. Qwen-Image only
                # uses text-only prompt encoding here, so fall back to the outer
                # config instead of keeping the nested raw dict.
                config.text_config = None

            kwargs["config"] = config
            kwargs.setdefault("ignore_mismatched_sizes", True)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs,
            )

    transformers._objects["Qwen2_5_VLConfig"] = Qwen25VLCompatConfig
    transformers._objects[
        "Qwen2_5_VLForConditionalGeneration"
    ] = Qwen25VLCompatForConditionalGeneration
    transformers._class_to_module["Qwen2_5_VLConfig"] = "models.qwen2_vl"
    transformers._class_to_module[
        "Qwen2_5_VLForConditionalGeneration"
    ] = "models.qwen2_vl"
    qwen2_vl_module.Qwen2_5_VLConfig = Qwen25VLCompatConfig
    qwen2_vl_module.Qwen2_5_VLForConditionalGeneration = (
        Qwen25VLCompatForConditionalGeneration
    )
    qwen2_vl_config_module.Qwen2_5_VLConfig = Qwen25VLCompatConfig
    qwen2_vl_model_module.Qwen2_5_VLForConditionalGeneration = (
        Qwen25VLCompatForConditionalGeneration
    )
    transformers.Qwen2_5_VLConfig = Qwen25VLCompatConfig
    transformers.Qwen2_5_VLForConditionalGeneration = (
        Qwen25VLCompatForConditionalGeneration
    )


def load_pipeline(model: str, torch_dtype: torch.dtype, device: torch.device) -> Any:
    architecture, model_type = _read_qwen_model_requirements(model)
    install_qwen25_transformers_compat()
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
                    component="the Qwen-Image scripts",
                    exc=exc,
                )
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            format_qwen_runtime_error(
                model=model,
                exc=exc,
                diffusers_version=_module_version("diffusers"),
                transformers_version=_module_version("transformers"),
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
    try:
        pipe = pipe.to(device)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            format_qwen_cuda_oom_error(
                model=model,
                device=device,
                torch_dtype=torch_dtype,
                exc=exc,
                accelerate_version=_module_version("accelerate"),
            )
        ) from exc
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


def maybe_compile(
    pipe: Any,
    args: argparse.Namespace,
    output_paths: sdxl_common.OutputPaths | None = None,
) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")

    if output_paths is not None and output_paths.llamasim_output_dir is not None:
        sdxl_common.configure_llamasim_inductor_markers(
            output_paths.llamasim_output_dir
        )

    # QwenImagePipeline relies on transformer.cache_context(...), so preserve
    # the module instance and compile only the forward method.
    pipe.transformer.forward = torch.compile(
        pipe.transformer.forward,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )


def build_generator(args: argparse.Namespace, device: torch.device) -> torch.Generator | None:
    if args.seed is None or args.seed < 0:
        return None
    return torch.Generator(device=str(device)).manual_seed(args.seed)


def run_pipeline(pipe: Any, args: argparse.Namespace, generator: torch.Generator | None) -> Any:
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": args.steps,
        "height": args.height,
        "width": args.width,
        "output_type": "latent",
        "max_sequence_length": args.max_sequence_length,
    }
    if args.guidance_scale is not None:
        kwargs["guidance_scale"] = args.guidance_scale
    if generator is not None:
        kwargs["generator"] = generator
    return pipe(**kwargs)


def decode_latents_to_pil(
    pipe: Any, latents: torch.Tensor, *, height: int, width: int
) -> list[Any]:
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"sample_step=0 phase={args.phase} component={args.component} "
        f"device={args.device} fusion={args.fusion}"
    )


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
    sdxl_common.validate_dot_args(args)
    device = sdxl_common.validate_device(args)
    sdxl_common.validate_fusion_runtime(args, device)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    warmup_runs = (
        args.warmup_runs if args.warmup_runs is not None else int(args.fusion != "none")
    )
    output_paths = sdxl_common.resolve_output_paths(args, default_output_prefix)

    pipe = load_pipeline(args.model, torch_dtype, device)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    maybe_compile(pipe, args, output_paths)
    generator = build_generator(args, device)

    captured_transformer_inputs: dict[str, Any] = {}
    capture_handle = None
    if output_paths.fx_dot_path is not None:
        captured_transformer_inputs, capture_handle = sdxl_common.capture_example_inputs(
            pipe.transformer
        )

    for _ in range(warmup_runs):
        warmup_output = run_pipeline(pipe, args, generator)
        sdxl_common.synchronize_device(device)
        del warmup_output

    scope_args = metadata_for_scope(args)
    execution_trace_observer = None
    if output_paths.execution_trace_path is not None:
        execution_trace_observer = torch.profiler.ExecutionTraceObserver()
        execution_trace_observer.register_callback(str(output_paths.execution_trace_path))
    with torch.profiler.profile(
        activities=sdxl_common.profile_activities(device),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_modules=True,
        execution_trace_observer=execution_trace_observer,
    ) as prof:
        with torch.autograd.profiler.record_function(
            RUN_RECORD_FUNCTION_NAME, scope_args
        ):
            output = run_pipeline(pipe, args, generator)
        sdxl_common.profiler_synchronize_device(device)

    if capture_handle is not None:
        capture_handle.remove()
    if execution_trace_observer is not None:
        execution_trace_observer.unregister_callback()

    metadata_json = None
    for event in prof.events():
        if event.name == RUN_RECORD_FUNCTION_NAME:
            metadata_json = event.metadata_json
            break

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    with torch.no_grad():
        images = decode_latents_to_pil(
            pipe,
            output.images,
            height=args.height,
            width=args.width,
        )
    images[0].save(output_paths.image_path)

    if output_paths.fx_dot_path is not None:
        if (
            "args" not in captured_transformer_inputs
            or "kwargs" not in captured_transformer_inputs
        ):
            raise RuntimeError(
                "Failed to capture transformer example inputs for FX DOT export"
            )
        fx_graph_module = sdxl_common.export_unet_graph_module(
            sdxl_common.underlying_unet_module(pipe.transformer),
            captured_transformer_inputs["args"],
            captured_transformer_inputs["kwargs"],
        )
        sdxl_common.write_fx_graph_dot(
            fx_graph_module,
            output_paths.fx_dot_path,
            output_paths.fx_dot_path.stem,
        )

    if output_paths.execution_dot_path is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError(
                "Execution DOT export requested without an execution trace path"
            )
        sdxl_common.write_execution_trace_dot(
            output_paths.execution_trace_path,
            output_paths.execution_dot_path,
            output_paths.execution_dot_path.stem,
        )
    if output_paths.memory_dot_path is not None:
        sdxl_common.write_memory_profile_dot(
            prof,
            output_paths.memory_dot_path,
            output_paths.memory_dot_path.stem,
        )
    if output_paths.kineto_dot_path is not None:
        sdxl_common.write_kineto_profile_dot(
            prof,
            output_paths.kineto_dot_path,
            output_paths.kineto_dot_path.stem,
            output_paths.kineto_map_path,
        )
    if output_paths.hybrid_dot_path is not None:
        sdxl_common.write_hybrid_profile_dot(
            prof,
            output_paths.hybrid_dot_path,
            output_paths.hybrid_dot_path.stem,
        )
    if output_paths.runtime_io_dot_path is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError(
                "Runtime-I/O DOT export requested without an execution trace path"
            )
        sdxl_common.write_runtime_io_profile_dot(
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
        _trans = sdxl_common.underlying_unet_module(pipe.transformer)
        # Use prof-derived observation-order mapping (see
        # `build_module_id_to_path_from_prof`).
        _pipe_catalog, _pipe_id_to_path = (
            sdxl_common.build_pipeline_module_index_from_prof(prof, pipe)
        )
        sdxl_common.write_llamasim_runtime_bundle(
            prof,
            output_paths.execution_trace_path,
            output_paths.llamasim_output_dir,
            trace_json_path=output_paths.trace_path,
            module_catalog=sdxl_common.build_module_catalog(_trans),
            module_id_to_path=sdxl_common.build_module_id_to_path_from_prof(
                prof, _trans,
            ),
            pipeline_module_catalog=_pipe_catalog,
            pipeline_module_id_to_path=_pipe_id_to_path,
            module_hierarchy=sdxl_common.build_module_hierarchy(pipe),
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
        sdxl_common.print_llamasim_runtime_summary(output_paths.llamasim_output_dir)
    print("latent_shape:", tuple(output.images.shape))
    print("metadata_json:", metadata_json)
    if metadata_json:
        print("metadata_parsed:", json.dumps(json.loads(metadata_json), sort_keys=True))
