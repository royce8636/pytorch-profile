#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_sdxl_turbo_common as sdxl_common


DTYPE_BY_NAME = sdxl_common.DTYPE_BY_NAME
RUN_RECORD_FUNCTION_NAME = "flux2_run"


def parse_args(
    *,
    default_model: str,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_component: str,
    default_fusion: str,
    default_offload_mode: str,
    default_profile_memory: bool,
    default_record_shapes: bool,
    default_with_stack: bool,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flux2 Klein profiler driver for the patched PyTorch build."
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
        help="Prompt passed to the pipeline.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text-encoder-out-layers",
        type=int,
        nargs="+",
        default=(9, 18, 27),
        help="Text encoder layer indices used by Flux2KleinPipeline.",
    )
    parser.add_argument("--device", default=default_device)
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
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the "
            "Flux2 transformer forward with torch.compile."
        ),
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--offload-mode",
        choices=("none", "model", "sequential"),
        default=default_offload_mode,
        help=(
            "'none' keeps the pipeline on the requested device, 'model' uses "
            "Diffusers enable_model_cpu_offload, and 'sequential' uses "
            "enable_sequential_cpu_offload."
        ),
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trace", default=f"/tmp/{default_output_prefix}_trace.json")
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--image", default=None)
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
        default="llamasim-runtime",
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
    parser.add_argument("--component", default=default_component)
    parser.add_argument("--phase", default="text_to_image")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument(
        "--profile-memory",
        dest="profile_memory",
        action="store_true",
        default=default_profile_memory,
    )
    parser.add_argument("--no-profile-memory", dest="profile_memory", action="store_false")
    parser.add_argument(
        "--record-shapes",
        dest="record_shapes",
        action="store_true",
        default=default_record_shapes,
    )
    parser.add_argument("--no-record-shapes", dest="record_shapes", action="store_false")
    parser.add_argument(
        "--with-stack",
        dest="with_stack",
        action="store_true",
        default=default_with_stack,
    )
    parser.add_argument("--no-with-stack", dest="with_stack", action="store_false")
    return parser.parse_args()


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) else None


def install_qwen3_transformers_compat() -> None:
    import transformers

    if hasattr(transformers, "Qwen3ForCausalLM"):
        return

    from transformers import Qwen2Config, Qwen2ForCausalLM

    class Qwen3CompatConfig(Qwen2Config):
        model_type = "qwen3"

    class Qwen3CompatForCausalLM(Qwen2ForCausalLM):
        config_class = Qwen3CompatConfig

    # Diffusers imports Flux2 with `from transformers import Qwen3ForCausalLM`.
    # Transformers 4.48 has Qwen2 but no Qwen3 export, and its lazy module does
    # not honor late-added attributes for from-imports. Replace only the package
    # root with a normal module that delegates all other attributes to the
    # original lazy package.
    wrapper = types.ModuleType("transformers")
    wrapper.__dict__.update(
        {
            "__file__": transformers.__file__,
            "__path__": transformers.__path__,
            "__spec__": transformers.__spec__,
            "__package__": transformers.__package__,
            "Qwen3Config": Qwen3CompatConfig,
            "Qwen3ForCausalLM": Qwen3CompatForCausalLM,
            "__getattr__": lambda name: getattr(transformers, name),
        }
    )
    sys.modules["transformers"] = wrapper


def format_flux2_runtime_error(
    *,
    model: str,
    exc: BaseException,
) -> str:
    return "\n".join(
        [
            "Failed to import Flux2 Klein runtime support needed by this profiler.",
            f"Model path: {model}",
            f"diffusers.__version__ = {_module_version('diffusers')!r}",
            f"transformers.__version__ = {_module_version('transformers')!r}",
            f"Original error: {exc}",
            "This checkout installs a local Qwen3 compatibility shim for older "
            "Transformers builds, but the underlying Qwen2 runtime must still be "
            "available.",
        ]
    )


def load_pipeline(
    model: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    *,
    offload_mode: str,
) -> Any:
    try:
        install_qwen3_transformers_compat()
        from diffusers import Flux2KleinPipeline
    except ModuleNotFoundError as exc:
        if sdxl_common.is_missing_diffusers_error(exc):
            raise RuntimeError(
                sdxl_common.format_missing_diffusers_runtime_error(
                    component="the Flux2 Klein scripts",
                    exc=exc,
                )
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(format_flux2_runtime_error(model=model, exc=exc)) from exc

    pipe = Flux2KleinPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    if offload_mode == "none":
        return pipe.to(device)
    if offload_mode == "model":
        pipe.enable_model_cpu_offload(device=device)
        return pipe
    pipe.enable_sequential_cpu_offload(device=device)
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


def run_pipeline(
    pipe: Any,
    args: argparse.Namespace,
    generator: torch.Generator | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "text_encoder_out_layers": tuple(args.text_encoder_out_layers),
        "output_type": "latent",
    }
    if generator is not None:
        kwargs["generator"] = generator
    return pipe(**kwargs)


def decode_latents_to_pil(pipe: Any, latents: torch.Tensor) -> list[Any]:
    latents = latents.to(pipe.vae.dtype)
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"sample_step=0 phase={args.phase} component={args.component} "
        f"device={args.device} fusion={args.fusion} offload_mode={args.offload_mode}"
    )


def main(
    *,
    default_model: str,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_component: str,
    default_fusion: str = "inductor",
    default_offload_mode: str = "none",
    default_profile_memory: bool = True,
    default_record_shapes: bool = True,
    default_with_stack: bool = True,
) -> None:
    args = parse_args(
        default_model=default_model,
        default_device=default_device,
        default_dtype=default_dtype,
        default_output_prefix=default_output_prefix,
        default_component=default_component,
        default_fusion=default_fusion,
        default_offload_mode=default_offload_mode,
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
    output_paths = sdxl_common.resolve_output_paths(
        args,
        default_output_prefix,
        default_llamasim_output_dirname="llama_bundle",
    )

    pipe = load_pipeline(
        args.model,
        torch_dtype,
        device,
        offload_mode=args.offload_mode,
    )
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

    execution_trace_observer = None
    if output_paths.execution_trace_path is not None:
        execution_trace_observer = torch.profiler.ExecutionTraceObserver()
        execution_trace_observer.register_callback(str(output_paths.execution_trace_path))
    try:
        with torch.profiler.profile(
            activities=sdxl_common.profile_activities(device),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            with_modules=True,
            execution_trace_observer=execution_trace_observer,
        ) as prof:
            with torch.autograd.profiler.record_function(
                RUN_RECORD_FUNCTION_NAME,
                metadata_for_scope(args),
            ):
                output = run_pipeline(pipe, args, generator)
            sdxl_common.profiler_synchronize_device(device)
    finally:
        if execution_trace_observer is not None:
            execution_trace_observer.unregister_callback()

    if capture_handle is not None:
        capture_handle.remove()

    metadata_json = None
    for event in prof.events():
        if event.name == RUN_RECORD_FUNCTION_NAME:
            metadata_json = event.metadata_json
            break

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    with torch.no_grad():
        images = decode_latents_to_pil(pipe, output.images)
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
        _pipe_catalog, _pipe_id_to_path = sdxl_common.build_pipeline_module_index(pipe)
        sdxl_common.write_llamasim_runtime_bundle(
            prof,
            output_paths.execution_trace_path,
            output_paths.llamasim_output_dir,
            trace_json_path=output_paths.trace_path,
            module_catalog=sdxl_common.build_module_catalog(_trans),
            module_id_to_path=sdxl_common.build_module_id_to_path(_trans),
            pipeline_module_catalog=_pipe_catalog,
            pipeline_module_id_to_path=_pipe_id_to_path,
        )

    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("model:", args.model)
    print("device:", device)
    print("dtype:", args.dtype)
    print("fusion:", args.fusion)
    print("offload_mode:", args.offload_mode)
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
