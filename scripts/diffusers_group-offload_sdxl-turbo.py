import time
import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from diffusers.hooks import apply_group_offloading
from huggingface_hub import login

model_id = "stabilityai/sdxl-turbo"

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.vae.config.force_upcast = False

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

group_offload_kwargs = dict(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=1,
    use_stream=True,
    non_blocking=False,
)

apply_group_offloading(pipe.unet, **group_offload_kwargs)
apply_group_offloading(pipe.text_encoder, **group_offload_kwargs)
apply_group_offloading(pipe.text_encoder_2, **group_offload_kwargs)
apply_group_offloading(pipe.vae, **group_offload_kwargs)

prompt = "A cute dog and a cat in a park"
height = 512
width = 512
num_inference_steps = 4

# Warmup to exclude one-time kernel compilation / autotune from the measurement.
_ = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=0.0,
).images[0]
torch.cuda.synchronize()
t1 = time.perf_counter()
peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
print(f"end-to-end time: {t1 - t0:.3f}s")
print(f"peak VRAM (allocated): {peak_allocated_gib:.3f} GiB")
print(f"peak VRAM (reserved):  {peak_reserved_gib:.3f} GiB")

image.save("sdxl_turbo_output.png")
