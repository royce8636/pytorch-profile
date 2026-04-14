#!/usr/bin/env python3
"""Weight streaming end-to-end example.

Usage:
    python3 weight_streaming_example.py

This compiles a simple model with weight streaming IO calls injected
between kernel dispatches, and exports the generated code to ./ws_output/.
"""
import json
import os
import tempfile

import torch
import torch._inductor.config as inductor_config
from torch._inductor.weight_streaming.plan import IOSchedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime


# ── Step 1: Define your model ──────────────────────────────────────────────

class TwoLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64, bias=False)
        self.fc2 = torch.nn.Linear(64, 64, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# ── Step 2: Create or load a schedule ─────────────────────────────────────
#
# In production, this comes from the llamasim scheduler:
#   python3 -m graph_modifiers.jit_sim_prune.main --profile-dir ... --output ...
#
# The schedule JSON has this structure:
#   - nodes: execution trace nodes with resource_kind ("gpu_stream" or "cpu_thread")
#   - io_operations: list of {"type": "prefetch"|"vram_prefetch_h2d"|"vram_evict_d2h", ...}
#   - spill_decisions: list of {"tensor_name", "evict_after_node", "reason", ...}
#   - cold_start_prefetches: list of {"tensor_name", "attach_before_node"}
#   - tensor_metadata: {name: {kind, size_bytes, dtype, shape, ...}}
#
# Here we create a minimal synthetic schedule for demonstration.

schedule_data = {
    # Execution trace nodes. "resource_kind" tells the adapter which are GPU kernels.
    # Only GPU nodes ("gpu_stream") map to Inductor wrapper kernel indices.
    "nodes": [
        {"idx": 0, "name": "aten::empty", "resource_kind": "cpu_thread"},
        {"idx": 1, "name": "triton_fused_relu_0", "resource_kind": "gpu_stream"},
        {"idx": 2, "name": "aten::empty", "resource_kind": "cpu_thread"},
    ],

    "io_operations": [
        # SSD -> DRAM prefetch before the relu kernel
        {
            "type": "prefetch",
            "tensor_name": "fc2.weight",
            "tensor_kind": "WEIGHT",
            "before_node": 1,        # needed by kernel at trace node 1
            "after_node": -1,        # can start immediately
            "eager_start": True,
        },
        # VRAM eviction after the relu kernel
        {
            "type": "vram_evict_d2h",
            "tensor_name": "fc1.weight",
            "tensor_kind": "WEIGHT",
            "after_node": 1,
        },
    ],

    "spill_decisions": [
        # DRAM eviction after the relu kernel
        {
            "tensor_name": "fc1.weight",
            "tensor_kind": "WEIGHT",
            "evict_after_node": 1,
            "reason": "immediate_evict_after_use",
            "writeback": False,
        },
    ],

    "cold_start_prefetches": [
        # Load fc1.weight into DRAM before anything runs
        {
            "tensor_name": "fc1.weight",
            "tensor_kind": "WEIGHT",
            "attach_before_node": 0,
        },
    ],

    # Optional: tensor metadata for SSD reads (not needed if tensors are pre-loaded)
    "tensor_metadata": {},
}

# Write schedule to a temp file (or use a real schedule path)
schedule_file = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False, prefix="ws_schedule_"
)
json.dump(schedule_data, schedule_file, indent=2)
schedule_file.flush()
schedule_path = schedule_file.name
print(f"Schedule written to: {schedule_path}")


# ── Step 3: Initialize runtime ────────────────────────────────────────────
#
# The runtime must be initialized before torch.compile runs, because the
# generated code calls WeightStreamRuntime.instance() at module load time.

WeightStreamRuntime.initialize(IOSchedule(tensors={}), torch.device("cuda"))

# In a real scenario, you'd pre-load weight tensors into DRAM here:
#   rt = WeightStreamRuntime.instance()
#   for name, param in model.named_parameters():
#       rt.register_dram_tensor(name, param.data.cpu().pin_memory())


# ── Step 4: Configure and compile ─────────────────────────────────────────

# Point Inductor at the schedule
inductor_config.weight_streaming_plan = schedule_path

# Optional: if nodes lack "resource_kind", provide a CSV
# inductor_config.weight_streaming_nodes_csv = "runtime_nodes.csv"

# Export the generated wrapper code to a directory
output_dir = os.path.join(os.getcwd(), "ws_output")
inductor_config.weight_streaming_output_code = output_dir

model = TwoLayerModel().cuda().eval()
inp = torch.randn(4, 64, device="cuda")

# Monkey-patch runtime methods to log + no-op (since we don't have real SSD files)
rt = WeightStreamRuntime.instance()
rt.ssd_prefetch = lambda n: print(f"  [runtime] ssd_prefetch({n!r})")
rt.h2d_prefetch = lambda n: print(f"  [runtime] h2d_prefetch({n!r})")
rt.evict_vram = lambda n: print(f"  [runtime] evict_vram({n!r})")
rt.evict_dram = lambda n: print(f"  [runtime] evict_dram({n!r})")
rt.cold_start_prefetch = lambda n: print(f"  [runtime] cold_start_prefetch({n!r})")

print("\nCompiling...")
compiled = torch.compile(model, fullgraph=True)

print("\nRunning (IO calls execute here):")
out = compiled(inp)
print(f"\nOutput shape: {out.shape}")


# ── Step 5: Inspect the generated code ────────────────────────────────────

out_file = os.path.join(output_dir, "weight_streaming_generated.py")
print(f"\nGenerated code exported to: {out_file}")
print("=" * 70)
with open(out_file) as f:
    code = f.read()

# Print just the call() function
lines = code.split("\n")
in_call = False
for i, line in enumerate(lines):
    if "def call(" in line:
        in_call = True
    if in_call:
        print(f"{i+1:4d}: {line}")
    if in_call and line.strip().startswith("return"):
        break

print("=" * 70)


# ── Cleanup ───────────────────────────────────────────────────────────────

inductor_config.weight_streaming_plan = ""
inductor_config.weight_streaming_nodes_csv = ""
inductor_config.weight_streaming_output_code = ""
WeightStreamRuntime.reset()
os.unlink(schedule_path)
