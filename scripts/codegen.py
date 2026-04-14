
import torch
import torch._inductor.config as inductor_config
from torch._inductor.weight_streaming.plan import IOSchedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime

# 1. Initialize runtime (before torch.compile)
WeightStreamRuntime.initialize(IOSchedule(tensors={}), torch.device("cuda"))

# 2. Point config at schedule + output dir
inductor_config.weight_streaming_plan = "/data/pytorch-source/llamasim_bundle/jit_sim_prune/jit_sim_prune_schedule.json"
inductor_config.weight_streaming_output_code = "./ws_output"

# Optional: if nodes in JSON lack "resource_kind"
# inductor_config.weight_streaming_nodes_csv = "runtime_nodes.csv"

# 3. Compile and run — IO calls injected automatically
compiled = torch.compile(model, fullgraph=True)
output = compiled(input_tensor)

