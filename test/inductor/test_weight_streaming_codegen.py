# Owner(s): ["module: inductor"]
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

import torch
from torch import nn
from torch._inductor import config as inductor_config
from torch._inductor.weight_streaming.codegen.inject import copy_list_path, reset_compile_position
from gpuco_torch.plan import load_plan
from gpuco_torch.runtime.copy_list import CopyList
from gpuco_torch.runtime.executor import Executor
from gpuco_torch.runtime.registry import install, WeightRegistry
from torch.testing._internal.common_utils import run_tests, TestCase


D, LAYERS = 64, 6


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(D, D) for _ in range(LAYERS))
        self.head = nn.Linear(D, 8, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.head(x)


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestInjection(TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda", 0)
        self.dir = tempfile.mkdtemp()
        self.addCleanup(install, None)

    def tearDown(self):
        torch._dynamo.reset()

    def model(self):
        torch.manual_seed(0)
        return Net().to(self.device).eval(), torch.randn(4, D, device=self.device)

    def compile_with(self, net, x, plan_path):
        registry = WeightRegistry(load_plan(plan_path), self.device)
        install(registry)
        torch._dynamo.reset()
        reset_compile_position()
        with inductor_config.patch(weight_streaming_plan=plan_path, force_disable_caches=True):
            compiled = torch.compile(net, backend="inductor", dynamic=False)
            with torch.no_grad():
                compiled(x)  # compile + registration
        return registry, lambda: compiled(x)

    def write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_compiled_model_streams_byte_exact(self):
        net, x = self.model()
        # Reference: the same compiled model, captured, weights resident.
        torch._dynamo.reset()
        with inductor_config.patch(force_disable_caches=True):
            ref_c = torch.compile(net, backend="inductor", dynamic=False)
            with torch.no_grad():
                ref_c(x)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    ref_out = ref_c(x)
                g.replay()
        torch.cuda.synchronize()
        ref = ref_out.clone()

        # Discovery compile with an empty plan: the graph's input names.
        empty = self.write("empty.json", {"summary": {"arena": {"slots": []}}, "io_operations": []})
        reg0, _ = self.compile_with(net, x, empty)
        pname = {id(p): n for n, p in net.named_parameters()}
        by_param = {pname[id(t)]: name for (gid, name), t in reg0.named_weights.items() if id(t) in pname}
        self.assertEqual(len(by_param), 2 * LAYERS + 1)
        install(None)

        # Layer i streams from DRAM through slot i % 2; the head stays resident.
        ops = [{"type": "vram_prefetch_h2d", "tensor_name": by_param[f"layers.{i}.{k}"],
                "compiled_graph_input_name": by_param[f"layers.{i}.{k}"], "compiled_graph_id": 0,
                "before_launch_id": 0, "group_id": i, "slot_id": i % 2, "tier": "dram", "source_pinned": True}
               for i in range(LAYERS) for k in ("weight", "bias")]
        plan = self.write("plan.json", {"summary": {"arena": {"slots": [{"slot_id": 0, "class_mb": 0.1},
                                                                        {"slot_id": 1, "class_mb": 0.1}]}},
                                        "io_operations": ops})
        head = net.head.weight.data_ptr()
        registry, fwd = self.compile_with(net, x, plan)
        self.assertEqual(net.head.weight.data_ptr(), head)
        self.assertEqual(net.layers[0].weight.untyped_storage().data_ptr(), registry.arena.storage.data_ptr())

        with open(copy_list_path(plan, 0)) as f:
            sidecar = json.load(f)
        rows = sidecar["copy_list"]
        self.assertEqual([r["group_id"] for r in rows], list(range(LAYERS)))
        self.assertTrue(all(r["first_use"] <= r["last_use"] < n["first_use"] for r, n in zip(rows, rows[1:])))
        self.assertEqual({r["source"] for r in rows}, {"dram"})

        ex = Executor(registry, CopyList.from_sidecar(rows), self.device)
        with torch.no_grad():
            ex.capture(fwd)
            for _ in range(5):  # 2 slots x 3 tenants
                p = ex.step()
                torch.cuda.synchronize()
                self.assertEqual(ex.static_output(p), ref, atol=0, rtol=0)

    def test_graph_without_plan_groups_is_untouched(self):
        net, x = self.model()
        empty = self.write("empty.json", {"summary": {"arena": {"slots": []}}, "io_operations": []})
        registry, fwd = self.compile_with(net, x, empty)
        with open(copy_list_path(empty, 0)) as f:
            self.assertEqual(json.load(f)["copy_list"], [])
        self.assertEqual(registry.group_members, {})
        self.assertEqual(len(registry.named_weights), 2 * LAYERS + 1)



class TestMicroBatchPasses(TestCase):
    def test_second_pass_of_each_sublayer(self):
        """Two micro-batches over an embedding and one OPT block: each FlexGen layer
        repeats its module sequence, and the second pass starts where the layer's
        first module starts again (consecutive kernels of one module are one start)."""
        from torch._inductor.weight_streaming.codegen.issue_order import _second_passes, _sublayers

        seq = []
        for _ in range(2):
            seq += ["decoder.embed_tokens", "", "decoder.embed_positions"]
        for _ in range(2):
            seq += ["decoder.layers.0.self_attn_layer_norm", "decoder.layers.0.self_attn_layer_norm",
                    "decoder.layers.0.self_attn.q_proj", "decoder.layers.0.self_attn.out_proj", ""]
        for _ in range(2):
            seq += ["decoder.layers.0.final_layer_norm", "decoder.layers.0.fc1", "decoder.layers.0.fc2", ""]
        for _ in range(2):
            seq += ["decoder.final_layer_norm", "lm_head"]
        stacks = {lid: [(m, "M")] if m else [] for lid, m in enumerate(seq)}
        sub_of = _sublayers(stacks)
        self.assertEqual(sorted(set(sub_of.values())), [-1, 0, 1, 2])
        self.assertEqual(_second_passes(stacks, sub_of), {-1: 3, 0: 11, 1: 20, 2: 26})
        one = {lid: st for lid, st in stacks.items() if lid < 3}
        self.assertEqual(_second_passes(one, _sublayers(one)), {})


if __name__ == "__main__":
    run_tests()
