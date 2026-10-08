# Owner(s): ["module: inductor"]
"""The DRAM and SSD tiers end to end on a small OPT checkpoint: tiered loading,
the blob, gpu-co's pread backend (no device, no root), every issue order and
the next-token prefix, byte-exact against the resident model."""
import hashlib
import json
import os
import struct
import tempfile
import unittest

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.runtime import triton_heuristics
from gpuco_torch import config as ws_config
from torch._inductor.weight_streaming.codegen.inject import copy_list_path, reset_compile_position
from gpuco_torch.plan import load_plan, parse_plan, Tier
from gpuco_torch.runtime.copy_list import CopyList
from gpuco_torch.runtime.executor import Executor
from gpuco_torch.runtime.handshake import Deadlock, simulate
from gpuco_torch.runtime.registry import install, WeightRegistry
from gpuco_torch.tiers import blob, loader
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

try:
    import gpuco  # noqa: F401
    import transformers
    HAVE_DEPS = torch.cuda.is_available()
except ImportError:
    HAVE_DEPS = False

LAYERS = 3


def row(g, slot, first, last, **extra):
    r = {"group_id": g, "episode_id": g, "slot_id": slot, "first_use": first, "last_use": last, "pos": 0,
         "source": "dram", "ring_off": -1, "issue_launch": -1, "issue_prev_iter": False}
    r.update(extra)
    return r


class TestHandshake(TestCase):
    def test_passes_a_feasible_schedule(self):
        cl = CopyList.from_sidecar([row(0, 0, 0, 0), row(1, 0, 1, 1), row(2, 1, 2, 2, source="ssd")])
        program = [["wait", [0]], ["free", [0]], ["wait", [1]], ["free", [1]], ["wait", [2]], ["free", [2]]]
        simulate([program], cl, {2: 0}, iterations=4)

    def test_finds_a_tenant_waited_before_its_predecessor_frees(self):
        cl = CopyList.from_sidecar([row(0, 0, 0, 2), row(1, 0, 3, 4)])
        program = [["wait", [0]], ["wait", [1]], ["free", [0]], ["free", [1]]]  # 1 waits before 0 frees
        with self.assertRaisesRegex(Deadlock, r"graph at wait\(1\)"):
            simulate([program], cl, {})

    def test_finds_an_issue_nobody_writes(self):
        cl = CopyList.from_sidecar([row(0, 0, 0, 0, issue_launch=5)])
        with self.assertRaises(Deadlock):
            simulate([[["wait", [0]], ["free", [0]]]], cl, {})


def tiny_opt(path):
    from transformers import OPTConfig, OPTForCausalLM

    torch.manual_seed(0)
    cfg = OPTConfig(vocab_size=128, hidden_size=64, num_hidden_layers=LAYERS, ffn_dim=128, num_attention_heads=4,
                    max_position_embeddings=64, word_embed_proj_dim=64, do_layer_norm_before=True)
    from safetensors.torch import save_file

    # Written directly (save_pretrained imports DeepSpeed, which needs torch.distributed):
    # three shards, split mid-layer so groups straddle shard files, and no tied lm_head.
    state = {k: v.contiguous() for k, v in OPTForCausalLM(cfg).state_dict().items() if k != "lm_head.weight"}
    keys = sorted(state)
    weight_map = {}
    for i in range(3):
        part = keys[i::3]
        name = f"model-{i + 1:05d}-of-00003.safetensors"
        save_file({k: state[k] for k in part}, os.path.join(path, name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(part, name))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    cfg.save_pretrained(path)


def sublayer_plan(names, shard_blob=False):
    """Layer i's attention from DRAM (slots 0/1), its MLP from SSD (slots 2/3), the
    input embedding from DRAM (slot 4, whose FlexGen issue point is the previous
    token), the rest resident. ``shard_blob``: one SSD group per tensor (the shard
    layout reads each tensor at its checkpoint offset)."""
    ops, gid = [], 0

    def group(params, slot, tier):
        nonlocal gid
        for p in params:
            ops.append({"type": "vram_prefetch_h2d", "tensor_name": names[p], "compiled_graph_input_name": names[p],
                        "compiled_graph_id": 0, "before_launch_id": 0, "group_id": gid, "slot_id": slot,
                        "tier": tier, "source_pinned": tier == "dram"})
        gid += 1

    group(["model.decoder.embed_tokens.weight"], 4, "dram")
    for i in range(LAYERS):
        pre = f"model.decoder.layers.{i}."
        group([n for n in names if n.startswith(pre + "self_attn")], i % 2, "dram")
        mlp = [n for n in names if n.startswith((pre + "fc", pre + "final_layer_norm"))]
        if shard_blob:
            for k, p in enumerate(n for n in mlp if n.endswith("weight") and "fc" in n):
                group([p], 2 + k, "ssd")
        else:
            group(mlp, 2 + i % 2, "ssd")
    slots = [{"slot_id": s, "class_mb": 0.2} for s in range(5)]
    return {"summary": {"arena": {"slots": slots, "ring_bytes": 0}}, "io_operations": ops,
            "compilation_hash": "tiny"}


@unittest.skipIf(not HAVE_DEPS, "needs CUDA, transformers and gpu-co")
class TestTiers(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt = tempfile.mkdtemp()
        tiny_opt(cls.ckpt)

    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda", 0)
        self.dir = tempfile.mkdtemp()
        self.x = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(1)).to(self.device)
        # The tiered loader turns autotuning off; the reference must pick kernels the same way.
        triton_heuristics.set_skip_autotune(True)
        self.addCleanup(triton_heuristics.set_skip_autotune, False)
        self.addCleanup(install, None)
        self.addCleanup(ws_config.set, ws_config.Config())
        self.addCleanup(torch._dynamo.reset)

    def resident(self):
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(self.ckpt, torch_dtype=torch.float32, attn_implementation="sdpa")
        model = model.to(self.device).eval()
        loader.untie(model)
        return model

    def reference(self):
        torch._dynamo.reset()
        with inductor_config.patch(force_disable_caches=True):
            c = torch.compile(self.resident(), backend="inductor", dynamic=False)
            with torch.no_grad():
                c(self.x)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = c(self.x).logits
                g.replay()
        torch.cuda.synchronize()
        return out.clone()

    def input_names(self):
        """Discovery compile with an empty plan: graph input name per parameter."""
        path = os.path.join(self.dir, "empty.json")
        with open(path, "w") as f:
            json.dump({"summary": {"arena": {"slots": []}}, "io_operations": []}, f)
        model = self.resident()
        reg = WeightRegistry(load_plan(path), self.device)
        install(reg)
        torch._dynamo.reset()
        reset_compile_position()
        with inductor_config.patch(weight_streaming_plan=path, force_disable_caches=True), torch.no_grad():
            torch.compile(model, backend="inductor", dynamic=False)(self.x)
        by_id = {id(p): n for n, p in model.named_parameters()}
        return {by_id[id(t)]: name for (_, name), t in reg.named_weights.items() if id(t) in by_id}

    def streamed(self, cfg, shard_blob=False):
        names = self.input_names()
        plan_path = os.path.join(self.dir, "plan.json")
        with open(plan_path, "w") as f:
            json.dump(sublayer_plan(names, shard_blob), f)
        ws_config.set(cfg)
        registry = WeightRegistry(load_plan(plan_path), self.device)
        install(registry)
        model = loader.tiered_load(self.ckpt, registry, torch.float32, untie_lm_head=True)
        torch._dynamo.reset()
        reset_compile_position()
        with inductor_config.patch(weight_streaming_plan=plan_path, force_disable_caches=True), torch.no_grad():
            compiled = torch.compile(model, backend="inductor", dynamic=False)
            compiled(self.x)
        with open(copy_list_path(plan_path, 0)) as f:
            sidecar = json.load(f)
        copies = CopyList.from_sidecar(sidecar["copy_list"])
        ex = Executor(registry, copies, self.device, [sidecar["program"]])
        with torch.no_grad():
            ex.capture(lambda: compiled(self.x).logits)
        return ex, registry, copies

    @parametrize("order", ["free", "flexgen", "flexgen_sync", "flexgen_mb", "at_use"])
    def test_tiered_dram_ssd_byte_exact(self, order):
        ref = self.reference()
        blob_path = os.path.join(self.dir, "blob.bin")
        ex, registry, copies = self.streamed(ws_config.Config(issue_order=order, ssd_blob=blob_path, ssd_lanes=2))
        self.assertEqual({e.source for e in copies.entries}, {Tier.DRAM, Tier.SSD})
        gated = [e for e in copies.entries if e.issue_launch >= 0]
        self.assertEqual(len(gated), 0 if order == "free" else len(copies.entries))
        # FlexGen issues the first sub-layer's loads (the input embedding, and here the
        # first layer norm fused into its kernel) during the previous token's last one.
        prefix = copies.prefix
        first = min(e.first_use for e in copies.entries)
        if order.startswith("flexgen"):
            self.assertIn(0, [e.group_id for e in prefix])
            self.assertTrue(all(e.first_use == first for e in prefix))
        else:
            self.assertEqual(prefix, ())
        prefix_ahead = bool(prefix)
        for i in range(6):
            p = ex.step(prefix_ahead=prefix_ahead and i < 5)
            ex.wait()
            self.assertEqual(ex.static_output(p), ref, atol=0, rtol=0)
        rep = ex.report()
        self.assertGreater(rep["ssd_mb_per_iter"], 0)
        self.assertGreater(rep["dram_mb_per_iter"], 0)
        self.assertEqual(rep["ssd"]["cqe_errors"], 0)
        self.assertGreater(registry.landed_bytes, 0)  # the resident set came from the shards

    def test_accel_issue_map(self):
        """Accelerate's measured anchors: each module's read starts while the module
        before it runs (graph ids are a process-wide counter, so the map covers a range).
        Unfused, as plan replays compile, so launches belong to leaf modules."""
        nofuse = {k: v for k, v in ws_config.NOFUSE_INDUCTOR.items() if hasattr(inductor_config, k)}
        with inductor_config.patch(nofuse):
            self._accel()

    def _accel(self):
        ref = self.reference()
        model = self.resident()
        named = [(n, m) for n, m in model.named_modules() if n]
        leaves = []  # leaf modules in execution order, as Accelerate's run observes them
        hooks = [m.register_forward_pre_hook(lambda _m, _a, n=n: leaves.append(n))
                 for n, m in named if not list(m.children())]
        with torch.no_grad():
            model(self.x)
        for h in hooks:
            h.remove()
        amap = {}
        for n, _ in named:  # anchor: the leaf that ran just before the module's first leaf
            k = next((i for i, leaf in enumerate(leaves) if leaf == n or leaf.startswith(n + ".")), 0)
            amap[n] = leaves[max(k - 1, 0)]
        path = os.path.join(self.dir, "issue_map.json")
        with open(path, "w") as f:
            json.dump({str(g): amap for g in range(512)}, f)
        cfg = ws_config.Config(issue_order="accel", issue_map=path, ssd_blob=os.path.join(self.dir, "blob.bin"))
        ex, _, copies = self.streamed(cfg)
        self.assertTrue(all(e.issue_launch >= 0 and e.issue_launch <= e.first_use for e in copies.entries))
        self.assertTrue(any(e.issue_launch < e.first_use for e in copies.entries))
        for _ in range(4):
            p = ex.step()
            ex.wait()
            self.assertEqual(ex.static_output(p), ref, atol=0, rtol=0)

    def test_shard_blob_and_move_trace(self):
        ref = self.reference()
        from safetensors import safe_open  # noqa: F401

        shards = sorted(os.path.join(self.ckpt, f) for f in os.listdir(self.ckpt) if f.endswith(".safetensors"))
        trace = os.path.join(self.dir, "moves.jsonl")
        cfg = ws_config.Config(ssd_blob=os.path.join(self.dir, "shards.bin"), ssd_blob_shards=tuple(shards),
                               move_trace=trace, timeline=True)
        ex, registry, copies = self.streamed(cfg, shard_blob=True)
        self.assertEqual(ex.engine.blob.header, b"")
        for _ in range(4):
            p = ex.step()
            ex.wait()
            self.assertEqual(ex.static_output(p), ref, atol=0, rtol=0)
        rep = ex.report()
        rows = [json.loads(line) for line in open(trace)]
        self.assertEqual(rep["move_trace_rows"], len(rows))
        self.assertEqual(len(rows), 4 * len(copies.entries))
        self.assertEqual({r["src"] for r in rows}, {"DISK", "CPU"})
        disk = [r for r in rows if r["src"] == "DISK"]
        self.assertTrue(all(r["t"] <= r["read_end"] <= r["h2d_start"] <= r["h2d_end"] for r in disk))
        self.assertEqual(rep["ssd"]["timeline"]["iterations"], 4)

    def _gen_callables(self, model, P, N, B=1):
        """The runner's generation shape: one StaticCache, a prefill and a decode callable."""
        from transformers import StaticCache

        cfg = model.config
        cache = StaticCache(config=cfg, max_cache_len=P + N)
        cache.early_initialization(batch_size=B, num_heads=cfg.num_attention_heads,
                                   head_dim=cfg.hidden_size // cfg.num_attention_heads, dtype=torch.float32,
                                   device=self.device)
        am = torch.ones((B, P + N), dtype=torch.long, device=self.device)
        ids = torch.randint(0, 128, (B, P), generator=torch.Generator().manual_seed(3)).to(self.device)
        st = {"pre_ids": ids, "pre_pos": torch.arange(P, device=self.device),
              "dec_ids": torch.zeros((B, 1), dtype=torch.long, device=self.device),
              "dec_pos": torch.tensor([P], device=self.device)}
        compiled = torch.compile(model, backend="inductor", dynamic=False)

        def call(ids, pos):
            return compiled(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=pos,
                            position_ids=pos[None], attention_mask=am).logits[:, -1, :]

        return cache, st, {None: lambda: call(st["dec_ids"], st["dec_pos"]),
                           "prefill": lambda: call(st["pre_ids"], st["pre_pos"])}

    @staticmethod
    def _generate(step, out_of, wait, cache, st, P, N, drain=None, after_prefill=None):
        if drain:
            drain()
        cache.reset()
        wait()
        toks, logits = [], []
        out = out_of(step("prefill"), "prefill")
        wait()
        logits.append(out.clone())
        nxt = out.argmax(-1)
        toks.append(nxt.tolist())
        if after_prefill:
            after_prefill()
        st["dec_ids"].copy_(nxt.view(-1, 1))
        st["dec_pos"].fill_(P)
        for _ in range(N - 1):
            p = step(None)
            wait()
            out = out_of(p, None)
            logits.append(out.clone())
            nxt = out.argmax(-1)
            toks.append(nxt.tolist())
            st["dec_ids"].copy_(nxt.view(-1, 1))
            st["dec_pos"] += 1
        return toks, torch.stack(logits)

    def _spill_run(self, mode, P=6, N=5, B=1, segments=None, **cfg):
        """FlexGen's KV store schedule for all layers, every row (prefill stores
        [0, P), decode step i stores P + i - 1, the last token nothing), on the
        pread backend: the resident reference tokens and the streamed executor."""
        with torch.no_grad():
            model = self.resident()
            cache, st, fns = self._gen_callables(model, P, N, B)
            fns[None]()  # compile
            ref = self._generate(lambda k: k, lambda _p, k: fns[k](), torch.cuda.synchronize, cache, st, P, N)  # (tokens, logits)
        # Graph input names of the decode graph (an empty-plan discovery compile).
        empty = os.path.join(self.dir, "empty.json")
        with open(empty, "w") as f:
            json.dump({"summary": {"arena": {"slots": []}}, "io_operations": []}, f)
        model = self.resident()
        reg = WeightRegistry(load_plan(empty), self.device)
        install(reg)
        torch._dynamo.reset()
        reset_compile_position()
        with inductor_config.patch(weight_streaming_plan=empty, force_disable_caches=True), torch.no_grad():
            _, _, d = self._gen_callables(model, P, N, B)
            d[None]()
        by_id = {id(p): n for n, p in model.named_parameters()}
        names = {by_id[id(t)]: name for (_, name), t in reg.named_weights.items() if id(t) in by_id}

        spill = {"format": "spill-v1", "source": "flexgen",
                 "sites": [{"site": j, "layer": j, "segments": segments or [{"tier": "disk", "bh": [0, 4 * B]}]}
                           for j in range(LAYERS)],
                 "steps": [{"kind": "prefill", "positions": [0, P]}] +
                          [{"kind": "decode", "positions": None if i == N - 1 else [P + i - 1, P + i]}
                           for i in range(1, N)]}
        spill_path, plan_path = os.path.join(self.dir, "spill.json"), os.path.join(self.dir, "plan.json")
        with open(spill_path, "w") as f:
            json.dump(spill, f)
        with open(plan_path, "w") as f:
            json.dump(sublayer_plan(names), f)
        ws_config.set(ws_config.Config(ssd_blob=os.path.join(self.dir, "blob.bin"), ssd_lanes=2,
                                       spill_plan=spill_path, ssd_write_region="1G:64M", ssd_write_qps=2,
                                       spill_mode=mode, **cfg))
        registry = WeightRegistry(load_plan(plan_path), self.device)
        install(registry)
        model = loader.tiered_load(self.ckpt, registry, torch.float32, untie_lm_head=True)
        torch._dynamo.reset()
        reset_compile_position()
        with inductor_config.patch(weight_streaming_plan=plan_path, force_disable_caches=True), torch.no_grad():
            cache, st, fns = self._gen_callables(model, P, N, B)
            site_of = {j: j for j in range(LAYERS)}
            if mode == "home":
                from gpuco_torch.plan import parse_spill_plan
                from gpuco_torch.runtime.spill import prepare_kv_home
                layout = prepare_kv_home(dict(enumerate(cache.layers)), parse_spill_plan(spill),
                                         ws_config.get().kv_slots, self.device)
                registry.bind_spill(layout.kv, site_of, kv_layout=layout)
            else:
                registry.bind_spill({j: (cache.layers[j].keys, cache.layers[j].values) for j in range(LAYERS)}, site_of)
            fns[None]()  # compile + registration
            with open(copy_list_path(plan_path, 0)) as f:
                sidecar = json.load(f)
            copies = CopyList.from_sidecar(sidecar["copy_list"])
            if mode == "home":  # the runner's order: the prefill graph also compiles over detached views
                layout.release()
                reset_compile_position()
                fns["prefill"]()
                torch.cuda.synchronize()
            ex = Executor(registry, copies, self.device, [sidecar["program"]])
            ex.capture(fns[None], pre_call=cache.reset)
            reset_compile_position()
            ex.capture(fns["prefill"], "prefill", n_warm=2, pre_call=cache.reset)
        return ex, cache, st, ref, P, N

    def test_spill_flexgen_kv_writes_and_readback(self):
        """Write-through: the caches stay resident and every store is also written;
        tokens equal the resident run's and every record read back two steps later
        through the read path equals the cache rows."""
        ex, cache, st, ref, P, N = self._spill_run("write_through")
        with torch.no_grad():
            for _ in range(2):
                toks, logits = self._generate(lambda k: ex.step(k), lambda p, k: ex.static_output(p, k), ex.wait,
                                              cache, st, P, N, drain=ex.drain)
                self.assertEqual(toks, ref[0])
                self.assertEqual(logits, ref[1], atol=0, rtol=0)  # every step's logits, bit for bit
                check = ex.spill_check()
                self.assertEqual(check["mismatches"], 0, check)
                self.assertEqual(check["records"], LAYERS * (N - 1))  # the last token stores nothing
                self.assertEqual(check["checked"], LAYERS * (N - 2))  # the last lag steps' records stay unread
        rep = ex.report()
        self.assertEqual((rep["ssd"]["cqe_errors"], rep["ssd"]["write_cqe_errors"]), (0, 0))
        self.assertEqual(rep["spill"]["records_per_generation"], LAYERS * (N - 1))
        cache.layers[1].values[0, 2, P].add_(1.0)  # negative control: one decode row of one layer changed
        self.assertEqual(ex.spill_check()["mismatches"], 1)

    @parametrize("order", ["free", "flexgen_sync"])
    @parametrize("nofuse", [False, True])
    @parametrize("kv_slots", [1, 2])
    def test_spill_kv_homed_on_ssd(self, kv_slots, nofuse, order):
        """KV homed on SSD (FlexGen's cache on disk): the caches are views into
        ``kv_slots`` slots shared by all layers; every decode step loads each
        layer's rows [0, pos) from the drive, the graph appends its row, the row is
        written back. Every step's logits equal the resident run's. Unfused (as plan
        replays compile), Inductor would write a copy of each cache back at the end
        of the graph; the slot update is an opaque in-place op so it cannot."""
        patch = {k: v for k, v in ws_config.NOFUSE_INDUCTOR.items() if hasattr(inductor_config, k)} if nofuse else {}
        with inductor_config.patch(patch):
            self._kv_home(kv_slots, order)

    @parametrize("order", ["free", "flexgen_sync"])
    @parametrize("split", ["cpu_disk", "gpu_cpu_disk", "cpu"])
    def test_spill_kv_split_tiers(self, split, order):
        """FlexGen's split cache home at batch 4 (16 batch x head rows, whole heads
        of 4): each layer's segments live on the drive, in pinned DRAM and in VRAM,
        every decode step loads each into the KV slot and stores the new rows back
        to each. Unfused, as plan replays compile. Every step's logits equal the
        resident run's."""
        segs = {"cpu_disk": [("cpu", 0, 8), ("disk", 8, 16)],
                "gpu_cpu_disk": [("gpu", 0, 4), ("cpu", 4, 8), ("disk", 8, 16)],
                "cpu": [("cpu", 0, 16)]}[split]
        nofuse = {k: v for k, v in ws_config.NOFUSE_INDUCTOR.items() if hasattr(inductor_config, k)}
        with inductor_config.patch(nofuse):
            ex, cache, st, ref, P, N = self._spill_run(
                "home", B=4, kv_slots=2, issue_order=order,
                segments=[{"tier": t, "bh": [lo, hi]} for t, lo, hi in segs])
            with torch.no_grad():
                for _ in range(2):
                    toks, logits = self._generate(lambda k: ex.step(k), lambda p, k: ex.static_output(p, k), ex.wait,
                                                  cache, st, P, N, drain=ex.drain)
                    self.assertEqual(toks, ref[0])
                    self.assertEqual(logits, ref[1], atol=0, rtol=0)
            rows = ex.spill.check()["rows_by_tier"]
            self.assertEqual(rows, {t: LAYERS * (hi - lo) for t, lo, hi in segs})

            home = next(h for h in ex.spill.homes[1] if h.tier != "disk")
            buf = next(b for b in ex.spill._buffers if b.data_ptr() == home.base)

            def corrupt():  # negative control: layer 1's off-drive home, after the prefill stored it
                torch.cuda.synchronize()
                buf.fill_(7)

            with torch.no_grad():
                _, bad = self._generate(lambda k: ex.step(k), lambda p, k: ex.static_output(p, k), ex.wait, cache,
                                        st, P, N, drain=ex.drain, after_prefill=corrupt)
            self.assertEqual(bad[0], ref[1][0], atol=0, rtol=0)
            self.assertFalse(any(torch.equal(bad[i], ref[1][i]) for i in range(1, N)))
            rep = ex.report()
            self.assertEqual((rep["ssd"]["cqe_errors"], rep["ssd"]["write_cqe_errors"]), (0, 0))

    def _kv_home(self, kv_slots, order):
        ex, cache, st, ref, P, N = self._spill_run("home", kv_slots=kv_slots, issue_order=order)
        # FlexGen's timing: block j's load_cache at the start of the FlexGen layer
        # before its attention (block 0's: the same token's input embedding).
        self.assertEqual(ex.spill.timed, order != "free")
        self.assertEqual(ex.flags.kv_prev_iter, set())
        storages = {cache.layers[j].keys.untyped_storage().data_ptr() for j in range(LAYERS)}
        self.assertEqual(len(storages), 1)  # every layer's KV lives in the one slot buffer
        self.assertEqual(len({cache.layers[j].keys.data_ptr() for j in range(LAYERS)}), kv_slots)
        with torch.no_grad():
            for _ in range(2):
                toks, logits = self._generate(lambda k: ex.step(k), lambda p, k: ex.static_output(p, k), ex.wait,
                                              cache, st, P, N, drain=ex.drain)
                self.assertEqual(toks, ref[0])
                self.assertEqual(logits, ref[1], atol=0, rtol=0)  # every step's logits, bit for bit
                ex.spill_check()
        rep = ex.report()
        self.assertEqual((rep["ssd"]["cqe_errors"], rep["ssd"]["write_cqe_errors"]), (0, 0))
        self.assertEqual(rep["spill"]["loads_per_generation"], LAYERS * (N - 1))  # every step but the prefill

        def corrupt():  # negative control: layer 1's prompt rows on the drive, after the prefill stored them
            torch.cuda.synchronize()
            home = ex.spill.disk_home(1)
            with open(ws_config.get().ssd_blob, "r+b") as f:
                f.seek(home.base)
                f.write(bytes(range(256)) * (P * home.segrow // 256))

        with torch.no_grad():
            _, bad = self._generate(lambda k: ex.step(k), lambda p, k: ex.static_output(p, k), ex.wait, cache, st,
                                    P, N, drain=ex.drain, after_prefill=corrupt)
        self.assertEqual(bad[0], ref[1][0], atol=0, rtol=0)  # the prefill attends to the slot, not the drive
        self.assertFalse(any(torch.equal(bad[i], ref[1][i]) for i in range(1, N)))  # every decode step reads it
        self.assertEqual(rep["spill"]["records_per_generation"], LAYERS * (N - 1))

    def test_blob_format(self):
        names = self.input_names()
        registry = WeightRegistry(parse_plan(sublayer_plan(names)), self.device)
        model = loader.tiered_load(self.ckpt, registry, torch.float32, untie_lm_head=True)
        for n, p in model.named_parameters():
            registry.register_weight_named(0, names[n], p)
        gids = [g.group_id for g in registry.plan.groups_on(Tier.SSD)]
        layout = blob.packed_layout(registry, gids)
        # The old tree's format: magic + sha1(compilation hash, per group: id, member offsets)[:16], padded.
        h = hashlib.sha1(b"tiny")
        for gid in sorted(gids):
            h.update(struct.pack("<q", gid))
            for w in registry.group_members[gid]:
                h.update(struct.pack("<qq", *registry.member_offset[id(w)]))
        self.assertEqual(layout.header, (b"WSB1" + h.hexdigest()[:16].encode()).ljust(4096, b"\0"))
        offs = [layout.records[g][0] for g in sorted(gids)]
        self.assertEqual(offs[0], 4096)
        self.assertTrue(all(o % 4096 == 0 for o in offs))
        path = os.path.join(self.dir, "blob.bin")
        self.assertFalse(blob.matches(layout, path))
        blob.write(layout, registry, path)
        self.assertTrue(blob.matches(layout, path))
        g = registry.group_members[gids[0]][0]
        off, n = registry.member_offset[id(g)]
        with open(path, "rb") as f:
            f.seek(layout.records[gids[0]][0] + off)
            self.assertEqual(f.read(n), registry.host_bytes(g).numpy().tobytes())


instantiate_parametrized_tests(TestTiers)


if __name__ == "__main__":
    run_tests()
