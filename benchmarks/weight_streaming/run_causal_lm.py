"""Run a Hugging Face causal LM under captured weight streaming, against an
existing plan, or as the resident reference a streamed run must reproduce.

Fixed-shape forward (the prompt, one iteration per replay):
    python run_causal_lm.py --model meta-llama/Llama-3.1-8B --plan plan.json --work-dir W
Generation (prefill once, then N-1 static-KV decode steps, argmax fed back):
    python run_causal_lm.py --model facebook/opt-30b --plan plan.json --work-dir W --dtype fp16 \\
        --gen-prompt-len 32 --gen-len 16 --tiered-load /ckpt/opt-30b --untie-lm-head --nofuse \\
        --ssd-pci 0000:53:00.0 --ssd-blob blob.bin [--issue-order flexgen_sync --prefix-ahead]
Reference (every weight resident, same compiled graphs, no streaming):
    python run_causal_lm.py --model ... [--gen-prompt-len 32 --gen-len 16] --reference --dump W/ref.pt

Streamed: load -> compile with the plan active -> one call (registration: host
copies or shard landing, arena repointing, copy lists) -> capture -> steps.
The plan is copied into the work dir, where the copy lists are written.
"""
import argparse
import json
import os
import shutil
import time

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

import torch
from torch._inductor import config as inductor_config
from gpuco_torch import config as ws_config
from torch._inductor.weight_streaming.codegen.inject import copy_list_path, reset_compile_position
from gpuco_torch.plan import load_plan
from gpuco_torch.runtime.copy_list import CopyList
from gpuco_torch.runtime.executor import Executor
from gpuco_torch.runtime.handshake import simulate
from gpuco_torch.runtime.registry import install, WeightRegistry
from gpuco_torch.tiers import loader


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face id or checkpoint dir (config + tokenizer)")
    ap.add_argument("--plan", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--reference", action="store_true", help="no streaming: every weight resident")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--gen-prompt-len", type=int, default=0)
    ap.add_argument("--gen-len", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=1, help="sequences per step, all the same prompt")
    ap.add_argument("--plan-crosswalk", default="",
                    help="param_crosswalk.json of the compile the plan was made for: its graph-input names are "
                         "renamed, by checkpoint name, to this compile's (needed when the inputs differ)")
    ap.add_argument("--gen-micro-batches", type=int, default=1,
                    help="OPT only: FlexGen's GPU batches, each layer run on every micro-batch in turn")
    ap.add_argument("--gen-repeats", type=int, default=2)
    ap.add_argument("--replays", type=int, default=20, help="fixed-forward mode: timed replays")
    ap.add_argument("--tiered-load", default="", help="safetensors checkpoint dir: load by tier, no full copy")
    ap.add_argument("--untie-lm-head", action="store_true")
    ap.add_argument("--attn", default="", help="attention implementation (the tiered loader's default: sdpa)")
    ap.add_argument("--nofuse", action="store_true", help="one kernel per op, program order (plan replays)")
    ap.add_argument("--vram-limit-gib", type=float, default=0.0)
    ap.add_argument("--issue-order", default="free", choices=ws_config.ISSUE_ORDERS)
    ap.add_argument("--issue-map", default="")
    ap.add_argument("--prefix-ahead", action="store_true", help="enqueue the next token's prefix before launch")
    ap.add_argument("--ssd-pci", default="")
    ap.add_argument("--ssd-blob", default="")
    ap.add_argument("--ssd-blob-shards", default="", help="comma-separated safetensors files forming the blob")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spill-plan", default="", help="KV write-back schedule (spill-v1, e.g. planners/import_flexgen_spill.py)")
    ap.add_argument("--write-region", default="", help="device bytes LO:BYTES for the KV writes, above the blob")
    ap.add_argument("--write-qps", type=int, default=4)
    ap.add_argument("--spill-readback-lag", type=int, default=2)
    ap.add_argument("--spill-mode", default="home", choices=("home", "write_through"),
                    help="home: the spilled KV lives on the drive, VRAM keeps --kv-slots slots; write_through: "
                         "the caches stay resident and are also written (a test of the write path)")
    ap.add_argument("--kv-slots", type=int, default=2)
    ap.add_argument("--move-trace", default="")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--hang-watchdog", type=float, default=0.0)
    ap.add_argument("--dump", default="", help="torch.save the outputs (logits / tokens)")
    ap.add_argument("--report-json", default="")
    return ap.parse_args()


def load_model(a, device, dtype, registry):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if a.tiered_load:
        return loader.tiered_load(a.tiered_load, registry, dtype, a.untie_lm_head, a.attn or "sdpa"), tokenizer
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, attn_implementation=a.attn or None)
    model = model.to(device).eval()
    if a.untie_lm_head:
        loader.untie(model)
    return model, tokenizer


class OPTMicroBatched(torch.nn.Module):
    """``opt_micro_forward`` as a module holding the model, so compiled kernels keep
    dotted module paths (``model.model.decoder.layers.3.fc1``) that the FlexGen issue
    orders map to sub-layers; a compiled closure gets mangled source names."""

    def __init__(self, model, caches, masks):
        super().__init__()
        self.model = model
        self.caches = caches
        self.masks = masks

    def forward(self, ids, pos):
        return opt_micro_forward(self.model, ids, self.caches, self.masks, pos[None])


def opt_micro_forward(model, ids, caches, masks, position_ids):
    """FlexGen's block schedule for OPT: every FlexGen layer (input embedding,
    each block's attention half, its MLP half, the output layer) runs on each
    micro-batch in turn before the next layer. Micro-batch k is rows
    [k*b, (k+1)*b) of ``ids`` with its own static KV cache. Last-position logits."""
    from transformers.masking_utils import create_causal_mask

    dec = model.model.decoder
    k_ids = ids.chunk(len(caches))
    hs, causal = [], []
    for x, cache, am in zip(k_ids, caches, masks):
        emb = dec.embed_tokens(x)
        past = cache.get_seq_length()
        causal.append(create_causal_mask(config=dec.config, inputs_embeds=emb, attention_mask=am, past_key_values=cache))
        hs.append(emb + dec.embed_positions(am, past, position_ids=position_ids))
    for layer in dec.layers:
        for k, cache in enumerate(caches):
            attn, _ = layer.self_attn(hidden_states=layer.self_attn_layer_norm(hs[k]), past_key_values=cache,
                                      position_ids=position_ids, attention_mask=causal[k])
            hs[k] = hs[k] + attn
        for k in range(len(caches)):
            shape = hs[k].shape
            h = hs[k].reshape(-1, shape[-1])
            hs[k] = (h + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(h))))).view(shape)
    return torch.cat([model.lm_head(dec.final_layer_norm(h[:, -1:, :]))[:, -1, :] for h in hs])


def input_positions(fn, args, model) -> dict[str, int]:
    """Checkpoint name -> position among the graph inputs Dynamo lifts for
    ``fn(*args)`` (Inductor names input i ``arg{i}_1``). Traced only: the
    backend records its inputs and raises, so nothing runs."""
    names = {id(t): n for n, t in list(model.named_parameters()) + list(model.named_buffers())}
    found: dict[str, int] = {}

    def backend(gm, example_inputs):
        for i, t in enumerate(example_inputs):
            if id(t) in names:
                found.setdefault(names[id(t)], i)
        raise RuntimeError("traced")

    torch._dynamo.reset()
    try:
        torch.compile(fn, backend=backend, dynamic=False)(*args)
    except Exception:
        if not found:
            raise
    torch._dynamo.reset()
    return found


def retarget_plan(plan_path: str, crosswalk_path: str, positions: dict[str, int]) -> None:
    """Rename the plan's graph inputs through their checkpoint names."""
    with open(plan_path) as f:
        plan = json.load(f)
    with open(crosswalk_path) as f:
        graphs = json.load(f)["graphs"]
    for op in plan["io_operations"]:
        if op.get("type") == "vram_prefetch_h2d":
            name = f"arg{positions[graphs[str(op['compiled_graph_id'])][op['compiled_graph_input_name']]]}_1"
            op["compiled_graph_input_name"] = op["tensor_name"] = name
    with open(plan_path, "w") as f:
        json.dump(plan, f)


def build_forward(a, model, tokenizer, compiled, device):
    """kind -> callable; plus the generation state when generating."""
    if not a.gen_len:
        x = tokenizer([a.prompt], return_tensors="pt", padding=True)["input_ids"].to(device)
        return {None: lambda: compiled(x).logits}, None
    from transformers import StaticCache

    P, N = a.gen_prompt_len, a.gen_len
    ids = tokenizer(a.prompt, return_tensors="pt").input_ids[0].tolist()
    while len(ids) < P:
        ids = ids + ids[1:]
    B = a.gen_batch
    pre_ids = torch.tensor([ids[:P]] * B, dtype=torch.long, device=device)
    cfg = model.config
    K = a.gen_micro_batches
    if B % K:
        raise SystemExit(f"--gen-batch {B} is not a multiple of --gen-micro-batches {K}")
    caches = [StaticCache(config=cfg, max_cache_len=P + N) for _ in range(K)]
    kv_layout = None
    if a.spill_plan and a.spill_mode == "home":
        # The spilled layers' caches become views into a few KV slots, set up one
        # layer at a time: the full cache is never allocated.
        from gpuco_torch.plan import load_spill_plan
        from gpuco_torch.runtime.spill import prepare_kv_home
        if K != 1:
            raise SystemExit("--spill-mode home supports one KV cache (no --gen-micro-batches)")
        sp = load_spill_plan(a.spill_plan)
        fake = torch.zeros((B, getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads, 0,
                            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads),
                           dtype=next(model.parameters()).dtype, device=device)
        kv_layout = prepare_kv_home({s.layer: caches[0].layers[s.layer] for s in sp.sites}, sp, a.kv_slots, device,
                                    init=lambda layer: layer.lazy_initialization(fake, fake))
    # Allocated now: lazily, the first compiled call would allocate it and Dynamo recompile
    # (layers already set up, the spilled ones, are skipped).
    for c in caches:
        c.early_initialization(batch_size=B // K, num_heads=getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
                               head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
                               dtype=next(model.parameters()).dtype, device=device)
    cache = caches[0]
    am = torch.ones((B, P + N), dtype=torch.long, device=device)
    pre_pos = torch.arange(P, device=device)
    dec_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
    dec_pos = torch.tensor([P], device=device)

    if K > 1:
        if cfg.model_type != "opt":
            raise SystemExit("--gen-micro-batches is implemented for OPT only")
        raw = OPTMicroBatched(model, caches, am.chunk(K))
        micro = torch.compile(raw, backend="inductor", mode="default", dynamic=False)

    def call(ids_, pos):
        if K > 1:
            return micro(ids_, pos)
        out = compiled(input_ids=ids_, past_key_values=cache, use_cache=True, cache_position=pos,
                       position_ids=pos[None], attention_mask=am)
        return out.logits[:, -1, :]

    # StaticCache writes at a static-address counter it advances in place and
    # ignores cache_position; reset it before every prefill, and each decode
    # replay advances it by one, exactly a generation.
    gen = {"P": P, "N": N, "reset": lambda: [c.reset() for c in caches], "dec_ids": dec_ids, "dec_pos": dec_pos,
           "kv": {i: (layer.keys, layer.values) for i, layer in enumerate(cache.layers)} if K == 1 else None,
           "layers": dict(enumerate(cache.layers)) if K == 1 else None, "kv_layout": kv_layout,
           "prompt_ids": ids[:P], "trace": (raw, (dec_ids, dec_pos)) if K > 1 else None}
    return {None: lambda: call(dec_ids, dec_pos), "prefill": lambda: call(pre_ids, pre_pos)}, gen


class Resident:
    """The reference: the same callables captured with no streaming."""

    def __init__(self):
        self.graphs, self.outs = {}, {}

    def capture(self, fn, kind=None, pre_call=None):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if pre_call:
                    pre_call()
                fn()
        torch.cuda.current_stream().wait_stream(s)
        if pre_call:
            pre_call()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.outs[kind] = fn()
        self.graphs[kind] = g
        torch.cuda.synchronize()

    def step(self, kind=None, prefix_ahead=False):
        self.graphs[kind].replay()
        return 0

    def static_output(self, p, kind=None):
        return self.outs[kind]

    def wait(self):
        torch.cuda.synchronize()


def row_tokens(nxt):
    """One step's argmax: a token id at batch 1, one per sequence otherwise."""
    return int(nxt.item()) if nxt.numel() == 1 else nxt.tolist()


def generate(runner, gen, repeats, prefix_ahead):
    """Prefill once, then N-1 decode replays with the argmax fed back in place;
    one warm run, then ``repeats`` timed."""
    P, N, dec_ids, dec_pos = gen["P"], gen["N"], gen["dec_ids"], gen["dec_pos"]
    runs, last = [], None
    for r in range(1 + repeats):
        toks, t_dec = [], []
        with torch.no_grad():
            getattr(runner, "drain", runner.wait)()  # spill stages read the caches the reset clears
            gen["reset"]()
            runner.wait()
            t_start = time.perf_counter()
            p = runner.step("prefill", prefix_ahead=prefix_ahead)
            runner.wait()
            t_pre = time.perf_counter() - t_start
            nxt = runner.static_output(p, "prefill").argmax(-1)
            toks.append(row_tokens(nxt))
            dec_ids.copy_(nxt.view(-1, 1))
            dec_pos.fill_(P)
            for i in range(N - 1):
                t0 = time.perf_counter()
                # The last replay of the last run leaves nothing enqueued ahead.
                p = runner.step(prefix_ahead=prefix_ahead and not (r == repeats and i == N - 2))
                runner.wait()
                t_dec.append(time.perf_counter() - t0)
                last = runner.static_output(p).clone()
                nxt = last.argmax(-1)
                toks.append(row_tokens(nxt))
                dec_ids.copy_(nxt.view(-1, 1))
                dec_pos += 1
        e2e_ms = (time.perf_counter() - t_start) * 1e3
        check = runner.spill_check() if hasattr(runner, "spill_check") else None
        if check is not None:
            print(f"→ [spill] run {r}: {check}", flush=True)
        runs.append({"prefill_ms": t_pre * 1e3, "decode_ms_per_token": sum(t_dec) / max(1, len(t_dec)) * 1e3,
                     "tokens": toks, "e2e_ms": e2e_ms, "spill": check, "decode_ms": [t * 1e3 for t in t_dec],
                     "wall": [time.time() - (time.perf_counter() - t_start), time.time()]})
        print(f"→ [gen] run {r}{' (warm)' if r == 0 else ''}: prefill {runs[-1]['prefill_ms']:.1f} ms, decode "
              f"{runs[-1]['decode_ms_per_token']:.1f} ms/token, tokens {toks[:8]}...", flush=True)
    timed = runs[1:] or runs
    res = {"P": P, "N": N, "prompt_ids": gen["prompt_ids"], "tokens": runs[0]["tokens"],
           "spill_checks": [r["spill"] for r in runs if r["spill"] is not None],
           "decode_ms_per_run": [r["decode_ms"] for r in runs],
           "tokens_identical_across_runs": all(r["tokens"] == runs[0]["tokens"] for r in runs),
           "runs_wall": [r["wall"] for r in runs]}  # epoch seconds, for aligning external samplers
    for k in ("prefill_ms", "decode_ms_per_token", "e2e_ms"):
        res[k] = sum(r[k] for r in timed) / len(timed)
    res["decode_e2e_ms_per_token"] = (res["e2e_ms"] - res["prefill_ms"]) / max(1, N - 1)
    return res, last


def replay_fixed(runner, replays):
    outs = {}
    with torch.no_grad():
        for _ in range(6):
            p = runner.step()
            runner.wait()
            out = runner.static_output(p)
            if p in outs and not torch.equal(outs[p], out):
                raise RuntimeError(f"bank {p} is not reproducible across replays")
            outs[p] = out.clone()
        t0 = time.perf_counter()
        for _ in range(replays):
            runner.step()
        runner.wait()
        elapsed = time.perf_counter() - t0
        p = runner.step()
        runner.wait()
        post = runner.static_output(p).clone()
    return outs, post, elapsed / max(replays, 1) * 1e3


def memory(stage: str) -> None:
    print(f"→ memory {stage}: allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB, reserved "
          f"{torch.cuda.memory_reserved() / 2**20:.0f} MiB, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB",
          flush=True)


def read_copy_list(plan_path):
    """The forward's merged copy list, per-graph digests and flag programs."""
    rows, digests, programs, pos = [], [], [], 0
    while os.path.exists(copy_list_path(plan_path, pos)):
        with open(copy_list_path(plan_path, pos)) as f:
            sidecar = json.load(f)
        rows += sidecar["copy_list"]
        digests.append(sidecar["graph_digest"])
        programs.append(sidecar["program"])
        pos += 1
    return rows, digests, programs


def main() -> None:
    a = parse_args()
    device = torch.device("cuda")
    if a.vram_limit_gib:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(a.vram_limit_gib * 2**30 / total, torch.cuda.current_device())
    if a.nofuse:
        for k, v in ws_config.NOFUSE_INDUCTOR.items():
            if hasattr(inductor_config, k):
                setattr(inductor_config, k, v)
    # A cache hit would skip the wrapper codegen that instruments the graph.
    inductor_config.force_disable_caches = True
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    report: dict = {}

    if a.reference:
        if a.tiered_load:
            # Resident, but with the streamed run's kernel-selection policy (the
            # tiered loader turns autotuning off), so the two compile the same kernels.
            from torch._inductor.runtime import triton_heuristics
            triton_heuristics.set_skip_autotune(True)
            a.tiered_load = ""
            a.attn = a.attn or "sdpa"
        model, tokenizer = load_model(a, device, dtype, None)
        torch._dynamo.reset()
        compiled = torch.compile(model, backend="inductor", mode="default", dynamic=False)
        fns, gen = build_forward(a, model, tokenizer, compiled, device)
        runner = Resident()
        with torch.no_grad():
            runner.capture(fns[None], pre_call=gen["reset"] if gen else None)
            if gen:
                runner.capture(fns["prefill"], "prefill", pre_call=gen["reset"])
        if gen:
            report["gen"], last = generate(runner, gen, a.gen_repeats, False)
            dump = {"tokens": report["gen"]["tokens"], "last_logits": last}
        else:
            outs, post, report["replay_ms_per_iter"] = replay_fixed(runner, a.replays)
            dump = {"logits": outs[0], "post_liveness": post}
        report["peak_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        if not (a.plan and a.work_dir):
            raise SystemExit("--plan and --work-dir are required unless --reference")
        os.makedirs(a.work_dir, exist_ok=True)
        plan_path = os.path.join(a.work_dir, "plan.json")
        if os.path.abspath(a.plan) != os.path.abspath(plan_path):
            shutil.copyfile(a.plan, plan_path)
        pos = 0
        while os.path.exists(copy_list_path(plan_path, pos)):  # stale lists would be merged
            os.remove(copy_list_path(plan_path, pos))
            pos += 1
        ws_config.set(ws_config.Config(
            issue_order=a.issue_order, issue_map=a.issue_map, ssd_pci=a.ssd_pci, ssd_blob=a.ssd_blob,
            ssd_blob_shards=tuple(p for p in a.ssd_blob_shards.split(",") if p), ssd_lanes=a.lanes,
            move_trace=a.move_trace, timeline=a.timeline, hang_watchdog_s=a.hang_watchdog,
            spill_plan=a.spill_plan, ssd_write_region=a.write_region, ssd_write_qps=a.write_qps,
            spill_readback_lag=a.spill_readback_lag, spill_mode=a.spill_mode, kv_slots=a.kv_slots))
        registry = WeightRegistry(load_plan(plan_path), device)
        install(registry)
        model, tokenizer = load_model(a, device, dtype, registry)
        report["load_peak_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
        memory("after load")
        inductor_config.weight_streaming_plan = plan_path
        torch._dynamo.reset()
        reset_compile_position()
        compiled = torch.compile(model, backend="inductor", mode="default", dynamic=False)
        fns, gen = build_forward(a, model, tokenizer, compiled, device)
        if a.spill_plan:
            from gpuco_torch.plan import load_spill_plan
            if not (gen and gen["kv"]):
                raise SystemExit("--spill-plan needs generation with one KV cache (--gen-len, no micro-batches)")
            sp = load_spill_plan(a.spill_plan)
            if len(sp.steps) != gen["N"]:
                raise SystemExit(f"the spill plan has {len(sp.steps)} steps, a generation here has {gen['N']}")
            site_of = {s.layer: s.site for s in sp.sites}
            if a.spill_mode == "home":
                kv_layout = gen["kv_layout"]
                registry.bind_spill(kv_layout.kv, site_of, kv_layout=kv_layout)
                print(f"→ [spill] KV of {len(sp.sites)} layers homed on SSD: {kv_layout.n_slots} slots x "
                      f"{kv_layout.slot_bytes / 1e6:.2f} MB, row stride {kv_layout.rowstride} B", flush=True)
            else:
                registry.bind_spill({s.layer: gen["kv"][s.layer] for s in sp.sites}, site_of)
        if a.plan_crosswalk:
            fn, args = gen["trace"] if gen and gen["trace"] else (model, ())
            if not args:
                raise SystemExit("--plan-crosswalk is supported with --gen-micro-batches > 1")
            with torch.no_grad():
                retarget_plan(plan_path, a.plan_crosswalk, input_positions(fn, args, model))
            registry.plan = load_plan(plan_path)
            reset_compile_position()
        with torch.no_grad():
            fns[None]()  # compile + registration; its output is not streamed
        torch.cuda.synchronize()
        memory("after registration")
        loader.write_crosswalk(registry, model, os.path.join(a.work_dir, "param_crosswalk.json"))
        rows, digests, programs = read_copy_list(plan_path)
        copies = CopyList.from_sidecar(rows)
        print(f"→ digest {':'.join(digests)}; copy list {len(rows)} entries from {len(digests)} graph(s)", flush=True)
        if registry.kv_home:
            # Every graph compiles over detached cache views (kv_bind moves them into
            # the slots when the graph runs): detach them again for the prefill graph.
            registry.kv_layout.release()
            reset_compile_position()
            with torch.no_grad():
                fns["prefill"]()
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        ex = Executor(registry, copies, device, programs)
        memory("after executor")
        with torch.no_grad():
            ex.capture(fns[None], pre_call=gen["reset"] if gen else None)
            memory("after capture")
            if gen:
                reset_compile_position()  # the prefill graph compiles at the decode graph's position
                ex.capture(fns["prefill"], "prefill", n_warm=2, pre_call=gen["reset"])
                pre_rows, _, pre_programs = read_copy_list(plan_path)
                copies.check_same_schedule(pre_rows, "the prefill graph")
                simulate(programs, copies, ex.engine.lane_of(), first=pre_programs)
        if gen:
            report["gen"], last = generate(ex, gen, a.gen_repeats, a.prefix_ahead)
            dump = {"tokens": report["gen"]["tokens"], "last_logits": last}
        else:
            outs, post, report["replay_ms_per_iter"] = replay_fixed(ex, a.replays)
            report["banks_agree"] = torch.equal(outs[0], outs[1])
            dump = {"logits": outs[0], "logits_bank1": outs[1], "post_liveness": post}
        report.update(ex.report(), digests=digests, peak_mb=torch.cuda.max_memory_allocated(device) / 1e6)
        from torch._inductor.runtime import triton_heuristics
        report["autotune_skipped"] = triton_heuristics.autotune_skipped
    print("→ " + json.dumps({k: v for k, v in report.items() if k != "ssd"}), flush=True)
    if a.dump:
        torch.save(dump, a.dump)
    if a.report_json:
        with open(a.report_json, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
