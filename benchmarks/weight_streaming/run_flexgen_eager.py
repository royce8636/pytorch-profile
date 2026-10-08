"""OPT generation with FlexGen's own compute, eager, captured per position.

FlexGen (FlexLLMGen flex_opt.py / pytorch_backend.py) runs OPT as eager PyTorch:
an input-embedding layer, each block's attention and MLP as two layers, an
output layer, every layer on each GPU batch (micro-batch) in turn. This runner
issues the same calls in the same order on the same layouts (its KV cache
(P+N-1, b*n_head, head_dim), int32 token ids, bool masks), so every compute
kernel is the one FlexGen launches. Each position (prefill, decode step i) is
its own CUDA graph, as its attention length differs. Weights stream through the
executor; its flag operations sit between FlexGen's layers:

  free          a layer's waits before its first micro-batch, frees after its last
  flexgen_sync  as free, and layer j+1 is issued after layer j's waits
                (layer 0 of the next token at the start of the output layer)
  flexgen_mb    FlexGen's per-micro-batch sync(): layer j+1 is issued with j's
                first micro-batch and waited before j's second

    python run_flexgen_eager.py --tiered-load /ckpt/opt-30b --plan plan.json \\
        --plan-crosswalk param_crosswalk.json --work-dir W --gen-batch 64 --micro-batches 8 \\
        --ssd-pci 0000:52:00.0 --ssd-blob blob.bin --issue-order flexgen_mb
"""
import argparse
import json
import os
import shutil
import time

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

import torch
import torch.nn.functional as F
from gpuco_torch import config as ws_config
from gpuco_torch.plan import load_plan
from gpuco_torch.runtime.copy_list import CopyList
from gpuco_torch.runtime.executor import Executor
from gpuco_torch.runtime.registry import WeightRegistry
from gpuco_torch.tiers import loader


ORDERS = ("free", "flexgen_sync", "flexgen_mb")


# ---- FlexGen's compute (pytorch_backend.TorchDevice), as its GPU path runs it ----

def _noop(stage):
    pass


def opt_input_embed(token_ids, mask, w_token, w_pos, pad_token_id, at=_noop):
    at("tok")
    token_embed = F.embedding(token_ids, w_token, pad_token_id)
    positions = torch.cumsum(mask, dim=1).int() * mask + 1
    past_key_values_length = mask.shape[1] - token_ids.shape[1]
    positions = positions[:, past_key_values_length:]
    at("pos")
    pos_embed = F.embedding(positions, w_pos)
    return token_embed + pos_embed


def opt_output_embed(inputs, w_ln, b_ln, w_token, at=_noop):
    b, s, h = inputs.shape
    at("ln")
    hidden = F.layer_norm(inputs, (h,), weight=w_ln, bias=b_ln)
    at("lm")
    logits = F.linear(hidden, w_token)
    last_token_logits = logits[:, -1, :]
    return last_token_logits.argmax(dim=1, keepdim=True), last_token_logits


def mha(inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, at=_noop):
    b, s, h = inputs.shape
    head_dim = h // n_head
    scaling = head_dim ** -0.5
    at("ln")
    hidden = F.layer_norm(inputs, (h,), weight=w_ln, bias=b_ln)
    at("q")
    q = F.linear(hidden, w_q, bias=b_q) * scaling
    at("k")
    k = F.linear(hidden, w_k, bias=b_k)
    at("v")
    v = F.linear(hidden, w_v, bias=b_v)
    q = q.view(b, s, n_head, head_dim)
    k = k.view(b, s, n_head, head_dim)
    v = v.view(b, s, n_head, head_dim)
    q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
    k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
    v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
    attn_weights = torch.bmm(q, k)
    idx = torch.arange(s, device=inputs.device)
    causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
    mask = attention_mask.view(b, 1, 1, s) & causal_mask
    attn_weights = attn_weights.view(b, n_head, s, s)
    attn_weights = torch.where(mask, attn_weights, -1e4)
    attn_weights = attn_weights.view(b * n_head, s, s)
    attn_weights = F.softmax(attn_weights, dim=2)
    value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
    value = value.transpose(1, 2).reshape(b, s, h)
    at("out")
    value = F.linear(value, w_out, bias=b_out)
    value.add_(inputs)
    k = k.permute(2, 0, 1)
    v = v.permute(1, 0, 2)
    return value, k, v


def mha_gen(inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
            k_cache, v_cache, at=_noop):
    b, tgt_s, h = inputs.shape
    src_s = attention_mask.shape[1]
    head_dim = h // n_head
    scaling = head_dim ** -0.5
    at("ln")
    hidden = F.layer_norm(inputs, (h,), weight=w_ln, bias=b_ln)
    at("q")
    q = F.linear(hidden, w_q, bias=b_q) * scaling
    at("k")
    k = F.linear(hidden, w_k, bias=b_k)
    at("v")
    v = F.linear(hidden, w_v, bias=b_v)
    q = q.view(b, tgt_s, n_head, head_dim)
    k = k.view(b, tgt_s, n_head, head_dim)
    v = v.view(b, tgt_s, n_head, head_dim)
    q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
    k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
    v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
    k = k_cache[:src_s]
    v = v_cache[:src_s]
    k[src_s - 1:src_s] = k_new
    v[src_s - 1:src_s] = v_new
    k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
    v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
    attn_weights = torch.bmm(q, k)
    mask = attention_mask.view(b, 1, 1, src_s)
    attn_weights = attn_weights.view(b, n_head, 1, src_s)
    attn_weights = torch.where(mask, attn_weights, -1e4)
    attn_weights = attn_weights.view(b * n_head, 1, src_s)
    attn_weights = F.softmax(attn_weights, dim=2)
    value = torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)
    value = value.transpose(1, 2).view(b, tgt_s, h)
    at("out")
    value = F.linear(value, w_out, bias=b_out)
    value.add_(inputs)
    return value, k_new, v_new


def mlp(inputs, wi, bi, wo, bo, w_ln, b_ln, at=_noop):
    b, s, h = inputs.shape
    at("ln")
    out = F.layer_norm(inputs, (h,), weight=w_ln, bias=b_ln)
    at("fc1")
    out = F.linear(out, wi, bias=bi)
    F.relu(out, inplace=True)
    at("fc2")
    out = F.linear(out, wo, bias=bo)
    out.add_(inputs)
    return out


# ---- FlexGen's layer list and the weights each layer uses ----

def flexgen_layers(n_blocks: int) -> list[tuple[str, int]]:
    return [("embed", -1)] + [(kind, b) for b in range(n_blocks) for kind in ("attn", "mlp")] + [("head", -1)]


def stage_params(kind: str, b: int) -> list[tuple[str, list[str]]]:
    """A FlexGen layer's weights by use stage, in call order (the ``at`` points)."""
    d = "model.decoder."
    if kind == "embed":
        return [("tok", [d + "embed_tokens.weight"]), ("pos", [d + "embed_positions.weight"])]
    if kind == "head":
        return [("ln", [d + "final_layer_norm.weight", d + "final_layer_norm.bias"]), ("lm", ["lm_head.weight"])]
    p = f"{d}layers.{b}."
    if kind == "attn":
        return [("ln", [p + "self_attn_layer_norm.weight", p + "self_attn_layer_norm.bias"])] + \
               [(m[0] if m != "out_proj" else "out", [p + f"self_attn.{m}.weight", p + f"self_attn.{m}.bias"])
                for m in ("q_proj", "k_proj", "v_proj", "out_proj")]
    return [("ln", [p + "final_layer_norm.weight", p + "final_layer_norm.bias"]),
            ("fc1", [p + "fc1.weight", p + "fc1.bias"]), ("fc2", [p + "fc2.weight", p + "fc2.bias"])]


def layer_params(kind: str, b: int) -> list[str]:
    """The weights a FlexGen layer's compute function takes, in its argument order."""
    d = "model.decoder."
    if kind == "embed":
        return [d + "embed_tokens.weight", d + "embed_positions.weight"]
    if kind == "head":
        return [d + "final_layer_norm.weight", d + "final_layer_norm.bias", "lm_head.weight"]
    p = f"{d}layers.{b}."
    if kind == "attn":
        return [p + f"self_attn.{m}.{t}" for m in ("q_proj", "k_proj", "v_proj", "out_proj") for t in ("weight", "bias")] + \
               [p + "self_attn_layer_norm.weight", p + "self_attn_layer_norm.bias"]
    return [p + n for n in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                            "final_layer_norm.weight", "final_layer_norm.bias")]


class Schedule:
    """Where the executor's flag operations go in FlexGen's loop: hook point
    (layer, micro-batch, stage, phase) -> [(kind, episodes)], plus the copy-list
    rows and the handshake program of that placement (the codegen's sidecar, for
    an eager forward). Points per (layer, micro-batch), in execution order:
    ("layer", "before"), each stage's ("<stage>", "before"), ("layer", "after").
    ``free`` waits per group before its first stage and frees it after its last
    (last micro-batch); the FlexGen orders hold a layer's groups for the whole
    layer, as FlexGen's weight_read_buf does."""

    def __init__(self, plan, layers, n_micro: int, order: str):
        K, last = n_micro, len(layers) - 1
        where_of: dict[str, tuple[int, int]] = {}  # param -> (layer, stage index)
        for j, (kind, b) in enumerate(layers):
            for si, (_, names) in enumerate(stage_params(kind, b)):
                for n in names:
                    where_of[n] = (j, si)
        groups: list[list[int]] = [[] for _ in layers]
        span: dict[int, tuple[int, int]] = {}
        for gid, g in sorted(plan.groups.items()):
            at = sorted(where_of[m.name] for m in g.members)
            if at[0][0] != at[-1][0]:
                raise ValueError(f"group {gid} spans FlexGen layers {at[0][0]} and {at[-1][0]}")
            groups[at[0][0]].append(gid)
            span[gid] = (at[0][1], at[-1][1])
        hooks: dict[tuple, list] = {}

        def add(point, kind, eids):
            if eids:
                hooks.setdefault(point, []).append((kind, sorted(eids)))

        for j, (kind, b) in enumerate(layers):
            stages = [name for name, _ in stage_params(kind, b)]
            nxt = groups[j + 1] if j < last else groups[0]
            if order == "free":
                for gid in groups[j]:
                    first_s, last_s = span[gid]
                    add((j, 0, stages[first_s], "before"), "wait", [gid])
                    end = (j, K - 1, stages[last_s + 1], "before") if last_s + 1 < len(stages) else (j, K - 1, "layer", "after")
                    hooks.setdefault(end, []).insert(0, ("free", [gid]))
                continue
            if order == "flexgen_sync" or K == 1:
                add((j, 0, "layer", "before"), "wait", groups[j])
                add((j, 0, "layer", "before"), "issue", nxt)
            else:  # flexgen_mb
                if j == 0:
                    add((0, 0, "layer", "before"), "wait", groups[0])
                add((j, 0, "layer", "before"), "issue", nxt)
                if j < last:
                    add((j, 1, "layer", "before"), "wait", groups[j + 1])
            add((j, K - 1, "layer", "after"), "free", groups[j])
        self.hooks = hooks
        pos, first, lastu, issued = 0, {}, {}, {}
        self.program = []
        for j, (kind, b) in enumerate(layers):
            points = [("layer", "before")] + [(name, "before") for name, _ in stage_params(kind, b)] + [("layer", "after")]
            for k in range(K):
                for stage, phase in points:
                    for hk, eids in hooks.get((j, k, stage, phase), []):
                        self.program.append([hk, eids])
                        for e in eids:
                            {"wait": first, "free": lastu, "issue": issued}[hk].setdefault(e, pos)
                    pos += 1
        self.rows = []
        for gid in sorted(plan.groups):
            g = plan.groups[gid]
            self.rows.append({"group_id": gid, "episode_id": gid, "slot_id": g.slot_id, "first_use": first[gid],
                              "last_use": lastu[gid], "pos": 0, "source": g.tier.value, "ring_off": g.ring_off,
                              "issue_launch": issued.get(gid, -1),
                              "issue_prev_iter": order != "free" and gid in groups[0]})

    def run(self, registry, point):
        for kind, eids in self.hooks.get(point, []):
            {"wait": registry.cudagraph_wait, "issue": registry.cudagraph_issue,
             "free": registry.cudagraph_free}[kind](eids)


class FlexGenOPT:
    """FlexGen's generation loop for one position (generation_loop_overlap_multi_batch
    compute order, single batch = one micro-batch), over static buffers."""

    def __init__(self, model, n_micro, gbs, P, N, device, registry, schedule):
        cfg = model.config
        self.cfg, self.K, self.gbs, self.P, self.N = cfg, n_micro, gbs, P, N
        self.n_head, h = cfg.num_attention_heads, cfg.hidden_size
        self.p = dict(model.named_parameters())
        self.layers = flexgen_layers(cfg.num_hidden_layers)
        self.registry, self.schedule = registry, schedule
        shape = (P + N - 1, gbs * self.n_head, h // self.n_head)
        self.k_home = [[torch.empty(shape, dtype=torch.float16, device=device) for _ in range(n_micro)]
                       for _ in range(cfg.num_hidden_layers)]
        self.v_home = [[torch.empty(shape, dtype=torch.float16, device=device) for _ in range(n_micro)]
                       for _ in range(cfg.num_hidden_layers)]
        self.masks = [[torch.ones((gbs, P + i), dtype=torch.bool, device=device) for _ in range(n_micro)]
                      for i in range(N)]
        self.pre_ids = [torch.zeros((gbs, P), dtype=torch.int32, device=device) for _ in range(n_micro)]
        self.dec_ids = [torch.zeros((gbs, 1), dtype=torch.int32, device=device) for _ in range(n_micro)]

    def w(self, kind, b):
        return [self.p[n] for n in layer_params(kind, b)]

    def step(self, i):
        K = self.K
        if i > 0:
            for k in range(K):  # update_attention_mask -> extend_attention_mask
                bs = self.masks[i - 1][k].shape[0]
                torch.concat((self.masks[i - 1][k], torch.ones((bs, 1), dtype=torch.bool, device=self.masks[i][k].device)),
                             dim=1, out=self.masks[i][k])
        hidden = [None] * K
        new_kv = {}
        out_ids, out_logits = [None] * K, [None] * K
        for j, (kind, b) in enumerate(self.layers):
            for k in range(K):
                def at(stage, j=j, k=k):
                    self.schedule.run(self.registry, (j, k, stage, "before"))

                self.schedule.run(self.registry, (j, k, "layer", "before"))
                mask = self.masks[i][k]
                if kind == "embed":
                    ids = self.pre_ids[k] if i == 0 else self.dec_ids[k]
                    w_token, w_pos = self.w(kind, b)
                    hidden[k] = opt_input_embed(ids, mask, w_token, w_pos, self.cfg.pad_token_id, at)
                elif kind == "attn":
                    (w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln) = self.w(kind, b)
                    if i == 0:
                        hidden[k], kn, vn = mha(hidden[k], mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
                                                w_ln, b_ln, self.n_head, at)
                    else:
                        hidden[k], kn, vn = mha_gen(hidden[k], mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
                                                    w_ln, b_ln, self.n_head, self.k_home[b][k], self.v_home[b][k], at)
                    new_kv[(j, k)] = (b, kn, vn)
                elif kind == "mlp":
                    hidden[k] = mlp(hidden[k], *self.w(kind, b), at)
                else:
                    out_ids[k], out_logits[k] = opt_output_embed(hidden[k], *self.w(kind, b), at)
                # store_cache(i, j, k - 1): the previous micro-batch, or the previous layer's last
                pj, pk = (j, k - 1) if k > 0 else (j - 1, K - 1)
                if (pj, pk) in new_kv and i != self.N - 1:
                    self.store_cache(i, *new_kv.pop((pj, pk)), pk)
                self.schedule.run(self.registry, (j, k, "layer", "after"))
        return out_ids, out_logits

    def store_cache(self, i, b, k_new, v_new, k):
        if i == 0:
            idx = (slice(0, k_new.shape[0]), slice(0, k_new.shape[1]))
        else:
            pos = self.P + i
            idx = (slice(pos - k_new.shape[0], pos), slice(0, k_new.shape[1]))
        self.k_home[b][k][idx].copy_(k_new)
        self.v_home[b][k][idx].copy_(v_new)


def retarget_to_checkpoint(plan_path: str, crosswalk_path: str) -> list[str]:
    """Plan graph-input names -> checkpoint names (graph 0); returns checkpoint
    names in the original graph-input order (registration order = slot offsets)."""
    with open(plan_path) as f:
        plan = json.load(f)
    with open(crosswalk_path) as f:
        names = json.load(f)["graphs"]["0"]
    for op in plan["io_operations"]:
        if op.get("type") == "vram_prefetch_h2d":
            ck = names[op["compiled_graph_input_name"]]
            op["compiled_graph_input_name"] = op["tensor_name"] = ck
            op["compiled_graph_id"] = 0
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    return [names[a] for a in sorted(names, key=lambda a: int(a[3:].split("_")[0]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiered-load", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--plan-crosswalk", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--gen-len", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--micro-batches", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--issue-order", default="free", choices=ORDERS)
    ap.add_argument("--prefix-ahead", action="store_true")
    ap.add_argument("--ssd-pci", default="")
    ap.add_argument("--ssd-blob", default="")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--vram-limit-gib", type=float, default=0.0)
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--move-trace", default="", help="per-transfer JSONL (SSD read, copy) of every step")
    ap.add_argument("--hang-watchdog", type=float, default=0.0)
    ap.add_argument("--profile-step", type=int, default=-1, help="Kineto kernel list of this decode step")
    ap.add_argument("--dump", default="")
    ap.add_argument("--report-json", default="")
    a = ap.parse_args()
    device = torch.device("cuda")
    if a.vram_limit_gib:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(a.vram_limit_gib * 2**30 / total, torch.cuda.current_device())
    os.makedirs(a.work_dir, exist_ok=True)
    plan_path = os.path.join(a.work_dir, "plan.json")
    shutil.copyfile(a.plan, plan_path)
    order = retarget_to_checkpoint(plan_path, a.plan_crosswalk)
    ws_config.set(ws_config.Config(ssd_pci=a.ssd_pci, ssd_blob=a.ssd_blob, ssd_lanes=a.lanes, timeline=a.timeline,
                                   move_trace=a.move_trace, hang_watchdog_s=a.hang_watchdog))
    plan = load_plan(plan_path)
    registry = WeightRegistry(plan, device)
    model = loader.tiered_load(a.tiered_load, registry, torch.float16, True, "eager")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tiered_load)
    ids = tok(a.prompt).input_ids
    while len(ids) < a.prompt_len:
        ids = ids + ids[1:]
    ids = ids[:a.prompt_len]
    K, B = a.micro_batches, a.gen_batch
    if B % K:
        raise SystemExit("--gen-batch must be a multiple of --micro-batches")
    params = dict(model.named_parameters())
    with torch.no_grad():
        registry.begin_graph_iter(0)
        for name in order + sorted(set(params) - set(order)):
            registry.register_weight_named(0, name, params[name])
    torch.cuda.synchronize()
    layers = flexgen_layers(model.config.num_hidden_layers)
    sched = Schedule(plan, layers, K, a.issue_order)
    copies = CopyList.from_sidecar(sched.rows)
    ex = Executor(registry, copies, device, [sched.program])
    net = FlexGenOPT(model, K, B // K, a.prompt_len, a.gen_len, device, registry, sched)
    for k in range(K):
        net.pre_ids[k].copy_(torch.tensor([ids] * (B // K), dtype=torch.int32))
    kinds = [f"step{i}" for i in range(a.gen_len)]
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, kind in enumerate(kinds):
            ex.capture(lambda i=i: net.step(i), kind=kind, n_warm=2 if i == 0 else 1)
    print(f"→ captured {len(kinds)} positions in {time.perf_counter() - t0:.1f} s; "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak", flush=True)

    runs = []
    dump = []
    for r in range(1 + a.repeats):
        toks, t_dec, logits = [], [], []
        ex.wait()
        t_start = time.perf_counter()
        for i, kind in enumerate(kinds):
            prof = None
            if i == a.profile_step and r == a.repeats:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
            t0 = time.perf_counter()
            p = ex.step(kind, prefix_ahead=a.prefix_ahead and not (r == a.repeats and i == a.gen_len - 1))
            ex.wait()
            dt = time.perf_counter() - t0
            if prof is not None:
                prof.__exit__(None, None, None)
                prof.export_chrome_trace(os.path.join(a.work_dir, f"kineto_step{i}.json"))
            if i == 0:
                t_pre = dt
            else:
                t_dec.append(dt)
            out_ids, out_logits = ex.static_output(p, kind)
            step_ids = torch.cat(out_ids).view(-1)
            toks.append(step_ids.tolist())
            if r == a.repeats:
                logits.append(torch.cat(out_logits).float().cpu())
            for k in range(K):
                net.dec_ids[k].copy_(out_ids[k])
        runs.append({"prefill_ms": t_pre * 1e3, "decode_ms_per_token": sum(t_dec) / len(t_dec) * 1e3,
                     "e2e_ms": (time.perf_counter() - t_start) * 1e3, "tokens": toks})
        print(f"→ [gen] run {r}{' (warm)' if r == 0 else ''}: prefill {runs[-1]['prefill_ms']:.1f} ms, decode "
              f"{runs[-1]['decode_ms_per_token']:.1f} ms/token, tokens {[t[0] for t in toks]}", flush=True)
        dump = logits
    timed = runs[1:] or runs
    report = {k: sum(x[k] for x in timed) / len(timed) for k in ("prefill_ms", "decode_ms_per_token", "e2e_ms")}
    report.update(tokens=runs[-1]["tokens"], tokens_identical_across_runs=all(x["tokens"] == runs[0]["tokens"] for x in runs),
                  micro_batches=K, gen_batch=B, issue_order=a.issue_order, peak_mb=torch.cuda.max_memory_allocated() / 1e6)
    report.update(ex.report())
    if a.dump:
        torch.save({"tokens": runs[-1]["tokens"], "logits": dump}, a.dump)
    if a.report_json:
        with open(a.report_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
    print("→ " + json.dumps({k: v for k, v in report.items() if k not in ("ssd", "tokens")}, default=str)[:600], flush=True)


if __name__ == "__main__":
    main()
