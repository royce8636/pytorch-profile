"""Issue-order policies: when each group's load may start, as in-graph ISSUE
writes the copy engine waits on. ``free`` emits none. The others replay another
system's loop relative to the compute:

* ``flexgen``: FlexGen issues sub-layer j+1's loads at the start of sub-layer j
  (an OPT block is two FlexGen layers, attention and MLP; the next token's input
  embedding is issued at the start of this token's output layer). The ISSUE is
  written before the previous sub-layer's first wait: two sub-layers in flight.
* ``flexgen_sync``: FlexGen's loop exactly, ``sync(); load(j+1); compute(j)``.
  Every wait of sub-layer j moves to its first kernel (the barrier) and j+1's
  ISSUE follows it: one sub-layer in flight.
* ``flexgen_mb``: ``flexgen_sync`` under FlexGen's block schedule (several GPU
  batches per layer): its ``sync()`` follows every micro-batch, and sub-layer
  j+1's whole load is issued with j's first one, so j+1's waits move to the
  start of j's second micro-batch pass; the rest of j runs with nothing in
  flight. One pass per sub-layer is ``flexgen_sync``.
* ``at_use``: HF Accelerate's hooks fetch a module's tensors in its pre_forward,
  so each group is issued at its own first use, before its wait.
* ``accel``: Accelerate, anchored where its own measured run started each read
  (``issue_map``: per graph, consumer module -> the leaf module the GPU was
  running); written after the anchor's wait, one transfer in flight. An anchor
  at or after the consumer's first launch falls back to ``at_use``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any

from torch._inductor.virtualized import V
from torch._inductor.weight_streaming.codegen.module_stack import stacks_by_launch


log = logging.getLogger(__name__)


@dataclasses.dataclass
class IssuePlan:
    waits: dict[int, list[int]]  # launch -> episodes waited before it (possibly moved to a barrier)
    before_wait: dict[int, list[int]]  # launch -> ISSUE writes emitted before its waits
    after_wait: dict[int, list[int]]  # launch -> ISSUE writes emitted after its waits
    meta: dict[int, tuple[int, bool]]  # episode -> (issuing launch, issued by the previous iteration)


def _sublayers(stacks: dict[int, list]) -> dict[int, int]:
    """launch -> FlexGen sub-layer: 2b (attention) and 2b+1 (MLP) of block b, -1
    before the first block (input embedding), 2*last+2 after it (output layer).
    Launches under neither (views, residual adds on the block itself) continue
    the sub-layer of the launch before them, as FlexGen's attention layer ends
    with its residual add; cache write-backs after the head keep their block's
    label, which moves no anchor (anchors take a sub-layer's first launch)."""
    blk_of: dict[int, Any] = {}
    for lid, st in stacks.items():
        names = [pair[0] for pair in st if pair]
        blk_of[lid] = next((int(m.group(1)) for nm in names
                            if (m := re.search(r"\.layers\.(\d+)(\.|$)", nm))), None)
    last_blk = max((b for b in blk_of.values() if b is not None), default=-1)
    head = 2 * last_blk + 2
    sub_of: dict[int, int] = {}
    prev = -1
    for lid in sorted(stacks):
        names = [pair[0] for pair in stacks[lid] if pair]
        blk = blk_of[lid]
        if blk is None:
            if names and prev in (2 * last_blk + 1, head):
                prev = head
        elif any("self_attn" in n for n in names):
            prev = 2 * blk
        elif any(k in n for n in names for k in ("fc1", "fc2", "final_layer_norm")):
            prev = 2 * blk + 1
        elif prev not in (2 * blk, 2 * blk + 1):
            prev = 2 * blk
        sub_of[lid] = prev
    return sub_of


def sublayer_starts(wrapper: Any, compute_lines: tuple) -> tuple[dict[int, int], dict[int, int]]:
    """(launch -> FlexGen sub-layer, sub-layer -> its first launch), as the flexgen
    orders anchor loads."""
    sub_of = _sublayers(stacks_by_launch(wrapper, compute_lines))
    first_of_sub: dict[int, int] = {}
    for lid, sb in sub_of.items():
        first_of_sub[sb] = min(lid, first_of_sub.get(sb, lid))
    return sub_of, first_of_sub


def _second_passes(stacks: dict[int, list], sub_of: dict[int, int]) -> dict[int, int]:
    """sub-layer -> first launch of its second micro-batch pass: where the leaf
    module of the sub-layer's first launch starts again. Absent with one pass."""
    leaf = {lid: next((pair[0] for pair in reversed(st or []) if pair), "") for lid, st in stacks.items()}
    first_leaf: dict[int, str] = {}
    out: dict[int, int] = {}
    prev_leaf: dict[int, str] = {}
    for lid in sorted(sub_of):
        sb = sub_of[lid]
        if sb not in first_leaf:
            first_leaf[sb] = leaf.get(lid, "")
        elif sb not in out and first_leaf[sb] and leaf.get(lid) == first_leaf[sb] != prev_leaf[sb]:
            out[sb] = lid
        prev_leaf[sb] = leaf.get(lid, "")
    return out


def plan_issues(mode: str, issue_map: str, wrapper: Any, compute_lines: tuple,
                episodes: dict[int, tuple[int, int]], slot_of: dict[int, int]) -> IssuePlan:
    waits: dict[int, list[int]] = {}
    for eid, (first, _last) in episodes.items():
        waits.setdefault(first, []).append(eid)
    before: dict[int, list[int]] = {}
    after: dict[int, list[int]] = {}
    meta: dict[int, tuple[int, bool]] = {}
    if mode == "free":
        return IssuePlan(waits, before, after, meta)

    if mode == "at_use":
        for eid, (first, _last) in episodes.items():
            before.setdefault(first, []).append(eid)
            meta[eid] = (first, False)
        return IssuePlan(waits, before, after, meta)

    stacks = stacks_by_launch(wrapper, compute_lines)
    if mode == "accel":
        amap_all = json.load(open(issue_map))
        # Keyed by the compile's graph id (decode and prefill share a compile position).
        amap = amap_all.get(str(V.graph.graph_id))
        if amap is None:
            raise RuntimeError(f"issue map has no entry for graph_id {V.graph.graph_id} (keys {sorted(amap_all)})")
        leaf_of = {lid: next((pair[0] for pair in reversed(st or []) if pair), "") for lid, st in stacks.items()}
        first_of_mod: dict[str, int] = {}
        for lid in sorted(stacks):
            first_of_mod.setdefault(leaf_of[lid], lid)
        unmapped = sorted({leaf_of.get(first, "") for first, _ in episodes.values()} - set(amap))
        if unmapped:
            raise RuntimeError(f"issue map graph {V.graph.graph_id}: no anchor for consumer modules {unmapped[:6]}")
        for eid, (first, _last) in episodes.items():
            anchor = first_of_mod.get(amap[leaf_of.get(first, "")], first)
            if anchor >= first:
                before.setdefault(first, []).append(eid)
                meta[eid] = (first, False)
            else:
                after.setdefault(anchor, []).append(eid)
                meta[eid] = (anchor, False)
        log.warning("[weight streaming] ISSUE anchors (accel, graph %s): %d episodes, %d at their own first use",
                    V.graph.graph_id, len(meta), sum(1 for e, (a, _) in meta.items() if a == episodes[e][0]))
        return IssuePlan(waits, before, after, meta)

    # flexgen / flexgen_sync
    sub_of = _sublayers(stacks)
    consumer_of_sub: dict[int, int] = {}
    for lid in waits:
        sb = sub_of.get(lid)
        if sb is not None:
            consumer_of_sub[sb] = min(lid, consumer_of_sub.get(sb, lid))
    first_of_sub: dict[int, int] = {}
    for lid, sb in sub_of.items():
        first_of_sub[sb] = min(lid, first_of_sub.get(sb, lid))
    subs = sorted(first_of_sub)
    target_map = before
    if mode in ("flexgen_sync", "flexgen_mb"):
        barrier = {lid: first_of_sub[sub_of[lid]] for lid in waits if lid in sub_of}
        if mode == "flexgen_mb":
            second = _second_passes(stacks, sub_of)
            pos = {sb: i for i, sb in enumerate(subs)}
            for lid in barrier:
                i = pos[sub_of[lid]]
                if i > 0 and subs[i - 1] in second:
                    barrier[lid] = second[subs[i - 1]]
            log.warning("[weight streaming] flexgen_mb: %d of %d sub-layers have a second micro-batch pass",
                        len(second), len(subs))
        # A wait moved to its sub-layer's barrier sits before the consumer, so the
        # group's copy must not need a slot freed after the barrier.
        by_slot: dict[int, list] = {}
        for eid, (first, last) in episodes.items():
            by_slot.setdefault(slot_of[eid], []).append((last, first, eid))
        for slot, eps in by_slot.items():
            eps.sort()
            for (p_last, _pf, p_eid), (_gl, g_first, g_eid) in zip(eps, eps[1:]):
                if p_last >= barrier.get(g_first, g_first):
                    raise RuntimeError(
                        f"flexgen_sync: episode {g_eid} would wait at its sub-layer barrier (launch "
                        f"{barrier.get(g_first)}) for slot {slot}, which episode {p_eid} frees only after "
                        f"launch {p_last}; the barrier would deadlock this slot coloring")
        moved: dict[int, list[int]] = {}
        for lid, eids in waits.items():
            moved.setdefault(barrier.get(lid, lid), []).extend(eids)
        waits = moved
        consumer_of_sub = dict(first_of_sub)
        target_map = after
    for eid, (first, _last) in episodes.items():
        target = sub_of.get(first, subs[0]) - 1
        prev_iter = target < subs[0]
        if prev_iter:
            target = subs[-1]
        anchor = consumer_of_sub.get(target, first_of_sub.get(target, first_of_sub[subs[0]]))
        target_map.setdefault(anchor, []).append(eid)
        meta[eid] = (anchor, prev_iter)
    log.warning("[weight streaming] ISSUE anchors (%s): %d sub-layers, %d groups gated, %d from the previous iteration",
                mode, len(subs), len(meta), sum(1 for v in meta.values() if v[1]))
    return IssuePlan(waits, before, after, meta)
