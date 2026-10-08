"""Instrument one compiled graph's wrapper for captured weight streaming.

Called from ``PythonWrapperCodegen.generate`` when ``config.weight_streaming_plan``
is set. For each group of the plan that this graph uses, the residency walk over
the graph's own kernel launches gives the first and the last launch that reads
any of its members. The wrapper then gets:

* a registration block, run on the graph's first call: every static input is
  named to the weight registry, which repoints streamed ones into the arena;
* ``cudagraph_wait([group])`` before the group's first-use launch (READY);
* ``cudagraph_free([group])`` after its last-use launch (FREED);
* ``cudagraph_issue([group])`` where the configured issue order starts its
  load (``issue_order.py``; none when free-running);
* with a spill plan, for each mutated static input (the KV caches; the
  registry maps a spilled layer's K and V to its site): ``kv_bind()`` at the
  top of every call (KV homed on SSD: the caches into their slots before any
  kernel), ``spill_wait`` before the first use (KVREADY), ``spill_produced``
  after the last (PRODUCED); under a flexgen order also the plan's timing:
  ``spill_issue`` where FlexGen's ``load_cache`` starts (the anchor the layer's
  weights get; the wait moves to its barrier as theirs do), and for the store,
  ``spill_store_issue`` at the start of the next sub-layer and
  ``spill_store_wait`` (on the drive) at the barrier after it, as FlexGen's
  ``store_cache(j - 1)`` after ``compute(j)`` and its ``sync()``;

and one copy-list row per group is written next to the plan
(``<plan>.copylist<pos>.json``) for the executor. A group is one residency
interval per iteration (its episode keeps the group id); the plan's reload
anchors are not used, because a second graph compiled against the same plan
(prefill next to decode) has its own launch numbering.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING

from torch._inductor.codegen.wrapper import (
    ExternKernelAllocLine,
    ExternKernelMultiOutLine,
    ExternKernelOutLine,
    KernelCallLine,
    WrapperLine,
)
from torch._inductor.ir import TensorBox
from torch._inductor.virtualized import V
from gpuco_torch import config as ws_config
from torch._inductor.weight_streaming.codegen.issue_order import plan_issues, sublayer_starts
from gpuco_torch.plan import load_plan


if TYPE_CHECKING:
    from torch._inductor.codegen.common import IndentedBuffer
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen


log = logging.getLogger(__name__)

COMPUTE_LINES = (KernelCallLine, ExternKernelOutLine, ExternKernelAllocLine, ExternKernelMultiOutLine)

# Position of the graph being compiled among the forward's graphs; the plan's
# graph ids and the registration keys use it. Reset before each full compile.
_compile_position = 0


def reset_compile_position() -> None:
    global _compile_position
    _compile_position = 0


@dataclasses.dataclass
class StreamingLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    calls: list[str]

    def codegen(self, code: IndentedBuffer) -> None:
        for line in self.calls:
            code.writeline(line)


def copy_list_path(plan_path: str, pos: int) -> str:
    return f"{plan_path}.copylist{pos}.json"


def _static_weight_inputs() -> list[str]:
    """Graph inputs that keep their address across calls (parameters, buffers)."""
    static = set(V.graph.static_input_idxs)
    return [name for i, name in enumerate(V.graph.graph_input_names)
            if i in static and isinstance(V.graph.graph_inputs.get(name), TensorBox)]


def _input_usage(lines: list) -> dict[str, list[int]]:
    """graph input name -> sorted launch ids of the kernels that read it."""
    names = set(V.graph.graph_input_names)
    if not names:
        return {}
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    usage: dict[str, set[int]] = {}
    for line in lines:
        if type(line) not in COMPUTE_LINES or line.launch_id < 0:
            continue
        if isinstance(line, KernelCallLine):
            args = list(line.call_args)
        else:
            args = [*line.node.codegen_args(), *line.node.codegen_kwargs()]
        for a in args:
            if isinstance(a, str):
                for m in pattern.finditer(a):
                    usage.setdefault(m.group(1), set()).add(line.launch_id)
    return {n: sorted(v) for n, v in usage.items()}


def _self_copies(lines: list) -> dict[str, set[int]]:
    """input name -> launches that only copy it onto itself: the copy-back Inductor
    emits at the end of a graph for a mutated input it already updated in place
    (``triton_poi_fused_copy_...(argN_1, argN_1, numel)``). They move no data, so
    they are not a use when placing a spilled cache's PRODUCED."""
    names = set(V.graph.graph_input_names)
    out: dict[str, set[int]] = {}
    for line in lines:
        if not isinstance(line, KernelCallLine) or line.launch_id < 0:
            continue
        refs = [a for a in line.call_args if isinstance(a, str) and a.isidentifier()]
        if len(refs) >= 2 and len(set(refs)) == 1 and refs[0] in names:
            out.setdefault(refs[0], set()).add(line.launch_id)
    return out


def usage_launches(lines: list) -> list[int]:
    return [line.launch_id for line in lines if type(line) in COMPUTE_LINES]


def inject(wrapper: PythonWrapperCodegen) -> None:
    from torch._inductor import config

    global _compile_position
    pos = _compile_position
    _compile_position += 1
    plan = load_plan(config.weight_streaming_plan)

    usage = _input_usage(wrapper.lines)
    episodes: dict[int, tuple[int, int]] = {}  # group -> (first use, last use)
    for g in plan.groups_in_graph(pos):
        uses = sorted({lid for m in g.members if m.graph_id in (pos, -1) for lid in usage.get(m.name, ())})
        if uses:
            episodes[g.group_id] = (uses[0], uses[-1])
    frees: dict[int, list[int]] = {}
    for gid, (_first, last) in episodes.items():
        frees.setdefault(last, []).append(gid)
    cfg = ws_config.get()
    issues = plan_issues(cfg.issue_order, cfg.issue_map, wrapper, COMPUTE_LINES, episodes,
                         {gid: plan.groups[gid].slot_id for gid in episodes})
    waits = issues.waits

    copy_list = sorted(
        ({"group_id": gid, "episode_id": gid, "slot_id": plan.groups[gid].slot_id, "last_use": last,
          "first_use": first, "pos": pos, "source": plan.groups[gid].tier.value,
          "ring_off": plan.groups[gid].ring_off,
          "issue_launch": issues.meta.get(gid, (-1, False))[0],
          "issue_prev_iter": issues.meta.get(gid, (-1, False))[1]}
         for gid, (first, last) in episodes.items()),
        key=lambda e: (e["last_use"], e["group_id"]))
    # The wait/free placement, as a function of the plan and the residency walk only.
    digest = hashlib.sha1(repr((
        sorted((k, sorted(v)) for k, v in waits.items()),
        sorted((k, sorted(v)) for k, v in frees.items()),
    )).encode()).hexdigest()[:16]
    # The flag operations in emission order, for the pre-capture handshake check.
    program = []
    for lid in sorted(usage_launches(wrapper.lines)):
        for kind, table in (("issue", issues.before_wait), ("wait", waits), ("issue", issues.after_wait),
                            ("free", frees)):
            if table.get(lid):
                program.append([kind, sorted(table[lid])])
    path = copy_list_path(config.weight_streaming_plan, pos)
    with open(path, "w") as f:
        json.dump({"compile_position": pos, "graph_digest": digest, "copy_list": copy_list,
                   "program": program}, f)
    log.warning("[weight streaming] graph %d: %d groups over %d slots, digest %s, copy list %s", pos,
                len(episodes), len({plan.groups[g].slot_id for g in episodes}), digest, path)

    wrapper.header.writeline("from gpuco_torch.runtime.registry import active_registry")
    wrapper.header.writeline("_ws_rt = active_registry()")
    inputs = _static_weight_inputs()
    new_lines: list = []
    if inputs:
        calls = [f"_ws_ref_{name} = {name}" for name in inputs]
        if cfg.spill_plan:
            calls.append("_ws_rt.kv_bind()")
        calls.append(f"if _ws_rt.begin_graph_iter({pos}) == 0:")
        calls += [f"    _ws_rt.register_weight_named({pos}, {name!r}, _ws_ref_{name})" for name in inputs]
        new_lines.append(StreamingLine(wrapper, calls))

    # launch -> spill hook calls: before its waits, with its waits, after its waits, after the launch
    kv_before: dict[int, list[str]] = {}
    kv_wait: dict[int, list[str]] = {}
    kv_after: dict[int, list[str]] = {}
    produced: dict[int, list[str]] = {}
    if cfg.spill_plan:
        self_copies = _self_copies(wrapper.lines)
        spilled = []
        for name in inputs:
            if name in V.graph.mutated_inputs and usage.get(name):
                real = [lid for lid in usage[name] if lid not in self_copies.get(name, ())] or usage[name]
                produced.setdefault(real[-1], []).append(f"_ws_rt.spill_produced(_ws_ref_{name})")
                spilled.append((name, real[0]))
        if cfg.issue_order == "free":
            for name, first in spilled:
                kv_wait.setdefault(first, []).append(f"_ws_rt.spill_wait(_ws_ref_{name})")
        elif cfg.issue_order in ("flexgen", "flexgen_sync"):
            pseudo = {i: (first, first) for i, (_n, first) in enumerate(spilled)}
            kv = plan_issues(cfg.issue_order, cfg.issue_map, wrapper, COMPUTE_LINES, pseudo,
                             {i: -1 - i for i in pseudo})
            for lid, ids in kv.waits.items():
                kv_wait.setdefault(lid, []).extend(f"_ws_rt.spill_wait(_ws_ref_{spilled[i][0]})" for i in ids)
            for table, out in ((kv.before_wait, kv_before), (kv.after_wait, kv_after)):
                for lid, ids in table.items():
                    out.setdefault(lid, []).extend(
                        f"_ws_rt.spill_issue(_ws_ref_{spilled[i][0]}, {kv.meta[i][1]})" for i in ids)
            sub_of, first_of_sub = sublayer_starts(wrapper, COMPUTE_LINES)
            for name, first in spilled:
                sb = sub_of[first]
                if sb + 2 not in first_of_sub:
                    raise RuntimeError(f"spill: no sub-layer after the one following {name}'s (sub-layer {sb})")
                kv_after.setdefault(first_of_sub[sb + 1], []).append(f"_ws_rt.spill_store_issue(_ws_ref_{name})")
                kv_wait.setdefault(first_of_sub[sb + 2], []).append(f"_ws_rt.spill_store_wait(_ws_ref_{name})")
        else:
            raise NotImplementedError(f"spill plans under issue order {cfg.issue_order!r}")

    def emit(call: str, eids: list[int] | None) -> None:
        if eids:
            new_lines.append(StreamingLine(wrapper, [f"_ws_rt.{call}({sorted(eids)!r})"]))

    for line in wrapper.lines:
        lid = line.launch_id if type(line) in COMPUTE_LINES else None
        if lid is not None:
            emit("cudagraph_issue", issues.before_wait.get(lid))
            for table in (kv_before, kv_wait):
                if table.get(lid):
                    new_lines.append(StreamingLine(wrapper, table[lid]))
            emit("cudagraph_wait", waits.get(lid))
            emit("cudagraph_issue", issues.after_wait.get(lid))
            if kv_after.get(lid):
                new_lines.append(StreamingLine(wrapper, kv_after[lid]))
        new_lines.append(line)
        if lid is not None:
            emit("cudagraph_free", frees.get(lid))
            if produced.get(lid):
                new_lines.append(StreamingLine(wrapper, produced[lid]))
    wrapper.lines = new_lines
