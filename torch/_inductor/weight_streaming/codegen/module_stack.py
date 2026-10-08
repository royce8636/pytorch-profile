"""Kernel -> nn.Module attribution, for issue orders that follow another
system's per-module loop. A fused kernel is attributed to the deepest module
all its FX origins share."""
from __future__ import annotations

import re
from typing import Any

import torch


Stack = list[tuple[str, str]]  # (qualified module name, module type), outermost first


def fx_node_stack(node: torch.fx.Node) -> Stack:
    from torch._inductor.fx_passes.graph_view import _clean_stack_name, _get_module_stack

    out: Stack = []
    for raw_name, module_class in _get_module_stack(node):
        # Dynamo sometimes records dot-form ``_modules.x`` the base cleaner leaves.
        cleaned = re.sub(r"(?:^|(?<=\.))_modules\.", "", _clean_stack_name(raw_name))
        type_name = (f"{module_class.__module__}.{module_class.__qualname__}"
                     if isinstance(module_class, type) else str(module_class))
        out.append((cleaned, type_name))
    return out


def common_stack(stacks: list[Stack]) -> Stack:
    stacks = [s for s in stacks if s]
    if not stacks:
        return []
    prefix = stacks[0]
    for other in stacks[1:]:
        n = 0
        for a, b in zip(prefix, other):
            if a != b:
                break
            n += 1
        prefix = prefix[:n]
    return prefix


def _ir_stack(ir_node: Any) -> Stack:
    origin = getattr(ir_node, "origin_node", None)
    nodes = [origin] if origin is not None else list(getattr(ir_node, "origins", None) or ())
    return common_stack([fx_node_stack(n) for n in nodes])


def scheduled_kernel_stack(node_schedule: Any) -> Stack:
    """The stack of a kernel about to be emitted from these scheduler nodes."""
    from torch._inductor.codegen.simd_kernel_features import DisableReduction, EnableReduction
    from torch._inductor.scheduler import BaseSchedulerNode

    stacks = []
    for snode in node_schedule if isinstance(node_schedule, (list, tuple)) else [node_schedule]:
        if snode in (EnableReduction, DisableReduction):
            continue
        ir_node = getattr(snode, "node", None)
        if ir_node is None and isinstance(snode, BaseSchedulerNode):
            continue
        stacks.append(_ir_stack(ir_node if ir_node is not None else snode))
    return common_stack(stacks)


def stacks_by_launch(wrapper: Any, compute_lines: tuple) -> dict[int, Stack]:
    """launch id -> stack, for every kernel launch of the graph."""
    from torch._inductor.codegen.wrapper import KernelCallLine

    out = {}
    for line in wrapper.lines:
        if type(line) not in compute_lines:
            continue
        if isinstance(line, KernelCallLine):
            out[line.launch_id] = wrapper._ws_launch_module_stacks.get(line.launch_id, [])
        else:
            out[line.launch_id] = _ir_stack(line.node)
    return out
