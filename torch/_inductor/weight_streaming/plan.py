from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field


@dataclass
class TensorEntry:
    name: str
    kind: str  # "WEIGHT" or "CONTEXT"
    size_bytes: int
    dtype: str
    shape: list[int]
    numel: int = 0
    file: str = ""
    file_offset: int = 0


@dataclass
class PrefetchOp:
    """SSD->DRAM prefetch (type="prefetch" in scheduler io_operations)."""
    tensor_name: str
    before_node: int  # trace node index where tensor is consumed
    after_node: int  # trace node index after which prefetch can start (-1 = eager)
    eager_start: bool = False
    before_launch_id: int = -1


@dataclass
class H2DPrefetchOp:
    """DRAM->VRAM prefetch (type="vram_prefetch_h2d" in scheduler io_operations).

    Placement uses two anchors when the scheduler provides early-start timing:
      - after_launch_id: where the async H2D copy is issued (early start)
      - before_launch_id: where the consumer waits for the copy to finish
    When only before_launch_id is set, the H2D is issued synchronously there.

    cross_iter=True means the op is a cyclic reload: iter N's post-ops at
    after_launch_id feed iter N+1's pre-ops at before_launch_id. For these,
    after_launch_id may equal or exceed before_launch_id; codegen treats them
    as always async.
    """
    tensor_name: str
    before_node: int
    before_launch_id: int = -1
    after_launch_id: int = -1  # early-start anchor from scheduler
    compiled_tensor_id: int = -1  # graph-local, from tensor_map sidecar
    cross_iter: bool = False


@dataclass
class EvictVramOp:
    """VRAM->DRAM eviction (type="vram_evict_d2h" in scheduler io_operations)."""
    tensor_name: str
    after_node: int
    after_launch_id: int = -1
    compiled_tensor_id: int = -1  # graph-local, from tensor_map sidecar


@dataclass
class EvictDramOp:
    """DRAM eviction (from scheduler spill_decisions)."""
    tensor_name: str
    evict_after_node: int
    after_launch_id: int = -1


@dataclass
class ColdStartOp:
    tensor_name: str
    attach_before_node: int
    before_launch_id: int = -1


@dataclass
class IOSchedule:
    nodes: list[dict] = field(default_factory=list)
    prefetches: list[PrefetchOp] = field(default_factory=list)
    h2d_prefetches: list[H2DPrefetchOp] = field(default_factory=list)
    evict_vram: list[EvictVramOp] = field(default_factory=list)
    evict_dram: list[EvictDramOp] = field(default_factory=list)
    cold_starts: list[ColdStartOp] = field(default_factory=list)
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    compilation_hash: str = ""


def _load_tensors_csv(path: str) -> dict[str, TensorEntry]:
    """Load pytorch_runtime_tensors.csv and return {tensor_name: TensorEntry}."""
    result: dict[str, TensorEntry] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["tensor_name"]
            if name in result:
                continue
            shape_str = row.get("shape", "[]")
            shape = [int(x) for x in shape_str.strip("[]").split(",") if x.strip()]
            numel = int(row.get("numel", 0))
            result[name] = TensorEntry(
                name=name,
                kind=row.get("tensor_kind", "WEIGHT"),
                size_bytes=int(row.get("tensor_size_bytes", 0)),
                dtype=row.get("dtype", ""),
                shape=shape,
                numel=numel,
            )
    return result


def _load_nodes_csv(path: str) -> dict[int, str]:
    """Load runtime_nodes.csv and return {node_idx: resource_kind}."""
    result: dict[int, str] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            result[idx] = row.get("resource_kind", "")
    return result


def load_io_schedule(
    path: str, nodes_csv: str = "", tensor_csv: str = ""
) -> IOSchedule:
    """Load scheduler JSON and optional runtime_nodes.csv / tensor CSV.

    The scheduler outputs:
      - io_operations: list of dicts with type "prefetch", "vram_prefetch_h2d",
        or "vram_evict_d2h"
      - spill_decisions: list of dicts with evict_after_node (DRAM evictions)
      - cold_start_prefetches: list of dicts with attach_before_node

    When tensor_csv is provided, tensor metadata is loaded from the CSV and
    VRAM ops targeting CONTEXT tensors (computed intermediates) are filtered out.
    """
    with open(path) as f:
        data = json.load(f)

    nodes = data.get("nodes", [])

    # If nodes don't have resource_kind and we have a CSV, enrich them
    if nodes_csv and nodes and "resource_kind" not in nodes[0]:
        csv_kinds = _load_nodes_csv(nodes_csv)
        for node in nodes:
            idx = node.get("idx", -1)
            if idx in csv_kinds:
                node["resource_kind"] = csv_kinds[idx]

    prefetches: list[PrefetchOp] = []
    h2d_prefetches: list[H2DPrefetchOp] = []
    evict_vram_ops: list[EvictVramOp] = []

    for op in data.get("io_operations", []):
        op_type = op.get("type", "")
        if op_type == "prefetch":
            prefetches.append(PrefetchOp(
                tensor_name=op["tensor_name"],
                before_node=op["before_node"],
                after_node=op.get("after_node", -1),
                eager_start=op.get("eager_start", False),
                before_launch_id=op.get("before_launch_id", -1),
            ))
        elif op_type == "vram_prefetch_h2d":
            h2d_prefetches.append(H2DPrefetchOp(
                tensor_name=op["tensor_name"],
                before_node=op["before_node"],
                before_launch_id=op.get("before_launch_id", -1),
                after_launch_id=op.get("after_launch_id", -1),
                compiled_tensor_id=op.get("compiled_tensor_id", -1),
                cross_iter=op.get("reason") == "h2d_cross_iter_reload",
            ))
        elif op_type == "vram_evict_d2h":
            evict_vram_ops.append(EvictVramOp(
                tensor_name=op["tensor_name"],
                after_node=op["after_node"],
                after_launch_id=op.get("after_launch_id", -1),
                compiled_tensor_id=op.get("compiled_tensor_id", -1),
            ))

    evict_dram_ops: list[EvictDramOp] = []
    for s in data.get("spill_decisions", []):
        evict_dram_ops.append(EvictDramOp(
            tensor_name=s["tensor_name"],
            evict_after_node=s["evict_after_node"],
            after_launch_id=s.get("after_launch_id", -1),
        ))

    cold_starts: list[ColdStartOp] = []
    for c in data.get("cold_start_prefetches", []):
        cold_starts.append(ColdStartOp(
            tensor_name=c["tensor_name"],
            attach_before_node=c.get("attach_before_node", 0),
            before_launch_id=c.get("before_launch_id", -1),
        ))

    # Load tensor metadata from CSV (preferred) or JSON fallback
    tensors: dict[str, TensorEntry] = {}
    if tensor_csv:
        tensors = _load_tensors_csv(tensor_csv)
    else:
        for name, t in data.get("tensor_metadata", {}).items():
            shape = t.get("shape", [])
            numel = 1
            for s in shape:
                numel *= s
            tensors[name] = TensorEntry(
                name=name,
                kind=t.get("kind", "WEIGHT"),
                size_bytes=t.get("size_bytes", 0),
                dtype=t.get("dtype", ""),
                shape=shape,
                numel=numel,
                file=t.get("safetensors_file", ""),
                file_offset=t.get("safetensors_offset", 0),
            )

    # Filter VRAM ops: skip CONTEXT tensors (computed intermediates already in VRAM)
    def _is_weight(name: str) -> bool:
        entry = tensors.get(name)
        return entry is None or entry.kind == "WEIGHT"

    if tensors:
        h2d_prefetches = [op for op in h2d_prefetches if _is_weight(op.tensor_name)]
        evict_vram_ops = [op for op in evict_vram_ops if _is_weight(op.tensor_name)]

    return IOSchedule(
        nodes=nodes,
        prefetches=prefetches,
        h2d_prefetches=h2d_prefetches,
        evict_vram=evict_vram_ops,
        evict_dram=evict_dram_ops,
        cold_starts=cold_starts,
        tensors=tensors,
        compilation_hash=data.get("compilation_hash", ""),
    )
