#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_VERSIONS = ("v0", "v1", "v2", "v3", "v4", "v5", "v6")
SUPPORTED_SUMMARY_VIEWS = ("compact", "split")


@dataclass(frozen=True)
class BundlePaths:
    bundle_dir: Path
    manifest_path: Path
    node_csv: Path
    edge_csv: Path
    timing_csv: Path


@dataclass(frozen=True)
class NodeRecord:
    raw: dict[str, str]
    step: int
    node_id: str
    node_n: int | None
    node_name: str
    op_name: str
    runtime_role: str
    node_kind: str
    resource_kind: str
    resource_id: str
    device_type: str
    device_index: int | None
    duration_ns: int

    @property
    def identity_text(self) -> str:
        return f"{self.node_name} {self.op_name}".strip().lower()


@dataclass(frozen=True)
class EdgeRecord:
    src_node_id: str
    dst_node_id: str
    edge_kind: str


@dataclass(frozen=True)
class NodeAttribution:
    version: str
    node_id: str
    classification: str
    pure_compute_time_ns: int
    ssd_time_ns: int
    reason: str


@dataclass(frozen=True)
class VersionReport:
    version: str
    raw_total_ns: int
    raw_total_ns_cpu: int
    raw_total_ns_gpu: int
    pure_compute_ns: int
    pure_compute_ns_cpu: int
    pure_compute_ns_gpu: int
    ssd_time_ns: int
    ssd_time_ns_cpu: int
    ssd_time_ns_gpu: int
    node_count: int
    compute_node_count: int
    stall_node_count: int
    classification_counts: dict[str, int]


@dataclass(frozen=True)
class BundleAnalysis:
    paths: BundlePaths
    nodes: list[NodeRecord]
    nodes_by_id: dict[str, NodeRecord]
    incoming_edges: dict[str, list[EdgeRecord]]
    outgoing_edges: dict[str, list[EdgeRecord]]


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def resolve_versions(version: str) -> list[str]:
    if version == "all":
        return list(SUPPORTED_VERSIONS)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported version {version!r}; expected one of {SUPPORTED_VERSIONS} or 'all'"
        )
    return [version]


def include_v0_version(versions: list[str]) -> list[str]:
    deduped = list(dict.fromkeys(versions))
    if "v0" in deduped:
        deduped.remove("v0")
    return ["v0", *deduped]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a llama bundle and split imported profile time into pure "
            "compute time and SSD/stall time using several heuristic versions."
        )
    )
    parser.add_argument(
        "llama_bundle",
        help=(
            "Path to a llama bundle directory or its manifest.json. The bundle "
            "must contain runtime_nodes.csv, runtime_edges.csv, and "
            "ggml_profile_node_records.csv."
        ),
    )
    parser.add_argument(
        "--version",
        choices=(*SUPPORTED_VERSIONS, "all"),
        default="v6",
        help="Which heuristic version to run. Defaults to v6.",
    )
    parser.add_argument(
        "--compare-bundle",
        default=None,
        help=(
            "Optional second llama bundle to compare against the primary one. "
            "When set, the script prints top raw deltas and the largest CPU "
            "nodes still retained as compute under the chosen version."
        ),
    )
    parser.add_argument(
        "--compare-limit",
        type=int,
        default=25,
        help="Maximum number of compare rows to print. Defaults to 25.",
    )
    parser.add_argument(
        "--compare-include-retained-compute",
        action="store_true",
        help=(
            "Include the slower section that attributes the primary bundle under "
            "the chosen version and prints the largest CPU nodes still retained as compute."
        ),
    )
    parser.add_argument(
        "--summary-view",
        choices=SUPPORTED_SUMMARY_VIEWS,
        default="split",
        help=(
            "How to print the stdout summary. 'compact' keeps the old totals-only "
            "view. 'split' adds CPU/GPU-separated raw, compute, and stall totals."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for the annotated CSV and summary JSON outputs. "
            "Reports are only written when this flag is passed or "
            "--write-reports is set."
        ),
    )
    parser.add_argument(
        "--write-reports",
        action="store_true",
        help="Write summary JSON and annotated CSV reports to disk.",
    )
    return parser.parse_args(argv)


def resolve_bundle_dir(llama_bundle: Path) -> Path:
    if llama_bundle.is_dir():
        return llama_bundle
    if llama_bundle.is_file() and llama_bundle.name == "manifest.json":
        return llama_bundle.parent
    raise FileNotFoundError(
        "Expected a llama bundle directory or manifest.json path, got "
        f"{llama_bundle}"
    )


def load_bundle_paths(bundle_dir: Path) -> BundlePaths:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    node_csv = bundle_dir / manifest.get("node_csv", "runtime_nodes.csv")
    edge_csv = bundle_dir / manifest.get("edge_csv", "runtime_edges.csv")
    timing_csv = bundle_dir / manifest.get("timing_csv", "ggml_profile_node_records.csv")
    for path in (node_csv, edge_csv, timing_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundle file: {path}")
    return BundlePaths(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        node_csv=node_csv,
        edge_csv=edge_csv,
        timing_csv=timing_csv,
    )


def _parse_required_duration(row: dict[str, str], timing_csv: Path) -> int:
    raw_duration = (
        row.get("node_compute_time_ns")
        or row.get("compute_time_ns")
        or row.get("duration_ns")
        or row.get("duration")
        or ""
    ).strip()
    if not raw_duration:
        raise RuntimeError(f"{timing_csv} is missing a duration for node_id={row.get('node_id', '<unknown>')}")
    try:
        return int(float(raw_duration))
    except ValueError as exc:
        raise RuntimeError(
            f"{timing_csv} has invalid duration {raw_duration!r} for node_id={row.get('node_id', '<unknown>')}"
        ) from exc


def load_timing_rows(timing_csv: Path) -> dict[str, dict[str, str]]:
    by_node_id: dict[str, dict[str, str]] = {}
    with timing_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [field.strip() for field in reader.fieldnames]
        fieldnames = reader.fieldnames or []
        if "node_id" not in fieldnames:
            raise RuntimeError(
                f"{timing_csv} must contain a node_id column for strict per-node timing lookup."
            )
        for row in reader:
            node_id = (row.get("node_id") or "").strip()
            if not node_id:
                raise RuntimeError(f"{timing_csv} has a timing row without node_id.")
            if node_id in by_node_id:
                raise RuntimeError(f"{timing_csv} has duplicate timing rows for node_id={node_id}.")
            strict_row = dict(row)
            strict_row["node_compute_time_ns"] = str(_parse_required_duration(strict_row, timing_csv))
            by_node_id[node_id] = strict_row
    if not by_node_id:
        raise RuntimeError(f"{timing_csv} contained no timing rows.")
    return by_node_id


def _duration_from_rows(
    node_row: dict[str, str],
    timing_row: dict[str, str] | None,
) -> int:
    if timing_row is not None and timing_row.get("node_compute_time_ns"):
        return int(timing_row["node_compute_time_ns"])
    raise ValueError(
        f"Node {node_row.get('node_id', '<unknown>')} is missing a strict timing row in ggml_profile_node_records.csv"
    )


def _node_from_row(node_row: dict[str, str], duration_ns: int) -> NodeRecord:
    raw = dict(node_row)
    raw["duration_ns"] = str(duration_ns)
    raw.setdefault("node_compute_time_ns", str(duration_ns))
    return NodeRecord(
        raw=raw,
        step=int(node_row.get("step", 0) or 0),
        node_id=node_row.get("node_id", ""),
        node_n=_parse_optional_int(node_row.get("node_n")),
        node_name=node_row.get("node_name", ""),
        op_name=node_row.get("op_name", node_row.get("node_name", "")),
        runtime_role=node_row.get("runtime_role", ""),
        node_kind=node_row.get("node_kind", ""),
        resource_kind=node_row.get("resource_kind", ""),
        resource_id=node_row.get("resource_id", ""),
        device_type=node_row.get("device_type", ""),
        device_index=_parse_optional_int(node_row.get("device_index")),
        duration_ns=duration_ns,
    )


def load_nodes(paths: BundlePaths) -> list[NodeRecord]:
    timing_by_node_id = load_timing_rows(paths.timing_csv)
    seen_node_ids: set[str] = set()
    nodes: list[NodeRecord] = []
    with paths.node_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for node_row in reader:
            node_id = (node_row.get("node_id") or "").strip()
            if not node_id:
                raise RuntimeError(f"{paths.node_csv} has a runtime node without node_id.")
            timing_row = timing_by_node_id.get(node_id)
            duration_ns = _duration_from_rows(node_row, timing_row)
            node = _node_from_row(node_row, duration_ns)
            nodes.append(node)
            if node.node_id:
                seen_node_ids.add(node.node_id)

    extra_timing_node_ids = sorted(set(timing_by_node_id) - seen_node_ids)
    if extra_timing_node_ids:
        sample = ", ".join(extra_timing_node_ids[:5])
        raise RuntimeError(
            f"{paths.timing_csv} has timing rows that are missing from {paths.node_csv}: {sample}"
        )

    return nodes


def load_edges(edge_csv: Path) -> tuple[dict[str, list[EdgeRecord]], dict[str, list[EdgeRecord]]]:
    incoming: dict[str, list[EdgeRecord]] = defaultdict(list)
    outgoing: dict[str, list[EdgeRecord]] = defaultdict(list)
    with edge_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edge = EdgeRecord(
                src_node_id=row["src_node_id"],
                dst_node_id=row["dst_node_id"],
                edge_kind=row["edge_kind"],
            )
            outgoing[edge.src_node_id].append(edge)
            incoming[edge.dst_node_id].append(edge)
    return dict(incoming), dict(outgoing)


def load_bundle_analysis(bundle_dir: str | Path) -> BundleAnalysis:
    bundle_path = resolve_bundle_dir(Path(bundle_dir).resolve())
    paths = load_bundle_paths(bundle_path)
    nodes = load_nodes(paths)
    incoming_edges, outgoing_edges = load_edges(paths.edge_csv)
    return BundleAnalysis(
        paths=paths,
        nodes=nodes,
        nodes_by_id={node.node_id: node for node in nodes if node.node_id},
        incoming_edges=incoming_edges,
        outgoing_edges=outgoing_edges,
    )


def _is_transfer_like(node: NodeRecord) -> bool:
    text = node.identity_text
    transfer_patterns = (
        "memcpy",
        "htod",
        "dtoh",
        "d2h",
        "h2d",
        "pageable -> device",
        "device -> host",
        "host -> device",
        "p2p",
        "prefetch",
        "offload",
        "evict",
        "reload",
        "ssd",
        "disk",
        "nvme",
        "pread",
        "pwrite",
        "io_uring",
    )
    return any(pattern in text for pattern in transfer_patterns)


def _is_wait_like(node: NodeRecord) -> bool:
    text = node.identity_text
    return (
        node.runtime_role == "wait"
        or "synchronize" in text
        or text.startswith("wait")
        or " wait" in text
        or "barrier" in text
    )


def _is_gpu_compute_like(node: NodeRecord) -> bool:
    return node.runtime_role == "gpu_runtime" and not _is_transfer_like(node)


def _is_cpu_runtime_api(node: NodeRecord) -> bool:
    text = node.identity_text
    runtime_prefixes = ("cuda", "cu", "cudnn", "nccl", "hip", "roc")
    return text.startswith(runtime_prefixes)


def _is_cpu_compute_like(node: NodeRecord) -> bool:
    if node.device_type != "CPU":
        return False
    if node.runtime_role != "cpu_leaf":
        return False
    if _is_transfer_like(node) or _is_wait_like(node):
        return False
    return not _is_cpu_runtime_api(node)


def _is_cpu_control_like(node: NodeRecord) -> bool:
    if node.device_type != "CPU":
        return False
    if _is_transfer_like(node) or _is_wait_like(node):
        return False
    if node.runtime_role == "submit":
        return True
    if node.runtime_role == "cpu_leaf" and _is_cpu_runtime_api(node):
        return True
    return False


def _is_cpu_offload_staging_like(node: NodeRecord) -> bool:
    if _device_bucket(node) != "cpu":
        return False
    text = node.identity_text
    staging_patterns = (
        "aten::empty",
        "aten::empty_strided",
        "aten::as_strided",
        "aten::view",
        "aten::_unsafe_view",
        "aten::_reshape_alias",
        "aten::to",
        "aten::copy_",
        "aten::clone",
        "aten::resize_",
        "detach",
    )
    return any(pattern in text for pattern in staging_patterns)


def _is_cpu_compiler_runtime_like(node: NodeRecord) -> bool:
    if _device_bucket(node) != "cpu":
        return False
    text = node.identity_text
    compiler_patterns = (
        "torchdynamo cache lookup",
        "aotdispatcher runtime wrapper prologue",
        "pregraph bytecode",
        "compiledfxgraph",
    )
    return any(pattern in text for pattern in compiler_patterns)


def _is_cpu_cuda_memory_runtime_like(node: NodeRecord) -> bool:
    if _device_bucket(node) != "cpu":
        return False
    text = node.identity_text
    runtime_patterns = (
        "cumemcreate",
        "cumemmap",
        "cumemunmap",
        "cumemrelease",
        "cudastreamiscapturing",
        "cudadevicegetattribute",
        "cudamemcpyasync",
        "cudastreamsynchronize",
    )
    return any(pattern in text for pattern in runtime_patterns)


def _is_cpu_offload_overhead_like(node: NodeRecord) -> bool:
    return (
        _is_cpu_offload_staging_like(node)
        or _is_cpu_compiler_runtime_like(node)
        or _is_cpu_cuda_memory_runtime_like(node)
    )


def _device_bucket(node: NodeRecord) -> str:
    device_type = node.device_type.upper()
    if device_type in {"CUDA", "GPU", "HIP", "XPU"}:
        return "gpu"
    resource_kind = node.resource_kind.lower()
    if node.runtime_role == "gpu_runtime" or "gpu" in resource_kind:
        return "gpu"
    return "cpu"


def _get_node(nodes_by_id: dict[str, NodeRecord], node_id: str) -> NodeRecord | None:
    return nodes_by_id.get(node_id)


def _incoming_sources(
    analysis: BundleAnalysis,
    node_id: str,
    edge_kind: str,
) -> list[NodeRecord]:
    sources: list[NodeRecord] = []
    nodes_by_id = {node.node_id: node for node in analysis.nodes if node.node_id}
    for edge in analysis.incoming_edges.get(node_id, []):
        if edge.edge_kind != edge_kind:
            continue
        source = _get_node(nodes_by_id, edge.src_node_id)
        if source is not None:
            sources.append(source)
    return sources


def _build_nodes_by_id(analysis: BundleAnalysis) -> dict[str, NodeRecord]:
    return analysis.nodes_by_id


def _immediate_transfer_sources_for_wait(
    analysis: BundleAnalysis,
    node_id: str,
    nodes_by_id: dict[str, NodeRecord] | None = None,
) -> list[str]:
    resolved_nodes_by_id = nodes_by_id or _build_nodes_by_id(analysis)
    transfer_source_ids: list[str] = []
    for edge in analysis.incoming_edges.get(node_id, []):
        if edge.edge_kind not in {"wait", "thread_order"}:
            continue
        source = resolved_nodes_by_id.get(edge.src_node_id)
        if source is None or not _is_transfer_like(source):
            continue
        transfer_source_ids.append(source.node_id)
    return transfer_source_ids


def _has_transfer_lineage_for_wait(
    analysis: BundleAnalysis,
    node_id: str,
    *,
    max_depth: int = 8,
    nodes_by_id: dict[str, NodeRecord] | None = None,
) -> tuple[bool, list[str]]:
    resolved_nodes_by_id = nodes_by_id or _build_nodes_by_id(analysis)
    seen = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    transfer_source_ids: list[str] = []

    while queue:
        current_node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in analysis.incoming_edges.get(current_node_id, []):
            if edge.edge_kind not in {"thread_order", "wait"}:
                continue
            source = resolved_nodes_by_id.get(edge.src_node_id)
            if source is None or source.node_id in seen:
                continue
            seen.add(source.node_id)
            if _is_transfer_like(source):
                transfer_source_ids.append(source.node_id)
                continue
            if _is_wait_like(source):
                immediate_transfer_source_ids = _immediate_transfer_sources_for_wait(
                    analysis,
                    source.node_id,
                    nodes_by_id=resolved_nodes_by_id,
                )
                if immediate_transfer_source_ids:
                    transfer_source_ids.extend(immediate_transfer_source_ids)
                    transfer_source_ids.append(source.node_id)
                    continue
            queue.append((source.node_id, depth + 1))

    unique_source_ids = sorted(set(transfer_source_ids))
    return bool(unique_source_ids), unique_source_ids


def _is_offload_lineage_carrier(node: NodeRecord) -> bool:
    if _is_wait_like(node):
        return True
    return _is_cpu_offload_overhead_like(node)


def _has_transfer_lineage_for_cpu_node(
    analysis: BundleAnalysis,
    node_id: str,
    *,
    max_depth: int = 12,
    nodes_by_id: dict[str, NodeRecord] | None = None,
) -> tuple[bool, list[str]]:
    resolved_nodes_by_id = nodes_by_id or _build_nodes_by_id(analysis)
    seen = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    transfer_source_ids: list[str] = []

    while queue:
        current_node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in analysis.incoming_edges.get(current_node_id, []):
            if edge.edge_kind not in {"thread_order", "wait"}:
                continue
            source = resolved_nodes_by_id.get(edge.src_node_id)
            if source is None or source.node_id in seen:
                continue
            seen.add(source.node_id)
            if _is_transfer_like(source):
                transfer_source_ids.append(source.node_id)
                continue
            if _is_wait_like(source):
                has_transfer_lineage, wait_transfer_source_ids = _has_transfer_lineage_for_wait(
                    analysis,
                    source.node_id,
                    nodes_by_id=resolved_nodes_by_id,
                )
                if has_transfer_lineage:
                    transfer_source_ids.extend(wait_transfer_source_ids)
                    transfer_source_ids.append(source.node_id)
                    continue
            if _is_offload_lineage_carrier(source):
                queue.append((source.node_id, depth + 1))

    unique_source_ids = sorted(set(transfer_source_ids))
    return bool(unique_source_ids), unique_source_ids


def _classify_v0(node: NodeRecord, analysis: BundleAnalysis) -> NodeAttribution:
    del analysis
    return NodeAttribution(
        version="v0",
        node_id=node.node_id,
        classification="raw_total",
        pure_compute_time_ns=node.duration_ns,
        ssd_time_ns=0,
        reason="Baseline: keep the imported profile duration unchanged.",
    )


def _classify_v1(node: NodeRecord, analysis: BundleAnalysis) -> NodeAttribution:
    del analysis
    if _is_gpu_compute_like(node):
        return NodeAttribution(
            version="v1",
            node_id=node.node_id,
            classification="pure_gpu_compute",
            pure_compute_time_ns=node.duration_ns,
            ssd_time_ns=0,
            reason="GPU runtime node is not an explicit transfer op.",
        )
    if _is_cpu_compute_like(node):
        return NodeAttribution(
            version="v1",
            node_id=node.node_id,
            classification="pure_cpu_compute",
            pure_compute_time_ns=node.duration_ns,
            ssd_time_ns=0,
            reason="CPU leaf node is not a CUDA runtime/control, wait, or transfer op.",
        )
    if _is_transfer_like(node):
        return NodeAttribution(
            version="v1",
            node_id=node.node_id,
            classification="transfer_op",
            pure_compute_time_ns=0,
            ssd_time_ns=node.duration_ns,
            reason="Explicit transfer/offload/memcpy node.",
        )
    if _is_wait_like(node):
        return NodeAttribution(
            version="v1",
            node_id=node.node_id,
            classification="wait_or_sync",
            pure_compute_time_ns=0,
            ssd_time_ns=node.duration_ns,
            reason="Strict mode excludes synchronization time from pure compute.",
        )
    if _is_cpu_control_like(node):
        return NodeAttribution(
            version="v1",
            node_id=node.node_id,
            classification="unknown_control",
            pure_compute_time_ns=0,
            ssd_time_ns=node.duration_ns,
            reason="Strict mode excludes CPU runtime/control work from pure compute.",
        )
    return NodeAttribution(
        version="v1",
        node_id=node.node_id,
        classification="unknown_other",
        pure_compute_time_ns=0,
        ssd_time_ns=node.duration_ns,
        reason="Node is not confidently classified as pure compute in strict mode.",
    )


def _classify_v2(node: NodeRecord, analysis: BundleAnalysis) -> NodeAttribution:
    del analysis
    if _is_gpu_compute_like(node):
        return NodeAttribution(
            version="v2",
            node_id=node.node_id,
            classification="pure_gpu_compute",
            pure_compute_time_ns=node.duration_ns,
            ssd_time_ns=0,
            reason="GPU runtime node is not an explicit transfer op.",
        )
    if _is_cpu_compute_like(node):
        return NodeAttribution(
            version="v2",
            node_id=node.node_id,
            classification="pure_cpu_compute",
            pure_compute_time_ns=node.duration_ns,
            ssd_time_ns=0,
            reason="CPU leaf node looks like regular host computation.",
        )
    if _is_transfer_like(node):
        return NodeAttribution(
            version="v2",
            node_id=node.node_id,
            classification="transfer_op",
            pure_compute_time_ns=0,
            ssd_time_ns=node.duration_ns,
            reason="Explicit transfer/offload/memcpy node.",
        )
    if _is_wait_like(node):
        return NodeAttribution(
            version="v2",
            node_id=node.node_id,
            classification="wait_or_sync",
            pure_compute_time_ns=0,
            ssd_time_ns=node.duration_ns,
            reason="This version still treats all wait/sync nodes as non-compute stall time.",
        )
    if _is_cpu_control_like(node):
        return NodeAttribution(
            version="v2",
            node_id=node.node_id,
            classification="pure_cpu_control",
            pure_compute_time_ns=node.duration_ns,
            ssd_time_ns=0,
            reason="CPU runtime/control node is preserved as pure CPU work.",
        )
    return NodeAttribution(
        version="v2",
        node_id=node.node_id,
        classification="unknown_other",
        pure_compute_time_ns=0,
        ssd_time_ns=node.duration_ns,
        reason="Node remains unclassified after compute/control heuristics.",
    )


def _classify_v3(node: NodeRecord, analysis: BundleAnalysis) -> NodeAttribution:
    if _is_wait_like(node):
        wait_sources = _incoming_sources(analysis, node.node_id, "wait")
        if any(_is_transfer_like(source) for source in wait_sources):
            source_ids = ", ".join(source.node_id for source in wait_sources if _is_transfer_like(source))
            return NodeAttribution(
                version="v3",
                node_id=node.node_id,
                classification="wait_for_transfer",
                pure_compute_time_ns=0,
                ssd_time_ns=node.duration_ns,
                reason=f"Wait node depends on transfer/offload source(s): {source_ids}.",
            )
        if any(_is_gpu_compute_like(source) or _is_cpu_compute_like(source) for source in wait_sources):
            source_ids = ", ".join(source.node_id for source in wait_sources)
            return NodeAttribution(
                version="v3",
                node_id=node.node_id,
                classification="pure_cpu_control",
                pure_compute_time_ns=node.duration_ns,
                ssd_time_ns=0,
                reason=f"Wait node synchronizes compute source(s) without transfer evidence: {source_ids}.",
            )
        thread_preds = _incoming_sources(analysis, node.node_id, "thread_order")
        if any(_is_transfer_like(source) for source in thread_preds):
            source_ids = ", ".join(source.node_id for source in thread_preds if _is_transfer_like(source))
            return NodeAttribution(
                version="v3",
                node_id=node.node_id,
                classification="wait_for_transfer",
                pure_compute_time_ns=0,
                ssd_time_ns=node.duration_ns,
                reason=f"Wait node immediately follows transfer-like predecessor(s): {source_ids}.",
            )

    v2_attribution = _classify_v2(node, analysis)
    return NodeAttribution(
        version="v3",
        node_id=node.node_id,
        classification=v2_attribution.classification,
        pure_compute_time_ns=v2_attribution.pure_compute_time_ns,
        ssd_time_ns=v2_attribution.ssd_time_ns,
        reason=v2_attribution.reason,
    )


def _classify_v4(node: NodeRecord, analysis: BundleAnalysis) -> NodeAttribution:
    if _is_wait_like(node):
        v3_attribution = _classify_v3(node, analysis)
        if v3_attribution.classification == "wait_for_transfer":
            return NodeAttribution(
                version="v4",
                node_id=node.node_id,
                classification=v3_attribution.classification,
                pure_compute_time_ns=v3_attribution.pure_compute_time_ns,
                ssd_time_ns=v3_attribution.ssd_time_ns,
                reason=v3_attribution.reason,
            )
        return NodeAttribution(
            version="v4",
            node_id=node.node_id,
            classification="pure_cpu_control",
            pure_compute_time_ns=node.duration_ns,
            ssd_time_ns=0,
            reason=(
                "Inclusive control mode keeps wait/sync time as CPU control "
                "unless the graph shows transfer/offload evidence."
            ),
        )

    v3_attribution = _classify_v3(node, analysis)
    return NodeAttribution(
        version="v4",
        node_id=node.node_id,
        classification=v3_attribution.classification,
        pure_compute_time_ns=v3_attribution.pure_compute_time_ns,
        ssd_time_ns=v3_attribution.ssd_time_ns,
        reason=v3_attribution.reason,
    )


def _build_v5_classifier(analysis: BundleAnalysis):
    nodes_by_id = _build_nodes_by_id(analysis)

    def classify(node: NodeRecord) -> NodeAttribution:
        if _is_wait_like(node) and _device_bucket(node) == "cpu":
            v4_attribution = _classify_v4(node, analysis)
            if v4_attribution.ssd_time_ns > 0:
                return NodeAttribution(
                    version="v5",
                    node_id=node.node_id,
                    classification=v4_attribution.classification,
                    pure_compute_time_ns=v4_attribution.pure_compute_time_ns,
                    ssd_time_ns=v4_attribution.ssd_time_ns,
                    reason=v4_attribution.reason,
                )

            has_transfer_lineage, transfer_source_ids = _has_transfer_lineage_for_wait(
                analysis,
                node.node_id,
                nodes_by_id=nodes_by_id,
            )
            if has_transfer_lineage:
                source_ids = ", ".join(transfer_source_ids[:8])
                return NodeAttribution(
                    version="v5",
                    node_id=node.node_id,
                    classification="wait_for_transfer_ancestor",
                    pure_compute_time_ns=0,
                    ssd_time_ns=node.duration_ns,
                    reason=(
                        "CPU wait/sync node inherits transfer/offload stall lineage "
                        f"through upstream thread-order or wait predecessors: {source_ids}."
                    ),
                )

            return NodeAttribution(
                version="v5",
                node_id=node.node_id,
                classification=v4_attribution.classification,
                pure_compute_time_ns=v4_attribution.pure_compute_time_ns,
                ssd_time_ns=v4_attribution.ssd_time_ns,
                reason=v4_attribution.reason,
            )

        v4_attribution = _classify_v4(node, analysis)
        return NodeAttribution(
            version="v5",
            node_id=node.node_id,
            classification=v4_attribution.classification,
            pure_compute_time_ns=v4_attribution.pure_compute_time_ns,
            ssd_time_ns=v4_attribution.ssd_time_ns,
            reason=v4_attribution.reason,
        )

    return classify


def _build_v6_classifier(analysis: BundleAnalysis):
    nodes_by_id = _build_nodes_by_id(analysis)
    v5_classifier = _build_v5_classifier(analysis)

    def classify(node: NodeRecord) -> NodeAttribution:
        v5_attribution = v5_classifier(node)
        if v5_attribution.ssd_time_ns > 0:
            return NodeAttribution(
                version="v6",
                node_id=node.node_id,
                classification=v5_attribution.classification,
                pure_compute_time_ns=v5_attribution.pure_compute_time_ns,
                ssd_time_ns=v5_attribution.ssd_time_ns,
                reason=v5_attribution.reason,
            )

        if _device_bucket(node) == "cpu" and _is_cpu_offload_overhead_like(node):
            has_transfer_lineage, transfer_source_ids = _has_transfer_lineage_for_cpu_node(
                analysis,
                node.node_id,
                nodes_by_id=nodes_by_id,
            )
            if has_transfer_lineage:
                source_ids = ", ".join(transfer_source_ids[:8])
                return NodeAttribution(
                    version="v6",
                    node_id=node.node_id,
                    classification="offload_cpu_staging_or_control",
                    pure_compute_time_ns=0,
                    ssd_time_ns=node.duration_ns,
                    reason=(
                        "CPU staging/control node inherits transfer/offload lineage "
                        f"through upstream thread-order or wait predecessors: {source_ids}."
                    ),
                )

        return NodeAttribution(
            version="v6",
            node_id=node.node_id,
            classification=v5_attribution.classification,
            pure_compute_time_ns=v5_attribution.pure_compute_time_ns,
            ssd_time_ns=v5_attribution.ssd_time_ns,
            reason=v5_attribution.reason,
        )

    return classify


CLASSIFIERS = {
    "v0": _classify_v0,
    "v1": _classify_v1,
    "v2": _classify_v2,
    "v3": _classify_v3,
    "v4": _classify_v4,
}


def attribute_nodes(
    analysis: BundleAnalysis,
    version: str,
) -> list[NodeAttribution]:
    if version == "v5":
        classifier = _build_v5_classifier(analysis)
        return [classifier(node) for node in analysis.nodes]
    if version == "v6":
        classifier = _build_v6_classifier(analysis)
        return [classifier(node) for node in analysis.nodes]
    classifier = CLASSIFIERS[version]
    return [classifier(node, analysis) for node in analysis.nodes]


def summarize_attributions(
    analysis: BundleAnalysis,
    attributions: list[NodeAttribution],
) -> VersionReport:
    raw_total_ns = sum(node.duration_ns for node in analysis.nodes)
    pure_compute_ns = sum(item.pure_compute_time_ns for item in attributions)
    ssd_time_ns = sum(item.ssd_time_ns for item in attributions)
    raw_total_ns_cpu = 0
    raw_total_ns_gpu = 0
    pure_compute_ns_cpu = 0
    pure_compute_ns_gpu = 0
    ssd_time_ns_cpu = 0
    ssd_time_ns_gpu = 0
    for node, attribution in zip(analysis.nodes, attributions):
        if _device_bucket(node) == "gpu":
            raw_total_ns_gpu += node.duration_ns
            pure_compute_ns_gpu += attribution.pure_compute_time_ns
            ssd_time_ns_gpu += attribution.ssd_time_ns
        else:
            raw_total_ns_cpu += node.duration_ns
            pure_compute_ns_cpu += attribution.pure_compute_time_ns
            ssd_time_ns_cpu += attribution.ssd_time_ns
    classification_counts = Counter(item.classification for item in attributions)
    return VersionReport(
        version=attributions[0].version if attributions else "unknown",
        raw_total_ns=raw_total_ns,
        raw_total_ns_cpu=raw_total_ns_cpu,
        raw_total_ns_gpu=raw_total_ns_gpu,
        pure_compute_ns=pure_compute_ns,
        pure_compute_ns_cpu=pure_compute_ns_cpu,
        pure_compute_ns_gpu=pure_compute_ns_gpu,
        ssd_time_ns=ssd_time_ns,
        ssd_time_ns_cpu=ssd_time_ns_cpu,
        ssd_time_ns_gpu=ssd_time_ns_gpu,
        node_count=len(analysis.nodes),
        compute_node_count=sum(item.pure_compute_time_ns > 0 for item in attributions),
        stall_node_count=sum(item.ssd_time_ns > 0 for item in attributions),
        classification_counts=dict(sorted(classification_counts.items())),
    )


def analyze_bundle(
    bundle_dir: str | Path,
    versions: list[str] | None = None,
) -> tuple[BundleAnalysis, dict[str, VersionReport], dict[str, list[NodeAttribution]]]:
    analysis = load_bundle_analysis(bundle_dir)
    resolved_versions = versions or list(SUPPORTED_VERSIONS)
    reports: dict[str, VersionReport] = {}
    attributions_by_version: dict[str, list[NodeAttribution]] = {}
    for version in resolved_versions:
        attributions = attribute_nodes(analysis, version)
        reports[version] = summarize_attributions(analysis, attributions)
        attributions_by_version[version] = attributions
    return analysis, reports, attributions_by_version


def _report_to_json_dict(report: VersionReport) -> dict[str, Any]:
    return {
        "version": report.version,
        "raw_total_ns": report.raw_total_ns,
        "raw_total_ns_cpu": report.raw_total_ns_cpu,
        "raw_total_ns_gpu": report.raw_total_ns_gpu,
        "pure_compute_ns": report.pure_compute_ns,
        "pure_compute_ns_cpu": report.pure_compute_ns_cpu,
        "pure_compute_ns_gpu": report.pure_compute_ns_gpu,
        "ssd_time_ns": report.ssd_time_ns,
        "ssd_time_ns_cpu": report.ssd_time_ns_cpu,
        "ssd_time_ns_gpu": report.ssd_time_ns_gpu,
        "raw_total_us": report.raw_total_ns / 1_000.0,
        "raw_total_us_cpu": report.raw_total_ns_cpu / 1_000.0,
        "raw_total_us_gpu": report.raw_total_ns_gpu / 1_000.0,
        "pure_compute_us": report.pure_compute_ns / 1_000.0,
        "pure_compute_us_cpu": report.pure_compute_ns_cpu / 1_000.0,
        "pure_compute_us_gpu": report.pure_compute_ns_gpu / 1_000.0,
        "ssd_time_us": report.ssd_time_ns / 1_000.0,
        "ssd_time_us_cpu": report.ssd_time_ns_cpu / 1_000.0,
        "ssd_time_us_gpu": report.ssd_time_ns_gpu / 1_000.0,
        "node_count": report.node_count,
        "compute_node_count": report.compute_node_count,
        "stall_node_count": report.stall_node_count,
        "classification_counts": report.classification_counts,
    }


def write_summary_json(
    analysis: BundleAnalysis,
    reports: dict[str, VersionReport],
    output_dir: Path,
) -> Path:
    output_path = output_dir / "summary.json"
    payload = {
        "bundle_dir": str(analysis.paths.bundle_dir),
        "manifest_path": str(analysis.paths.manifest_path),
        "versions": [_report_to_json_dict(report) for report in reports.values()],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def write_annotated_csv(
    analysis: BundleAnalysis,
    version: str,
    attributions: list[NodeAttribution],
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{version}_annotated_nodes.csv"
    attribution_by_node_id = {item.node_id: item for item in attributions}
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for node in analysis.nodes:
        for key in node.raw.keys():
            if key in seen_fields:
                continue
            fieldnames.append(key)
            seen_fields.add(key)
    fieldnames.extend(
        [
            "version",
            "device_bucket",
            "classification",
            "raw_time_ns",
            "raw_cpu_time_ns",
            "raw_gpu_time_ns",
            "pure_compute_time_ns",
            "pure_compute_time_cpu_ns",
            "pure_compute_time_gpu_ns",
            "ssd_time_ns",
            "ssd_time_cpu_ns",
            "ssd_time_gpu_ns",
            "reason",
        ]
    )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node in analysis.nodes:
            attribution = attribution_by_node_id[node.node_id]
            device_bucket = _device_bucket(node)
            row = dict(node.raw)
            row.update(
                {
                    "version": version,
                    "device_bucket": device_bucket,
                    "classification": attribution.classification,
                    "raw_time_ns": str(node.duration_ns),
                    "raw_cpu_time_ns": str(node.duration_ns if device_bucket == "cpu" else 0),
                    "raw_gpu_time_ns": str(node.duration_ns if device_bucket == "gpu" else 0),
                    "pure_compute_time_ns": str(attribution.pure_compute_time_ns),
                    "pure_compute_time_cpu_ns": str(
                        attribution.pure_compute_time_ns if device_bucket == "cpu" else 0
                    ),
                    "pure_compute_time_gpu_ns": str(
                        attribution.pure_compute_time_ns if device_bucket == "gpu" else 0
                    ),
                    "ssd_time_ns": str(attribution.ssd_time_ns),
                    "ssd_time_cpu_ns": str(attribution.ssd_time_ns if device_bucket == "cpu" else 0),
                    "ssd_time_gpu_ns": str(attribution.ssd_time_ns if device_bucket == "gpu" else 0),
                    "reason": attribution.reason,
                }
            )
            writer.writerow(row)

    return output_path


def write_reports(
    analysis: BundleAnalysis,
    reports: dict[str, VersionReport],
    attributions_by_version: dict[str, list[NodeAttribution]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, Path] = {}
    written_paths["summary"] = write_summary_json(analysis, reports, output_dir)
    for version, attributions in attributions_by_version.items():
        written_paths[version] = write_annotated_csv(
            analysis,
            version,
            attributions,
            output_dir,
        )
    return written_paths


def _format_us(value_ns: int) -> str:
    return f"{value_ns / 1_000.0:.3f}"


def render_summary_table(reports: dict[str, VersionReport]) -> str:
    return render_summary_table_with_view(reports, summary_view="split")


def render_summary_table_with_view(
    reports: dict[str, VersionReport],
    *,
    summary_view: str,
) -> str:
    if summary_view == "compact":
        rows = [
            [
                report.version,
                _format_us(report.raw_total_ns),
                _format_us(report.pure_compute_ns),
                _format_us(report.ssd_time_ns),
                str(report.compute_node_count),
                str(report.stall_node_count),
            ]
            for report in reports.values()
        ]
        headers = [
            "version",
            "raw_total_us",
            "pure_compute_us",
            "ssd_time_us",
            "compute_nodes",
            "stall_nodes",
        ]
    else:
        rows = [
            [
                report.version,
                _format_us(report.raw_total_ns),
                _format_us(report.raw_total_ns_cpu),
                _format_us(report.raw_total_ns_gpu),
                _format_us(report.pure_compute_ns),
                _format_us(report.pure_compute_ns_cpu),
                _format_us(report.pure_compute_ns_gpu),
                _format_us(report.ssd_time_ns),
                _format_us(report.ssd_time_ns_cpu),
                _format_us(report.ssd_time_ns_gpu),
                str(report.compute_node_count),
                str(report.stall_node_count),
            ]
            for report in reports.values()
        ]
        headers = [
            "version",
            "raw_total_us",
            "raw_cpu_us",
            "raw_gpu_us",
            "pure_compute_us",
            "pure_cpu_us",
            "pure_gpu_us",
            "ssd_time_us",
            "ssd_cpu_us",
            "ssd_gpu_us",
            "compute_nodes",
            "stall_nodes",
        ]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    table_lines = [_format_row(headers), _format_row(["-" * width for width in widths])]
    table_lines.extend(_format_row(row) for row in rows)
    return "\n".join(table_lines)


def _summarize_raw_bundle(analysis: BundleAnalysis) -> VersionReport:
    raw_total_ns = sum(node.duration_ns for node in analysis.nodes)
    raw_total_ns_cpu = sum(
        node.duration_ns for node in analysis.nodes if _device_bucket(node) == "cpu"
    )
    raw_total_ns_gpu = raw_total_ns - raw_total_ns_cpu
    return VersionReport(
        version=str(analysis.paths.bundle_dir),
        raw_total_ns=raw_total_ns,
        raw_total_ns_cpu=raw_total_ns_cpu,
        raw_total_ns_gpu=raw_total_ns_gpu,
        pure_compute_ns=0,
        pure_compute_ns_cpu=0,
        pure_compute_ns_gpu=0,
        ssd_time_ns=0,
        ssd_time_ns_cpu=0,
        ssd_time_ns_gpu=0,
        node_count=len(analysis.nodes),
        compute_node_count=0,
        stall_node_count=0,
        classification_counts={},
    )


def _aggregate_duration_by_key(
    analysis: BundleAnalysis,
) -> Counter[tuple[str, str, str]]:
    durations: Counter[tuple[str, str, str]] = Counter()
    for node in analysis.nodes:
        durations[(_device_bucket(node), node.runtime_role, node.node_name)] += node.duration_ns
    return durations


def _aggregate_retained_compute_by_key(
    analysis: BundleAnalysis,
    attributions: list[NodeAttribution],
) -> Counter[tuple[str, str, str, str]]:
    retained: Counter[tuple[str, str, str, str]] = Counter()
    for node, attribution in zip(analysis.nodes, attributions):
        if attribution.pure_compute_time_ns <= 0:
            continue
        retained[
            (
                _device_bucket(node),
                attribution.classification,
                node.runtime_role,
                node.node_name,
            )
        ] += attribution.pure_compute_time_ns
    return retained


def render_compare_report(
    primary_analysis: BundleAnalysis,
    secondary_analysis: BundleAnalysis,
    *,
    version: str,
    compare_limit: int,
    include_retained_compute: bool,
) -> str:
    raw_reports = {
        "primary": _summarize_raw_bundle(primary_analysis),
        "secondary": _summarize_raw_bundle(secondary_analysis),
    }
    lines = [
        "Raw Bundle Summary",
        render_summary_table_with_view(raw_reports, summary_view="split"),
    ]

    primary_durations = _aggregate_duration_by_key(primary_analysis)
    secondary_durations = _aggregate_duration_by_key(secondary_analysis)
    delta_rows: list[list[str]] = []
    for key in set(primary_durations) | set(secondary_durations):
        device_bucket, runtime_role, node_name = key
        if device_bucket != "cpu":
            continue
        primary_ns = primary_durations[key]
        secondary_ns = secondary_durations[key]
        delta_ns = primary_ns - secondary_ns
        if delta_ns <= 0:
            continue
        delta_rows.append(
            [
                _format_us(delta_ns),
                _format_us(primary_ns),
                _format_us(secondary_ns),
                runtime_role,
                node_name,
            ]
        )
    delta_rows.sort(key=lambda row: float(row[0]), reverse=True)
    lines.append("\nTop CPU Raw Deltas (primary - secondary)")
    lines.append(
        _render_generic_table(
            headers=[
                "delta_us",
                "primary_us",
                "secondary_us",
                "runtime_role",
                "node_name",
            ],
            rows=delta_rows[:compare_limit],
        )
    )

    if include_retained_compute:
        primary_attributions = attribute_nodes(primary_analysis, version)
        retained_compute = _aggregate_retained_compute_by_key(
            primary_analysis,
            primary_attributions,
        )
        retained_rows: list[list[str]] = []
        for key, pure_compute_ns in retained_compute.items():
            device_bucket, classification, runtime_role, node_name = key
            if device_bucket != "cpu":
                continue
            retained_rows.append(
                [
                    _format_us(pure_compute_ns),
                    classification,
                    runtime_role,
                    node_name,
                ]
            )
        retained_rows.sort(key=lambda row: float(row[0]), reverse=True)
        lines.append(f"\nTop CPU Nodes Still Counted As Compute In Primary ({version})")
        lines.append(
            _render_generic_table(
                headers=[
                    "pure_compute_us",
                    "classification",
                    "runtime_role",
                    "node_name",
                ],
                rows=retained_rows[:compare_limit],
            )
        )
    return "\n".join(lines)


def _render_generic_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    table_lines = [_format_row(headers), _format_row(["-" * width for width in widths])]
    table_lines.extend(_format_row(row) for row in rows)
    return "\n".join(table_lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bundle_dir = resolve_bundle_dir(Path(args.llama_bundle).resolve())
    if args.compare_bundle is not None:
        if args.version == "all":
            raise ValueError("--compare-bundle requires a single --version, not 'all'")
        compare_bundle_dir = resolve_bundle_dir(Path(args.compare_bundle).resolve())
        print(
            render_compare_report(
                load_bundle_analysis(bundle_dir),
                load_bundle_analysis(compare_bundle_dir),
                version=args.version,
                compare_limit=args.compare_limit,
                include_retained_compute=args.compare_include_retained_compute,
            )
        )
        return

    versions = include_v0_version(resolve_versions(args.version))
    analysis, reports, attributions_by_version = analyze_bundle(bundle_dir, versions)
    print(render_summary_table_with_view(reports, summary_view=args.summary_view))
    if args.write_reports or args.out_dir is not None:
        output_dir = (
            Path(args.out_dir).resolve()
            if args.out_dir is not None
            else Path("/tmp") / f"{bundle_dir.name}_time_split"
        )
        written_paths = write_reports(analysis, reports, attributions_by_version, output_dir)
        print(f"\nsummary_json={written_paths['summary']}")
        for version in versions:
            print(f"{version}_annotated_csv={written_paths[version]}")


if __name__ == "__main__":
    main()
