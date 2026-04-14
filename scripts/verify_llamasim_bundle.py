#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
PROFILE_COMMON_PATH = THIS_DIR / "profile_sdxl_turbo_common.py"
PROFILE_COMMON_MODULE_NAME = "_verify_llamasim_bundle_profile_common"


def load_profile_common():
    module = sys.modules.get(PROFILE_COMMON_MODULE_NAME)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(
        PROFILE_COMMON_MODULE_NAME,
        PROFILE_COMMON_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load profiler helpers from {PROFILE_COMMON_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PROFILE_COMMON_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


COMMON = load_profile_common()


_NODE_LINE_RE = re.compile(
    r'^\s*"(?P<node_id>[^"]+)"\s*\[(?P<attrs>.*)\]\s*;?\s*$'
)
_EDGE_LINE_RE = re.compile(
    r'^\s*"(?P<src>[^"]+)"\s*->\s*"(?P<dst>[^"]+)"(?:\s*\[(?P<attrs>.*)\])?\s*;?\s*$'
)
_LABEL_RE = re.compile(r'label\s*=\s*"(?P<label>[^"]*)"')
_TRACE_EVENT_MATCH_TOLERANCE_US = 1e-3


@dataclass(frozen=True)
class SourceArtifacts:
    trace_csv: Path
    trace_json: Path
    kineto_map_csv: Path
    execution_trace: Path


@dataclass(frozen=True)
class EventSignature:
    device_type: str
    device_index: int
    resource_kind: str
    resource_id: int
    op_name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    correlation_id: int
    linked_correlation_id: int


@dataclass(frozen=True)
class TensorSignature:
    tensor_id: int
    storage_id: int
    offset: int
    numel: int
    itemsize: int
    device: str
    tensor_kind: str
    dtype: str
    shape_json: str


@dataclass(frozen=True)
class TensorEdgeSignature:
    tensor_key: str
    event: EventSignature
    edge_kind: str


@dataclass
class VerificationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class FakeKinetoEvent:
    def __init__(
        self,
        *,
        name: str,
        device_type: Any,
        device_index: int,
        device_resource_id: int,
        start_ns: int,
        duration_ns: int,
        correlation_id: int,
        linked_correlation_id: int,
        thread_id: int,
        is_async: bool,
        is_user_annotation: bool,
    ) -> None:
        self._name = name
        self._device_type = device_type
        self._device_index = device_index
        self._device_resource_id = device_resource_id
        self._start_ns = start_ns
        self._duration_ns = duration_ns
        self._correlation_id = correlation_id
        self._linked_correlation_id = linked_correlation_id
        self._thread_id = thread_id
        self._is_async = is_async
        self._is_user_annotation = is_user_annotation

    def name(self) -> str:
        return self._name

    def device_type(self) -> Any:
        return self._device_type

    def device_index(self) -> int:
        return self._device_index

    def device_resource_id(self) -> int:
        return self._device_resource_id

    def start_ns(self) -> int:
        return self._start_ns

    def end_ns(self) -> int:
        return self._start_ns + self._duration_ns

    def duration_ns(self) -> int:
        return self._duration_ns

    def correlation_id(self) -> int:
        return self._correlation_id

    def linked_correlation_id(self) -> int:
        return self._linked_correlation_id

    def start_thread_id(self) -> int:
        return self._thread_id

    def end_thread_id(self) -> int:
        return self._thread_id

    def is_async(self) -> bool:
        return self._is_async

    def is_user_annotation(self) -> bool:
        return self._is_user_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a llamasim runtime bundle is internally consistent and, "
            "when the matching profiler artifacts are available, that it matches "
            "the recorded runtime trace."
        )
    )
    parser.add_argument(
        "bundle",
        help="Path to the bundle directory or its manifest.json file.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help=(
            "Optional profile output directory containing *_trace.csv, "
            "*_trace.json, *_kineto_map.csv, and *_execution_trace.json."
        ),
    )
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--trace-json", default=None)
    parser.add_argument("--kineto-map", default=None)
    parser.add_argument("--execution-trace", default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the verification report as JSON.",
    )
    return parser.parse_args()


def resolve_bundle_dir(bundle: str | Path) -> Path:
    bundle_path = Path(bundle).resolve()
    if bundle_path.is_dir():
        return bundle_path
    if bundle_path.is_file() and bundle_path.name == "manifest.json":
        return bundle_path.parent
    raise RuntimeError(
        f"Expected a bundle directory or manifest.json path, got {bundle_path}"
    )


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def maybe_resolve_source_artifacts(
    *,
    profile_dir: str | None,
    trace_csv: str | None,
    trace_json: str | None,
    kineto_map: str | None,
    execution_trace: str | None,
) -> SourceArtifacts | None:
    explicit_values = [trace_csv, trace_json, kineto_map, execution_trace]
    if profile_dir is None and not any(explicit_values):
        return None

    base_dir = Path(profile_dir).resolve() if profile_dir is not None else None

    def resolve_explicit(value: str | None) -> Path | None:
        return None if value is None else Path(value).resolve()

    def find_single(
        pattern: str,
        *,
        exclude_suffixes: tuple[str, ...] = (),
    ) -> Path:
        if base_dir is None:
            raise RuntimeError(
                f"Missing required source artifact {pattern}; pass it explicitly."
            )
        matches = sorted(
            path
            for path in base_dir.glob(pattern)
            if not any(path.name.endswith(suffix) for suffix in exclude_suffixes)
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one match for {pattern} under {base_dir}, found {len(matches)}"
            )
        return matches[0]

    resolved_trace_csv = resolve_explicit(trace_csv) or find_single("*_trace.csv")
    resolved_trace_json = resolve_explicit(trace_json) or find_single(
        "*_trace.json",
        exclude_suffixes=("_execution_trace.json",),
    )
    resolved_kineto_map = resolve_explicit(kineto_map) or find_single("*_kineto_map.csv")
    resolved_execution_trace = resolve_explicit(execution_trace) or (
        find_single("*_execution_trace.json")
        if base_dir is not None
        else None
    )
    if resolved_execution_trace is None:
        raise RuntimeError("Missing required execution trace path")

    return SourceArtifacts(
        trace_csv=resolved_trace_csv,
        trace_json=resolved_trace_json,
        kineto_map_csv=resolved_kineto_map,
        execution_trace=resolved_execution_trace,
    )


def parse_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def parse_json_list(value: str) -> list[Any]:
    parsed = json.loads(value)
    if isinstance(parsed, list):
        return parsed
    raise RuntimeError(f"Expected JSON list, got {value}")


def device_type_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def runtime_resource_kind(event: FakeKinetoEvent) -> str:
    if device_type_name(event.device_type()) == "CPU":
        return "cpu_thread"
    return "gpu_stream"


def runtime_resource_id(event: FakeKinetoEvent) -> int:
    if device_type_name(event.device_type()) == "CPU":
        return event.start_thread_id()
    return event.device_resource_id()


def event_signature(event: FakeKinetoEvent) -> EventSignature:
    return EventSignature(
        device_type=device_type_name(event.device_type()),
        device_index=event.device_index(),
        resource_kind=runtime_resource_kind(event),
        resource_id=runtime_resource_id(event),
        op_name=event.name(),
        start_ns=event.start_ns(),
        end_ns=event.end_ns(),
        duration_ns=event.duration_ns(),
        correlation_id=event.correlation_id(),
        linked_correlation_id=event.linked_correlation_id(),
    )


def bundle_row_signature(row: dict[str, str]) -> EventSignature:
    return EventSignature(
        device_type=row["device_type"],
        device_index=parse_int(row.get("device_index")),
        resource_kind=row["resource_kind"],
        resource_id=parse_int(row.get("resource_id")),
        op_name=row["op_name"],
        start_ns=parse_int(row.get("start_ns")),
        end_ns=parse_int(row.get("end_ns")),
        duration_ns=parse_int(row.get("duration_ns")),
        correlation_id=parse_int(row.get("correlation_id")),
        linked_correlation_id=parse_int(row.get("linked_correlation_id")),
    )


def tensor_key_from_row(row: dict[str, str]) -> str:
    return json.dumps(
        [
            parse_int(row.get("tensor_id")),
            parse_int(row.get("storage_id")),
            parse_int(row.get("offset")),
            parse_int(row.get("numel")),
            parse_int(row.get("itemsize")),
            row.get("device", ""),
        ],
        separators=(",", ":"),
    )


def tensor_signature_from_row(row: dict[str, str]) -> TensorSignature:
    return TensorSignature(
        tensor_id=parse_int(row.get("tensor_id")),
        storage_id=parse_int(row.get("storage_id")),
        offset=parse_int(row.get("offset")),
        numel=parse_int(row.get("numel")),
        itemsize=parse_int(row.get("itemsize")),
        device=row.get("device", ""),
        tensor_kind=row.get("tensor_kind", ""),
        dtype=row.get("dtype", ""),
        shape_json=row.get("shape", ""),
    )


def parse_dot_graph(path: Path) -> tuple[set[str], list[tuple[str, str, str | None]]]:
    node_ids: set[str] = set()
    edges: list[tuple[str, str, str | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        edge_match = _EDGE_LINE_RE.match(line)
        if edge_match is not None:
            attrs = edge_match.group("attrs") or ""
            label_match = _LABEL_RE.search(attrs)
            label = None if label_match is None else label_match.group("label")
            edges.append(
                (
                    edge_match.group("src"),
                    edge_match.group("dst"),
                    label,
                )
            )
            continue
        node_match = _NODE_LINE_RE.match(line)
        if node_match is not None:
            node_ids.add(node_match.group("node_id"))
    return node_ids, edges


def sample_sorted(values: set[Any] | list[Any]) -> list[Any]:
    return sorted(values, key=str)[:10]


def source_trace_event_lookup(
    trace_json: dict[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    cpu_events_by_external_id: dict[int, list[dict[str, Any]]] = {}
    gpu_events_by_correlation_id: dict[int, list[dict[str, Any]]] = {}
    for event in trace_json.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        if "External id" in args and event.get("cat") == "cpu_op":
            cpu_events_by_external_id.setdefault(
                int(args["External id"]), []
            ).append(event)
        if "correlation" in args and event.get("cat") == "kernel":
            gpu_events_by_correlation_id.setdefault(
                int(args["correlation"]), []
            ).append(event)
    return cpu_events_by_external_id, gpu_events_by_correlation_id


def match_trace_json_event(
    *,
    candidates: list[dict[str, Any]],
    expected_name: str,
    expected_start_us: float,
    expected_duration_us: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    same_name = [event for event in candidates if event.get("name") == expected_name]
    if same_name:
        candidates = same_name
    exact = [
        event
        for event in candidates
        if abs(float(event.get("dur", 0.0)) - expected_duration_us)
        <= _TRACE_EVENT_MATCH_TOLERANCE_US
        and abs(float(event.get("ts", 0.0)) - expected_start_us)
        <= _TRACE_EVENT_MATCH_TOLERANCE_US
    ]
    if len(exact) == 1:
        return exact[0]
    return min(
        candidates,
        key=lambda event: (
            abs(float(event.get("ts", 0.0)) - expected_start_us),
            abs(float(event.get("dur", 0.0)) - expected_duration_us),
        ),
    )


def load_fake_kineto_events(artifacts: SourceArtifacts) -> list[FakeKinetoEvent]:
    from torch.profiler import DeviceType

    trace_csv_rows_by_index = {
        parse_int(row["event_index"]): row for row in read_csv_rows(artifacts.trace_csv)
    }
    with artifacts.trace_json.open(encoding="utf-8") as f:
        trace_json = json.load(f)
    cpu_events_by_external_id, gpu_events_by_correlation_id = source_trace_event_lookup(
        trace_json
    )

    events: list[FakeKinetoEvent] = []
    for row in read_csv_rows(artifacts.kineto_map_csv):
        matches = parse_json_list(row["csv_event_indexes"])
        thread_id = 0
        is_async = False
        is_user_annotation = False

        if len(matches) == 1:
            csv_row = trace_csv_rows_by_index[int(matches[0])]
            thread_id = parse_int(csv_row.get("thread"))
            is_async = csv_row.get("is_async", "").lower() == "true"
            is_user_annotation = (
                csv_row.get("is_user_annotation", "").lower() == "true"
            )
        else:
            expected_start_us = float(row["kineto_start_us"])
            expected_duration_us = float(row["kineto_duration_us"])
            correlation_id = parse_int(row["kineto_correlation_id"])
            device_type = row["kineto_device_type"]
            trace_event = None
            if device_type == "CPU":
                trace_event = match_trace_json_event(
                    candidates=cpu_events_by_external_id.get(correlation_id, []),
                    expected_name=row["kineto_name"],
                    expected_start_us=expected_start_us,
                    expected_duration_us=expected_duration_us,
                )
                if trace_event is not None:
                    thread_id = parse_int(str(trace_event.get("tid")))
            else:
                trace_event = match_trace_json_event(
                    candidates=gpu_events_by_correlation_id.get(correlation_id, []),
                    expected_name=row["kineto_name"],
                    expected_start_us=expected_start_us,
                    expected_duration_us=expected_duration_us,
                )

            if trace_event is None:
                raise RuntimeError(
                    "Failed to recover unmatched Kineto event from trace JSON: "
                    f"{row['kineto_name']} cid={correlation_id}"
                )

        events.append(
            FakeKinetoEvent(
                name=row["kineto_name"],
                device_type=getattr(DeviceType, row["kineto_device_type"]),
                device_index=parse_int(row["kineto_device_index"]),
                device_resource_id=parse_int(row["kineto_device_resource_id"]),
                start_ns=COMMON.scalarize_trace_us(float(row["kineto_start_us"])),
                duration_ns=COMMON.scalarize_trace_us(float(row["kineto_duration_us"])),
                correlation_id=parse_int(row["kineto_correlation_id"]),
                linked_correlation_id=parse_int(row["kineto_linked_correlation_id"]),
                thread_id=thread_id,
                is_async=is_async,
                is_user_annotation=is_user_annotation,
            )
        )
    return events


def expected_runtime_roles(
    selected_events: list[FakeKinetoEvent],
    selected_cpu_events: list[FakeKinetoEvent],
    selected_gpu_events: list[FakeKinetoEvent],
    all_events: list[FakeKinetoEvent],
    *,
    ac2g_correlation_ids: set[int] | None = None,
    ac2g_endpoints: Any | None = None,
    ac2g_matches: Any | None = None,
) -> tuple[
    list[tuple[EventSignature, EventSignature, str]],
    dict[EventSignature, str],
]:
    node_ids = {id(event): f"k{index}" for index, event in enumerate(selected_events)}
    selected_event_by_node_id = {
        node_id: event for event, node_id in ((event, node_ids[id(event)]) for event in selected_events)
    }
    thread_edges = COMMON.build_llamasim_thread_order_edges(node_ids, selected_cpu_events)
    stream_edges = COMMON.build_llamasim_stream_order_edges(node_ids, selected_gpu_events)
    if ac2g_correlation_ids:
        submit_edges, runtime_role_by_event_id, _ = (
            COMMON.build_llamasim_submit_edges_from_ac2g(
                node_ids,
                selected_cpu_events,
                selected_gpu_events,
                all_events,
                ac2g_correlation_ids,
                ac2g_endpoints=ac2g_endpoints,
                trace_start_ns=0,
                ac2g_matches=ac2g_matches,
                strict=ac2g_endpoints is not None,
            )
        )
    else:
        submit_edges, runtime_role_by_event_id = COMMON.build_llamasim_submit_edges(
            node_ids,
            selected_cpu_events,
            selected_gpu_events,
            all_events,
        )
    # Exact wait edges from ws_sync markers
    ws_launch_index = COMMON.build_ws_launch_index(all_events)
    ws_sync_index = COMMON.propagate_ws_sync_to_leaf_events(
        all_events,
        COMMON.build_ws_sync_index(all_events),
    )
    exact_wait_edges: list[tuple[str, str, str]] = []
    if ws_sync_index:
        exact_wait_edges, runtime_role_by_event_id, _ = (
            COMMON.build_llamasim_exact_wait_edges(
                node_ids,
                selected_cpu_events,
                selected_gpu_events,
                ws_launch_index,
                ws_sync_index,
                runtime_role_by_event_id,
            )
        )

    aten_wait_edges, runtime_role_by_event_id, _, _ = (
        COMMON.build_llamasim_aten_sync_wait_edges(
            node_ids,
            selected_cpu_events,
            selected_gpu_events,
            all_events,
            runtime_role_by_event_id,
        )
    )

    # Heuristic wait edges (fallback)
    wait_edges, runtime_role_by_event_id = COMMON.build_llamasim_wait_edges(
        node_ids,
        selected_cpu_events,
        selected_gpu_events,
        runtime_role_by_event_id,
    )
    sync_wait_edges, runtime_role_by_event_id = COMMON.build_llamasim_sync_wait_edges(
        node_ids,
        selected_cpu_events,
        selected_gpu_events,
        runtime_role_by_event_id,
    )
    wait_edges.extend(sync_wait_edges)
    wait_edges = exact_wait_edges + aten_wait_edges + wait_edges

    edges: list[tuple[EventSignature, EventSignature, str]] = []
    for src_node_id, dst_node_id, edge_kind in [
        *thread_edges,
        *stream_edges,
        *submit_edges,
        *wait_edges,
    ]:
        edges.append(
            (
                event_signature(selected_event_by_node_id[src_node_id]),
                event_signature(selected_event_by_node_id[dst_node_id]),
                edge_kind,
            )
        )

    roles: dict[EventSignature, str] = {}
    for event in selected_events:
        signature = event_signature(event)
        roles[signature] = runtime_role_by_event_id.get(
            id(event),
            "cpu_leaf"
            if device_type_name(event.device_type()) == "CPU"
            else "gpu_runtime",
        )
    return edges, roles


def verify_exact_submit_edges_against_ac2g(
    bundle_state: dict[str, Any],
    ac2g_matches: Any,
    report: VerificationReport,
) -> None:
    bundle_row_by_signature = {
        bundle_row_signature(row): row for row in bundle_state["node_rows"]
    }
    submit_row_by_node_pair = {
        (row["src_node_id"], row["dst_node_id"]): row
        for row in bundle_state["edge_rows"]
        if row["edge_kind"] == "submit"
    }

    for pair in ac2g_matches.pairs_by_corr_id.values():
        src_signature = event_signature(pair.cpu_event)
        dst_signature = event_signature(pair.gpu_event)
        src_row = bundle_row_by_signature.get(src_signature)
        dst_row = bundle_row_by_signature.get(dst_signature)
        if src_row is None or dst_row is None:
            report.errors.append(
                "Bundle is missing an exact ac2g submit endpoint node "
                f"cid={pair.correlation_id} src={src_signature} dst={dst_signature}"
            )
            continue
        edge_row = submit_row_by_node_pair.get(
            (src_row["node_id"], dst_row["node_id"])
        )
        if edge_row is None:
            report.errors.append(
                "Bundle is missing exact ac2g submit edge "
                f"cid={pair.correlation_id} src_node={src_row['node_id']} "
                f"dst_node={dst_row['node_id']}"
            )
            continue
        if edge_row.get("submit_exact") != "true":
            report.errors.append(
                "Bundle submit edge is not marked exact for ac2g endpoint "
                f"cid={pair.correlation_id} src_node={src_row['node_id']} "
                f"dst_node={dst_row['node_id']}"
            )


def expected_tensor_overlay(
    all_events: list[FakeKinetoEvent],
    selected_events: list[FakeKinetoEvent],
    execution_trace_path: Path,
) -> tuple[
    dict[str, TensorSignature],
    list[TensorEdgeSignature],
    list[TensorEdgeSignature],
]:
    execution_nodes = COMMON.load_execution_trace_nodes(execution_trace_path)
    matched_execution_node_by_event, match_errors, _match_warnings = (
        COMMON.match_execution_nodes_to_gpu_kernels(all_events, execution_nodes)
    )
    if match_errors:
        raise RuntimeError(
            "Source trace has ambiguous GPU kernel matches.\n" + "\n".join(match_errors)
        )

    execution_nodes_by_rf_id = COMMON.index_execution_trace_nodes_by_rf_id(
        execution_nodes
    )
    cpu_matched, _ambiguous = COMMON.match_execution_nodes_to_kineto_runtime(
        all_events,
        execution_nodes_by_rf_id,
    )
    matched_execution_node_by_event.update(cpu_matched)
    COMMON._propagate_parent_tensor_io(
        all_events,
        matched_execution_node_by_event,
        execution_nodes_by_rf_id,
    )

    tensor_signatures_by_key: dict[str, TensorSignature] = {}
    produced_tensor_keys: set[str] = set()
    input_edges: list[TensorEdgeSignature] = []
    output_edges: list[TensorEdgeSignature] = []

    def ensure_tensor_signature(
        tensor_tuple: tuple[int, int, int, int, int, str],
        *,
        type_name: str | None,
        shape: Any,
    ) -> str:
        tensor_key = COMMON.execution_tensor_key(list(tensor_tuple))
        if tensor_key is None:
            raise RuntimeError("Failed to compute execution tensor key")
        if tensor_key not in tensor_signatures_by_key:
            tensor_signatures_by_key[tensor_key] = TensorSignature(
                tensor_id=tensor_tuple[0],
                storage_id=tensor_tuple[1],
                offset=tensor_tuple[2],
                numel=tensor_tuple[3],
                itemsize=tensor_tuple[4],
                device=tensor_tuple[5],
                tensor_kind="WEIGHT",
                dtype=COMMON.execution_type_to_llamasim_dtype(type_name) or "",
                shape_json=json.dumps(
                    list(COMMON.execution_shape_to_tuple(shape)),
                    separators=(",", ":"),
                ),
            )
        return tensor_key

    for event in selected_events:
        execution_node = matched_execution_node_by_event.get(id(event))
        if execution_node is None:
            continue

        inputs = execution_node.get("inputs", {})
        for input_index, (type_name, shape, value) in enumerate(
            zip(
                inputs.get("types", []),
                inputs.get("shapes", []),
                inputs.get("values", []),
            )
        ):
            for entry in COMMON.flatten_execution_tensor_entries(
                value,
                path=str(input_index),
                shape=shape,
                type_name=type_name,
            ):
                tensor_key = ensure_tensor_signature(
                    entry["tensor_tuple"],
                    type_name=entry["type_name"],
                    shape=entry["shape"],
                )
                input_edges.append(
                    TensorEdgeSignature(
                        tensor_key=tensor_key,
                        event=event_signature(event),
                        edge_kind="data_input",
                    )
                )

        outputs = execution_node.get("outputs", {})
        for output_index, (type_name, shape, value) in enumerate(
            zip(
                outputs.get("types", []),
                outputs.get("shapes", []),
                outputs.get("values", []),
            )
        ):
            for entry in COMMON.flatten_execution_tensor_entries(
                value,
                path=str(output_index),
                shape=shape,
                type_name=type_name,
            ):
                tensor_key = ensure_tensor_signature(
                    entry["tensor_tuple"],
                    type_name=entry["type_name"],
                    shape=entry["shape"],
                )
                produced_tensor_keys.add(tensor_key)
                output_edges.append(
                    TensorEdgeSignature(
                        tensor_key=tensor_key,
                        event=event_signature(event),
                        edge_kind="data_output",
                    )
                )

    for tensor_key in produced_tensor_keys:
        signature = tensor_signatures_by_key[tensor_key]
        tensor_signatures_by_key[tensor_key] = TensorSignature(
            tensor_id=signature.tensor_id,
            storage_id=signature.storage_id,
            offset=signature.offset,
            numel=signature.numel,
            itemsize=signature.itemsize,
            device=signature.device,
            tensor_kind="CONTEXT",
            dtype=signature.dtype,
            shape_json=signature.shape_json,
        )

    return tensor_signatures_by_key, input_edges, output_edges


def verify_internal_consistency(bundle_dir: Path, report: VerificationReport) -> dict[str, Any]:
    manifest = load_manifest(bundle_dir)
    node_csv = bundle_dir / manifest.get("node_csv", "runtime_nodes.csv")
    edge_csv = bundle_dir / manifest.get("edge_csv", "runtime_edges.csv")
    tensor_csv = bundle_dir / manifest.get("tensor_csv", "pytorch_runtime_tensors.csv")
    timing_csv = bundle_dir / manifest.get("timing_csv", "ggml_profile_node_records.csv")
    dot_path = bundle_dir / manifest.get("step_dot_files", ["step_0_compute_graph.dot"])[0]

    node_rows = read_csv_rows(node_csv)
    edge_rows = read_csv_rows(edge_csv)
    tensor_rows = read_csv_rows(tensor_csv)
    timing_rows = read_csv_rows(timing_csv)

    report.infos.append(
        f"Loaded bundle with {len(node_rows)} runtime nodes, {len(edge_rows)} edges, and {len(tensor_rows)} tensors."
    )

    node_ids = [row["node_id"] for row in node_rows]
    if len(node_ids) != len(set(node_ids)):
        report.errors.append("runtime_nodes.csv contains duplicate node_id values")

    tensor_node_ids = [row["tensor_node_id"] for row in tensor_rows]
    if len(tensor_node_ids) != len(set(tensor_node_ids)):
        report.errors.append("pytorch_runtime_tensors.csv contains duplicate tensor_node_id values")

    expected_node_numbers = list(range(len(node_rows)))
    actual_node_numbers = sorted(parse_int(row["node_n"]) for row in node_rows)
    if actual_node_numbers != expected_node_numbers:
        report.errors.append("runtime_nodes.csv node_n values are not contiguous from 0")

    timing_by_node_id = {row["node_id"]: row for row in timing_rows}
    if len(timing_by_node_id) != len(timing_rows):
        report.errors.append("ggml_profile_node_records.csv contains duplicate node_id values")
    if set(timing_by_node_id) != set(node_ids):
        report.errors.append(
            "ggml_profile_node_records.csv does not have a one-to-one match with runtime_nodes.csv"
        )

    for row in node_rows:
        timing_row = timing_by_node_id.get(row["node_id"])
        if timing_row is None:
            continue
        for field in (
            "node_n",
            "node_name",
            "start_ns",
            "end_ns",
            "correlation_id",
            "linked_correlation_id",
            "resource_kind",
            "resource_id",
            "runtime_role",
        ):
            timing_field = "node_compute_time_ns" if field == "duration_ns" else field
            if field in row and field in timing_row and row[field] != timing_row[field]:
                report.errors.append(
                    f"Mismatch between runtime_nodes.csv and ggml_profile_node_records.csv for node_id={row['node_id']} field={field}"
                )

    valid_runtime_edge_kinds = {
        "thread_order",
        "stream_order",
        "submit",
        "wait",
    }
    valid_tensor_edge_kinds = {"data_input", "data_output"}
    known_node_ids = set(node_ids)
    known_tensor_ids = set(tensor_node_ids)
    for row in edge_rows:
        src = row["src_node_id"]
        dst = row["dst_node_id"]
        edge_kind = row["edge_kind"]
        if edge_kind in valid_runtime_edge_kinds:
            if src not in known_node_ids or dst not in known_node_ids:
                report.errors.append(
                    f"Runtime edge {edge_kind} points to non-runtime nodes: {src}->{dst}"
                )
        elif edge_kind == "data_input":
            if src not in known_tensor_ids or dst not in known_node_ids:
                report.errors.append(
                    f"data_input edge points to invalid endpoints: {src}->{dst}"
                )
        elif edge_kind == "data_output":
            if src not in known_node_ids or dst not in known_tensor_ids:
                report.errors.append(
                    f"data_output edge points to invalid endpoints: {src}->{dst}"
                )
        else:
            report.errors.append(f"Unknown edge_kind in runtime_edges.csv: {edge_kind}")

    tensor_input_counts: dict[str, int] = {}
    tensor_output_counts: dict[str, int] = {}
    for row in edge_rows:
        edge_kind = row["edge_kind"]
        if edge_kind == "data_input":
            tensor_input_counts[row["src_node_id"]] = (
                tensor_input_counts.get(row["src_node_id"], 0) + 1
            )
        elif edge_kind == "data_output":
            tensor_output_counts[row["dst_node_id"]] = (
                tensor_output_counts.get(row["dst_node_id"], 0) + 1
            )

    for row in tensor_rows:
        tensor_node_id = row["tensor_node_id"]
        expected_consumers = parse_int(row.get("consumer_count"))
        expected_producers = parse_int(row.get("producer_count"))
        if tensor_input_counts.get(tensor_node_id, 0) != expected_consumers:
            report.errors.append(
                f"consumer_count mismatch for tensor {tensor_node_id}: "
                f"csv={expected_consumers} edges={tensor_input_counts.get(tensor_node_id, 0)}"
            )
        if tensor_output_counts.get(tensor_node_id, 0) != expected_producers:
            report.errors.append(
                f"producer_count mismatch for tensor {tensor_node_id}: "
                f"csv={expected_producers} edges={tensor_output_counts.get(tensor_node_id, 0)}"
            )

    dot_node_ids, dot_edges = parse_dot_graph(dot_path)
    expected_dot_node_ids = known_node_ids | known_tensor_ids
    if dot_node_ids != expected_dot_node_ids:
        missing_from_dot = sample_sorted(expected_dot_node_ids - dot_node_ids)
        extra_in_dot = sample_sorted(dot_node_ids - expected_dot_node_ids)
        report.errors.append(
            "step_0_compute_graph.dot node set does not match the CSV exports "
            f"(missing={missing_from_dot}, extra={extra_in_dot})"
        )

    dot_edge_counter: dict[tuple[str, str, str], int] = {}
    for src, dst, label in dot_edges:
        if label in valid_runtime_edge_kinds:
            key = (src, dst, label)
        elif label is None and src in known_tensor_ids and dst in known_node_ids:
            key = (src, dst, "data_input")
        elif label is None and src in known_node_ids and dst in known_tensor_ids:
            key = (src, dst, "data_output")
        else:
            report.errors.append(
                f"Unable to classify DOT edge {src}->{dst} label={label!r}"
            )
            continue
        dot_edge_counter[key] = dot_edge_counter.get(key, 0) + 1

    csv_edge_counter: dict[tuple[str, str, str], int] = {}
    for row in edge_rows:
        key = (row["src_node_id"], row["dst_node_id"], row["edge_kind"])
        csv_edge_counter[key] = csv_edge_counter.get(key, 0) + 1

    if dot_edge_counter != csv_edge_counter:
        missing_edges = sample_sorted(set(csv_edge_counter) - set(dot_edge_counter))
        extra_edges = sample_sorted(set(dot_edge_counter) - set(csv_edge_counter))
        report.errors.append(
            "step_0_compute_graph.dot edge set does not match runtime_edges.csv "
            f"(missing={missing_edges}, extra={extra_edges})"
        )

    manifest_counts = {
        "node_count": len(node_rows),
        "gpu_node_count": sum(1 for row in node_rows if row["device_type"] != "CPU"),
        "cpu_node_count": sum(1 for row in node_rows if row["device_type"] == "CPU"),
        "tensor_count": len(tensor_rows),
        "data_input_edge_count": sum(
            1 for row in edge_rows if row["edge_kind"] == "data_input"
        ),
        "data_output_edge_count": sum(
            1 for row in edge_rows if row["edge_kind"] == "data_output"
        ),
        "submit_edge_count": sum(
            1 for row in edge_rows if row["edge_kind"] == "submit"
        ),
        "wait_edge_count": sum(1 for row in edge_rows if row["edge_kind"] == "wait"),
        "thread_order_edge_count": sum(
            1 for row in edge_rows if row["edge_kind"] == "thread_order"
        ),
        "stream_order_edge_count": sum(
            1 for row in edge_rows if row["edge_kind"] == "stream_order"
        ),
    }
    for field, actual in manifest_counts.items():
        expected = manifest.get(field)
        if expected is not None and expected != actual:
            report.errors.append(
                f"manifest.json field {field}={expected} does not match exported files ({actual})"
            )

    return {
        "manifest": manifest,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "tensor_rows": tensor_rows,
        "timing_rows": timing_rows,
    }


def verify_against_source(
    bundle_state: dict[str, Any],
    artifacts: SourceArtifacts,
    report: VerificationReport,
) -> None:
    raw_events = load_fake_kineto_events(artifacts)
    ac2g_endpoints = COMMON.parse_ac2g_flow_endpoints(artifacts.trace_json)
    ac2g_ids = ac2g_endpoints.paired_correlation_ids
    ac2g_matches = None
    exact_submit_cpu_event_ids: set[int] = set()
    if ac2g_ids:
        ac2g_matches = COMMON.match_ac2g_endpoints_to_raw_events(
            raw_events,
            ac2g_endpoints,
            trace_start_ns=0,
        )
        if ac2g_matches.errors:
            report.errors.append(
                "Source trace ac2g endpoints could not be matched exactly: "
                + "; ".join(ac2g_matches.errors[:5])
            )
            return
        exact_submit_cpu_event_ids = {
            id(pair.cpu_event) for pair in ac2g_matches.pairs_by_corr_id.values()
        }

    selected_events, selected_cpu_events, selected_gpu_events = (
        COMMON.select_llamasim_runtime_events(
            raw_events,
            extra_cpu_event_ids=exact_submit_cpu_event_ids,
            exclude_cpu_event_ids=(
                COMMON.find_profiler_sync_excluded_cpu_event_ids(raw_events)
            ),
        )
    )

    expected_event_counter: dict[EventSignature, int] = {}
    for event in selected_events:
        signature = event_signature(event)
        expected_event_counter[signature] = expected_event_counter.get(signature, 0) + 1

    bundle_event_counter: dict[EventSignature, int] = {}
    for row in bundle_state["node_rows"]:
        signature = bundle_row_signature(row)
        bundle_event_counter[signature] = bundle_event_counter.get(signature, 0) + 1

    if bundle_event_counter != expected_event_counter:
        missing = sample_sorted(set(expected_event_counter) - set(bundle_event_counter))
        extra = sample_sorted(set(bundle_event_counter) - set(expected_event_counter))
        report.errors.append(
            "Bundle runtime nodes do not match the selected source runtime events "
            f"(missing={missing}, extra={extra})"
        )
        return

    bundle_row_by_signature = {
        bundle_row_signature(row): row for row in bundle_state["node_rows"]
    }
    expected_edges, expected_roles = expected_runtime_roles(
        selected_events,
        selected_cpu_events,
        selected_gpu_events,
        raw_events,
        ac2g_correlation_ids=ac2g_ids or None,
        ac2g_endpoints=ac2g_endpoints if ac2g_ids else None,
        ac2g_matches=ac2g_matches,
    )
    if ac2g_matches is not None:
        verify_exact_submit_edges_against_ac2g(
            bundle_state,
            ac2g_matches,
            report,
        )

    bundle_runtime_edge_counter: dict[tuple[EventSignature, EventSignature, str], int] = {}
    node_signature_by_id = {
        row["node_id"]: bundle_row_signature(row) for row in bundle_state["node_rows"]
    }
    for row in bundle_state["edge_rows"]:
        edge_kind = row["edge_kind"]
        if edge_kind not in {"thread_order", "stream_order", "submit", "wait"}:
            continue
        src = node_signature_by_id[row["src_node_id"]]
        dst = node_signature_by_id[row["dst_node_id"]]
        key = (src, dst, edge_kind)
        bundle_runtime_edge_counter[key] = bundle_runtime_edge_counter.get(key, 0) + 1

    expected_runtime_edge_counter: dict[tuple[EventSignature, EventSignature, str], int] = {}
    for src, dst, edge_kind in expected_edges:
        key = (src, dst, edge_kind)
        expected_runtime_edge_counter[key] = (
            expected_runtime_edge_counter.get(key, 0) + 1
        )

    if bundle_runtime_edge_counter != expected_runtime_edge_counter:
        missing = sample_sorted(
            set(expected_runtime_edge_counter) - set(bundle_runtime_edge_counter)
        )
        extra = sample_sorted(
            set(bundle_runtime_edge_counter) - set(expected_runtime_edge_counter)
        )
        report.errors.append(
            "Bundle runtime edges do not match the source-derived runtime graph "
            f"(missing={missing}, extra={extra})"
        )

    for signature, expected_role in expected_roles.items():
        row = bundle_row_by_signature[signature]
        actual_role = row.get("runtime_role", "")
        if actual_role != expected_role:
            report.errors.append(
                f"runtime_role mismatch for {signature}: bundle={actual_role} source={expected_role}"
            )

    expected_tensors, expected_input_edges, expected_output_edges = expected_tensor_overlay(
        raw_events,
        selected_events,
        artifacts.execution_trace,
    )
    bundle_tensor_counter: dict[TensorSignature, int] = {}
    bundle_tensor_key_by_node_id: dict[str, str] = {}
    for row in bundle_state["tensor_rows"]:
        tensor_signature = tensor_signature_from_row(row)
        bundle_tensor_counter[tensor_signature] = (
            bundle_tensor_counter.get(tensor_signature, 0) + 1
        )
        bundle_tensor_key_by_node_id[row["tensor_node_id"]] = tensor_key_from_row(row)

    expected_tensor_counter: dict[TensorSignature, int] = {}
    for signature in expected_tensors.values():
        expected_tensor_counter[signature] = (
            expected_tensor_counter.get(signature, 0) + 1
        )

    if bundle_tensor_counter != expected_tensor_counter:
        missing = sample_sorted(set(expected_tensor_counter) - set(bundle_tensor_counter))
        extra = sample_sorted(set(bundle_tensor_counter) - set(expected_tensor_counter))
        report.errors.append(
            "Bundle tensor set does not match the source-derived tensor overlay "
            f"(missing={missing}, extra={extra})"
        )

    bundle_tensor_edge_counter: dict[TensorEdgeSignature, int] = {}
    for row in bundle_state["edge_rows"]:
        edge_kind = row["edge_kind"]
        if edge_kind == "data_input":
            tensor_key = bundle_tensor_key_by_node_id[row["src_node_id"]]
            event_sig = node_signature_by_id[row["dst_node_id"]]
        elif edge_kind == "data_output":
            tensor_key = bundle_tensor_key_by_node_id[row["dst_node_id"]]
            event_sig = node_signature_by_id[row["src_node_id"]]
        else:
            continue
        signature = TensorEdgeSignature(
            tensor_key=tensor_key,
            event=event_sig,
            edge_kind=edge_kind,
        )
        bundle_tensor_edge_counter[signature] = (
            bundle_tensor_edge_counter.get(signature, 0) + 1
        )

    expected_tensor_edge_counter: dict[TensorEdgeSignature, int] = {}
    for edge_signature in [*expected_input_edges, *expected_output_edges]:
        expected_tensor_edge_counter[edge_signature] = (
            expected_tensor_edge_counter.get(edge_signature, 0) + 1
        )

    if bundle_tensor_edge_counter != expected_tensor_edge_counter:
        missing = sample_sorted(
            set(expected_tensor_edge_counter) - set(bundle_tensor_edge_counter)
        )
        extra = sample_sorted(
            set(bundle_tensor_edge_counter) - set(expected_tensor_edge_counter)
        )
        report.errors.append(
            "Bundle tensor I/O edges do not match the source-derived execution trace overlay "
            f"(missing={missing}, extra={extra})"
        )

    report.infos.append(
        "Source-backed verification completed against trace.csv, trace.json, kineto_map.csv, and execution_trace.json."
    )


def verify_bundle(
    bundle: str | Path,
    *,
    source_artifacts: SourceArtifacts | None = None,
) -> VerificationReport:
    report = VerificationReport()
    bundle_dir = resolve_bundle_dir(bundle)
    bundle_state = verify_internal_consistency(bundle_dir, report)
    if source_artifacts is None:
        if report.ok:
            report.warnings.append(
                "Source-backed verification was not run. Internal consistency passed, "
                "but this alone cannot prove the bundle covers the exact source trace."
            )
        else:
            report.warnings.append(
                "Source-backed verification was not run."
            )
        return report

    verify_against_source(bundle_state, source_artifacts, report)
    return report


def main() -> None:
    args = parse_args()
    source_artifacts = maybe_resolve_source_artifacts(
        profile_dir=args.profile_dir,
        trace_csv=args.trace_csv,
        trace_json=args.trace_json,
        kineto_map=args.kineto_map,
        execution_trace=args.execution_trace,
    )
    report = verify_bundle(args.bundle, source_artifacts=source_artifacts)
    payload = {
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "infos": report.infos,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"[{status}] verify_llamasim_bundle")
        for message in report.infos:
            print(f"info: {message}")
        for message in report.warnings:
            print(f"warning: {message}")
        for message in report.errors:
            print(f"error: {message}")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
