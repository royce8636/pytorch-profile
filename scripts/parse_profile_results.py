#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileSummary:
    peak_vram_bytes: int
    e2e_time_us: float
    cpu_busy_time_us: float | None = None
    gpu_busy_time_us: float | None = None

    @property
    def peak_vram_kb(self) -> int:
        return self.peak_vram_bytes // 1024

    @property
    def peak_vram_mb(self) -> float:
        return self.peak_vram_bytes / (1024**2)

    @property
    def peak_vram_gb(self) -> float:
        return self.peak_vram_bytes / (1024**3)

    @property
    def e2e_time_ms(self) -> float:
        return self.e2e_time_us / 1000.0

    @property
    def e2e_time_s(self) -> float:
        return self.e2e_time_us / 1_000_000.0

    @property
    def cpu_busy_time_ms(self) -> float | None:
        if self.cpu_busy_time_us is None:
            return None
        return self.cpu_busy_time_us / 1000.0

    @property
    def cpu_busy_time_s(self) -> float | None:
        if self.cpu_busy_time_us is None:
            return None
        return self.cpu_busy_time_us / 1_000_000.0

    @property
    def gpu_busy_time_ms(self) -> float | None:
        if self.gpu_busy_time_us is None:
            return None
        return self.gpu_busy_time_us / 1000.0

    @property
    def gpu_busy_time_s(self) -> float | None:
        if self.gpu_busy_time_us is None:
            return None
        return self.gpu_busy_time_us / 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a saved SDXL/Qwen profiler output directory or llama bundle "
            "and print peak VRAM plus end-to-end timing."
        )
    )
    parser.add_argument(
        "path",
        help=(
            "Profile output directory, llama bundle directory, manifest.json, "
            "trace.csv, or trace.json."
        ),
    )
    parser.add_argument(
        "--vram-kind",
        choices=("reserved", "allocated"),
        default="reserved",
        help=(
            "Which peak VRAM value to report from profiler memory events. "
            "Defaults to reserved."
        ),
    )
    parser.add_argument(
        "--time-source",
        choices=("auto", "trace", "calibration-profiled", "calibration-unprofiled"),
        default="auto",
        help=(
            "Where to read end-to-end timing from. Defaults to profiler trace "
            "data when available."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _iter_candidate_profile_dirs(path: Path) -> list[Path]:
    if path.is_dir():
        if path.name == "llama_bundle":
            return [path, path.parent]
        return [path]
    if path.name == "manifest.json":
        return [path.parent, path.parent.parent]
    return [path.parent]


def _find_manifest_path(path: Path) -> Path | None:
    if path.is_file() and path.name == "manifest.json":
        return path
    for base in _iter_candidate_profile_dirs(path):
        direct = base / "manifest.json"
        if direct.is_file():
            return direct
        candidate = base / "llama_bundle" / "manifest.json"
        if candidate.is_file():
            return candidate
    return None


def _find_trace_csv_path(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".csv":
        return path
    for base in _iter_candidate_profile_dirs(path):
        csv_paths = sorted(base.glob("*_trace.csv"))
        if csv_paths:
            return csv_paths[0]
    return None


def _is_profiler_trace_json_path(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix == ".json"
        and path.name != "manifest.json"
        and "execution_trace" not in path.name
    )


def _find_trace_json_path(path: Path) -> Path | None:
    if _is_profiler_trace_json_path(path):
        return path

    trace_csv_path = _find_trace_csv_path(path)
    if trace_csv_path is not None:
        sibling = trace_csv_path.with_suffix(".json")
        if _is_profiler_trace_json_path(sibling):
            return sibling

    for base in _iter_candidate_profile_dirs(path):
        direct = base / "trace.json"
        if _is_profiler_trace_json_path(direct):
            return direct
        for candidate in sorted(base.glob("*_trace.json")):
            if _is_profiler_trace_json_path(candidate):
                return candidate
    return None


def _find_runtime_nodes_csv_path(path: Path) -> Path | None:
    if path.is_file() and path.name == "runtime_nodes.csv":
        return path

    manifest_path = _find_manifest_path(path)
    if manifest_path is not None:
        manifest = _load_json(manifest_path)
        node_csv = manifest.get("node_csv") or "runtime_nodes.csv"
        candidate = manifest_path.parent / str(node_csv)
        if candidate.is_file():
            return candidate

    for base in _iter_candidate_profile_dirs(path):
        direct = base / "runtime_nodes.csv"
        if direct.is_file():
            return direct
        candidate = base / "llama_bundle" / "runtime_nodes.csv"
        if candidate.is_file():
            return candidate
    return None


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _parse_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _looks_like_run_scope_name(name: str) -> bool:
    return name.endswith("_run") and not name.startswith(("ws_", "## "))


def _row_depth(row: dict[str, str]) -> int:
    depth = _parse_int(row.get("depth"))
    if depth is None:
        return 0
    return depth


def _parse_trace_scope_duration_us(
    trace_csv_path: Path,
    *,
    require_metadata: bool = False,
) -> float:
    candidates: list[tuple[bool, bool, float]] = []
    with trace_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_user_annotation_column = "is_user_annotation" in (reader.fieldnames or [])
        for row in reader:
            name = row.get("name") or ""
            if not _looks_like_run_scope_name(name):
                continue
            if has_user_annotation_column and not _parse_bool(
                row.get("is_user_annotation")
            ):
                continue
            duration_us = _parse_float(row.get("duration_us"))
            if duration_us <= 0.0:
                continue
            has_metadata = bool((row.get("metadata_json") or "").strip())
            if require_metadata and not has_metadata:
                continue
            candidates.append(
                (
                    has_metadata,
                    _row_depth(row) == 0,
                    duration_us,
                )
            )
    if candidates:
        return max(candidates)[2]
    raise RuntimeError(
        f"Did not find a profiled *_run row in profiler CSV: {trace_csv_path}"
    )


def _is_device_memory_event_args(args: dict) -> bool:
    device_id = _parse_int(args.get("Device Id"))
    if device_id is not None:
        return device_id >= 0

    device_type = _parse_int(args.get("Device Type"))
    if device_type is not None:
        return device_type != 0

    return False


def _parse_peak_vram_from_trace_json(trace_json_path: Path, vram_kind: str) -> int:
    trace = _load_json(trace_json_path)
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise RuntimeError(
            f"{trace_json_path} does not contain Chrome traceEvents memory data"
        )

    total_key = "Total Reserved" if vram_kind == "reserved" else "Total Allocated"
    peak: int | None = None
    for event in events:
        if not isinstance(event, dict) or event.get("name") != "[memory]":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or not _is_device_memory_event_args(args):
            continue
        value = _parse_int(args.get(total_key))
        if value is None:
            continue
        peak = value if peak is None else max(peak, value)

    if peak is None:
        raise RuntimeError(
            f"{trace_json_path} does not contain device {total_key} memory events"
        )
    return peak


def _resolve_peak_vram_bytes(path: Path, vram_kind: str) -> int:
    trace_json_path = _find_trace_json_path(path)
    trace_error: RuntimeError | None = None
    if trace_json_path is not None:
        try:
            return _parse_peak_vram_from_trace_json(trace_json_path, vram_kind)
        except RuntimeError as exc:
            trace_error = exc

    manifest_path = _find_manifest_path(path)
    if manifest_path is None:
        if trace_error is not None:
            raise trace_error
        raise RuntimeError(
            "Could not find profiler trace memory events or manifest.json. "
            "Pass a profile output directory containing a *_trace.json file."
        )
    manifest = _load_json(manifest_path)
    key = (
        "vram_peak_reserved_bytes"
        if vram_kind == "reserved"
        else "vram_peak_allocated_bytes"
    )
    value = manifest.get(key)
    if value is None:
        raise RuntimeError(f"{manifest_path} does not contain {key}")
    return int(value)


def _duration_ns_from_row(row: dict[str, str]) -> float:
    try:
        return float(row.get("duration_ns") or 0.0)
    except ValueError:
        return 0.0


def _is_cpu_runtime_row(row: dict[str, str]) -> bool:
    return row.get("device_type") == "CPU" or row.get("resource_kind") == "cpu_thread"


def _is_gpu_runtime_row(row: dict[str, str]) -> bool:
    device_type = row.get("device_type")
    return (
        device_type in {"CUDA", "GPU", "XPU"}
        or row.get("resource_kind") == "gpu_stream"
        or row.get("node_kind") == "gpu_runtime"
    )


def _resolve_busy_times_us(path: Path) -> tuple[float | None, float | None]:
    runtime_nodes_csv_path = _find_runtime_nodes_csv_path(path)
    if runtime_nodes_csv_path is None:
        return None, None

    cpu_busy_ns = 0.0
    gpu_busy_ns = 0.0
    with runtime_nodes_csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            duration_ns = _duration_ns_from_row(row)
            if _is_cpu_runtime_row(row):
                cpu_busy_ns += duration_ns
            elif _is_gpu_runtime_row(row):
                gpu_busy_ns += duration_ns

    return cpu_busy_ns / 1000.0, gpu_busy_ns / 1000.0


def _find_calibration_path(path: Path) -> Path | None:
    for base in _iter_candidate_profile_dirs(path):
        direct = base / "calibration.json"
        if direct.is_file():
            return direct
        candidate = base / "llama_bundle" / "calibration.json"
        if candidate.is_file():
            return candidate
    return None


def _resolve_calibration_time_us(path: Path, key: str) -> float:
    calibration_path = _find_calibration_path(path)
    if calibration_path is None:
        raise RuntimeError(f"Could not find calibration.json for {key}")
    calibration = _load_json(calibration_path)
    value = calibration.get(key)
    if value is None:
        raise RuntimeError(f"{calibration_path} does not contain {key}")
    return float(value) / 1000.0


def _resolve_trace_e2e_time_us(
    path: Path,
    *,
    require_metadata: bool = False,
) -> float:
    trace_csv_path = _find_trace_csv_path(path)
    if trace_csv_path is None:
        raise RuntimeError("Could not find a *_trace.csv file to derive trace timing")
    return _parse_trace_scope_duration_us(
        trace_csv_path,
        require_metadata=require_metadata,
    )


def _resolve_e2e_time_us(path: Path, time_source: str) -> float:
    if time_source == "trace":
        return _resolve_trace_e2e_time_us(path)
    if time_source == "calibration-profiled":
        return _resolve_calibration_time_us(path, "profiled_wall_ns")
    if time_source == "calibration-unprofiled":
        return _resolve_calibration_time_us(path, "unprofiled_wall_ns")
    if time_source != "auto":
        raise ValueError(f"Unsupported time_source: {time_source}")

    trace_error: RuntimeError | None = None
    try:
        return _resolve_trace_e2e_time_us(path, require_metadata=True)
    except RuntimeError as exc:
        trace_error = exc

    try:
        return _resolve_calibration_time_us(path, "unprofiled_wall_ns")
    except RuntimeError:
        pass

    try:
        return _resolve_trace_e2e_time_us(path)
    except RuntimeError as exc:
        trace_error = exc

    try:
        return _resolve_calibration_time_us(path, "profiled_wall_ns")
    except RuntimeError:
        if trace_error is not None:
            raise trace_error
        raise


def summarize_profile(
    path: Path,
    *,
    vram_kind: str,
    time_source: str = "auto",
) -> ProfileSummary:
    cpu_busy_time_us, gpu_busy_time_us = _resolve_busy_times_us(path)
    return ProfileSummary(
        peak_vram_bytes=_resolve_peak_vram_bytes(path, vram_kind),
        e2e_time_us=_resolve_e2e_time_us(path, time_source),
        cpu_busy_time_us=cpu_busy_time_us,
        gpu_busy_time_us=gpu_busy_time_us,
    )


def _format_optional_time_us(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.3f}"


def _format_optional_time_ms(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.3f}"


def _format_optional_time_s(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6f}"


def print_summary(summary: ProfileSummary) -> None:
    print(f"peak_vram_kb: {summary.peak_vram_kb}")
    print(f"peak_vram_mb: {summary.peak_vram_mb:.2f}")
    print(f"peak_vram_gb: {summary.peak_vram_gb:.4f}")
    print(f"e2e_time_us: {summary.e2e_time_us:.3f}")
    print(f"e2e_time_ms: {summary.e2e_time_ms:.3f}")
    print(f"e2e_time_s: {summary.e2e_time_s:.6f}")
    print(f"cpu_busy_time_us: {_format_optional_time_us(summary.cpu_busy_time_us)}")
    print(f"cpu_busy_time_ms: {_format_optional_time_ms(summary.cpu_busy_time_ms)}")
    print(f"cpu_busy_time_s: {_format_optional_time_s(summary.cpu_busy_time_s)}")
    print(f"gpu_busy_time_us: {_format_optional_time_us(summary.gpu_busy_time_us)}")
    print(f"gpu_busy_time_ms: {_format_optional_time_ms(summary.gpu_busy_time_ms)}")
    print(f"gpu_busy_time_s: {_format_optional_time_s(summary.gpu_busy_time_s)}")


def main() -> None:
    args = parse_args()
    summary = summarize_profile(
        Path(args.path),
        vram_kind=args.vram_kind,
        time_source=args.time_source,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
