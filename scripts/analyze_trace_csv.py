#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TraceRow:
    event_index: int
    event_id: str
    name: str
    start_us: float
    end_us: float
    duration_us: float
    device_type: str
    device_resource_id: str
    thread: str
    depth: int
    cpu_parent_id: str
    cpu_children: int
    kernel_count: int
    is_user_annotation: bool


@dataclass(frozen=True)
class OverlapStats:
    total_rows: int
    overlapping_rows: int
    overlap_pairs: int
    peak_concurrency: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a PyTorch profiler CSV exported with export_csv()."
    )
    parser.add_argument("trace_csv", help="Path to the exported profiler CSV.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many rows to print for each section.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def parse_int(value: str) -> int:
    return int(value) if value else 0


def parse_float(value: str) -> float:
    return float(value) if value else 0.0


def load_rows(path: Path) -> list[TraceRow]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            TraceRow(
                event_index=parse_int(row["event_index"]),
                event_id=row["id"],
                name=row["name"],
                start_us=parse_float(row["start_us"]),
                end_us=parse_float(row["end_us"]),
                duration_us=parse_float(row["duration_us"]),
                device_type=row["device_type"],
                device_resource_id=row["device_resource_id"],
                thread=row["thread"],
                depth=parse_int(row["depth"]),
                cpu_parent_id=row["cpu_parent_id"],
                cpu_children=parse_int(row["cpu_children"]),
                kernel_count=parse_int(row["kernel_count"]),
                is_user_annotation=parse_bool(row["is_user_annotation"]),
            )
            for row in reader
        ]


def is_leaf_cpu(row: TraceRow) -> bool:
    return row.device_type == "CPU" and row.cpu_children == 0


def is_top_level_cpu(row: TraceRow) -> bool:
    return row.device_type == "CPU" and row.depth == 0


def is_cuda_activity(row: TraceRow) -> bool:
    return row.device_type == "CUDA"


def is_true_cuda_kernel(row: TraceRow) -> bool:
    if row.device_type != "CUDA":
        return False
    if row.is_user_annotation:
        return False
    if row.name.startswith("Memcpy") or row.name.startswith("Memset"):
        return False
    return True


def sort_by_start(rows: Iterable[TraceRow]) -> list[TraceRow]:
    return sorted(rows, key=lambda row: (row.start_us, -row.end_us, row.event_index))


def overlap_stats(rows: Iterable[TraceRow]) -> OverlapStats:
    sorted_rows = sort_by_start(rows)
    active: list[TraceRow] = []
    overlapping_indices: set[int] = set()
    overlap_pairs = 0
    peak_concurrency = 0

    for row in sorted_rows:
        active = [candidate for candidate in active if candidate.end_us > row.start_us]
        if active:
            overlapping_indices.add(row.event_index)
            overlap_pairs += len(active)
            for candidate in active:
                overlapping_indices.add(candidate.event_index)
        active.append(row)
        if len(active) > peak_concurrency:
            peak_concurrency = len(active)

    return OverlapStats(
        total_rows=len(sorted_rows),
        overlapping_rows=len(overlapping_indices),
        overlap_pairs=overlap_pairs,
        peak_concurrency=peak_concurrency,
    )


def find_overlap_pairs(
    rows: Iterable[TraceRow], limit: int = 5
) -> list[tuple[TraceRow, TraceRow]]:
    sorted_rows = sort_by_start(rows)
    active: list[TraceRow] = []
    pairs: list[tuple[TraceRow, TraceRow]] = []
    for row in sorted_rows:
        active = [candidate for candidate in active if candidate.end_us > row.start_us]
        for candidate in active:
            pairs.append((candidate, row))
            if len(pairs) >= limit:
                return pairs
        active.append(row)
    return pairs


def overlap_stats_by_key(
    rows: Iterable[TraceRow], key_fn
) -> dict[str, OverlapStats]:
    groups: dict[str, list[TraceRow]] = {}
    for row in rows:
        key = key_fn(row)
        groups.setdefault(key, []).append(row)
    return {key: overlap_stats(group_rows) for key, group_rows in groups.items()}


def print_section(title: str, rows: Iterable[TraceRow], top_n: int) -> None:
    print(title)
    for row in list(rows)[:top_n]:
        print(
            "  "
            f"start={row.start_us:12.3f}us "
            f"dur={row.duration_us:10.3f}us "
            f"id={row.event_id:>6} "
            f"depth={row.depth:>2} "
            f"thread={row.thread:>4} "
            f"stream={row.device_resource_id:>4} "
            f"name={row.name}"
        )
    print()


def print_overlap_summary(title: str, stats: OverlapStats) -> None:
    print(title)
    print(f"  total_rows={stats.total_rows}")
    print(f"  overlapping_rows={stats.overlapping_rows}")
    print(f"  overlap_pairs={stats.overlap_pairs}")
    print(f"  peak_concurrency={stats.peak_concurrency}")
    print()


def print_overlap_examples(
    title: str, pairs: list[tuple[TraceRow, TraceRow]]
) -> None:
    print(title)
    if not pairs:
        print("  none")
        print()
        return
    for left, right in pairs:
        print(
            "  "
            f"[{left.start_us:12.3f}, {left.end_us:12.3f}] "
            f"id={left.event_id:>6} thread={left.thread:>4} stream={left.device_resource_id:>4} "
            f"name={left.name}"
        )
        print(
            "  "
            f"[{right.start_us:12.3f}, {right.end_us:12.3f}] "
            f"id={right.event_id:>6} thread={right.thread:>4} stream={right.device_resource_id:>4} "
            f"name={right.name}"
        )
        print()


def main() -> None:
    args = parse_args()
    trace_csv = Path(args.trace_csv)
    rows = load_rows(trace_csv)

    top_level_cpu = sort_by_start(row for row in rows if is_top_level_cpu(row))
    leaf_cpu = sort_by_start(row for row in rows if is_leaf_cpu(row))
    cuda_kernels = sort_by_start(row for row in rows if is_true_cuda_kernel(row))
    cuda_activities = sort_by_start(row for row in rows if is_cuda_activity(row))

    print(f"trace_csv={trace_csv}")
    print(f"total_rows={len(rows)}")
    print(f"top_level_cpu_rows={len(top_level_cpu)}")
    print(f"leaf_cpu_rows={len(leaf_cpu)}")
    print(f"cuda_activity_rows={len(cuda_activities)}")
    print(f"true_cuda_kernel_rows={len(cuda_kernels)}")
    print()

    print_section("Top-Level CPU Rows", top_level_cpu, args.top_n)
    print_section("Leaf CPU Rows", leaf_cpu, args.top_n)
    print_section("True CUDA Kernel Rows", cuda_kernels, args.top_n)

    cpu_overlap = overlap_stats(leaf_cpu)
    print_overlap_summary("Leaf CPU Overlap", cpu_overlap)
    print_overlap_examples("Leaf CPU Overlap Examples", find_overlap_pairs(leaf_cpu))

    cpu_overlap_by_thread = overlap_stats_by_key(leaf_cpu, lambda row: row.thread)
    if cpu_overlap_by_thread:
        print("Leaf CPU Overlap By Thread")
        for thread, stats in sorted(
            cpu_overlap_by_thread.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        ):
            print(
                "  "
                f"thread={thread} total={stats.total_rows} "
                f"overlapping_rows={stats.overlapping_rows} "
                f"peak_concurrency={stats.peak_concurrency}"
            )
        print()

    kernel_overlap = overlap_stats(cuda_kernels)
    print_overlap_summary("True CUDA Kernel Overlap", kernel_overlap)
    print_overlap_examples(
        "True CUDA Kernel Overlap Examples", find_overlap_pairs(cuda_kernels)
    )

    kernel_overlap_by_stream = overlap_stats_by_key(
        cuda_kernels, lambda row: row.device_resource_id
    )
    if kernel_overlap_by_stream:
        print("True CUDA Kernel Overlap By Stream")
        for stream, stats in sorted(
            kernel_overlap_by_stream.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        ):
            print(
                "  "
                f"stream={stream} total={stats.total_rows} "
                f"overlapping_rows={stats.overlapping_rows} "
                f"peak_concurrency={stats.peak_concurrency}"
            )
        print()


if __name__ == "__main__":
    main()
