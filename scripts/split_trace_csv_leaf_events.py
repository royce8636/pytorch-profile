#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from html import escape
from pathlib import Path


CPU_DEVICE_TYPE = "CPU"
GPU_DEVICE_TYPE = "CUDA"
DEFAULT_TOP_N = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split an exported PyTorch profiler CSV into leaf CPU rows and true GPU kernel rows."
        )
    )
    parser.add_argument("trace_csv", help="Path to the profiler CSV export.")
    parser.add_argument(
        "--cpu-out",
        default=None,
        help="Output CSV path for leaf CPU rows. Defaults to <trace>_leaf_cpu.csv.",
    )
    parser.add_argument(
        "--gpu-out",
        default=None,
        help=(
            "Output CSV path for true GPU kernel rows. "
            "Defaults to <trace>_leaf_gpu.csv."
        ),
    )
    parser.add_argument(
        "--graph-out",
        default=None,
        help=(
            "Output SVG path for the summary bar chart. "
            "Defaults to <trace>_leaf_activity_summary.svg."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of activities to print and graph. Defaults to {DEFAULT_TOP_N}.",
    )
    return parser.parse_args()


def parse_int(value: str) -> int:
    return int(value) if value else 0


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def parse_float(value: str) -> float:
    return float(value) if value else 0.0


def is_leaf_cpu(row: dict[str, str]) -> bool:
    return row.get("device_type", "") == CPU_DEVICE_TYPE and parse_int(
        row.get("cpu_children", "")
    ) == 0


def is_true_gpu_kernel(row: dict[str, str]) -> bool:
    if row.get("device_type", "") != GPU_DEVICE_TYPE:
        return False
    if parse_bool(row.get("is_user_annotation", "false")):
        return False
    name = row.get("name", "")
    if name.startswith("Memcpy") or name.startswith("Memset"):
        return False
    return True


def default_output_path(trace_csv: Path, suffix: str) -> Path:
    return trace_csv.with_name(f"{trace_csv.stem}_{suffix}.csv")


def default_graph_path(trace_csv: Path) -> Path:
    return trace_csv.with_name(f"{trace_csv.stem}_leaf_activity_summary.svg")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def total_duration_us(rows: list[dict[str, str]]) -> float:
    return sum(parse_float(row.get("duration_us", "")) for row in rows)


def merge_intervals(
    rows: list[dict[str, str]],
) -> list[tuple[float, float]]:
    intervals = sorted(
        (
            parse_float(row.get("start_us", "")),
            parse_float(row.get("end_us", "")),
        )
        for row in rows
    )
    merged: list[tuple[float, float]] = []
    for start_us, end_us in intervals:
        if not merged or start_us > merged[-1][1]:
            merged.append((start_us, end_us))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end_us))
    return merged


def merged_wall_time_us(rows: list[dict[str, str]]) -> float:
    return sum(end_us - start_us for start_us, end_us in merge_intervals(rows))


def overlap_wall_time_us(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
) -> float:
    left = merge_intervals(left_rows)
    right = merge_intervals(right_rows)
    overlap_us = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        left_start, left_end = left[i]
        right_start, right_end = right[j]
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_end > overlap_start:
            overlap_us += overlap_end - overlap_start
        if left_end <= right_end:
            i += 1
        else:
            j += 1
    return overlap_us


def wall_time_us(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    start_us = min(parse_float(row.get("start_us", "")) for row in rows)
    end_us = max(parse_float(row.get("end_us", "")) for row in rows)
    return end_us - start_us


def summarize_rows_by_name(rows: list[dict[str, str]]) -> list[tuple[str, int, float]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for row in rows:
        name = row.get("name", "")
        totals[name][0] += 1
        totals[name][1] += parse_float(row.get("duration_us", ""))
    return [
        (name, int(count), float(total_us))
        for name, (count, total_us) in totals.items()
    ]


def print_top_summary(
    title: str,
    rows: list[tuple[str, int, float]],
    *,
    key_fn,
    top_n: int = 5,
) -> None:
    print(title)
    sorted_rows = sorted(rows, key=key_fn)
    for name, count, total_us in sorted_rows[:top_n]:
        avg_us = total_us / count if count else 0.0
        print(
            f"  name={name} count={count} total_us={total_us:.3f} avg_us={avg_us:.3f}"
        )
    if not sorted_rows:
        print("  none")
    print()


def top_rows(
    rows: list[tuple[str, int, float]],
    *,
    key_fn,
    top_n: int,
) -> list[tuple[str, int, float]]:
    return sorted(rows, key=key_fn)[:top_n]


def truncate_label(label: str, max_chars: int = 44) -> str:
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 3]}..."


def write_summary_svg(
    path: Path,
    *,
    cpu_count_rows: list[tuple[str, int, float]],
    cpu_time_rows: list[tuple[str, int, float]],
    gpu_count_rows: list[tuple[str, int, float]],
    gpu_time_rows: list[tuple[str, int, float]],
    top_n: int,
) -> None:
    panel_width = 760
    panel_height = 360
    margin = 30
    gap_x = 20
    gap_y = 20
    width = panel_width * 2 + gap_x + margin * 2
    height = panel_height * 2 + gap_y + margin * 2 + 30
    bar_left = 300
    bar_right = 730
    row_height = 28
    bar_height = 16

    def format_metric(title: str, count: int, total_us: float) -> str:
        avg_us = total_us / count if count else 0.0
        if "Count" in title:
            return f"count={count} total_us={total_us:.1f} avg_us={avg_us:.1f}"
        return f"total_us={total_us:.1f} count={count} avg_us={avg_us:.1f}"

    def value_for_panel(title: str, row: tuple[str, int, float]) -> float:
        _, count, total_us = row
        return float(count) if "Count" in title else total_us

    def render_panel(
        title: str,
        rows: list[tuple[str, int, float]],
        x: int,
        y: int,
        bar_color: str,
    ) -> str:
        parts = [
            f'<g transform="translate({x},{y})">',
            (
                f'<rect x="0" y="0" width="{panel_width}" height="{panel_height}" '
                'rx="10" fill="#ffffff" stroke="#d0d7de" />'
            ),
            (
                f'<text x="20" y="28" font-size="18" font-family="monospace" '
                f'font-weight="bold" fill="#1f2328">{escape(title)}</text>'
            ),
        ]
        if not rows:
            parts.append(
                '<text x="20" y="60" font-size="14" font-family="monospace" '
                'fill="#57606a">none</text>'
            )
            parts.append("</g>")
            return "\n".join(parts)

        max_value = max(value_for_panel(title, row) for row in rows) or 1.0
        for index, row in enumerate(rows):
            name, count, total_us = row
            value = value_for_panel(title, row)
            top = 60 + index * row_height
            bar_width = int((value / max_value) * (bar_right - bar_left))
            parts.append(
                f'<text x="20" y="{top + 12}" font-size="12" '
                f'font-family="monospace" fill="#24292f">{escape(truncate_label(name))}</text>'
            )
            parts.append(
                f'<rect x="{bar_left}" y="{top}" width="{bar_width}" height="{bar_height}" '
                f'fill="{bar_color}" rx="4" />'
            )
            parts.append(
                f'<text x="{bar_right + 10}" y="{top + 12}" font-size="11" '
                f'font-family="monospace" fill="#57606a">'
                f"{escape(format_metric(title, count, total_us))}</text>"
            )
        parts.append("</g>")
        return "\n".join(parts)

    svg_parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            'viewBox="0 0 {width} {height}">'
        ).format(width=width, height=height),
        '<rect width="100%" height="100%" fill="#f6f8fa" />',
        (
            f'<text x="{margin}" y="24" font-size="20" font-family="monospace" '
            'font-weight="bold" fill="#1f2328">'
            f"{escape(f'Leaf Event Activity Summary (top {top_n})')}</text>"
        ),
        render_panel(
            f"Leaf CPU Top {top_n} by Count",
            cpu_count_rows,
            margin,
            margin + 20,
            "#0969da",
        ),
        render_panel(
            f"Leaf CPU Top {top_n} by Total Time (us)",
            cpu_time_rows,
            margin + panel_width + gap_x,
            margin + 20,
            "#1a7f37",
        ),
        render_panel(
            f"True GPU Top {top_n} by Count",
            gpu_count_rows,
            margin,
            margin + 20 + panel_height + gap_y,
            "#bf8700",
        ),
        render_panel(
            f"True GPU Top {top_n} by Total Time (us)",
            gpu_time_rows,
            margin + panel_width + gap_x,
            margin + 20 + panel_height + gap_y,
            "#8250df",
        ),
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_parts))


def main() -> None:
    args = parse_args()
    trace_csv = Path(args.trace_csv)
    top_n = args.top_n
    cpu_out = Path(args.cpu_out) if args.cpu_out is not None else default_output_path(
        trace_csv, "leaf_cpu"
    )
    gpu_out = Path(args.gpu_out) if args.gpu_out is not None else default_output_path(
        trace_csv, "leaf_gpu"
    )
    graph_out = (
        Path(args.graph_out)
        if args.graph_out is not None
        else default_graph_path(trace_csv)
    )

    with trace_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Trace CSV at {trace_csv} has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    leaf_cpu_rows = [row for row in rows if is_leaf_cpu(row)]
    leaf_gpu_rows = [row for row in rows if is_true_gpu_kernel(row)]
    selected_rows = leaf_cpu_rows + leaf_gpu_rows

    write_rows(cpu_out, fieldnames, leaf_cpu_rows)
    write_rows(gpu_out, fieldnames, leaf_gpu_rows)

    print(f"trace_csv={trace_csv}")
    print(f"leaf_cpu_out={cpu_out}")
    print(f"leaf_gpu_out={gpu_out}")
    print(f"summary_graph_out={graph_out}")
    print(f"leaf_cpu_rows={len(leaf_cpu_rows)}")
    print(f"leaf_gpu_rows={len(leaf_gpu_rows)}")
    print(f"total_leaf_cpu_time_us={total_duration_us(leaf_cpu_rows):.3f}")
    print(f"total_leaf_gpu_time_us={total_duration_us(leaf_gpu_rows):.3f}")
    print(f"leaf_cpu_active_wall_time_us={merged_wall_time_us(leaf_cpu_rows):.3f}")
    print(f"leaf_gpu_active_wall_time_us={merged_wall_time_us(leaf_gpu_rows):.3f}")
    print(
        f"cpu_gpu_overlap_wall_time_us="
        f"{overlap_wall_time_us(leaf_cpu_rows, leaf_gpu_rows):.3f}"
    )
    print(f"e2e_compute_wall_time_us={wall_time_us(selected_rows):.3f}")
    print()

    cpu_summary = summarize_rows_by_name(leaf_cpu_rows)
    gpu_summary = summarize_rows_by_name(leaf_gpu_rows)

    cpu_count_rows = top_rows(
        cpu_summary,
        key_fn=lambda row: (-row[1], -row[2], row[0]),
        top_n=top_n,
    )
    cpu_time_rows = top_rows(
        cpu_summary,
        key_fn=lambda row: (-row[2], -row[1], row[0]),
        top_n=top_n,
    )
    gpu_count_rows = top_rows(
        gpu_summary,
        key_fn=lambda row: (-row[1], -row[2], row[0]),
        top_n=top_n,
    )
    gpu_time_rows = top_rows(
        gpu_summary,
        key_fn=lambda row: (-row[2], -row[1], row[0]),
        top_n=top_n,
    )

    write_summary_svg(
        graph_out,
        cpu_count_rows=cpu_count_rows,
        cpu_time_rows=cpu_time_rows,
        gpu_count_rows=gpu_count_rows,
        gpu_time_rows=gpu_time_rows,
        top_n=top_n,
    )

    print_top_summary(
        f"Top {top_n} Most Frequent Leaf CPU Activities",
        cpu_count_rows,
        key_fn=lambda row: 0,
        top_n=top_n,
    )
    print_top_summary(
        f"Top {top_n} Longest Accumulated Leaf CPU Activities",
        cpu_time_rows,
        key_fn=lambda row: 0,
        top_n=top_n,
    )
    print_top_summary(
        f"Top {top_n} Most Frequent True GPU Activities",
        gpu_count_rows,
        key_fn=lambda row: 0,
        top_n=top_n,
    )
    print_top_summary(
        f"Top {top_n} Longest Accumulated True GPU Activities",
        gpu_time_rows,
        key_fn=lambda row: 0,
        top_n=top_n,
    )


if __name__ == "__main__":
    main()
