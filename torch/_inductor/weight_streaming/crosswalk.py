"""Diagnostic crosswalk builder: match scheduler tensor names to compiled_tensor_ids.

Uses shape+dtype matching. NOT a production mechanism — a diagnostic bootstrap for
models where shapes are unique per weight. The real production crosswalk requires
FQN on both sides.

Two modes:
- strict: fail if any compiled weight has 0 or >1 scheduler matches
- allow-partial: emit only confident matches, warn on ambiguity
"""

from __future__ import annotations

import csv
import json
import logging
import sys

log = logging.getLogger(__name__)


def _load_tensor_map(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("tensors", [])


def _load_scheduler_tensors(csv_path: str) -> list[dict]:
    result = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tensor_kind") != "WEIGHT":
                continue
            shape_str = row.get("shape", "[]")
            shape = tuple(
                x.strip() for x in shape_str.strip("[]").split(",") if x.strip()
            )
            result.append(
                {
                    "name": row["tensor_name"],
                    "dtype": row.get("dtype", ""),
                    "shape": shape,
                }
            )
    return result


def build_crosswalk(
    tensor_map_path: str,
    scheduler_tensor_csv: str,
    output_path: str,
    strict: bool = True,
) -> dict[str, int]:
    """Match scheduler tensor names to compiled_tensor_ids by dtype+shape.

    In strict mode, ambiguous matches (0 or >1 candidates) cause a hard failure.
    In allow-partial mode, only confident matches are emitted.
    """
    compiled = _load_tensor_map(tensor_map_path)
    scheduler = _load_scheduler_tensors(scheduler_tensor_csv)

    # Build index: (dtype, shape_tuple) -> list of scheduler names
    sched_by_sig: dict[tuple, list[str]] = {}
    for s in scheduler:
        sig = (s["dtype"], s["shape"])
        sched_by_sig.setdefault(sig, []).append(s["name"])

    mapping: dict[str, int] = {}
    ambiguous = []
    unmatched = []

    for entry in compiled:
        tid = entry["compiled_tensor_id"]
        dtype = entry["dtype"]
        shape = tuple(str(s) for s in entry["shape"])
        sig = (dtype, shape)
        candidates = sched_by_sig.get(sig, [])
        if len(candidates) == 1:
            mapping[candidates[0]] = tid
        elif len(candidates) == 0:
            unmatched.append(entry["graph_input_name"])
        else:
            ambiguous.append(
                (entry["graph_input_name"], tid, len(candidates))
            )

    if ambiguous:
        msg = (
            f"{len(ambiguous)} compiled weights have ambiguous matches "
            f"(multiple scheduler tensors with same dtype+shape)"
        )
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    if unmatched:
        msg = f"{len(unmatched)} compiled weights have no scheduler match"
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    # Read compilation_hash from tensor_map for inclusion
    with open(tensor_map_path) as f:
        tensor_map_data = json.load(f)
    compilation_hash = tensor_map_data.get("compilation_hash", "")

    result = {
        "compilation_hash": compilation_hash,
        "mapping": mapping,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info(
        "Crosswalk: %d matched, %d ambiguous, %d unmatched -> %s",
        len(mapping),
        len(ambiguous),
        len(unmatched),
        output_path,
    )
    return mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build tensor crosswalk")
    parser.add_argument("--tensor-map", required=True)
    parser.add_argument("--scheduler-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Don't fail on ambiguous/unmatched entries",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        build_crosswalk(
            args.tensor_map,
            args.scheduler_csv,
            args.output,
            strict=not args.allow_partial,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
