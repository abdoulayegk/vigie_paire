"""Bench matching: compare run_strict_intra_section_compare output to a reference set.

Reference format (JSON): list of objects with t1_uid, t2_uid, expected_match (bool).
  [{"t1_uid": "section|id|p1", "t2_uid": "section|id|p2", "expected_match": true}, ...]

Usage (once reference and result files exist):
  python scripts/bench_matching_metrics.py --reference path/to/reference.json --result path/to/compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_reference(path: Path) -> list[dict[str, Any]]:
    """Load reference pairs. Each item: t1_uid, t2_uid, expected_match (bool)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and "t1_uid" in x and "t2_uid" in x]


def extract_pairs_from_result(payload: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract (t1_uid, t2_uid) from run_strict_intra_section_compare result."""
    pairs = payload.get("pairs") or []
    if not isinstance(pairs, list):
        return set()
    out: set[tuple[str, str]] = set()
    for p in pairs:
        if isinstance(p, dict):
            u1 = str(p.get("t1_uid", ""))
            u2 = str(p.get("t2_uid", ""))
            if u1 and u2:
                out.add((u1, u2))
    return out


def compute_metrics(
    result_pairs: set[tuple[str, str]],
    reference: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute precision and recall vs reference. Reference expected_match=True are the gold pairs."""
    gold = {(r["t1_uid"], r["t2_uid"]) for r in reference if r.get("expected_match") is True}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(result_pairs & gold)
    precision = tp / len(result_pairs) if result_pairs else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "gold": float(len(gold)), "pred": float(len(result_pairs))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench matching vs reference set")
    parser.add_argument("--reference", type=Path, help="JSON reference: list of {t1_uid, t2_uid, expected_match}")
    parser.add_argument("--result", type=Path, help="JSON result from run_strict_intra_section_compare (or comparison_runner)")
    args = parser.parse_args()
    if not args.reference or not args.reference.exists():
        print("No reference file or file not found. Create a reference JSON and pass --reference.")
        return
    if not args.result or not args.result.exists():
        print("No result file or file not found. Run a comparison and pass --result.")
        return
    ref = load_reference(args.reference)
    result_payload = json.loads(args.result.read_text(encoding="utf-8"))
    result_pairs = extract_pairs_from_result(result_payload)
    metrics = compute_metrics(result_pairs, ref)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
