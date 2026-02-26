"""Evaluation harness for table matching and indicator rename stability.

Input: comparison JSON (from run_comparison_with_sections) and optional gold CSV/JSON.
Output: table_match precision@1, #tables unmatched, indicator counts, rename precision.

Gold CSV format (optional):
  - Table matches: columns t1_uid, t2_uid, expected_match (true/false)
  - Rename pairs: columns removed_text, added_text, expected_rename (true/false)

If no gold: exports top N changes for manual review (manual review sampler).

Usage:
  python scripts/eval_matching_harness.py --result path/to/comparison.json [--gold path/to/gold.csv]
  python scripts/eval_matching_harness.py --result path/to/comparison.json --export-review 50 --out review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_result(path: Path) -> dict[str, Any]:
    """Load comparison runner output JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_gold_table_matches(path: Path) -> list[dict[str, Any]]:
    """Load gold table matches from CSV: t1_uid, t2_uid, expected_match."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u1 = (row.get("t1_uid") or "").strip()
            u2 = (row.get("t2_uid") or "").strip()
            if u1 and u2:
                exp = row.get("expected_match", "true").strip().lower()
                rows.append({"t1_uid": u1, "t2_uid": u2, "expected_match": exp in ("true", "1", "yes")})
    return rows


def load_gold_renames(path: Path) -> list[dict[str, Any]]:
    """Load gold rename pairs from CSV: removed_text, added_text, expected_rename."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = (row.get("removed_text") or "").strip()
            a = (row.get("added_text") or "").strip()
            if r or a:
                exp = row.get("expected_rename", "true").strip().lower()
                rows.append({"removed_text": r, "added_text": a, "expected_rename": exp in ("true", "1", "yes")})
    return rows


def extract_table_pairs(payload: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract (t1_uid, t2_uid) from comparison payload table_comparisons."""
    out: set[tuple[str, str]] = set()
    for c in payload.get("table_comparisons") or []:
        if not isinstance(c, dict):
            continue
        u1 = str(c.get("table_id_t1", "") or "").strip()
        u2 = str(c.get("table_id_t2", "") or "").strip()
        section = str(c.get("section", "") or "")
        if u1 and u2 and section:
            out.add((f"{section}|{u1}|p{c.get('page_t1', 0)}", f"{section}|{u2}|p{c.get('page_t2', 0)}"))
    return out


def extract_renames(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (from, to) rename pairs from table_comparisons."""
    out: list[tuple[str, str]] = []
    for c in payload.get("table_comparisons") or []:
        if not isinstance(c, dict):
            continue
        for r in c.get("renamed_indicators") or []:
            if isinstance(r, dict):
                out.append((str(r.get("from", "") or ""), str(r.get("to", "") or "")))
            else:
                out.append(("", ""))
    return out


def table_metrics(
    result_pairs: set[tuple[str, str]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    """Precision@1 and counts vs gold table matches."""
    gold_set = {(r["t1_uid"], r["t2_uid"]) for r in gold if r.get("expected_match") is True}
    if not gold_set:
        return {"precision_at_1": None, "tp": 0, "gold_count": 0, "pred_count": len(result_pairs)}
    tp = len(result_pairs & gold_set)
    prec = tp / len(result_pairs) if result_pairs else 0.0
    return {"precision_at_1": prec, "tp": tp, "gold_count": len(gold_set), "pred_count": len(result_pairs)}


def rename_metrics(
    result_renames: list[tuple[str, str]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rename precision on sampled set (gold lists expected pairs)."""
    gold_set = {(r["removed_text"].strip(), r["added_text"].strip()) for r in gold if r.get("expected_rename") is True}
    if not gold_set:
        return {"rename_precision": None, "tp": 0, "gold_renames": 0, "pred_renames": len(result_renames)}
    result_set = set(result_renames)
    tp = len(result_set & gold_set)
    prec = tp / len(result_set) if result_set else 0.0
    return {"rename_precision": prec, "tp": tp, "gold_renames": len(gold_set), "pred_renames": len(result_renames)}


def summary_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Summary: tables matched, added, removed, indicator added/removed/renamed."""
    comps = payload.get("table_comparisons") or []
    tables_added = len(payload.get("tables_added") or [])
    tables_removed = len(payload.get("tables_removed") or [])
    total_added = sum(len(c.get("added_indicators") or []) for c in comps if isinstance(c, dict))
    total_removed = sum(len(c.get("removed_indicators") or []) for c in comps if isinstance(c, dict))
    total_renamed = sum(len(c.get("renamed_indicators") or []) for c in comps if isinstance(c, dict))
    return {
        "tables_matched": len(comps),
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "total_added_indicators": total_added,
        "total_removed_indicators": total_removed,
        "total_renamed_indicators": total_renamed,
    }


def export_manual_review(payload: dict[str, Any], top_n: int, out_path: Path) -> None:
    """Export top N changes (by table status and impact) for manual verdict."""
    rows: list[dict[str, Any]] = []
    for c in payload.get("table_comparisons") or []:
        if not isinstance(c, dict):
            continue
        status = c.get("table_status") or ""
        n_added = len(c.get("added_indicators") or [])
        n_removed = len(c.get("removed_indicators") or [])
        n_renamed = len(c.get("renamed_indicators") or [])
        impact = n_added + n_removed + (2 * n_renamed)
        rows.append({
            "section": c.get("section"),
            "table_id_t1": c.get("table_id_t1"),
            "table_id_t2": c.get("table_id_t2"),
            "table_status": status,
            "match_score": c.get("match_score"),
            "added_count": n_added,
            "removed_count": n_removed,
            "renamed_count": n_renamed,
            "impact_score": impact,
            "manual_verdict": "",
        })
    rows.sort(key=lambda x: (-(x["impact_score"] or 0), x["section"] or "", x["table_id_t1"] or ""))
    to_export = rows[:top_n]
    if not to_export:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(to_export[0].keys()))
        writer.writeheader()
        writer.writerows(to_export)
    print(f"Exported {len(to_export)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval harness for table/indicator matching")
    parser.add_argument("--result", type=Path, help="Comparison JSON (run_comparison_with_sections output)")
    parser.add_argument("--gold-tables", type=Path, help="Optional CSV: t1_uid,t2_uid,expected_match")
    parser.add_argument("--gold-renames", type=Path, help="Optional CSV: removed_text,added_text,expected_rename")
    parser.add_argument("--export-review", type=int, metavar="N", help="Export top N changes for manual review")
    parser.add_argument("--out", type=Path, help="Output path for export-review CSV")
    args = parser.parse_args()

    if not args.result or not args.result.exists():
        print("--result path required and must exist.")
        return

    payload = load_result(args.result)
    counts = summary_counts(payload)
    print("Summary counts:", json.dumps(counts, indent=2))

    if args.gold_tables and args.gold_tables.exists():
        gold_t = load_gold_table_matches(args.gold_tables)
        pairs = extract_table_pairs(payload)
        metrics_t = table_metrics(pairs, gold_t)
        print("Table match metrics:", json.dumps(metrics_t, indent=2))

    if args.gold_renames and args.gold_renames.exists():
        gold_r = load_gold_renames(args.gold_renames)
        renames = extract_renames(payload)
        metrics_r = rename_metrics(renames, gold_r)
        print("Rename metrics:", json.dumps(metrics_r, indent=2))

    if args.export_review is not None and args.export_review > 0:
        out = args.out or Path("review_export.csv")
        export_manual_review(payload, args.export_review, out)


if __name__ == "__main__":
    main()
