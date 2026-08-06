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
    """Load comparison runner output JSON from disk.

    Args:
        path: Path to the comparison JSON file produced by run_comparison_with_sections.

    Returns:
        Parsed JSON payload as a dictionary (table_comparisons, tables_added, etc.).

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_gold_table_matches(path: Path) -> list[dict[str, Any]]:
    """Load gold table matches from a CSV file.

    Expected columns: t1_uid, t2_uid, expected_match. Rows with empty t1_uid or
    t2_uid are skipped. expected_match is parsed as true for "true", "1", "yes"
    (case-insensitive).

    Args:
        path: Path to the CSV file.

    Returns:
        List of dicts, each with keys t1_uid, t2_uid, expected_match (bool).
    """
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
    """Load gold rename pairs from a CSV file.

    Expected columns: removed_text, added_text, expected_rename. Rows with both
    removed_text and added_text empty are skipped. expected_rename is parsed as
    true for "true", "1", "yes" (case-insensitive).

    Args:
        path: Path to the CSV file.

    Returns:
        List of dicts, each with keys removed_text, added_text, expected_rename (bool).
    """
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
    """Extract matched table pairs from comparison payload.

    Builds canonical keys as "section|table_id|page" for each side of the pair.
    Only includes pairs where section, table_id_t1, and table_id_t2 are non-empty.

    Args:
        payload: Comparison JSON from run_comparison_with_sections (must have
            table_comparisons list of dicts with table_id_t1, table_id_t2,
            section, page_t1, page_t2).

    Returns:
        Set of (key_t1, key_t2) tuples, where each key is "section|uid|pN".
    """
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
    """Extract indicator rename pairs from comparison payload.

    Iterates over table_comparisons and their renamed_indicators; each rename
    yields a (from, to) tuple. Malformed entries yield ("", "").

    Args:
        payload: Comparison JSON with table_comparisons[].renamed_indicators[],
            each element being a dict with "from" and "to" keys.

    Returns:
        List of (from_text, to_text) tuples (may contain duplicates).
    """
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
    """Compute precision@1 and counts against gold table matches.

    Gold pairs with expected_match=True define the positive set. Precision@1
    is the fraction of predicted pairs that are in the gold set.

    Args:
        result_pairs: Set of predicted (key_t1, key_t2) pairs from extract_table_pairs.
        gold: List of gold dicts from load_gold_table_matches (t1_uid, t2_uid,
            expected_match).

    Returns:
        Dict with keys: precision_at_1 (float or None if no gold), tp (int),
        gold_count (int), pred_count (int).
    """
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
    """Compute rename precision against gold rename pairs.

    Gold pairs with expected_rename=True define the positive set. Precision is
    the fraction of unique predicted renames that are in the gold set.

    Args:
        result_renames: List of (from_text, to_text) pairs from extract_renames.
        gold: List of gold dicts from load_gold_renames (removed_text, added_text,
            expected_rename).

    Returns:
        Dict with keys: rename_precision (float or None if no gold), tp (int),
        gold_renames (int), pred_renames (int).
    """
    gold_set = {(r["removed_text"].strip(), r["added_text"].strip()) for r in gold if r.get("expected_rename") is True}
    if not gold_set:
        return {"rename_precision": None, "tp": 0, "gold_renames": 0, "pred_renames": len(result_renames)}
    result_set = set(result_renames)
    tp = len(result_set & gold_set)
    prec = tp / len(result_set) if result_set else 0.0
    return {"rename_precision": prec, "tp": tp, "gold_renames": len(gold_set), "pred_renames": len(result_renames)}


def summary_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Compute summary counts from comparison payload.

    Counts tables matched, tables added/removed, and total indicators
    added/removed/renamed across all table comparisons.

    Args:
        payload: Comparison JSON with table_comparisons, tables_added,
            tables_removed, and per-comparison added_indicators, removed_indicators,
            renamed_indicators.

    Returns:
        Dict with keys: tables_matched, tables_added, tables_removed,
        total_added_indicators, total_removed_indicators, total_renamed_indicators.
    """
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


def extended_summary_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Compute extended summary counts including confirmed/pending/review.

    Extends summary_counts with tables_added_confirmed, tables_removed_confirmed,
    tables_added_pending_review, tables_removed_pending_review, review_candidates.

    Args:
        payload: Comparison JSON (must include the keys used by summary_counts
            plus tables_added_confirmed, tables_removed_confirmed,
            tables_added_pending_review, tables_removed_pending_review,
            review_candidates).

    Returns:
        Dict with all keys from summary_counts plus the extended keys above.
    """
    counts = summary_counts(payload)
    counts.update(
        {
            "tables_added_confirmed": len(payload.get("tables_added_confirmed") or []),
            "tables_removed_confirmed": len(payload.get("tables_removed_confirmed") or []),
            "tables_added_pending_review": len(payload.get("tables_added_pending_review") or []),
            "tables_removed_pending_review": len(payload.get("tables_removed_pending_review") or []),
            "review_candidates": len(payload.get("review_candidates") or []),
        }
    )
    return counts


def _diff_numeric_dict(
    old_values: dict[str, Any],
    new_values: dict[str, Any],
) -> dict[str, float]:
    """Compute per-key numeric deltas (new - old) for comparable dicts.

    Only keys present in either dict are included. Values must be int or float;
    missing or non-numeric values are treated as 0.

    Args:
        old_values: Baseline dict of numeric values.
        new_values: New dict of numeric values (same or overlapping keys).

    Returns:
        Dict mapping each key to float(new_value - old_value).
    """
    keys = set(old_values) | set(new_values)
    diff: dict[str, float] = {}
    for key in sorted(keys):
        old_val = old_values.get(key, 0) or 0
        new_val = new_values.get(key, 0) or 0
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            diff[key] = float(new_val) - float(old_val)
    return diff


def compare_payloads(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compare two comparison payloads and report deltas.

    Uses extended_summary_counts and _diff_numeric_dict to produce old, new,
    and delta views. Used for legacy vs recall-first pipeline comparison.

    Args:
        old_payload: Baseline comparison JSON.
        new_payload: New comparison JSON to compare against.

    Returns:
        Dict with keys: old (extended counts), new (extended counts),
        delta (per-key numeric differences).
    """
    old_counts = extended_summary_counts(old_payload)
    new_counts = extended_summary_counts(new_payload)
    return {
        "old": old_counts,
        "new": new_counts,
        "delta": _diff_numeric_dict(old_counts, new_counts),
    }


def export_manual_review(payload: dict[str, Any], top_n: int, out_path: Path) -> None:
    """Export top N table comparisons for manual review.

    Sorts by impact_score (added + removed + 2*renamed), then section and
    table_id. Writes CSV with section, table ids, status, match_score,
    indicator counts, impact_score, and an empty manual_verdict column.

    Args:
        payload: Comparison JSON with table_comparisons.
        top_n: Maximum number of rows to export.
        out_path: Path for the output CSV file.

    Returns:
        None. Writes to out_path and prints export confirmation. Does nothing
        if there are no comparisons to export.
    """
    rows: list[dict[str, Any]] = []
    for c in payload.get("table_comparisons") or []:
        if not isinstance(c, dict):
            continue
        status = c.get("table_status") or ""
        n_added = len(c.get("added_indicators") or [])
        n_removed = len(c.get("removed_indicators") or [])
        n_renamed = len(c.get("renamed_indicators") or [])
        impact = n_added + n_removed + (2 * n_renamed)
        rows.append(
            {
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
            }
        )
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
    """CLI entry point for the evaluation harness.

    Supports two modes:
    1. Single-result: --result loads one comparison JSON, prints summary counts,
       and optionally evaluates table/rename metrics against --gold-tables and
       --gold-renames CSVs. Can export top N changes via --export-review.
    2. Compare mode: --result-old and --result-new load two payloads; prints
       extended summary deltas and optionally table/rename metrics for both,
       with deltas.

    Exits early with a message if required paths are missing or invalid.
    """
    parser = argparse.ArgumentParser(description="Eval harness for table/indicator matching")
    parser.add_argument("--result", type=Path, help="Comparison JSON (run_comparison_with_sections output)")
    parser.add_argument("--result-old", type=Path, help="Legacy comparison JSON for old/new diff mode")
    parser.add_argument("--result-new", type=Path, help="Recall-first comparison JSON for old/new diff mode")
    parser.add_argument("--gold-tables", type=Path, help="Optional CSV: t1_uid,t2_uid,expected_match")
    parser.add_argument("--gold-renames", type=Path, help="Optional CSV: removed_text,added_text,expected_rename")
    parser.add_argument("--export-review", type=int, metavar="N", help="Export top N changes for manual review")
    parser.add_argument("--out", type=Path, help="Output path for export-review CSV")
    args = parser.parse_args()

    compare_mode = bool(args.result_old or args.result_new)
    if compare_mode:
        if not args.result_old or not args.result_old.exists():
            print("--result-old path required and must exist in compare mode.")
            return
        if not args.result_new or not args.result_new.exists():
            print("--result-new path required and must exist in compare mode.")
            return
    elif not args.result or not args.result.exists():
        print("--result path required and must exist.")
        return

    if compare_mode:
        old_payload = load_result(args.result_old)
        new_payload = load_result(args.result_new)
        print("Old/New comparison:", json.dumps(compare_payloads(old_payload, new_payload), indent=2))

        if args.gold_tables and args.gold_tables.exists():
            gold_t = load_gold_table_matches(args.gold_tables)
            old_metrics_t = table_metrics(extract_table_pairs(old_payload), gold_t)
            new_metrics_t = table_metrics(extract_table_pairs(new_payload), gold_t)
            print(
                "Table match metrics old/new:",
                json.dumps(
                    {
                        "old": old_metrics_t,
                        "new": new_metrics_t,
                        "delta": _diff_numeric_dict(old_metrics_t, new_metrics_t),
                    },
                    indent=2,
                ),
            )

        if args.gold_renames and args.gold_renames.exists():
            gold_r = load_gold_renames(args.gold_renames)
            old_metrics_r = rename_metrics(extract_renames(old_payload), gold_r)
            new_metrics_r = rename_metrics(extract_renames(new_payload), gold_r)
            print(
                "Rename metrics old/new:",
                json.dumps(
                    {
                        "old": old_metrics_r,
                        "new": new_metrics_r,
                        "delta": _diff_numeric_dict(old_metrics_r, new_metrics_r),
                    },
                    indent=2,
                ),
            )
        return

    payload = load_result(args.result)
    counts = extended_summary_counts(payload)
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
