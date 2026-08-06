"""Audit strict intra-section comparison outputs.

Usage:
  python scripts/audit_intra_section.py --input path/to/compare.json --output artifacts/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_pairs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = payload.get("pairs", [])
    return pairs if isinstance(pairs, list) else []


def _extract_entries(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    values = payload.get(field, [])
    return values if isinstance(values, list) else []


def _is_unknown(section: Any) -> bool:
    return str(section or "").strip().lower() in UNKNOWN_SECTIONS


def build_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Construit le rapport d'audit des appariements intra-section d'un payload."""
    pairs = _extract_pairs(payload)
    unmatched_t1 = _extract_entries(payload, "unmatched_t1")
    unmatched_t2 = _extract_entries(payload, "unmatched_t2")

    total_pairs = len(pairs)
    cross_section_pairs = 0
    unknown_matched_pairs = 0
    for pair in pairs:
        section_t1 = pair.get("section_t1", pair.get("section"))
        section_t2 = pair.get("section_t2", pair.get("section"))
        if _is_unknown(section_t1) or _is_unknown(section_t2):
            unknown_matched_pairs += 1
        if (
            section_t1 is not None
            and section_t2 is not None
            and str(section_t1).strip()
            and str(section_t2).strip()
            and str(section_t1).strip() != str(section_t2).strip()
        ):
            cross_section_pairs += 1

    unknown_unmatched = sum(
        1 for item in [*unmatched_t1, *unmatched_t2] if _is_unknown(item.get("section"))
    )
    potential_false_positive_signals = cross_section_pairs + unknown_matched_pairs

    return {
        "summary": {
            "total_pairs": total_pairs,
            "cross_section_pairs": cross_section_pairs,
            "unknown_matched_pairs": unknown_matched_pairs,
            "unknown_unmatched_items": unknown_unmatched,
            "potential_false_positive_signals": potential_false_positive_signals,
        },
        "status": {
            "strict_intra_section_ok": cross_section_pairs == 0,
            "unknown_never_matched_ok": unknown_matched_pairs == 0,
        },
    }


def main() -> None:
    """Point d'entrée CLI: lit un payload de comparaison et écrit le rapport d'audit."""
    parser = argparse.ArgumentParser(description="Audit strict intra-section comparison payloads.")
    parser.add_argument("--input", required=True, help="Path to comparison JSON output")
    parser.add_argument("--output", required=True, help="Path to write audit JSON report")
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit 1 when cross-section or unknown matched violations are detected.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(input_path)
    report = build_report(payload)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fail_on_violations:
        status = report["status"]
        if not status["strict_intra_section_ok"] or not status["unknown_never_matched_ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
