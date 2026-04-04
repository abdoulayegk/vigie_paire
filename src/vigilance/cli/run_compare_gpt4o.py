"""Point d'entree CLI pour la comparaison GPT-4o sur les artefacts d'extraction canoniques."""

from __future__ import annotations

import argparse
from pathlib import Path

from vigilance.compare_gpt import (
    REFERENCE_RESOLUTION_RULE,
    compare_reports_gpt4o,
    normalize_quarter,
    resolve_reference_period,
)

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_EXTRACTION_ROOT = "outputs/extractions"
DEFAULT_OUT_ROOT = "outputs/comparisons"


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur d'arguments pour la comparaison GPT-4o."""
    parser = argparse.ArgumentParser(
        description="Compare two extracted reports with GPT-4o using canonical tables.json artifacts."
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. bnc)")
    parser.add_argument(
        "--year-current", required=True, type=int, help="Current report year"
    )
    parser.add_argument(
        "--quarter-current", required=True, help="Current report quarter (e.g. t2)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument(
        "--extraction-root",
        default=DEFAULT_EXTRACTION_ROOT,
        help="Root directory containing extracted report artifacts",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="Root directory for comparison outputs",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional OpenAI model override (defaults to config role default_genai)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executer la comparaison GPT-4o entre deux repertoires d'extraction."""
    parser = build_parser()
    args = parser.parse_args(argv)

    current_quarter = normalize_quarter(args.quarter_current)
    year_previous, previous_quarter = resolve_reference_period(
        args.year_current, current_quarter
    )
    extraction_root = Path(args.extraction_root)
    previous_dir = extraction_root / args.bank / str(year_previous) / previous_quarter
    current_dir = extraction_root / args.bank / str(args.year_current) / current_quarter

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=Path(args.out_root),
        model=str(args.model or "").strip() or None,
        config_path=args.config,
        reference_resolution={
            "mode": "automatique",
            "year_previous": year_previous,
            "quarter_previous": previous_quarter,
            "rule": REFERENCE_RESOLUTION_RULE,
        },
    )
    print(str(comparison_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
