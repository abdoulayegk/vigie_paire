"""CLI pour la comparaison sémantique de paragraphes entre deux trimestres.

Charge les text_extraction.json T1 et T2, lance le diff sémantique GPT-4o
(Pass 2), puis le triage réglementaire (Pass 3), et écrit text_comparison.json.

Usage:
    python -m vigilance.cli.run_text_compare --bank BNS --year 2025 --quarter T2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUT_ROOT_EXTRACTIONS = "outputs/text_extractions"
DEFAULT_OUT_ROOT_COMPARISONS = "outputs/text_comparisons"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare text paragraphs between two quarters using GPT-4o semantic diff + triage."
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. bns, bnc, rbc)")
    parser.add_argument("--year", required=True, type=int, help="Current report year")
    parser.add_argument("--quarter", required=True, help="Current report quarter (e.g. t2)")
    parser.add_argument(
        "--extraction-root",
        default=DEFAULT_OUT_ROOT_EXTRACTIONS,
        help=f"Root directory of text extractions (default: {DEFAULT_OUT_ROOT_EXTRACTIONS})",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT_COMPARISONS,
        help=f"Root directory for text comparison output (default: {DEFAULT_OUT_ROOT_COMPARISONS})",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Comparer les paragraphes texte entre deux trimestres."""
    args = build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from vigilance.cli.quarter_logic import normalize_quarter, resolve_previous_quarter
    from vigilance.text_extraction.text_extraction_writer import (
        get_text_extraction_path,
        load_text_extraction,
    )
    from vigilance.text_comparison import (
        build_text_comparison,
        get_text_comparison_path,
        run_section_diff,
        run_text_triage,
        write_text_comparison,
    )

    bank_code = args.bank.lower()
    year_t2 = args.year
    quarter_t2 = normalize_quarter(args.quarter)
    extraction_root = Path(args.extraction_root)
    out_root = Path(args.out_root)
    model = args.model

    # Resolve previous quarter
    year_t1, quarter_t1 = resolve_previous_quarter(year_t2, quarter_t2)

    logger.info(
        "Comparaison texte : %s %s_%s vs %s_%s",
        bank_code.upper(), year_t2, quarter_t2.upper(), year_t1, quarter_t1.upper(),
    )

    # Load extractions
    path_t1 = get_text_extraction_path(extraction_root, bank_code, year_t1, quarter_t1)
    path_t2 = get_text_extraction_path(extraction_root, bank_code, year_t2, quarter_t2)

    try:
        extraction_t1 = load_text_extraction(path_t1)
    except FileNotFoundError:
        logger.error(
            "text_extraction.json T1 introuvable : %s\n"
            "Lancez d'abord : python -m vigilance.cli.run_text_extract "
            "--bank %s --year %d --quarter %s --pdf <path>",
            path_t1, bank_code.upper(), year_t1, quarter_t1.upper(),
        )
        return 1

    try:
        extraction_t2 = load_text_extraction(path_t2)
    except FileNotFoundError:
        logger.error(
            "text_extraction.json T2 introuvable : %s\n"
            "Lancez d'abord : python -m vigilance.cli.run_text_extract "
            "--bank %s --year %d --quarter %s --pdf <path>",
            path_t2, bank_code.upper(), year_t2, quarter_t2.upper(),
        )
        return 1

    logger.info(
        "Extractions chargées : T1=%d blocs, T2=%d blocs",
        len(extraction_t1.get("blocks", [])),
        len(extraction_t2.get("blocks", [])),
    )

    # Pass 2 — Semantic diff
    logger.info("Pass 2 — Diff sémantique par section ...")
    section_diffs = run_section_diff(
        extraction_t1=extraction_t1,
        extraction_t2=extraction_t2,
        model=model,
    )

    if not section_diffs:
        logger.warning("Aucun diff trouvé entre les deux trimestres.")
        return 1

    total_changes = sum(len(v) for v in section_diffs.values())
    logger.info("Pass 2 terminé : %d sections, %d changements détectés", len(section_diffs), total_changes)

    # Pass 3 — Triage
    logger.info("Pass 3 — Triage réglementaire ...")
    triage_result = run_text_triage(
        section_diffs=section_diffs,
        model=model,
    )

    relevant = triage_result.get("global_summary", {}).get("counts", {}).get("total_relevant", 0)
    logger.info("Pass 3 terminé : %d changements pertinents", relevant)

    # Build and write comparison
    payload = build_text_comparison(
        bank_code=bank_code,
        quarter_t1=quarter_t1,
        quarter_t2=quarter_t2,
        year_t1=year_t1,
        year_t2=year_t2,
        extraction_t1=extraction_t1,
        extraction_t2=extraction_t2,
        triage_result=triage_result,
    )

    out_path = get_text_comparison_path(out_root, bank_code, year_t2, quarter_t2, year_t1, quarter_t1)
    write_text_comparison(payload, out_path)

    logger.info(
        "Comparaison texte terminée → %s\n"
        "  %d sections | %d changements | %d pertinents",
        out_path,
        len(payload.get("section_comparisons", [])),
        total_changes,
        relevant,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
