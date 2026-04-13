"""CLI du pipeline texte canonique GPT-first.

Cette commande ne charge plus d'artefact intermédiaire public. Elle localise
les PDFs T1/T2, exécute l'analyse sémantique Vision + comparaison + triage,
et écrit directement ``text_comparison.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUT_ROOT_EXTRACTIONS = "outputs/text_extractions"
DEFAULT_OUT_ROOT_COMPARISONS = "outputs/resultats"
DEFAULT_INPUTS_ROOT = "Inputs"


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour la commande run_text_compare."""
    parser = argparse.ArgumentParser(
        description="Compare text paragraphs between two quarters using GPT-4o semantic diff + triage."
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. bns, bnc, rbc)")
    parser.add_argument("--year", required=True, type=int, help="Current report year")
    quarter_group = parser.add_mutually_exclusive_group(required=True)
    quarter_group.add_argument(
        "--T1", dest="quarter_flag", action="store_const", const="T1", help="Current report quarter T1"
    )
    quarter_group.add_argument(
        "--T2", dest="quarter_flag", action="store_const", const="T2", help="Current report quarter T2"
    )
    quarter_group.add_argument(
        "--T3", dest="quarter_flag", action="store_const", const="T3", help="Current report quarter T3"
    )
    quarter_group.add_argument(
        "--T4", dest="quarter_flag", action="store_const", const="T4", help="Current report quarter T4"
    )
    parser.add_argument(
        "--extraction-root",
        default=DEFAULT_OUT_ROOT_EXTRACTIONS,
        help=(
            "Paramètre conservé pour compatibilité CLI. "
            "Le pipeline canonique n'écrit plus d'extraction publique."
        ),
    )
    parser.add_argument(
        "--inputs-root",
        default=DEFAULT_INPUTS_ROOT,
        help=f"Root directory of quarterly PDFs (default: {DEFAULT_INPUTS_ROOT})",
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--strict-sections",
        action="store_true",
        help="Ne comparer que gestion_capital et gestion_risques (ignorer gestion_reglementation dans les JSON).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exécuter l'analyse texte canonique entre deux trimestres."""
    args = build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from vigilance.cli.quarter_logic import (
        find_pdf_pair,
        normalize_quarter,
        resolve_previous_quarter,
    )
    from vigilance.text_analysis_pipeline import TextAnalysisQualityError, run_text_analysis_pipeline

    bank_code = args.bank.lower()
    year_t2 = args.year
    quarter_t2 = normalize_quarter(args.quarter_flag)
    out_root = Path(args.out_root)
    model = args.model
    project_root = Path.cwd()
    inputs_root = project_root / args.inputs_root

    year_t1, quarter_t1 = resolve_previous_quarter(year_t2, quarter_t2)

    logger.info(
        "Comparaison texte : %s %s_%s vs %s_%s",
        bank_code.upper(),
        year_t2,
        quarter_t2.upper(),
        year_t1,
        quarter_t1.upper(),
    )

    try:
        pdf_t1, pdf_t2 = find_pdf_pair(
            bank=bank_code.upper(),
            year_current=year_t2,
            quarter_current=quarter_t2,
            inputs_root=inputs_root if inputs_root.is_dir() else None,
            legacy_data_root=(project_root / "data") if (project_root / "data").is_dir() else None,
        )
    except FileNotFoundError as exc:
        logger.error(
            "PDFs introuvables pour %s %s_%s vs %s_%s: %s",
            bank_code.upper(),
            year_t2,
            quarter_t2.upper(),
            year_t1,
            quarter_t1.upper(),
            exc,
        )
        return 1

    from vigilance.text_comparison import generate_text_comparison_excel
    try:
        allowed_section_keys = None
        if args.strict_sections:
            allowed_section_keys = {"gestion_capital", "gestion_risques"}
            logger.info("Mode --strict-sections activé : %s", sorted(allowed_section_keys))
        payload, out_path = run_text_analysis_pipeline(
            bank_code=bank_code,
            year_current=year_t2,
            quarter_current=quarter_t2,
            pdf_previous=pdf_t1,
            pdf_current=pdf_t2,
            out_root=out_root,
            model=model,
            allowed_section_keys=allowed_section_keys,
        )
    except TextAnalysisQualityError as exc:
        logger.error("Erreur qualité pipeline texte: %s", exc)
        return 1

    retained = payload.get("global_summary", {}).get("counts", {}).get("total_relevant", 0)
    excel_path = out_path.with_suffix(".xlsx")
    generate_text_comparison_excel(payload, excel_path)
    logger.info("Excel analyste généré → %s", excel_path)
    logger.info(
        "Analyse texte canonique terminée → %s\n  %d sections | %d changements retenus",
        out_path,
        len(payload.get("section_comparisons", [])),
        retained,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
