"""CLI du pipeline texte canonique GPT-first.

Cette commande ne charge plus d'artefact intermediaire public. Elle localise
les PDFs T1/T2, execute l'analyse semantique Vision + comparaison + triage,
et ecrit directement ``text_comparison.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vigie.analyse_texte.models import TextAnalysisQualityError
from vigie.analyse_texte.pipeline import run_text_analysis_pipeline
from vigie.analyse_texte.text_comparison import generate_text_comparison_excel
from vigie.support.batch_quarter import find_pdf_pair, normalize_quarter, resolve_previous_quarter

logger = logging.getLogger(__name__)

DEFAULT_OUT_ROOT_EXTRACTIONS = "outputs/text_extractions"
DEFAULT_OUT_ROOT_COMPARISONS = "outputs/resultats"
DEFAULT_INPUTS_ROOT = "Inputs"


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour la commande run_text_compare."""
    parser = argparse.ArgumentParser(description="Comparer les paragraphes texte entre deux trimestres (GPT-4o).")
    parser.add_argument("--banque", required=True, help="Code banque (ex: bns, bnc, rbc)")
    parser.add_argument("--annee", required=True, type=int, help="Annee du rapport courant")
    trimestre = parser.add_mutually_exclusive_group(required=True)
    trimestre.add_argument("--T1", dest="trimestre", action="store_const", const="T1", help="Trimestre courant T1")
    trimestre.add_argument("--T2", dest="trimestre", action="store_const", const="T2", help="Trimestre courant T2")
    trimestre.add_argument("--T3", dest="trimestre", action="store_const", const="T3", help="Trimestre courant T3")
    trimestre.add_argument("--T4", dest="trimestre", action="store_const", const="T4", help="Trimestre courant T4")
    parser.add_argument(
        "--racine-extraction",
        default=DEFAULT_OUT_ROOT_EXTRACTIONS,
        help=("Parametre conserve pour compatibilite CLI. Le pipeline canonique n'ecrit plus d'extraction publique."),
    )
    parser.add_argument(
        "--entrees",
        default=DEFAULT_INPUTS_ROOT,
        help=f"Racine des PDFs trimestriels (defaut: {DEFAULT_INPUTS_ROOT})",
    )
    parser.add_argument(
        "--sortie",
        default=DEFAULT_OUT_ROOT_COMPARISONS,
        help=f"Racine des sorties de comparaison texte (defaut: {DEFAULT_OUT_ROOT_COMPARISONS})",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Modele OpenAI (defaut: gpt-5.4)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Activer les logs detailles")
    parser.add_argument(
        "--strict-sections",
        action="store_true",
        help="Ne comparer que gestion_capital et gestion_risques.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executer l'analyse texte canonique entre deux trimestres."""
    args = build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    bank_code = args.banque.lower()
    year_t2 = args.annee
    quarter_t2 = normalize_quarter(args.trimestre)
    out_root = Path(args.sortie)
    model = args.model
    project_root = Path.cwd()
    inputs_root = project_root / args.entrees

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

    try:
        allowed_section_keys = None
        if args.strict_sections:
            allowed_section_keys = {"gestion_capital", "gestion_risques"}
            logger.info("Mode --strict-sections active : %s", sorted(allowed_section_keys))
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
        logger.error("Erreur qualite pipeline texte: %s", exc)
        return 1

    retained = payload.get("global_summary", {}).get("counts", {}).get("total_relevant", 0)
    excel_path = out_path.with_suffix(".xlsx")
    generate_text_comparison_excel(payload, excel_path)
    logger.info("Excel analyste genere → %s", excel_path)
    logger.info(
        "Analyse texte canonique terminee → %s\n  %d sections | %d changements retenus",
        out_path,
        len(payload.get("section_comparisons", [])),
        retained,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
