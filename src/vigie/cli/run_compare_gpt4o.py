"""Point d'entree CLI pour la comparaison GPT-4o sur les artefacts d'extraction canoniques."""

from __future__ import annotations

import argparse
from pathlib import Path

from vigie.comparaison.io import normalize_quarter, resolve_reference_period
from vigie.comparaison.pipeline.construction_resultat import REFERENCE_RESOLUTION_RULE
from vigie.comparaison.pipeline.orchestration import compare_reports_gpt4o

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_EXTRACTION_ROOT = "outputs/extractions"
DEFAULT_OUT_ROOT = "outputs/resultats"


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur d'arguments pour la comparaison GPT-4o."""
    parser = argparse.ArgumentParser(
        description="Comparer deux rapports extraits avec GPT-4o (artefacts tables.json)."
    )
    parser.add_argument("--banque", required=True, help="Code banque (ex: bnc)")
    parser.add_argument(
        "--annee-courante", required=True, type=int, help="Annee du rapport courant"
    )
    parser.add_argument(
        "--trimestre-courant", required=True, help="Trimestre courant (ex: t2)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Chemin YAML de configuration")
    parser.add_argument(
        "--racine-extraction",
        default=DEFAULT_EXTRACTION_ROOT,
        help="Racine des artefacts d'extraction",
    )
    parser.add_argument(
        "--sortie",
        default=DEFAULT_OUT_ROOT,
        help="Racine des sorties de comparaison",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Modele OpenAI optionnel (defaut: role config default_genai)",
    )
    parser.add_argument(
        "--pdf-precedent",
        default="",
        help="Chemin du PDF du trimestre de reference (preuves visuelles UI)",
    )
    parser.add_argument(
        "--pdf-courant",
        default="",
        help="Chemin du PDF du trimestre courant (preuves visuelles UI)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executer la comparaison GPT-4o entre deux repertoires d'extraction."""
    parser = build_parser()
    args = parser.parse_args(argv)

    current_quarter = normalize_quarter(args.trimestre_courant)
    year_previous, previous_quarter = resolve_reference_period(
        args.annee_courante, current_quarter
    )
    extraction_root = Path(args.racine_extraction)
    previous_dir = extraction_root / args.banque / str(year_previous) / previous_quarter
    current_dir = extraction_root / args.banque / str(args.annee_courante) / current_quarter

    comparison_path = compare_reports_gpt4o(
        previous_dir=previous_dir,
        current_dir=current_dir,
        out_root=Path(args.sortie),
        model=str(args.model or "").strip() or None,
        config_path=args.config,
        reference_resolution={
            "mode": "automatique",
            "year_previous": year_previous,
            "quarter_previous": previous_quarter,
            "rule": REFERENCE_RESOLUTION_RULE,
        },
        source_pdf_previous=str(args.pdf_precedent or "").strip() or None,
        source_pdf_current=str(args.pdf_courant or "").strip() or None,
    )
    print(str(comparison_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
