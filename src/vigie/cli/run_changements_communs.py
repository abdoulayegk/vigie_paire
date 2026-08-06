"""CLI pour generer l'analyse des changements communs entre banques."""

from __future__ import annotations

import argparse
from pathlib import Path

from vigie.comparaison.changements_communs import (
    changements_communs_output_path,
    generate_changements_communs_report,
    write_changements_communs_report,
)
from vigie.interface.ui_config import RESULTATS_DIR


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Genere l'artefact JSON des changements communs entre banques apres les comparaisons banque par banque."
        )
    )
    parser.add_argument(
        "--theme",
        default="changements communs de divulgation entre banques",
        help="Theme ou question metier analysee par le LLM.",
    )
    parser.add_argument(
        "--periode",
        required=True,
        help=(
            "Periode de comparaison a analyser, par exemple "
            "2025_t2_vs_2025_t1. L'analyse ne melange pas les trimestres."
        ),
    )
    parser.add_argument(
        "--seuil-banques",
        type=int,
        default=3,
        help="Nombre minimal de banques distinctes pour declarer un consensus.",
    )
    parser.add_argument(
        "--resultats-dir",
        type=Path,
        default=RESULTATS_DIR,
        help="Racine outputs/resultats contenant les dossiers de banques.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Chemin JSON de sortie pour le validateur. Par defaut: "
            "outputs/resultats/changements_communs_banques/<periode>/"
            "changements_communs_banques.json."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=160,
        help="Nombre maximal total de changements candidats envoyes au juge LLM.",
    )
    parser.add_argument(
        "--candidats-par-theme",
        type=int,
        default=35,
        help="Nombre maximal de candidats recuperes pour chaque theme de decouverte.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modele OpenAI optionnel pour le juge LLM.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Modele OpenAI d'embeddings pour la recuperation RAG.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the common-changes generation command."""
    args = build_parser().parse_args(argv)
    report = generate_changements_communs_report(
        args.theme,
        min_banks=max(2, int(args.seuil_banques)),
        period=args.periode,
        root_dir=args.resultats_dir,
        max_candidates=max(1, int(args.max_candidates)),
        candidates_per_query=max(1, int(args.candidats_par_theme)),
        model=args.model,
        embedding_model=args.embedding_model,
    )
    output_path = args.output or changements_communs_output_path(args.periode)
    output = write_changements_communs_report(report, path=output_path)
    print(f"Analyse des changements communs sauvegardee: {output}")
    print(f"Periode analysee: {report.get('period')}")
    print(f"Changements sources: {report.get('source_change_count')}")
    counts = report.get("signal_counts") or {}
    print(f"Regroupements detectes: {counts.get('total', len(report.get('signals') or []))}")
    print(f"Consensus >= {report.get('min_banks')} banques: {counts.get('consensus', 0)}")
    print(f"Signaux mineurs 2 banques: {counts.get('minor', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
