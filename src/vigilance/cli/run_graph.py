"""Point d'entrée CLI pour l'exécution du pipeline Multi-Agents LangGraph (vigie-graph-run)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from vigilance.graph.runner import arun_langgraph_comparison, run_langgraph_comparison

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour la commande vigie-graph-run."""
    parser = argparse.ArgumentParser(
        description="Exécute la comparaison bancaire via le graphe Multi-Agents LangGraph."
    )
    parser.add_argument(
        "--bank",
        "-b",
        required=True,
        help="Code de la banque (ex: RBC, BMO, TD, BNC, CIBC, BNS)",
    )
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=2025,
        help="Année du trimestre courant (défaut: 2025)",
    )
    parser.add_argument(
        "--quarter",
        "-q",
        default="T4",
        help="Trimestre courant (défaut: T4)",
    )
    parser.add_argument(
        "--year-prev",
        type=int,
        default=2024,
        help="Année du trimestre précédent (défaut: 2024)",
    )
    parser.add_argument(
        "--quarter-prev",
        default="T4",
        help="Trimestre précédent (défaut: T4)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Dossier de sortie optionnel pour les fichiers comparison.json et comparison.xlsx",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Exécuter en mode asynchrone non-bloquant",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Point d'entrée principal de la commande CLI vigie-graph-run."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    parsed = parser.parse_args(args)

    print(f"🚀 Lancement de la comparaison LangGraph pour {parsed.bank.upper()} ({parsed.quarter} {parsed.year} vs {parsed.quarter_prev} {parsed.year_prev})...")

    if parsed.async_mode:
        json_path = asyncio.run(
            arun_langgraph_comparison(
                bank=parsed.bank,
                year_current=parsed.year,
                quarter_current=parsed.quarter,
                year_previous=parsed.year_prev,
                quarter_previous=parsed.quarter_prev,
                output_dir=parsed.output_dir,
            )
        )
    else:
        json_path = run_langgraph_comparison(
            bank=parsed.bank,
            year_current=parsed.year,
            quarter_current=parsed.quarter,
            year_previous=parsed.year_prev,
            quarter_previous=parsed.quarter_prev,
            output_dir=parsed.output_dir,
        )

    print(f"✅ Comparaison LangGraph terminée avec succès !")
    print(f"📂 Fichier JSON produit : {json_path}")
    print(f"📊 Fichier Excel produit : {json_path.parent / 'comparison.xlsx'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
