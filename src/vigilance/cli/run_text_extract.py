"""CLI legacy d'extraction texte.

Le pipeline texte canonique n'expose plus d'artefact d'extraction public.
Cette commande est conservée pour compatibilité d'interface mais redirige
l'utilisateur vers les entrées canoniques de comparaison.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/text_extractions"


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour la commande run_text_extract."""
    parser = argparse.ArgumentParser(
        description="Extract text blocks (paragraphs) from targeted sections of a bank report PDF."
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. bns, bnc, rbc)")
    parser.add_argument("--year", required=True, type=int, help="Report year (e.g. 2025)")
    quarter_group = parser.add_mutually_exclusive_group(required=True)
    quarter_group.add_argument("--T1", dest="quarter_flag", action="store_const", const="T1", help="Current report quarter T1")
    quarter_group.add_argument("--T2", dest="quarter_flag", action="store_const", const="T2", help="Current report quarter T2")
    quarter_group.add_argument("--T3", dest="quarter_flag", action="store_const", const="T3", help="Current report quarter T3")
    quarter_group.add_argument("--T4", dest="quarter_flag", action="store_const", const="T4", help="Current report quarter T4")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML bank profiles config")
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--strict-sections",
        action="store_true",
        help="Limiter l'extraction aux sections gestion_capital et gestion_risques "
        "(ignore gestion_reglementation même si présente au profil banque).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrée legacy désormais désactivée."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.error(
        "run_text_extract est désactivé. Utilisez "
        "`uv run python -m vigilance.cli.run_text_compare --bank %s --year %d --%s` "
        "ou `uv run run_text_pipeline.py --bank %s --year %d --%s`.",
        args.bank.upper(),
        args.year,
        args.quarter_flag,
        args.bank.upper(),
        args.year,
        args.quarter_flag,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
