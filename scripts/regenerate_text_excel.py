"""Regenere text_comparison.xlsx depuis un text_comparison.json existant.

N'invoque pas le pipeline GPT (extraction/comparaison/triage) : reutilise les
donnees deja calculees dans le JSON pour reproduire l'export Excel avec le
generateur actuel (colonnes et justifications a jour).

Usage:
  python scripts/regenerate_text_excel.py --json outputs/resultats/bnc/2025_t4_vs_2024_t4/text_comparison.json
  python scripts/regenerate_text_excel.py --json path/to/text_comparison.json --output path/to/out.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigilance.text_comparison import generate_text_comparison_excel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Chemin vers text_comparison.json")
    parser.add_argument(
        "--output",
        help="Chemin de sortie .xlsx (defaut: meme chemin que --json avec suffixe .xlsx)",
    )
    return parser


def main() -> None:
    """Point d'entree CLI."""
    args = build_parser().parse_args()
    json_path = Path(args.json)
    if not json_path.exists():
        raise SystemExit(f"Fichier introuvable: {json_path}")

    output_path = Path(args.output) if args.output else json_path.with_suffix(".xlsx")

    text_data = json.loads(json_path.read_text(encoding="utf-8"))
    result_path = generate_text_comparison_excel(text_data, output_path)

    print(f"Excel regenere: {result_path}")


if __name__ == "__main__":
    main()
