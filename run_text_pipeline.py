#!/usr/bin/env python
"""Pipeline batch texte canonique GPT-first.

Usage::

    python run_text_pipeline.py --bank BNS --year 2025 --T2

Ce pipeline:
1. Déduit automatiquement le trimestre précédent (T2→T1, T1→T3 N-1, …).
2. Localise les deux PDFs dans Inputs/{BANK}/{YEAR}/.
3. Extrait des unités sémantiques propres via GPT-4o Vision.
4. Compare explicitement les idées T1 vs T2.
5. Trie les changements et ne garde que les majeurs ou modérés réellement nouveaux.
6. Écrit text_comparison.json dans outputs/resultats/.

Architecture:
    outputs/resultats/{bank}/{year_q_vs_year_q}/text_comparison.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vigilance.cli.quarter_logic import (
    find_pdf_pair,
    normalize_quarter,
    resolve_previous_quarter,
)

DEFAULT_INPUTS_ROOT = "Inputs"
DEFAULT_LEGACY_DATA_ROOT = "data"
DEFAULT_COMPARISON_ROOT = "outputs/resultats"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI du pipeline texte."""
    p = argparse.ArgumentParser(
        description="Vigilance — Pipeline Texte Batch (Extraction + Comparaison sémantique).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python run_text_pipeline.py --bank BNS --year 2025 --T2\n"
            "  python run_text_pipeline.py --bank RBC --year 2025 --T2\n"
            "  python run_text_pipeline.py --bank BMO --year 2025 --T4 --extract-only --force-extraction\n"
        ),
    )
    p.add_argument("--bank", required=True, help="Code de la banque (ex: BNS, BNC, RBC)")
    p.add_argument("--year", required=True, type=int, help="Année du rapport courant (ex: 2025)")
    quarter_group = p.add_mutually_exclusive_group(required=True)
    quarter_group.add_argument("--T1", dest="quarter_flag", action="store_const", const="T1", help="Trimestre courant T1")
    quarter_group.add_argument("--T2", dest="quarter_flag", action="store_const", const="T2", help="Trimestre courant T2")
    quarter_group.add_argument("--T3", dest="quarter_flag", action="store_const", const="T3", help="Trimestre courant T3")
    quarter_group.add_argument("--T4", dest="quarter_flag", action="store_const", const="T4", help="Trimestre courant T4")
    p.add_argument("--inputs-root", default=DEFAULT_INPUTS_ROOT, help="Répertoire racine des PDFs")
    p.add_argument(
        "--out-root",
        default=DEFAULT_COMPARISON_ROOT,
        help=f"Répertoire racine des comparaisons texte (défaut: {DEFAULT_COMPARISON_ROOT})",
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="Modèle OpenAI (défaut: gpt-4o)",
    )
    p.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Faire seulement l'extraction texte, sans comparaison GPT",
    )
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="Alias explicite de --skip-comparison",
    )
    p.add_argument(
        "--force-extraction",
        action="store_true",
        help="Ignorer les text_extraction.md existants et régénérer les artefacts",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_compare_text(
    bank: str,
    year_current: int,
    quarter_current: str,
    previous_pdf: Path,
    current_pdf: Path,
    out_root: Path,
    model: str,
    force_extraction: bool,
) -> Path:
    """Lance l'analyse texte canonique. Retourne le chemin text_comparison.json."""
    from vigilance.text_analysis_pipeline import run_text_analysis_pipeline
    from vigilance.text_comparison.text_comparison_excel import generate_text_comparison_excel

    payload, out_path = run_text_analysis_pipeline(
        bank_code=bank,
        year_current=year_current,
        quarter_current=quarter_current,
        pdf_previous=previous_pdf,
        pdf_current=current_pdf,
        out_root=out_root,
        model=model,
        force_extraction=force_extraction,
        progress_callback=lambda message: print(
            f"  {message}",
            flush=True,
        ),
    )
    generate_text_comparison_excel(payload, out_path.with_suffix(".xlsx"))
    if not out_path.exists():
        raise FileNotFoundError(
            f"Analyse texte terminée mais text_comparison.json introuvable: {out_path}"
        )
    return out_path


def _step_extract_text(
    bank: str,
    year_current: int,
    quarter_current: str,
    previous_pdf: Path,
    current_pdf: Path,
    out_root: Path,
    force_extraction: bool,
) -> dict[str, object]:
    """Lance l'extraction texte seule et retourne le manifeste d'artefacts."""
    from vigilance.text_analysis_pipeline import run_text_extraction_pipeline

    return run_text_extraction_pipeline(
        bank_code=bank,
        year_current=year_current,
        quarter_current=quarter_current,
        pdf_previous=previous_pdf,
        pdf_current=current_pdf,
        out_root=out_root,
        force_extraction=force_extraction,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Point d'entrée du pipeline texte batch."""
    args = build_parser().parse_args(argv)

    bank = args.bank.upper()
    bank_lower = bank.lower()
    year_current = args.year
    quarter_value = str(args.quarter_flag or "").strip()
    q_current = normalize_quarter(quarter_value)
    year_previous, q_previous = resolve_previous_quarter(year_current, q_current)

    project_root = Path(__file__).resolve().parent
    inputs_root = project_root / args.inputs_root
    legacy_data = project_root / DEFAULT_LEGACY_DATA_ROOT
    out_root = project_root / args.out_root

    print("=" * 70)
    print("  VIGILANCE — Pipeline Texte Batch")
    print(f"  Banque:              {bank}")
    print(f"  Trimestre courant:   {q_current.upper()}-{year_current}")
    print(f"  Trimestre précédent: {q_previous.upper()}-{year_previous}  (déduit automatiquement)")
    print(f"  Modèle:              {args.model}")
    print("=" * 70)

    # ── Locate PDFs ──────────────────────────────────────────────────────
    print("\n📂 Recherche des PDFs...")
    try:
        previous_pdf, current_pdf = find_pdf_pair(
            bank=bank,
            year_current=year_current,
            quarter_current=q_current,
            inputs_root=inputs_root if inputs_root.is_dir() else None,
            legacy_data_root=legacy_data if legacy_data.is_dir() else None,
        )
    except FileNotFoundError as exc:
        print(f"\n  ERREUR : {exc}")
        print(
            "\n  Placer les PDFs dans Inputs/{BANK}/{YEAR}/{BANK}_{YEAR}_{Q}.pdf\n"
            "  Ex: Inputs/BNS/2025/BNS_2025_T2.pdf"
        )
        return 1

    print(f"  Période précédente : {previous_pdf}")
    print(f"  Période courante   : {current_pdf}")

    extraction_only = bool(args.skip_comparison or args.extract_only)
    if extraction_only:
        print(
            f"\n🧾 Extraction texte seule "
            f"({q_current.upper()}-{year_current} vs {q_previous.upper()}-{year_previous})..."
        )
        t0 = time.time()
        try:
            extraction_payload = _step_extract_text(
                bank=bank_lower,
                year_current=year_current,
                quarter_current=q_current,
                previous_pdf=previous_pdf,
                current_pdf=current_pdf,
                out_root=out_root,
                force_extraction=args.force_extraction,
            )
            elapsed = time.time() - t0
            print(f"  ✓ Extraction texte terminée ({elapsed:.1f}s)")
            print(f"  Artefact précédent : {extraction_payload['extraction_artifact_t1']}")
            print(f"  Artefact courant   : {extraction_payload['extraction_artifact_t2']}")
        except Exception as exc:
            print(f"\n  ERREUR extraction : {exc}")
            return 1
    else:
        print(f"\n🔍 Analyse texte canonique ({q_current.upper()}-{year_current} vs {q_previous.upper()}-{year_previous})...")
        t0 = time.time()
        try:
            comparison_path = _step_compare_text(
                bank=bank_lower,
                year_current=year_current,
                quarter_current=q_current,
                previous_pdf=previous_pdf,
                current_pdf=current_pdf,
                out_root=out_root,
                model=args.model,
                force_extraction=args.force_extraction,
            )
            elapsed = time.time() - t0
            print(f"  ✓ Comparaison texte terminée ({elapsed:.1f}s) : {comparison_path}")
        except Exception as exc:
            print(f"\n  ERREUR comparaison : {exc}")
            return 1

    print("\n" + "=" * 70)
    print("  Pipeline texte terminé avec succès.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
