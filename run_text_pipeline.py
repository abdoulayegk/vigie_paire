#!/usr/bin/env python
"""Pipeline batch texte — Extraction + Comparaison sémantique de paragraphes.

Usage::

    python run_text_pipeline.py --bank BNS --year 2025 --quarter T2

Ce pipeline:
1. Déduit automatiquement le trimestre précédent (T2→T1, T1→T3 N-1, …).
2. Localise les deux PDFs dans Inputs/{BANK}/{YEAR}/.
3. Extrait les blocs de texte des sections ciblées (Pass 1 — Docling).
4. Compare sémantiquement les paragraphes T1 vs T2 (Pass 2 — GPT-4o diff).
5. Classe chaque changement réglementairement (Pass 3 — GPT-4o triage).
6. Écrit text_comparison.json dans outputs/text_comparisons/.

Architecture:
    outputs/text_extractions/{bank}/{year}/{quarter}/text_extraction.json
    outputs/text_comparisons/{bank}/{year_q_vs_year_q}/text_comparison.json
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
DEFAULT_EXTRACTION_ROOT = "outputs/text_extractions"
DEFAULT_COMPARISON_ROOT = "outputs/text_comparisons"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vigilance — Pipeline Texte Batch (Extraction + Comparaison sémantique).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python run_text_pipeline.py --bank BNS --year 2025 --quarter T2\n"
            "  python run_text_pipeline.py --bank RBC --year 2025 --quarter T2 --skip-extraction\n"
        ),
    )
    p.add_argument("--bank", required=True, help="Code de la banque (ex: BNS, BNC, RBC)")
    p.add_argument("--year", required=True, type=int, help="Année du rapport courant (ex: 2025)")
    p.add_argument("--quarter", required=True, help="Trimestre courant (ex: T1, T2, T3)")
    p.add_argument("--inputs-root", default=DEFAULT_INPUTS_ROOT, help="Répertoire racine des PDFs")
    p.add_argument(
        "--extraction-root",
        default=DEFAULT_EXTRACTION_ROOT,
        help=f"Répertoire racine des extractions texte (défaut: {DEFAULT_EXTRACTION_ROOT})",
    )
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
        "--skip-extraction",
        action="store_true",
        help="Sauter l'extraction (utile si text_extraction.json existe déjà)",
    )
    p.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Sauter la comparaison (ne faire que l'extraction)",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_extract_text(
    pdf_path: Path,
    bank: str,
    year: int,
    quarter: str,
    extraction_root: Path,
) -> Path:
    """Extrait les blocs de texte d'un PDF. Retourne le chemin text_extraction.json."""
    from vigilance.cli.run_text_extract import main as extract_main

    extract_main([
        "--bank", bank,
        "--year", str(year),
        "--quarter", quarter,
        "--pdf", str(pdf_path),
        "--out-root", str(extraction_root),
    ])

    from vigilance.text_extraction.text_extraction_writer import get_text_extraction_path
    out_path = get_text_extraction_path(extraction_root, bank.lower(), year, quarter)
    if not out_path.exists():
        raise FileNotFoundError(
            f"Extraction texte terminée mais text_extraction.json introuvable: {out_path}"
        )
    return out_path


def _step_compare_text(
    bank: str,
    year_current: int,
    quarter_current: str,
    extraction_root: Path,
    out_root: Path,
    model: str,
) -> Path:
    """Lance la comparaison texte. Retourne le chemin text_comparison.json."""
    from vigilance.cli.run_text_compare import main as compare_main

    compare_main([
        "--bank", bank,
        "--year", str(year_current),
        "--quarter", quarter_current,
        "--extraction-root", str(extraction_root),
        "--out-root", str(out_root),
        "--model", model,
    ])

    from vigilance.cli.quarter_logic import resolve_previous_quarter
    from vigilance.text_comparison.text_comparison_writer import get_text_comparison_path
    year_t1, quarter_t1 = resolve_previous_quarter(year_current, quarter_current)
    out_path = get_text_comparison_path(
        out_root, bank.lower(), year_current, quarter_current, year_t1, quarter_t1
    )
    if not out_path.exists():
        raise FileNotFoundError(
            f"Comparaison texte terminée mais text_comparison.json introuvable: {out_path}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    bank = args.bank.upper()
    bank_lower = bank.lower()
    year_current = args.year
    q_current = normalize_quarter(args.quarter)
    year_previous, q_previous = resolve_previous_quarter(year_current, q_current)

    project_root = Path(__file__).resolve().parent
    inputs_root = project_root / args.inputs_root
    legacy_data = project_root / DEFAULT_LEGACY_DATA_ROOT
    extraction_root = project_root / args.extraction_root
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

    print(f"  T1 : {previous_pdf}")
    print(f"  T2 : {current_pdf}")

    # ── Step 1: Extraction ────────────────────────────────────────────────
    if args.skip_extraction:
        print("\n⏭️  Extraction ignorée (--skip-extraction).")
    else:
        print(f"\n📄 Extraction texte T1 ({q_previous.upper()}-{year_previous})...")
        t0 = time.time()
        try:
            path_t1 = _step_extract_text(
                pdf_path=previous_pdf,
                bank=bank_lower,
                year=year_previous,
                quarter=q_previous,
                extraction_root=extraction_root,
            )
            print(f"  ✓ T1 extrait : {path_t1} ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"\n  ERREUR extraction T1 : {exc}")
            return 1

        print(f"\n📄 Extraction texte T2 ({q_current.upper()}-{year_current})...")
        t0 = time.time()
        try:
            path_t2 = _step_extract_text(
                pdf_path=current_pdf,
                bank=bank_lower,
                year=year_current,
                quarter=q_current,
                extraction_root=extraction_root,
            )
            print(f"  ✓ T2 extrait : {path_t2} ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"\n  ERREUR extraction T2 : {exc}")
            return 1

    # ── Step 2+3: Comparison ──────────────────────────────────────────────
    if args.skip_comparison:
        print("\n⏭️  Comparaison ignorée (--skip-comparison).")
    else:
        print(f"\n🔍 Comparaison sémantique ({q_current.upper()}-{year_current} vs {q_previous.upper()}-{year_previous})...")
        t0 = time.time()
        try:
            comparison_path = _step_compare_text(
                bank=bank_lower,
                year_current=year_current,
                quarter_current=q_current,
                extraction_root=extraction_root,
                out_root=out_root,
                model=args.model,
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
