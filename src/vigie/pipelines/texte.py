#!/usr/bin/env python
"""Pipeline batch texte canonique GPT-first.

Usage::

    python run_text_pipeline.py --banque BNS --annee 2025 --T2

Ce pipeline:
1. Deduit automatiquement le trimestre precedent (T2→T1, T1→T3 N-1, …).
2. Localise les deux PDFs dans Inputs/{BANK}/{YEAR}/.
3. Extrait des unites semantiques propres via GPT-4o Vision.
4. Compare explicitement les idees T1 vs T2.
5. Trie les changements et ne garde que les majeurs ou moderes reellement nouveaux.
6. Ecrit text_comparison.json dans outputs/resultats/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Racine du depot (src/vigie/pipelines/ -> 3 niveaux au-dessus).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vigie.support.batch_quarter import (
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
        description="Vigie — Pipeline texte (extraction + comparaison semantique).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python run_text_pipeline.py --banque BNS --annee 2025 --T2\n"
            "  python run_text_pipeline.py --banque RBC --annee 2025 --T2\n"
            "  python run_text_pipeline.py --banque BMO --annee 2025 --T4 "
            "--extraction-seule --forcer-extraction\n"
        ),
    )
    p.add_argument("--banque", required=True, help="Code de la banque (ex: BNS, BNC, RBC)")
    p.add_argument("--annee", required=True, type=int, help="Annee du rapport courant (ex: 2025)")
    trimestre = p.add_mutually_exclusive_group(required=True)
    trimestre.add_argument("--T1", dest="trimestre", action="store_const", const="T1", help="Trimestre courant T1")
    trimestre.add_argument("--T2", dest="trimestre", action="store_const", const="T2", help="Trimestre courant T2")
    trimestre.add_argument("--T3", dest="trimestre", action="store_const", const="T3", help="Trimestre courant T3")
    trimestre.add_argument("--T4", dest="trimestre", action="store_const", const="T4", help="Trimestre courant T4")
    p.add_argument("--entrees", default=DEFAULT_INPUTS_ROOT, help="Repertoire racine des PDFs")
    p.add_argument(
        "--sortie",
        default=DEFAULT_COMPARISON_ROOT,
        help=f"Repertoire racine des comparaisons texte (defaut: {DEFAULT_COMPARISON_ROOT})",
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="Modele OpenAI (defaut: gpt-4o)",
    )
    p.add_argument(
        "--sans-comparaison",
        action="store_true",
        help="Faire seulement l'extraction texte, sans comparaison GPT",
    )
    p.add_argument(
        "--extraction-seule",
        action="store_true",
        help="Alias explicite de --sans-comparaison",
    )
    p.add_argument(
        "--forcer-extraction",
        action="store_true",
        help="Ignorer les text_extraction.md existants et regenerer les artefacts",
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
    from vigie.analyse_texte.pipeline import run_text_analysis_pipeline
    from vigie.analyse_texte.text_comparison.text_comparison_excel import generate_text_comparison_excel

    payload, out_path = run_text_analysis_pipeline(
        bank_code=bank,
        year_current=year_current,
        quarter_current=quarter_current,
        pdf_previous=previous_pdf,
        pdf_current=current_pdf,
        out_root=out_root,
        model=model,
        force_extraction=force_extraction,
    )
    generate_text_comparison_excel(payload, out_path.with_suffix(".xlsx"))
    if not out_path.exists():
        raise FileNotFoundError(f"Analyse texte terminée mais text_comparison.json introuvable: {out_path}")
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
    from vigie.analyse_texte.pipeline import run_text_extraction_pipeline

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

    bank = args.banque.upper()
    bank_lower = bank.lower()
    year_current = args.annee
    quarter_value = str(args.trimestre or "").strip()
    q_current = normalize_quarter(quarter_value)
    year_previous, q_previous = resolve_previous_quarter(year_current, q_current)

    project_root = _PROJECT_ROOT
    inputs_root = project_root / args.entrees
    legacy_data = project_root / DEFAULT_LEGACY_DATA_ROOT
    out_root = project_root / args.sortie

    print("=" * 70)
    print("  VIGILANCE — Pipeline Texte Batch")
    print(f"  Banque:              {bank}")
    print(f"  Trimestre courant:   {q_current.upper()}-{year_current}")
    print(f"  Trimestre précédent: {q_previous.upper()}-{year_previous}  (déduit automatiquement)")
    print(f"  Modèle:              {args.model}")
    print("=" * 70)

    # ── Locate PDFs ──────────────────────────────────────────────────────
    print("\nRecherche des PDFs...")
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
            "\n  Placer les PDFs dans Inputs/{BANK}/{YEAR}/{BANK}_{YEAR}_{Q}.pdf\n  Ex: Inputs/BNS/2025/BNS_2025_T2.pdf"
        )
        return 1

    print(f"  Période précédente : {previous_pdf}")
    print(f"  Période courante   : {current_pdf}")

    extraction_only = bool(args.sans_comparaison or args.extraction_seule)
    if extraction_only:
        print(
            f"\nExtraction texte seule ({q_current.upper()}-{year_current} vs {q_previous.upper()}-{year_previous})..."
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
                force_extraction=args.forcer_extraction,
            )
            elapsed = time.time() - t0
            print(f"  Extraction texte terminée ({elapsed:.1f}s)")
            print(f"  Artefact précédent : {extraction_payload['extraction_artifact_t1']}")
            print(f"  Artefact courant   : {extraction_payload['extraction_artifact_t2']}")
        except Exception as exc:
            print(f"\n  ERREUR extraction : {exc}")
            return 1
    else:
        print(
            f"\nAnalyse texte canonique ({q_current.upper()}-{year_current} vs {q_previous.upper()}-{year_previous})..."
        )
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
                force_extraction=args.forcer_extraction,
            )
            elapsed = time.time() - t0
            print(f"  Comparaison texte terminée ({elapsed:.1f}s) : {comparison_path}")
        except Exception as exc:
            print(f"\n  ERREUR comparaison : {exc}")
            return 1

    print("\n" + "=" * 70)
    print("  Pipeline texte terminé avec succès.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
