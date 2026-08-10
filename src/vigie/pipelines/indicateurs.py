#!/usr/bin/env python
"""Orchestrateur batch du pipeline indicateurs (extraction + comparaison).

Usage::

    python run_pipeline.py --banque BNC --annee 2025 --T2

Ce script :

1. Deduit le trimestre precedent (T2→T1, T1→T3 N-1, …).
2. Localise les deux PDF.
3. Extrait les tableaux (Docling + Vision).
4. Compare semantiquement (GPT-4o).
5. Ecrit comparison.json / manifest.json sous outputs/resultats/.
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

from vigie.cli.output_builder import write_run_manifest
from vigie.support.batch_quarter import (
    find_pdf_pair,
    normalize_quarter,
    resolve_previous_quarter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_INPUTS_ROOT = "Inputs"
DEFAULT_LEGACY_DATA_ROOT = "data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI du pipeline indicateurs."""
    p = argparse.ArgumentParser(
        description="Vigie — Pipeline indicateurs (extraction + comparaison).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python run_pipeline.py --banque BNC --annee 2025 --T2\n"
            "  python run_pipeline.py --banque RBC --annee 2025 --T1 --sans-extraction\n"
        ),
    )
    p.add_argument("--banque", required=True, help="Code de la banque (ex: BNC, RBC, TD)")
    p.add_argument("--annee", required=True, type=int, help="Annee du rapport courant (ex: 2025)")
    trimestre = p.add_mutually_exclusive_group(required=True)
    trimestre.add_argument("--T1", dest="trimestre", action="store_const", const="T1")
    trimestre.add_argument("--T2", dest="trimestre", action="store_const", const="T2")
    trimestre.add_argument("--T3", dest="trimestre", action="store_const", const="T3")
    trimestre.add_argument("--T4", dest="trimestre", action="store_const", const="T4")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Chemin YAML de configuration")
    p.add_argument("--entrees", default=DEFAULT_INPUTS_ROOT, help="Repertoire racine des PDFs")
    p.add_argument(
        "--sortie",
        default="outputs/resultats",
        help="Repertoire racine des sorties de comparaison (defaut: outputs/resultats)",
    )
    p.add_argument(
        "--sans-extraction",
        action="store_true",
        help="Sauter l'etape d'extraction (si les tables.json existent deja)",
    )
    p.add_argument(
        "--sans-comparaison",
        action="store_true",
        help="Sauter l'etape de comparaison (utile pour reextraire sans recomparer)",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------


def _step_extract(
    pdf_path: Path,
    bank: str,
    year: int,
    quarter: str,
    config: str,
    extraction_root: Path,
) -> Path:
    """Step 1: Run extraction on a single PDF and return the tables.json path."""
    from vigie.cli.run_extract_report import main as extract_main

    extract_main(
        [
            "--banque",
            bank,
            "--pdf",
            str(pdf_path),
            "--annee",
            str(year),
            "--trimestre",
            quarter,
            "--config",
            config,
            "--sortie",
            str(extraction_root),  # extraction writes {root}/{bank}/{year}/{quarter}/
        ]
    )

    # The extraction writes into {out_root}/{bank}/{year}/{quarter}/
    extraction_dir = extraction_root / bank.lower() / str(year) / quarter
    tables_json = extraction_dir / "tables.json"
    if not tables_json.exists():
        raise FileNotFoundError(f"Extraction terminée mais tables.json introuvable: {tables_json}")
    return tables_json


def _step_compare(
    bank: str,
    year_current: int,
    quarter_current: str,
    config: str,
    extraction_root: Path,
    out_root: Path,
    expected_comparison_dir: Path,
    pdf_previous: Path | None = None,
    pdf_current: Path | None = None,
) -> Path:
    """Step 2: Run GPT-4o comparison and return the comparison output path.

    Args:
        bank: Code banque.
        year_current: Annee du rapport courant.
        quarter_current: Trimestre courant (forme ``t1``..``t4``).
        config: Chemin YAML de configuration.
        extraction_root: Racine des artefacts d'extraction.
        out_root: Racine des sorties de comparaison.
        expected_comparison_dir: Chemin canonique attendu pour ce run
            (``outputs/resultats/{bank}/{year_q_vs_year_q}/``). Utilisé pour
            résoudre le ``comparison.json`` produit sans risque de retomber
            sur un fichier d'un autre run via une recherche globale.
        pdf_previous: PDF du trimestre de reference, optionnel.
        pdf_current: PDF du trimestre courant, optionnel.
    """
    from vigie.cli.run_compare_gpt4o import main as compare_main

    cmd = [
        "--banque",
        bank,
        "--annee-courante",
        str(year_current),
        "--trimestre-courant",
        quarter_current,
        "--config",
        config,
        "--racine-extraction",
        str(extraction_root),
        "--sortie",
        str(out_root),
    ]
    if pdf_previous:
        cmd += ["--pdf-precedent", str(pdf_previous)]
    if pdf_current:
        cmd += ["--pdf-courant", str(pdf_current)]

    compare_main(cmd)

    # Résolution stricte : on attend le ``comparison.json`` au chemin canonique
    # de ce run, jamais une recherche globale (qui retomberait sur un fichier
    # d'un autre run lexicographiquement « plus grand »).
    expected_path = expected_comparison_dir / "comparison.json"
    if expected_path.exists():
        return expected_path
    raise FileNotFoundError(
        f"Comparaison terminée mais ``{expected_path}`` est introuvable. "
        f"Vérifier que la commande compare_gpt4o a bien écrit dans ce dossier."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Executer le pipeline indicateurs de bout en bout."""
    args = build_parser().parse_args(argv)

    bank = args.banque.upper()
    year_current = args.annee
    q_current = normalize_quarter(args.trimestre)
    year_previous, q_previous = resolve_previous_quarter(year_current, q_current)
    config = args.config

    project_root = _PROJECT_ROOT
    inputs_root = project_root / args.entrees
    legacy_data = project_root / DEFAULT_LEGACY_DATA_ROOT

    print("=" * 70)
    print("  VIGILANCE — Pipeline Batch")
    print(f"  Banque:              {bank}")
    print(f"  Trimestre courant:   {q_current.upper()}-{year_current}")
    print(f"  Trimestre précédent: {q_previous.upper()}-{year_previous}  (déduit automatiquement)")
    print("=" * 70)

    # ── Locate PDFs ──────────────────────────────────────────────────────
    print("\nRecherche des PDFs…")
    previous_pdf, current_pdf = find_pdf_pair(
        bank=bank,
        year_current=year_current,
        quarter_current=q_current,
        inputs_root=inputs_root if inputs_root.is_dir() else None,
        legacy_data_root=legacy_data if legacy_data.is_dir() else None,
    )
    print(f"   Courant:   {current_pdf}")
    print(f"   Précédent: {previous_pdf}")

    # ── Canonical output paths ───────────────────────────────────────────
    extraction_root = project_root / "outputs" / "extractions"
    extraction_root.mkdir(parents=True, exist_ok=True)

    comparison_root = project_root / args.sortie
    comparison_dir = (
        comparison_root / bank.lower() / f"{year_current}_{q_current.lower()}_vs_{year_previous}_{q_previous.lower()}"
    )

    cur_extraction_dir = extraction_root / bank.lower() / str(year_current) / q_current
    prev_extraction_dir = extraction_root / bank.lower() / str(year_previous) / q_previous

    # ── Step 1: Extraction ───────────────────────────────────────────────
    if not args.sans_extraction:
        print("\n" + "─" * 70)
        print("ÉTAPE 1 — Extraction des tableaux")
        print("─" * 70)

        t0 = time.time()
        print(f"\n   Extraction du rapport courant ({q_current.upper()}-{year_current})…")
        _step_extract(current_pdf, bank.lower(), year_current, q_current, config, extraction_root)
        print(f"   tables.json → {cur_extraction_dir}")

        print(f"\n   Extraction du rapport précédent ({q_previous.upper()}-{year_previous})…")
        _step_extract(previous_pdf, bank.lower(), year_previous, q_previous, config, extraction_root)
        print(f"   tables.json → {prev_extraction_dir}")

        elapsed = time.time() - t0
        print(f"\n   Extraction terminée en {elapsed:.1f}s")
    else:
        print("\nExtraction ignoree (--sans-extraction)")

    # ── Step 2: Comparison ───────────────────────────────────────────────
    comparison_path: Path | None = None

    if not args.sans_comparaison:
        print("\n" + "─" * 70)
        print("ÉTAPE 2 — Comparaison sémantique (GPT-4o)")
        print("─" * 70)

        t0 = time.time()
        comparison_out = comparison_root
        comparison_path = _step_compare(
            bank=bank.lower(),
            year_current=year_current,
            quarter_current=q_current,
            config=config,
            extraction_root=extraction_root,
            out_root=comparison_out,
            expected_comparison_dir=comparison_dir,
            pdf_previous=previous_pdf,
            pdf_current=current_pdf,
        )
        elapsed = time.time() - t0
        print(f"\n   comparison.json → {comparison_dir}")
        print(f"   Comparaison terminée en {elapsed:.1f}s")
    else:
        print("\nComparaison ignoree (--sans-comparaison)")

    # ── Step 2.5: GenAI Triage (Batch LLM Analysis) ────────────────────
    if comparison_path and comparison_path.exists():
        print("\n" + "─" * 70)
        print("ÉTAPE 2.5 — Analyse GenAI (Triage de pertinence)")
        print("─" * 70)

        from vigie.comparaison.triage.genai_triage import enrich_comparison_with_genai_triage, inject_llm_resume_metier

        t0 = time.time()
        enrich_comparison_with_genai_triage(comparison_path)
        # Injection du résumé LLM dans genai_analysis["resume_metier"]
        # Charger, modifier, puis réécrire comparison.json
        import json

        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison = inject_llm_resume_metier(comparison)
        comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        elapsed = time.time() - t0
        print("   comparison.json enrichi avec l'analyse GenAI et résumé métier LLM injecté")
        print(f"   Triage GenAI terminé en {elapsed:.1f}s")

        # Export Excel avec justifications IA à côté de comparison.json
        from vigie.comparaison.excel import generate_comparison_excel

        excel_path = comparison_path.with_suffix(".xlsx")
        generate_comparison_excel(comparison, excel_path)
        print(f"   comparison.xlsx → {excel_path}")

    # ── Step 3: Manifest ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("ÉTAPE 3 — Génération du manifeste")
    print("─" * 70)

    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        comparison_dir,
        bank=bank,
        year_current=year_current,
        quarter_current=q_current,
        year_previous=year_previous,
        quarter_previous=q_previous,
        status="completed",
        extraction_dir_current=cur_extraction_dir,
        extraction_dir_previous=prev_extraction_dir,
        pdf_path_current=current_pdf,
        pdf_path_previous=previous_pdf,
    )
    print(f"   manifest.json → {comparison_dir / 'manifest.json'}")

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"   Extractions:       {extraction_root}")
    if comparison_path:
        print(f"   Comparaison:       {comparison_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
