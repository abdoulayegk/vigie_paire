#!/usr/bin/env python
"""Batch pipeline orchestrator for the Vigilance system.

Usage::

    python run_pipeline.py --bank BNC --year 2025 --quarter T2

This single command will:

1. Deduce the previous quarter automatically (T2→T1, T1→T3 N-1, …).
2. Locate the two PDF reports.
3. Extract tables from both reports (Docling for layout, GPT-4o Vision for table content).
4. Compare the two sets of tables semantically (GPT-4o).
5. Generate all output files (comparison.json, indicators.json, …).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the src/ directory is importable when running from project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vigilance.cli.output_builder import (
    build_run_dir,
    split_audit_files,
    write_run_manifest,
)
from vigilance.cli.quarter_logic import (
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
DEFAULT_OUTPUTS_ROOT = "Outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vigilance — Pipeline Batch de Nuit (Extraction + Comparaison).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python run_pipeline.py --bank BNC --year 2025 --quarter T2\n"
            "  python run_pipeline.py --bank RBC --year 2025 --quarter T1 --skip-extraction\n"
        ),
    )
    p.add_argument("--bank", required=True, help="Code de la banque (ex: BNC, RBC, TD)")
    p.add_argument(
        "--year", required=True, type=int, help="Année du rapport courant (ex: 2025)"
    )
    p.add_argument(
        "--quarter", required=True, help="Trimestre courant (ex: T1, T2, T3)"
    )
    p.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Chemin YAML de configuration"
    )
    p.add_argument(
        "--inputs-root", default=DEFAULT_INPUTS_ROOT, help="Répertoire racine des PDFs"
    )
    p.add_argument(
        "--outputs-root",
        default=DEFAULT_OUTPUTS_ROOT,
        help="Répertoire racine de sortie",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Sauter l'étape d'extraction (utile si les tables.json existent déjà)",
    )
    p.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Sauter l'étape de comparaison (utile pour réextraire sans recomparer)",
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
    from vigilance.cli.run_extract_report import main as extract_main

    extract_main(
        [
            "--bank",
            bank,
            "--pdf",
            str(pdf_path),
            "--year",
            str(year),
            "--quarter",
            quarter,
            "--config",
            config,
            "--out-root",
            str(extraction_root),  # extraction writes {root}/{bank}/{year}/{quarter}/
        ]
    )

    # The extraction writes into {out_root}/{bank}/{year}/{quarter}/
    extraction_dir = extraction_root / bank.lower() / str(year) / quarter
    tables_json = extraction_dir / "tables.json"
    if not tables_json.exists():
        raise FileNotFoundError(
            f"Extraction terminée mais tables.json introuvable: {tables_json}"
        )
    return tables_json


def _step_compare(
    bank: str,
    year_current: int,
    quarter_current: str,
    config: str,
    extraction_root: Path,
    out_root: Path,
) -> Path:
    """Step 2: Run GPT-4o comparison and return the comparison output path."""
    from vigilance.cli.run_compare_gpt4o import main as compare_main

    compare_main(
        [
            "--bank",
            bank,
            "--year-current",
            str(year_current),
            "--quarter-current",
            quarter_current,
            "--config",
            config,
            "--extraction-root",
            str(extraction_root),
            "--out-root",
            str(out_root),
        ]
    )

    # Find the comparison.json that was produced
    comparison_dir = out_root
    candidates = sorted(comparison_dir.rglob("comparison.json"))
    if not candidates:
        # Also check for any JSON output
        candidates = sorted(comparison_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Comparaison terminée mais aucun fichier de sortie trouvé dans: {out_root}"
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    bank = args.bank.upper()
    year_current = args.year
    q_current = normalize_quarter(args.quarter)
    year_previous, q_previous = resolve_previous_quarter(year_current, q_current)
    config = args.config

    project_root = Path(__file__).resolve().parent
    inputs_root = project_root / args.inputs_root
    outputs_root = project_root / args.outputs_root
    legacy_data = project_root / DEFAULT_LEGACY_DATA_ROOT

    print("=" * 70)
    print("  VIGILANCE — Pipeline Batch")
    print(f"  Banque:              {bank}")
    print(f"  Trimestre courant:   {q_current.upper()}-{year_current}")
    print(
        f"  Trimestre précédent: {q_previous.upper()}-{year_previous}  (déduit automatiquement)"
    )
    print("=" * 70)

    # ── Locate PDFs ──────────────────────────────────────────────────────
    print("\n📂 Recherche des PDFs…")
    previous_pdf, current_pdf = find_pdf_pair(
        bank=bank,
        year_current=year_current,
        quarter_current=q_current,
        inputs_root=inputs_root if inputs_root.is_dir() else None,
        legacy_data_root=legacy_data if legacy_data.is_dir() else None,
    )
    print(f"   ✓ Courant:   {current_pdf}")
    print(f"   ✓ Précédent: {previous_pdf}")

    # ── Build Run Directory ──────────────────────────────────────────────
    run_dir = build_run_dir(
        out_root=outputs_root,
        bank=bank,
        year_current=year_current,
        quarter_current=q_current,
        year_previous=year_previous,
        quarter_previous=q_previous,
    )
    cur_sub = run_dir / f"{q_current.upper()}-{year_current}"
    prev_sub = run_dir / f"{q_previous.upper()}-{year_previous}"

    print(f"\n📁 Dossier du Run: {run_dir}")

    # ── Step 1: Extraction ───────────────────────────────────────────────
    extraction_root = run_dir / "_extractions"
    extraction_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_extraction:
        print("\n" + "─" * 70)
        print("⚗️  ÉTAPE 1 — Extraction des tableaux")
        print("─" * 70)

        t0 = time.time()
        print(
            f"\n   Extraction du rapport courant ({q_current.upper()}-{year_current})…"
        )
        cur_tables = _step_extract(
            current_pdf, bank.lower(), year_current, q_current, config, extraction_root
        )
        shutil.copy2(cur_tables, cur_sub / "tables.json")
        split_audit_files(cur_tables, cur_sub)
        print(f"   ✓ tables.json, indicators.json, footnotes.json → {cur_sub}")

        print(
            f"\n   Extraction du rapport précédent ({q_previous.upper()}-{year_previous})…"
        )
        prev_tables = _step_extract(
            previous_pdf,
            bank.lower(),
            year_previous,
            q_previous,
            config,
            extraction_root,
        )
        shutil.copy2(prev_tables, prev_sub / "tables.json")
        split_audit_files(prev_tables, prev_sub)
        print(f"   ✓ tables.json, indicators.json, footnotes.json → {prev_sub}")

        elapsed = time.time() - t0
        print(f"\n   ⏱  Extraction terminée en {elapsed:.1f}s")
    else:
        print("\n⏩ Extraction ignorée (--skip-extraction)")

    # ── Step 2: Comparison ───────────────────────────────────────────────
    if not args.skip_comparison:
        print("\n" + "─" * 70)
        print("🔍 ÉTAPE 2 — Comparaison sémantique (GPT-4o)")
        print("─" * 70)

        t0 = time.time()
        comparison_out = run_dir / "_comparisons"
        comparison_path = _step_compare(
            bank=bank.lower(),
            year_current=year_current,
            quarter_current=q_current,
            config=config,
            extraction_root=extraction_root,
            out_root=comparison_out,
        )
        # Copy comparison.json to run root
        final_comparison = run_dir / "comparison.json"
        shutil.copy2(comparison_path, final_comparison)
        elapsed = time.time() - t0
        print(f"\n   ✓ comparison.json → {final_comparison}")
        print(f"   ⏱  Comparaison terminée en {elapsed:.1f}s")
    else:
        print("\n⏩ Comparaison ignorée (--skip-comparison)")

    # ── Step 2.5: GenAI Triage (Batch LLM Analysis) ────────────────────
    if (
        not args.skip_comparison
        and "final_comparison" in locals()
        and final_comparison.exists()
    ):
        print("\n" + "─" * 70)
        print("🧠 ÉTAPE 2.5 — Analyse GenAI (Triage de pertinence)")
        print("─" * 70)

        from vigilance.genai_triage import enrich_comparison_with_genai_triage

        t0 = time.time()
        enrich_comparison_with_genai_triage(final_comparison)
        elapsed = time.time() - t0
        print("   ✓ comparison.json enrichi avec l'analyse GenAI")
        print(f"   ⏱  Triage GenAI terminé en {elapsed:.1f}s")

    # ── Step 3: Manifest & Summary ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("📋 ÉTAPE 3 — Génération du manifeste")
    print("─" * 70)

    write_run_manifest(
        run_dir,
        bank=bank,
        year_current=year_current,
        quarter_current=q_current,
        year_previous=year_previous,
        quarter_previous=q_previous,
        status="completed",
    )
    print(f"   ✓ manifest.json → {run_dir / 'manifest.json'}")

    # ── Step 4: Deploy to Dash ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🚀 ÉTAPE 4 — Déploiement vers le Dashboard (Dash)")
    print("─" * 70)
    if (
        not args.skip_comparison
        and "final_comparison" in locals()
        and final_comparison.exists()
    ):
        dash_target_dir = (
            project_root
            / "outputs"
            / "comparisons"
            / bank.lower()
            / f"{year_current}_{q_current.lower()}_vs_{year_previous}_{q_previous.lower()}"
        )
        dash_target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_comparison, dash_target_dir / "comparison.json")
        if current_pdf and current_pdf.exists():
            shutil.copy2(current_pdf, dash_target_dir / "current_report.pdf")
        if previous_pdf and previous_pdf.exists():
            shutil.copy2(previous_pdf, dash_target_dir / "previous_report.pdf")
        print(
            f"   ✓ Données déployées et prêtes pour Dash dans: {dash_target_dir.relative_to(project_root)}"
        )

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"   Dossier du Run: {run_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
