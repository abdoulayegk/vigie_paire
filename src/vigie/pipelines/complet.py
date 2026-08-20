#!/usr/bin/env python
"""Orchestrateur unifié des pipelines Vigie (indicateurs + texte).

Usage::

    python run_full_pipeline.py --banque BNC --annee 2025 --T2

Ce script lance sequentiellement :

1. Le pipeline indicateurs (tableaux chiffres) via ``run_pipeline.main()``.
2. Le pipeline texte (risques, capital, etc.) via ``run_text_pipeline.main()``.

Toutes les sorties sont consolidees dans un dossier unique :
    outputs/resultats/{banque}/{annee_q_vs_annee_q}/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vigie.interface.ui_config import RESULTATS_DIR
from vigie.pipelines.indicateurs import main as indicateurs_main
from vigie.pipelines.texte import main as texte_main

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    DEFAULT_OUT_ROOT = str(RESULTATS_DIR.relative_to(_PROJECT_ROOT))
except ValueError:
    DEFAULT_OUT_ROOT = str(RESULTATS_DIR)


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI du pipeline complet."""
    p = argparse.ArgumentParser(
        description="Vigie -- Pipeline complet (indicateurs + texte).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python run_full_pipeline.py --banque BNC --annee 2025 --T2\n"
            "  python run_full_pipeline.py --banque BNC --annee 2025 --T2 --forcer-extraction\n"
            "  python run_full_pipeline.py --banque BNC --annee 2025 --T2 --sans-extraction\n"
            "  python run_full_pipeline.py --banque BNC --annee 2025 --T2 --sans-indicateurs\n"
            "  python run_full_pipeline.py --banque BNC --annee 2025 --T2 --sans-texte\n"
        ),
    )
    p.add_argument("--banque", required=True, help="Code de la banque (ex: BNC, RBC, TD)")
    p.add_argument("--annee", required=True, type=int, help="Annee du rapport courant (ex: 2025)")

    trimestre = p.add_mutually_exclusive_group(required=True)
    trimestre.add_argument("--T1", dest="trimestre", action="store_const", const="T1")
    trimestre.add_argument("--T2", dest="trimestre", action="store_const", const="T2")
    trimestre.add_argument("--T3", dest="trimestre", action="store_const", const="T3")
    trimestre.add_argument("--T4", dest="trimestre", action="store_const", const="T4")

    p.add_argument(
        "--sortie",
        default=DEFAULT_OUT_ROOT,
        help=f"Repertoire racine des resultats (defaut: {DEFAULT_OUT_ROOT})",
    )
    p.add_argument(
        "--sans-extraction",
        action="store_true",
        help="Sauter l'extraction des tableaux (reutiliser les tables.json existants)",
    )
    p.add_argument(
        "--forcer-extraction",
        action="store_true",
        help="Forcer la re-extraction (indicateurs: ignore cache Vision; texte: ignore text_extraction.md)",
    )
    p.add_argument(
        "--sans-comparaison",
        action="store_true",
        help="Sauter la comparaison semantique (indicateurs et texte)",
    )
    p.add_argument(
        "--sans-indicateurs",
        action="store_true",
        help="Ne pas executer le pipeline indicateurs (tableaux chiffres)",
    )
    p.add_argument(
        "--sans-texte",
        action="store_true",
        help="Ne pas executer le pipeline texte (risques, capital, etc.)",
    )
    return p


def _run_pipeline_indicateurs(
    banque: str,
    annee: int,
    trimestre: str,
    out_root: str,
    sans_extraction: bool,
    sans_comparaison: bool,
    forcer_extraction: bool,
) -> int:
    """Lance le pipeline indicateurs et retourne le code de sortie."""
    argv = [
        "--banque",
        banque,
        "--annee",
        str(annee),
        f"--{trimestre}",
        "--sortie",
        out_root,
    ]
    if sans_extraction:
        argv.append("--sans-extraction")
    if sans_comparaison:
        argv.append("--sans-comparaison")
    if forcer_extraction:
        argv.append("--forcer-extraction")

    return indicateurs_main(argv)


def _run_pipeline_texte(
    banque: str,
    annee: int,
    trimestre: str,
    out_root: str,
    sans_comparaison: bool,
    forcer_extraction: bool,
) -> int:
    """Lance le pipeline texte et retourne le code de sortie."""
    argv = [
        "--banque",
        banque,
        "--annee",
        str(annee),
        f"--{trimestre}",
        "--sortie",
        out_root,
    ]
    if sans_comparaison:
        argv.append("--sans-comparaison")
    if forcer_extraction:
        argv.append("--forcer-extraction")

    return texte_main(argv)


def main(argv: list[str] | None = None) -> int:
    """Exécute les pipelines indicateurs et texte selon les options CLI."""
    args = build_parser().parse_args(argv)

    banque = args.banque.upper()
    annee = args.annee
    trimestre = args.trimestre
    out_root = args.sortie

    print("=" * 70)
    print("  VIGIE -- Pipeline complet")
    print(f"  Banque:    {banque}")
    print(f"  Periode:   {trimestre}-{annee}")
    print(f"  Sortie:    {out_root}/")
    print("=" * 70)

    resultats: dict[str, str] = {}
    t_global = time.time()

    # -- Pipeline indicateurs --------------------------------------------------
    if args.sans_indicateurs:
        print("\n>> Pipeline indicateurs : ignore (--sans-indicateurs)")
        resultats["Indicateurs"] = "IGNORE"
    else:
        print("\n" + "=" * 70)
        print("  PIPELINE INDICATEURS (tableaux chiffres)")
        print("=" * 70)
        t0 = time.time()
        try:
            rc = _run_pipeline_indicateurs(
                banque=banque,
                annee=annee,
                trimestre=trimestre,
                out_root=out_root,
                sans_extraction=args.sans_extraction,
                sans_comparaison=args.sans_comparaison,
                forcer_extraction=args.forcer_extraction,
            )
            elapsed = time.time() - t0
            if rc == 0:
                resultats["Indicateurs"] = f"OK ({elapsed:.1f}s)"
            else:
                resultats["Indicateurs"] = f"ECHEC (code {rc}, {elapsed:.1f}s)"
        except Exception as exc:
            elapsed = time.time() - t0
            resultats["Indicateurs"] = f"ERREUR ({elapsed:.1f}s) -- {exc}"
            print(f"\n  ERREUR pipeline indicateurs : {exc}")

    # -- Pipeline texte --------------------------------------------------------
    if args.sans_texte:
        print("\n>> Pipeline texte : ignore (--sans-texte)")
        resultats["Texte"] = "IGNORE"
    else:
        print("\n" + "=" * 70)
        print("  PIPELINE TEXTE (risques, capital, etc.)")
        print("=" * 70)
        t0 = time.time()
        try:
            rc = _run_pipeline_texte(
                banque=banque,
                annee=annee,
                trimestre=trimestre,
                out_root=out_root,
                sans_comparaison=args.sans_comparaison,
                forcer_extraction=args.forcer_extraction,
            )
            elapsed = time.time() - t0
            if rc == 0:
                resultats["Texte"] = f"OK ({elapsed:.1f}s)"
            else:
                resultats["Texte"] = f"ECHEC (code {rc}, {elapsed:.1f}s)"
        except Exception as exc:
            elapsed = time.time() - t0
            resultats["Texte"] = f"ERREUR ({elapsed:.1f}s) -- {exc}"
            print(f"\n  ERREUR pipeline texte : {exc}")

    # -- Recapitulatif ---------------------------------------------------------
    elapsed_total = time.time() - t_global
    print("\n" + "=" * 70)
    print("  RECAPITULATIF")
    print("=" * 70)
    for nom, statut in resultats.items():
        print(f"  {nom:20s} {statut}")
    print(f"\n  Duree totale : {elapsed_total:.1f}s")
    print(f"  Resultats    : {out_root}/")
    print("=" * 70)

    has_failure = any(s.startswith("ECHEC") or s.startswith("ERREUR") for s in resultats.values())
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
