"""CLI minimal pour l'extraction ciblee par section des artefacts de rapport.

Ce point d'entree preserve le pipeline d'extraction existant :
- detecter les sections pertinentes dans le PDF
- extraire les tableaux uniquement sur les plages de pages ciblees
- ecrire les artefacts compacts ``tables.json``, ``indicators.json`` et ``footnotes.json``
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import Any

from vigie.extraction.section_taxonomy import canonicalize_section
from vigie.extraction.vision_extraction_writer import (
    write_compact_report_artifacts,
)
from vigie.support.config import resolve_openai_model
from vigie.support.config.loader import get_bank_cfg, load_config

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/extractions"


def _normalize_storage_quarter(quarter: str) -> str:
    """Normaliser un libelle de trimestre en ``t1``..``t4``."""
    value = str(quarter or "").strip().lower()
    match = re.search(r"([qt])\s*([1-4])", value, flags=re.IGNORECASE)
    if match:
        return f"t{match.group(2)}"
    return value or "t1"


def _is_t4(quarter: str) -> bool:
    """Indiquer si le libelle correspond au T4."""
    return bool(re.search(r"\b[qt]\s*4\b|^t4(?:[_-]|\b)", str(quarter), re.I))


def _filter_t4_target_ranges(ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conserver uniquement les sections cibles T4, sans combler les trous."""
    target_sections = {"capital_management", "risk_management"}
    target_ranges = [item for item in ranges if item.get("section") in target_sections]
    return target_ranges or ranges


def _build_section_ranges(mapping: Any, quarter: str = "") -> list[dict[str, Any]]:
    """Convertir le resultat du locator en plages de sections pour l'extraction."""
    ranges: list[dict[str, Any]] = []
    for located in getattr(mapping, "sections", []) or []:
        start = int(getattr(located, "start_page", 0) or 0)
        if start <= 0:
            continue
        end = int(getattr(located, "end_page", start) or start)
        if end < start:
            end = start
        section = canonicalize_section(str(getattr(located, "section_type", "") or ""))
        if not section:
            section = "unknown_section"
        ranges.append({"section": section, "start": start, "end": end})
    if _is_t4(quarter):
        return _filter_t4_target_ranges(ranges)
    return ranges


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur d'arguments pour l'extraction de rapport."""
    parser = argparse.ArgumentParser(description="Extraire les artefacts tables/indicateurs/notes pour un rapport.")
    parser.add_argument("--banque", required=True, help="Code banque (ex: bnc)")
    parser.add_argument("--pdf", required=True, help="Chemin du PDF d'entree")
    parser.add_argument("--annee", required=True, type=int, help="Annee du rapport (ex: 2025)")
    parser.add_argument("--trimestre", required=True, help="Libelle trimestre (ex: t1)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Chemin YAML de configuration")
    parser.add_argument(
        "--sortie",
        default=DEFAULT_OUT_ROOT,
        help="Racine des sorties (defaut: outputs/extractions)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Detecter les sections d'un PDF et extraire les artefacts compacts."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.banque)

    processor_module = importlib.import_module("vigie.extraction.docling.processor")
    locator_module = importlib.import_module("vigie.extraction.localisation_sections.section_locator")
    extract_tables_docling_by_sections = processor_module.extract_tables_docling_by_sections
    locate_sections_in_pdf = locator_module.locate_sections_in_pdf

    quarter_norm = _normalize_storage_quarter(args.trimestre)
    mapping = locate_sections_in_pdf(
        args.pdf,
        bank_code=args.banque,
        quarter=quarter_norm,
        year=int(args.annee),
    )
    section_ranges = _build_section_ranges(mapping, args.trimestre)
    if not section_ranges:
        raise ValueError(f"No valid section ranges detected for {args.pdf}")

    tables = extract_tables_docling_by_sections(
        pdf_path=args.pdf,
        bank_code=args.banque,
        quarter=quarter_norm,
        year=int(args.annee),
        section_ranges=section_ranges,
    )

    out_dir = Path(args.sortie) / str(args.banque).lower() / str(args.annee) / quarter_norm
    model_version = ""
    try:
        model_version = resolve_openai_model("extraction_primary", config_path=args.config)
    except Exception:
        model_version = ""

    paths = write_compact_report_artifacts(
        tables=tables,
        out_dir=out_dir,
        bank_code=str(args.banque).lower(),
        year=int(args.annee),
        quarter=quarter_norm,
        meta={
            "model_version": model_version,
            "prompt_version": "",
        },
    )
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing output artifacts after extraction: {', '.join(missing)}")
    print(out_dir)


if __name__ == "__main__":
    main()
