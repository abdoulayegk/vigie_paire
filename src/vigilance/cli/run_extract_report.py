"""Minimal CLI for section-targeted report extraction artifacts.

This entrypoint preserves the existing extraction pipeline:
- detect relevant sections in the PDF
- extract tables only on those page ranges
- write compact ``tables.json``, ``indicators.json``, and ``footnotes.json``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from vigilance.config import resolve_openai_model
from vigilance.config.loader import get_bank_cfg, load_config
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.extraction.vision_extraction_writer import (
    write_compact_report_artifacts,
)

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/extractions"


def _normalize_storage_quarter(quarter: str) -> str:
    value = str(quarter or "").strip().lower()
    match = re.search(r"([qt])\s*([1-4])", value, flags=re.IGNORECASE)
    if match:
        return f"t{match.group(2)}"
    return value or "t1"


def _build_section_ranges(mapping: Any) -> list[dict[str, Any]]:
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
    return ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract compact tables/indicators/footnotes artifacts for one report."
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. bnc)")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--year", required=True, type=int, help="Report year (e.g. 2025)")
    parser.add_argument("--quarter", required=True, help="Quarter label (e.g. t1)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="Output root directory (default: outputs/extractions)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.bank)

    try:
        from vigilance.extraction.docling_processor import extract_tables_docling_by_sections
        from vigilance.extraction.section_locator import locate_sections_in_pdf
    except Exception as exc:
        raise NotImplementedError(
            "Extraction backends are not importable in this environment."
        ) from exc

    quarter_norm = _normalize_storage_quarter(args.quarter)
    mapping = locate_sections_in_pdf(
        args.pdf,
        bank_code=args.bank,
        quarter=quarter_norm,
        year=int(args.year),
    )
    section_ranges = _build_section_ranges(mapping)
    if not section_ranges:
        raise ValueError(f"No valid section ranges detected for {args.pdf}")

    tables = extract_tables_docling_by_sections(
        pdf_path=args.pdf,
        bank_code=args.bank,
        quarter=quarter_norm,
        year=int(args.year),
        section_ranges=section_ranges,
    )

    out_dir = Path(args.out_root) / str(args.bank).lower() / str(args.year) / quarter_norm
    model_version = ""
    try:
        model_version = resolve_openai_model("extraction_primary", config_path=args.config)
    except Exception:
        model_version = ""

    paths = write_compact_report_artifacts(
        tables=tables,
        out_dir=out_dir,
        bank_code=str(args.bank).lower(),
        year=int(args.year),
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
