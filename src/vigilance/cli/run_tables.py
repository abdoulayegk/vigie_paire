"""CLI facade for table extraction on previously detected page ranges."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from vigilance.config.loader import get_bank_cfg, load_config
from vigilance.models.table_models import TableArtifact
from vigilance.utils.rbc_table_signals import (
    build_rbc_first_column_signals,
    classify_rbc_title_reliability,
    is_rbc_bank,
)
from vigilance.report.export_json import write_tables_docling

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/runs"


def _canonicalize_section(raw: str | None) -> str | None:
    if raw is None:
        return None
    try:
        from vigilance.extraction.section_taxonomy import canonicalize_section

        return canonicalize_section(raw)
    except Exception:
        fallback = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        return fallback or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract tables on selected page ranges.")
    parser.add_argument("--bank", required=True, help="Bank code (e.g. rbc)")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--quarter", required=True, help="Quarter label (e.g. t1-2025)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--ranges_json", required=True, help="Path to section_ranges.json")
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Output root directory")
    parser.add_argument(
        "--vigie_extract",
        action="store_true",
        default=False,
        help="Also produce a single vigie_extract_v1 JSON per PDF",
    )
    parser.add_argument("--language", default="fr", help="Language code for vigie_extract (default: fr)")
    return parser


def _infer_year(quarter: str) -> int:
    match = re.search(r"(19|20)\d{2}", quarter)
    return int(match.group(0)) if match else 2025


def _load_section_ranges(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    section_ranges: list[dict[str, Any]] = []

    ranges = data.get("section_ranges")
    if isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, dict):
                continue
            section = _canonicalize_section(str(item.get("section", "")).strip())
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", start) or start)
            if section and start > 0 and end >= start:
                entry: dict[str, Any] = {"section": section, "start": start, "end": end}
                if item.get("evidence"):
                    entry["evidence"] = item["evidence"]
                section_ranges.append(entry)

    ranges = data.get("ranges")
    if not section_ranges and isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, dict):
                continue
            section = _canonicalize_section(str(item.get("section", "")).strip())
            start = int(item.get("start_page_pdf", 0) or 0)
            end = int(item.get("end_page_pdf", start) or start)
            if section and start > 0 and end >= start:
                entry = {"section": section, "start": start, "end": end}
                if item.get("evidence"):
                    entry["evidence"] = item["evidence"]
                section_ranges.append(entry)

    if not section_ranges and isinstance(data.get("sections"), dict):
        for section_name, item in data["sections"].items():
            if not isinstance(item, dict):
                continue
            section = _canonicalize_section(str(item.get("section", section_name)).strip())
            start = int(item.get("start_page", 0) or 0)
            end = int(item.get("end_page", start) or start)
            if section and start > 0 and end >= start:
                entry = {"section": section, "start": start, "end": end}
                if item.get("evidence"):
                    entry["evidence"] = item["evidence"]
                section_ranges.append(entry)

    if not section_ranges:
        raise ValueError(f"No valid section ranges found in {path}")
    return section_ranges


def _to_artifacts(raw_tables: list[Any], bank: str, quarter: str, pdf_path: str) -> list[TableArtifact]:
    artifacts: list[TableArtifact] = []
    for index, table in enumerate(raw_tables, start=1):
        raw_indicators = list(getattr(table, "first_column_indicators", []) or [])
        if raw_indicators:
            indicators = [str(item).strip() for item in raw_indicators if str(item).strip()]
        else:
            indicators = []
            for raw_row in list(getattr(table, "rows", []) or []):
                if isinstance(raw_row, list) and raw_row:
                    label = str(raw_row[0]).strip()
                    if label:
                        indicators.append(label)

        raw = getattr(table, "first_column_indicators_raw", None)
        if raw is not None:
            raw = [str(x).strip() for x in raw if str(x).strip()]
        else:
            raw = None

        rows = [list(row) for row in (getattr(table, "rows", []) or [])]
        first_column_groups: list[str] | None = None
        hierarchical_indicator_signature: list[str] | None = None
        if is_rbc_bank(bank):
            rbc_signals = build_rbc_first_column_signals(
                rows=rows,
                raw_indicators=raw or indicators,
            )
            if rbc_signals.indicators_raw:
                indicators = list(rbc_signals.indicators_clean)
                raw = list(rbc_signals.indicators_raw)
            first_column_groups = list(rbc_signals.groups_raw)
            hierarchical_indicator_signature = list(
                rbc_signals.hierarchical_indicator_signature
            )

        section = _canonicalize_section(getattr(table, "section", None)) or "unknown_section"
        artifacts.append(
            TableArtifact(
                bank_code=bank,
                section=section,
                page_pdf=int(getattr(table, "page_number", 0) or 0),
                table_id=str(getattr(table, "table_id", f"table_{index}")),
                title=getattr(table, "title", None),
                headers=list(getattr(table, "headers", []) or []),
                rows=rows,
                first_column_indicators=indicators,
                first_column_indicators_raw=raw,
                first_column_groups=first_column_groups,
                hierarchical_indicator_signature=hierarchical_indicator_signature,
                table_number=getattr(table, "table_number", None),
                bbox=getattr(table, "bbox", None),
                extraction_method=getattr(table, "extraction_method", None) or "docling",
                quarter=quarter,
                pdf_path=pdf_path,
                title_reliability=classify_rbc_title_reliability(
                    getattr(table, "title_clean", None) or getattr(table, "title", None),
                    bank_code=bank,
                ),
            )
        )
    return artifacts


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.bank)
    section_ranges = _load_section_ranges(args.ranges_json)

    try:
        from vigilance.extraction.docling_processor import extract_tables_docling_by_sections
    except Exception as exc:
        raise NotImplementedError(
            "Docling extraction backend from extraction/ is not importable in this environment."
        ) from exc

    year = _infer_year(args.quarter)
    raw_tables = extract_tables_docling_by_sections(
        pdf_path=args.pdf,
        bank_code=args.bank,
        quarter=args.quarter,
        year=year,
        section_ranges=section_ranges,
    )
    artifacts = _to_artifacts(raw_tables, bank=args.bank, quarter=args.quarter, pdf_path=args.pdf)
    out_dir = Path(args.out_root) / args.quarter / args.bank
    out_path = write_tables_docling(out_dir=out_dir, tables=artifacts)
    print(out_path)

    if args.vigie_extract:
        from vigilance.report.vigie_extract_schema import build_vigie_extract, write_vigie_extract

        payload = build_vigie_extract(
            pdf_path=args.pdf,
            bank_code=args.bank,
            quarter=args.quarter,
            year=year,
            language=args.language,
            section_ranges=section_ranges,
            tables=raw_tables,
        )
        vigie_path = write_vigie_extract(out_dir=out_dir, payload=payload)
        print(vigie_path)


if __name__ == "__main__":
    main()
