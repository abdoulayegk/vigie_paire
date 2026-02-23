"""CLI facade for section range detection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from vigilance.config.loader import get_bank_cfg, load_config
from vigilance.models.section_models import SectionRange, SectionRangesResult
from vigilance.report.export_json import write_section_ranges

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/runs"


def _canonicalize_section(raw: str) -> str:
    try:
        from vigilance.extraction.section_taxonomy import canonicalize_section

        return canonicalize_section(raw)
    except Exception:
        fallback = re.sub(r"[^a-z0-9]+", "_", (raw or "").lower()).strip("_")
        return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect section ranges for one PDF.")
    parser.add_argument("--bank", required=True, help="Bank code (e.g. rbc)")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--quarter", required=True, help="Quarter label (e.g. t1-2025)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Output root directory")
    return parser


def _to_result(mapping: Any, bank_code: str, quarter: str, pdf_path: str) -> SectionRangesResult:
    ranges: list[SectionRange] = []
    for located in getattr(mapping, "sections", []):
        start_page = int(getattr(located, "start_page", 0) or 0)
        if start_page <= 0:
            continue
        end_page = int(getattr(located, "end_page", start_page) or start_page)
        ranges.append(
            SectionRange(
                section=_canonicalize_section(str(getattr(located, "section_type", ""))),
                start_page_pdf=start_page,
                end_page_pdf=end_page,
                method=str(getattr(located, "detection_method", "")),
                confidence=float(getattr(located, "confidence", 0.0) or 0.0),
                evidence={
                    "title_found": getattr(located, "title_found", ""),
                    "end_detection_method": getattr(located, "end_detection_method", ""),
                    "detected_span": getattr(located, "detected_span", None),
                    "final_span": getattr(located, "final_span", None),
                    "constraint_applied": getattr(located, "constraint_applied", False),
                    "constraint_reason": getattr(located, "constraint_reason", ""),
                },
            )
        )
    return SectionRangesResult(bank_code=bank_code, quarter=quarter, pdf_path=pdf_path, ranges=ranges)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.bank)

    try:
        from vigilance.extraction.section_locator import locate_sections_in_pdf
    except Exception as exc:
        raise NotImplementedError(
            "Section detection backend from extraction/ is not importable in this environment."
        ) from exc

    mapping = locate_sections_in_pdf(args.pdf, bank_code=args.bank, quarter=args.quarter)
    result = _to_result(mapping, bank_code=args.bank, quarter=args.quarter, pdf_path=args.pdf)
    out_dir = Path(args.out_root) / args.quarter / args.bank
    out_path = write_section_ranges(out_dir=out_dir, result=result)
    print(out_path)


if __name__ == "__main__":
    main()
