"""Section detection and PDF preview helpers for Dash."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from vigilance.extraction.pdf_preview import get_pdf_info, render_pdf_page, render_pdf_pages

_SECTION_LABELS = {
    "gestion_capital": "Gestion du capital",
    "capital_management": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "risk_management": "Gestion des risques",
    "gestion_reglementation": "Réglementation",
    "regulatory_updates": "Réglementation",
    "reglementation": "Réglementation",
}


def _label_for(section_type: str) -> str:
    return _SECTION_LABELS.get(section_type, section_type.replace("_", " ").title())


def _normalize_section_type(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return "unknown_section"
    lowered = raw.lower()
    lowered = lowered.replace("é", "e").replace("è", "e")
    if "capital" in lowered or "fonds propres" in lowered:
        return "gestion_capital"
    if "risque" in lowered:
        return "gestion_risques"
    if "reglement" in lowered:
        return "gestion_reglementation"
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered or "unknown_section"


def _fallback_sections(total_pages: int) -> list[dict[str, Any]]:
    if total_pages <= 0:
        total_pages = 1
    midpoint = max(1, total_pages // 2)
    return [
        {
            "type": "gestion_capital",
            "label": "Gestion du capital",
            "start_page": 1,
            "end_page": midpoint,
        },
        {
            "type": "gestion_risques",
            "label": "Gestion des risques",
            "start_page": midpoint + 1 if midpoint < total_pages else midpoint,
            "end_page": total_pages,
        },
    ]


def _detect_sections_core(pdf_path: str | Path, bank_code: str | None = None) -> dict[str, Any]:
    """Detect key sections in a PDF and return UI-friendly ranges."""
    path = str(pdf_path)
    info = get_pdf_info(path)
    total_pages = int(info.get("total_pages", 0) or 0)

    try:
        from vigilance.extraction.section_locator import locate_sections_in_pdf

        mapping = locate_sections_in_pdf(path, bank_code=bank_code, quarter="dash")
        sections: list[dict[str, Any]] = []
        for item in getattr(mapping, "sections", []) or []:
            section_type = _normalize_section_type(str(getattr(item, "section_type", "")))
            start = int(getattr(item, "start_page", 1) or 1)
            end = int(getattr(item, "end_page", start) or start)
            if end < start:
                end = start
            sections.append(
                {
                    "type": section_type,
                    "label": _label_for(section_type),
                    "start_page": start,
                    "end_page": end,
                }
            )

        if not sections:
            sections = _fallback_sections(total_pages)
        sections.sort(key=lambda s: int(s.get("start_page", 0)))
        return {"sections": sections, "total_pages": int(getattr(mapping, "total_pages", total_pages) or total_pages)}
    except Exception:
        return {"sections": _fallback_sections(total_pages), "total_pages": total_pages}


def get_pdf_preview(pdf_path: str | Path, page: int, scale: float = 1.5) -> bytes | None:
    """Render one page as bytes for inline display."""
    if page is None:
        return None
    return render_pdf_page(pdf_path=pdf_path, page_number=int(page), scale=scale)


def get_section_preview_images(
    pdf_path: str | Path,
    section: dict[str, Any],
    *,
    max_pages: int = 5,
    scale: float = 1.2,
) -> list[str]:
    """Render section pages and return base64 PNG images for Dash."""
    start = int(section.get("start_page", 1) or 1)
    end = int(section.get("end_page", start) or start)
    if end < start:
        end = start

    previews = render_pdf_pages(
        pdf_path=pdf_path,
        start_page=start,
        end_page=end,
        max_pages=max_pages,
        scale=scale,
    )

    result: list[str] = []
    for preview in previews:
        raw = getattr(preview, "image_bytes", b"")
        if raw:
            result.append(base64.b64encode(raw).decode("ascii"))
    return result
