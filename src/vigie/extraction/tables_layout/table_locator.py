"""Detection des tableaux via PyMuPDF4LLM + Layout (sans OCR)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from vigie.extraction.table_locator.models import TableAnchor, TableLocationResult
from vigie.extraction.tables_layout.table_bbox import normalize_pymupdf_bbox, page_number_from_layout
from vigie.extraction.tables_layout.table_reference_text import build_reference_text_from_table_block
from vigie.support.utils.pymupdf_utils import configure_mupdf_runtime

logger = logging.getLogger("vigie.extraction.tables_layout")

SOURCE = "tables_layout"


class TablesLayoutLocator:
    """Localiser les tableaux PDF natifs avec PyMuPDF Layout (``use_ocr=False``)."""

    def locate(
        self,
        pdf_path: Path,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        reference_text_max_chars: int = 6000,
    ) -> TableLocationResult:
        """Detecter les tableaux et retourner des ancres normalisees.

        Raises:
            RuntimeError: Si Layout / pymupdf4llm echoue (pas de fallback Docling).
        """
        from vigie.extraction.tables_layout.tables_layout_pass import (  # noqa: PLC0415 - orchestration dediee
            run_tables_layout_pass,
        )

        return run_tables_layout_pass(
            pdf_path,
            page_ranges=page_ranges,
            reference_text_max_chars=reference_text_max_chars,
        )


def _ensure_layout_enabled() -> None:
    """Activer le module Layout de PyMuPDF4LLM si disponible."""
    try:
        import pymupdf.layout  # noqa: F401, PLC0415 - verification disponibilite Layout
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF Layout is required for TABLE_LOCATOR_ENGINE=pymupdf_layout but pymupdf.layout is not installed"
        ) from exc
    pymupdf4llm.use_layout(True)


def _page_indices_0_based(
    page_ranges: list[tuple[int, int]] | None,
    total_pages: int,
) -> list[int] | None:
    """Convertir des plages 1-indexees en indices 0-based pour pymupdf4llm."""
    if not page_ranges:
        return None
    indices: list[int] = []
    for start, end in page_ranges:
        for page_1based in range(int(start), int(end) + 1):
            if 1 <= page_1based <= total_pages:
                indices.append(page_1based - 1)
    return sorted(set(indices))


def _load_layout_json(
    pdf_path: Path,
    *,
    pages_0_based: list[int] | None,
) -> dict[str, Any]:
    """Appeler ``pymupdf4llm.to_json`` avec OCR desactive."""
    _ensure_layout_enabled()
    kwargs: dict[str, Any] = {"use_ocr": False, "force_ocr": False}
    if pages_0_based is not None:
        kwargs["pages"] = pages_0_based
    raw = pymupdf4llm.to_json(str(pdf_path), **kwargs)
    if isinstance(raw, str):
        data = json.loads(raw)
    elif isinstance(raw, dict):
        data = raw
    else:
        raise RuntimeError(f"Unexpected pymupdf4llm.to_json return type: {type(raw)!r}")
    if not isinstance(data, dict):
        raise RuntimeError("pymupdf4llm.to_json did not return a JSON object")
    return data


def _inventory_pages_from_ranges(
    page_ranges: list[tuple[int, int]] | None,
    total_pages: int,
) -> list[int]:
    """Pages 1-indexees a proposer a l'inventaire Vision page-context."""
    if total_pages <= 0:
        return []
    if not page_ranges:
        return list(range(1, total_pages + 1))
    pages: list[int] = []
    for start, end in page_ranges:
        for page_1based in range(int(start), int(end) + 1):
            if 1 <= page_1based <= total_pages:
                pages.append(page_1based)
    return sorted(set(pages))


def detect_table_anchors(
    pdf_path: Path,
    page_ranges: list[tuple[int, int]] | None = None,
    *,
    reference_text_max_chars: int = 6000,
) -> TableLocationResult:
    """Executer la detection Layout et construire le resultat d'ancrage."""
    configure_mupdf_runtime(pymupdf)
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF non trouve: {path}")

    with pymupdf.open(str(path)) as doc:
        total_pages = len(doc)

    pages_0_based = _page_indices_0_based(page_ranges, total_pages)
    try:
        layout_data = _load_layout_json(path, pages_0_based=pages_0_based)
    except Exception as exc:
        raise RuntimeError(f"tables_layout detection failed for {path}: {exc}") from exc

    pages = layout_data.get("pages") or []
    if not isinstance(pages, list):
        raise RuntimeError("tables_layout JSON missing pages list")

    anchors: list[TableAnchor] = []
    table_idx = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page_number_from_layout(int(page.get("page_number") or 0))
        page_width = float(page.get("width") or 0.0)
        page_height = float(page.get("height") or 0.0)
        boxes = page.get("boxes") or []
        if not isinstance(boxes, list):
            continue
        page_table_count = 0
        for box in boxes:
            if not isinstance(box, dict) or box.get("boxclass") != "table":
                continue
            table_payload = box.get("table") if isinstance(box.get("table"), dict) else {}
            raw_bbox = table_payload.get("bbox") if table_payload else box.get("bbox")
            table_bbox: list[float] | None = None
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                try:
                    table_bbox = normalize_pymupdf_bbox(raw_bbox, page_width, page_height)
                except ValueError as exc:
                    logger.warning(
                        "tables_layout invalid bbox page=%s table=%s: %s",
                        page_number,
                        table_idx,
                        exc,
                    )
                    table_bbox = None
            reference_text = build_reference_text_from_table_block(
                table_payload if isinstance(table_payload, dict) else None,
                max_chars=reference_text_max_chars,
            )
            table_id = f"tableau_{table_idx}"
            anchors.append(
                TableAnchor(
                    table_id=table_id,
                    page_number=page_number,
                    bbox=table_bbox,
                    reference_text=reference_text,
                    source=SOURCE,
                )
            )
            logger.info(
                "tables_layout table_id=%s page=%s bbox=%s",
                table_id,
                page_number,
                table_bbox,
            )
            table_idx += 1
            page_table_count += 1
        logger.info(
            "tables_layout page=%s tables_detected=%s",
            page_number,
            page_table_count,
        )

    text_content = ""
    try:
        md_kwargs: dict[str, Any] = {"use_ocr": False, "force_ocr": False}
        if pages_0_based is not None:
            md_kwargs["pages"] = pages_0_based
        md = pymupdf4llm.to_markdown(str(path), **md_kwargs)
        text_content = md if isinstance(md, str) else str(md or "")
    except Exception as exc:
        logger.warning("tables_layout to_markdown failed (non-fatal): %s", exc)

    inventory_pages = _inventory_pages_from_ranges(page_ranges, total_pages)
    logger.info(
        "tables_layout locate done: tables=%s total_pages=%s",
        len(anchors),
        total_pages,
    )
    return TableLocationResult(
        anchors=anchors,
        text_content=text_content,
        total_pages=total_pages,
        inventory_pages=inventory_pages,
    )
