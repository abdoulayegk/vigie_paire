"""Tests de contrat des integrations PyMuPDF sur un vrai document PDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from vigie.extraction.pdf_preview import (
    create_thumbnail,
    extract_text_from_pages,
    get_pdf_info,
    render_pdf_page,
    render_pdf_pages,
)
from vigie.analyse_texte.extraction import _extract_pymupdf_fallback_blocks
from vigie.support.utils.pdf_crop import (
    crop_footnote_region_to_bytes,
    crop_page_region_bytes,
    crop_table_image,
    crop_table_region_to_bytes,
    render_page_with_bbox_highlight_to_bytes,
)
from vigie.support.utils.pdf_highlight import find_text_bboxes_in_region

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Construire un PDF minimal contenant texte, tableau et note."""
    pdf_path = tmp_path / "pymupdf-contract.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.insert_text((30, 55), "Revenue 2026", fontsize=14)
        page.draw_rect(pymupdf.Rect(25, 80, 275, 250), color=(0, 0, 0))
        page.insert_text((40, 120), "Operating income 125", fontsize=11)
        page.insert_text((30, 300), "Footnote detail", fontsize=10)
        document.save(pdf_path)
    return pdf_path


def _assert_png(raw: bytes | None) -> None:
    assert raw is not None
    assert raw.startswith(_PNG_SIGNATURE)


def test_preview_and_text_contract(sample_pdf: Path) -> None:
    """Verifier ouverture, texte, rendu multi-page et miniature."""
    info = get_pdf_info(sample_pdf)
    assert info["available"] is True
    assert info["total_pages"] == 1

    _assert_png(render_pdf_page(sample_pdf, 1, scale=1.0))
    _assert_png(create_thumbnail(sample_pdf, 1, width=120))

    previews = render_pdf_pages(sample_pdf, 1, 1, scale=1.0)
    assert len(previews) == 1
    assert previews[0].width == 300
    assert previews[0].height == 400
    assert "Revenue 2026" in previews[0].text_content

    extracted = extract_text_from_pages(sample_pdf, 1, 1)
    assert "--- Page 1 ---" in extracted
    assert "Operating income 125" in extracted


def test_crop_render_and_highlight_contract(sample_pdf: Path, tmp_path: Path) -> None:
    """Verifier les crops utilises par Vision et les preuves Dash."""
    table_bbox = [0.08, 0.18, 0.92, 0.65]
    footnote_bbox = [0.08, 0.18, 0.92, 0.70]
    highlight = [[0.10, 0.25, 0.80, 0.34]]

    cropped = crop_table_region_to_bytes(
        str(sample_pdf),
        1,
        table_bbox,
        scale=1.0,
        highlight_rects=highlight,
    )
    _assert_png(cropped)

    full = render_page_with_bbox_highlight_to_bytes(
        str(sample_pdf), 1, table_bbox, scale=1.0
    )
    _assert_png(full)

    footnote = crop_footnote_region_to_bytes(
        str(sample_pdf), 1, footnote_bbox, scale=1.0
    )
    _assert_png(footnote)

    region = crop_page_region_bytes(
        str(sample_pdf), 1, bbox_norm=table_bbox, dpi=72
    )
    _assert_png(region)

    output_path = tmp_path / "crop.png"
    assert crop_table_image(
        str(sample_pdf), 1, table_bbox, str(output_path), dpi=72
    )
    _assert_png(output_path.read_bytes())

    matches = find_text_bboxes_in_region(
        str(sample_pdf), 1, "Operating income 125", [0.0, 0.0, 1.0, 1.0]
    )
    assert len(matches) == 1
    assert all(0.0 <= coordinate <= 1.0 for coordinate in matches[0])


def test_fallback_blocks_contract(sample_pdf: Path) -> None:
    """Verifier le fallback du pipeline texte."""
    blocks_by_page = _extract_pymupdf_fallback_blocks(sample_pdf, [1, 2])
    assert blocks_by_page[2] == []
    assert any("Revenue 2026" in block.text for block in blocks_by_page[1])
    assert all(
        block.source_label == "pymupdf_fallback"
        and all(0.0 <= coordinate <= 1.0 for coordinate in block.bbox_norm)
        for block in blocks_by_page[1]
    )
