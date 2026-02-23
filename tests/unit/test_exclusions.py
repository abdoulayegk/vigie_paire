"""Tests for vigilance.extract.exclusions."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from vigilance.extract.exclusions import compile_exclusion_patterns, get_skipped_pages


@pytest.fixture()
def three_page_pdf(tmp_path: Path) -> Path:
    """Create a 3-page PDF where page 1 (0-based) contains exclusion keywords."""
    pdf_path = tmp_path / "three_pages.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 0 — normal text
    c.drawString(100, 700, "Rapport trimestriel - contenu normal")
    c.showPage()

    # Page 1 — contains EDTF + Index des recommandations
    c.drawString(100, 700, "Conformite EDTF et reglementation")
    c.drawString(100, 680, "Index des recommandations du pilier 3")
    c.showPage()

    # Page 2 — normal text
    c.drawString(100, 700, "Bilan consolide du trimestre")
    c.showPage()

    c.save()
    return pdf_path


@pytest.fixture()
def exclusion_cfg() -> dict:
    """Minimal exclusion config requiring 2 pattern hits to skip a page."""
    return {
        "exclusions": {
            "block_title_patterns": [
                r"\bedtf\b",
                r"index\s+des\s+recommandations",
            ],
            "page_skip_rules": {"min_hits_to_skip": 2},
        },
    }


def test_compile_exclusion_patterns(exclusion_cfg: dict) -> None:
    patterns = compile_exclusion_patterns(exclusion_cfg)
    assert len(patterns) == 2


def test_get_skipped_pages(three_page_pdf: Path, exclusion_cfg: dict) -> None:
    skipped = get_skipped_pages(str(three_page_pdf), exclusion_cfg)
    assert skipped == [1]


def test_no_skip_when_threshold_high(three_page_pdf: Path, exclusion_cfg: dict) -> None:
    """If threshold is higher than possible matches, nothing is skipped."""
    exclusion_cfg["exclusions"]["page_skip_rules"]["min_hits_to_skip"] = 99
    skipped = get_skipped_pages(str(three_page_pdf), exclusion_cfg)
    assert skipped == []
