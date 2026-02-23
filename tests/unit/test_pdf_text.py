"""Tests for vigilance.extract.pdf_text."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from vigilance.extract.pdf_text import extract_page_text, extract_pages_text


@pytest.fixture()
def simple_pdf(tmp_path: Path) -> Path:
    """Create a 1-page PDF containing 'Bonjour'."""
    pdf_path = tmp_path / "simple.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 700, "Bonjour")
    c.showPage()
    c.save()
    return pdf_path


def test_extract_page_text(simple_pdf: Path) -> None:
    text = extract_page_text(str(simple_pdf), 0)
    assert "Bonjour" in text


def test_extract_pages_text(simple_pdf: Path) -> None:
    result = extract_pages_text(str(simple_pdf), [0])
    assert 0 in result
    assert "Bonjour" in result[0]
