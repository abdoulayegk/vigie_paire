"""Tests unitaires pour les générateurs d'exportation modulaires (Excel, PDF)."""

from __future__ import annotations

from vigilance.export import (
    export_comparison_to_excel,
    export_summary_to_pdf,
)


def test_export_comparison_to_excel() -> None:
    path = export_comparison_to_excel({"bank": "rbc"}, output_path="test.xlsx")
    assert path == "test.xlsx"


def test_export_summary_to_pdf() -> None:
    path = export_summary_to_pdf({"bank": "td"}, output_path="test.pdf")
    assert path == "test.pdf"
