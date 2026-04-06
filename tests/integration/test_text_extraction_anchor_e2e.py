"""Integration test for text extraction anchored on real section titles."""

from __future__ import annotations

from pathlib import Path

from vigilance.cli.run_text_extract import main
from vigilance.text_extraction.text_extraction_writer import load_text_extraction


def test_text_extraction_starts_at_section_title_for_bnc_t2(tmp_path: Path) -> None:
    """Run text extraction on a real BNC PDF and assert section output starts at the title."""
    pdf_path = Path("data/bnc/t2-rapport-actionnaire-2025.pdf")
    assert pdf_path.exists(), f"Missing fixture PDF: {pdf_path}"

    out_root = tmp_path / "text_anchor_e2e"
    exit_code = main(
        [
            "--bank",
            "bnc",
            "--year",
            "2025",
            "--T2",
            "--pdf",
            str(pdf_path),
            "--out-root",
            str(out_root),
        ]
    )

    assert exit_code == 0

    extraction_path = out_root / "bnc" / "2025" / "t2" / "text_extraction.json"
    payload = load_text_extraction(extraction_path)
    blocks = payload["blocks"]

    capital_blocks = [b for b in blocks if b["section"] == "gestion_capital"]
    risk_blocks = [b for b in blocks if b["section"] == "gestion_risques"]

    assert capital_blocks
    assert risk_blocks

    assert capital_blocks[0]["page"] == 25
    assert capital_blocks[0]["text"] == "Gestion du capital"
    assert capital_blocks[1]["page"] == 25
    assert capital_blocks[1]["text"].startswith("La Gestion du capital assume le double rôle")

    assert risk_blocks[0]["page"] == 31
    assert risk_blocks[0]["text"] == "Gestion des risques"
    assert risk_blocks[1]["page"] == 31
    assert risk_blocks[1]["text"].startswith("Le risque de crédit représente la possibilité")
