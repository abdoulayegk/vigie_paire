from __future__ import annotations

import pytest

from vigie.extraction.docling import DoclingProcessor


def test_extract_document_rejects_none_pdf_path_before_path_conversion() -> None:
    processor = DoclingProcessor()

    with pytest.raises(ValueError, match="Chemin PDF requis pour l'extraction"):
        processor.extract_document(
            pdf_path=None,
            bank_code="bnc",
            quarter="t1",
            year=2026,
            use_vision_extraction=False,
        )
