"""Tests du selecteur de moteur de localisation de tableaux."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vigie.extraction.table_locator import (
    ENGINE_DOCLING,
    ENGINE_PYMUPDF_LAYOUT,
    TableAnchor,
    get_table_locator,
    resolve_table_locator_engine,
)
from vigie.extraction.tables_layout.table_bbox import normalize_pymupdf_bbox, page_number_from_layout


def test_resolve_default_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans env, le defaut est pymupdf_layout."""
    monkeypatch.delenv("TABLE_LOCATOR_ENGINE", raising=False)
    assert resolve_table_locator_engine() == ENGINE_PYMUPDF_LAYOUT


def test_resolve_docling_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """TABLE_LOCATOR_ENGINE=docling selectionne Docling."""
    monkeypatch.setenv("TABLE_LOCATOR_ENGINE", "docling")
    assert resolve_table_locator_engine() == ENGINE_DOCLING


def test_resolve_invalid_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une valeur inconnue leve une erreur claire."""
    monkeypatch.setenv("TABLE_LOCATOR_ENGINE", "foo")
    with pytest.raises(ValueError, match="Unsupported TABLE_LOCATOR_ENGINE"):
        resolve_table_locator_engine()


def test_get_table_locator_pymupdf_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """pymupdf_layout n'instancie jamais DoclingTableLocator."""
    monkeypatch.setenv("TABLE_LOCATOR_ENGINE", "pymupdf_layout")
    with patch("vigie.extraction.docling.table_locator.DoclingTableLocator") as docling_cls:
        locator = get_table_locator()
        assert locator.__class__.__name__ == "TablesLayoutLocator"
        docling_cls.assert_not_called()


def test_get_table_locator_docling_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """docling n'instancie jamais TablesLayoutLocator."""
    monkeypatch.setenv("TABLE_LOCATOR_ENGINE", "docling")
    converter = MagicMock(name="converter")
    with patch("vigie.extraction.tables_layout.table_locator.TablesLayoutLocator") as pymu_cls:
        locator = get_table_locator(converter=converter)
        assert locator.__class__.__name__ == "DoclingTableLocator"
        pymu_cls.assert_not_called()


def test_no_fallback_when_tables_layout_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si tables_layout echoue, Docling n'est pas appele."""
    monkeypatch.setenv("TABLE_LOCATOR_ENGINE", "pymupdf_layout")
    locator = get_table_locator()
    with (
        patch(
            "vigie.extraction.tables_layout.tables_layout_pass.detect_table_anchors",
            side_effect=RuntimeError("layout boom"),
        ),
        patch("vigie.extraction.docling.table_locator.DoclingTableLocator") as docling_cls,
    ):
        with pytest.raises(RuntimeError, match="layout boom"):
            locator.locate(Path("/tmp/missing.pdf"))
        docling_cls.assert_not_called()


def test_normalize_pymupdf_bbox_matches_vision_contract() -> None:
    """Les bbox Layout sont normalisees [l,t,r,b] dans [0,1], origine haut-gauche."""
    bbox = normalize_pymupdf_bbox([61.2, 79.2, 550.8, 633.6], page_width=612.0, page_height=792.0)
    assert bbox == pytest.approx([0.1, 0.1, 0.9, 0.8])
    assert all(0.0 <= value <= 1.0 for value in bbox)
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]


def test_page_number_from_layout_json_is_already_one_based() -> None:
    """Le champ page_number du JSON Layout est deja 1-indexe."""
    assert page_number_from_layout(11) == 11


def test_page_number_zero_based_api_conversion() -> None:
    """L'API pages= de pymupdf4llm est 0-indexee et doit etre convertie."""
    assert page_number_from_layout(10, zero_based_input=True) == 11


def test_tables_layout_propagates_use_ocr_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """to_json / to_markdown recoivent use_ocr=False."""
    monkeypatch.setenv("TABLE_LOCATOR_ENGINE", "pymupdf_layout")
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    fake_json = {
        "pages": [
            {
                "page_number": 1,
                "width": 100.0,
                "height": 200.0,
                "boxes": [
                    {
                        "boxclass": "table",
                        "table": {
                            "bbox": [10.0, 20.0, 90.0, 180.0],
                            "markdown": "a | b | c\n1 | 2 | 3\nmore reference text here",
                        },
                    }
                ],
            }
        ]
    }
    captured: dict[str, object] = {}

    def fake_to_json(_path: str, **kwargs: object) -> dict:
        captured["to_json"] = kwargs
        return fake_json

    def fake_to_markdown(_path: str, **kwargs: object) -> str:
        captured["to_markdown"] = kwargs
        return "# ok"

    with (
        patch("vigie.extraction.tables_layout.table_locator.pymupdf4llm.to_json", side_effect=fake_to_json),
        patch(
            "vigie.extraction.tables_layout.table_locator.pymupdf4llm.to_markdown",
            side_effect=fake_to_markdown,
        ),
        patch("vigie.extraction.tables_layout.table_locator.pymupdf4llm.use_layout"),
        patch("vigie.extraction.tables_layout.table_locator.pymupdf.open") as open_pdf,
    ):
        doc = MagicMock()
        doc.__enter__.return_value = doc
        doc.__exit__.return_value = False
        doc.__len__.return_value = 1
        open_pdf.return_value = doc

        from vigie.extraction.tables_layout.table_locator import detect_table_anchors

        result = detect_table_anchors(pdf_path, page_ranges=[(1, 1)])

    assert captured["to_json"]["use_ocr"] is False
    assert captured["to_json"]["force_ocr"] is False
    assert captured["to_markdown"]["use_ocr"] is False
    assert len(result.anchors) == 1
    assert result.anchors[0].source == "tables_layout"
    assert result.anchors[0].page_number == 1
    assert result.anchors[0].bbox == pytest.approx([0.1, 0.1, 0.9, 0.9])


def test_anchors_to_vision_items_contract() -> None:
    """Les ancres se convertissent vers le tuple Vision historique."""
    from vigie.extraction.table_locator import anchors_to_vision_items

    anchors = [
        TableAnchor(
            table_id="tableau_0",
            page_number=3,
            bbox=[0.1, 0.2, 0.8, 0.9],
            reference_text="ref",
            source="tables_layout",
        )
    ]
    items = anchors_to_vision_items(anchors)
    assert items == [(0, 3, [0.1, 0.2, 0.8, 0.9], "tableau_0", "ref")]
