from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

from vigilance.cli.run_text_extract import _get_section_ranges_from_locator
from vigilance.extraction.section_locator import (
    LocatedSection,
    SectionLocator,
    VisualTextElement,
)
from vigilance.text_extraction.text_extractor import (
    TextExtractor,
    _extract_text_items,
)


class _FakeBBox:
    def __init__(self, bbox: list[float]) -> None:
        self._bbox = bbox

    def to_top_left_origin(self, page_height: float) -> "_FakeBBox":
        return self

    def normalized(self, size: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            l=self._bbox[0],
            t=self._bbox[1],
            r=self._bbox[2],
            b=self._bbox[3],
        )


class _FakeProv:
    def __init__(self, page_no: int, bbox: list[float]) -> None:
        self.page_no = page_no
        self.bbox = _FakeBBox(bbox)


class _FakeLabel:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeTextItem:
    def __init__(self, text: str, page_no: int, bbox: list[float], label: str) -> None:
        self.text = text
        self.prov = [_FakeProv(page_no=page_no, bbox=bbox)]
        self.label = _FakeLabel(label)


class _FakeDoc:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, 0


def _page_objects() -> dict[int, SimpleNamespace]:
    return {
        10: SimpleNamespace(size=SimpleNamespace(height=1000)),
        11: SimpleNamespace(size=SimpleNamespace(height=1000)),
    }


def test_resolve_section_anchor_prefers_exact_title_found_over_alias() -> None:
    locator = SectionLocator(bank_code="bnc", quarter="t1", year=2025)
    section = LocatedSection(
        section_type="gestion_risques",
        title_found="Gestion des risques",
        start_page=42,
    )
    visual_elements = {
        42: [
            VisualTextElement(
                text="Risk management",
                page=42,
                x0=100,
                y0=60,
                x1=700,
                y1=95,
                font_size=18,
                is_bold=True,
                line_number=1,
                page_width=1000,
                page_height=1000,
            ),
            VisualTextElement(
                text="Gestion des risques",
                page=42,
                x0=120,
                y0=210,
                x1=760,
                y1=250,
                font_size=17,
                is_bold=True,
                line_number=4,
                page_width=1000,
                page_height=1000,
            ),
        ]
    }

    resolved = locator._resolve_section_anchor(section, visual_elements)

    assert resolved.anchor_found is True
    assert resolved.anchor_page == 42
    assert resolved.anchor_text == "Gestion des risques"
    assert resolved.anchor_bbox_norm == [0.12, 0.21, 0.76, 0.25]


def test_resolve_section_anchor_uses_alias_and_highest_header_candidate() -> None:
    locator = SectionLocator(bank_code="bnc", quarter="t1", year=2025)
    section = LocatedSection(
        section_type="gestion_reglementation",
        title_found="Contexte réglementaire et perspectives",
        start_page=18,
    )
    visual_elements = {
        18: [
            VisualTextElement(
                text="Réglementation",
                page=18,
                x0=90,
                y0=260,
                x1=420,
                y1=295,
                font_size=15,
                is_bold=True,
                line_number=7,
                page_width=1000,
                page_height=1000,
            ),
            VisualTextElement(
                text="Réglementation",
                page=18,
                x0=80,
                y0=120,
                x1=430,
                y1=155,
                font_size=18,
                is_bold=True,
                line_number=2,
                page_width=1000,
                page_height=1000,
            ),
        ]
    }

    resolved = locator._resolve_section_anchor(section, visual_elements)

    assert resolved.anchor_found is True
    assert resolved.anchor_text == "Réglementation"
    assert resolved.anchor_bbox_norm == [0.08, 0.12, 0.43, 0.155]


def test_get_section_ranges_from_locator_propagates_anchor_metadata(
    monkeypatch,
) -> None:
    locator_module = types.ModuleType("vigilance.extraction.section_locator")

    def fake_locate_sections_in_pdf(
        pdf_path: Path,
        bank_code: str | None = None,
        quarter: str | None = None,
        year: int = 2025,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            sections=[
                SimpleNamespace(
                    section_type="gestion_risques",
                    start_page=12,
                    end_page=14,
                    anchor_page=12,
                    anchor_text="Gestion des risques",
                    anchor_bbox_norm=[0.1, 0.2, 0.7, 0.24],
                    anchor_found=True,
                )
            ]
        )

    locator_module.locate_sections_in_pdf = fake_locate_sections_in_pdf
    monkeypatch.setitem(sys.modules, "vigilance.extraction.section_locator", locator_module)

    ranges = _get_section_ranges_from_locator(
        pdf_path=Path("dummy.pdf"),
        bank_code="bnc",
        year=2025,
        quarter="t1",
    )

    assert ranges == [{
        "section": "gestion_risques",
        "start": 12,
        "end": 14,
        "anchor_page": 12,
        "anchor_text": "Gestion des risques",
        "anchor_bbox_norm": [0.1, 0.2, 0.7, 0.24],
        "anchor_found": True,
    }]


def test_extract_text_items_keeps_title_and_blocks_after_anchor_only() -> None:
    doc = _FakeDoc([
        _FakeTextItem(
            "Texte de la section precedente qui doit etre elimine completement.",
            page_no=10,
            bbox=[0.1, 0.08, 0.9, 0.14],
            label="paragraph",
        ),
        _FakeTextItem(
            "Gestion des risques",
            page_no=10,
            bbox=[0.1, 0.30, 0.8, 0.34],
            label="heading",
        ),
        _FakeTextItem(
            "Le risque de credit est gere via des limites consolidees detaillees.",
            page_no=10,
            bbox=[0.1, 0.36, 0.92, 0.44],
            label="paragraph",
        ),
        _FakeTextItem(
            "Le risque de liquidite est surveille sur les pages suivantes de la section.",
            page_no=11,
            bbox=[0.1, 0.05, 0.9, 0.11],
            label="paragraph",
        ),
    ])

    section_ranges = [{
        "section": "gestion_risques",
        "start": 10,
        "end": 11,
        "anchor_page": 10,
        "anchor_text": "Gestion des risques",
        "anchor_bbox_norm": [0.1, 0.30, 0.8, 0.34],
        "anchor_found": True,
    }]

    items = _extract_text_items(
        doc=doc,
        page_objects=_page_objects(),
        table_zones={},
        section_ranges=section_ranges,
    )

    assert items == [
        ("gestion_risques", 10, "Gestion des risques"),
        (
            "gestion_risques",
            10,
            "Le risque de credit est gere via des limites consolidees detaillees.",
        ),
        (
            "gestion_risques",
            11,
            "Le risque de liquidite est surveille sur les pages suivantes de la section.",
        ),
    ]


def test_extract_text_items_keeps_anchor_but_filters_text_near_tables() -> None:
    doc = _FakeDoc([
        _FakeTextItem(
            "Gestion du capital",
            page_no=10,
            bbox=[0.1, 0.20, 0.8, 0.24],
            label="heading",
        ),
        _FakeTextItem(
            "Le ratio CET1 est presente dans ce paragraphe narratif proche du tableau.",
            page_no=10,
            bbox=[0.1, 0.36, 0.92, 0.44],
            label="paragraph",
        ),
    ])

    section_ranges = [{
        "section": "gestion_capital",
        "start": 10,
        "end": 10,
        "anchor_page": 10,
        "anchor_text": "Gestion du capital",
        "anchor_bbox_norm": [0.1, 0.20, 0.8, 0.24],
        "anchor_found": True,
    }]

    items = _extract_text_items(
        doc=doc,
        page_objects=_page_objects(),
        table_zones={10: [[0.0, 0.34, 1.0, 0.70]]},
        section_ranges=section_ranges,
    )

    assert items == [("gestion_capital", 10, "Gestion du capital")]


def test_extract_text_blocks_skips_sections_without_anchor() -> None:
    extractor = TextExtractor()
    extractor._initialized = True

    class _UnusedConverter:
        def convert(self, *args, **kwargs):
            raise AssertionError("convert should not be called when no section anchor is valid")

    extractor._converter = _UnusedConverter()

    blocks = extractor.extract_text_blocks(
        pdf_path=Path(__file__),
        bank_code="bnc",
        quarter="t1",
        year=2025,
        section_ranges=[{
            "section": "gestion_risques",
            "start": 10,
            "end": 11,
            "anchor_page": None,
            "anchor_text": None,
            "anchor_bbox_norm": None,
            "anchor_found": False,
        }],
    )

    assert blocks == []


def test_extract_text_items_splits_overlapping_sections_by_anchor_height() -> None:
    doc = _FakeDoc([
        _FakeTextItem(
            "Autres faits nouveaux en matière de réglementation",
            page_no=43,
            bbox=[0.1, 0.04, 0.8, 0.08],
            label="heading",
        ),
        _FakeTextItem(
            "Nous continuons de surveiller l'évolution de la réglementation de près.",
            page_no=43,
            bbox=[0.1, 0.10, 0.92, 0.16],
            label="paragraph",
        ),
        _FakeTextItem(
            "Gestion des risques",
            page_no=43,
            bbox=[0.1, 0.27, 0.7, 0.30],
            label="heading",
        ),
        _FakeTextItem(
            "Le risque de crédit demeure le principal risque supporté par la banque.",
            page_no=43,
            bbox=[0.1, 0.32, 0.92, 0.38],
            label="paragraph",
        ),
    ])

    section_ranges = [
        {
            "section": "gestion_reglementation",
            "start": 43,
            "end": 43,
            "anchor_page": 43,
            "anchor_text": "Autres faits nouveaux en matière de réglementation",
            "anchor_bbox_norm": [0.1, 0.04, 0.8, 0.08],
            "anchor_found": True,
        },
        {
            "section": "gestion_risques",
            "start": 43,
            "end": 55,
            "anchor_page": 43,
            "anchor_text": "Gestion des risques",
            "anchor_bbox_norm": [0.1, 0.27, 0.7, 0.30],
            "anchor_found": True,
        },
    ]

    items = _extract_text_items(
        doc=doc,
        page_objects={43: SimpleNamespace(size=SimpleNamespace(height=1000))},
        table_zones={},
        section_ranges=section_ranges,
    )

    assert items == [
        (
            "gestion_reglementation",
            43,
            "Autres faits nouveaux en matière de réglementation",
        ),
        (
            "gestion_reglementation",
            43,
            "Nous continuons de surveiller l'évolution de la réglementation de près.",
        ),
        ("gestion_risques", 43, "Gestion des risques"),
        (
            "gestion_risques",
            43,
            "Le risque de crédit demeure le principal risque supporté par la banque.",
        ),
    ]
