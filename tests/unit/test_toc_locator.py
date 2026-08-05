"""Tests de la localisation TDM structurelle T4."""

from __future__ import annotations

from vigie.extraction.localisation_sections.toc_locator import (
    apply_offset_to_entries,
    find_management_report_opening_page,
    is_edtf_decoy_page,
    locate_toc_structure,
    parse_toc_entries_from_text,
    resolve_printed_to_physical_offset,
    TocStructureEntry,
)


def _edtf_page_text() -> str:
    lines = [f"{i}. Divulgation item numero {i} ........ {10 + i}" for i in range(1, 33)]
    return "\n".join(lines)


def _rg_page_text() -> str:
    return "\n".join(
        [
            "Rapport de gestion",
            "Gestion du risque ........................ 74",
            "Gestion des fonds propres ............... 125",
            "Questions en matiere de comptabilite .... 138",
            "Introduction au rapport",
        ]
    )


def test_is_edtf_decoy_page_detects_numbered_table() -> None:
    assert is_edtf_decoy_page(_edtf_page_text()) is True
    assert is_edtf_decoy_page(_rg_page_text()) is False


def test_find_rg_opening_rejects_edtf() -> None:
    text_by_page = {
        16: _edtf_page_text(),
        17: _rg_page_text(),
        18: "Autre contenu narratif sans titre RG",
    }
    rg_page, rejected = find_management_report_opening_page(text_by_page)
    assert rg_page == 17
    assert 16 in rejected


def test_collapse_letter_spaced_gestion_du_risque() -> None:
    from vigie.extraction.localisation_sections.toc_locator import collapse_letter_spaced_text

    collapsed = collapse_letter_spaced_text("Ge s t i o n d u ri s q u e 7 2")
    assert "gestion" in collapsed.lower()
    assert "risque" in collapsed.lower()
    assert collapsed.strip().endswith("72")
    spaced_entries = parse_toc_entries_from_text("Ge s t i o n d u ri s q u e 7 2")
    assert any(e.printed_page == 72 and "risque" in e.title.lower() for e in spaced_entries)


def test_parse_toc_entries_from_text() -> None:
    entries = parse_toc_entries_from_text(_rg_page_text())
    assert len(entries) >= 3
    titles = {e.title.lower() for e in entries}
    assert any("risque" in t for t in titles)
    assert any("fonds propres" in t for t in titles)
    assert any(e.printed_page == 125 for e in entries)


def test_resolve_offset_from_physical_title_anchor() -> None:
    text_by_page = {
        17: _rg_page_text(),
        127: "Gestion des fonds propres\nNous gerons activement nos fonds propres.",
    }
    entries = [
        TocStructureEntry(title="Gestion des fonds propres", printed_page=125),
        TocStructureEntry(title="Gestion du risque", printed_page=74),
    ]
    offset, anomalies = resolve_printed_to_physical_offset(text_by_page, entries, scan_from=100)
    assert offset == 2
    applied = apply_offset_to_entries(entries, offset)
    assert applied[0].physical_page == 127


def test_locate_toc_structure_end_to_end_text_only() -> None:
    text_by_page = {
        16: _edtf_page_text(),
        17: _rg_page_text(),
        127: "Gestion des fonds propres\nTexte capital",
        76: "Gestion du risque\nTexte risques",
    }
    toc = locate_toc_structure(text_by_page, configured_offset=0)
    assert toc.rg_page == 17
    assert 16 in toc.rejected_edtf_pages
    assert toc.confidence >= 0.55
    assert len(toc.entries) >= 2
