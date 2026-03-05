"""Unit tests for SectionLocator page_number_offset (document vs physical)."""

from __future__ import annotations

from vigilance.extraction.section_locator import (
    LocatedSection,
    SectionLocator,
)


def test_uses_document_page_numbers_toc() -> None:
    """toc et variantes (toc_*) sont en numerotation document -> offset applique."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("toc") is True
    assert locator._uses_document_page_numbers("toc_corrected") is True
    assert locator._uses_document_page_numbers("toc_cibc_recalibrated") is True


def test_uses_document_page_numbers_manual_override() -> None:
    """manual_override et manual_override_guardrail sont en numerotation document."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("manual_override") is True
    assert locator._uses_document_page_numbers("manual_override_guardrail") is True


def test_uses_document_page_numbers_scan_visual_genai_false() -> None:
    """scan, genai_fallback, visual donnent deja des numeros physiques -> pas d'offset."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("scan") is False
    assert locator._uses_document_page_numbers("scan_subsection") is False
    assert locator._uses_document_page_numbers("genai_fallback") is False
    assert locator._uses_document_page_numbers("visual") is False
    assert locator._uses_document_page_numbers("visual_temp") is False


def test_uses_document_page_numbers_empty_false() -> None:
    """detection_method vide -> pas d'offset."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._uses_document_page_numbers("") is False


def test_cibc_page_number_offset_from_config() -> None:
    """CIBC a page_number_offset=3 dans bank_profiles."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    assert locator._get_page_number_offset() == 3


def test_offset_applied_only_to_document_sections() -> None:
    """Apres etape 4.5: section toc -> start_page augmente de l'offset; section scan -> inchange."""
    locator = SectionLocator(bank_code="cibc", quarter="t1", year=2025)
    offset = locator._get_page_number_offset()
    assert offset == 3

    section_toc = LocatedSection(
        section_type="gestion_capital",
        title_found="Gestion des fonds propres",
        start_page=20,
        end_page=24,
        detection_method="toc",
    )
    section_scan = LocatedSection(
        section_type="gestion_risques",
        title_found="Gestion du risque",
        start_page=25,
        end_page=45,
        detection_method="scan",
    )

    # Logique identique a l'etape 4.5: appliquer offset uniquement si document
    adjusted = []
    for s in [section_toc, section_scan]:
        if locator._uses_document_page_numbers(s.detection_method):
            new_start = s.start_page + offset
            new_end = (s.end_page + offset) if s.end_page is not None else None
            adjusted.append(
                LocatedSection(
                    section_type=s.section_type,
                    title_found=s.title_found,
                    start_page=new_start,
                    end_page=new_end,
                    confidence=s.confidence,
                    detection_method=s.detection_method,
                    end_detection_method=s.end_detection_method,
                    detected_span=s.detected_span,
                    final_span=s.final_span,
                    constraint_applied=s.constraint_applied,
                    constraint_reason=s.constraint_reason,
                )
            )
        else:
            adjusted.append(s)

    toc_result = next(s for s in adjusted if s.detection_method == "toc")
    scan_result = next(s for s in adjusted if s.detection_method == "scan")

    assert toc_result.start_page == 23, "toc: document 20 + offset 3 = physique 23"
    assert toc_result.end_page == 27, "toc: document 24 + offset 3 = physique 27"
    assert scan_result.start_page == 25, "scan: deja physique, pas d'offset"
    assert scan_result.end_page == 45, "scan: deja physique, pas d'offset"
