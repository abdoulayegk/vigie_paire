"""Constat de fiabilité des deux sections cibles.

Remplace l'ancien déclencheur du repli GenAI, dont les deux critères étaient
faux : il comptait le nombre d'entrées (une section réglementaire pouvait
masquer une section risques absente) et moyennait les confiances (un override
manuel à 1.0 pouvait dissimuler un scan de titres au plancher).
"""

from __future__ import annotations

from vigie.extraction.localisation_sections.models import LocatedSection
from vigie.extraction.localisation_sections.validation import assess_target_section_health


def _section(section_type: str, confidence: float) -> LocatedSection:
    return LocatedSection(
        section_type=section_type,
        title_found=section_type,
        start_page=10,
        end_page=20,
        confidence=confidence,
        detection_method="test",
    )


def test_two_target_sections_with_solid_confidence_are_complete() -> None:
    health = assess_target_section_health([_section("gestion_capital", 0.95), _section("gestion_risques", 0.9)])

    assert health["status"] == "complete"
    assert health["missing"] == []
    assert health["min_confidence"] == 0.9


def test_regulatory_section_does_not_hide_a_missing_target() -> None:
    """Deux sections détectées, mais la section risques manque.

    L'ancien critère ``len(sections) < 2`` ne se déclenchait pas ici.
    """
    health = assess_target_section_health([_section("gestion_reglementation", 0.95), _section("gestion_capital", 0.95)])

    assert health["status"] == "missing_target_section"
    assert health["missing"] == ["risk_management"]
    assert health["located"] == ["capital_management"]


def test_strong_section_does_not_mask_a_weak_one() -> None:
    """Override manuel à 1.0 + scan de titres à 0.5.

    L'ancienne moyenne valait 0.75 et ne déclenchait rien; le minimum expose
    la section faible.
    """
    health = assess_target_section_health([_section("gestion_capital", 1.0), _section("gestion_risques", 0.5)])

    assert health["status"] == "low_confidence"
    assert health["min_confidence"] == 0.5
    assert health["confidence_by_concept"]["risk_management"] == 0.5


def test_no_section_at_all_is_reported_as_missing() -> None:
    health = assess_target_section_health([])

    assert health["status"] == "missing_target_section"
    assert health["missing"] == ["capital_management", "risk_management"]
    assert health["min_confidence"] == 0.0
