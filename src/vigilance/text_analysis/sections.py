"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from pathlib import Path

from vigilance.cli.quarter_logic import normalize_quarter
from vigilance.extraction.section_locator import locate_sections_in_pdf
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.text_analysis.constants import (
    _CANONICAL_TO_TEXT_KEY,
    _SECTION_LABELS,
    _T4_TEXT_TARGET_SECTIONS,
    _TARGET_SECTIONS_BY_BANK,
)
from vigilance.text_analysis.models import ResolvedSection


def _sorted_sections(sections: dict[str, ResolvedSection]) -> list[ResolvedSection]:
    """Retourne les sections triées par ordre d'apparition dans le PDF.

    Tri par page de début, puis par position verticale de l'ancre, puis par clé
    de section pour garantir un ordre stable lors des itérations.
    """
    return sorted(
        sections.values(),
        key=lambda sec: (
            sec.start_page,
            float(sec.anchor_bbox_norm[1]) if sec.anchor_bbox_norm else 0.0,
            sec.section_key,
        ),
    )


def _next_section_by_key(sections: dict[str, ResolvedSection]) -> dict[str, ResolvedSection | None]:
    """Construit un mapping section_key → section suivante dans le PDF.

    Utilisé pour délimiter la fenêtre basse d'une section quand deux sections
    partagent la même page de fin/début.
    """
    ordered = _sorted_sections(sections)
    next_map: dict[str, ResolvedSection | None] = {section.section_key: None for section in ordered}
    for current, nxt in zip(ordered, ordered[1:]):
        next_map[current.section_key] = nxt
    return next_map


def _section_window_for_page(
    section: ResolvedSection,
    page_number: int,
    next_section: ResolvedSection | None = None,
) -> tuple[float, float]:
    """Calcule la fenêtre verticale (top, bottom) d'une section sur une page donnée.

    Sur la première page de la section, la fenêtre commence sous l'ancre pour
    éviter d'inclure le titre de section lui-même. Sur la dernière page partagée
    avec la section suivante, la fenêtre se ferme au-dessus de l'ancre suivante.
    Les coordonnées sont normalisées dans [0, 1].

    Returns:
        Tuple ``(top, bottom)`` délimitant la zone de la section sur cette page.
    """
    top = 0.0
    bottom = 1.0
    if page_number == section.start_page and section.anchor_page == section.start_page and section.anchor_bbox_norm:
        top = max(top, float(section.anchor_bbox_norm[3]))
    if (
        next_section is not None
        and page_number == section.end_page == next_section.start_page
        and next_section.anchor_page == next_section.start_page
        and next_section.anchor_bbox_norm
    ):
        bottom = min(bottom, float(next_section.anchor_bbox_norm[1]))
    elif (
        page_number == section.end_page
        and section.end_anchor_page == page_number
        and section.end_anchor_bbox_norm
    ):
        bottom = min(bottom, float(section.end_anchor_bbox_norm[1]))
    if bottom <= top:
        return top, 1.0
    return top, bottom


def _allowed_target_sections(bank_code: str) -> set[str]:
    """Retourne l'ensemble des clés de sections autorisées pour une banque donnée.

    Si la banque n'a pas de configuration spécifique dans ``_TARGET_SECTIONS_BY_BANK``,
    toutes les sections du catalogue ``_SECTION_LABELS`` sont autorisées.
    """
    return set(_TARGET_SECTIONS_BY_BANK.get(str(bank_code or "").strip().lower(), set(_SECTION_LABELS)))


def _resolve_sections(
    pdf_path: Path,
    bank_code: str,
    *,
    quarter: str | None = None,
    year: int | None = None,
) -> dict[str, ResolvedSection]:
    """Localise et normalise les sections textuelles cibles d'un PDF.

    Le resultat est deja filtre selon les sections autorisees pour la banque,
    ce qui stabilise le flux de comparaison inter-trimestrielle.
    """
    mapping = locate_sections_in_pdf(
        str(pdf_path),
        bank_code=bank_code.lower(),
        quarter=quarter,
        year=year or 2025,
    )
    allowed_sections = _allowed_target_sections(bank_code)
    if quarter is not None and normalize_quarter(quarter) == "t4":
        allowed_sections &= _T4_TEXT_TARGET_SECTIONS
    sections: dict[str, ResolvedSection] = {}
    for item in getattr(mapping, "sections", []) or []:
        canonical = canonicalize_section(getattr(item, "section_type", ""))
        section_key = _CANONICAL_TO_TEXT_KEY.get(canonical)
        if not section_key or section_key not in allowed_sections or section_key in sections:
            continue
        start_page = int(getattr(item, "start_page", 0) or 0)
        end_page = int(getattr(item, "end_page", 0) or 0)
        if start_page <= 0 or end_page < start_page:
            continue
        sections[section_key] = ResolvedSection(
            section_key=section_key,
            title=_SECTION_LABELS[section_key],
            start_page=start_page,
            end_page=end_page,
            anchor_page=int(getattr(item, "anchor_page", 0) or 0) or None,
            anchor_text=str(getattr(item, "anchor_text", "") or "") or None,
            anchor_bbox_norm=list(getattr(item, "anchor_bbox_norm", []) or []) or None,
            end_anchor_page=int(getattr(item, "end_anchor_page", 0) or 0) or None,
            end_anchor_text=str(getattr(item, "end_anchor_text", "") or "") or None,
            end_anchor_bbox_norm=list(getattr(item, "end_anchor_bbox_norm", []) or []) or None,
        )
    return sections
