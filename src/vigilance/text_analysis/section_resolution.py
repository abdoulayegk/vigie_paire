"""Resolution des sections texte cibles et audits par PDF."""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)

from vigilance.cli.quarter_logic import normalize_quarter
from vigilance.extraction.section_locator import locate_sections_in_pdf
from vigilance.extraction.section_taxonomy import canonicalize_section

from .constants import _CANONICAL_TO_TEXT_KEY, _SECTION_LABELS, _T4_TEXT_TARGET_SECTIONS, _TARGET_SECTIONS_BY_BANK
from .models import ResolvedSection, SectionAudit
from .pdf_block_classification import _build_section_audit, _repeated_text_counts
from .pdf_block_extraction import _extract_docling_page_blocks

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
        )
    return sections


def _extract_audits_for_pdf(
    *,
    pdf_path: Path,
    sections: dict[str, ResolvedSection],
) -> list[SectionAudit]:
    """Extrait les blocs narratifs de chaque section via Docling + heuristiques.

    Cette etape construit les audits qui seront convertis en markdown
    ``source of truth``. Les appels GPT de comparaison/triage relisent ensuite
    exclusivement ce markdown, pas les PDFs directement.
    """
    if not sections:
        return []
    section_order = _next_section_by_key(sections)
    unique_pages = sorted({page for section in sections.values() for page in section.pages})
    page_blocks, table_bboxes_by_page, footnote_bboxes_by_page = _extract_docling_page_blocks(pdf_path, unique_pages)
    repeated_counts = _repeated_text_counts(page_blocks)
    audits: list[SectionAudit] = []
    for section_key, section in sections.items():
        audit = _build_section_audit(
            section=section,
            next_section=section_order.get(section_key),
            page_blocks=page_blocks,
            repeated_text_counts=repeated_counts,
            table_bboxes_by_page=table_bboxes_by_page,
            footnote_bboxes_by_page=footnote_bboxes_by_page,
        )
        audits.append(audit)
    return audits
