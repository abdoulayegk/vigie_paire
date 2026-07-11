"""Construction du markdown filtré à partir du markdown natif Docling."""

from __future__ import annotations

import re
from dataclasses import dataclass

from vigilance.text_analysis.constants import _SECTION_LABELS
from vigilance.text_analysis.markdown import (
    _format_heading_line,
    _is_docling_heading_block,
    _is_out_of_scope_accounting_heading,
)
from vigilance.text_analysis.models import PDFBlock, SectionAudit
from vigilance.text_analysis.normalization import (
    _looks_like_footnote,
    _looks_like_table_or_financial_grid,
    _normalized_block_text,
)

_SECTION_ORDER = {
    "gestion_capital": 0,
    "gestion_risques": 1,
    "gestion_reglementation": 2,
}

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_LIST_ITEM_RE = re.compile(r"^[-*]\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_FOOTNOTE_LINE_RE = re.compile(r"^\d+\s+\S")
_PAGE_HEADER_RE = re.compile(
    r"groupe banque td.*rapport de gestion\s+\d+",
    flags=re.IGNORECASE,
)
_END_BOUNDARY_HEADING_PATTERNS = [
    re.compile(r"normes\s+et\s+m[eé]thodes\s+comptables", re.IGNORECASE),
    re.compile(r"m[eé]thodes\s+et\s+estimations\s+comptables", re.IGNORECASE),
    re.compile(r"m[eé]thodes\s+comptables\s+significatives", re.IGNORECASE),
    re.compile(r"[eé]tats?\s+financiers?", re.IGNORECASE),
]


def _is_end_boundary_heading(segment: DoclingSegment, audits: list[SectionAudit]) -> bool:
    """Indique si un titre Docling marque la fin de la section courante."""
    if segment.kind != "heading":
        return False
    text = str(segment.text or "").strip()
    if not text:
        return False
    segment_norm = _normalized_block_text(text)
    for audit in audits:
        end_text = str(audit.end_anchor_text or "").strip()
        if not end_text:
            continue
        end_norm = _normalized_block_text(end_text)
        if segment_norm == end_norm or end_norm in segment_norm or segment_norm in end_norm:
            return True
    for pattern in _END_BOUNDARY_HEADING_PATTERNS:
        if pattern.search(text):
            return True
    return False


@dataclass(slots=True)
class DoclingSegment:
    """Segment parsé depuis le markdown natif Docling."""

    kind: str
    text: str
    heading_level: int = 0


def _parse_docling_markdown(md_content: str) -> list[DoclingSegment]:
    """Découpe le markdown Docling en segments ordonnés."""
    segments: list[DoclingSegment] = []
    in_table = False
    for raw_line in md_content.splitlines():
        line = raw_line.strip()
        if not line:
            in_table = False
            continue

        if _TABLE_ROW_RE.match(line):
            in_table = True
            continue
        if in_table:
            continue

        heading_match = _HEADING_LINE_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            if text:
                segments.append(DoclingSegment(kind="heading", text=text, heading_level=level))
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            text = list_match.group(1).strip()
            if text:
                segments.append(DoclingSegment(kind="list_item", text=text))
            continue

        segments.append(DoclingSegment(kind="paragraph", text=line))
    return segments


def _segment_matches_included_block(segment: DoclingSegment, audit: SectionAudit) -> bool:
    """Vérifie si un segment correspond à un bloc narratif inclus."""
    segment_norm = _normalized_block_text(segment.text)
    if not segment_norm:
        return False
    for block in audit.included_blocks:
        block_norm = _normalized_block_text(block.text)
        if not block_norm:
            continue
        if segment_norm == block_norm:
            return True
        if len(segment_norm) >= 20 and segment_norm in block_norm:
            return True
        if len(block_norm) >= 20 and block_norm in segment_norm:
            return True
    return False


def _segment_heading_allowed(segment: DoclingSegment, audit: SectionAudit) -> bool:
    """N'accepte un titre Docling que s'il n'est pas un artefact de tableau."""
    if _is_out_of_scope_accounting_heading(segment.text):
        return False
    segment_norm = _normalized_block_text(segment.text)
    if not segment_norm:
        return False
    for block in _audit_blocks(audit):
        block_norm = _normalized_block_text(block.text)
        if segment_norm != block_norm:
            continue
        if block.block_type == "table" or block.exclusion_reason == "table_like_block":
            return False
        return True
    return _segment_matches_audit(segment, audit)


def _should_keep_docling_segment(segment: DoclingSegment, audits: list[SectionAudit] | None = None) -> bool:
    """Filtre tableaux, notes et en-têtes de page du flux Docling."""
    if audits and any(_segment_matches_included_block(segment, audit) for audit in audits):
        return True

    text = str(segment.text or "").strip()
    if not text:
        return False
    if _PAGE_HEADER_RE.search(text):
        return False
    if re.search(r"rapport de gestion\s+\d+", text, flags=re.IGNORECASE):
        return False
    if segment.kind == "heading":
        if _is_out_of_scope_accounting_heading(text):
            return False
        if _looks_like_table_or_financial_grid(text) or _looks_like_footnote(text):
            return False
        if re.search(r"\(en millions", text, flags=re.IGNORECASE):
            return False
        return True
    if _looks_like_table_or_financial_grid(text) or _looks_like_footnote(text):
        return False
    if _FOOTNOTE_LINE_RE.match(text):
        return False
    if re.search(r"\(en millions", text, flags=re.IGNORECASE) and re.search(r"\b20\d{2}\b", text):
        return False
    return True


def _block_below_end_anchor(block: PDFBlock, audit: SectionAudit) -> bool:
    """Indique si un bloc se trouve sous l'ancre de fin sur une page partagée."""
    if audit.end_anchor_page is None or not audit.end_anchor_bbox_norm:
        return False
    if int(block.page) != int(audit.end_anchor_page):
        return False
    return float(block.y0) >= float(audit.end_anchor_bbox_norm[1])


def _audit_blocks(audit: SectionAudit) -> list[PDFBlock]:
    """Retourne tous les blocs utiles à l'alignement texte↔section."""
    blocks: list[PDFBlock] = list(audit.included_blocks)
    for block in audit.excluded_blocks:
        if not _is_docling_heading_block(block):
            continue
        if _is_out_of_scope_accounting_heading(block.text):
            continue
        if _block_below_end_anchor(block, audit):
            continue
        blocks.append(block)
    return blocks


def _segment_matches_audit(segment: DoclingSegment, audit: SectionAudit) -> bool:
    """Vérifie si un segment Docling correspond à un bloc de la section."""
    segment_norm = _normalized_block_text(segment.text)
    if not segment_norm:
        return False
    for block in _audit_blocks(audit):
        block_norm = _normalized_block_text(block.text)
        if not block_norm:
            continue
        if segment_norm == block_norm:
            return True
        if len(segment_norm) >= 20 and segment_norm in block_norm:
            return True
        if len(block_norm) >= 20 and block_norm in segment_norm:
            return True
    return False


def _matching_section_keys(segment: DoclingSegment, audits: list[SectionAudit]) -> list[str]:
    """Retourne les section_key candidates pour un segment."""
    return [audit.section_key for audit in audits if _segment_matches_audit(segment, audit)]


def _assign_segments_to_sections(
    segments: list[DoclingSegment],
    audits: list[SectionAudit],
) -> dict[str, list[DoclingSegment]]:
    """Répartit les segments Docling dans les sections AMF en conservant l'ordre."""
    assigned: dict[str, list[DoclingSegment]] = {audit.section_key: [] for audit in audits}
    if not audits:
        return assigned

    audits_by_start = sorted(audits, key=lambda audit: (audit.start_page, _SECTION_ORDER.get(audit.section_key, 99)))
    current_section: str | None = None
    stopped_sections: set[str] = set()

    for segment in segments:
        if segment.kind == "heading" and _is_end_boundary_heading(segment, audits):
            if current_section is not None:
                stopped_sections.add(current_section)
            current_section = None
            continue

        if not _should_keep_docling_segment(segment, audits):
            continue

        if segment.kind == "heading":
            allowed_sections = [
                audit.section_key
                for audit in audits
                if _segment_heading_allowed(segment, audit) and _segment_matches_audit(segment, audit)
            ]
            if not allowed_sections:
                continue
            matched_keys = allowed_sections
        else:
            matched_keys = _matching_section_keys(segment, audits)
        if matched_keys:
            for audit in audits_by_start:
                if audit.section_key in matched_keys:
                    current_section = audit.section_key
                    break
        elif segment.kind != "heading" or current_section is None:
            continue

        if current_section is None or current_section in stopped_sections:
            continue
        assigned[current_section].append(segment)

    return assigned


def _page_lookup_for_section(audit: SectionAudit) -> dict[str, int]:
    """Construit un index texte normalisé → page pour une section."""
    lookup: dict[str, int] = {}
    for block in _audit_blocks(audit):
        block_norm = _normalized_block_text(block.text)
        if block_norm:
            lookup[block_norm] = block.page
    return lookup


def _page_for_segment(
    segment: DoclingSegment,
    *,
    page_lookup: dict[str, int],
    fallback_page: int | None,
) -> int | None:
    """Retrouve la page PDF d'un segment via les blocs audités."""
    segment_norm = _normalized_block_text(segment.text)
    if segment_norm in page_lookup:
        return page_lookup[segment_norm]

    best_page: int | None = None
    best_score = 0.0
    segment_words = set(segment_norm.split())
    for block_norm, page in page_lookup.items():
        if segment_norm in block_norm or block_norm in segment_norm:
            return page
        block_words = set(block_norm.split())
        if not block_words:
            continue
        overlap = len(segment_words & block_words)
        score = overlap / max(len(segment_words), len(block_words))
        if score > best_score:
            best_score = score
            best_page = page
    if best_score >= 0.30:
        return best_page
    return fallback_page


def _build_text_extraction_markdown_from_docling(
    section_audits: list[SectionAudit],
    *,
    raw_docling_markdown: str,
) -> str:
    """Construit le markdown filtré en s'alignant sur la structure Docling."""
    segments = _parse_docling_markdown(raw_docling_markdown)
    assigned = _assign_segments_to_sections(segments, section_audits)
    ordered_audits = sorted(
        section_audits,
        key=lambda audit: (_SECTION_ORDER.get(audit.section_key, 99), audit.start_page),
    )

    lines: list[str] = []
    seen_heading_norms: dict[str, set[str]] = {}

    for audit in ordered_audits:
        section_segments = assigned.get(audit.section_key, [])
        if not section_segments:
            continue

        page_lookup = _page_lookup_for_section(audit)
        seen_heading_norms.setdefault(audit.section_key, set())
        section_title_norm = _normalized_block_text(audit.section_title)

        lines.append(_format_heading_line("##", audit.section_title, audit.start_page))
        lines.append("")

        last_page: int | None = audit.start_page
        for segment in section_segments:
            page = _page_for_segment(segment, page_lookup=page_lookup, fallback_page=last_page)
            if page is not None:
                last_page = page

            if segment.kind == "heading":
                heading_norm = _normalized_block_text(segment.text)
                if not heading_norm or heading_norm == section_title_norm:
                    continue
                if _is_out_of_scope_accounting_heading(segment.text):
                    continue
                if heading_norm in seen_heading_norms[audit.section_key]:
                    continue
                seen_heading_norms[audit.section_key].add(heading_norm)
                lines.append(_format_heading_line("###", segment.text, page))
                lines.append("")
                continue

            lines.append(segment.text)
            lines.append("")

    return "\n".join(lines).strip() + "\n"
