"""Construction du markdown filtré à partir du markdown natif Docling."""

from __future__ import annotations

import logging
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
    _is_chart_axis_label_row,
    _is_not_applicable_marker,
    _is_running_report_chrome,
    _is_table_unit_label,
    _normalized_block_text,
)

logger = logging.getLogger(__name__)

_SECTION_ORDER = {
    "gestion_capital": 0,
    "gestion_risques": 1,
    "gestion_reglementation": 2,
}

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_LIST_ITEM_RE = re.compile(r"^[-*]\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_EXPLICIT_FOOTNOTE_MARKER_RE = re.compile(
    r"^\s*(?:\(?\d{1,2}\)|[¹²³⁴⁵⁶⁷⁸⁹]+|[*†‡]{1,3}|"
    r"(?:note|source|s\.?\s*o\.?|n\.?\s*s\.?)\b)",
    flags=re.IGNORECASE,
)
_BARE_NUMERIC_NOTE_RE = re.compile(r"^\s*\d{1,2}\s+(?!\.)")
_TABLE_FOOTNOTE_CUE_RE = re.compile(
    r"\b(?:comprennent|excluent|incluent|les\s+m[eé]thodes|"
    r"pour\s+de\s+plus\s+amples|se\s+reporter|voir\s+le\s+tableau|"
    r"au\s+tableau|notes?\s+\d+|non\s+significatif|sans\s+objet|"
    r"ratio\s+pr[eê]t[-–\s]valeur|calculons\s+le\s+ratio|"
    r"ligne\s+directrice\s+b-\d+)\b",
    flags=re.IGNORECASE,
)
_DATED_NARRATIVE_RE = re.compile(
    r"\b(?:le\s+)?\d{1,2}\s+"
    r"(?:janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
    r"septembre|octobre|novembre|d[eé]cembre)\s+20\d{2}\b",
    flags=re.IGNORECASE,
)
_END_BOUNDARY_HEADING_PATTERNS = [
    re.compile(r"normes\s+et\s+m[eé]thodes\s+comptables", re.IGNORECASE),
    re.compile(r"m[eé]thodes\s+et\s+estimations\s+comptables", re.IGNORECASE),
    re.compile(r"m[eé]thodes\s+comptables\s+significatives", re.IGNORECASE),
    re.compile(r"[eé]tats?\s+financiers?", re.IGNORECASE),
]
_INTERNAL_RISK_ACCOUNTING_HEADING_RE = re.compile(
    r"^conventions?,?\s+m[eé]thodes\s+et\s+estimations\s+comptables\s+"
    r"utilis[eé]es\s+par\s+la\s+banque$",
    re.IGNORECASE,
)


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

    # Les rapports BNC contiennent cette sous-section comptable précise à
    # l'intérieur même de « Gestion des risques », avant « Risque de crédit ».
    # L'exception reste volontairement étroite : les vrais chapitres comptables
    # de TD et des autres banques doivent continuer à fermer la section.
    if _INTERNAL_RISK_ACCOUNTING_HEADING_RE.fullmatch(text):
        for audit in audits:
            for block in [*audit.included_blocks, *audit.excluded_blocks]:
                if block.exclusion_reason == "outside_target_section":
                    continue
                if _normalized_block_text(block.text) != segment_norm:
                    continue
                if int(block.page) < int(audit.end_page):
                    return False

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
    follows_table: bool = False


def _has_table_footnote_cue(text: str) -> bool:
    """Indique si un texte contient un indice lexical propre aux notes de tableau."""
    return _TABLE_FOOTNOTE_CUE_RE.search(str(text or "")) is not None


def _is_bare_numeric_table_footnote(text: str, *, follows_table: bool) -> bool:
    """Distingue une note de tableau numérotée d'une divulgation narrative numérotée."""
    value = str(text or "").strip()
    if not _BARE_NUMERIC_NOTE_RE.match(value):
        return False
    if _DATED_NARRATIVE_RE.search(value) and not _has_table_footnote_cue(value):
        return False
    return _has_table_footnote_cue(value) or (
        follows_table and not _DATED_NARRATIVE_RE.search(value)
    )


def _parse_docling_markdown(md_content: str) -> list[DoclingSegment]:
    """Découpe le markdown Docling en segments ordonnés."""
    segments: list[DoclingSegment] = []
    in_table = False
    follows_table = False
    for raw_line in md_content.splitlines():
        line = raw_line.strip()
        if not line:
            follows_table = follows_table or in_table
            in_table = False
            continue

        if _TABLE_ROW_RE.match(line):
            in_table = True
            follows_table = False
            continue
        if in_table:
            continue

        heading_match = _HEADING_LINE_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            if text:
                segments.append(
                    DoclingSegment(
                        kind="heading",
                        text=text,
                        heading_level=level,
                        follows_table=follows_table,
                    )
                )
            follows_table = False
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            text = list_match.group(1).strip()
            if text:
                segments.append(
                    DoclingSegment(
                        kind="list_item",
                        text=text,
                        follows_table=follows_table,
                    )
                )
            follows_table = False
            continue

        text = line
        if not text:
            follows_table = False
            continue
        segments.append(
            DoclingSegment(
                kind="paragraph",
                text=text,
                follows_table=follows_table,
            )
        )
        follows_table = False
    return segments


def _segment_matches_block(segment: DoclingSegment, block: PDFBlock) -> bool:
    """Vérifie si un segment Docling correspond à un bloc PDF audité."""
    segment_norm = _normalized_block_text(segment.text)
    if not segment_norm:
        return False
    block_norm = _normalized_block_text(block.text)
    if not block_norm:
        return False
    if segment_norm == block_norm:
        return True
    if len(segment_norm) >= 20 and segment_norm in block_norm:
        return True
    if len(block_norm) >= 20 and block_norm in segment_norm:
        return True
    return False


def _segment_matches_included_block(segment: DoclingSegment, audit: SectionAudit) -> bool:
    """Vérifie si un segment correspond à un bloc narratif inclus."""
    return any(_segment_matches_block(segment, block) for block in audit.included_blocks)


def _is_confirmed_narrative_segment(
    segment: DoclingSegment,
    audits: list[SectionAudit] | None,
) -> bool:
    """Indique qu'un segment est confirmé comme narratif par l'audit PDF.

    L'audit combine le libellé Docling, la géométrie de la page et les régions
    de tableaux pour décider qu'un bloc appartient réellement au narratif de
    la section. Cette information est plus fiable qu'une heuristique fondée
    uniquement sur des mots comme ``total`` et des pourcentages.

    Les titres restent exclus : ils sont structurels et sont traités par les
    règles spécifiques aux en-têtes plus bas dans le flux.
    """
    if segment.kind not in {"paragraph", "list_item"} or not audits:
        return False
    return any(_segment_matches_included_block(segment, audit) for audit in audits)


def _is_confirmed_footnote_segment(
    segment: DoclingSegment,
    audits: list[SectionAudit] | None,
) -> bool:
    """Indique que l'audit PDF a identifié le segment comme note de tableau.

    Ce repli couvre les formats propres aux banques qui ne portent pas tous un
    marqueur universel, par exemple une note BMO commençant simplement par
    ``1`` sous une figure. La géométrie de la note et son libellé Docling sont
    alors plus fiables que son préfixe textuel seul.
    """
    if segment.kind not in {"paragraph", "list_item"} or not audits:
        return False
    for audit in audits:
        for block in audit.excluded_blocks:
            if block.block_type != "table_footnote" and block.exclusion_reason != "table_footnote":
                continue
            if _segment_matches_block(segment, block):
                return True
    return False


def _is_confirmed_table_segment(
    segment: DoclingSegment,
    audits: list[SectionAudit] | None,
) -> bool:
    """Indique que l'audit spatial a rattaché le segment à un tableau."""
    if not audits:
        return False
    for audit in audits:
        for block in audit.excluded_blocks:
            if block.block_type != "table" and block.exclusion_reason != "table_like_block":
                continue
            if _segment_matches_block(segment, block):
                return True
    return False


def _segment_heading_allowed(segment: DoclingSegment, audit: SectionAudit) -> bool:
    """N'accepte un titre Docling que s'il correspond à un vrai bloc-titre audité.

    Une correspondance approximative avec un paragraphe est interdite : par
    exemple, le titre ``Contrôle des risques`` ne doit pas être rattaché au
    capital parce qu'une phrase capital contient les mêmes mots.
    """
    if _is_out_of_scope_accounting_heading(segment.text):
        return False
    segment_norm = _normalized_block_text(segment.text)
    if not segment_norm:
        return False
    for block in _audit_blocks(audit):
        block_norm = _normalized_block_text(block.text)
        if segment_norm != block_norm:
            continue
        if block.block_type in {"table", "table_footnote"} or block.exclusion_reason in {
            "table_like_block",
            "table_footnote",
        }:
            return False
        return True
    return False


def _is_table_footnote_segment(segment: DoclingSegment) -> bool:
    """Indique si un segment est une note de tableau à exclure du narratif.

    Le repli sans géométrie reste volontairement strict : un marqueur ne suffit
    que s'il arrive immédiatement après une vraie table Markdown. Les notes
    numérotées ou « Source » ailleurs dans la section sont conservées.
    """
    if segment.kind == "heading":
        return False

    text = str(segment.text or "").strip()
    if not text:
        return False
    if not segment.follows_table:
        return False
    return bool(
        _EXPLICIT_FOOTNOTE_MARKER_RE.match(text)
        or _is_bare_numeric_table_footnote(text, follows_table=True)
    )


def _should_keep_docling_segment(segment: DoclingSegment, audits: list[SectionAudit] | None = None) -> bool:
    """Conserve tout segment hors table ou note de table confirmée.

    Les tables Markdown sont déjà retirées par le parseur. Les notes de table
    sont éliminées par leur lien géométrique audité ou, en repli, par une
    position immédiatement après une table. Aucun filtre lexical ne doit
    supprimer un montant, un pourcentage, un texte court ou une note autonome,
    à l'exception du marqueur autonome « s.o. ».
    """
    text = str(segment.text or "").strip()
    if not text:
        return False
    if _is_running_report_chrome(text) or _is_table_unit_label(text) or _is_chart_axis_label_row(text):
        return False
    if _is_not_applicable_marker(text):
        return False

    # Les notes confirmées par leur zone sous tableau sont les seules notes
    # exclues par les métadonnées PDF.
    if _is_confirmed_footnote_segment(segment, audits):
        return False

    if _is_confirmed_table_segment(segment, audits):
        return False

    if _is_table_footnote_segment(segment):
        return False

    if _is_confirmed_narrative_segment(segment, audits):
        return True
    if segment.kind == "heading":
        return not _is_out_of_scope_accounting_heading(text)
    return True


def _missing_audited_blocks(
    markdown: str,
    audits: list[SectionAudit],
) -> list[PDFBlock]:
    """Retourne les blocs audités absents du Markdown rendu."""
    rendered = _normalized_block_text(markdown)
    if not rendered:
        return [block for audit in audits for block in audit.included_blocks]

    missing: list[PDFBlock] = []
    for audit in audits:
        for block in audit.included_blocks:
            block_norm = _normalized_block_text(block.text)
            if not block_norm or block_norm in rendered:
                continue
            missing.append(block)
    return missing


def _warn_on_missing_audited_narrative_blocks(
    markdown: str,
    audits: list[SectionAudit],
) -> list[PDFBlock]:
    """Journalise et retourne toute perte entre l'audit et le Markdown."""
    missing = _missing_audited_blocks(markdown, audits)

    if missing:
        logger.warning(
            "Markdown narratif incomplet: %d bloc(s) audité(s) absent(s): %s",
            len(missing),
            ", ".join(
                f"{block.block_id}:pdf.{block.page}"
                for block in missing
            ),
        )
    return missing


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
        # Un titre hors de la fenêtre géométrique de la section ne doit jamais
        # redevenir une ancre de structure. Le markdown brut Docling peut le
        # placer juste avant la vraie ancre sur une page partagée.
        if block.exclusion_reason == "outside_target_section":
            continue
        if not _is_docling_heading_block(block):
            continue
        if _is_out_of_scope_accounting_heading(block.text):
            continue
        if _block_below_end_anchor(block, audit):
            continue
        blocks.append(block)
    return blocks


def _matchable_section_segments(segments: list[DoclingSegment]) -> list[DoclingSegment]:
    """Retire les titres sans contenu narratif propre du flux de matching.

    Les tableaux, légendes et notes sont déjà absents des ``segments``. Un
    titre suivi immédiatement d'un autre titre est donc soit un titre de
    tableau, soit un parent structurel, soit un en-tête parasite. Il reste dans
    l'artefact Docling brut et dans l'audit, mais ne devient pas une
    sous-section ``###`` à comparer.
    """
    selected: list[DoclingSegment] = []
    pending_heading: DoclingSegment | None = None

    for segment in segments:
        if segment.kind == "heading":
            # Seul le titre immédiatement associé au prochain bloc narratif
            # possède un contenu propre. Les parents/titres de tableaux qui le
            # précèdent sont volontairement exclus du matching.
            pending_heading = segment
            continue

        if pending_heading is not None:
            selected.append(pending_heading)
            pending_heading = None
        selected.append(segment)

    # Ne pas vider ``pending_heading`` : un titre final sans texte propre n'est
    # pas une unité comparable.
    return selected


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
        section_segments = _matchable_section_segments(assigned.get(audit.section_key, []))
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
                heading_text = segment.text.strip()
                heading_norm = _normalized_block_text(heading_text)
                if not heading_norm or heading_norm == section_title_norm:
                    continue
                if _is_out_of_scope_accounting_heading(segment.text):
                    continue
                if heading_norm in seen_heading_norms[audit.section_key]:
                    continue
                seen_heading_norms[audit.section_key].add(heading_norm)
                lines.append(_format_heading_line("###", heading_text, page))
                lines.append("")
                continue

            lines.append(segment.text)
            lines.append("")

    rendered = "\n".join(lines).strip() + "\n"
    missing = _warn_on_missing_audited_narrative_blocks(rendered, section_audits)
    if missing:
        # La structure Markdown de Docling est préférée lorsqu'elle couvre tout
        # l'audit. Dès qu'elle omet un bloc non-tabulaire, le rendu ordonné des
        # blocs PDF devient la source de vérité : conserver le contenu prévaut
        # sur une hiérarchie Docling incomplète.
        logger.warning(
            "Repli vers les blocs PDF audités pour préserver %d bloc(s) absent(s).",
            len(missing),
        )
        from vigilance.text_analysis.markdown import _build_text_extraction_markdown_from_blocks

        return _build_text_extraction_markdown_from_blocks(section_audits)
    return rendered
