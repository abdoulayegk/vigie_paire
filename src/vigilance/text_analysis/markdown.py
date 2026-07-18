"""Composants modulaires du pipeline texte."""

from __future__ import annotations

import logging
import re
from typing import Any

from vigilance.cli.quarter_logic import normalize_quarter
from vigilance.config.loader import load_config
from vigilance.text_analysis.constants import _OUT_OF_SCOPE_ACCOUNTING_HEADING_PATTERNS, _SECTION_LABELS
from vigilance.text_analysis.list_items import format_list_item_markdown, parse_list_item_line
from vigilance.text_analysis.models import PDFBlock, SectionAudit
from vigilance.text_analysis.normalization import (
    _looks_like_footnote,
    _looks_like_table_or_financial_grid,
    _normalized_block_text,
)

logger = logging.getLogger(__name__)


def _is_out_of_scope_accounting_heading(text: str) -> bool:
    """Indique si un titre appartient aux notes comptables / états financiers."""
    value = str(text or "").strip()
    if not value:
        return False
    return any(pattern.search(value) for pattern in _OUT_OF_SCOPE_ACCOUNTING_HEADING_PATTERNS)


def _looks_like_section_heading_text(text: str) -> bool:
    """Heuristique prudente pour garder les titres Docling non étiquetés."""
    value = str(text or "").strip()
    if not value:
        return False
    if len(value) < 5 or len(value) > 180:
        return False
    if _looks_like_table_or_financial_grid(value) or _looks_like_footnote(value):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)
    if len(words) < 2 or len(words) > 22:
        return False
    if re.search(r"[.!?]\s*$", value):
        return False
    alpha_chars = sum(1 for ch in value if ch.isalpha())
    digit_chars = sum(1 for ch in value if ch.isdigit())
    return alpha_chars >= 5 and digit_chars <= max(2, alpha_chars // 6)


def _is_docling_heading_block(block: PDFBlock) -> bool:
    """Indique si un bloc doit être conservé comme titre/sous-titre markdown."""
    label = str(block.source_label or "").lower()
    text = str(block.text or "").strip()
    if not text:
        return False

    if _is_out_of_scope_accounting_heading(text):
        return False

    if label in {"page_header", "page_footer", "caption", "footnote"}:
        return False
    if re.search(r"rapport de gestion\s+\d+", text, flags=re.IGNORECASE):
        return False
    if re.search(r"groupe banque td", text, flags=re.IGNORECASE) and re.search(
        r"rapport",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    if _looks_like_table_or_financial_grid(text) or _looks_like_footnote(text):
        return False
    if re.search(r"\(en millions", text, flags=re.IGNORECASE):
        return False
    if re.search(r"[;]\s*$", text):
        return False
    if re.search(r":\s*$", text) and not text.isupper():
        return False
    if re.search(r"\b\d{2,}(?:\s+\d{2,}){2,}\b", text):
        return False

    if label in {"title", "section_header"}:
        return True
    if block.heading_level is not None and int(block.heading_level) > 0:
        return True
    if label == "pymupdf_fallback" and _looks_like_section_heading_text(text):
        return True
    return False


def _is_structural_markdown_heading(block: PDFBlock) -> bool:
    """Indique qu'un bloc peut réellement devenir un titre Markdown.

    Docling peut étiqueter un paragraphe entier ``section_header``. Le rendu de
    repli ne doit pas transformer ce contenu en ``###`` : les titres Markdown
    ne sont pas chunkés pour la comparaison. On exige donc aussi la forme d'un
    titre court et non phrastique.
    """
    return _is_docling_heading_block(block) and _looks_like_section_heading_text(block.text)


def _markdown_blocks_for_section(section: SectionAudit) -> list[PDFBlock]:
    """Retourne les blocs à inclure dans le markdown source de vérité d'une section.

    Sélectionne les blocs narratifs inclus, plus les titres/en-têtes exclus pour
    raison ``non_narrative_block`` (afin de préserver la structure ``###`` dans le
    markdown). Les doublons sont dédupliqués par ``block_id``. Le résultat est
    trié par (page, line_number, y0) pour respecter l'ordre de lecture.
    """
    selected: list[PDFBlock] = []
    seen_ids: set[str] = set()
    for block in section.included_blocks:
        if block.block_id not in seen_ids:
            selected.append(block)
            seen_ids.add(block.block_id)
    for block in section.excluded_blocks:
        if block.block_id in seen_ids:
            continue
        # La classification spatiale des tableaux est une décision durable.
        # Un renderer de repli ne doit jamais la renverser sous prétexte que
        # Docling avait aussi attribué une étiquette de titre à la cellule.
        if block.block_type in {"table", "table_footnote"} or block.exclusion_reason in {
            "table_like_block",
            "table_footnote",
        }:
            continue
        if not _is_structural_markdown_heading(block):
            continue
        if block.exclusion_reason not in {"", "non_narrative_block"}:
            continue
        if _looks_like_table_or_financial_grid(block.text) or _looks_like_footnote(block.text):
            continue
        selected.append(block)
        seen_ids.add(block.block_id)
    return sorted(selected, key=lambda block: (block.page, block.line_number, block.y0))


def _get_page_number_offset_for_period(
    bank_code: str,
    *,
    year: int,
    quarter: str,
) -> int:
    """Retourne l'offset page imprimée -> page PDF physique pour une période."""
    try:
        cfg = load_config("configs/bank_profiles.yaml")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Impossible de charger les offsets de pages (%s); offset=0", exc)
        return 0

    bank_data = (cfg.get("banks") or {}).get(str(bank_code or "").lower(), {})
    if not isinstance(bank_data, dict):
        return 0

    quarter_key = normalize_quarter(quarter)
    period_offsets = bank_data.get("page_number_offsets", {})
    if isinstance(period_offsets, dict):
        for key in (f"{quarter_key}_{year}", quarter_key):
            if key in period_offsets:
                try:
                    return int(period_offsets.get(key) or 0)
                except (TypeError, ValueError):
                    return 0

    try:
        return int(bank_data.get("page_number_offset") or 0)
    except (TypeError, ValueError):
        return 0


def _format_page_suffix(physical_page: int) -> str:
    """Retourne le suffixe inline `` [pdf.N]`` pour un titre."""
    return f" [pdf.{int(physical_page)}]"


def _format_heading_line(prefix: str, title: str, physical_page: int | None) -> str:
    """Formate un titre markdown avec page PDF inline."""
    clean_title = str(title or "").strip()
    if physical_page is not None and int(physical_page) > 0:
        return f"{prefix} {clean_title}{_format_page_suffix(int(physical_page))}"
    return f"{prefix} {clean_title}"


def _format_page_marker(physical_page: int, *, page_number_offset: int = 0) -> str:
    """Formate un marqueur de page PDF autonome (migration legacy)."""
    _ = page_number_offset
    return f"[pdf.{int(physical_page)}]"


_INLINE_PDF_MARKER_RE = re.compile(r"\s*\[pdf\.(\d+)\]\s*$")
_INLINE_LEGACY_MARKER_RE = re.compile(r"\s*\[p\.(\d+)(?:\s*\|\s*pdf\.(\d+))?\]\s*$")
_STANDALONE_PDF_MARKER_RE = re.compile(r"^\[pdf\.(\d+)\]\s*$")
_STANDALONE_LEGACY_MARKER_RE = re.compile(
    r"^\[p\.(\d+)(?:\s*\|\s*pdf\.(\d+))?\]\s*$",
)


def _extract_inline_pdf_page(line: str) -> int | None:
    """Extrait la page PDF physique d'un suffixe inline sur un titre."""
    match = _INLINE_PDF_MARKER_RE.search(line)
    if match:
        return int(match.group(1))
    legacy = _INLINE_LEGACY_MARKER_RE.search(line)
    if legacy:
        return int(legacy.group(2) or legacy.group(1))
    return None


def _strip_inline_page_suffix(text: str) -> str:
    """Retire les suffixes de page inline d'un titre."""
    value = _INLINE_PDF_MARKER_RE.sub("", text)
    return _INLINE_LEGACY_MARKER_RE.sub("", value).strip()


def _extract_physical_page_from_standalone_marker(line: str) -> int | None:
    """Extrait la page PDF d'une ligne marqueur autonome (formats legacy inclus)."""
    stripped = line.strip()
    pdf_match = _STANDALONE_PDF_MARKER_RE.match(stripped)
    if pdf_match:
        return int(pdf_match.group(1))
    legacy_match = _STANDALONE_LEGACY_MARKER_RE.match(stripped)
    if legacy_match:
        return int(legacy_match.group(2) or legacy_match.group(1))
    return None


def _rewrite_page_markers_for_display(md_content: str) -> str:
    """Migre les marqueurs autonomes vers ``[pdf.N]`` inline sur les titres seulement."""
    lines = md_content.splitlines()
    result: list[str] = []
    pending_page: int | None = None

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            result.append(line)
            continue

        standalone_page = _extract_physical_page_from_standalone_marker(line)
        if standalone_page is not None:
            pending_page = standalone_page
            continue

        if line.startswith("### "):
            title = _strip_inline_page_suffix(line[4:]).strip()
            page = _extract_inline_pdf_page(line) or pending_page
            pending_page = None
            result.append(_format_heading_line("###", title, page))
            continue

        if line.startswith("## "):
            title = _strip_inline_page_suffix(line[3:]).strip()
            page = _extract_inline_pdf_page(line) or pending_page
            pending_page = None
            result.append(_format_heading_line("##", title, page))
            continue

        pending_page = None
        result.append(line)

    return "\n".join(result).strip() + "\n"


def _build_text_extraction_markdown(
    section_audits: list[SectionAudit],
    *,
    raw_docling_markdown: str | None = None,
) -> str:
    """Convertit une liste d'audits de sections en markdown source de vérité.

    Quand ``raw_docling_markdown`` est fourni, la structure (titres/sous-titres
    et ordre) provient du markdown natif Docling. Sinon, repli sur les blocs PDF.
    """
    if raw_docling_markdown and raw_docling_markdown.strip():
        from vigilance.text_analysis.docling_markdown import _build_text_extraction_markdown_from_docling

        return _build_text_extraction_markdown_from_docling(
            section_audits,
            raw_docling_markdown=raw_docling_markdown,
        )

    return _build_text_extraction_markdown_from_blocks(section_audits)


def _build_text_extraction_markdown_from_blocks(
    section_audits: list[SectionAudit],
) -> str:
    """Convertit une liste d'audits de sections en markdown source de vérité.

    Chaque section devient un bloc ``## Titre [pdf.N]``. Les sous-titres détectés
    deviennent des blocs ``### Sous-titre [pdf.N]`` positionnés juste avant le
    premier paragraphe qui les suit. Les paragraphes narratifs n'ont pas de
    marqueur de page. Les suffixes sont strippés avant tout appel GPT.
    """
    lines: list[str] = []
    for section in section_audits:
        lines.append(_format_heading_line("##", section.section_title, section.start_page))
        lines.append("")
        seen_heading_norms: set[str] = set()
        section_title_norm = _normalized_block_text(section.section_title)
        pending_headings: list[tuple[str, int]] = []
        for block in _markdown_blocks_for_section(section):
            text = str(block.text or "").strip()
            if not text:
                continue
            norm = _normalized_block_text(text)
            if _is_structural_markdown_heading(block):
                if not norm or norm == section_title_norm or norm in seen_heading_norms:
                    continue
                pending_headings.append((text, int(block.page)))
                seen_heading_norms.add(norm)
                continue
            if pending_headings:
                # Les titres successifs sans bloc narratif intermédiaire sont
                # des parents structurels ou des titres de tableau. Seul le
                # dernier introduit le texte qui suit et devient comparable.
                heading_text, heading_page = pending_headings[-1]
                lines.append(_format_heading_line("###", heading_text, heading_page))
                lines.append("")
                pending_headings.clear()
            parsed_list_item = parse_list_item_line(text)
            if parsed_list_item is not None:
                lines.append(
                    format_list_item_markdown(
                        parsed_list_item.text,
                        marker=parsed_list_item.marker,
                        indent=parsed_list_item.indent,
                    )
                )
            else:
                lines.append(text)
            lines.append("")
    # Un titre final sans bloc narratif propre est conservé par l'audit, pas
    # par le markdown utilisé pour le matching.
    return "\n".join(lines).strip() + "\n"


def _build_block_page_index(section: SectionAudit) -> list[tuple[int, str]]:
    """Retourne la liste ordonnée (page, texte) des blocs narratifs d'une section.

    Utilisée pour retrouver la page exacte d'un fragment de texte retourné par GPT.
    Les blocs titres/en-têtes sont exclus car GPT ne les retourne pas comme corps de texte.
    """
    index: list[tuple[int, str]] = []
    for block in _markdown_blocks_for_section(section):
        if _is_structural_markdown_heading(block):
            continue
        text = str(block.text or "").strip()
        if text:
            index.append((block.page, text))
    return index


def _find_page_for_fragment(fragment: str, block_index: list[tuple[int, str]]) -> int | None:
    """Retrouve la page du bloc qui correspond le mieux au fragment de texte GPT.

    Stratégie en deux passes :
    1. Correspondance substring normalisée (exact ou inclusion)
    2. Score de recouvrement de mots (retenu si ≥ 30 %)
    """
    if not fragment or not block_index:
        return None
    frag_norm = _normalized_block_text(fragment)
    if not frag_norm:
        return None

    # Pass 1 — substring
    for page, text in block_index:
        text_norm = _normalized_block_text(text)
        if frag_norm in text_norm or text_norm in frag_norm:
            return page

    # Pass 2 — word overlap
    frag_words = set(frag_norm.split())
    best_score = 0.0
    best_page: int | None = None
    for page, text in block_index:
        text_words = set(_normalized_block_text(text).split())
        if not text_words:
            continue
        overlap = len(frag_words & text_words)
        score = overlap / max(len(frag_words), len(text_words))
        if score > best_score:
            best_score = score
            best_page = page

    return best_page if best_score >= 0.30 else None


def _strip_page_markers(text: str) -> str:
    """Supprime les marqueurs de page d'un bloc de texte avant appel GPT."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            lines.append(line)
            continue
        if _extract_physical_page_from_standalone_marker(line) is not None:
            continue
        if line.startswith("### "):
            lines.append(_format_heading_line("###", _strip_inline_page_suffix(line[4:]), None))
            continue
        if line.startswith("## "):
            lines.append(_format_heading_line("##", _strip_inline_page_suffix(line[3:]), None))
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_section_text_from_markdown(md_content: str, section_key: str) -> str:
    """Extrait le texte d'une section depuis le markdown auditable.

    Le markdown ecrit sur disque est la source de verite fonctionnelle du flux
    texte; ce helper redecoupe exactement ce meme contenu avant comparaison.
    Les marqueurs ``[pdf.N]`` sont strippés avant retour — c'est l'unique
    gatekeeper vers les appels GPT du flux texte.
    """
    title = _SECTION_LABELS.get(section_key, section_key)
    lines = md_content.splitlines()
    in_section = False
    section_lines: list[str] = []
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            if in_section:
                break
            heading_title = _strip_inline_page_suffix(line[3:]).strip()
            if heading_title == title:
                in_section = True
                continue
        if in_section:
            section_lines.append(line)
    joined = "\n".join(section_lines).strip()
    return _strip_page_markers(joined).strip()


def _parse_page_index_from_markdown(
    md_content: str,
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, int]]:
    """Reconstruit l'index page→texte et les start_pages par section depuis le .md.

    Remplace ``_build_block_page_index`` quand on réutilise un .md existant
    plutôt que de relancer Docling. Retourne un tuple ``(index, section_start_pages)``:

    - ``index[section_key] = [(page, text), …]`` — même forme qu'attendue par
      ``_find_page_for_fragment``. Titres/sous-titres ``##``/``###`` exclus,
      comme dans la version basée sur ``SectionAudit``.
    - ``section_start_pages[section_key] = N`` — la page PDF du suffixe ``[pdf.N]`` sur
      ``## Title``. Reproduit fidèlement ``ResolvedSection.start_page``.
    """
    title_to_key: dict[str, str] = {v: k for k, v in _SECTION_LABELS.items()}

    index: dict[str, list[tuple[int, str]]] = {}
    section_start_pages: dict[str, int] = {}
    current_section_key: str | None = None
    current_page: int | None = None

    lines = md_content.splitlines()
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        standalone_page = _extract_physical_page_from_standalone_marker(line)
        if standalone_page is not None:
            current_page = standalone_page
            continue

        if line.startswith("### "):
            inline_page = _extract_inline_pdf_page(line)
            if inline_page is not None:
                current_page = inline_page
            continue

        if line.startswith("## ") and not line.startswith("### "):
            title = _strip_inline_page_suffix(line[3:]).strip()
            inline_page = _extract_inline_pdf_page(line)
            if inline_page is not None:
                current_page = inline_page
            current_section_key = title_to_key.get(title)
            if current_section_key is not None:
                if current_section_key not in index:
                    index[current_section_key] = []
                if current_page is not None and current_section_key not in section_start_pages:
                    section_start_pages[current_section_key] = current_page
            continue

        if current_section_key is None:
            continue
        if current_page is None:
            continue
        index[current_section_key].append((current_page, line.strip()))
    return index, section_start_pages


def _section_page_range_from_index(
    page_index: list[tuple[int, str]],
    *,
    start_page_hint: int | None = None,
) -> tuple[int | None, int | None]:
    """Déduit ``(start_page, end_page)`` d'une section à partir de son index.

    Si ``start_page_hint`` est fourni (ex: extrait du ``[p.N]`` précédant
    le header ``## Title`` dans le .md), il prime sur le min des pages des
    blocs pour rester cohérent avec ``ResolvedSection.start_page``.
    """
    if not page_index:
        if start_page_hint is not None:
            return start_page_hint, start_page_hint
        return None, None
    pages = [p for p, _ in page_index if isinstance(p, int) and p > 0]
    if not pages:
        if start_page_hint is not None:
            return start_page_hint, start_page_hint
        return None, None
    start = start_page_hint if start_page_hint is not None else min(pages)
    return start, max(pages)
