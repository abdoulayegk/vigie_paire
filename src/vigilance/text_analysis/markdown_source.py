"""Construction et lecture du markdown source de verite."""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

from vigilance.cli.quarter_logic import normalize_quarter
from vigilance.config.loader import load_config

from .constants import _PAGE_MARKER_RE, _SECTION_LABELS
from .models import PDFBlock, SectionAudit
from .pdf_block_classification import _looks_like_footnote, _looks_like_table_or_financial_grid
from .text_normalization import _normalized_block_text

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
        if block.exclusion_reason != "non_narrative_block":
            continue
        if block.source_label not in {"title", "section_header"}:
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


def _format_page_marker(physical_page: int, *, page_number_offset: int = 0) -> str:
    """Formate le marqueur markdown: page imprimée visible, page PDF conservée."""
    page = int(physical_page)
    if page_number_offset <= 0:
        return f"[p.{page}]"
    printed_page = max(1, page - int(page_number_offset))
    if printed_page == page:
        return f"[p.{page}]"
    return f"[p.{printed_page} | pdf.{page}]"


def _rewrite_page_markers_for_display(md_content: str, *, page_number_offset: int) -> str:
    """Réécrit les marqueurs existants pour afficher la page imprimée."""
    if page_number_offset <= 0:
        return md_content

    def _replace_marker(match: re.Match[str]) -> str:
        physical_page = int(match.group(2) or match.group(1))
        return _format_page_marker(physical_page, page_number_offset=page_number_offset)

    return _PAGE_MARKER_RE.sub(_replace_marker, md_content)


def _build_text_extraction_markdown(
    section_audits: list[SectionAudit],
    *,
    page_number_offset: int = 0,
) -> str:
    """Convertit une liste d'audits de sections en markdown source de vérité.

    Chaque section devient un bloc ``## Titre``. Les sous-titres détectés deviennent
    des blocs ``### Sous-titre`` positionnés juste avant le premier paragraphe qui
    les suit. Les doublons de titres et les titres identiques à la section parente
    sont supprimés. Chaque bloc (titre, sous-titre, paragraphe) est précédé d'un
    marqueur ``[p.N]`` indiquant sa page d'origine ; ces marqueurs sont strippés
    avant tout appel GPT (cf ``_extract_section_text_from_markdown``) et servent
    à reconstruire l'index page→texte lors de la réutilisation du .md. Quand un
    offset est connu, le marqueur affiche la page imprimée et conserve la page
    physique sous la forme ``[p.58 | pdf.60]``.
    """
    lines: list[str] = []
    for section in section_audits:
        lines.append(_format_page_marker(section.start_page, page_number_offset=page_number_offset))
        lines.append(f"## {section.section_title}")
        lines.append("")
        seen_heading_norms: set[str] = set()
        section_title_norm = _normalized_block_text(section.section_title)
        pending_heading: tuple[str, int] | None = None
        for block in _markdown_blocks_for_section(section):
            text = str(block.text or "").strip()
            if not text:
                continue
            norm = _normalized_block_text(text)
            if block.source_label in {"title", "section_header"}:
                if not norm or norm == section_title_norm or norm in seen_heading_norms:
                    continue
                pending_heading = (text, int(block.page))
                seen_heading_norms.add(norm)
                continue
            if pending_heading is not None:
                heading_text, heading_page = pending_heading
                lines.append(_format_page_marker(heading_page, page_number_offset=page_number_offset))
                lines.append(f"### {heading_text}")
                lines.append("")
                pending_heading = None
            lines.append(_format_page_marker(int(block.page), page_number_offset=page_number_offset))
            lines.append(text)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_block_page_index(section: SectionAudit) -> list[tuple[int, str]]:
    """Retourne la liste ordonnée (page, texte) des blocs narratifs d'une section.

    Utilisée pour retrouver la page exacte d'un fragment de texte retourné par GPT.
    Les blocs titres/en-têtes sont exclus car GPT ne les retourne pas comme corps de texte.
    """
    index: list[tuple[int, str]] = []
    for block in _markdown_blocks_for_section(section):
        if block.source_label in {"title", "section_header"}:
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
    """Supprime les marqueurs ``[p.N]`` / ``[p.N | pdf.M]`` d'un bloc de texte.

    Utilisé exclusivement avant tout appel GPT : les marqueurs servent à
    l'audit humain et à la reconstruction de l'index page→texte, mais ne
    doivent jamais entrer dans les prompts pour éviter de biaiser la
    comparaison sémantique.
    """
    return re.sub(
        r"^\[p\.\d+(?:\s*\|\s*pdf\.\d+)?\]\s*\n?",
        "",
        text,
        flags=re.MULTILINE,
    )


def _extract_section_text_from_markdown(md_content: str, section_key: str) -> str:
    """Extrait le texte d'une section depuis le markdown auditable.

    Le markdown ecrit sur disque est la source de verite fonctionnelle du flux
    texte; ce helper redecoupe exactement ce meme contenu avant comparaison.
    Les marqueurs ``[p.N]`` sont strippés avant retour — c'est l'unique
    gatekeeper vers les appels GPT du flux texte.
    """
    title = _SECTION_LABELS.get(section_key, section_key)
    lines = md_content.splitlines()
    in_section = False
    section_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if in_section:
                break
            # Skip leading page marker if present before the section header
            stripped = line.strip()
            if stripped == f"## {title}":
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
    - ``section_start_pages[section_key] = N`` — la page PDF physique du marqueur qui précède
      immédiatement ``## Title``. Reproduit fidèlement ``ResolvedSection.start_page``
      même quand la page du header diffère de la première page narrative.
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
        page_match = _PAGE_MARKER_RE.match(line)
        if page_match:
            current_page = int(page_match.group(2) or page_match.group(1))
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            current_section_key = title_to_key.get(title)
            if current_section_key is not None:
                if current_section_key not in index:
                    index[current_section_key] = []
                if current_page is not None and current_section_key not in section_start_pages:
                    section_start_pages[current_section_key] = current_page
            continue
        if line.startswith("### "):
            # Sub-heading: ignored in the index (matches _build_block_page_index).
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
