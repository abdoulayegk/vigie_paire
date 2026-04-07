"""Pipeline texte canonique GPT-first.

Ce module remplace la chaîne extraction + alignement heuristique + diff/triage
par un seul orchestrateur qui:

1. localise les sections texte dans les deux PDFs,
2. extrait des unités sémantiques propres via GPT-4o Vision,
3. compare explicitement T1 vs T2 via GPT-4o,
4. trie les changements métiers,
5. ne conserve que les changements vraiment majeurs.

Le pipeline écrit ``text_comparison.json`` et deux artefacts d'audit
``text_extraction_*.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz

from vigilance.cli.quarter_logic import normalize_quarter, resolve_previous_quarter
from vigilance.extraction.section_locator import locate_sections_in_pdf
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.text_comparison.text_comparison_writer import (
    get_text_comparison_path,
    write_text_comparison,
)
from vigilance.text_extraction.text_extraction_audit_writer import (
    TEXT_EXTRACTION_AUDIT_SCHEMA_VERSION,
    get_text_extraction_audit_path,
    write_text_extraction_audit,
)
from vigilance.utils.genai import get_openai_api_key
from vigilance.utils.pymupdf_utils import configure_mupdf_runtime

logger = logging.getLogger(__name__)

UNIFIED_TEXT_SCHEMA_VERSION = 3

_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_CANONICAL_TO_TEXT_KEY: dict[str, str] = {
    "capital_management": "gestion_capital",
    "capital": "gestion_capital",
    "risk_management": "gestion_risques",
    "risk": "gestion_risques",
    "regulatory_updates": "gestion_reglementation",
    "regulatory": "gestion_reglementation",
}

_THEME_BY_SECTION: dict[str, str] = {
    "gestion_capital": "capital",
    "gestion_risques": "risque",
    "gestion_reglementation": "changement",
}

_REGULATORY_REF_RE = re.compile(
    r"\b(?:OSFI|BSIF|Bâle|Basel|TLAC|LCR|NSFR|CET1|Tier\s*1|Tier\s*2|Pilier\s*[123]|IFRS|IAS|NIIF|BISM|VaR)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\S*\d\S*\b")
_ROMAN_NUMERAL_RE = re.compile(r"\b[IVX]{1,4}\b")
_PERCENT_RE = re.compile(r"[%‰]+")
_BPS_RE = re.compile(r"\b(?:pb|pbs|bp|bps|point(?:s)?\s+de\s+base)\b", flags=re.IGNORECASE)
_PUNCT_SPACING_RE = re.compile(r"\s+([,;:.])")
_MULTISPACE_RE = re.compile(r"\s+")
_SEMANTIC_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bcadre de capacité totale d[’']absorption des pertes\b", flags=re.IGNORECASE),
        "un cadre renforcé d'absorption des pertes",
    ),
    (
        re.compile(r"\bligne directrice sur le levier\b", flags=re.IGNORECASE),
        "des exigences de levier",
    ),
    (
        re.compile(r"\bréformes de\s+[IVX]{1,4}\b", flags=re.IGNORECASE),
        "des réformes prudentielles",
    ),
    (
        re.compile(r"\bexigences?\s+réglementaires?\b", flags=re.IGNORECASE),
        "des exigences prudentielles",
    ),
    (
        re.compile(r"\bexigence\s+réglementaire\s+minimale\b", flags=re.IGNORECASE),
        "exigence minimale",
    ),
    (
        re.compile(r"\bBISM\b", flags=re.IGNORECASE),
        "les banques d'importance systémique",
    ),
    (
        re.compile(r"\bVaR\b", flags=re.IGNORECASE),
        "la mesure de risque de marché",
    ),
]


class TextAnalysisQualityError(RuntimeError):
    """Raised when a targeted text section cannot yield analyzable semantic units."""


@dataclass(slots=True)
class SemanticUnit:
    unit_id: str
    section_key: str
    theme: str
    semantic_text: str
    source_text: str
    source_block_ids: list[str]
    source_resolution: str
    evidence_pages: list[int]
    evidence_snippet: str


@dataclass(slots=True)
class ResolvedSection:
    section_key: str
    title: str
    start_page: int
    end_page: int
    anchor_page: int | None = None
    anchor_text: str | None = None
    anchor_bbox_norm: list[float] | None = None

    @property
    def pages(self) -> list[int]:
        return list(range(self.start_page, self.end_page + 1))


@dataclass(slots=True)
class PDFBlock:
    block_id: str
    page: int
    bbox_norm: list[float]
    text: str
    line_number: int
    block_type: str = "other"
    included: bool = False
    exclusion_reason: str = ""

    @property
    def y0(self) -> float:
        return float(self.bbox_norm[1])

    @property
    def y1(self) -> float:
        return float(self.bbox_norm[3])


@dataclass(slots=True)
class SectionAudit:
    section_key: str
    section_title: str
    start_page: int
    end_page: int
    anchor_page: int | None
    anchor_text: str | None
    anchor_bbox_norm: list[float] | None
    included_blocks: list[PDFBlock]
    excluded_blocks: list[PDFBlock]
    semantic_units: list[SemanticUnit]


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_to_payload(block: PDFBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "page": block.page,
        "bbox": [round(v, 6) for v in block.bbox_norm],
        "text": block.text,
        "block_type": block.block_type,
        "included": block.included,
        "exclusion_reason": block.exclusion_reason,
        "line_number": block.line_number,
    }


def _semantic_unit_to_payload(unit: SemanticUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "theme": unit.theme,
        "semantic_text": unit.semantic_text,
        "source_text": unit.source_text,
        "source_block_ids": list(unit.source_block_ids),
        "source_resolution": unit.source_resolution,
        "pages": list(unit.evidence_pages),
        "evidence_snippet": unit.evidence_snippet,
    }


def _sanitize_semantic_text(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    for pattern, replacement in _SEMANTIC_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    value = _REGULATORY_REF_RE.sub("", value)
    value = _NUMERIC_TOKEN_RE.sub("", value)
    value = _ROMAN_NUMERAL_RE.sub("", value)
    value = _PERCENT_RE.sub("", value)
    value = _BPS_RE.sub("", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\([^)]*\d[^)]*\)", "", value)
    value = re.sub(r"\s*[-–—]\s*", " ", value)
    value = re.sub(r"\b(?:Le|La|Les)\s+a\b", "La banque a", value)
    value = re.sub(r"\bLa Banque\b", "La banque", value)
    value = re.sub(r"\bLe Groupe\b", "La banque", value)
    value = re.sub(r"\bConseil d'administration\b", "gouvernance", value, flags=re.IGNORECASE)
    value = _PUNCT_SPACING_RE.sub(r"\1", value)
    value = _MULTISPACE_RE.sub(" ", value).strip(" ,;:.")
    return value.strip()


def _normalized_block_text(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9 ]+", "", value)
    return value.strip()


def _sanitize_explanation(text: str) -> str:
    value = _sanitize_semantic_text(text)
    return value[:1200]


def _is_new_major_or_allowed_moderate(triage: dict[str, Any]) -> bool:
    if not triage.get("is_relevant", False):
        return False
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    if impact == "MAJEUR":
        return True
    if impact != "MODERE":
        return False
    if triage.get("nouvelle_idee", False):
        return True
    signals = triage.get("signals") or {}
    return bool(signals.get("regulatory_reference_added") or signals.get("methodology_change"))


def _tokenize_semantic_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]{4,}", (text or "").lower())
        if token not in {"banque", "risque", "risques", "cadre", "mesure", "mesures"}
    }


def _lexical_shift_is_large(text_t1: str, text_t2: str) -> bool:
    tokens_t1 = _tokenize_semantic_text(text_t1)
    tokens_t2 = _tokenize_semantic_text(text_t2)
    if not tokens_t1 or not tokens_t2:
        return True
    overlap = len(tokens_t1 & tokens_t2)
    base = max(1, min(len(tokens_t1), len(tokens_t2)))
    return (overlap / base) < 0.45


def _compute_conservative_new_idea(change: dict[str, Any], triage: dict[str, Any]) -> bool:
    if not triage.get("is_relevant", False):
        return False

    diff_type = str(change.get("diff_type") or "").lower()
    return diff_type == "added" and bool(change.get("semantic_text_t2"))


def _sorted_sections(sections: dict[str, ResolvedSection]) -> list[ResolvedSection]:
    return sorted(
        sections.values(),
        key=lambda sec: (
            sec.start_page,
            float(sec.anchor_bbox_norm[1]) if sec.anchor_bbox_norm else 0.0,
            sec.section_key,
        ),
    )


def _next_section_by_key(sections: dict[str, ResolvedSection]) -> dict[str, ResolvedSection | None]:
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
    top = 0.0
    bottom = 1.0
    if (
        page_number == section.start_page
        and section.anchor_page == section.start_page
        and section.anchor_bbox_norm
    ):
        top = max(top, float(section.anchor_bbox_norm[3]))
    if (
        next_section is not None
        and page_number == section.end_page == next_section.start_page
        and next_section.anchor_page == next_section.start_page
        and next_section.anchor_bbox_norm
    ):
        bottom = min(bottom, float(next_section.anchor_bbox_norm[1]))
    if bottom <= top:
        return top, 1.0
    return top, bottom


def _page_block_candidates(pdf_path: Path, page_number: int) -> list[PDFBlock]:
    configure_mupdf_runtime(fitz)
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        page_rect = page.rect
        raw_blocks = page.get_text("blocks") or []
        ordered = sorted(
            raw_blocks,
            key=lambda block: (round(float(block[1]), 2), round(float(block[0]), 2)),
        )
        results: list[PDFBlock] = []
        for idx, block in enumerate(ordered, start=1):
            text = _MULTISPACE_RE.sub(" ", str(block[4] or "").replace("\n", " ").strip()).strip()
            if not text:
                continue
            x0, y0, x1, y1 = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
            bbox_norm = [
                max(0.0, min(1.0, x0 / page_rect.width)),
                max(0.0, min(1.0, y0 / page_rect.height)),
                max(0.0, min(1.0, x1 / page_rect.width)),
                max(0.0, min(1.0, y1 / page_rect.height)),
            ]
            results.append(
                PDFBlock(
                    block_id=f"p{page_number:03d}_b{idx:03d}",
                    page=page_number,
                    bbox_norm=bbox_norm,
                    text=text,
                    line_number=idx,
                )
            )
        return results
    finally:
        doc.close()


def _classify_block_type(block: PDFBlock, repeated_text_counts: dict[str, int]) -> str:
    norm = _normalized_block_text(block.text)
    if not norm:
        return "other"
    text = block.text.strip()
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    digits = re.findall(r"\d", text)
    digit_ratio = len(digits) / max(1, len(text))
    upper_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    repeated = repeated_text_counts.get(norm, 0)

    if repeated >= 2 and (block.y1 <= 0.12 or block.y0 >= 0.88):
        return "header_footer"
    if block.y0 >= 0.75 and re.match(r"^\s*(?:\(?\d+\)?|[*†‡]|note\b|source\b)", text, flags=re.IGNORECASE):
        return "footnote"
    if (
        digit_ratio >= 0.12
        or "\t" in text
        or "  " in text
        or ("|" in text)
        or (len(words) <= 8 and len(digits) >= 4)
        or (upper_ratio >= 0.7 and len(words) <= 12)
    ):
        return "table"
    if len(words) >= 8 and len(text) >= 45:
        return "narrative"
    return "other"


def _exclusion_reason_for_block(block_type: str, in_window: bool) -> str:
    if not in_window:
        return "outside_target_section"
    return {
        "table": "table_like_block",
        "footnote": "footnote",
        "header_footer": "header_footer",
        "other": "non_narrative_block",
    }.get(block_type, "")


def _concat_source_blocks(blocks: list[PDFBlock], block_ids: list[str]) -> str:
    lookup = {block.block_id: block for block in blocks}
    ordered = sorted(
        (lookup[block_id] for block_id in block_ids if block_id in lookup),
        key=lambda block: (block.page, block.line_number, block.y0),
    )
    return "\n".join(block.text for block in ordered).strip()


def _resolve_source_block_ids(
    candidate_blocks: list[PDFBlock],
    provided_ids: list[str],
    reference_text: str,
    semantic_text: str,
) -> tuple[list[str], str]:
    valid_ids = [block_id for block_id in provided_ids if any(block.block_id == block_id for block in candidate_blocks)]
    if valid_ids:
        return valid_ids, "matched"

    if not candidate_blocks:
        return [], "fallback"

    reference = _normalized_match_text(reference_text or semantic_text)
    semantic = _normalized_match_text(semantic_text)
    scored: list[tuple[float, str]] = []
    for block in candidate_blocks:
        norm_candidate = _normalized_match_text(block.text)
        score = 0.0
        if reference:
            score = max(score, SequenceMatcher(None, reference[:800], norm_candidate[:800]).ratio())
        if semantic:
            score = max(score, SequenceMatcher(None, semantic[:800], norm_candidate[:800]).ratio())
        scored.append((score, block.block_id))
    scored.sort(reverse=True)
    best_score = scored[0][0]
    fallback_ids = [
        block_id
        for score, block_id in scored
        if score >= 0.28 and score >= max(0.28, best_score - 0.08)
    ][:3]
    if not fallback_ids:
        fallback_ids = [scored[0][1]]
    return fallback_ids, "fallback"


def _validate_pages(raw_pages: Any, allowed_pages: set[int]) -> list[int]:
    pages: list[int] = []
    for value in raw_pages or []:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page in allowed_pages and page not in pages:
            pages.append(page)
    return pages


def _make_data_url(pdf_path: Path, page_number: int, dpi: int = 200) -> str:
    configure_mupdf_runtime(fitz)
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        payload = base64.b64encode(pix.tobytes("png")).decode("ascii")
        return f"data:image/png;base64,{payload}"
    finally:
        doc.close()


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


@lru_cache(maxsize=512)
def _page_blocks(pdf_path_str: str, page_number: int) -> tuple[str, ...]:
    configure_mupdf_runtime(fitz)
    doc = fitz.open(pdf_path_str)
    try:
        page = doc.load_page(page_number - 1)
        blocks = page.get_text("blocks") or []
        values: list[str] = []
        for block in blocks:
            text = str(block[4] or "").strip()
            text = _MULTISPACE_RE.sub(" ", text).strip()
            if len(text) >= 35:
                values.append(text)
        return tuple(values)
    finally:
        doc.close()


def _normalized_match_text(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9 ]+", "", value)
    return value.strip()


def _resolve_exact_source_text(pdf_path: Path, pages: list[int], reference_text: str, semantic_text: str) -> str:
    candidates: list[str] = []
    for page in pages:
        candidates.extend(_page_blocks(str(pdf_path), page))

    if not candidates:
        return (reference_text or semantic_text or "").strip()

    reference = _normalized_match_text(reference_text or semantic_text)
    semantic = _normalized_match_text(semantic_text)
    best_text = candidates[0]
    best_score = 0.0
    for candidate in candidates:
        norm_candidate = _normalized_match_text(candidate)
        score = 0.0
        if reference:
            score = max(score, SequenceMatcher(None, reference[:800], norm_candidate[:800]).ratio())
        if semantic:
            score = max(score, SequenceMatcher(None, semantic[:800], norm_candidate[:800]).ratio())
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text if best_score >= 0.28 else (reference_text or best_text or semantic_text).strip()


def _repeated_text_counts(page_blocks: dict[int, list[PDFBlock]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for blocks in page_blocks.values():
        seen_on_page: set[str] = set()
        for block in blocks:
            norm = _normalized_block_text(block.text)
            if not norm or norm in seen_on_page:
                continue
            seen_on_page.add(norm)
            counts[norm] = counts.get(norm, 0) + 1
    return counts


def _build_section_audit(
    *,
    section: ResolvedSection,
    next_section: ResolvedSection | None,
    page_blocks: dict[int, list[PDFBlock]],
    repeated_text_counts: dict[str, int],
) -> SectionAudit:
    included_blocks: list[PDFBlock] = []
    excluded_blocks: list[PDFBlock] = []
    for page in section.pages:
        blocks = page_blocks.get(page, [])
        top_cutoff, bottom_cutoff = _section_window_for_page(section, page, next_section)
        for block in blocks:
            section_block = PDFBlock(
                block_id=block.block_id,
                page=block.page,
                bbox_norm=list(block.bbox_norm),
                text=block.text,
                line_number=block.line_number,
            )
            midpoint = (block.y0 + block.y1) / 2.0
            in_window = top_cutoff <= midpoint < bottom_cutoff
            block_type = _classify_block_type(section_block, repeated_text_counts)
            section_block.block_type = block_type
            section_block.included = in_window and block_type == "narrative"
            section_block.exclusion_reason = "" if section_block.included else _exclusion_reason_for_block(block_type, in_window)
            if section_block.included:
                included_blocks.append(section_block)
            else:
                excluded_blocks.append(section_block)
    return SectionAudit(
        section_key=section.section_key,
        section_title=section.title,
        start_page=section.start_page,
        end_page=section.end_page,
        anchor_page=section.anchor_page,
        anchor_text=section.anchor_text,
        anchor_bbox_norm=section.anchor_bbox_norm,
        included_blocks=included_blocks,
        excluded_blocks=excluded_blocks,
        semantic_units=[],
    )


def _build_text_extraction_audit_payload(
    *,
    bank_code: str,
    year: int,
    quarter_label: str,
    pdf_path: Path,
    model: str,
    section_audits: list[SectionAudit],
) -> dict[str, Any]:
    return {
        "schema_version": TEXT_EXTRACTION_AUDIT_SCHEMA_VERSION,
        "artifact_type": "text_extraction_audit",
        "pipeline": "gpt4o_vision_unified",
        "bank_code": bank_code,
        "year": year,
        "quarter": quarter_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source_pdf": {
            "path": str(pdf_path),
            "sha256": _sha256_file(pdf_path),
        },
        "sections": [
            {
                "section_key": section.section_key,
                "section_title": section.section_title,
                "start_page": section.start_page,
                "end_page": section.end_page,
                "anchor_page": section.anchor_page,
                "anchor_text": section.anchor_text,
                "anchor_bbox_norm": section.anchor_bbox_norm,
                "included_blocks": [_block_to_payload(block) for block in section.included_blocks],
                "excluded_blocks": [_block_to_payload(block) for block in section.excluded_blocks],
                "semantic_units": [_semantic_unit_to_payload(unit) for unit in section.semantic_units],
            }
            for section in section_audits
        ],
        "summary": {
            "sections_count": len(section_audits),
            "included_blocks_count": sum(len(section.included_blocks) for section in section_audits),
            "excluded_blocks_count": sum(len(section.excluded_blocks) for section in section_audits),
            "semantic_units_count": sum(len(section.semantic_units) for section in section_audits),
        },
    }


def _build_openai_client():
    from openai import OpenAI

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY absent: le pipeline texte GPT-first ne peut pas s'exécuter.")
    return OpenAI(api_key=api_key)


def _call_json_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    payload = response.choices[0].message.content or "{}"
    return json.loads(payload)


def _resolve_sections(pdf_path: Path, bank_code: str) -> dict[str, ResolvedSection]:
    mapping = locate_sections_in_pdf(str(pdf_path), bank_code.lower())
    sections: dict[str, ResolvedSection] = {}
    for item in getattr(mapping, "sections", []) or []:
        canonical = canonicalize_section(getattr(item, "section_type", ""))
        section_key = _CANONICAL_TO_TEXT_KEY.get(canonical)
        if not section_key or section_key in sections:
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


def _extract_semantic_units_from_chunk(
    *,
    client: Any,
    model: str,
    pdf_path: Path,
    section_audit: SectionAudit,
    page_numbers: list[int],
) -> list[SemanticUnit]:
    candidate_blocks = [
        block
        for block in section_audit.included_blocks
        if block.page in set(page_numbers)
    ]
    if not candidate_blocks:
        return []

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Analyse ces pages de rapport bancaire et retourne uniquement du JSON.\n"
                "Objectif: extraire uniquement des unités sémantiques narratives qui parlent"
                " des risques, du capital, des stratégies ou des changements significatifs.\n"
                "Exclus strictement: tableaux, footnotes, headers/footers, valeurs numériques,"
                " références réglementaires explicites, titres hors périmètre.\n"
                "Tu dois t'appuyer sur la liste des blocs narratifs déjà filtrés ci-dessous."
                " N'utilise que leurs block_ids comme preuve source.\n"
                "Quand un passage cite un cadre, un acronyme prudentiel ou une ligne directrice,"
                " reformule-le en langage métier générique au lieu de le nommer explicitement.\n"
                "Nettoie le texte final pour qu'il soit fluide, uniquement sémantique et sans chiffres.\n"
                "Les pages sont fournies dans cet ordre: "
                f"{page_numbers}.\n"
                'Réponds sous la forme {"units":[{"semantic_text": "...", "pages":[...],'
                ' "evidence_snippet":"...", "theme":"risque|capital|strategie|changement",'
                ' "source_block_ids":["p001_b001"]}]}.\n'
                f"Section: {section_audit.section_key}\n"
                f"Blocs narratifs filtrés:\n{_json_dumps([_block_to_payload(block) for block in candidate_blocks])}\n"
                "Omet tout élément vide ou purement rédactionnel."
            ),
        }
    ]
    allowed_pages = set(page_numbers)
    for page in page_numbers:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _make_data_url(pdf_path, page), "detail": "high"},
            }
        )

    raw = _call_json_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un analyste senior des rapports bancaires. "
                    "Tu extrais uniquement le sens utile, sans bruit éditorial."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=6000,
    )

    units: list[SemanticUnit] = []
    for idx, item in enumerate(raw.get("units") or [], start=1):
        semantic_text = _sanitize_semantic_text(str(item.get("semantic_text") or ""))
        if not semantic_text:
            continue
        pages = _validate_pages(item.get("pages"), allowed_pages)
        evidence_snippet = str(item.get("evidence_snippet") or "").strip()[:800]
        theme = str(item.get("theme") or _THEME_BY_SECTION.get(section_audit.section_key, "changement")).strip().lower()
        provided_ids = [
            str(value).strip()
            for value in (item.get("source_block_ids") or [])
            if str(value).strip()
        ]
        source_block_ids, source_resolution = _resolve_source_block_ids(
            candidate_blocks=candidate_blocks,
            provided_ids=provided_ids,
            reference_text=evidence_snippet,
            semantic_text=semantic_text,
        )
        source_text = _concat_source_blocks(candidate_blocks, source_block_ids)
        source_pages = sorted({block.page for block in candidate_blocks if block.block_id in source_block_ids})
        units.append(
            SemanticUnit(
                unit_id=f"{section_audit.section_key}_chunk_{page_numbers[0]}_{idx:03d}",
                section_key=section_audit.section_key,
                theme=theme or _THEME_BY_SECTION.get(section_audit.section_key, "changement"),
                semantic_text=semantic_text,
                source_text=source_text,
                source_block_ids=source_block_ids,
                source_resolution=source_resolution,
                evidence_pages=source_pages or pages or [page_numbers[0]],
                evidence_snippet=evidence_snippet or semantic_text,
            )
        )
    return units


def _dedupe_units(units: list[SemanticUnit]) -> list[SemanticUnit]:
    unique: list[SemanticUnit] = []
    seen: set[tuple[str, str]] = set()
    for unit in units:
        key = (unit.section_key, unit.semantic_text.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(unit)
    for idx, unit in enumerate(unique, start=1):
        unit.unit_id = f"{unit.section_key}_unit_{idx:03d}"
    return unique


def _extract_semantic_units_for_pdf(
    *,
    client: Any,
    model: str,
    pdf_path: Path,
    sections: dict[str, ResolvedSection],
) -> tuple[dict[str, list[SemanticUnit]], list[SectionAudit]]:
    section_order = _next_section_by_key(sections)
    unique_pages = sorted({page for section in sections.values() for page in section.pages})
    page_blocks = {page: _page_block_candidates(pdf_path, page) for page in unique_pages}
    repeated_counts = _repeated_text_counts(page_blocks)
    extracted: dict[str, list[SemanticUnit]] = {}
    audits: list[SectionAudit] = []
    for section_key, section in sections.items():
        section_audit = _build_section_audit(
            section=section,
            next_section=section_order.get(section_key),
            page_blocks=page_blocks,
            repeated_text_counts=repeated_counts,
        )
        units: list[SemanticUnit] = []
        for chunk in _chunked(section.pages, chunk_size=4):
            units.extend(
                _extract_semantic_units_from_chunk(
                    client=client,
                    model=model,
                    pdf_path=pdf_path,
                    section_audit=section_audit,
                    page_numbers=chunk,
                )
            )
        units = _dedupe_units(units)
        if not units:
            raise TextAnalysisQualityError(
                f"Section ciblée vide après nettoyage sémantique: {section.section_key} ({pdf_path})"
            )
        extracted[section_key] = units
        section_audit.semantic_units = units
        audits.append(section_audit)
    return extracted, audits


def _serialize_units(units: list[SemanticUnit]) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": unit.unit_id,
            "semantic_text": unit.semantic_text,
            "source_text": unit.source_text,
            "source_block_ids": list(unit.source_block_ids),
            "source_resolution": unit.source_resolution,
            "theme": unit.theme,
            "pages": unit.evidence_pages,
            "evidence_snippet": unit.evidence_snippet,
        }
        for unit in units
    ]


def _compare_section_units(
    *,
    client: Any,
    model: str,
    section_key: str,
    units_t1: list[SemanticUnit],
    units_t2: list[SemanticUnit],
) -> list[dict[str, Any]]:
    lookup_t1 = {unit.unit_id: unit for unit in units_t1}
    lookup_t2 = {unit.unit_id: unit for unit in units_t2}
    payload = {
        "section_key": section_key,
        "t1_units": _serialize_units(units_t1),
        "t2_units": _serialize_units(units_t2),
    }
    raw = _call_json_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu alignes des idées entre deux trimestres de rapport bancaire. "
                    "Décide explicitement si T1 et T2 parlent de la même idée ou non."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Compare ces unités sémantiques et retourne uniquement du JSON.\n"
                    'Format: {"changes":[{"diff_type":"unchanged|modified|added|removed",'
                    ' "unit_id_t1":"...", "unit_id_t2":"...", "change_summary":"..."}]}.\n'
                    "Unchanged signifie même idée malgré reformulation. "
                    "Modified signifie même idée mais vraie évolution métier. "
                    "Added et removed signifient nouvelle idée ou disparition d'idée.\n"
                    f"{_json_dumps(payload)}"
                ),
            },
        ],
        max_tokens=6000,
    )

    validated: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("changes") or [], start=1):
        diff_type = str(item.get("diff_type") or "").strip().lower()
        if diff_type not in {"unchanged", "modified", "added", "removed"}:
            continue
        unit_t1 = lookup_t1.get(str(item.get("unit_id_t1") or "").strip())
        unit_t2 = lookup_t2.get(str(item.get("unit_id_t2") or "").strip())
        if diff_type in {"unchanged", "modified"} and (unit_t1 is None or unit_t2 is None):
            continue
        if diff_type == "added" and unit_t2 is None:
            continue
        if diff_type == "removed" and unit_t1 is None:
            continue
        validated.append(
            {
                "change_id": f"{section_key}_change_{idx:03d}",
                "section_key": section_key,
                "diff_type": diff_type,
                "semantic_text_t1": unit_t1.semantic_text if unit_t1 else "",
                "semantic_text_t2": unit_t2.semantic_text if unit_t2 else "",
                "source_text_t1": unit_t1.source_text if unit_t1 else "",
                "source_text_t2": unit_t2.source_text if unit_t2 else "",
                "source_block_ids_t1": list(unit_t1.source_block_ids) if unit_t1 else [],
                "source_block_ids_t2": list(unit_t2.source_block_ids) if unit_t2 else [],
                "source_refs_t1": list(unit_t1.source_block_ids) if unit_t1 else [],
                "source_refs_t2": list(unit_t2.source_block_ids) if unit_t2 else [],
                "pages_t1": list(unit_t1.evidence_pages) if unit_t1 else [],
                "pages_t2": list(unit_t2.evidence_pages) if unit_t2 else [],
                "source_resolution_t1": unit_t1.source_resolution if unit_t1 else "",
                "source_resolution_t2": unit_t2.source_resolution if unit_t2 else "",
                "evidence_t1": {
                    "pages": unit_t1.evidence_pages if unit_t1 else [],
                    "snippet": unit_t1.evidence_snippet if unit_t1 else "",
                },
                "evidence_t2": {
                    "pages": unit_t2.evidence_pages if unit_t2 else [],
                    "snippet": unit_t2.evidence_snippet if unit_t2 else "",
                },
                "change_summary": _sanitize_explanation(str(item.get("change_summary") or "")),
            }
        )
    return validated


def _default_triage() -> dict[str, Any]:
    return {
        "is_relevant": False,
        "category": "COSMETIQUE",
        "impact_level": "MINEUR",
        "risk_type": "autre",
        "relevance_score": "FAIBLE",
        "risk_level": "FAIBLE",
        "explanation": "",
        "impact_description": "",
        "action_requise": "aucune",
        "reference_reglementaire": "",
        "nouvelle_idee": False,
        "confidence": 0.0,
        "source": "gpt4o_triage",
        "signals": {
            "regulatory_reference_added": False,
            "methodology_change": False,
            "tone_changed": False,
            "forward_looking": False,
            "quantitative_changed": False,
        },
    }


def _triage_section_changes(
    *,
    client: Any,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not changes:
        return []
    triage_inputs = []
    for idx, change in enumerate(changes, start=1):
        triage_inputs.append(
            {
                "change_index": idx,
                "diff_type": change["diff_type"],
                "semantic_text_t1": change.get("semantic_text_t1", ""),
                "semantic_text_t2": change.get("semantic_text_t2", ""),
                "change_summary": change.get("change_summary", ""),
            }
        )
    raw = _call_json_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu fais un triage métier ultra-sélectif des changements de rapports bancaires. "
                    "Tu ne gardes que les changements vraiment majeurs, ou les modérés "
                    "qui introduisent une idée réellement nouvelle."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Retourne uniquement du JSON.\n"
                    'Format: {"triages":[{"change_index":1,"is_relevant":true,'
                    ' "category":"REGLEMENTAIRE|RISQUE|CAPITAL|STRUCTURE|COSMETIQUE",'
                    ' "impact_level":"MAJEUR|MODERE|MINEUR",'
                    ' "action_requise":"escalade|investigation|confirmation|information|aucune",'
                    ' "nouvelle_idee":true, "explanation":"...", "impact_description":"...",'
                    ' "risk_type":"credit|marche|liquidite|capital|conformite|autre",'
                    ' "signals":{"regulatory_reference_added":false,"methodology_change":false,'
                    ' "tone_changed":false,"forward_looking":false,"quantitative_changed":false}}]}.\n'
                    "Considère rédactionnel/cosmétique par défaut. "
                    "Un changement modéré ne doit être pertinent que s'il introduit "
                    "une nouvelle règle, contrainte, nuance de risque ou idée métier.\n"
                    "N'utilise pas d'acronymes prudentiels ni de références réglementaires explicites "
                    "dans l'explication; reformule en langage métier générique.\n"
                    f"Section: {section_key}\n{_json_dumps(triage_inputs)}"
                ),
            },
        ],
        max_tokens=5000,
    )

    triage_map: dict[int, dict[str, Any]] = {}
    for item in raw.get("triages") or []:
        try:
            idx = int(item.get("change_index"))
        except (TypeError, ValueError):
            continue
        triage = _default_triage()
        triage.update(
            {
                "is_relevant": bool(item.get("is_relevant", False)),
                "category": str(item.get("category") or "COSMETIQUE").upper(),
                "impact_level": str(item.get("impact_level") or "MINEUR").upper(),
                "risk_type": str(item.get("risk_type") or "autre").lower(),
                "explanation": _sanitize_explanation(str(item.get("explanation") or "")),
                "impact_description": _sanitize_explanation(str(item.get("impact_description") or "")),
                "action_requise": str(item.get("action_requise") or "aucune").lower(),
                "nouvelle_idee": bool(item.get("nouvelle_idee", False)),
                "source": "gpt4o_triage",
                "signals": {
                    "regulatory_reference_added": bool(
                        (item.get("signals") or {}).get("regulatory_reference_added", False)
                    ),
                    "methodology_change": bool((item.get("signals") or {}).get("methodology_change", False)),
                    "tone_changed": bool((item.get("signals") or {}).get("tone_changed", False)),
                    "forward_looking": bool((item.get("signals") or {}).get("forward_looking", False)),
                    "quantitative_changed": bool((item.get("signals") or {}).get("quantitative_changed", False)),
                },
            }
        )
        triage_map[idx] = triage

    enriched: list[dict[str, Any]] = []
    for idx, change in enumerate(changes, start=1):
        triage = triage_map.get(idx, _default_triage())
        triage["nouvelle_idee"] = _compute_conservative_new_idea(change, triage)
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return enriched


def _build_global_summary(section_comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    all_changes = [
        block
        for section in section_comparisons
        for block in (section.get("block_comparisons") or [])
    ]
    by_impact: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_action: dict[str, int] = {}
    highlights: list[str] = []

    for change in all_changes:
        triage = change.get("genai_triage") or {}
        impact = str(triage.get("impact_level") or "MINEUR").upper()
        category = str(triage.get("category") or "INCONNU").upper()
        action = str(triage.get("action_requise") or "aucune").lower()
        by_impact[impact] = by_impact.get(impact, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
        summary = str(change.get("change_summary") or "").strip()
        if summary and len(highlights) < 5:
            highlights.append(summary)

    overview = (
        "Aucun changement textuel majeur retenu."
        if not all_changes
        else f"{len(all_changes)} changement(s) textuel(s) majeur(s) ou modéré(s) réellement nouveaux retenus."
    )
    pertinence = "FAIBLE"
    if by_impact.get("MAJEUR", 0) >= 3:
        pertinence = "ELEVEE"
    elif all_changes:
        pertinence = "MOYENNE"

    return {
        "executive_overview": overview,
        "key_highlights": highlights,
        "pertinence_globale": pertinence,
        "counts": {
            "total": len(all_changes),
            "total_relevant": len(all_changes),
            "by_impact": by_impact,
            "by_category": by_category,
            "by_action": by_action,
        },
    }


def run_text_analysis_pipeline(
    *,
    bank_code: str,
    year_current: int,
    quarter_current: str,
    pdf_previous: Path,
    pdf_current: Path,
    out_root: Path,
    model: str = "gpt-4o",
    allowed_section_keys: set[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run the unified GPT-first text pipeline and persist ``text_comparison.json``."""
    quarter_current = normalize_quarter(quarter_current)
    year_previous, quarter_previous = resolve_previous_quarter(year_current, quarter_current)
    bank_code = bank_code.lower()

    client = _build_openai_client()
    sections_previous = _resolve_sections(pdf_previous, bank_code)
    sections_current = _resolve_sections(pdf_current, bank_code)
    section_keys = sorted(set(sections_previous) | set(sections_current))
    if allowed_section_keys is not None:
        section_keys = [key for key in section_keys if key in allowed_section_keys]
    if not section_keys:
        raise TextAnalysisQualityError("Aucune section texte ciblée localisée dans les rapports.")

    semantic_previous, audits_previous = _extract_semantic_units_for_pdf(
        client=client,
        model=model,
        pdf_path=pdf_previous,
        sections={key: sections_previous[key] for key in section_keys if key in sections_previous},
    )
    semantic_current, audits_current = _extract_semantic_units_for_pdf(
        client=client,
        model=model,
        pdf_path=pdf_current,
        sections={key: sections_current[key] for key in section_keys if key in sections_current},
    )

    section_comparisons: list[dict[str, Any]] = []
    for section_key in section_keys:
        changes = _compare_section_units(
            client=client,
            model=model,
            section_key=section_key,
            units_t1=semantic_previous.get(section_key, []),
            units_t2=semantic_current.get(section_key, []),
        )
        non_unchanged = [change for change in changes if change.get("diff_type") != "unchanged"]
        enriched = _triage_section_changes(
            client=client,
            model=model,
            section_key=section_key,
            changes=non_unchanged,
        )
        all_changes = [dict(change) for change in enriched]
        retained = [change for change in enriched if _is_new_major_or_allowed_moderate(change["genai_triage"])]
        section_comparisons.append(
            {
                "section_key": section_key,
                "section_title": _SECTION_LABELS.get(section_key, section_key),
                "block_comparisons": retained,
                "all_block_comparisons": all_changes,
                "summary": {
                    "retained_changes": len(retained),
                    "all_changes": len(all_changes),
                    "pages_previous": [s.start_page for s in [sections_previous.get(section_key)] if s]
                    + [s.end_page for s in [sections_previous.get(section_key)] if s],
                    "pages_current": [s.start_page for s in [sections_current.get(section_key)] if s]
                    + [s.end_page for s in [sections_current.get(section_key)] if s],
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": UNIFIED_TEXT_SCHEMA_VERSION,
        "artifact_type": "text_comparison",
        "pipeline": "gpt4o_vision_unified",
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": f"{year_previous}_{quarter_previous}",
        "year_current": year_current,
        "quarter_current": f"{year_current}_{quarter_current}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "extraction_artifact_t1": f"text_extraction_{year_previous}_{quarter_previous}.json",
        "extraction_artifact_t2": f"text_extraction_{year_current}_{quarter_current}.json",
        "section_comparisons": section_comparisons,
    }
    payload["global_summary"] = _build_global_summary(section_comparisons)
    payload["all_changes_summary"] = _build_global_summary(
        [
            {
                "section_key": section["section_key"],
                "block_comparisons": section.get("all_block_comparisons") or [],
            }
            for section in section_comparisons
        ]
    )

    out_path = get_text_comparison_path(
        out_root=out_root,
        bank_code=bank_code,
        year_t2=year_current,
        quarter_t2=quarter_current,
        year_t1=year_previous,
        quarter_t1=quarter_previous,
    )
    out_dir = out_path.parent
    previous_extraction_path = get_text_extraction_audit_path(
        out_dir,
        f"{year_previous}_{quarter_previous}",
    )
    current_extraction_path = get_text_extraction_audit_path(
        out_dir,
        f"{year_current}_{quarter_current}",
    )
    write_text_extraction_audit(
        _build_text_extraction_audit_payload(
            bank_code=bank_code,
            year=year_previous,
            quarter_label=f"{year_previous}_{quarter_previous}",
            pdf_path=pdf_previous,
            model=model,
            section_audits=audits_previous,
        ),
        previous_extraction_path,
    )
    write_text_extraction_audit(
        _build_text_extraction_audit_payload(
            bank_code=bank_code,
            year=year_current,
            quarter_label=f"{year_current}_{quarter_current}",
            pdf_path=pdf_current,
            model=model,
            section_audits=audits_current,
        ),
        current_extraction_path,
    )
    write_text_comparison(payload, out_path)
    return payload, out_path
