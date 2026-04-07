"""Pipeline texte canonique GPT-first.

Ce module remplace la chaîne extraction + alignement heuristique + diff/triage
par un seul orchestrateur qui:

1. localise les sections texte dans les deux PDFs,
2. extrait des unités sémantiques propres via GPT-4o Vision,
3. compare explicitement T1 vs T2 via GPT-4o,
4. trie les changements métiers,
5. ne conserve que les changements vraiment majeurs.

Le seul artefact public reste ``text_comparison.json``.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
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
    evidence_pages: list[int]
    evidence_snippet: str


@dataclass(slots=True)
class ResolvedSection:
    section_key: str
    title: str
    start_page: int
    end_page: int

    @property
    def pages(self) -> list[int]:
        return list(range(self.start_page, self.end_page + 1))


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


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


def _should_keep_for_expert_excel(triage: dict[str, Any]) -> bool:
    if not triage.get("is_relevant", False):
        return False
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    return impact in {"MAJEUR", "MODERE"}


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

    impact = str(triage.get("impact_level") or "MINEUR").upper()
    diff_type = str(change.get("diff_type") or "").lower()
    category = str(triage.get("category") or "INCONNU").upper()
    signals = triage.get("signals") or {}
    regulatory = bool(signals.get("regulatory_reference_added", False))
    methodology = bool(signals.get("methodology_change", False))
    text_t1 = str(change.get("semantic_text_t1") or "")
    text_t2 = str(change.get("semantic_text_t2") or "")
    semantic_shift = _lexical_shift_is_large(text_t1, text_t2)

    if diff_type in {"added", "removed"}:
        return impact == "MAJEUR" or regulatory

    if diff_type == "modified":
        if impact != "MAJEUR":
            return False
        if regulatory:
            return True
        return methodology and semantic_shift and category in {"RISQUE", "CAPITAL", "REGLEMENTAIRE"}

    return False


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
        )
    return sections


def _extract_semantic_units_from_chunk(
    *,
    client: Any,
    model: str,
    pdf_path: Path,
    section: ResolvedSection,
    page_numbers: list[int],
) -> list[SemanticUnit]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Analyse ces pages de rapport bancaire et retourne uniquement du JSON.\n"
                "Objectif: extraire uniquement des unités sémantiques narratives qui parlent"
                " des risques, du capital, des stratégies ou des changements significatifs.\n"
                "Exclus strictement: tableaux, footnotes, headers/footers, valeurs numériques,"
                " références réglementaires explicites, titres hors périmètre.\n"
                "Quand un passage cite un cadre, un acronyme prudentiel ou une ligne directrice,"
                " reformule-le en langage métier générique au lieu de le nommer explicitement.\n"
                "Nettoie le texte final pour qu'il soit fluide, uniquement sémantique et sans chiffres.\n"
                "Les pages sont fournies dans cet ordre: "
                f"{page_numbers}.\n"
                'Réponds sous la forme {"units":[{"semantic_text": "...", "pages":[...],'
                ' "evidence_snippet":"...", "theme":"risque|capital|strategie|changement"}]}.\n'
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
        theme = str(item.get("theme") or _THEME_BY_SECTION.get(section.section_key, "changement")).strip().lower()
        units.append(
            SemanticUnit(
                unit_id=f"{section.section_key}_chunk_{page_numbers[0]}_{idx:03d}",
                section_key=section.section_key,
                theme=theme or _THEME_BY_SECTION.get(section.section_key, "changement"),
                semantic_text=semantic_text,
                evidence_pages=pages or [page_numbers[0]],
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
) -> dict[str, list[SemanticUnit]]:
    extracted: dict[str, list[SemanticUnit]] = {}
    for section_key, section in sections.items():
        units: list[SemanticUnit] = []
        for chunk in _chunked(section.pages, chunk_size=4):
            units.extend(
                _extract_semantic_units_from_chunk(
                    client=client,
                    model=model,
                    pdf_path=pdf_path,
                    section=section,
                    page_numbers=chunk,
                )
            )
        units = _dedupe_units(units)
        if not units:
            raise TextAnalysisQualityError(
                f"Section ciblée vide après nettoyage sémantique: {section.section_key} ({pdf_path})"
            )
        extracted[section_key] = units
    return extracted


def _serialize_units(units: list[SemanticUnit]) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": unit.unit_id,
            "semantic_text": unit.semantic_text,
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

    semantic_previous = _extract_semantic_units_for_pdf(
        client=client,
        model=model,
        pdf_path=pdf_previous,
        sections={key: sections_previous[key] for key in section_keys if key in sections_previous},
    )
    semantic_current = _extract_semantic_units_for_pdf(
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
        expert_changes = [change for change in enriched if _should_keep_for_expert_excel(change["genai_triage"])]
        retained = [change for change in enriched if _is_new_major_or_allowed_moderate(change["genai_triage"])]
        section_comparisons.append(
            {
                "section_key": section_key,
                "section_title": _SECTION_LABELS.get(section_key, section_key),
                "block_comparisons": retained,
                "expert_block_comparisons": expert_changes,
                "summary": {
                    "retained_changes": len(retained),
                    "expert_changes": len(expert_changes),
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
        "section_comparisons": section_comparisons,
    }
    payload["global_summary"] = _build_global_summary(section_comparisons)
    payload["expert_summary"] = _build_global_summary(
        [
            {
                "section_key": section["section_key"],
                "block_comparisons": section.get("expert_block_comparisons") or [],
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
    write_text_comparison(payload, out_path)
    return payload, out_path
