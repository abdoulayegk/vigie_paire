"""
Calcul des signaux pour le matching de tableaux (reduction faux positifs).

Signaux retournes: table_number_match, title_similarity, header_overlap,
indicator_overlap (Jaccard, conserve pour compatibilite), plus les signaux
structurels quand disponibles: indicator_jaccard, indicator_containment_min,
indicator_lcs_ratio, indicator_size_ratio, indicator_prefix_ratio.

Signaux forts: table_number_match, title_similarity>=0.90, header_overlap>=0.85,
et pour les indicateurs: robust indicator score>=0.75 lorsque les signaux
structurels sont presents, sinon indicator_overlap>=0.75 (repli).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from difflib import SequenceMatcher

from vigilance.comparison.indicator_match_helpers import (
    compute_indicator_signals,
    compute_robust_indicator_score_from_signals,
)
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.matching_normalizer import normalize_for_matching

TABLE_NUMBER_PATTERNS = [
    r"TABLEAU[\s_]*(\d+[a-z]?)",
    r"TABLE[\s_]*(\d+[a-z]?)",
    r"\bT[\s_]*(\d+[a-z]?)",
    r"Tableau[\s_]*(\d+[a-z]?)",
]
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}


def extract_table_number(text: Optional[str], table_id: Optional[str] = None) -> str:
    """
    Extrait le numero de tableau depuis titre ou table_id.

    Supporte: TABLEAU 24, TABLEAU 11A, T5, T11A, Table 12.
    """
    del table_id  # Intentionally ignored: technical IDs must not drive table-number matching.
    src = text or ""
    if src:
        for pattern in TABLE_NUMBER_PATTERNS:
            match = re.search(pattern, src, re.IGNORECASE)
            if match:
                return (match.group(1) or "").strip().upper()
    return ""


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_match_signals(
    t1: Any,
    t2: Any,
    has_headers: bool = True,
) -> dict[str, Any]:
    """
    Calcule les signaux de matching entre deux tableaux.

    Args:
        t1: Tableau 1 (doit avoir: title, headers?, indicators?, section?, page?)
        t2: Tableau 2
        has_headers: True si les deux ont des headers non vides

    Returns:
        Dict avec table_number_match, title_similarity, header_overlap,
        indicator_overlap, section_match, page_distance
    """
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    title1 = _get(t1, "title") or ""
    title2 = _get(t2, "title") or ""
    headers1 = _get(t1, "headers") or []
    headers2 = _get(t2, "headers") or []
    indicators1 = _get(t1, "indicators") or _get(t1, "first_column_indicators") or []
    indicators2 = _get(t2, "indicators") or _get(t2, "first_column_indicators") or []
    section1 = _get(t1, "section") or ""
    section2 = _get(t2, "section") or ""
    page1 = int(_get(t1, "page") or _get(t1, "page_number") or 0)
    page2 = int(_get(t2, "page") or _get(t2, "page_number") or 0)
    num1 = extract_table_number(title1)
    num2 = extract_table_number(title2)
    table_number_match = bool(num1 and num2 and num1 == num2)

    norm_title1 = normalize_for_matching(title1, target="title")
    norm_title2 = normalize_for_matching(title2, target="title")
    if norm_title1 and norm_title2:
        title_similarity = SequenceMatcher(None, norm_title1, norm_title2).ratio()
    else:
        title_similarity = 0.0

    header_overlap = 0.0
    if headers1 and headers2 and has_headers:
        h1_tokens = set()
        for h in headers1:
            h1_tokens.update(normalize_for_matching(str(h), target="header").split())
        h2_tokens = set()
        for h in headers2:
            h2_tokens.update(normalize_for_matching(str(h), target="header").split())
        header_overlap = _jaccard(h1_tokens, h2_tokens)

    def _norm_indicators(items: list) -> set[str]:
        out = set()
        for item in items:
            if isinstance(item, str):
                text = item
            elif hasattr(item, "text"):
                text = getattr(item, "text", "") or ""
            elif isinstance(item, dict):
                text = item.get("text", "") or ""
            else:
                continue
            n = normalize_indicator_for_comparison(text)
            if n:
                out.add(n)
        return out

    def _norm_indicators_ordered(items: list) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, str):
                text = item
            elif hasattr(item, "text"):
                text = getattr(item, "text", "") or ""
            elif isinstance(item, dict):
                text = item.get("text", "") or ""
            else:
                continue
            n = normalize_indicator_for_comparison(text)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    ind1 = _norm_indicators(indicators1 if isinstance(indicators1, list) else [])
    ind2 = _norm_indicators(indicators2 if isinstance(indicators2, list) else [])
    if hasattr(t1, "indicator_norm_set") and t1.indicator_norm_set:
        ind1 = t1.indicator_norm_set
    if hasattr(t2, "indicator_norm_set") and t2.indicator_norm_set:
        ind2 = t2.indicator_norm_set
    indicator_overlap = _jaccard(ind1, ind2)

    ordered_ind1 = _norm_indicators_ordered(indicators1 if isinstance(indicators1, list) else [])
    ordered_ind2 = _norm_indicators_ordered(indicators2 if isinstance(indicators2, list) else [])
    if hasattr(t1, "indicator_norm_set") and t1.indicator_norm_set:
        ordered_ind1 = list(t1.indicator_norm_set)
    if hasattr(t2, "indicator_norm_set") and t2.indicator_norm_set:
        ordered_ind2 = list(t2.indicator_norm_set)
    ind_signals = compute_indicator_signals(ordered_ind1, ordered_ind2)

    section1_norm = normalize_for_matching(section1, target="section")
    section2_norm = normalize_for_matching(section2, target="section")
    section_state = "unknown_present"
    if (
        section1_norm
        and section2_norm
        and section1_norm not in UNKNOWN_SECTIONS
        and section2_norm not in UNKNOWN_SECTIONS
    ):
        section_state = "same_known" if section1_norm == section2_norm else "mismatch_known"
    section_match = bool(
        section1_norm
        and section2_norm
        and section1_norm not in UNKNOWN_SECTIONS
        and section2_norm not in UNKNOWN_SECTIONS
        and section1_norm == section2_norm
    )
    page_distance = abs(page1 - page2)

    result: dict[str, Any] = {
        "table_number_match": table_number_match,
        "title_similarity": title_similarity,
        "header_overlap": header_overlap,
        "indicator_overlap": indicator_overlap,
        "section_match": section_match,
        "section_state": section_state,
        "page_distance": page_distance,
    }
    result.update(ind_signals)
    return result


def count_strong_signals(signals: dict[str, Any], has_headers: bool = True) -> int:
    """
    Compte le nombre de signaux forts.

    Quand les signaux structurels sont presents (indicator_lcs_ratio etc.), le
    critere indicateur est le robust indicator score >= 0.75. Sinon on utilise
    indicator_overlap >= 0.75 pour compatibilite avec les anciens appels.
    """
    count = 0
    if signals.get("table_number_match"):
        count += 1
    if signals.get("title_similarity", 0) >= 0.90:
        count += 1
    if has_headers and signals.get("header_overlap", 0) >= 0.85:
        count += 1
    if "indicator_lcs_ratio" in signals:
        robust = compute_robust_indicator_score_from_signals(signals)
        if robust >= 0.75:
            count += 1
    elif signals.get("indicator_overlap", 0) >= 0.75:
        count += 1
    return count
