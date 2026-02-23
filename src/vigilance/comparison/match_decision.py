"""
Logique de decision MATCH / PROBABLE / NO_MATCH avec gating et titres generiques.
"""

from __future__ import annotations

from typing import Any, Optional

from vigilance.comparison.hard_negative_checks import check_hard_negative
from vigilance.comparison.match_signals import count_strong_signals
from vigilance.utils.matching_normalizer import is_generic_title

try:
    from vigilance.config import get_matching_thresholds
except ImportError:
    get_matching_thresholds = lambda: {}


def _get_threshold(key: str, default: float) -> float:
    try:
        t = get_matching_thresholds()
        return float(t.get(key, default))
    except Exception:
        return default


def compute_composite_score(signals: dict[str, Any]) -> float:
    """
    Score composite pour classer les candidats.

    Priorite: table_number > header_overlap > indicator_overlap > title_similarity
    > section_match > page_distance (inverse).
    """
    tn = 1.0 if signals.get("table_number_match") else 0.0
    ho = signals.get("header_overlap", 0)
    io = signals.get("indicator_overlap", 0)
    ts = signals.get("title_similarity", 0)
    sm = 1.0 if signals.get("section_match") else 0.0
    pd = signals.get("page_distance", 99)
    page_score = max(0, 1.0 - pd / 15.0)
    score = (
        0.35 * tn
        + 0.25 * ho
        + 0.25 * io
        + 0.10 * ts
        + 0.03 * sm
        + 0.02 * page_score
    )
    section_state = str(signals.get("section_state", "") or "")
    if section_state == "unknown_present":
        penalty = _get_threshold("unknown_section_penalty", 0.15)
        score *= max(0.0, 1.0 - penalty)
    return score


def compute_decision(
    signals: dict[str, Any],
    table_type_t1: str,
    table_type_t2: str,
    title_t1: Optional[str] = None,
    title_t2: Optional[str] = None,
    headers_t1: Optional[list] = None,
    headers_t2: Optional[list] = None,
    has_headers: bool = True,
    generic_titles: Optional[set[str] | frozenset] = None,
) -> dict[str, Any]:
    """
    Determine la decision MATCH / PROBABLE / NO_MATCH.

    Args:
        signals: Resultat de compute_match_signals
        table_type_t1, table_type_t2: Types des tableaux
        title_t1, title_t2: Titres pour detection generique
        headers_t1, headers_t2: Pour hard negative
        has_headers: True si headers disponibles
        generic_titles: Liste titres generiques (optionnel)

    Returns:
        Dict avec decision, composite_score, strong_signals_count,
        hard_negative_triggered, reason
    """
    composite_score = compute_composite_score(signals)
    strong_count = count_strong_signals(signals, has_headers)
    hn = check_hard_negative(
        table_type_t1=table_type_t1,
        table_type_t2=table_type_t2,
        table_number_match=signals.get("table_number_match", False),
        header_overlap=signals.get("header_overlap", 0),
        headers_t1=headers_t1,
        headers_t2=headers_t2,
    )

    section_match = signals.get("section_match", False)
    section_state = str(signals.get("section_state", "unknown_present") or "unknown_present")
    generic_1 = is_generic_title(title_t1 or "", generic_titles)
    generic_2 = is_generic_title(title_t2 or "", generic_titles)
    title_generic = generic_1 or generic_2

    min_signals = _get_threshold("section_mismatch_min_signals", 2)
    min_signals_generic = _get_threshold("section_mismatch_min_signals_generic", 3)
    table_match_score = _get_threshold("table_match_score", 0.45)
    probable_band = _get_threshold("probable_band_width", 0.08)
    unknown_match_min_containment = _get_threshold("unknown_match_min_containment", 0.65)
    unknown_match_min_score = _get_threshold("unknown_match_min_score", 0.74)

    reason_parts = []
    decision = "NO_MATCH"

    # Hard business rule: known cross-section matching is forbidden.
    if section_state == "mismatch_known":
        return {
            "decision": "NO_MATCH",
            "composite_score": round(composite_score, 4),
            "strong_signals_count": strong_count,
            "hard_negative_triggered": hn.triggered,
            "reason": "cross_section_forbidden",
        }

    if hn.triggered and not hn.can_override:
        decision = "NO_MATCH"
        reason_parts.append(f"hard_negative: {hn.reason}")
    elif section_match or section_state == "unknown_present":
        if composite_score >= table_match_score and not hn.triggered:
            if title_generic:
                if signals.get("table_number_match") or signals.get("header_overlap", 0) >= 0.85:
                    decision = "MATCH"
                    reason_parts.append("section_match, generic_title override ok")
                elif composite_score >= table_match_score + 0.10:
                    decision = "MATCH"
                    reason_parts.append("section_match, generic_title, score high")
                else:
                    decision = "PROBABLE"
                    reason_parts.append("section_match, generic_title, needs validation")
            else:
                decision = "MATCH"
                reason_parts.append("section_match, score ok")
        elif composite_score >= table_match_score - probable_band:
            decision = "PROBABLE"
            reason_parts.append("section_match, score in probable band")

    if section_state == "unknown_present":
        containment_like = float(signals.get("indicator_overlap", 0.0) or 0.0)
        if decision == "MATCH" and (
            composite_score < unknown_match_min_score
            or containment_like < unknown_match_min_containment
        ):
            if composite_score >= table_match_score - probable_band:
                decision = "PROBABLE"
            else:
                decision = "NO_MATCH"
            reason_parts.append("unknown_section_penalized")
    return {
        "decision": decision,
        "composite_score": round(composite_score, 4),
        "strong_signals_count": strong_count,
        "hard_negative_triggered": hn.triggered,
        "reason": "; ".join(reason_parts),
    }


def build_output_json(
    t1_table_id: str,
    best: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construit le JSON de sortie strict pour l'audit."""
    return {
        "t1_table_id": t1_table_id,
        "best": best,
        "ranked_candidates": ranked_candidates,
    }
