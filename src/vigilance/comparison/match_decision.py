"""
Logique de decision MATCH / PROBABLE / NO_MATCH avec gating et titres generiques.
"""

from __future__ import annotations

from typing import Any, Optional

from vigilance.comparison.hard_negative_checks import check_hard_negative
from vigilance.comparison.indicator_match_helpers import compute_robust_indicator_score_from_signals
from vigilance.comparison.match_signals import count_strong_signals
from vigilance.utils.matching_normalizer import is_generic_title

try:
    from vigilance.config import get_matching_thresholds
except ImportError:

    def get_matching_thresholds() -> dict:
        return {}


def _get_threshold(key: str, default: float) -> float:
    try:
        t = get_matching_thresholds()
        return float(t.get(key, default))
    except Exception:
        return default


def _robust_indicator_score(signals: dict[str, Any]) -> float:
    """Canonical robust indicator score from signals dict."""
    return compute_robust_indicator_score_from_signals(signals)


def _corroborated_table_number(signals: dict[str, Any]) -> float:
    """
    Table number contribution (0, partial, or 1) based on robust indicator support only.
    Full value only when robust_indicator_score >= full_threshold; else partial or near-zero.
    """
    if not signals.get("table_number_match"):
        return 0.0
    robust = _robust_indicator_score(signals)
    full_min = _get_threshold("indicator_robust_full_match_min", 0.40)
    partial_min = _get_threshold("indicator_robust_partial_match_min", 0.20)
    if robust >= full_min:
        return 1.0
    if robust >= partial_min:
        return 0.5
    return 0.15


def should_reject_prefix_only_structure(signals: dict[str, Any]) -> bool:
    """
    Reject when prefix_ratio high, lcs_ratio low, size_ratio poor (prefix-only pattern).
    """
    prefix_min = _get_threshold("indicator_prefix_reject_min", 0.70)
    lcs_max = _get_threshold("indicator_lcs_reject_max", 0.25)
    size_max = _get_threshold("indicator_size_ratio_reject_max", 0.40)
    prefix_ratio = float(signals.get("indicator_prefix_ratio", 0) or 0)
    lcs_ratio = float(signals.get("indicator_lcs_ratio", 0) or 0)
    size_ratio = float(signals.get("indicator_size_ratio", 1.0) or 1.0)
    return (
        prefix_ratio >= prefix_min
        and lcs_ratio <= lcs_max
        and size_ratio <= size_max
    )


def compute_composite_score(signals: dict[str, Any]) -> float:
    """
    Score composite pour classer les candidats.

    Uses corroborated table number and robust indicator score when new signals present.
    """
    tn = _corroborated_table_number(signals)
    ho = signals.get("header_overlap", 0)
    if "indicator_lcs_ratio" in signals:
        io = _robust_indicator_score(signals)
    else:
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
    robust_ind = _robust_indicator_score(signals)
    composite_score = compute_composite_score(signals)
    strong_count = count_strong_signals(signals, has_headers)
    override_ind_min = _get_threshold("hard_negative_override_indicator_min", 0.30)
    hn = check_hard_negative(
        table_type_t1=table_type_t1,
        table_type_t2=table_type_t2,
        table_number_match=signals.get("table_number_match", False),
        header_overlap=signals.get("header_overlap", 0),
        headers_t1=headers_t1,
        headers_t2=headers_t2,
        robust_indicator_score=robust_ind,
        override_indicator_min=override_ind_min,
    )

    section_match = signals.get("section_match", False)
    section_state = str(signals.get("section_state", "unknown_present") or "unknown_present")
    generic_1 = is_generic_title(title_t1 or "", generic_titles)
    generic_2 = is_generic_title(title_t2 or "", generic_titles)
    title_generic = generic_1 or generic_2

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

    if should_reject_prefix_only_structure(signals):
        return {
            "decision": "NO_MATCH",
            "composite_score": round(composite_score, 4),
            "strong_signals_count": strong_count,
            "hard_negative_triggered": hn.triggered,
            "reason": "indicator_structure_mismatch",
        }

    if hn.triggered and not hn.can_override:
        decision = "NO_MATCH"
        reason_parts.append(f"hard_negative: {hn.reason}")
    elif section_match or section_state == "unknown_present":
        if composite_score >= table_match_score and not hn.triggered:
            if title_generic:
                corroborated_tn = _corroborated_table_number(signals)
                if corroborated_tn >= 0.5:
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
        containment_like = (
            robust_ind if "indicator_lcs_ratio" in signals else float(signals.get("indicator_overlap", 0.0) or 0.0)
        )
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
