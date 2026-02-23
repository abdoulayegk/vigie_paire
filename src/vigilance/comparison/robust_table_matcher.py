"""
Matcher robuste: match_best(t1, t2_candidates) -> JSON strict.

Enchaine: Normalisation -> table_type -> Hard Negative -> Signaux -> Decision -> Ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from vigilance.comparison.hard_negative_checks import check_hard_negative
from vigilance.comparison.match_decision import compute_composite_score, compute_decision
from vigilance.comparison.match_signals import compute_match_signals
from vigilance.comparison.table_type_classifier import classify_table_type

logger = logging.getLogger(__name__)

try:
    from vigilance.config import get_matching_thresholds
except ImportError:
    get_matching_thresholds = lambda: {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _get_table_id(obj: Any) -> str:
    return str(_get(obj, "table_id") or _get(obj, "table_key") or "")


def match_best(
    t1: Any,
    t2_candidates: list[Any],
    bank_code: Optional[str] = None,
    use_vision_validation: bool = False,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Evalue les candidats et retourne le meilleur match + classement.

    Args:
        t1: Tableau source (Tn)
        t2_candidates: Liste de candidats (Tn+1), Top-K
        bank_code: Code banque pour config
        use_vision_validation: Appeler Vision si PROBABLE (necessite images)
        api_key: Cle API pour Vision

    Returns:
        JSON strict avec t1_table_id, best, ranked_candidates
    """
    t1_id = _get_table_id(t1)
    t1_headers = _get(t1, "headers") or []
    t1_has_headers = bool(t1_headers)
    t1_title = _get(t1, "title") or ""
    t1_context_before = _get(t1, "context_before")
    t1_context_after = _get(t1, "context_after")

    type_result_t1 = classify_table_type(
        title=t1_title,
        headers=t1_headers,
        context_before=t1_context_before,
        context_after=t1_context_after,
        bank_code=bank_code,
    )

    generic_titles = None
    try:
        th = get_matching_thresholds()
        gt = th.get("generic_titles", [])
        if gt:
            generic_titles = frozenset(str(x).lower().strip() for x in gt)
    except Exception:
        pass

    candidates_with_scores: list[tuple[dict, float]] = []

    for t2 in t2_candidates:
        t2_id = _get_table_id(t2)
        t2_headers = _get(t2, "headers") or []
        t2_has_headers = bool(t2_headers)
        has_headers = t1_has_headers and t2_has_headers

        type_result_t2 = classify_table_type(
            title=_get(t2, "title"),
            headers=t2_headers,
            context_before=_get(t2, "context_before"),
            context_after=_get(t2, "context_after"),
            bank_code=bank_code,
        )

        signals = compute_match_signals(t1, t2, has_headers=has_headers)

        decision_result = compute_decision(
            signals=signals,
            table_type_t1=type_result_t1.table_type,
            table_type_t2=type_result_t2.table_type,
            title_t1=t1_title,
            title_t2=_get(t2, "title"),
            headers_t1=t1_headers if t1_headers else None,
            headers_t2=t2_headers if t2_headers else None,
            has_headers=has_headers,
            generic_titles=generic_titles,
        )

        entry = {
            "t2_table_id": t2_id,
            "decision": decision_result["decision"],
            "composite_score": decision_result["composite_score"],
            "table_type_t2": type_result_t2.table_type,
            "signals": signals,
            "strong_signals_count": decision_result["strong_signals_count"],
            "hard_negative_triggered": decision_result["hard_negative_triggered"],
            "reason": decision_result["reason"],
        }
        candidates_with_scores.append((entry, decision_result["composite_score"]))

    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

    ranked = []
    for entry, _ in candidates_with_scores[: max(5, len(candidates_with_scores))]:
        ranked.append({
            "t2_table_id": entry["t2_table_id"],
            "decision": entry["decision"],
            "composite_score": entry["composite_score"],
            "table_type_t2": entry["table_type_t2"],
            "strong_signals_count": entry["strong_signals_count"],
            "hard_negative_triggered": entry["hard_negative_triggered"],
            "notes": entry["reason"],
        })

    best_t2_id = None
    best_entry = None
    for entry, _ in candidates_with_scores:
        if entry["decision"] == "MATCH":
            best_t2_id = entry["t2_table_id"]
            best_entry = dict(entry)
            best_entry["table_type_t1"] = type_result_t1.table_type
            break
        if entry["decision"] == "PROBABLE" and best_entry is None:
            best_entry = dict(entry)
            best_t2_id = entry["t2_table_id"]
            best_entry["table_type_t1"] = type_result_t1.table_type

    if best_entry is None and candidates_with_scores:
        entry = candidates_with_scores[0][0]
        best_entry = dict(entry)
        best_entry["table_type_t1"] = type_result_t1.table_type
        best_t2_id = None

    if best_entry is None:
        best_entry = {
            "t2_table_id": None,
            "decision": "NO_MATCH",
            "composite_score": 0.0,
            "table_type_t1": type_result_t1.table_type,
            "table_type_t2": "unknown",
            "signals": {},
            "strong_signals_count": 0,
            "hard_negative_triggered": False,
            "reason": "no candidates",
        }
    else:
        best_entry["t2_table_id"] = best_t2_id

    # Vision validation: si decision PROBABLE et config active, appeler Vision
    best_entry["vision_reason"] = None
    if (
        best_entry["decision"] == "PROBABLE"
        and best_t2_id is not None
        and use_vision_validation
        and api_key
    ):
        try:
            th = get_matching_thresholds()
            only_on_probable = th.get("vision_validation_only_on_probable", True)
            if not only_on_probable:
                pass  # Skip if not only on probable (would run on more cases)
            else:
                # Trouver le t2 correspondant
                best_t2_obj = None
                for t2 in t2_candidates:
                    if _get_table_id(t2) == best_t2_id:
                        best_t2_obj = t2
                        break
                img1 = _get(t1, "image_base64")
                img2 = _get(best_t2_obj, "image_base64") if best_t2_obj else None
                if img1 and img2:
                    from vigilance.analysis.vision_table_matcher import VisionTableMatcher

                    matcher = VisionTableMatcher(api_key=api_key)
                    vision_result = matcher.match_tables(img1, img2)
                    min_conf = th.get("vision_validation_min_confidence", 80)
                    if vision_result.matched and vision_result.confidence >= min_conf:
                        best_entry["decision"] = "MATCH"
                        best_entry["vision_reason"] = (
                            f"Vision confirme: {vision_result.reason} (conf={vision_result.confidence})"
                        )
                    else:
                        best_entry["decision"] = "NO_MATCH"
                        best_entry["t2_table_id"] = None
                        best_entry["vision_reason"] = (
                            f"Vision rejette: {vision_result.reason} (conf={vision_result.confidence})"
                        )
        except Exception as e:
            logger.warning("Vision validation failed: %s", e)
            best_entry["vision_reason"] = f"Vision error: {e}"

    return {
        "t1_table_id": t1_id,
        "best": best_entry,
        "ranked_candidates": ranked,
    }
