"""Controlled semantic table validation via GPT-4o.

Validates structural table matches after deterministic pairing in gray-zone
cases only. GPT never replaces structural matching; it acts as an optional
validator or unmatched-table reviewer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_BANKS = frozenset({"rbc", "bns", "td"})
_GRAY_ZONE_OVERLAP_THRESHOLD = 0.40
_DEFAULT_TOP_K = 3
_MAX_ANCHORS = 40

_SYSTEM_PROMPT = (
    "You are a regulatory audit expert specialized in Canadian bank quarterly disclosures. "
    "You must determine whether any candidate table corresponds to the same regulatory concept "
    "as the reference table. "
    "You must ignore: dates, numeric changes, punctuation, formatting differences, "
    "unit-only lines, and footnotes. "
    "You must choose at most one candidate or return null. "
    "You must respond ONLY in strict JSON with the following schema: "
    '{"best_match_id": "<candidate_id or null>", '
    '"decision": "match" | "no_match" | "unsure", '
    '"confidence": <float 0.0-1.0>, '
    '"reasons": ["<short reason 1>", ...], '
    '"risk_flags": ["<optional warning>", ...]}'
)


def is_bank_allowed(
    bank_code: str,
    allowed_banks: list[str] | frozenset[str] | None = None,
) -> bool:
    code = (bank_code or "").strip().lower()
    if allowed_banks is not None:
        codes = (
            {b.strip().lower() for b in allowed_banks}
            if isinstance(allowed_banks, list)
            else allowed_banks
        )
        return code in codes
    return code in _ALLOWED_BANKS


def _needs_semantic_validation(
    pair: dict[str, Any],
    indicator_overlap: float,
    suspicious_low_overlap: bool,
) -> bool:
    if indicator_overlap < _GRAY_ZONE_OVERLAP_THRESHOLD:
        return True
    if pair.get("rescue_type"):
        return True
    if suspicious_low_overlap:
        return True
    return False


def _needs_semantic_validation_unmatched(change_type: str) -> bool:
    return change_type in ("table_added", "table_removed")


def _table_summary(
    table: Any,
    *,
    structural_score: float = 0.0,
    indicator_overlap: float = 0.0,
) -> dict[str, Any]:
    indicators = list(getattr(table, "first_column_indicators", []) or [])
    return {
        "table_id": getattr(table, "table_id", ""),
        "title_clean": getattr(table, "title", "") or "",
        "section_path": getattr(table, "section", ""),
        "page": getattr(table, "page_pdf", 0),
        "anchors": indicators[:_MAX_ANCHORS],
        "structural_score": round(structural_score, 4),
        "indicator_overlap": round(indicator_overlap, 4),
        "fragmentation_detected": getattr(table, "fragmentation_detected", False),
    }


def _extract_top_k_candidates(
    t1_uid: str,
    strict_result: dict[str, Any],
    t2_by_uid: dict[str, Any],
    current_t2_uid: str | None,
    k: int = _DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    seen_uids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    if current_t2_uid and current_t2_uid in t2_by_uid:
        seen_uids.add(current_t2_uid)

    for entry in strict_result.get("debug_unmatched_candidates", []):
        if str(entry.get("t1_uid", "")) != t1_uid:
            continue
        for cand in entry.get("candidates", []):
            uid = str(cand.get("t2_uid", ""))
            if uid in seen_uids or uid not in t2_by_uid:
                continue
            seen_uids.add(uid)
            candidates.append(
                {
                    "t2_uid": uid,
                    "t2_table": t2_by_uid[uid],
                    "score": float(cand.get("score", 0.0) or 0.0),
                    "indicator_overlap": float(cand.get("indicator_overlap", 0.0) or 0.0),
                }
            )

    for pp in strict_result.get("probable_pairs", []):
        uid = str(pp.get("t2_uid", ""))
        if str(pp.get("t1_uid", "")) != t1_uid or uid in seen_uids or uid not in t2_by_uid:
            continue
        seen_uids.add(uid)
        candidates.append(
            {
                "t2_uid": uid,
                "t2_table": t2_by_uid[uid],
                "score": float(pp.get("score", 0.0) or 0.0),
                "indicator_overlap": float(pp.get("indicator_overlap", 0.0) or 0.0),
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:k]


def _build_prompt(reference: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    payload = {
        "reference_table": reference,
        "candidates": candidates,
    }
    return (
        "Compare the reference table to each candidate. "
        "Determine if any candidate represents the same regulatory table "
        "(same concept, same disclosure requirement). "
        "Respond ONLY in strict JSON.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _call_gpt(
    api_key: str,
    reference: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed; semantic judge disabled")
        return {"error": "openai_not_installed"}

    client = OpenAI(api_key=api_key)
    user_prompt = _build_prompt(reference, candidates)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("GPT-4o semantic judge API call failed: %s", exc)
        return {"error": str(exc)}

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("GPT-4o returned non-JSON: %s", raw_text[:200])
        return {"error": "non_json_response", "raw": raw_text[:500]}

    if not isinstance(parsed, dict):
        return {"error": "invalid_response_type", "raw": raw_text[:500]}

    return {
        "best_match_id": parsed.get("best_match_id"),
        "decision": parsed.get("decision", "unsure"),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reasons": parsed.get("reasons", []),
        "risk_flags": parsed.get("risk_flags", []),
        "raw": raw_text,
    }


def _apply_guard_rails(
    gpt_result: dict[str, Any],
    top_k_uids: set[str],
    structural_score: float,
    indicator_overlap: float,
) -> dict[str, Any]:
    if "error" in gpt_result:
        return {
            "final_decision": "structural_fallback",
            "final_match_id": None,
            "guard_action": "gpt_error_fallback",
            "original_gpt_decision": gpt_result,
        }

    decision = str(gpt_result.get("decision", "unsure"))
    best_id = gpt_result.get("best_match_id")

    if best_id is not None and str(best_id) not in top_k_uids:
        logger.warning(
            "GPT returned best_match_id=%s outside Top-K %s; rejecting",
            best_id,
            top_k_uids,
        )
        return {
            "final_decision": "structural_fallback",
            "final_match_id": None,
            "guard_action": "rejected_outside_topk",
            "original_gpt_decision": gpt_result,
        }

    if decision == "unsure":
        return {
            "final_decision": "structural_fallback",
            "final_match_id": None,
            "guard_action": "unsure_fallback",
            "original_gpt_decision": gpt_result,
        }

    if decision == "match" and indicator_overlap < 0.20:
        logger.info(
            "GPT says match but indicator_overlap=%.3f < 0.20; downgrade to review",
            indicator_overlap,
        )
        return {
            "final_decision": "review",
            "final_match_id": str(best_id) if best_id else None,
            "guard_action": "low_overlap_downgrade",
            "original_gpt_decision": gpt_result,
        }

    if decision == "no_match" and structural_score > 0.85 and indicator_overlap > 0.6:
        logger.info(
            "GPT says no_match but structural=%.3f overlap=%.3f; keeping structural match",
            structural_score,
            indicator_overlap,
        )
        return {
            "final_decision": "structural_kept",
            "final_match_id": None,
            "guard_action": "structural_override",
            "original_gpt_decision": gpt_result,
        }

    if decision == "match" and best_id:
        return {
            "final_decision": "match",
            "final_match_id": str(best_id),
            "guard_action": "none",
            "original_gpt_decision": gpt_result,
        }

    return {
        "final_decision": "no_match",
        "final_match_id": None,
        "guard_action": "none",
        "original_gpt_decision": gpt_result,
    }


def _extract_gpt_raw(result: dict[str, Any]) -> str:
    gpt = result.get("original_gpt_decision")
    if isinstance(gpt, dict):
        return str(gpt.get("raw", ""))
    return ""


def _write_semantic_judge_log(log_path: Any, record: dict[str, Any]) -> None:
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("semantic_judge log write failed: %s", exc)


def run_semantic_judge_for_pair(
    *,
    bank_code: str,
    api_key: str,
    table_t1: Any,
    table_t2: Any | None,
    pair: dict[str, Any],
    indicator_overlap: float,
    suspicious_low_overlap: bool,
    t1_uid: str,
    t2_uid: str | None,
    strict_result: dict[str, Any],
    t2_by_uid: dict[str, Any],
    log_path: Any,
    top_k: int = _DEFAULT_TOP_K,
) -> dict[str, Any]:
    del suspicious_low_overlap
    structural_score = float(pair.get("score", 0.0) or 0.0)

    ref_summary = _table_summary(
        table_t1,
        structural_score=structural_score,
        indicator_overlap=indicator_overlap,
    )

    alt_candidates = _extract_top_k_candidates(
        t1_uid, strict_result, t2_by_uid, t2_uid, k=top_k
    )

    all_candidates: list[dict[str, Any]] = []
    top_k_uids: set[str] = set()

    if t2_uid and table_t2 is not None:
        all_candidates.append(
            _table_summary(
                table_t2,
                structural_score=structural_score,
                indicator_overlap=indicator_overlap,
            )
        )
        top_k_uids.add(t2_uid)

    for cand in alt_candidates:
        cand_uid = cand["t2_uid"]
        if cand_uid in top_k_uids:
            continue
        cand_overlap = _compute_overlap_for_candidate(table_t1, cand["t2_table"])
        all_candidates.append(
            _table_summary(
                cand["t2_table"],
                structural_score=cand["score"],
                indicator_overlap=cand_overlap,
            )
        )
        top_k_uids.add(cand_uid)
        if len(all_candidates) >= top_k:
            break

    if not all_candidates:
        result = {
            "final_decision": "structural_fallback",
            "final_match_id": None,
            "guard_action": "no_candidates",
            "original_gpt_decision": None,
        }
    else:
        gpt_result = _call_gpt(api_key, ref_summary, all_candidates)
        result = _apply_guard_rails(
            gpt_result,
            top_k_uids=top_k_uids,
            structural_score=structural_score,
            indicator_overlap=indicator_overlap,
        )

    log_record = {
        "bank": bank_code,
        "t1_id": getattr(table_t1, "table_id", ""),
        "top_k_candidates": [c.get("table_id", "") for c in all_candidates],
        "structural_scores": [c.get("structural_score", 0.0) for c in all_candidates],
        "overlap_values": [c.get("indicator_overlap", 0.0) for c in all_candidates],
        "gpt_raw_response": _extract_gpt_raw(result),
        "final_decision_after_guard": result.get("final_decision", ""),
        "guard_action": result.get("guard_action", ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _write_semantic_judge_log(log_path, log_record)

    return result


def _compute_overlap_for_candidate(t1: Any, t2: Any) -> float:
    from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison

    def _keys(t: Any) -> set[str]:
        result: set[str] = set()
        for ind in getattr(t, "first_column_indicators", []) or []:
            s = str(ind).strip()
            if not s:
                continue
            key = normalize_indicator_for_comparison(s)
            if key:
                result.add(key)
        return result

    a, b = _keys(t1), _keys(t2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_semantic_judge_for_unmatched(
    *,
    bank_code: str,
    api_key: str,
    unmatched_table: Any,
    change_type: str,
    opposite_tables: list[Any],
    strict_result: dict[str, Any],
    t_by_uid: dict[str, Any],
    log_path: Any,
    top_k: int = _DEFAULT_TOP_K,
) -> dict[str, Any]:
    del strict_result, t_by_uid
    ref_summary = _table_summary(unmatched_table)

    candidates_with_overlap: list[tuple[Any, float]] = []
    for opp in opposite_tables:
        overlap = _compute_overlap_for_candidate(unmatched_table, opp)
        candidates_with_overlap.append((opp, overlap))

    candidates_with_overlap.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates_with_overlap[:top_k]

    all_summaries: list[dict[str, Any]] = []
    top_k_uids: set[str] = set()
    for opp, ov in top_candidates:
        uid = f"{opp.section}|{opp.table_id}|p{opp.page_pdf}"
        all_summaries.append(_table_summary(opp, indicator_overlap=ov))
        top_k_uids.add(uid)

    if not all_summaries:
        result = {
            "final_decision": "no_match",
            "final_match_id": None,
            "guard_action": "no_candidates",
            "original_gpt_decision": None,
        }
    else:
        gpt_result = _call_gpt(api_key, ref_summary, all_summaries)
        result = _apply_guard_rails(
            gpt_result,
            top_k_uids=top_k_uids,
            structural_score=0.0,
            indicator_overlap=0.0,
        )

    log_record = {
        "bank": bank_code,
        "t1_id": getattr(unmatched_table, "table_id", ""),
        "change_type": change_type,
        "top_k_candidates": [c.get("table_id", "") for c in all_summaries],
        "overlap_values": [c.get("indicator_overlap", 0.0) for c in all_summaries],
        "gpt_raw_response": _extract_gpt_raw(result),
        "final_decision_after_guard": result.get("final_decision", ""),
        "guard_action": result.get("guard_action", ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _write_semantic_judge_log(log_path, log_record)

    return result
