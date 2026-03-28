"""GPT-backed semantic diff for already-matched canonical financial tables."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


INDICATOR_DIFF_SYSTEM_PROMPT = """
You are a precision-first banking table indicator diff engine.

You compare two already-matched canonical banking tables from adjacent quarterly reports.

Your task is to report only meaningful semantic indicator changes:
- indicators_added
- indicators_removed
- indicators_renamed

Rules:
- The table pair is already matched. Do not question the pairing.
- Compare only indicator meaning and role in the table.
- Ignore numeric values, dates, periods, formatting, OCR noise, row order changes, and line wrapping.
- IGNORE footnote marker changes: if two indicators differ ONLY by a footnote reference like (1), (2), (3), (4) being added, removed, or changed, they are THE SAME indicator. Do NOT classify this as renamed. Example: 'catégorie 1 (4)' and 'catégorie 1' are identical — ignore this.
- Indicator present only in current = indicators_added.
- Indicator present only in previous = indicators_removed.
- Classify indicators_renamed only when the previous and current indicators clearly represent the exact same business concept with the same scope and the same role in the table.
- If the change could instead be explained by addition, removal, scope change, row split, or row merge, do NOT classify it as renamed.
- When unsure between rename and add/remove, prefer add/remove.
- If one previous indicator appears split into multiple current indicators, treat the new rows as additions rather than rename.
- If multiple previous indicators appear merged into one current indicator, do not classify as rename unless the scope is clearly identical.
- Be conservative and report only clear semantic differences.
- For each change (added, removed, renamed), you MUST act as a Senior Risk Analyst and provide an 'analyst_assessment'.
- The 'analyst_assessment' MUST include:
  1. A 'relevance_level' (integer: 1 for Critical/Regulatory, 2 for High/Structural, 3 for Low/Cosmetic).
  2. A 'justification' (A clear, articulate, and complete descriptive sentence explaining the business impact and exactly WHY this change matters to guide the analyst).

Output must be valid JSON following the response_schema.
"""


FOOTNOTE_DIFF_SYSTEM_PROMPT = """
You are a precision-first banking table footnote diff engine.

You compare footnotes from two already-matched canonical banking tables from adjacent quarterly reports.

Your task is to report only meaningful footnote changes:
- footnotes_added
- footnotes_removed
- footnotes_renamed

Rules:
- The table pair is already matched. Do not question the pairing.
- Compare only semantic footnote meaning, not numbering or formatting.
- Ignore pure footnote renumbering when the meaning is unchanged.
- Ignore changes caused only by dates, quarter references, formatting, punctuation, or minor drafting changes that do not alter meaning.
- IGNORE page number changes: if two footnotes differ ONLY by page references (e.g. 'pages 6 à 10' vs 'pages 6 à 12'), they are THE SAME footnote. Do NOT classify this as renamed.
- IGNORE quarter/date reference updates: if a footnote text changes only the quarter or date reference (e.g. '31 janvier 2025' vs '30 avril 2025'), this is NOT a meaningful change.
- Footnote present only in current = footnotes_added.
- Footnote present only in previous = footnotes_removed.
- Footnote with the same semantic meaning but materially revised wording = footnotes_renamed.
- Compare footnotes within the logical scope of the same table and in the context of the already-matched pair.
- Be conservative and report only clear semantic differences.
- For each change (added, removed, renamed), you MUST act as a Senior Risk Analyst and provide an 'analyst_assessment'.
- The 'analyst_assessment' MUST include:
  1. A 'relevance_level' (integer: 1 for Critical/Regulatory, 2 for High/Structural, 3 for Low/Cosmetic).
  2. A 'justification' (A clear, articulate, and complete descriptive sentence explaining the business impact and exactly WHY this change matters to guide the analyst).

Output must be valid JSON following the response_schema.
"""


def _normalize_footnotes(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not fid and not text:
            continue
        normalized.append({"id": fid, "text": text})
    return normalized


def _normalize_reasoned_values(
    items: Any,
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get(value_key, "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if not value:
            continue
        assessment = item.get("analyst_assessment")
        out.append(
            {
                value_key: value,
                "reason": reason,
                "analyst_assessment": dict(assessment)
                if isinstance(assessment, dict)
                else {},
            }
        )
    return out


def _normalize_indicator_renames(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        previous = str(item.get("previous", "") or "").strip()
        current = str(item.get("current", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if not previous or not current:
            continue
        assessment = item.get("analyst_assessment")
        out.append(
            {
                "previous": previous,
                "current": current,
                "reason": reason,
                "analyst_assessment": dict(assessment)
                if isinstance(assessment, dict)
                else {},
            }
        )
    return out


def _normalize_footnote_reasoned_values(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if not fid and not text:
            continue
        assessment = item.get("analyst_assessment")
        out.append(
            {
                "id": fid,
                "text": text,
                "reason": reason,
                "analyst_assessment": dict(assessment)
                if isinstance(assessment, dict)
                else {},
            }
        )
    return out


def _normalize_footnote_renames(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        previous_id = str(item.get("previous_id", "") or "").strip()
        current_id = str(item.get("current_id", "") or "").strip()
        previous_text = str(item.get("previous_text", "") or "").strip()
        current_text = str(item.get("current_text", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if (
            not previous_id
            and not current_id
            and not previous_text
            and not current_text
        ):
            continue
        assessment = item.get("analyst_assessment")
        out.append(
            {
                "previous_id": previous_id,
                "current_id": current_id,
                "previous_text": previous_text,
                "current_text": current_text,
                "reason": reason,
                "analyst_assessment": dict(assessment)
                if isinstance(assessment, dict)
                else {},
            }
        )
    return out


# ---------------------------------------------------------------------------
# Deterministic diff helpers (safety net)
# ---------------------------------------------------------------------------

_FOOTNOTE_MARKER_RE = re.compile(r"\s*[\(\[]\d{1,2}[\)\]]\s*")
_SUPERSCRIPT_DIGITS = str.maketrans(
    "", "", "\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070"
)


def _normalize_indicator_text(name: str) -> str:
    """Normalise an indicator name for deterministic set comparison."""
    text = str(name or "").strip()
    text = _FOOTNOTE_MARKER_RE.sub("", text)
    text = text.translate(_SUPERSCRIPT_DIGITS)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _token_overlap_ratio(a: str, b: str) -> float:
    """Return Jaccard-like token overlap ratio between two normalised strings."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _deterministic_indicator_diff(
    prev_indicators: list[str],
    curr_indicators: list[str],
    *,
    fuzzy_threshold: float = 0.80,
) -> dict[str, Any]:
    """Compute set-based indicator diff before GPT call."""
    prev_norm = {_normalize_indicator_text(ind): ind for ind in prev_indicators}
    curr_norm = {_normalize_indicator_text(ind): ind for ind in curr_indicators}

    prev_keys = set(prev_norm.keys())
    curr_keys = set(curr_norm.keys())

    only_prev = prev_keys - curr_keys
    only_curr = curr_keys - prev_keys

    # Attempt fuzzy matching between the unmatched sets
    det_renamed: list[dict[str, str]] = []
    matched_prev: set[str] = set()
    matched_curr: set[str] = set()
    for pkey in sorted(only_prev):
        best_score = 0.0
        best_ckey = ""
        for ckey in sorted(only_curr):
            if ckey in matched_curr:
                continue
            score = _token_overlap_ratio(pkey, ckey)
            if score > best_score:
                best_score = score
                best_ckey = ckey
        if best_score >= fuzzy_threshold and best_ckey:
            det_renamed.append(
                {"previous": prev_norm[pkey], "current": curr_norm[best_ckey]}
            )
            matched_prev.add(pkey)
            matched_curr.add(best_ckey)

    det_removed = [prev_norm[k] for k in sorted(only_prev - matched_prev)]
    det_added = [curr_norm[k] for k in sorted(only_curr - matched_curr)]

    return {
        "det_added": det_added,
        "det_removed": det_removed,
        "det_renamed": det_renamed,
    }


_DATE_QUARTER_RE = re.compile(
    r"\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4}"
    r"|T[1-4]\s*[-–]?\s*\d{4}"
    r"|\d{4}\s*[-–]?\s*T[1-4]"
    r"|(?:premier|deuxième|troisième|quatrième)\s+trimestre\s+\d{4}",
    re.IGNORECASE,
)
_PAGE_REF_RE_DET = re.compile(r"pages?\s+\d+\s*[àa]\s*\d+", re.IGNORECASE)


def _normalize_footnote_text(text: str) -> str:
    """Normalise footnote text for deterministic comparison (strip dates/pages/whitespace)."""
    text = str(text or "").strip()
    text = _DATE_QUARTER_RE.sub("__DATE__", text)
    text = _PAGE_REF_RE_DET.sub("__PAGE__", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _deterministic_footnote_diff(
    prev_footnotes: list[dict[str, str]],
    curr_footnotes: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute set-based footnote diff before GPT call."""
    prev_by_id: dict[str, dict[str, str]] = {}
    for fn in prev_footnotes:
        fid = str(fn.get("id", "") or "").strip()
        if fid:
            prev_by_id[fid] = fn

    curr_by_id: dict[str, dict[str, str]] = {}
    for fn in curr_footnotes:
        fid = str(fn.get("id", "") or "").strip()
        if fid:
            curr_by_id[fid] = fn

    prev_ids = set(prev_by_id.keys())
    curr_ids = set(curr_by_id.keys())

    det_added = [curr_by_id[fid] for fid in sorted(curr_ids - prev_ids)]
    det_removed = [prev_by_id[fid] for fid in sorted(prev_ids - curr_ids)]

    # For IDs present in both, check if text changed materially
    det_modified: list[dict[str, Any]] = []
    for fid in sorted(prev_ids & curr_ids):
        prev_text = _normalize_footnote_text(prev_by_id[fid].get("text", ""))
        curr_text = _normalize_footnote_text(curr_by_id[fid].get("text", ""))
        if prev_text != curr_text:
            det_modified.append(
                {
                    "previous_id": fid,
                    "current_id": fid,
                    "previous_text": prev_by_id[fid].get("text", ""),
                    "current_text": curr_by_id[fid].get("text", ""),
                }
            )

    # Cross-match removed/added by text similarity (re-numbered footnotes)
    unmatched_removed = list(det_removed)
    unmatched_added = list(det_added)
    cross_renamed: list[dict[str, Any]] = []
    still_removed: list[dict[str, str]] = []
    for rfn in unmatched_removed:
        r_text = _normalize_footnote_text(rfn.get("text", ""))
        best_idx = -1
        best_match = False
        for idx, afn in enumerate(unmatched_added):
            a_text = _normalize_footnote_text(afn.get("text", ""))
            if r_text == a_text:
                best_idx = idx
                best_match = True
                break
        if best_match:
            afn = unmatched_added.pop(best_idx)
            # Same text, different ID → pure renumbering, not a real change
        else:
            still_removed.append(rfn)
    det_removed = still_removed
    det_added = unmatched_added

    return {
        "det_added": det_added,
        "det_removed": det_removed,
        "det_modified": det_modified,
    }


def _table_context(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = list(entry.get("indicators", []) or [])
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": entry.get("page"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [
            str(value).strip()
            for value in list(entry.get("headers", []) or [])
            if str(value).strip()
        ],
        "indicators": [
            str(value).strip() for value in indicators if str(value).strip()
        ],
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _call_validated_diff_json(
    *,
    system_prompt: str,
    prompt: dict[str, Any],
    required_list_fields: tuple[str, ...],
    model: str,
    call_kind: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None,
    max_validation_attempts: int,
) -> dict[str, Any]:
    validation_feedback = ""
    data: dict[str, Any] | None = None
    for attempt in range(max_validation_attempts):
        request_prompt = dict(prompt)
        if validation_feedback:
            request_prompt["validation_feedback"] = validation_feedback
            request_prompt["rules"] = list(prompt["rules"]) + [
                "Your previous response was structurally invalid. Fix the validation issue and return corrected JSON."
            ]
        data = call_openai_json(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(request_prompt, ensure_ascii=False),
                },
            ],
            usage_recorder=usage_recorder,
            call_kind=call_kind,
        )
        if all(isinstance(data.get(field, []), list) for field in required_list_fields):
            return data
        validation_feedback = (
            "Diff response must contain list-valued fields for: "
            + ", ".join(required_list_fields)
        )
        if attempt + 1 >= max_validation_attempts:
            raise RuntimeError(
                f"GPT {call_kind} output remained structurally invalid after retries."
            )
    raise RuntimeError("Unreachable diff validation loop")


def diff_indicators_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    prev_ctx = _table_context(previous_table)
    curr_ctx = _table_context(current_table)

    # --- Deterministic pre-diff (safety net) ---
    det_diff = _deterministic_indicator_diff(
        prev_ctx["indicators"],
        curr_ctx["indicators"],
    )
    det_hints: dict[str, Any] = {}
    if det_diff["det_removed"] or det_diff["det_added"] or det_diff["det_renamed"]:
        det_hints = {
            "deterministic_analysis": {
                "mechanically_absent_from_current": det_diff["det_removed"],
                "mechanically_absent_from_previous": det_diff["det_added"],
                "potential_renames_by_similarity": [
                    {"previous": r["previous"], "current": r["current"]}
                    for r in det_diff["det_renamed"]
                ],
            }
        }

    rules = [
        "Return JSON only and strictly follow the response_schema.",
        "The two tables are already matched. Do not question the pairing.",
        "Compare only the canonical indicators.",
        "Ignore numeric values, dates, periods, formatting differences, OCR noise, row order changes, and line wrapping.",
        "Indicator present only in current = indicators_added.",
        "Indicator present only in previous = indicators_removed.",
        "Classify indicators_renamed only when the concept, scope, and role are clearly identical.",
        "When unsure between rename and add/remove, prefer add/remove.",
        "Do not treat row splits or row merges as renamed indicators unless the scope is clearly identical.",
    ]
    if det_hints:
        rules.append(
            "A deterministic set analysis is provided in 'deterministic_analysis'. "
            "You MUST account for every indicator listed there: classify each as truly "
            "added/removed/renamed, or explain why it is OCR noise / footnote marker difference."
        )

    prompt: dict[str, Any] = {
        "task": (
            "Compare two already-matched banking tables and report only meaningful semantic indicator changes."
        ),
        "rules": rules,
        "response_schema": {
            "indicators_added": [
                {
                    "value": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "indicators_removed": [
                {
                    "value": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "indicators_renamed": [
                {
                    "previous": "string",
                    "current": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "reason": "string",
        },
        "previous_table": {
            key: value for key, value in prev_ctx.items() if key != "footnotes"
        },
        "current_table": {
            key: value for key, value in curr_ctx.items() if key != "footnotes"
        },
    }
    if det_hints:
        prompt.update(det_hints)

    data = _call_validated_diff_json(
        system_prompt=INDICATOR_DIFF_SYSTEM_PROMPT,
        prompt=prompt,
        required_list_fields=(
            "indicators_added",
            "indicators_removed",
            "indicators_renamed",
        ),
        model=model,
        call_kind="diff_indicators",
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )
    return {
        "indicators_added": _normalize_reasoned_values(
            data.get("indicators_added", []), value_key="value"
        ),
        "indicators_removed": _normalize_reasoned_values(
            data.get("indicators_removed", []), value_key="value"
        ),
        "indicators_renamed": _normalize_indicator_renames(
            data.get("indicators_renamed", [])
        ),
        "reason": str(data.get("reason", "") or "").strip(),
    }


def diff_footnotes_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    indicator_diff: dict[str, Any],
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    previous_footnotes = _normalize_footnotes(previous_table.get("footnotes", []))
    current_footnotes = _normalize_footnotes(current_table.get("footnotes", []))

    if not previous_footnotes and not current_footnotes:
        return {
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "",
        }

    if not previous_footnotes and current_footnotes:
        return {
            "footnotes_added": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "reason": "Footnote present only in current table.",
                    "analyst_assessment": {
                        "relevance_level": 3,
                        "justification": "L'ajout d'une nouvelle note de bas de page sans contexte détaillé nécessite une vérification manuelle pour confirmer son impact.",
                    },
                }
                for item in current_footnotes
            ],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Current table contains footnotes while previous table had none.",
        }

    if previous_footnotes and not current_footnotes:
        return {
            "footnotes_added": [],
            "footnotes_removed": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "reason": "Footnote present only in previous table.",
                    "analyst_assessment": {
                        "relevance_level": 3,
                        "justification": "La suppression d'une ancienne note de bas de page sans contexte détaillé nécessite une vérification manuelle pour confirmer son impact.",
                    },
                }
                for item in previous_footnotes
            ],
            "footnotes_renamed": [],
            "reason": "Previous table contains footnotes while current table has none.",
        }

    # --- Deterministic footnote pre-diff (safety net) ---
    det_fn_diff = _deterministic_footnote_diff(previous_footnotes, current_footnotes)
    det_fn_hints: dict[str, Any] = {}
    if (
        det_fn_diff["det_added"]
        or det_fn_diff["det_removed"]
        or det_fn_diff["det_modified"]
    ):
        det_fn_hints = {
            "deterministic_footnote_analysis": {
                "footnotes_only_in_current": [
                    {"id": fn.get("id", ""), "text": fn.get("text", "")}
                    for fn in det_fn_diff["det_added"]
                ],
                "footnotes_only_in_previous": [
                    {"id": fn.get("id", ""), "text": fn.get("text", "")}
                    for fn in det_fn_diff["det_removed"]
                ],
                "footnotes_with_text_changes": [
                    {
                        "id": fn["previous_id"],
                        "previous_text": fn["previous_text"],
                        "current_text": fn["current_text"],
                    }
                    for fn in det_fn_diff["det_modified"]
                ],
            }
        }

    fn_rules = [
        "Return JSON only and strictly follow the response_schema.",
        "The two tables are already matched. Do not question the pairing.",
        "Ignore pure footnote renumbering when meaning is unchanged.",
        "Ignore changes caused only by dates, quarter references, formatting, punctuation, or minor drafting changes that do not alter meaning.",
        "Footnote present only in current = footnotes_added.",
        "Footnote present only in previous = footnotes_removed.",
        "Same semantic note with materially revised wording = footnotes_renamed.",
        "Be conservative and report only clear semantic differences.",
    ]
    if det_fn_hints:
        fn_rules.append(
            "A deterministic footnote analysis is provided in 'deterministic_footnote_analysis'. "
            "You MUST account for every footnote listed there: classify each change as truly "
            "added/removed/renamed, or explain why it is pure renumbering / date-only change."
        )

    prompt: dict[str, Any] = {
        "task": (
            "Compare footnotes for two already-matched banking tables and report only meaningful semantic footnote changes."
        ),
        "rules": fn_rules,
        "response_schema": {
            "footnotes_added": [
                {
                    "id": "string",
                    "text": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "footnotes_removed": [
                {
                    "id": "string",
                    "text": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "footnotes_renamed": [
                {
                    "previous_id": "string",
                    "current_id": "string",
                    "previous_text": "string",
                    "current_text": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "reason": "string",
        },
        "examples": [
            {
                "description": "Pure footnote renumbering should not produce a published change.",
                "previous_footnotes": [
                    {"id": "7", "text": "Comprennent les engagements de la Banque."}
                ],
                "current_footnotes": [
                    {"id": "8", "text": "Comprennent les engagements de la Banque."}
                ],
                "expected_output": {
                    "footnotes_added": [],
                    "footnotes_removed": [],
                    "footnotes_renamed": [],
                },
            },
            {
                "description": "Material semantic wording update should be exposed as footnotes_renamed.",
                "previous_footnotes": [
                    {"id": "7", "text": "Comprennent les engagements de la Banque."}
                ],
                "current_footnotes": [
                    {
                        "id": "8",
                        "text": "Comprennent aussi les engagements de la Banque.",
                    }
                ],
                "expected_output": {
                    "footnotes_added": [],
                    "footnotes_removed": [],
                    "footnotes_renamed": [
                        {
                            "previous_id": "7",
                            "current_id": "8",
                            "previous_text": "Comprennent les engagements de la Banque.",
                            "current_text": "Comprennent aussi les engagements de la Banque.",
                            "reason": "Same note with materially revised wording.",
                            "analyst_assessment": {
                                "relevance_level": 2,
                                "justification": "La clarification de la portée des engagements élargit le périmètre d'inclusion comptable, ce qui justifie une révision analytique.",
                            },
                        }
                    ],
                },
            },
        ],
        "pair_context": {
            "previous_table": _table_context(previous_table),
            "current_table": _table_context(current_table),
            "indicator_diff": {
                "indicators_added": list(
                    indicator_diff.get("indicators_added", []) or []
                ),
                "indicators_removed": list(
                    indicator_diff.get("indicators_removed", []) or []
                ),
                "indicators_renamed": list(
                    indicator_diff.get("indicators_renamed", []) or []
                ),
            },
        },
    }
    if det_fn_hints:
        prompt.update(det_fn_hints)

    data = _call_validated_diff_json(
        system_prompt=FOOTNOTE_DIFF_SYSTEM_PROMPT,
        prompt=prompt,
        required_list_fields=(
            "footnotes_added",
            "footnotes_removed",
            "footnotes_renamed",
        ),
        model=model,
        call_kind="diff_footnotes",
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )
    return {
        "footnotes_added": _normalize_footnote_reasoned_values(
            data.get("footnotes_added", [])
        ),
        "footnotes_removed": _normalize_footnote_reasoned_values(
            data.get("footnotes_removed", [])
        ),
        "footnotes_renamed": _normalize_footnote_renames(
            data.get("footnotes_renamed", [])
        ),
        "reason": str(data.get("reason", "") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Post-GPT deterministic guard
# ---------------------------------------------------------------------------

_GUARD_ASSESSMENT = {
    "relevance_level": 2,
    "justification": "Détecté par le filet de sécurité déterministe — absent du résultat GPT.",
}


def _is_covered_by_gpt_indicators(
    needle: str,
    gpt_added: list[dict[str, Any]],
    gpt_removed: list[dict[str, Any]],
    gpt_renamed: list[dict[str, Any]],
) -> bool:
    """Check if a normalised indicator is already accounted for in GPT output."""
    norm = _normalize_indicator_text(needle)
    for item in gpt_removed:
        if _normalize_indicator_text(item.get("value", "")) == norm:
            return True
    for item in gpt_added:
        if _normalize_indicator_text(item.get("value", "")) == norm:
            return True
    for item in gpt_renamed:
        if _normalize_indicator_text(item.get("previous", "")) == norm:
            return True
        if _normalize_indicator_text(item.get("current", "")) == norm:
            return True
    return False


def _apply_indicator_guard(
    indicator_diff: dict[str, Any],
    prev_indicators: list[str],
    curr_indicators: list[str],
) -> dict[str, Any]:
    """Inject indicators missed by GPT but found by deterministic set diff."""
    det = _deterministic_indicator_diff(prev_indicators, curr_indicators)
    gpt_added = list(indicator_diff.get("indicators_added", []) or [])
    gpt_removed = list(indicator_diff.get("indicators_removed", []) or [])
    gpt_renamed = list(indicator_diff.get("indicators_renamed", []) or [])

    injected = 0
    for val in det["det_removed"]:
        if not _is_covered_by_gpt_indicators(val, gpt_added, gpt_removed, gpt_renamed):
            gpt_removed.append(
                {
                    "value": val,
                    "reason": "Filet déterministe : indicateur absent du tableau courant, non signalé par GPT.",
                    "analyst_assessment": dict(_GUARD_ASSESSMENT),
                    "source": "deterministic_guard",
                }
            )
            injected += 1

    for val in det["det_added"]:
        if not _is_covered_by_gpt_indicators(val, gpt_added, gpt_removed, gpt_renamed):
            gpt_added.append(
                {
                    "value": val,
                    "reason": "Filet déterministe : indicateur absent du tableau précédent, non signalé par GPT.",
                    "analyst_assessment": dict(_GUARD_ASSESSMENT),
                    "source": "deterministic_guard",
                }
            )
            injected += 1

    if injected:
        logger.info("Deterministic guard injected %d indicator change(s).", injected)

    return {
        **indicator_diff,
        "indicators_added": gpt_added,
        "indicators_removed": gpt_removed,
        "indicators_renamed": gpt_renamed,
    }


def _is_covered_by_gpt_footnotes(
    fn_id: str,
    gpt_added: list[dict[str, Any]],
    gpt_removed: list[dict[str, Any]],
    gpt_renamed: list[dict[str, Any]],
) -> bool:
    """Check if a footnote ID is already accounted for in GPT output."""
    for item in gpt_added:
        if str(item.get("id", "")).strip() == fn_id:
            return True
    for item in gpt_removed:
        if str(item.get("id", "")).strip() == fn_id:
            return True
    for item in gpt_renamed:
        if str(item.get("previous_id", "")).strip() == fn_id:
            return True
        if str(item.get("current_id", "")).strip() == fn_id:
            return True
    return False


def _apply_footnote_guard(
    footnote_diff: dict[str, Any],
    prev_footnotes: list[dict[str, str]],
    curr_footnotes: list[dict[str, str]],
) -> dict[str, Any]:
    """Inject footnote changes missed by GPT but found by deterministic diff."""
    det = _deterministic_footnote_diff(prev_footnotes, curr_footnotes)
    gpt_added = list(footnote_diff.get("footnotes_added", []) or [])
    gpt_removed = list(footnote_diff.get("footnotes_removed", []) or [])
    gpt_renamed = list(footnote_diff.get("footnotes_renamed", []) or [])

    injected = 0
    for fn in det["det_removed"]:
        fid = str(fn.get("id", "")).strip()
        if fid and not _is_covered_by_gpt_footnotes(
            fid, gpt_added, gpt_removed, gpt_renamed
        ):
            gpt_removed.append(
                {
                    "id": fid,
                    "text": fn.get("text", ""),
                    "reason": "Filet déterministe : note absente du tableau courant, non signalée par GPT.",
                    "analyst_assessment": dict(_GUARD_ASSESSMENT),
                    "source": "deterministic_guard",
                }
            )
            injected += 1

    for fn in det["det_added"]:
        fid = str(fn.get("id", "")).strip()
        if fid and not _is_covered_by_gpt_footnotes(
            fid, gpt_added, gpt_removed, gpt_renamed
        ):
            gpt_added.append(
                {
                    "id": fid,
                    "text": fn.get("text", ""),
                    "reason": "Filet déterministe : note absente du tableau précédent, non signalée par GPT.",
                    "analyst_assessment": dict(_GUARD_ASSESSMENT),
                    "source": "deterministic_guard",
                }
            )
            injected += 1

    for fn in det["det_modified"]:
        fid = str(fn.get("previous_id", "")).strip()
        if fid and not _is_covered_by_gpt_footnotes(
            fid, gpt_added, gpt_removed, gpt_renamed
        ):
            gpt_renamed.append(
                {
                    "previous_id": fid,
                    "current_id": str(fn.get("current_id", "")).strip(),
                    "previous_text": fn.get("previous_text", ""),
                    "current_text": fn.get("current_text", ""),
                    "reason": "Filet déterministe : texte de note modifié matériellement, non signalé par GPT.",
                    "analyst_assessment": dict(_GUARD_ASSESSMENT),
                    "source": "deterministic_guard",
                }
            )
            injected += 1

    if injected:
        logger.info("Deterministic guard injected %d footnote change(s).", injected)

    return {
        **footnote_diff,
        "footnotes_added": gpt_added,
        "footnotes_removed": gpt_removed,
        "footnotes_renamed": gpt_renamed,
    }


def _compose_pair_reason(
    *,
    indicator_reason: str,
    footnote_reason: str,
    has_indicator_changes: bool,
    has_footnote_changes: bool,
) -> str:
    indicator_reason = str(indicator_reason or "").strip()
    footnote_reason = str(footnote_reason or "").strip()
    if has_indicator_changes and has_footnote_changes:
        parts = [part for part in (indicator_reason, footnote_reason) if part]
        if parts:
            if len(parts) == 2 and parts[0] == parts[1]:
                return parts[0]
            return " ".join(parts)
        return "Des changements sémantiques ont été détectés sur les indicateurs et les notes de bas de page."
    if has_indicator_changes:
        return (
            indicator_reason
            or "Des changements sémantiques ont été détectés sur les indicateurs."
        )
    if has_footnote_changes:
        return (
            footnote_reason
            or "Des changements sémantiques ont été détectés sur les notes de bas de page."
        )
    return indicator_reason or footnote_reason or "Aucun changement sémantique détecté."


def diff_table_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    indicator_diff = diff_indicators_pair_gpt(
        previous_table,
        current_table,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )
    footnote_diff = diff_footnotes_pair_gpt(
        previous_table,
        current_table,
        indicator_diff=indicator_diff,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )

    # --- Post-GPT deterministic guard ---
    prev_ctx = _table_context(previous_table)
    curr_ctx = _table_context(current_table)
    indicator_diff = _apply_indicator_guard(
        indicator_diff,
        prev_ctx["indicators"],
        curr_ctx["indicators"],
    )

    previous_footnotes = _normalize_footnotes(previous_table.get("footnotes", []))
    current_footnotes = _normalize_footnotes(current_table.get("footnotes", []))
    footnote_gpt_called = bool(previous_footnotes and current_footnotes)
    footnote_diff = _apply_footnote_guard(
        footnote_diff,
        previous_footnotes,
        current_footnotes,
    )

    technical_diff: dict[str, Any] = {
        "indicators_added": list(indicator_diff.get("indicators_added", []) or []),
        "indicators_removed": list(indicator_diff.get("indicators_removed", []) or []),
        "indicators_renamed": list(indicator_diff.get("indicators_renamed", []) or []),
        "footnotes_added": list(footnote_diff.get("footnotes_added", []) or []),
        "footnotes_removed": list(footnote_diff.get("footnotes_removed", []) or []),
        "footnotes_renamed": list(footnote_diff.get("footnotes_renamed", []) or []),
    }
    has_changes = any(technical_diff.values())
    technical_diff["table_level_change"] = "modifie" if has_changes else "inchange"
    reason = _compose_pair_reason(
        indicator_reason=str(indicator_diff.get("reason", "") or ""),
        footnote_reason=str(footnote_diff.get("reason", "") or ""),
        has_indicator_changes=any(
            technical_diff[field]
            for field in (
                "indicators_added",
                "indicators_removed",
                "indicators_renamed",
            )
        ),
        has_footnote_changes=any(
            technical_diff[field]
            for field in (
                "footnotes_added",
                "footnotes_removed",
                "footnotes_renamed",
            )
        ),
    )
    return {
        "technical_diff": technical_diff,
        "reason": reason,
        "diff_mode": "gpt",
        "diff_calls_total": 2 if footnote_gpt_called else 1,
    }
