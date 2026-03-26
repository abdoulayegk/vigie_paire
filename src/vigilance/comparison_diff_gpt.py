"""GPT-backed semantic diff for already-matched canonical financial tables."""

from __future__ import annotations

import json
from typing import Any, Callable


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
- Indicator present only in current = indicators_added.
- Indicator present only in previous = indicators_removed.
- Classify indicators_renamed only when the previous and current indicators clearly represent the exact same business concept with the same scope and the same role in the table.
- If the change could instead be explained by addition, removal, scope change, row split, or row merge, do NOT classify it as renamed.
- When unsure between rename and add/remove, prefer add/remove.
- If one previous indicator appears split into multiple current indicators, treat the new rows as additions rather than rename.
- If multiple previous indicators appear merged into one current indicator, do not classify as rename unless the scope is clearly identical.
- Be conservative and report only clear semantic differences.

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
- Footnote present only in current = footnotes_added.
- Footnote present only in previous = footnotes_removed.
- Footnote with the same semantic meaning but materially revised wording = footnotes_renamed.
- Compare footnotes within the logical scope of the same table and in the context of the already-matched pair.
- Be conservative and report only clear semantic differences.

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
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get(value_key, "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if not value:
            continue
        out.append({value_key: value, "reason": reason})
    return out


def _normalize_indicator_renames(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        previous = str(item.get("previous", "") or "").strip()
        current = str(item.get("current", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if not previous or not current:
            continue
        out.append({"previous": previous, "current": current, "reason": reason})
    return out


def _normalize_footnote_reasoned_values(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if not fid and not text:
            continue
        out.append({"id": fid, "text": text, "reason": reason})
    return out


def _normalize_footnote_renames(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
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
        out.append(
            {
                "previous_id": previous_id,
                "current_id": current_id,
                "previous_text": previous_text,
                "current_text": current_text,
                "reason": reason,
            }
        )
    return out


def _table_context(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = list(entry.get("indicators", []) or [])
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": entry.get("page"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [str(value).strip() for value in list(entry.get("headers", []) or []) if str(value).strip()],
        "indicators": [str(value).strip() for value in indicators if str(value).strip()],
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
                {"role": "user", "content": json.dumps(request_prompt, ensure_ascii=False)},
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
    prompt = {
        "task": (
            "Compare two already-matched banking tables and report only meaningful semantic indicator changes."
        ),
        "rules": [
            "Return JSON only and strictly follow the response_schema.",
            "The two tables are already matched. Do not question the pairing.",
            "Compare only the canonical indicators.",
            "Ignore numeric values, dates, periods, formatting differences, OCR noise, row order changes, and line wrapping.",
            "Indicator present only in current = indicators_added.",
            "Indicator present only in previous = indicators_removed.",
            "Classify indicators_renamed only when the concept, scope, and role are clearly identical.",
            "When unsure between rename and add/remove, prefer add/remove.",
            "Do not treat row splits or row merges as renamed indicators unless the scope is clearly identical.",
        ],
        "response_schema": {
            "indicators_added": [{"value": "string", "reason": "string"}],
            "indicators_removed": [{"value": "string", "reason": "string"}],
            "indicators_renamed": [
                {"previous": "string", "current": "string", "reason": "string"}
            ],
            "reason": "string",
        },
        "previous_table": {
            key: value
            for key, value in _table_context(previous_table).items()
            if key != "footnotes"
        },
        "current_table": {
            key: value
            for key, value in _table_context(current_table).items()
            if key != "footnotes"
        },
    }
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
                }
                for item in previous_footnotes
            ],
            "footnotes_renamed": [],
            "reason": "Previous table contains footnotes while current table has none.",
        }

    prompt = {
        "task": (
            "Compare footnotes for two already-matched banking tables and report only meaningful semantic footnote changes."
        ),
        "rules": [
            "Return JSON only and strictly follow the response_schema.",
            "The two tables are already matched. Do not question the pairing.",
            "Ignore pure footnote renumbering when meaning is unchanged.",
            "Ignore changes caused only by dates, quarter references, formatting, punctuation, or minor drafting changes that do not alter meaning.",
            "Footnote present only in current = footnotes_added.",
            "Footnote present only in previous = footnotes_removed.",
            "Same semantic note with materially revised wording = footnotes_renamed.",
            "Be conservative and report only clear semantic differences.",
        ],
        "response_schema": {
            "footnotes_added": [{"id": "string", "text": "string", "reason": "string"}],
            "footnotes_removed": [{"id": "string", "text": "string", "reason": "string"}],
            "footnotes_renamed": [
                {
                    "previous_id": "string",
                    "current_id": "string",
                    "previous_text": "string",
                    "current_text": "string",
                    "reason": "string",
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
                        }
                    ],
                },
            },
        ],
        "pair_context": {
            "previous_table": _table_context(previous_table),
            "current_table": _table_context(current_table),
            "indicator_diff": {
                "indicators_added": list(indicator_diff.get("indicators_added", []) or []),
                "indicators_removed": list(indicator_diff.get("indicators_removed", []) or []),
                "indicators_renamed": list(indicator_diff.get("indicators_renamed", []) or []),
            },
        },
    }
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
        return indicator_reason or "Des changements sémantiques ont été détectés sur les indicateurs."
    if has_footnote_changes:
        return footnote_reason or "Des changements sémantiques ont été détectés sur les notes de bas de page."
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
    previous_footnotes = _normalize_footnotes(previous_table.get("footnotes", []))
    current_footnotes = _normalize_footnotes(current_table.get("footnotes", []))
    footnote_gpt_called = bool(previous_footnotes and current_footnotes)

    technical_diff = {
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
