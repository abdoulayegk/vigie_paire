"""GPT-4o comparison pipeline on canonical tables.json artifacts."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vigilance.config import resolve_openai_model
from vigilance.comparison_analyst import build_analyst_assessment
from vigilance.comparison_diff_gpt import diff_table_pair_gpt
from vigilance.utils.genai import get_openai_api_key
from vigilance.utils.model_cost import estimate_openai_cost_usd

logger = logging.getLogger(__name__)

MATCH_PROMPT_VERSION = "table_match_v6"
DIFF_PROMPT_VERSION = "table_diff_v4"
COMPARISON_SCHEMA_VERSION = 2
_BUSINESS_EXTRACTION_STATUSES = frozenset({"ok", "rescued"})
_ARTIFACT_EXTRACTION_STATUSES = frozenset({"confirmed_no_table"})
_SUSPECT_EXTRACTION_STATUSES = frozenset({"suspect_unresolved"})

PRIMARY_MATCH_SYSTEM_PROMPT = """
You are a precision-first financial table matcher for bank quarterly reports.

Given lists of business tables from a previous report and a current report, produce a strict one-to-one matching ledger over current tables.

Rules:
- Every current table must be classified exactly once as `matched` or `unresolved`.
- Each previous table can be used at most once.
- This is the strict pass: return `unresolved` if any material doubt remains.
- Never skip a current table.
- Never force a speculative match.
- When 2 or 3 candidates are similar, resolve the best one-to-one arrangement jointly instead of independently.

Evidence (strong -> weak):
1. Indicators - semantic overlap, ordering, distinctive sequence. Generic labels alone (e.g. "Total", "Other", "Canada", "Stage 1") are weak.
2. Headers - similar structure and meaning.
3. Title - only when specific and distinctive.
4. Table summary and footnotes - reinforce business purpose.
5. Row count - diff <= 2 strong, diff <= 5 acceptable only with other strong evidence.
6. Section and page - tie-breakers only.

Important:
- Do not match tables only because they are in the same section or on nearby pages.
- Do not match tables only because they have a similar generic title.
- Cross-section matches are allowed only when semantic evidence clearly outweighs the section mismatch.
- Ignore numeric values, dates, formatting, and footnote numbering.

Examples of good `matched` decisions:
{
  "current_table_id": "tbl_p053_i01",
  "decision": "matched",
  "previous_table_id": "tbl_p051_i01",
  "reason": "Title identical: 'SOMMAIRE DU FINANCEMENT PROVENANT DES DÉPÔTS'; indicators: same 3 rows (personnels, commerciaux, total); row_count 3 vs 3.",
  "match_confidence": 0.95
}

{
  "current_table_id": "tbl_p047_i02",
  "decision": "matched",
  "previous_table_id": "tbl_p045_i02",
  "reason": "Indicators: same 4 entities (société mère, filiales bancaires, succursales étrangères, total); row_count 4 vs 4; headers similar structure.",
  "match_confidence": 0.90
}

Example of correct `unresolved` decision:
{
  "current_table_id": "tbl_p045_i01",
  "decision": "unresolved",
  "reason": "Two previous tables remain plausible because indicators overlap only partially and the titles are generic; strict pass keeps this unresolved."
}

Output must be valid JSON following the response_schema.
"""

RECOVERY_MATCH_SYSTEM_PROMPT = """
You are a recovery matcher for bank quarterly tables.

You receive:
- a set of current tables that were unresolved in the primary stage
- a set of unused previous tables that were not matched in the primary stage

Your task is to assign each current table to either:
- `matched` with one unused previous table
- `added` if no sufficiently credible remaining match exists

Rules:
- Every current table must have exactly one decision: `matched` or `added`.
- Each previous table can be used at most once.
- This is the recovery pass: you may accept a match that is weaker than the strict primary pass, but only if it is still the best globally coherent non-speculative match among the remaining tables.
- If the best available candidate is still materially ambiguous, prefer `added` over forcing a match.
- Never skip a current table.
- When 2 or 3 candidates are similar, resolve the best one-to-one arrangement jointly instead of independently.

Evidence ranking (strong -> weak):
1. Indicators - semantic overlap, ordering, distinctive sequence.
2. Headers - similar structure and meaning.
3. Title - only when specific and distinctive.
4. Table summary and footnotes - reinforce business purpose.
5. Row count - diff <= 2 strong, diff <= 5 acceptable only with other strong evidence.
6. Section and page - tie-breakers only.

Important:
- Cross-section matches are allowed if semantic evidence is clearly stronger than the section mismatch.
- Generic labels alone are not sufficient.
- Do not match tables only because they are in the same section or on nearby pages.
- Do not match tables only because they have a similar generic title.
- Ignore numeric values, dates, formatting, OCR noise, and footnote numbering when meaning is unchanged.

Examples:

Recovery match:
{
  "current_table_id": "tbl_p053_i02",
  "decision": "matched",
  "previous_table_id": "tbl_p051_i03",
  "reason": "Title identical: 'FINANCEMENT À LONG TERME¹'; indicators: same 5 currency rows (dollar canadien, américain, euro, livre sterling, total); row_count 5 vs 5.",
  "match_confidence": 0.85
}

If no reasonable remaining match exists:
{
  "current_table_id": "tbl_p039_i01",
  "decision": "added",
  "reason": "Indicators are unique; no remaining previous table has a sufficiently similar indicator structure, title, or business purpose."
}

Output must be valid JSON following the response_schema.
"""

REFERENCE_RESOLUTION_RULE = (
    "t2->t1 meme annee; t3->t2 meme annee; "
    "t1->t3 annee precedente; t4->t4 annee precedente"
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _coerce_pathlike(value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Chemin requis manquant: {field}")
    try:
        return Path(text)
    except TypeError as exc:
        raise ValueError(f"Chemin invalide pour {field}: {value!r}") from exc


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def _load_tables_payload(extraction_dir: Path) -> dict[str, Any]:
    tables_path = extraction_dir / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"Missing required file: {tables_path}")
    payload = _load_json(tables_path)
    schema_version = _coerce_int(payload.get("schema_version"))
    if schema_version != 7:
        raise ValueError(
            f"Unsupported tables.json schema_version={schema_version} in {tables_path}; expected 7"
        )
    tables = payload.get("tables", [])
    if not isinstance(tables, list):
        raise ValueError(f"Invalid tables list in {tables_path}")
    return payload


def normalize_quarter(value: str) -> str:
    text = str(value or "").strip().lower()
    match = re.search(r"([qt])\s*([1-4])", text)
    if match:
        return f"t{match.group(2)}"
    raise ValueError(f"Trimestre invalide: {value!r}. Attendu: t1, t2, t3 ou t4.")


def resolve_reference_period(
    year_current: int,
    quarter_current: str,
) -> tuple[int, str]:
    quarter = normalize_quarter(quarter_current)
    year = int(year_current)
    if quarter == "t2":
        return year, "t1"
    if quarter == "t3":
        return year, "t2"
    if quarter == "t1":
        return year - 1, "t3"
    if quarter == "t4":
        return year - 1, "t4"
    raise ValueError(f"Trimestre invalide: {quarter_current!r}")


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


def _normalize_extraction_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in _BUSINESS_EXTRACTION_STATUSES | _ARTIFACT_EXTRACTION_STATUSES | _SUSPECT_EXTRACTION_STATUSES:
        return status
    return "ok"


def _table_card(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = [
        str(value).strip()
        for value in list(entry.get("indicators", []) or [])
        if str(value).strip()
    ]
    row_count = int(entry.get("row_count", len(indicators)) or 0)

    header_source = entry.get("headers", [])
    header_list = list(header_source) if isinstance(header_source, list) else []
    headers = []
    for h in header_list:
        cell = str(h).strip()
        if cell:
            headers.append(cell)
    page_raw = entry.get("page")
    page: int | None
    try:
        page = (
            int(page_raw)
            if page_raw is not None and str(page_raw).strip() != ""
            else None
        )
    except (TypeError, ValueError):
        page = None

    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": page,
        "row_count": row_count,
        "headers": headers,
        "indicators": indicators,
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }

def _table_detail(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = [
        str(value).strip()
        for value in list(entry.get("indicators", []) or [])
        if str(value).strip()
    ]
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
        "indicators": indicators,
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _table_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    indicators = [
        str(value).strip()
        for value in list(entry.get("indicators", []) or [])
        if str(value).strip()
    ]
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "extraction_status": str(entry.get("extraction_status", "") or "ok"),
        "page": entry.get("page"),
        "section": str(entry.get("section", "") or "unknown_section"),
        "bbox": entry.get("bbox"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [
            str(value).strip()
            for value in list(entry.get("headers", []) or [])
            if str(value).strip()
        ],
        "indicators": indicators,
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _partition_tables_by_status(
    tables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    business: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    suspects: list[dict[str, str]] = []
    for entry in tables:
        if not isinstance(entry, dict):
            continue
        table_id = str(entry.get("table_id", "") or "").strip()
        if not table_id:
            continue
        status = _normalize_extraction_status(entry.get("extraction_status"))
        if status in _BUSINESS_EXTRACTION_STATUSES:
            business.append(entry)
        elif status in _ARTIFACT_EXTRACTION_STATUSES:
            artifacts.append(
                {
                    "table_id": table_id,
                    "reason": "Excluded from business matching by extraction_status=confirmed_no_table.",
                }
            )
        else:
            suspects.append(
                {
                    "table_id": table_id,
                    "reason": "Excluded from business matching by extraction_status=suspect_unresolved.",
                }
            )
    return business, artifacts, suspects


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _unique_run_dir(base_dir: Path, run_id: str) -> tuple[str, Path]:
    candidate = base_dir / run_id
    if not candidate.exists():
        return run_id, candidate
    counter = 2
    while True:
        suffix_id = f"{run_id}_{counter:02d}"
        candidate = base_dir / suffix_id
        if not candidate.exists():
            return suffix_id, candidate
        counter += 1


def _archive_pdf(source_path: str | None, out_dir: Path, filename: str) -> str:
    text = str(source_path or "").strip()
    if not text:
        return ""
    source = Path(text)
    if not source.exists() or not source.is_file():
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    try:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return str(target)
    except OSError:
        return ""


def _require_string(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required string field: {field}")
    return text


def _extract_usage_metrics(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    try:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        prompt = 0
    try:
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        completion = 0
    try:
        total = int(getattr(usage, "total_tokens", 0) or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        total = prompt + completion
    return prompt, completion, total


def _call_openai_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int = 4000,
    temperature: float = 0.0,
    api_retry_max: int = 2,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison",
) -> dict[str, Any]:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    last_error: Exception | None = None
    for attempt in range(api_retry_max + 1):
        if attempt > 0:
            time.sleep(1.5 * (2 ** (attempt - 1)))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("OpenAI response is not a JSON object")
            if usage_recorder is not None:
                prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(
                    response
                )
                usage_recorder.append(
                    {
                        "model": model,
                        "call_kind": call_kind,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
            return data
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = (
                "rate" in message
                and "limit" in message
                or "timeout" in message
                or "timed out" in message
                or "connection" in message
                or "connect" in message
            )
            if not retryable or attempt >= api_retry_max:
                break
    raise RuntimeError(f"OpenAI comparison call failed: {last_error}")

_MATCHING_VALIDATION_ATTEMPTS = 3


class _MatchingValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        duplicate_count: int = 0,
        validation_failures: int = 1,
    ) -> None:
        super().__init__(message)
        self.duplicate_count = int(max(0, duplicate_count))
        self.validation_failures = int(max(1, validation_failures))


def _normalize_matching_warnings(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _sort_matched_pairs(
    pairs: list[dict[str, Any]],
    previous_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    order = {
        str(card.get("table_id", "") or ""): index
        for index, card in enumerate(previous_cards)
    }
    return sorted(
        pairs,
        key=lambda item: (
            order.get(str(item.get("previous_table_id", "") or ""), 10**9),
            str(item.get("previous_table_id", "") or ""),
            str(item.get("current_table_id", "") or ""),
        ),
    )

def _normalize_matching_response(
    data: dict[str, Any],
    *,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    current_table_decisions: list[dict[str, Any]] = []
    used_previous: set[str] = set()
    used_current: set[str] = set()
    duplicate_total = 0
    raw_total = 0
    consumed_previous = set(consumed_previous_ids or ())

    for item in list(data.get("current_table_decisions", []) or []):
        if not isinstance(item, dict):
            raise _MatchingValidationError(
                "current_table_decisions items must be objects"
            )
        current_table_id = _require_string(
            item.get("current_table_id"), "current_table_id"
        )
        decision = _require_string(item.get("decision"), "decision").lower()
        if decision not in allowed_decisions:
            raise _MatchingValidationError(
                "Invalid matching decision returned by GPT: "
                f"{decision!r}; allowed={sorted(allowed_decisions)}"
            )
        if current_table_id not in current_ids:
            raise _MatchingValidationError(
                f"Unknown current_table_id returned by GPT: {current_table_id}"
            )
        if current_table_id in used_current:
            duplicate_total += 1
            raise _MatchingValidationError(
                f"Duplicate current_table_id returned by GPT: {current_table_id}",
                duplicate_count=duplicate_total,
            )
        used_current.add(current_table_id)

        normalized_item: dict[str, Any] = {
            "current_table_id": current_table_id,
            "decision": decision,
            "reason": str(item.get("reason", "") or "").strip(),
        }

        previous_table_id = str(item.get("previous_table_id", "") or "").strip()
        confidence_raw = item.get("match_confidence")
        confidence_supplied = confidence_raw is not None and str(confidence_raw).strip() != ""

        if decision == "matched":
            if not previous_table_id:
                raise _MatchingValidationError(
                    "Matched decisions must include previous_table_id."
                )
            if previous_table_id not in previous_ids:
                raise _MatchingValidationError(
                    f"Unknown previous_table_id returned by GPT: {previous_table_id}"
                )
            raw_total += 1
            if previous_table_id in consumed_previous or previous_table_id in used_previous:
                duplicate_total += 1
                raise _MatchingValidationError(
                    f"Duplicate or reused previous_table_id returned by GPT: {previous_table_id}",
                    duplicate_count=duplicate_total,
                )
            try:
                confidence = float(confidence_raw or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            normalized_item["previous_table_id"] = previous_table_id
            normalized_item["match_confidence"] = max(0.0, min(1.0, confidence))
            used_previous.add(previous_table_id)
        else:
            if previous_table_id:
                raise _MatchingValidationError(
                    f"{decision!r} decisions must not include previous_table_id."
                )
            if confidence_supplied:
                raise _MatchingValidationError(
                    f"{decision!r} decisions must not include match_confidence."
                )

        current_table_decisions.append(normalized_item)

    if used_current != current_ids:
        missing = sorted(current_ids - used_current)
        extra = sorted(used_current - current_ids)
        raise _MatchingValidationError(
            "Matching output must cover exactly the business current tables. "
            f"missing={missing} extra={extra}"
        )

    return {
        "current_table_decisions": current_table_decisions,
        "warnings": _normalize_matching_warnings(data.get("warnings", [])),
        "matching_pairs_llm_duplicates_total": duplicate_total,
        "matching_pairs_llm_deduped_total": max(0, raw_total - len(used_previous)),
    }


def _matching_decisions_to_pairs(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in decisions:
        if item.get("decision") != "matched":
            continue
        out.append(
            {
                "previous_table_id": str(item.get("previous_table_id", "") or ""),
                "current_table_id": str(item.get("current_table_id", "") or ""),
                "match_confidence": _coerce_float(item.get("match_confidence")),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )
    return out


def _matching_decisions_to_table_refs(
    decisions: list[dict[str, Any]],
    *,
    decision: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in decisions:
        if item.get("decision") != decision:
            continue
        out.append(
            {
                "table_id": str(item.get("current_table_id", "") or ""),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )
    return out


def _empty_matching_result(
    *,
    tables_removed: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "executed": False,
        "matched_pairs": [],
        "tables_added": [],
        "tables_removed": list(tables_removed or []),
        "warnings": [],
        "matching_pairs_llm_duplicates_total": 0,
        "matching_pairs_llm_deduped_total": 0,
        "validation_retries_total": 0,
        "matching_validation_failures_total": 0,
        "stage1_validation_retries_total": 0,
        "stage2_validation_retries_total": 0,
        "unresolved_after_stage1_total": 0,
        "matched_in_stage2_total": 0,
        "matching_passes_total": 0,
    }


def _build_matching_stage_prompt(
    *,
    stage: str,
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    current_ids: set[str],
    previous_ids: set[str],
    allowed_decisions: set[str],
    validation_feedback: str,
) -> dict[str, Any]:
    decision_values = sorted(allowed_decisions)
    response_item: dict[str, Any] = {
        "current_table_id": "string",
        "decision": f"one_of_{decision_values}",
        "reason": "short explanation grounded in indicators, headers, title, table_summary, footnotes, row_count, section, and page only if needed",
    }
    if "matched" in allowed_decisions:
        response_item["previous_table_id"] = (
            "string_required_when_decision_is_matched"
        )
        response_item["match_confidence"] = (
            "number_0_to_1_required_when_decision_is_matched"
        )

    if stage == "primary":
        task = (
            "Perform a strict global reconciliation of business banking tables between the previous and current reports. "
            "For each current table, return either matched or unresolved."
        )
        rules = [
            "Return JSON only, strictly following response_schema.",
            "Every current_table_id must appear exactly once in current_table_decisions.",
            "Each previous_table_id can be used at most once across matched decisions.",
            "This is the strict precision-first pass: if any material doubt remains, return unresolved instead of forcing a match.",
            "Do not classify any table as added in this stage.",
        ]
    else:
        task = (
            "Resolve the remaining ambiguous current business tables against the remaining unused previous business tables. "
            "For each current table, return either matched or added."
        )
        rules = [
            "Return JSON only, strictly following response_schema.",
            "This is a recovery pass over unresolved current tables only.",
            "Every current_table_id must appear exactly once in current_table_decisions.",
            "Each previous_table_id can be used at most once across matched decisions.",
            "If the best remaining candidate is still ambiguous, return added instead of forcing a speculative match.",
            "Do not skip any current table.",
        ]

    prompt = {
        "stage": stage,
        "task": task,
        "rules": rules,
        "response_schema": {
            "current_table_decisions": [response_item],
            "warnings": ["string"],
        },
        "allowed_decisions": decision_values,
        "required_current_table_ids": sorted(current_ids),
        "allowed_previous_table_ids": sorted(previous_ids),
        "previous_tables": previous_cards,
        "current_tables": current_cards,
    }
    if validation_feedback:
        prompt["validation_feedback"] = validation_feedback
        prompt["rules"].append(
            "Your previous response was structurally invalid. Fix the listed validation problem and return a corrected JSON object."
        )
    return prompt


def _run_matching_stage(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    stage: str,
    allowed_decisions: set[str],
    model: str,
    usage_recorder: list[dict[str, Any]] | None = None,
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    previous_ids = {card["table_id"] for card in previous_cards}
    current_ids = {card["table_id"] for card in current_cards}
    system_prompt = (
        PRIMARY_MATCH_SYSTEM_PROMPT
        if stage == "primary"
        else RECOVERY_MATCH_SYSTEM_PROMPT
    )
    validation_feedback = ""
    validation_retries_total = 0
    matching_validation_failures_total = 0
    matching_pairs_llm_duplicates_total = 0

    for attempt in range(_MATCHING_VALIDATION_ATTEMPTS):
        prompt = _build_matching_stage_prompt(
            stage=stage,
            previous_cards=previous_cards,
            current_cards=current_cards,
            current_ids=current_ids,
            previous_ids=previous_ids,
            allowed_decisions=allowed_decisions,
            validation_feedback=validation_feedback,
        )
        data = _call_openai_json(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            usage_recorder=usage_recorder,
            call_kind="matching",
        )
        try:
            normalized = _normalize_matching_response(
                data,
                previous_ids=previous_ids,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                consumed_previous_ids=consumed_previous_ids,
            )
            normalized["executed"] = True
            normalized["validation_retries_total"] = validation_retries_total
            normalized["matching_validation_failures_total"] = (
                matching_validation_failures_total
            )
            normalized["matching_pairs_llm_duplicates_total"] += (
                matching_pairs_llm_duplicates_total
            )
            return normalized
        except _MatchingValidationError as exc:
            validation_feedback = str(exc)
            validation_retries_total += 1
            matching_validation_failures_total += int(
                getattr(exc, "validation_failures", 1)
            )
            matching_pairs_llm_duplicates_total += int(
                getattr(exc, "duplicate_count", 0)
            )
            if attempt + 1 >= _MATCHING_VALIDATION_ATTEMPTS:
                raise RuntimeError(
                    f"GPT matching output remained structurally invalid: {exc}"
                ) from exc

    raise RuntimeError("Unreachable matching validation loop")


def _match_tables(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not previous_cards and not current_cards:
        return _empty_matching_result()

    if not current_cards:
        tables_removed = [
            {
                "table_id": str(card.get("table_id", "") or ""),
                "reason": "No current business table was available for matching.",
            }
            for card in previous_cards
        ]
        return _empty_matching_result(tables_removed=tables_removed)

    stage1 = _run_matching_stage(
        previous_cards,
        current_cards,
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model=model,
        usage_recorder=usage_recorder,
    )
    stage1_decisions = list(stage1.get("current_table_decisions", []) or [])
    stage1_pairs = _matching_decisions_to_pairs(stage1_decisions)
    used_previous_stage1 = {
        item["previous_table_id"]
        for item in stage1_pairs
        if str(item.get("previous_table_id", "") or "").strip()
    }
    unresolved_ids = [
        item["current_table_id"]
        for item in stage1_decisions
        if item.get("decision") == "unresolved"
    ]
    unresolved_lookup = {card["table_id"]: card for card in current_cards}
    unresolved_current_cards = [
        unresolved_lookup[table_id]
        for table_id in unresolved_ids
        if table_id in unresolved_lookup
    ]
    remaining_previous_cards = [
        card
        for card in previous_cards
        if card["table_id"] not in used_previous_stage1
    ]

    stage2_decisions: list[dict[str, Any]] = []
    stage2_metrics = {
        "executed": False,
        "validation_retries_total": 0,
        "matching_validation_failures_total": 0,
        "matching_pairs_llm_duplicates_total": 0,
        "matching_pairs_llm_deduped_total": 0,
        "warnings": [],
    }
    tables_added: list[dict[str, str]] = []

    if unresolved_current_cards and remaining_previous_cards:
        stage2 = _run_matching_stage(
            remaining_previous_cards,
            unresolved_current_cards,
            stage="recovery",
            allowed_decisions={"matched", "added"},
            model=model,
            usage_recorder=usage_recorder,
            consumed_previous_ids=used_previous_stage1,
        )
        stage2_metrics = stage2
        stage2_decisions = list(stage2.get("current_table_decisions", []) or [])
        tables_added = _matching_decisions_to_table_refs(
            stage2_decisions,
            decision="added",
        )
    elif unresolved_current_cards:
        tables_added = [
            {
                "table_id": str(item.get("current_table_id", "") or ""),
                "reason": (
                    str(item.get("reason", "") or "").strip()
                    or "No previous business table remained available for matching."
                ),
            }
            for item in stage1_decisions
            if item.get("decision") == "unresolved"
        ]

    matched_pairs = stage1_pairs + _matching_decisions_to_pairs(stage2_decisions)
    used_previous_all = {
        str(item.get("previous_table_id", "") or "").strip()
        for item in matched_pairs
        if str(item.get("previous_table_id", "") or "").strip()
    }
    tables_removed = [
        {
            "table_id": str(card.get("table_id", "") or ""),
            "reason": "No current business table was matched to this previous table.",
        }
        for card in previous_cards
        if card["table_id"] not in used_previous_all
    ]
    warnings = _normalize_matching_warnings(
        list(stage1.get("warnings", []) or []) + list(stage2_metrics.get("warnings", []) or [])
    )

    return {
        "executed": bool(stage1.get("executed") or stage2_metrics.get("executed")),
        "matched_pairs": matched_pairs,
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "warnings": warnings,
        "matching_pairs_llm_duplicates_total": _coerce_int(
            stage1.get("matching_pairs_llm_duplicates_total")
        )
        + _coerce_int(stage2_metrics.get("matching_pairs_llm_duplicates_total")),
        "matching_pairs_llm_deduped_total": _coerce_int(
            stage1.get("matching_pairs_llm_deduped_total")
        )
        + _coerce_int(stage2_metrics.get("matching_pairs_llm_deduped_total")),
        "validation_retries_total": _coerce_int(
            stage1.get("validation_retries_total")
        )
        + _coerce_int(stage2_metrics.get("validation_retries_total")),
        "matching_validation_failures_total": _coerce_int(
            stage1.get("matching_validation_failures_total")
        )
        + _coerce_int(stage2_metrics.get("matching_validation_failures_total")),
        "stage1_validation_retries_total": _coerce_int(
            stage1.get("validation_retries_total")
        ),
        "stage2_validation_retries_total": _coerce_int(
            stage2_metrics.get("validation_retries_total")
        ),
        "unresolved_after_stage1_total": len(unresolved_current_cards),
        "matched_in_stage2_total": len(
            [
                item
                for item in stage2_decisions
                if item.get("decision") == "matched"
            ]
        ),
        "matching_passes_total": int(bool(stage1.get("executed")))
        + int(bool(stage2_metrics.get("executed"))),
        "unmatched_after_primary_total": len(unresolved_current_cards)
        + len(remaining_previous_cards),
        "unmatched_after_rescue_total": len(tables_added) + len(tables_removed),
    }


def _run_table_matching(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = _match_tables(
        previous_cards,
        current_cards,
        model=model,
        usage_recorder=usage_recorder,
    )
    result["matched_pairs"] = _sort_matched_pairs(
        result["matched_pairs"], previous_cards
    )
    tables_added = list(result.get("tables_added", []) or [])
    tables_removed = list(result.get("tables_removed", []) or [])
    return {
        "matched_pairs": result["matched_pairs"],
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "matching_passes_total": _coerce_int(result.get("matching_passes_total")),
        "audit_passes_total": 0,
        "matching_output_retries_total": _coerce_int(
            result.get("validation_retries_total")
        ),
        "matching_validation_failures_total": _coerce_int(
            result.get("matching_validation_failures_total")
        ),
        "stage1_validation_retries_total": _coerce_int(
            result.get("stage1_validation_retries_total")
        ),
        "stage2_validation_retries_total": _coerce_int(
            result.get("stage2_validation_retries_total")
        ),
        "unresolved_after_stage1_total": _coerce_int(
            result.get("unresolved_after_stage1_total")
        ),
        "matched_in_stage2_total": _coerce_int(result.get("matched_in_stage2_total")),
        "unmatched_previous_table_ids": [item["table_id"] for item in tables_removed],
        "unmatched_current_table_ids": [item["table_id"] for item in tables_added],
        "unmatched_after_primary_total": _coerce_int(
            result.get("unmatched_after_primary_total")
        ),
        "unmatched_after_rescue_total": _coerce_int(
            result.get("unmatched_after_rescue_total")
        ),
        "matching_pairs_llm_duplicates_total": _coerce_int(
            result.get("matching_pairs_llm_duplicates_total")
        ),
        "matching_pairs_llm_deduped_total": _coerce_int(
            result.get("matching_pairs_llm_deduped_total")
        ),
        "warnings": _normalize_matching_warnings(result.get("warnings", [])),
    }


def _count_pair_changes(
    pair_comparisons: list[dict[str, Any]],
) -> tuple[int, int]:
    indicator_total = 0
    footnote_total = 0
    for item in pair_comparisons:
        technical_diff = item.get("technical_diff", {}) or {}
        indicator_total += len(technical_diff.get("indicators_added", []) or [])
        indicator_total += len(technical_diff.get("indicators_removed", []) or [])
        indicator_total += len(technical_diff.get("indicators_renamed", []) or [])
        footnote_total += len(technical_diff.get("footnotes_added", []) or [])
        footnote_total += len(technical_diff.get("footnotes_removed", []) or [])
        footnote_total += len(technical_diff.get("footnotes_renamed", []) or [])
    return indicator_total, footnote_total


def _count_high_priority_items(
    pair_comparisons: list[dict[str, Any]],
    tables_added: list[dict[str, Any]],
    tables_removed: list[dict[str, Any]],
) -> int:
    total = 0
    for item in pair_comparisons:
        assessment = item.get("analyst_assessment", {}) or {}
        if str(assessment.get("review_priority", "") or "") in {
            "prioritaire",
            "critique",
        }:
            total += 1
    for item in tables_added + tables_removed:
        assessment = item.get("analyst_assessment", {}) or {}
        if str(assessment.get("review_priority", "") or "") in {
            "prioritaire",
            "critique",
        }:
            total += 1
    return total


def _aggregate_usage_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        prompt_tokens += _coerce_int(item.get("prompt_tokens"))
        completion_tokens += _coerce_int(item.get("completion_tokens"))
        total_tokens += _coerce_int(item.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "total_tokens_total": total_tokens,
        "comparison_calls_total": len(records),
    }


def _aggregate_extraction_run_metrics(
    extraction_run_metrics: dict[str, Any] | None,
    *,
    runtime_extraction_sec: float,
) -> dict[str, Any]:
    previous = dict((extraction_run_metrics or {}).get("previous") or {})
    current = dict((extraction_run_metrics or {}).get("current") or {})
    vision_calls_total = _coerce_int(previous.get("vision_calls_total")) + _coerce_int(
        current.get("vision_calls_total")
    )
    vision_rescue_total = _coerce_int(
        previous.get("vision_rescue_total")
    ) + _coerce_int(current.get("vision_rescue_total"))
    prompt_tokens_total = _coerce_int(
        previous.get("prompt_tokens_total")
    ) + _coerce_int(current.get("prompt_tokens_total"))
    completion_tokens_total = _coerce_int(
        previous.get("completion_tokens_total")
    ) + _coerce_int(current.get("completion_tokens_total"))
    total_tokens_total = _coerce_int(previous.get("total_tokens_total")) + _coerce_int(
        current.get("total_tokens_total")
    )
    estimated_cost = _coerce_float(previous.get("estimated_cost_usd")) + _coerce_float(
        current.get("estimated_cost_usd")
    )
    return {
        "runtime_extraction_sec": round(
            max(0.0, float(runtime_extraction_sec or 0.0)), 3
        ),
        "vision_calls_total": vision_calls_total,
        "vision_rescue_total": vision_rescue_total,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "estimated_cost_usd": round(estimated_cost, 6),
        "cache_hits_total": int(bool(previous.get("cache_hit")))
        + int(bool(current.get("cache_hit"))),
        "previous": previous,
        "current": current,
    }


def compare_reports_gpt4o(
    previous_dir: Path | str,
    current_dir: Path | str,
    out_root: Path | str,
    *,
    model: str | None = None,
    config_path: str | None = None,
    reference_resolution: dict[str, Any] | None = None,
    source_pdf_previous: str | None = None,
    source_pdf_current: str | None = None,
    runtime_extraction_sec: float | None = None,
    extraction_run_metrics: dict[str, Any] | None = None,
) -> Path:
    """Run the full report-to-report comparison pipeline and write the artifact.

    This is the public entry point used by the CLI and Dash app. It loads the
    canonical ``tables.json`` artifacts for both quarters, enriches the tables
    for matching, runs the layered matcher, computes pair-level semantic diffs,
    aggregates summaries and metrics, and finally writes ``comparison.json`` to
    a timestamped output directory.

    Args:
        previous_dir: Extraction directory for the reference quarter.
        current_dir: Extraction directory for the current quarter.
        out_root: Root directory where the comparison run folder is created.
        model: Optional OpenAI model override.
        config_path: Optional model configuration path.
        reference_resolution: Optional metadata describing how the reference
            quarter was resolved.
        source_pdf_previous: Optional source PDF path for the previous report.
        source_pdf_current: Optional source PDF path for the current report.
        runtime_extraction_sec: Optional extraction runtime propagated to final
            run metrics.
        extraction_run_metrics: Optional extraction metrics merged into final
            run metrics.

    Returns:
        The path to the generated ``comparison.json`` artifact.
    """
    comparison_started_at = time.monotonic()
    previous_dir_path = _coerce_pathlike(previous_dir, "previous_dir")
    current_dir_path = _coerce_pathlike(current_dir, "current_dir")
    out_root_path = _coerce_pathlike(out_root, "out_root")

    previous_payload = _load_tables_payload(previous_dir_path)
    current_payload = _load_tables_payload(current_dir_path)

    previous_tables = [
        entry
        for entry in list(previous_payload.get("tables", []) or [])
        if isinstance(entry, dict)
    ]
    current_tables = [
        entry
        for entry in list(current_payload.get("tables", []) or [])
        if isinstance(entry, dict)
    ]
    (
        previous_business_tables,
        previous_artifact_refs,
        previous_suspect_refs,
    ) = _partition_tables_by_status(previous_tables)
    (
        current_business_tables,
        current_artifact_refs,
        current_suspect_refs,
    ) = _partition_tables_by_status(current_tables)

    def _build_views() -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        return (
            [_table_card(entry) for entry in previous_business_tables],
            [_table_card(entry) for entry in current_business_tables],
            {entry["table_id"]: _table_detail(entry) for entry in previous_business_tables},
            {entry["table_id"]: _table_detail(entry) for entry in current_business_tables},
            {entry["table_id"]: _table_snapshot(entry) for entry in previous_tables},
            {entry["table_id"]: _table_snapshot(entry) for entry in current_tables},
        )

    (
        previous_cards,
        current_cards,
        previous_lookup,
        current_lookup,
        previous_snapshots,
        current_snapshots,
    ) = _build_views()

    bank_code = str(
        current_payload.get("bank_code") or previous_payload.get("bank_code") or ""
    )
    if not bank_code:
        raise ValueError("Missing bank_code in tables.json payloads")
    year_previous = int(previous_payload.get("year", 0) or 0)
    year_current = int(current_payload.get("year", 0) or 0)
    quarter_previous = str(previous_payload.get("quarter", "") or "")
    quarter_current = str(current_payload.get("quarter", "") or "")
    model_name = str(
        model or resolve_openai_model("default_genai", config_path=config_path)
    )
    usage_records: list[dict[str, Any]] = []

    match_result = _run_table_matching(
        previous_cards,
        current_cards,
        model=model_name,
        usage_recorder=usage_records,
    )
    tables_added: list[dict[str, Any]] = []
    for item in match_result["tables_added"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "ajoute",
        }
        tables_added.append(
            {
                **item,
                **current_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="ajoute",
                ),
            }
        )

    tables_removed: list[dict[str, Any]] = []
    for item in match_result["tables_removed"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "supprime",
        }
        tables_removed.append(
            {
                **item,
                **previous_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=previous_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="supprime",
                ),
            }
        )

    artifacts_confirmed_previous: list[dict[str, Any]] = []
    for item in previous_artifact_refs:
        table_id = item["table_id"]
        artifacts_confirmed_previous.append({**item, **previous_snapshots[table_id]})

    artifacts_confirmed_current: list[dict[str, Any]] = []
    for item in current_artifact_refs:
        table_id = item["table_id"]
        artifacts_confirmed_current.append({**item, **current_snapshots[table_id]})

    extraction_suspects_previous: list[dict[str, Any]] = []
    for item in previous_suspect_refs:
        table_id = item["table_id"]
        extraction_suspects_previous.append({**item, **previous_snapshots[table_id]})

    extraction_suspects_current: list[dict[str, Any]] = []
    for item in current_suspect_refs:
        table_id = item["table_id"]
        extraction_suspects_current.append({**item, **current_snapshots[table_id]})

    pair_comparisons: list[dict[str, Any]] = []
    diff_calls_total = 0
    for pair in match_result["matched_pairs"]:
        previous_table_id = pair["previous_table_id"]
        current_table_id = pair["current_table_id"]
        diff = diff_table_pair_gpt(
            previous_lookup[previous_table_id],
            current_lookup[current_table_id],
            model=model_name,
            call_openai_json=_call_openai_json,
            usage_recorder=usage_records,
            max_validation_attempts=_MATCHING_VALIDATION_ATTEMPTS,
        )
        diff_calls_total += _coerce_int(diff.get("diff_calls_total"))
        pair_comparisons.append(
            {
                "previous_table_id": previous_table_id,
                "current_table_id": current_table_id,
                "match_confidence": pair["match_confidence"],
                "match_reason": pair.get("reason", ""),
                "diff_mode": str(diff.get("diff_mode", "") or ""),
                "previous_table": previous_snapshots[previous_table_id],
                "current_table": current_snapshots[current_table_id],
                "technical_diff": diff["technical_diff"],
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[current_table_id],
                    technical_diff=diff["technical_diff"],
                    change_kind="modifie",
                ),
                "reason": diff["reason"],
            }
        )

    indicator_changes_total, footnote_changes_total = _count_pair_changes(
        pair_comparisons
    )
    high_priority_items_total = _count_high_priority_items(
        pair_comparisons,
        tables_added,
        tables_removed,
    )
    comparison_runtime_sec = round(
        max(0.0, time.monotonic() - comparison_started_at), 3
    )
    comparison_metrics = _aggregate_usage_metrics(usage_records)
    comparison_metrics["runtime_comparison_sec"] = comparison_runtime_sec
    comparison_metrics["matching_passes_total"] = _coerce_int(
        match_result.get("matching_passes_total")
    )
    comparison_metrics["audit_passes_total"] = _coerce_int(
        match_result.get("audit_passes_total")
    )
    comparison_metrics["matching_output_retries_total"] = _coerce_int(
        match_result.get("matching_output_retries_total")
    )
    comparison_metrics["matching_validation_failures_total"] = _coerce_int(
        match_result.get("matching_validation_failures_total")
    )
    comparison_metrics["stage1_validation_retries_total"] = _coerce_int(
        match_result.get("stage1_validation_retries_total")
    )
    comparison_metrics["stage2_validation_retries_total"] = _coerce_int(
        match_result.get("stage2_validation_retries_total")
    )
    comparison_metrics["unresolved_after_stage1_total"] = _coerce_int(
        match_result.get("unresolved_after_stage1_total")
    )
    comparison_metrics["matched_in_stage2_total"] = _coerce_int(
        match_result.get("matched_in_stage2_total")
    )
    comparison_metrics["unmatched_after_primary_total"] = _coerce_int(
        match_result.get("unmatched_after_primary_total")
    )
    comparison_metrics["unmatched_after_rescue_total"] = _coerce_int(
        match_result.get("unmatched_after_rescue_total")
    )
    comparison_metrics["matching_pairs_llm_duplicates_total"] = _coerce_int(
        match_result.get("matching_pairs_llm_duplicates_total")
    )
    comparison_metrics["matching_pairs_llm_deduped_total"] = _coerce_int(
        match_result.get("matching_pairs_llm_deduped_total")
    )
    comparison_metrics["comparison_calls_total"] = max(
        _coerce_int(comparison_metrics.get("comparison_calls_total")),
        comparison_metrics["matching_passes_total"]
        + comparison_metrics["audit_passes_total"]
        + diff_calls_total,
    )
    comparison_metrics["estimated_cost_usd"] = estimate_openai_cost_usd(
        model_name,
        prompt_tokens=comparison_metrics["prompt_tokens_total"],
        completion_tokens=comparison_metrics["completion_tokens_total"],
    )
    extraction_metrics = _aggregate_extraction_run_metrics(
        extraction_run_metrics,
        runtime_extraction_sec=float(runtime_extraction_sec or 0.0),
    )
    run_metrics = {
        "runtime_extraction_sec": extraction_metrics["runtime_extraction_sec"],
        "runtime_comparison_sec": comparison_metrics["runtime_comparison_sec"],
        "vision_calls_total": extraction_metrics["vision_calls_total"],
        "vision_rescue_total": extraction_metrics["vision_rescue_total"],
        "comparison_calls_total": comparison_metrics["comparison_calls_total"],
        "matching_passes_total": comparison_metrics["matching_passes_total"],
        "audit_passes_total": comparison_metrics["audit_passes_total"],
        "matching_output_retries_total": comparison_metrics[
            "matching_output_retries_total"
        ],
        "matching_validation_failures_total": comparison_metrics[
            "matching_validation_failures_total"
        ],
        "stage1_validation_retries_total": comparison_metrics[
            "stage1_validation_retries_total"
        ],
        "stage2_validation_retries_total": comparison_metrics[
            "stage2_validation_retries_total"
        ],
        "unresolved_after_stage1_total": comparison_metrics[
            "unresolved_after_stage1_total"
        ],
        "matched_in_stage2_total": comparison_metrics["matched_in_stage2_total"],
        "unmatched_after_primary_total": comparison_metrics[
            "unmatched_after_primary_total"
        ],
        "unmatched_after_rescue_total": comparison_metrics[
            "unmatched_after_rescue_total"
        ],
        "matching_pairs_llm_duplicates_total": comparison_metrics[
            "matching_pairs_llm_duplicates_total"
        ],
        "matching_pairs_llm_deduped_total": comparison_metrics[
            "matching_pairs_llm_deduped_total"
        ],
        "prompt_tokens_total": extraction_metrics["prompt_tokens_total"]
        + comparison_metrics["prompt_tokens_total"],
        "completion_tokens_total": extraction_metrics["completion_tokens_total"]
        + comparison_metrics["completion_tokens_total"],
        "total_tokens_total": extraction_metrics["total_tokens_total"]
        + comparison_metrics["total_tokens_total"],
        "estimated_cost_usd": round(
            float(extraction_metrics["estimated_cost_usd"] or 0.0)
            + float(comparison_metrics["estimated_cost_usd"] or 0.0),
            6,
        ),
        "extraction": extraction_metrics,
        "comparison": comparison_metrics,
    }

    base_out_dir = (
        out_root_path
        / bank_code
        / f"{year_current}_{quarter_current}_vs_{year_previous}_{quarter_previous}"
    )
    run_id, out_dir = _unique_run_dir(base_out_dir, _make_run_id())
    archived_pdf_previous = _archive_pdf(
        source_pdf_previous,
        out_dir,
        "previous_report.pdf",
    )
    archived_pdf_current = _archive_pdf(
        source_pdf_current,
        out_dir,
        "current_report.pdf",
    )
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "artifact_type": "report_comparison",
        "run_id": run_id,
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": quarter_previous,
        "year_current": year_current,
        "quarter_current": quarter_current,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_pdf_previous": str(source_pdf_previous or "").strip(),
        "source_pdf_current": str(source_pdf_current or "").strip(),
        "archived_pdf_previous": archived_pdf_previous,
        "archived_pdf_current": archived_pdf_current,
        "model_version": model_name,
        "prompt_version_match": MATCH_PROMPT_VERSION,
        "prompt_version_diff": DIFF_PROMPT_VERSION,
        "reference_resolution": (
            dict(reference_resolution)
            if isinstance(reference_resolution, dict)
            else {
                "mode": "automatique",
                "year_previous": year_previous,
                "quarter_previous": quarter_previous,
                "rule": REFERENCE_RESOLUTION_RULE,
            }
        ),
        "matching": {
            "matched_pairs": match_result["matched_pairs"],
            "tables_added": tables_added,
            "tables_removed": tables_removed,
            "artifacts_confirmed_previous": artifacts_confirmed_previous,
            "artifacts_confirmed_current": artifacts_confirmed_current,
            "extraction_suspects_previous": extraction_suspects_previous,
            "extraction_suspects_current": extraction_suspects_current,
        },
        "pair_comparisons": pair_comparisons,
        "run_metrics": run_metrics,
        "summary": {
            "matched_pairs_total": len(match_result["matched_pairs"]),
            "tables_added_total": len(tables_added),
            "tables_removed_total": len(tables_removed),
            "artifacts_confirmed_previous_total": len(artifacts_confirmed_previous),
            "artifacts_confirmed_current_total": len(artifacts_confirmed_current),
            "extraction_suspects_previous_total": len(extraction_suspects_previous),
            "extraction_suspects_current_total": len(extraction_suspects_current),
            "indicator_changes_total": indicator_changes_total,
            "footnote_changes_total": footnote_changes_total,
            "high_priority_items_total": high_priority_items_total,
        },
    }
    return _atomic_write_json(out_dir / "comparison.json", payload)
