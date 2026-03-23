"""GPT-4o comparison pipeline on canonical tables.json artifacts."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from vigilance.config import resolve_openai_model
from vigilance.utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)

MATCH_PROMPT_VERSION = "table_match_v2"
DIFF_PROMPT_VERSION = "table_diff_v2"
COMPARISON_SCHEMA_VERSION = 1

_HIGH_IMPACT_THEMES = frozenset({"capital", "liquidite", "financement"})
_RISK_THEMES = frozenset({"risque_credit", "risque_marche", "risque_operationnel"})
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


def _table_card(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "indicators_normalized": [
            str(value).strip()
            for value in list(entry.get("indicators_normalized", []) or [])
            if str(value).strip()
        ],
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _table_detail(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "indicators_raw": [
            str(value).strip()
            for value in list(entry.get("indicators_raw", []) or [])
            if str(value).strip()
        ],
        "indicators_normalized": [
            str(value).strip()
            for value in list(entry.get("indicators_normalized", []) or [])
            if str(value).strip()
        ],
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _table_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "title": str(entry.get("title", "") or ""),
        "page": entry.get("page"),
        "section": str(entry.get("section", "") or "unknown_section"),
        "bbox": entry.get("bbox"),
        "indicators_raw": [
            str(value).strip()
            for value in list(entry.get("indicators_raw", []) or [])
            if str(value).strip()
        ],
        "indicators_normalized": [
            str(value).strip()
            for value in list(entry.get("indicators_normalized", []) or [])
            if str(value).strip()
        ],
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


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
        out.append(
            {
                "previous": previous,
                "current": current,
                "reason": reason,
            }
        )
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
        if not previous_id and not current_id and not previous_text and not current_text:
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


def _call_openai_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int = 4000,
    temperature: float = 0.0,
    api_retry_max: int = 2,
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
            return data
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = (
                "rate" in message and "limit" in message
                or "timeout" in message
                or "timed out" in message
                or "connection" in message
                or "connect" in message
            )
            if not retryable or attempt >= api_retry_max:
                break
    raise RuntimeError(f"OpenAI comparison call failed: {last_error}")


def _ascii_fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text or ""))
        if not unicodedata.combining(c)
    ).lower()


def _theme_text(*parts: Any) -> str:
    raw = " ".join(str(part or "") for part in parts if str(part or "").strip())
    folded = _ascii_fold(raw)
    folded = re.sub(r"[^a-z0-9\s]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _classify_theme(
    *,
    section: str,
    title: str,
    indicators: list[str],
    footnotes: list[dict[str, str]],
) -> str:
    text = _theme_text(
        section,
        title,
        " ".join(indicators),
        " ".join(item.get("text", "") for item in footnotes),
    )
    if any(
        token in text
        for token in (
            "capital",
            "fonds propres",
            "cet1",
            "tier 1",
            "tlac",
            "levier",
            "leverage",
        )
    ):
        return "capital"
    if any(
        token in text
        for token in ("liquidite", "liquidity", "lcr", "nsfr", "hqla")
    ):
        return "liquidite"
    if any(
        token in text
        for token in ("funding", "financement", "depot", "depots")
    ):
        return "financement"
    if any(
        token in text
        for token in (
            "credit risk",
            "risque de credit",
            "perte de credit",
            "expected credit",
        )
    ):
        return "risque_credit"
    if any(
        token in text
        for token in ("market risk", "risque de marche", "trading", "var")
    ):
        return "risque_marche"
    if any(
        token in text
        for token in (
            "operational risk",
            "risque operationnel",
            "operational",
        )
    ):
        return "risque_operationnel"
    return "autre"


def _technical_change_counts(technical_diff: dict[str, Any]) -> tuple[int, int, int]:
    indicator_add_remove = len(technical_diff.get("indicators_added", []) or []) + len(
        technical_diff.get("indicators_removed", []) or []
    )
    renamed = len(technical_diff.get("indicators_renamed", []) or []) + len(
        technical_diff.get("footnotes_renamed", []) or []
    )
    footnote_add_remove = len(technical_diff.get("footnotes_added", []) or []) + len(
        technical_diff.get("footnotes_removed", []) or []
    )
    return indicator_add_remove, renamed, footnote_add_remove


def _build_analyst_summary(
    *,
    theme: str,
    change_kind: str,
    technical_diff: dict[str, Any],
) -> str:
    if change_kind == "ajoute":
        return f"Nouveau tableau sur le theme {theme} a revoir par l'analyste."
    if change_kind == "supprime":
        return f"Tableau supprime sur le theme {theme} a confirmer par l'analyste."
    total_indicators = (
        len(technical_diff.get("indicators_added", []) or [])
        + len(technical_diff.get("indicators_removed", []) or [])
        + len(technical_diff.get("indicators_renamed", []) or [])
    )
    total_footnotes = (
        len(technical_diff.get("footnotes_added", []) or [])
        + len(technical_diff.get("footnotes_removed", []) or [])
        + len(technical_diff.get("footnotes_renamed", []) or [])
    )
    if total_indicators == 0 and total_footnotes == 0:
        return f"Aucun changement semantique detecte sur le theme {theme}."
    return (
        f"Changements semantiques detectes sur le theme {theme}: "
        f"{total_indicators} changement(s) d'indicateur et {total_footnotes} changement(s) de footnote."
    )


def _build_analyst_assessment(
    *,
    table_context: dict[str, Any],
    technical_diff: dict[str, Any],
    change_kind: str,
) -> dict[str, str]:
    theme = _classify_theme(
        section=str(table_context.get("section", "") or ""),
        title=str(table_context.get("title", "") or ""),
        indicators=[
            str(value)
            for value in list(
                table_context.get("indicators_raw")
                or table_context.get("indicators_normalized")
                or []
            )
        ],
        footnotes=_normalize_footnotes(table_context.get("footnotes", [])),
    )
    indicator_add_remove, renamed, footnote_add_remove = _technical_change_counts(
        technical_diff
    )
    total_changes = indicator_add_remove + renamed + footnote_add_remove

    if change_kind in {"ajoute", "supprime"}:
        if theme in _HIGH_IMPACT_THEMES:
            significance, priority = "eleve", "critique"
        elif theme in _RISK_THEMES:
            significance, priority = "moyen", "prioritaire"
        else:
            significance, priority = "moyen", "normale"
    elif total_changes == 0:
        significance, priority = "faible", "normale"
    elif theme in _HIGH_IMPACT_THEMES:
        significance = "eleve"
        priority = (
            "critique"
            if indicator_add_remove + footnote_add_remove >= 2
            else "prioritaire"
        )
    elif theme in _RISK_THEMES:
        significance = "moyen"
        priority = "prioritaire" if total_changes >= 2 else "normale"
    else:
        significance = "moyen" if total_changes >= 2 else "faible"
        priority = "prioritaire" if total_changes >= 3 else "normale"

    return {
        "theme": theme,
        "change_significance": significance,
        "review_priority": priority,
        "analyst_summary": _build_analyst_summary(
            theme=theme,
            change_kind=change_kind,
            technical_diff=technical_diff,
        ),
    }


def _match_tables(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    prompt = {
        "task": "Match canonical financial tables across two report extracts.",
        "rules": [
            "Return JSON only.",
            "Each previous table can match at most one current table.",
            "Each current table can match at most one previous table.",
            "Match tables semantically using first-column indicators and associated footnotes.",
            "Section and title are weak context only; prioritize indicators_normalized and footnotes.",
            "Ignore all numeric changes, dates, period labels, and date formats.",
            "Unmatched current tables go to tables_added.",
            "Unmatched previous tables go to tables_removed.",
        ],
        "response_schema": {
            "matched_pairs": [
                {
                    "previous_table_id": "string",
                    "current_table_id": "string",
                    "match_confidence": "number_0_to_1",
                    "reason": "string",
                }
            ],
            "tables_added": [{"table_id": "string", "reason": "string"}],
            "tables_removed": [{"table_id": "string", "reason": "string"}],
        },
        "previous_tables": previous_cards,
        "current_tables": current_cards,
    }
    data = _call_openai_json(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You compare extracted financial tables for a banking analyst. "
                    "Match tables semantically using first-column indicators and footnotes only. "
                    "Ignore numeric changes, dates, period labels, and date formats. "
                    "Return valid JSON only and do not add commentary."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )

    previous_ids = {card["table_id"] for card in previous_cards}
    current_ids = {card["table_id"] for card in current_cards}
    matched_pairs: list[dict[str, Any]] = []
    used_previous: set[str] = set()
    used_current: set[str] = set()

    for item in list(data.get("matched_pairs", []) or []):
        if not isinstance(item, dict):
            continue
        previous_table_id = _require_string(
            item.get("previous_table_id"), "previous_table_id"
        )
        current_table_id = _require_string(
            item.get("current_table_id"), "current_table_id"
        )
        if previous_table_id not in previous_ids:
            raise ValueError(
                f"Unknown previous_table_id in match output: {previous_table_id}"
            )
        if current_table_id not in current_ids:
            raise ValueError(
                f"Unknown current_table_id in match output: {current_table_id}"
            )
        if previous_table_id in used_previous:
            raise ValueError(
                f"Duplicate previous_table_id in match output: {previous_table_id}"
            )
        if current_table_id in used_current:
            raise ValueError(
                f"Duplicate current_table_id in match output: {current_table_id}"
            )
        used_previous.add(previous_table_id)
        used_current.add(current_table_id)
        try:
            confidence = float(item.get("match_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        matched_pairs.append(
            {
                "previous_table_id": previous_table_id,
                "current_table_id": current_table_id,
                "match_confidence": max(0.0, min(1.0, confidence)),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )

    def _normalize_unmatched(
        items: Any,
        *,
        valid_ids: set[str],
        field: str,
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, dict):
                continue
            table_id = _require_string(item.get("table_id"), field)
            if table_id not in valid_ids:
                raise ValueError(f"Unknown table_id in {field}: {table_id}")
            out.append(
                {
                    "table_id": table_id,
                    "reason": str(item.get("reason", "") or "").strip(),
                }
            )
        return out

    return {
        "matched_pairs": matched_pairs,
        "tables_added": _normalize_unmatched(
            data.get("tables_added", []),
            valid_ids=current_ids,
            field="tables_added",
        ),
        "tables_removed": _normalize_unmatched(
            data.get("tables_removed", []),
            valid_ids=previous_ids,
            field="tables_removed",
        ),
    }


def _diff_pair(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    prompt = {
        "task": "Compare two matched financial tables and identify semantic changes.",
        "rules": [
            "Return JSON only.",
            "Compare only first-column indicators and associated footnotes.",
            "Use indicators and footnotes semantically, not by exact string only.",
            "Only classify as renamed when the underlying meaning is the same.",
            "Ignore all numeric changes, dates, period labels, and date formats.",
            "Do not use numeric values or temporal labels to infer changes.",
        ],
        "response_schema": {
            "indicators_added": [{"value": "string", "reason": "string"}],
            "indicators_removed": [{"value": "string", "reason": "string"}],
            "indicators_renamed": [
                {"previous": "string", "current": "string", "reason": "string"}
            ],
            "footnotes_added": [
                {"id": "string", "text": "string", "reason": "string"}
            ],
            "footnotes_removed": [
                {"id": "string", "text": "string", "reason": "string"}
            ],
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
        "previous_table": previous_table,
        "current_table": current_table,
    }
    data = _call_openai_json(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You compare two already-matched financial tables for a banking analyst. "
                    "Only compare first-column indicators and associated footnotes. "
                    "Ignore numeric changes, dates, period labels, and date formats. "
                    "Return valid JSON only and do not add commentary."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )

    technical_diff = {
        "indicators_added": _normalize_reasoned_values(
            data.get("indicators_added", []), value_key="value"
        ),
        "indicators_removed": _normalize_reasoned_values(
            data.get("indicators_removed", []), value_key="value"
        ),
        "indicators_renamed": _normalize_indicator_renames(
            data.get("indicators_renamed", [])
        ),
        "footnotes_added": _normalize_footnote_reasoned_values(
            data.get("footnotes_added", [])
        ),
        "footnotes_removed": _normalize_footnote_reasoned_values(
            data.get("footnotes_removed", [])
        ),
        "footnotes_renamed": _normalize_footnote_renames(
            data.get("footnotes_renamed", [])
        ),
    }
    has_changes = any(technical_diff.values())
    technical_diff["table_level_change"] = "modifie" if has_changes else "inchange"
    return {
        "technical_diff": technical_diff,
        "reason": str(data.get("reason", "") or "").strip(),
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
        if str(assessment.get("review_priority", "") or "") in {"prioritaire", "critique"}:
            total += 1
    for item in tables_added + tables_removed:
        assessment = item.get("analyst_assessment", {}) or {}
        if str(assessment.get("review_priority", "") or "") in {"prioritaire", "critique"}:
            total += 1
    return total


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
) -> Path:
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
    previous_cards = [_table_card(entry) for entry in previous_tables]
    current_cards = [_table_card(entry) for entry in current_tables]
    previous_lookup = {
        entry["table_id"]: _table_detail(entry) for entry in previous_tables
    }
    current_lookup = {
        entry["table_id"]: _table_detail(entry) for entry in current_tables
    }
    previous_snapshots = {
        entry["table_id"]: _table_snapshot(entry) for entry in previous_tables
    }
    current_snapshots = {
        entry["table_id"]: _table_snapshot(entry) for entry in current_tables
    }

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

    match_result = _match_tables(previous_cards, current_cards, model=model_name)

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
                "analyst_assessment": _build_analyst_assessment(
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
                "analyst_assessment": _build_analyst_assessment(
                    table_context=previous_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="supprime",
                ),
            }
        )

    pair_comparisons: list[dict[str, Any]] = []
    for pair in match_result["matched_pairs"]:
        previous_table_id = pair["previous_table_id"]
        current_table_id = pair["current_table_id"]
        diff = _diff_pair(
            previous_lookup[previous_table_id],
            current_lookup[current_table_id],
            model=model_name,
        )
        pair_comparisons.append(
            {
                "previous_table_id": previous_table_id,
                "current_table_id": current_table_id,
                "match_confidence": pair["match_confidence"],
                "match_reason": pair.get("reason", ""),
                "previous_table": previous_snapshots[previous_table_id],
                "current_table": current_snapshots[current_table_id],
                "technical_diff": diff["technical_diff"],
                "analyst_assessment": _build_analyst_assessment(
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
        },
        "pair_comparisons": pair_comparisons,
        "summary": {
            "matched_pairs_total": len(match_result["matched_pairs"]),
            "tables_added_total": len(tables_added),
            "tables_removed_total": len(tables_removed),
            "indicator_changes_total": indicator_changes_total,
            "footnote_changes_total": footnote_changes_total,
            "high_priority_items_total": high_priority_items_total,
        },
    }
    return _atomic_write_json(out_dir / "comparison.json", payload)
