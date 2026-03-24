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
from vigilance.utils.model_cost import estimate_openai_cost_usd

logger = logging.getLogger(__name__)

MATCH_PROMPT_VERSION = "table_match_v2"
DIFF_PROMPT_VERSION = "table_diff_v2"
COMPARISON_SCHEMA_VERSION = 1

TABLE_MATCH_SYSTEM_PROMPT = """
You are a senior financial disclosure table matching engine working for a banking analyst.

You compare canonical financial tables extracted from two quarterly bank reports:
- previous report
- current report

Your job is to recover the maximum number of true one-to-one table correspondences across both reports, while keeping false matches extremely low.

You must perform a GLOBAL one-to-one assignment:
- each previous table can match at most one current table
- each current table can match at most one previous table

Your target is high-recall, high-precision matching:
- match every table that clearly represents the same business table across the two reports
- never force a weak, speculative, or low-evidence pair
- leave unmatched only when evidence is genuinely insufficient or when a table appears to be an extraction artifact

Core matching logic:

PRIMARY evidence:
- indicators_normalized overlap in business meaning
- overall row structure similarity
- row ordering pattern
- distinctive row sequence
- whether the two tables represent the same business purpose

STRONG supporting evidence:
- row_count similarity
- header similarity
- distinctive footnote semantics
- distinctive, stable title similarity when present

WEAK supporting evidence / tie-break only:
- section similarity
- page proximity
- extraction order proximity
- same page
- nearby index

Important rules:
- A small overlap of only a few generic indicators is NOT enough for a match.
- Do not match tables only because they are in the same section, on the same page, or have nearby extraction order.
- Do not match tables only because they have a similar generic title.
- Use title as strong evidence only when it is specific, distinctive, and clearly points to the same business sub-table.
- Use headers as strong supporting evidence, especially to separate similar tables within the same section.
- Use row_count as a structural consistency signal:
  - a small row_count difference is positive evidence
  - a moderate row_count difference can still be acceptable if semantic evidence is strong
  - a large row_count difference is negative evidence unless the business meaning and structure are still clearly aligned
- Relative order in the extraction lists is important as a tie-breaker, especially when multiple candidates are similar on the same page, but it is never sufficient by itself.
- Tables with no rows and no indicators should be treated as extraction artifacts unless there is strong evidence they are real standalone tables.
- Ignore numeric value changes, percentages, currency amounts, dates, date formats, period labels, page references, punctuation differences, OCR noise, and footnote marker formatting differences.
- Ignore simple footnote renumbering if semantic content is unchanged.

Decision policy:
1. First identify the strongest high-confidence pairs.
2. Then resolve the remaining unmatched tables using the remaining free candidates.
3. Prefer semantic and structural correctness over blindly maximizing pair count.
4. If a real corresponding table clearly exists in both reports, it should be matched.
5. If evidence remains genuinely ambiguous after considering all signals, do not guess.

Confidence guidance:
- 0.90 to 1.00 = very strong match
- 0.75 to 0.89 = strong match
- 0.60 to 0.74 = plausible but weaker; acceptable only if it is still the best globally coherent candidate
- below 0.60 = do not match

Return valid JSON only.
Do not return markdown.
Do not add commentary outside JSON.
"""

_HIGH_IMPACT_THEMES = frozenset({"capital", "liquidite", "financement"})
_RISK_THEMES = frozenset({"risque_credit", "risque_marche", "risque_operationnel"})
REFERENCE_RESOLUTION_RULE = (
    "t2->t1 meme annee; t3->t2 meme annee; "
    "t1->t3 annee precedente; t4->t4 annee precedente"
)

_SECTION_THEME_MAP = {
    "capital": "capital",
    "capital_management": "capital",
    "liquidite": "liquidite",
    "liquidity": "liquidite",
    "funding": "financement",
    "funding_management": "financement",
}

_TITLE_THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "risque_credit",
        (
            "risque de credit",
            "credit risk",
            "perte de credit",
            "pertes de credit",
            "expected credit",
            "expositions au risque de credit",
            "provisions pour pertes sur creances",
        ),
    ),
    (
        "risque_marche",
        (
            "risque de marche",
            "market risk",
            "trading",
            "valeur a risque",
            "value at risk",
            "var",
        ),
    ),
    (
        "risque_operationnel",
        (
            "risque operationnel",
            "operational risk",
            "cyber",
            "fraude",
        ),
    ),
    (
        "liquidite",
        (
            "liquidite",
            "liquidity",
            "lcr",
            "nsfr",
            "hqla",
            "ratio structurel de liquidite",
        ),
    ),
    (
        "capital",
        (
            "capital",
            "fonds propres",
            "cet1",
            "tier 1",
            "tlac",
            "levier",
            "leverage",
        ),
    ),
    (
        "financement",
        (
            "financement",
            "funding",
            "depot",
            "depots",
        ),
    ),
]

_INDICATOR_THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "capital",
        (
            "ratio cet1",
            "ratio tier 1",
            "ratio total",
            "ratio de levier",
            "tlac",
        ),
    ),
    (
        "liquidite",
        (
            "lcr",
            "nsfr",
            "hqla",
            "liquidite",
            "ratio structurel de liquidite",
        ),
    ),
    (
        "risque_credit",
        (
            "pertes de credit",
            "perte de credit",
            "expected credit",
            "provision",
            "provisions",
            "exposition",
            "expositions",
            "ecl",
        ),
    ),
    (
        "risque_marche",
        (
            "trading",
            "var",
            "value at risk",
            "valeur a risque",
            "sensibilite",
        ),
    ),
    (
        "risque_operationnel",
        (
            "operational",
            "operationnel",
            "cyber",
            "fraude",
        ),
    ),
    (
        "financement",
        (
            "depots",
            "depot",
            "billets",
            "papier commercial",
            "emprunts",
            "obligations",
            "financement",
            "funding",
        ),
    ),
]


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


_TABLE_MATCH_CARD_MAX_HEADERS = 24


def _table_card(
    entry: dict[str, Any],
    *,
    extraction_index: int,
    max_headers_in_card: int = _TABLE_MATCH_CARD_MAX_HEADERS,
) -> dict[str, Any]:
    rows_raw = entry.get("rows", [])
    row_count = len(rows_raw) if isinstance(rows_raw, list) else 0

    header_source = entry.get("headers", [])
    header_list = list(header_source) if isinstance(header_source, list) else []
    cap = max(0, int(max_headers_in_card))
    headers = []
    for h in header_list[:cap]:
        cell = str(h).strip()
        if cell:
            headers.append(cell)
    header_columns_total = len(header_list)

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
        "extraction_index": int(extraction_index),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "page": page,
        "row_count": row_count,
        "headers": headers,
        "header_columns_total": header_columns_total,
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


def _match_theme_from_text(
    text: str,
    rules: list[tuple[str, tuple[str, ...]]],
) -> str | None:
    for theme, tokens in rules:
        if any(token in text for token in tokens):
            return theme
    return None


def _normalize_label(value: Any) -> str:
    text = _theme_text(value)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_footnote_signature(item: dict[str, str]) -> str:
    fid = _normalize_label(item.get("id", ""))
    text = _normalize_label(item.get("text", ""))
    return f"{fid}|{text}"


def _is_trivial_no_change(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
) -> bool:
    previous_indicators = [
        _normalize_label(value)
        for value in list(previous_table.get("indicators_normalized", []) or [])
        if _normalize_label(value)
    ]
    current_indicators = [
        _normalize_label(value)
        for value in list(current_table.get("indicators_normalized", []) or [])
        if _normalize_label(value)
    ]
    if previous_indicators != current_indicators:
        return False
    previous_footnotes = [
        _normalize_footnote_signature(item)
        for item in _normalize_footnotes(previous_table.get("footnotes", []))
    ]
    current_footnotes = [
        _normalize_footnote_signature(item)
        for item in _normalize_footnotes(current_table.get("footnotes", []))
    ]
    return previous_footnotes == current_footnotes


def _zero_technical_diff() -> dict[str, Any]:
    return {
        "indicators_added": [],
        "indicators_removed": [],
        "indicators_renamed": [],
        "footnotes_added": [],
        "footnotes_removed": [],
        "footnotes_renamed": [],
        "table_level_change": "inchange",
    }


def _classify_theme(
    *,
    section: str,
    title: str,
    indicators: list[str],
    footnotes: list[dict[str, str]],
) -> str:
    title_theme = _match_theme_from_text(_theme_text(title), _TITLE_THEME_RULES)
    if title_theme:
        return title_theme

    section_key = str(section or "").strip().lower()
    if section_key in _SECTION_THEME_MAP:
        return _SECTION_THEME_MAP[section_key]

    section_theme = _match_theme_from_text(_theme_text(section), _TITLE_THEME_RULES)
    if section_theme:
        return section_theme

    indicator_theme = _match_theme_from_text(
        _theme_text(" ".join(indicators)),
        _INDICATOR_THEME_RULES,
    )
    if indicator_theme:
        return indicator_theme

    footnote_theme = _match_theme_from_text(
        _theme_text(" ".join(item.get("text", "") for item in footnotes)),
        _INDICATOR_THEME_RULES,
    )
    if footnote_theme:
        return footnote_theme

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
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = {
        "task": (
            "Match canonical financial tables between the previous extract and the "
            "current extract using a partial bijection, maximizing true matches while "
            "avoiding false matches."
        ),
        "rules": [
            "Return JSON only, strictly following response_schema.",
            "Each previous_table_id can appear at most once in matched_pairs.",
            "Each current_table_id can appear at most once in matched_pairs.",
            "Use indicators_normalized and overall business meaning as the main evidence.",
            "Use row structure, row ordering pattern, and row_count similarity as strong structural evidence.",
            "Use header similarity as strong supporting evidence, especially to distinguish similar tables within the same section.",
            "Use footnotes as supporting semantic evidence, especially when they are distinctive.",
            "If a specific and distinctive title is present and aligns strongly across both reports, treat it as strong corroborating evidence.",
            "Do not use title alone as sufficient evidence for a match.",
            "Do not match on a small overlap of generic indicators only.",
            "Do not match tables only because they are on the same page, in the same section, or close in extraction order.",
            "When multiple candidates are similar, prefer the candidate with the strongest overall semantic and structural alignment.",
            "Use extraction order and page proximity only as tie-breakers when semantic evidence is otherwise close.",
            "Treat empty or near-empty tables as extraction artifacts unless strong evidence shows they are real tables.",
            "If a table clearly exists in both reports, match it.",
            "Leave unmatched only when evidence is genuinely insufficient or the table is likely an extraction artifact.",
        ],
        "response_schema": {
            "matched_pairs": [
                {
                    "previous_table_id": "string",
                    "current_table_id": "string",
                    "match_confidence": "number_0_to_1",
                    "reason": (
                        "short explanation grounded in business meaning, indicators, "
                        "row structure, row_count, headers, footnotes, title, and "
                        "tie-break evidence if used"
                    ),
                }
            ],
            "tables_added": [
                {
                    "table_id": "string",
                    "reason": "short explanation",
                }
            ],
            "tables_removed": [
                {
                    "table_id": "string",
                    "reason": "short explanation",
                }
            ],
            "warnings": ["string"],
        },
        "previous_tables": previous_cards,
        "current_tables": current_cards,
    }
    data = _call_openai_json(
        model=model,
        messages=[
            {
                "role": "system",
                "content": TABLE_MATCH_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        usage_recorder=usage_recorder,
        call_kind="matching",
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
        legacy_key = (
            "current_table_id" if field == "tables_added" else "previous_table_id"
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            table_id = _require_string(
                item.get("table_id") or item.get(legacy_key),
                field,
            )
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
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if _is_trivial_no_change(previous_table, current_table):
        return {
            "technical_diff": _zero_technical_diff(),
            "reason": (
                "Aucun changement semantique: indicateurs et footnotes "
                "identiques apres normalisation locale."
            ),
            "diff_mode": "local_exact_match",
        }
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
            "footnotes_added": [{"id": "string", "text": "string", "reason": "string"}],
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
        usage_recorder=usage_recorder,
        call_kind="diff_pair",
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
        "diff_mode": "gpt",
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
    previous_cards = [
        _table_card(entry, extraction_index=i)
        for i, entry in enumerate(previous_tables)
    ]
    current_cards = [
        _table_card(entry, extraction_index=i) for i, entry in enumerate(current_tables)
    ]
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
    usage_records: list[dict[str, Any]] = []

    match_result = _match_tables(
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
            usage_recorder=usage_records,
        )
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
    comparison_runtime_sec = round(
        max(0.0, time.monotonic() - comparison_started_at), 3
    )
    comparison_metrics = _aggregate_usage_metrics(usage_records)
    comparison_metrics["runtime_comparison_sec"] = comparison_runtime_sec
    comparison_metrics["comparison_local_diff_skips"] = sum(
        1
        for item in pair_comparisons
        if str(item.get("diff_mode", "") or "") == "local_exact_match"
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
        "comparison_local_diff_skips": comparison_metrics[
            "comparison_local_diff_skips"
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
        },
        "pair_comparisons": pair_comparisons,
        "run_metrics": run_metrics,
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
