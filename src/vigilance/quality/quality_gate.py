"""Strict Quality Gate for extraction outputs (indicators + footnotes) and extraction certification."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from vigilance.models.table_models import (
    EXTRACTION_STATUS_BLOCKED,
    EXTRACTION_STATUS_CERTIFIED,
    EXTRACTION_STATUS_REVIEW_REQUIRED,
    derive_extraction_blockers,
    get_extraction_confidence,
    get_extraction_quality_flags,
    get_extraction_status,
)
from vigilance.utils.indicator_cleaner import (
    clean_table_title_contamination,
    is_table_title_contaminated,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "duplicate_ratio_threshold": 0.15,
    "max_tables_duplicate_excess": 3,
    "max_contaminated_titles": 2,
    "allow_date_header_titles": True,
    "line_split_max_per_table": 2,
    "missing_marker_majority_threshold": 0.5,
    "title_numeric_density_threshold": 0.45,
}

# Extraction certification: strict quality-first defaults (fail on any blocked table).
_DEFAULT_EXTRACTION_CONFIG: dict[str, Any] = {
    "max_tables_blocked": 0,
    "max_tables_review_required": 10,
    "fail_on_budget_exhausted": True,
}

_NUMERIC_TOKEN_RE = re.compile(r"^\d[\d,.\u00a0]*$")
_MARKER_ONLY_RE = re.compile(
    r"^\s*(?:[\(\[\{]?\d+[\)\]\}]?|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|[*†‡§¶‖]+)\s*$"
)
_PUNCT_ONLY_RE = re.compile(r"^\s*[\-–—.,:;!?/\\|_+=~`\"'()\[\]{}]+\s*$")
_DICTISH_RE = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)
_MONTH_TOKEN_RE = (
    r"(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|"
    r"novembre|decembre|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
)
_FULL_DATE_RE = re.compile(rf"\b\d{{1,2}}\s+{_MONTH_TOKEN_RE}\s+\d{{4}}\b")

_SUPERSCRIPT_TO_DIGIT = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_CONFIG)
    if isinstance(config, dict):
        merged.update(config)
    return merged


def evaluate_extraction_quality(
    tables: list[Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate extraction certification across tables. Returns status, fail_reasons, and summary.
    Used by the run-level quality gate; strict defaults (any blocked => FAIL).
    """
    cfg = dict(_DEFAULT_EXTRACTION_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    max_blocked = int(cfg.get("max_tables_blocked", 0))
    max_review_required = int(cfg.get("max_tables_review_required", 10))
    fail_on_budget = bool(cfg.get("fail_on_budget_exhausted", True))

    tables_blocked = 0
    tables_review_required = 0
    tables_certified = 0
    tables_crop_rejected = 0
    tables_low_confidence = 0
    tables_budget_exhausted = 0

    for table in tables or []:
        status = get_extraction_status(table)
        if status == EXTRACTION_STATUS_BLOCKED:
            tables_blocked += 1
        elif status == EXTRACTION_STATUS_REVIEW_REQUIRED:
            tables_review_required += 1
        else:
            tables_certified += 1
        flags = get_extraction_quality_flags(table)
        if flags.get("crop_rejected"):
            tables_crop_rejected += 1
        if get_extraction_confidence(table) < 0.5:
            tables_low_confidence += 1
        dm = getattr(table, "debug_metrics", None)
        if isinstance(dm, dict) and dm.get("vision_budget_exhausted"):
            tables_budget_exhausted += 1

    fail_reasons: list[str] = []
    if tables_blocked > max_blocked:
        fail_reasons.append(
            f"extraction_blocked_tables={tables_blocked}>max({max_blocked})"
        )
    if tables_review_required > max_review_required:
        fail_reasons.append(
            f"extraction_review_required_tables={tables_review_required}>max({max_review_required})"
        )
    if fail_on_budget and tables_budget_exhausted > 0:
        fail_reasons.append(
            f"extraction_budget_exhausted_tables={tables_budget_exhausted}"
        )

    status = "FAIL" if fail_reasons else "PASS"
    eligible_for_review = status == "PASS"

    return {
        "status": status,
        "eligible_for_review": eligible_for_review,
        "fail_reasons": fail_reasons,
        "summary": {
            "tables_total": len(tables or []),
            "tables_certified": tables_certified,
            "tables_review_required": tables_review_required,
            "tables_blocked": tables_blocked,
            "tables_crop_rejected": tables_crop_rejected,
            "tables_low_confidence": tables_low_confidence,
            "tables_budget_exhausted": tables_budget_exhausted,
        },
    }


def _safe_read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def _normalize_text(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _looks_numeric_token(tok: str) -> bool:
    value = str(tok or "").replace("\u00a0", "").strip()
    return bool(value) and bool(_NUMERIC_TOKEN_RE.match(value))


def _title_numeric_density(title: str) -> float:
    tokens = [t for t in str(title or "").split() if t.strip()]
    if not tokens:
        return 0.0
    numeric = sum(1 for tok in tokens if _looks_numeric_token(tok))
    return numeric / len(tokens)


def _has_trailing_numeric_run(title: str, min_run: int = 2) -> bool:
    tokens = [t for t in str(title or "").split() if t.strip()]
    if not tokens:
        return False
    run = 0
    for tok in reversed(tokens):
        if _looks_numeric_token(tok):
            run += 1
            if run >= min_run:
                return True
        else:
            break
    return False


def _is_repr_like(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if not _DICTISH_RE.match(value):
        return False
    low = value.lower()
    return (
        "'text':" in low
        or "\"text\":" in low
        or "'id':" in low
        or "\"id\":" in low
        or "'marker':" in low
        or "\"marker\":" in low
        or "'value':" in low
        or "\"value\":" in low
    )


def _duplicate_ratio(lines: list[str]) -> float:
    cleaned = [_normalize_text(x) for x in lines if _normalize_text(x)]
    if not cleaned:
        return 0.0
    unique = len(set(cleaned))
    return max(0.0, 1.0 - (unique / len(cleaned)))


def _normalize_marker(marker: str) -> str:
    raw = str(marker or "").translate(_SUPERSCRIPT_TO_DIGIT).strip()
    raw = raw.replace("\u00a0", " ")
    raw = re.sub(r"^[\(\[\{]\s*", "", raw)
    raw = re.sub(r"\s*[\)\]\}]$", "", raw)
    raw = re.sub(r"\s+", "", raw)
    return raw


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9¹²³⁴⁵⁶⁷⁸⁹⁰]+", str(text or ""))


def _ascii_fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text or ""))
        if not unicodedata.combining(c)
    ).lower()


def _is_line_split_suspicious(text: str) -> bool:
    value = _normalize_text(text)
    if not value:
        return False

    if _MARKER_ONLY_RE.match(value):
        return True
    if _PUNCT_ONLY_RE.match(value):
        return True

    words = _words(value)
    if not words:
        return False

    first = words[0]
    if len(words) <= 2 and first and first[0].islower():
        return True

    folded = _ascii_fold(value)
    continuation_roots = (
        "supplementaire",
        "additional",
        "continue",
        "suite",
        "seulement",
    )
    if len(words) <= 3 and any(root in folded for root in continuation_roots):
        return True

    return False


def _is_date_header_title(title: str) -> bool:
    """Return True if title looks like a date header rather than a semantic title."""
    folded = _ascii_fold(_normalize_text(title))
    if not folded:
        return False
    has_full_date = bool(_FULL_DATE_RE.search(folded))
    if not has_full_date:
        return False

    # Remove explicit dates and verify only lightweight header glue remains.
    remainder = _FULL_DATE_RE.sub(" ", folded)
    remainder = re.sub(r"[^\w\s]", " ", remainder)
    tokens = [t for t in remainder.split() if t]
    if not tokens:
        return True

    allowed = {
        "au",
        "as",
        "at",
        "of",
        "for",
        "the",
        "pour",
        "le",
        "la",
        "les",
        "des",
        "de",
        "du",
        "d",
        "trimestre",
        "quarter",
        "clos",
        "ended",
        "en",
        "date",
        "au",
    }
    return all(tok in allowed for tok in tokens)


def _is_title_contaminated_basic(title: str, *, title_density_threshold: float) -> bool:
    return bool(
        is_table_title_contaminated(title)
        or _has_trailing_numeric_run(title, min_run=2)
        or (
            _title_numeric_density(title) >= title_density_threshold
            and len(title.split()) >= 3
        )
    )


def _assess_title_quality(
    title: str,
    *,
    title_density_threshold: float,
    allow_date_header_titles: bool,
) -> tuple[bool, bool, bool, str]:
    """
    Return (title_contaminated, date_header_title, title_auto_cleaned, title_effective).

    - date_header_title: date-like heading (allowed, does not fail contamination gate)
    - title_auto_cleaned: contamination detected on raw title but removed by canonical cleaner
    - title_effective: cleaned title used for quality decision
    """
    raw = _normalize_text(title)
    if not raw:
        return False, False, False, raw

    if allow_date_header_titles and _is_date_header_title(raw):
        return False, True, False, raw

    contaminated_raw = _is_title_contaminated_basic(
        raw,
        title_density_threshold=title_density_threshold,
    )
    if not contaminated_raw:
        return False, False, False, raw

    cleaned = _normalize_text(clean_table_title_contamination(raw))
    if cleaned and cleaned != raw:
        has_alpha = bool(re.search(r"[A-Za-zÀ-ÿ]", cleaned))
        contaminated_cleaned = _is_title_contaminated_basic(
            cleaned,
            title_density_threshold=title_density_threshold,
        )
        if has_alpha and not contaminated_cleaned and not _is_date_header_title(cleaned):
            return False, False, True, cleaned

    return True, False, False, (cleaned or raw)


def _make_footnote_index(
    foot_tables: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    exact: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_source_id: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in foot_tables:
        source = str(entry.get("source", "") or "")
        table_id = str(entry.get("table_id", "") or "")
        page = int(entry.get("page", 0) or 0)
        exact[(source, table_id, page)] = entry
        by_source_id.setdefault((source, table_id), []).append(entry)
    return exact, by_source_id


def _find_footnote_entry(
    indicators_entry: dict[str, Any],
    *,
    exact_idx: dict[tuple[str, str, int], dict[str, Any]],
    by_source_id: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    source = str(indicators_entry.get("source", "") or "")
    table_id = str(indicators_entry.get("table_id", "") or "")
    page = int(indicators_entry.get("page", 0) or 0)
    exact = exact_idx.get((source, table_id, page))
    if exact is not None:
        return exact
    candidates = by_source_id.get((source, table_id), [])
    if candidates:
        return candidates[0]
    return None


def _analyze_footnote_integrity(
    foot_entry: dict[str, Any] | None,
    *,
    missing_marker_majority_threshold: float,
) -> tuple[str, list[str]]:
    if not foot_entry:
        return "pass", []

    has_footnotes = bool(foot_entry.get("has_footnotes"))
    if not has_footnotes:
        return "pass", []

    reasons: list[str] = []
    content = foot_entry.get("footnotes_content", {})
    markers = foot_entry.get("footnote_markers", []) or []

    if not isinstance(content, dict) or not content:
        reasons.append("missing_or_invalid_footnotes_content")
        return "fail", reasons

    empty_values = 0
    repr_like_values = 0
    for value in content.values():
        txt = str(value or "").strip()
        if not txt:
            empty_values += 1
            continue
        if _is_repr_like(txt):
            repr_like_values += 1
    if empty_values > 0:
        reasons.append(f"empty_footnote_values({empty_values})")
    if repr_like_values > 0:
        reasons.append(f"repr_like_footnote_values({repr_like_values})")

    marker_norm = {_normalize_marker(m) for m in markers if _normalize_marker(m)}
    key_norm = {_normalize_marker(k) for k in content.keys() if _normalize_marker(k)}

    if key_norm and not marker_norm:
        reasons.append("missing_footnote_markers")
    elif marker_norm:
        missing = marker_norm - key_norm
        missing_ratio = len(missing) / len(marker_norm) if marker_norm else 0.0
        if missing_ratio > float(missing_marker_majority_threshold):
            reasons.append(f"marker_key_mismatch_ratio({missing_ratio:.2f})")

    return ("fail", reasons) if reasons else ("pass", [])


def _score_table(
    *,
    title_contaminated: bool,
    duplicate_ratio: float,
    duplicate_threshold: float,
    suspicious_line_splits: int,
    footnote_integrity: str,
) -> int:
    score = 100.0
    if title_contaminated:
        score -= 20.0
    if duplicate_ratio > duplicate_threshold:
        severity = min(1.0, (duplicate_ratio - duplicate_threshold) / max(0.01, 1.0 - duplicate_threshold))
        score -= 30.0 * severity
    score -= min(25.0, suspicious_line_splits * 8.0)
    if footnote_integrity == "fail":
        score -= 35.0
    return int(max(0, min(100, round(score))))


def evaluate_quality(
    indicators_payload: dict[str, Any],
    footnotes_payload: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _merge_config(config)
    duplicate_threshold = float(cfg["duplicate_ratio_threshold"])
    max_dup = int(cfg["max_tables_duplicate_excess"])
    max_titles = int(cfg["max_contaminated_titles"])
    allow_date_header_titles = bool(cfg.get("allow_date_header_titles", True))
    title_density_threshold = float(cfg["title_numeric_density_threshold"])
    missing_marker_majority_threshold = float(cfg["missing_marker_majority_threshold"])

    indicators_tables = indicators_payload.get("tables", []) or []
    footnotes_tables = footnotes_payload.get("tables", []) or []
    if not isinstance(indicators_tables, list):
        indicators_tables = []
    if not isinstance(footnotes_tables, list):
        footnotes_tables = []

    foot_exact, foot_by_sid = _make_footnote_index(
        [e for e in footnotes_tables if isinstance(e, dict)]
    )

    per_table: list[dict[str, Any]] = []
    duplicates_excess_count = 0
    contaminated_titles_count = 0
    date_header_titles_count = 0
    titles_auto_cleaned_count = 0
    footnote_fails_count = 0

    for idx, entry in enumerate(indicators_tables):
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "") or "")
        sections = entry.get("sections", []) or []
        if not isinstance(sections, list):
            sections = []

        max_dup_ratio = 0.0
        dup_by_section: list[dict[str, Any]] = []
        suspicious_count = 0
        for section in sections:
            if not isinstance(section, dict):
                continue
            sec_name = str(section.get("section", "") or "unknown")
            indicators = section.get("indicators", []) or []
            if not isinstance(indicators, list):
                indicators = []
            lines = [str(x) for x in indicators]
            sec_dup = _duplicate_ratio(lines)
            max_dup_ratio = max(max_dup_ratio, sec_dup)
            dup_by_section.append(
                {
                    "section": sec_name,
                    "indicator_count": len(lines),
                    "duplicate_ratio": round(sec_dup, 4),
                }
            )
            suspicious_count += sum(1 for line in lines if _is_line_split_suspicious(line))

        (
            title_contaminated,
            date_header_title,
            title_auto_cleaned,
            title_effective,
        ) = _assess_title_quality(
            title,
            title_density_threshold=title_density_threshold,
            allow_date_header_titles=allow_date_header_titles,
        )
        if date_header_title:
            date_header_titles_count += 1
        if title_auto_cleaned:
            titles_auto_cleaned_count += 1
        if title_contaminated:
            contaminated_titles_count += 1

        if max_dup_ratio > duplicate_threshold:
            duplicates_excess_count += 1

        foot_entry = _find_footnote_entry(
            entry, exact_idx=foot_exact, by_source_id=foot_by_sid
        )
        foot_status, foot_reasons = _analyze_footnote_integrity(
            foot_entry,
            missing_marker_majority_threshold=missing_marker_majority_threshold,
        )
        if foot_status == "fail":
            footnote_fails_count += 1

        score = _score_table(
            title_contaminated=title_contaminated,
            duplicate_ratio=max_dup_ratio,
            duplicate_threshold=duplicate_threshold,
            suspicious_line_splits=suspicious_count,
            footnote_integrity=foot_status,
        )

        reasons: list[str] = []
        if title_contaminated:
            reasons.append("title_contaminated")
        if max_dup_ratio > duplicate_threshold:
            reasons.append(f"high_duplicate_ratio({max_dup_ratio:.3f})")
        if suspicious_count > int(cfg["line_split_max_per_table"]):
            reasons.append(f"suspicious_line_splits({suspicious_count})")
        reasons.extend(foot_reasons)

        per_table.append(
            {
                "table_key": {
                    "source": str(entry.get("source", "") or ""),
                    "table_id": str(entry.get("table_id", "") or ""),
                    "page": int(entry.get("page", 0) or 0),
                },
                "title": title,
                "title_effective": title_effective,
                "date_header_title": date_header_title,
                "title_auto_cleaned": title_auto_cleaned,
                "title_contaminated": title_contaminated,
                "duplicate_ratio": round(max_dup_ratio, 4),
                "duplicate_ratio_by_section": dup_by_section,
                "suspicious_line_splits": int(suspicious_count),
                "footnote_integrity": foot_status,
                "footnote_reasons": foot_reasons,
                "overall_table_quality_score": score,
                "reasons": reasons,
            }
        )

    fail_reasons: list[str] = []
    if footnote_fails_count > 0:
        fail_reasons.append(f"footnote_integrity_failed_tables={footnote_fails_count}")
    if duplicates_excess_count > max_dup:
        fail_reasons.append(
            f"duplicate_ratio_excess_tables={duplicates_excess_count}>max({max_dup})"
        )
    if contaminated_titles_count > max_titles:
        fail_reasons.append(
            f"contaminated_titles={contaminated_titles_count}>max({max_titles})"
        )

    status = "FAIL" if fail_reasons else "PASS"
    eligible_for_review = status == "PASS"

    worst = sorted(
        per_table,
        key=lambda t: (int(t.get("overall_table_quality_score", 0)), -len(t.get("reasons", []))),
    )

    recommended_actions: list[str] = []
    if contaminated_titles_count:
        recommended_actions.append(
            "Renforcer le nettoyage des titres contamines (runs numeriques en fin de titre)."
        )
    if titles_auto_cleaned_count:
        recommended_actions.append(
            "Ameliorer l'extraction upstream des titres pour eviter la contamination nettoyee a posteriori."
        )
    if duplicates_excess_count:
        recommended_actions.append(
            "Verifier les sections avec duplication d'indicateurs et ajuster dedupe/segmentation."
        )
    if footnote_fails_count:
        recommended_actions.append(
            "Corriger l'integrite des footnotes (contenus vides/repr-like, markers incoherents)."
        )
    if any(int(t.get("suspicious_line_splits", 0)) > 0 for t in per_table):
        recommended_actions.append(
            "Revoir les lignes suspectes de split OCR (tokens isoles, ponctuation seule)."
        )
    if not recommended_actions:
        recommended_actions.append("Aucune action corrective critique detectee.")

    return {
        "status": status,
        "eligible_for_review": eligible_for_review,
        "thresholds": {
            "duplicate_ratio_threshold": duplicate_threshold,
            "max_tables_duplicate_excess": max_dup,
            "max_contaminated_titles": max_titles,
            "allow_date_header_titles": allow_date_header_titles,
            "line_split_max_per_table": int(cfg["line_split_max_per_table"]),
            "missing_marker_majority_threshold": missing_marker_majority_threshold,
            "title_numeric_density_threshold": title_density_threshold,
        },
        "summary": {
            "tables_total": len(per_table),
            "tables_failed_footnote_integrity": footnote_fails_count,
            "tables_duplicate_ratio_excess": duplicates_excess_count,
            "tables_title_contaminated": contaminated_titles_count,
            "tables_date_header_titles": date_header_titles_count,
            "tables_title_auto_cleaned": titles_auto_cleaned_count,
        },
        "fail_reasons": fail_reasons,
        "recommended_actions": recommended_actions,
        "top_worst_tables": worst[:10],
        "tables": per_table,
    }


def _report_markdown(report: dict[str, Any]) -> str:
    status = str(report.get("status", "FAIL"))
    summary = report.get("summary", {}) or {}
    thresholds = report.get("thresholds", {}) or {}
    fail_reasons = report.get("fail_reasons", []) or []
    actions = report.get("recommended_actions", []) or []
    worst = report.get("top_worst_tables", []) or []

    lines: list[str] = []
    lines.append("# Quality Gate Report")
    lines.append("")
    lines.append(f"- Status: **{status}**")
    lines.append(
        f"- Eligible for analyst review: **{bool(report.get('eligible_for_review', False))}**"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Tables analyzed: {int(summary.get('tables_total', 0) or 0)}"
    )
    lines.append(
        f"- Footnote integrity fails: {int(summary.get('tables_failed_footnote_integrity', 0) or 0)}"
    )
    lines.append(
        f"- High duplicate ratio tables: {int(summary.get('tables_duplicate_ratio_excess', 0) or 0)}"
    )
    lines.append(
        f"- Contaminated titles: {int(summary.get('tables_title_contaminated', 0) or 0)}"
    )
    lines.append(
        f"- Date-header titles (exempted): {int(summary.get('tables_date_header_titles', 0) or 0)}"
    )
    lines.append(
        f"- Auto-cleaned titles (non-blocking): {int(summary.get('tables_title_auto_cleaned', 0) or 0)}"
    )
    lines.append("")
    lines.append("## Thresholds")
    for key in (
        "duplicate_ratio_threshold",
        "max_tables_duplicate_excess",
        "max_contaminated_titles",
        "allow_date_header_titles",
        "line_split_max_per_table",
        "missing_marker_majority_threshold",
        "title_numeric_density_threshold",
    ):
        if key in thresholds:
            lines.append(f"- {key}: {thresholds[key]}")

    lines.append("")
    lines.append("## Top 10 Worst Tables")
    if not worst:
        lines.append("- No degraded tables detected.")
    for idx, table in enumerate(worst, start=1):
        key = table.get("table_key", {}) or {}
        source = key.get("source", "")
        table_id = key.get("table_id", "")
        page = key.get("page", 0)
        score = int(table.get("overall_table_quality_score", 0) or 0)
        reasons = table.get("reasons", []) or []
        lines.append(
            f"{idx}. [{source}] {table_id} p.{page} | score={score} | reasons={', '.join(reasons) if reasons else 'none'}"
        )

    lines.append("")
    lines.append("## Recommended Actions")
    for action in actions:
        lines.append(f"- {action}")

    if fail_reasons:
        lines.append("")
        lines.append("## Gate Failure Reasons")
        for reason in fail_reasons:
            lines.append(f"- {reason}")

    lines.append("")
    return "\n".join(lines)


def run_quality_gate(
    *,
    indicators_path: Path,
    footnotes_path: Path,
    out_dir: Path,
    bank_code: str,
    run_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run quality checks and write quality_report.json/.md + quality_gate_status.json."""
    cfg = _merge_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().isoformat(timespec="seconds")
    try:
        indicators_payload = _safe_read_json(indicators_path)
        footnotes_payload = _safe_read_json(footnotes_path)
        report = evaluate_quality(
            indicators_payload, footnotes_payload, config=cfg
        )
    except Exception as exc:
        logger.exception("Quality gate failed to analyze extraction outputs")
        report = {
            "status": "FAIL",
            "eligible_for_review": False,
            "thresholds": {
                "duplicate_ratio_threshold": float(cfg["duplicate_ratio_threshold"]),
                "max_tables_duplicate_excess": int(cfg["max_tables_duplicate_excess"]),
                "max_contaminated_titles": int(cfg["max_contaminated_titles"]),
                "allow_date_header_titles": bool(
                    cfg.get("allow_date_header_titles", True)
                ),
                "line_split_max_per_table": int(cfg["line_split_max_per_table"]),
                "missing_marker_majority_threshold": float(
                    cfg["missing_marker_majority_threshold"]
                ),
                "title_numeric_density_threshold": float(
                    cfg["title_numeric_density_threshold"]
                ),
            },
            "summary": {
                "tables_total": 0,
                "tables_failed_footnote_integrity": 0,
                "tables_duplicate_ratio_excess": 0,
                "tables_title_contaminated": 0,
                "tables_date_header_titles": 0,
                "tables_title_auto_cleaned": 0,
            },
            "fail_reasons": [f"quality_gate_runtime_error({exc})"],
            "recommended_actions": [
                "Verifier la presence/validite de indicators.json et footnotes.json."
            ],
            "top_worst_tables": [],
            "tables": [],
        }

    payload: dict[str, Any] = {
        "bank_code": bank_code,
        "run_id": run_id,
        "generated_at": now,
        **report,
    }

    report_json_path = out_dir / "quality_report.json"
    report_md_path = out_dir / "quality_report.md"
    status_path = out_dir / "quality_gate_status.json"

    report_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_md_path.write_text(_report_markdown(payload), encoding="utf-8")

    status_payload = {
        "bank_code": bank_code,
        "run_id": run_id,
        "generated_at": now,
        "status": payload.get("status", "FAIL"),
        "eligible_for_review": bool(payload.get("eligible_for_review", False)),
        "fail_reasons": list(payload.get("fail_reasons", []) or []),
        "quality_report_json": str(report_json_path),
        "quality_report_md": str(report_md_path),
    }
    status_path.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        **status_payload,
        "quality_gate_status_path": str(status_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extraction quality gate.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to outputs/comparisons/{bank}/extractions/{run_id}",
    )
    parser.add_argument("--bank", default="", help="Bank code override")
    parser.add_argument("--run-id", default="", help="Run id override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    indicators_path = run_dir / "indicators.json"
    footnotes_path = run_dir / "footnotes.json"

    if not indicators_path.exists() or not footnotes_path.exists():
        print(
            f"Missing required files in {run_dir}: indicators.json and footnotes.json",
            flush=True,
        )
        return 2

    bank_code = str(args.bank or "").strip()
    run_id = str(args.run_id or "").strip()
    try:
        indicators_payload = _safe_read_json(indicators_path)
    except Exception:
        indicators_payload = {}
    if not bank_code:
        bank_code = str(indicators_payload.get("bank_code", "") or "unknown")
    if not run_id:
        run_id = str(indicators_payload.get("run_id", "") or run_dir.name)

    result = run_quality_gate(
        indicators_path=indicators_path,
        footnotes_path=footnotes_path,
        out_dir=run_dir,
        bank_code=bank_code,
        run_id=run_id,
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "eligible_for_review": result.get("eligible_for_review"),
                "quality_report_json": result.get("quality_report_json"),
                "quality_report_md": result.get("quality_report_md"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
