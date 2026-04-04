"""Metrics aggregation and quality signal functions for the comparison pipeline.

Extracted from compare_gpt.py. compare_gpt.py re-exports all names from this module
so that all existing imports remain valid.
"""

from __future__ import annotations

from typing import Any, TypedDict

from vigilance.comparison_io import _coerce_float, _coerce_int
from vigilance.utils.model_cost import (
    estimate_openai_cost_usd,  # noqa: E402 — used in _build_run_metrics
)

# ---------------------------------------------------------------------------
# TypedDict interfaces (annotation-only)
# ---------------------------------------------------------------------------


class UsageMetrics(TypedDict):
    prompt_tokens_total: int
    completion_tokens_total: int
    total_tokens_total: int
    comparison_calls_total: int


# ---------------------------------------------------------------------------
# Pair-level change counts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Usage / cost aggregation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Full run-metrics assembly
# ---------------------------------------------------------------------------

_MATCHING_METRIC_KEYS = (
    "matching_passes_total",
    "inspector_passes_total",
    "audit_passes_total",
    "matching_output_retries_total",
    "matching_validation_failures_total",
    "stage1_validation_retries_total",
    "stage2_validation_retries_total",
    "unresolved_after_stage1_total",
    "matched_in_stage2_total",
    "unmatched_after_primary_total",
    "unmatched_after_rescue_total",
    "matching_pairs_llm_duplicates_total",
    "matching_pairs_llm_deduped_total",
    "inspector_rejected_total",
    "inspector_confirmed_total",
)


def _build_run_metrics(
    *,
    usage_records: list[dict[str, Any]],
    match_result: dict[str, Any],
    diff_calls_total: int,
    comparison_runtime_sec: float,
    model_name: str,
    extraction_run_metrics: dict[str, Any] | None,
    runtime_extraction_sec: float,
) -> dict[str, Any]:
    """Assemble the final ``run_metrics`` dict for ``comparison.json``."""
    comparison_metrics = _aggregate_usage_metrics(usage_records)
    comparison_metrics["runtime_comparison_sec"] = comparison_runtime_sec
    for key in _MATCHING_METRIC_KEYS:
        comparison_metrics[key] = _coerce_int(match_result.get(key))
    comparison_metrics["comparison_calls_total"] = max(
        _coerce_int(comparison_metrics.get("comparison_calls_total")),
        comparison_metrics["matching_passes_total"]
        + comparison_metrics["inspector_passes_total"]
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

    run_metrics: dict[str, Any] = {
        "runtime_extraction_sec": extraction_metrics["runtime_extraction_sec"],
        "runtime_comparison_sec": comparison_metrics["runtime_comparison_sec"],
        "vision_calls_total": extraction_metrics["vision_calls_total"],
        "vision_rescue_total": extraction_metrics["vision_rescue_total"],
        "comparison_calls_total": comparison_metrics["comparison_calls_total"],
    }
    for key in _MATCHING_METRIC_KEYS:
        run_metrics[key] = comparison_metrics[key]
    run_metrics["prompt_tokens_total"] = (
        extraction_metrics["prompt_tokens_total"]
        + comparison_metrics["prompt_tokens_total"]
    )
    run_metrics["completion_tokens_total"] = (
        extraction_metrics["completion_tokens_total"]
        + comparison_metrics["completion_tokens_total"]
    )
    run_metrics["total_tokens_total"] = (
        extraction_metrics["total_tokens_total"]
        + comparison_metrics["total_tokens_total"]
    )
    run_metrics["estimated_cost_usd"] = round(
        float(extraction_metrics["estimated_cost_usd"] or 0.0)
        + float(comparison_metrics["estimated_cost_usd"] or 0.0),
        6,
    )
    run_metrics["extraction"] = extraction_metrics
    run_metrics["comparison"] = comparison_metrics
    return run_metrics
