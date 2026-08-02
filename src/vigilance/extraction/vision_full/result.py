"""Resultat d'extraction Vision et sa serialisation vers le cache.

Extrait de ``vision_full_extractor.py`` sans modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .constants import _EXTRACTION_METHOD


@dataclass
class VisionFullResult:
    """Resultat de l'extraction minimale par Vision."""

    table_title: str
    table_summary: str
    headers: list[str]
    indicators: list[str]
    footnotes_content: list[dict[str, str]]
    no_table_detected: bool = False
    extraction_method: str = _EXTRACTION_METHOD
    vision_status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    retry_reasons: list[str] = field(default_factory=list)
    requested_max_completion_tokens: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    rescue_used: bool = False
    recrop_attempted: bool = False
    recrop_used: bool = False
    recrop_failed_incomplete: bool = False
    extraction_status: str = "ok"
    acceptance_reason: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    selected_candidate_name: str | None = None
    no_table_evidence_count: int = 0
    summary_present: bool = False
    indicator_count: int = 0
    candidate_quality_rank: list[int] = field(default_factory=list)
    qa_inspected: bool = False
    confidence_score: float = 0.0
    selected_bbox_norm: list[float] | None = None
    bbox_source: str = "docling"
    bbox_confidence: float | None = None
    page_context_title: str = ""
    page_context_continuation: bool | None = None
    page_context_table_count: int | None = None

    def to_footnotes_list(self) -> list[dict[str, str]]:
        """Retourne une copie de la liste des notes de bas de page."""
        return list(self.footnotes_content)


def _cache_payload_from_result(result: VisionFullResult) -> dict[str, Any]:
    """Construit le payload de cache a partir d'un resultat d'extraction."""
    return {
        "table_title": result.table_title,
        "table_summary": result.table_summary,
        "headers": result.headers,
        "indicators": result.indicators,
        "footnotes_content": result.footnotes_content,
        "no_table_detected": result.no_table_detected,
        "vision_status": result.vision_status,
        "warnings": result.warnings,
        "retry_reasons": result.retry_reasons,
        "requested_max_completion_tokens": result.requested_max_completion_tokens,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "rescue_used": result.rescue_used,
        "recrop_attempted": result.recrop_attempted,
        "recrop_used": result.recrop_used,
        "recrop_failed_incomplete": result.recrop_failed_incomplete,
        "extraction_status": result.extraction_status,
        "acceptance_reason": result.acceptance_reason,
        "rejection_reasons": result.rejection_reasons,
        "selected_candidate_name": result.selected_candidate_name,
        "no_table_evidence_count": result.no_table_evidence_count,
        "summary_present": result.summary_present,
        "indicator_count": result.indicator_count,
        "candidate_quality_rank": result.candidate_quality_rank,
        "qa_inspected": result.qa_inspected,
        "confidence_score": result.confidence_score,
        "selected_bbox_norm": result.selected_bbox_norm,
        "bbox_source": result.bbox_source,
        "bbox_confidence": result.bbox_confidence,
        "page_context_title": result.page_context_title,
        "page_context_continuation": result.page_context_continuation,
        "page_context_table_count": result.page_context_table_count,
    }


def _result_from_cache_payload(payload: dict[str, Any]) -> VisionFullResult | None:
    """Reconstruire un resultat Vision valide depuis un payload de cache."""
    indicators_raw = payload.get("indicators")
    if not isinstance(indicators_raw, list):
        return None
    indicators: list[str] = []
    for item in indicators_raw:
        if isinstance(item, str) and item.strip():
            indicators.append(item.strip())
        elif isinstance(item, dict) and item.get("text"):
            indicators.append(str(item.get("text") or "").strip())

    footnotes_raw = payload.get("footnotes_content", [])
    if isinstance(footnotes_raw, dict):
        footnotes = [
            {"id": str(key), "text": str(value)}
            for key, value in footnotes_raw.items()
            if str(key).strip() and str(value).strip()
        ]
    elif isinstance(footnotes_raw, list):
        footnotes = [
            {
                "id": str(item.get("id") or item.get("marker") or "").strip(),
                "text": str(item.get("text") or "").strip(),
            }
            for item in footnotes_raw
            if isinstance(item, dict)
            and (item.get("id") or item.get("marker"))
            and item.get("text")
        ]
    else:
        footnotes = []

    values = {
        key: value
        for key, value in payload.items()
        if key in {item.name for item in fields(VisionFullResult)}
    }
    values.update(
        {
            "table_title": str(payload.get("table_title") or ""),
            "table_summary": str(payload.get("table_summary") or ""),
            "headers": list(payload.get("headers") or []),
            "indicators": indicators,
            "footnotes_content": footnotes,
            "no_table_detected": bool(payload.get("no_table_detected", False)),
            "extraction_method": _EXTRACTION_METHOD,
        }
    )
    try:
        return VisionFullResult(**values)
    except (TypeError, ValueError):
        return None
