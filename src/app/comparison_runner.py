"""Dash orchestration for section-targeted extraction + GPT comparison."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.comparison_canonical import new_empty_ui_comparison_payload, to_canonical_payload
from app.extraction_storage import build_extraction_manifest, is_stored_manifest_compatible
from app.ui_config import OUTPUT_DIR
from vigilance.config import resolve_openai_model
from vigilance.compare_gpt import (
    REFERENCE_RESOLUTION_RULE,
    compare_reports_gpt4o,
    normalize_quarter,
    resolve_reference_period,
)
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.extraction.vision_extraction_writer import write_compact_report_artifacts
from vigilance.utils.genai import get_openai_api_key
from vigilance.utils.model_cost import estimate_openai_cost_usd

EXTRACTION_ROOT = OUTPUT_DIR / "extractions"
COMPARISON_ROOT = OUTPUT_DIR / "comparisons"
logger = logging.getLogger(__name__)


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _empty_extraction_metrics(*, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "runtime_sec": 0.0,
        "vision_calls_total": 0,
        "vision_rescue_total": 0,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "total_tokens_total": 0,
        "estimated_cost_usd": 0.0,
        "tables_total": 0,
        "cache_hit": mode == "stored",
    }


def _extract_metrics_from_tables(
    tables: list[Any],
    *,
    runtime_sec: float,
    mode: str,
) -> dict[str, Any]:
    metrics = _empty_extraction_metrics(mode=mode)
    metrics["runtime_sec"] = round(max(0.0, float(runtime_sec or 0.0)), 3)
    metrics["tables_total"] = len(tables)
    model_name = ""
    for table in tables:
        dm = getattr(table, "debug_metrics", None)
        if not isinstance(dm, dict):
            continue
        if dm.get("vision_extraction_attempted"):
            metrics["vision_calls_total"] += 1
        if dm.get("vision_max_completion_tokens_rescue_used"):
            metrics["vision_rescue_total"] += 1
        metrics["prompt_tokens_total"] += _coerce_int(dm.get("vision_prompt_tokens"))
        metrics["completion_tokens_total"] += _coerce_int(
            dm.get("vision_completion_tokens")
        )
        metrics["total_tokens_total"] += _coerce_int(dm.get("vision_total_tokens"))
        if not model_name:
            model_name = str(dm.get("vision_model", "") or "").strip()
    if metrics["total_tokens_total"] == 0:
        metrics["total_tokens_total"] = (
            metrics["prompt_tokens_total"] + metrics["completion_tokens_total"]
        )
    metrics["estimated_cost_usd"] = estimate_openai_cost_usd(
        model_name,
        prompt_tokens=metrics["prompt_tokens_total"],
        completion_tokens=metrics["completion_tokens_total"],
    )
    if model_name:
        metrics["model_version"] = model_name
    return metrics


def _stored_manifest_matches(
    payload: dict[str, Any],
    expected_manifest: dict[str, Any],
) -> bool:
    stored_manifest = payload.get("extraction_manifest")
    if not isinstance(stored_manifest, dict) or not stored_manifest:
        return False
    return is_stored_manifest_compatible(
        {"extraction_manifest": stored_manifest},
        expected_manifest,
    )


def _quarter_label(quarter: str, year: int) -> str:
    code = normalize_quarter(quarter)
    return f"Q{code[1]}-{int(year)}"


def _extract_year(value: Any) -> int | None:
    text = str(value or "").strip()
    match = re.search(r"((?:19|20)\d{2})", text)
    if match:
        return int(match.group(1))
    return None


def _build_section_ranges(sections: Any) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for entry in list(sections or []):
        if not isinstance(entry, dict):
            continue
        start = int(entry.get("start_page", 0) or 0)
        end = int(entry.get("end_page", start) or start)
        if start <= 0 or end < start:
            continue
        section = str(
            entry.get("section")
            or entry.get("type")
            or entry.get("section_key")
            or ""
        ).strip()
        section = canonicalize_section(section) or "unknown_section"
        ranges.append({"section": section, "start": start, "end": end})
    return ranges


def _artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "tables": out_dir / "tables.json",
        "indicators": out_dir / "indicators.json",
        "footnotes": out_dir / "footnotes.json",
    }


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalide: {path}")
    return data


def _resolve_pdf_input_path(value: Any, *, label: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        logger.warning("[run_comparison_with_sections] missing_pdf_path label=%s", label)
        raise ValueError(f"{label} introuvable. Veuillez recharger le PDF avant l'analyse.")
    path = Path(raw)
    if not path.exists() or not path.is_file():
        logger.warning(
            "[run_comparison_with_sections] invalid_pdf_path label=%s path=%s",
            label,
            path,
        )
        raise ValueError(f"{label} introuvable: {path}")
    return str(path)


def _extract_tables(
    *,
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int,
    sections: Any,
    use_vision_extraction: bool | None = None,
    use_stored_extraction_if_available: bool = False,
    return_provenance: bool = False,
) -> Any:
    from vigilance.extraction.docling_processor import extract_tables_docling_by_sections

    extraction_started_at = time.monotonic()
    quarter_code = normalize_quarter(quarter)
    out_dir = EXTRACTION_ROOT / str(bank_code).lower() / str(int(year)) / quarter_code
    paths = _artifact_paths(out_dir)
    artifacts_present = {name: path.exists() for name, path in paths.items()}
    section_ranges = _build_section_ranges(sections)
    if not section_ranges:
        raise ValueError("Aucune section valide fournie pour l'extraction.")
    extraction_mode = (
        "vision_full_gpt4o" if use_vision_extraction is not False else "docling_only"
    )
    expected_manifest = build_extraction_manifest(
        pdf_path=pdf_path,
        section_ranges=section_ranges,
        extraction_mode=extraction_mode,
    )

    mode = "fresh"
    source_metrics: dict[str, Any] = {}
    if use_stored_extraction_if_available and all(artifacts_present.values()):
        payload = _load_json(paths["tables"])
        if _stored_manifest_matches(payload, expected_manifest):
            tables = list(payload.get("tables", []) or [])
            mode = "stored"
            source_metrics = dict(payload.get("metrics") or {})
            extraction_metrics = _empty_extraction_metrics(mode=mode)
            extraction_metrics["runtime_sec"] = round(
                max(0.0, time.monotonic() - extraction_started_at), 3
            )
            extraction_metrics["tables_total"] = len(tables)
        else:
            logger.info(
                "[_extract_tables] manifest_mismatch bank=%s year=%s quarter=%s -> reextract",
                str(bank_code).lower(),
                int(year),
                quarter_code,
            )

    if mode == "fresh":
        tables = extract_tables_docling_by_sections(
            pdf_path=pdf_path,
            bank_code=str(bank_code).lower(),
            quarter=quarter_code,
            year=int(year),
            section_ranges=section_ranges,
            use_vision_extraction=use_vision_extraction,
        )
        extraction_metrics = _extract_metrics_from_tables(
            tables,
            runtime_sec=time.monotonic() - extraction_started_at,
            mode=mode,
        )
        try:
            extraction_model_version = resolve_openai_model("extraction_primary")
        except Exception:
            extraction_model_version = ""
        write_compact_report_artifacts(
            tables=tables,
            out_dir=out_dir,
            bank_code=str(bank_code).lower(),
            year=int(year),
            quarter=quarter_code,
            meta={
                "model_version": extraction_model_version,
                "prompt_version": "extract_v1",
                "pdf_fingerprint": expected_manifest.get("pdf_fingerprint", ""),
                "extraction_manifest": expected_manifest,
                "metrics": extraction_metrics,
            },
        )
        artifacts_present = {name: path.exists() for name, path in paths.items()}
        source_metrics = dict(extraction_metrics)

    if mode == "fresh" and "extraction_metrics" not in locals():
        extraction_metrics = _empty_extraction_metrics(mode=mode)
    if mode == "stored" and "extraction_metrics" not in locals():
        extraction_metrics = _empty_extraction_metrics(mode=mode)
        extraction_metrics["runtime_sec"] = round(
            max(0.0, time.monotonic() - extraction_started_at), 3
        )
        extraction_metrics["tables_total"] = len(tables)

    provenance = {
        "mode": mode,
        "artifact_dir": str(out_dir),
        "tables_path": str(paths["tables"]),
        "indicators_path": str(paths["indicators"]),
        "footnotes_path": str(paths["footnotes"]),
        "artifacts_present": artifacts_present,
        "quarter": quarter_code,
        "year": int(year),
        "extraction_manifest": expected_manifest,
        "run_metrics": extraction_metrics,
        "source_metrics": source_metrics,
    }
    if return_provenance:
        return tables, provenance
    return tables


def _empty_result(
    bank_code: str,
    year: int,
    message: str,
    *,
    quarter_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = new_empty_ui_comparison_payload()
    payload["bank_code"] = str(bank_code).lower()
    payload["year"] = int(year)
    if isinstance(quarter_context, dict):
        previous = quarter_context.get("previous", {}) or {}
        current = quarter_context.get("current", {}) or {}
        payload["quarter_from"] = str(previous.get("label", "") or "")
        payload["quarter_to"] = str(current.get("label", "") or "")
        payload["previous_quarter"] = payload["quarter_from"]
        payload["current_quarter"] = payload["quarter_to"]
        payload["meta"]["quarter_context"] = quarter_context
    payload["meta"]["source_format"] = "dash_runner_empty"
    payload["meta"]["executive_summary"] = {"content": str(message or "")}
    payload["meta"]["extraction_sources"] = {
        "previous": {
            "mode": "unknown",
            "quarter": str((quarter_context or {}).get("previous", {}).get("code") or ""),
            "artifacts_present": {"tables": False, "indicators": False, "footnotes": False},
        },
        "current": {
            "mode": "unknown",
            "quarter": str((quarter_context or {}).get("current", {}).get("code") or ""),
            "artifacts_present": {"tables": False, "indicators": False, "footnotes": False},
        },
    }
    return payload


def run_comparison_with_sections(
    *,
    pdf_path_previous: str | None = None,
    pdf_path_current: str | None = None,
    pdf_path_t1: str | None = None,
    pdf_path_t2: str | None = None,
    bank_code: str,
    sections_previous: Any | None = None,
    sections_current: Any | None = None,
    sections_t1: Any | None = None,
    sections_t2: Any | None = None,
    current_quarter: str | None = None,
    previous_quarter: str | None = None,
    current_year: int | None = None,
    previous_year: int | None = None,
    use_genai: bool = True,
    api_key: str | None = None,
    use_vision_extraction: bool | None = None,
    include_footnotes: bool = True,
    include_genai_classification: bool = False,
    use_stored_extraction_if_available: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Extract validated sections, run GPT comparison, and return Dash canonical payload."""
    del include_footnotes, include_genai_classification

    path_previous = _resolve_pdf_input_path(
        pdf_path_previous or pdf_path_t1,
        label="Rapport précédent",
    )
    path_current = _resolve_pdf_input_path(
        pdf_path_current or pdf_path_t2,
        label="Rapport courant",
    )
    logger.info(
        "[run_comparison_with_sections] pdf_previous=%s pdf_current=%s",
        path_previous,
        path_current,
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key).strip()
    if use_genai and not get_openai_api_key():
        raise RuntimeError(
            "OPENAI_API_KEY absente. Ajouter la clé dans .env avant de lancer l'analyse."
        )

    current_quarter_value = current_quarter or "Q2"
    current_year_value = int(
        current_year
        or _extract_year(current_quarter_value)
        or datetime.now().year
    )
    current_quarter_code = normalize_quarter(current_quarter_value)
    resolved_previous_year, resolved_previous_quarter_code = resolve_reference_period(
        current_year_value,
        current_quarter_code,
    )
    previous_year_value = int(previous_year or resolved_previous_year)
    previous_quarter_code = normalize_quarter(
        previous_quarter or resolved_previous_quarter_code
    )

    if previous_year is None or previous_quarter is None:
        previous_year_value = resolved_previous_year
        previous_quarter_code = resolved_previous_quarter_code

    current_label = _quarter_label(current_quarter_code, current_year_value)
    previous_label = _quarter_label(previous_quarter_code, previous_year_value)
    quarter_context = {
        "previous": {
            "code": previous_quarter_code,
            "label": previous_label,
            "year": previous_year_value,
        },
        "current": {
            "code": current_quarter_code,
            "label": current_label,
            "year": current_year_value,
        },
        "comparison_direction": "current_vs_previous",
        "comparison_label": f"{current_label} vs {previous_label}",
    }

    previous_sections_value = sections_previous if sections_previous is not None else sections_t1
    current_sections_value = sections_current if sections_current is not None else sections_t2
    if not previous_sections_value or not current_sections_value:
        return _empty_result(
            bank_code,
            current_year_value,
            "Aucune section valide fournie.",
            quarter_context=quarter_context,
        )

    extraction_started_at = time.monotonic()
    _, previous_provenance = _extract_tables(
        pdf_path=path_previous,
        bank_code=bank_code,
        quarter=previous_quarter_code,
        year=previous_year_value,
        sections=previous_sections_value,
        use_vision_extraction=use_vision_extraction,
        use_stored_extraction_if_available=use_stored_extraction_if_available,
        return_provenance=True,
    )
    _, current_provenance = _extract_tables(
        pdf_path=path_current,
        bank_code=bank_code,
        quarter=current_quarter_code,
        year=current_year_value,
        sections=current_sections_value,
        use_vision_extraction=use_vision_extraction,
        use_stored_extraction_if_available=use_stored_extraction_if_available,
        return_provenance=True,
    )
    extraction_runtime_sec = round(max(0.0, time.monotonic() - extraction_started_at), 3)

    comparison_path = compare_reports_gpt4o(
        previous_dir=Path(previous_provenance["artifact_dir"]),
        current_dir=Path(current_provenance["artifact_dir"]),
        out_root=COMPARISON_ROOT,
        reference_resolution={
            "mode": "automatique",
            "year_previous": previous_year_value,
            "quarter_previous": previous_quarter_code,
            "rule": REFERENCE_RESOLUTION_RULE,
        },
        source_pdf_previous=path_previous,
        source_pdf_current=path_current,
        runtime_extraction_sec=extraction_runtime_sec,
        extraction_run_metrics={
            "previous": dict(previous_provenance.get("run_metrics") or {}),
            "current": dict(current_provenance.get("run_metrics") or {}),
        },
    )
    raw_payload = _load_json(comparison_path)
    canonical = to_canonical_payload(raw_payload)
    meta = canonical.setdefault("meta", {})
    meta["compare_path"] = str(comparison_path)
    meta["quarter_context"] = quarter_context
    meta["reference_resolution"] = dict(raw_payload.get("reference_resolution") or {})
    meta["run_metrics"] = dict(raw_payload.get("run_metrics") or {})
    meta["extraction_sources"] = {
        "previous": previous_provenance,
        "current": current_provenance,
    }
    return canonical
