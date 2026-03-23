"""Dash orchestration for section-targeted extraction + GPT comparison."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.comparison_canonical import new_empty_ui_comparison_payload, to_canonical_payload
from app.ui_config import OUTPUT_DIR
from vigilance.compare_gpt import (
    REFERENCE_RESOLUTION_RULE,
    compare_reports_gpt4o,
    normalize_quarter,
    resolve_reference_period,
)
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.extraction.vision_extraction_writer import write_compact_report_artifacts
from vigilance.utils.genai import get_openai_api_key

EXTRACTION_ROOT = OUTPUT_DIR / "extractions"
COMPARISON_ROOT = OUTPUT_DIR / "comparisons"
logger = logging.getLogger(__name__)


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

    quarter_code = normalize_quarter(quarter)
    out_dir = EXTRACTION_ROOT / str(bank_code).lower() / str(int(year)) / quarter_code
    paths = _artifact_paths(out_dir)
    artifacts_present = {name: path.exists() for name, path in paths.items()}

    mode = "fresh"
    if use_stored_extraction_if_available and all(artifacts_present.values()):
        payload = _load_json(paths["tables"])
        tables = list(payload.get("tables", []) or [])
        mode = "stored"
    else:
        section_ranges = _build_section_ranges(sections)
        if not section_ranges:
            raise ValueError("Aucune section valide fournie pour l'extraction.")
        tables = extract_tables_docling_by_sections(
            pdf_path=pdf_path,
            bank_code=str(bank_code).lower(),
            quarter=quarter_code,
            year=int(year),
            section_ranges=section_ranges,
            use_vision_extraction=use_vision_extraction,
        )
        write_compact_report_artifacts(
            tables=tables,
            out_dir=out_dir,
            bank_code=str(bank_code).lower(),
            year=int(year),
            quarter=quarter_code,
            meta={},
        )
        artifacts_present = {name: path.exists() for name, path in paths.items()}

    provenance = {
        "mode": mode,
        "artifact_dir": str(out_dir),
        "tables_path": str(paths["tables"]),
        "indicators_path": str(paths["indicators"]),
        "footnotes_path": str(paths["footnotes"]),
        "artifacts_present": artifacts_present,
        "quarter": quarter_code,
        "year": int(year),
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
    )
    raw_payload = _load_json(comparison_path)
    canonical = to_canonical_payload(raw_payload)
    meta = canonical.setdefault("meta", {})
    meta["compare_path"] = str(comparison_path)
    meta["quarter_context"] = quarter_context
    meta["reference_resolution"] = dict(raw_payload.get("reference_resolution") or {})
    meta["extraction_sources"] = {
        "previous": previous_provenance,
        "current": current_provenance,
    }
    return canonical
