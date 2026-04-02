"""Context helpers for Dash callbacks: PDF paths, quarter context, export names.

Extracted from dash_app/app.py. app.py re-exports all names from this module
so that all existing monkeypatches (setattr on dash_app) continue to work.
"""

from __future__ import annotations

from pathlib import Path

from vigilance.quarter_utils import build_quarter_context


def _comparison_path_from_meta(
    indicator_meta: dict | None, indicator_result: dict | None = None
) -> str:
    """Return the persisted comparison path from available Dash state."""
    meta = indicator_meta if isinstance(indicator_meta, dict) else {}
    compare_path = str(meta.get("compare_path") or "").strip()
    if compare_path:
        return compare_path
    if isinstance(indicator_result, dict):
        result_meta = indicator_result.get("meta", {}) or {}
        return str(result_meta.get("compare_path") or "").strip()
    return ""


def _quarter_context_from_store(data: dict | None) -> dict:
    if isinstance(data, dict):
        current = data.get("current")
        previous = data.get("previous")
        if isinstance(current, dict) and isinstance(previous, dict):
            return data
    return build_quarter_context("T2", year=2025)


def _pdf_paths_from_comparison_meta(
    indicator_meta: dict | None,
    indicator_result: dict | None = None,
) -> dict[str, str]:
    meta: dict[str, object] = {}
    top_level: dict[str, object] = {}
    if isinstance(indicator_result, dict):
        top_level = indicator_result
        result_meta = indicator_result.get("meta", {})
        if isinstance(result_meta, dict):
            meta.update(result_meta)
    if isinstance(indicator_meta, dict):
        meta.update(indicator_meta)

    raw_paths = meta.get("pdf_paths")
    if isinstance(raw_paths, dict):
        previous = str(
            raw_paths.get("pdf_previous")
            or raw_paths.get("pdf_t1")
            or meta.get("archived_pdf_previous")
            or meta.get("source_pdf_previous")
            or ""
        ).strip()
        current = str(
            raw_paths.get("pdf_current")
            or raw_paths.get("pdf_t2")
            or meta.get("archived_pdf_current")
            or meta.get("source_pdf_current")
            or ""
        ).strip()
    else:
        previous = str(
            meta.get("archived_pdf_previous")
            or top_level.get("archived_pdf_previous")
            or meta.get("source_pdf_previous")
            or top_level.get("source_pdf_previous")
            or ""
        ).strip()
        current = str(
            meta.get("archived_pdf_current")
            or top_level.get("archived_pdf_current")
            or meta.get("source_pdf_current")
            or top_level.get("source_pdf_current")
            or ""
        ).strip()

    compare_path_raw = str(
        meta.get("compare_path") or top_level.get("compare_path") or ""
    ).strip()
    if compare_path_raw and (not previous or not current):
        compare_path = Path(compare_path_raw)
        run_dir = compare_path.parent if compare_path.suffix else compare_path
        sibling_previous = run_dir / "previous_report.pdf"
        sibling_current = run_dir / "current_report.pdf"
        if not previous and sibling_previous.exists():
            previous = str(sibling_previous)
        if not current and sibling_current.exists():
            current = str(sibling_current)

    return {
        "pdf_t1": previous,
        "pdf_t2": current,
        "pdf_previous": previous,
        "pdf_current": current,
    }


def _normalize_pdf_paths_store(paths: dict | None) -> dict[str, str]:
    source = paths if isinstance(paths, dict) else {}
    previous = str(source.get("pdf_previous") or source.get("pdf_t1") or "").strip()
    current = str(source.get("pdf_current") or source.get("pdf_t2") or "").strip()
    return {
        "pdf_t1": previous,
        "pdf_t2": current,
        "pdf_previous": previous,
        "pdf_current": current,
    }


def _missing_pdf_warning(paths: dict[str, str] | None) -> str:
    if not isinstance(paths, dict):
        return (
            "Comparaison chargée, mais les PDF archivés de preuve sont indisponibles."
        )
    missing: list[str] = []
    previous = str(paths.get("pdf_previous") or paths.get("pdf_t1") or "").strip()
    current = str(paths.get("pdf_current") or paths.get("pdf_t2") or "").strip()
    if not previous or not Path(previous).exists():
        missing.append("rapport précédent")
    if not current or not Path(current).exists():
        missing.append("rapport courant")
    if not missing:
        return ""
    joined = " et ".join(missing)
    return (
        "Comparaison chargée, mais la preuve PDF archivée est indisponible pour le "
        f"{joined}."
    )
