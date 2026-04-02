"""File-first comparison store used by Dash.

This module centralizes all Dash access to saved comparison artifacts so the
UI stops coupling directly to raw filesystem helpers. The initial backend is
the local ``comparison.json`` tree, but the interface is shaped so a future
database-backed store can replace it without rewriting every callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict

from vigilance.comparison_canonical import (
    is_canonical_comparison,
    to_canonical_payload,
)
from vigilance.dash_app.services.comparison_context import (
    _missing_pdf_warning,
    _normalize_pdf_paths_store,
    _pdf_paths_from_comparison_meta,
)
from vigilance.quarter_utils import build_quarter_context
from vigilance.review_storage import load_review_state, save_review_state
from vigilance.ui_config import INDICATOR_COMPARISON_DIR
from vigilance.ui_io import load_comparison_result


class ComparisonPayloadBundle(TypedDict):
    """Canonical payload + resolved filesystem context for Dash."""

    compare_path: str
    raw_data: dict[str, Any]
    indicator_result: dict[str, Any]
    indicator_meta: dict[str, Any]
    pdf_paths: dict[str, str]
    warning: str


class ComparisonStore(Protocol):
    """Interface for comparison persistence used by Dash."""

    def list_comparison_options(self) -> list[dict[str, str]]: ...

    def list_saved_run_options(
        self,
        *,
        bank_code: str,
        year: int | str,
        current_quarter: str,
    ) -> list[dict[str, Any]]: ...

    def load_dash_payload(
        self,
        target: str | Path,
        *,
        source: str,
        source_label: str,
    ) -> ComparisonPayloadBundle | None: ...

    def load_review_state(self, compare_path: str | Path | None) -> dict[str, Any] | None: ...

    def save_review_state(
        self,
        compare_path: str | Path | None,
        **kwargs: Any,
    ) -> Path | None: ...


class FileComparisonStore:
    """Comparison store backed by local ``comparison.json`` artifacts."""

    def __init__(self, *, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else INDICATOR_COMPARISON_DIR

    def list_comparison_options(self) -> list[dict[str, str]]:
        if not self.root_dir.exists():
            return []

        files = sorted(
            self.root_dir.glob("**/*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        options: list[dict[str, str]] = []
        for path in files:
            if path.name != "comparison.json":
                continue
            try:
                rel = path.relative_to(self.root_dir)
            except ValueError:
                rel = path
            options.append({"label": rel.as_posix(), "value": rel.as_posix()})
        return options

    def list_saved_run_options(
        self,
        *,
        bank_code: str,
        year: int | str,
        current_quarter: str,
    ) -> list[dict[str, Any]]:
        try:
            ctx = build_quarter_context(current_quarter, year=year)
            curr_y = ctx["current"]["year"]
            curr_q_t = ctx["current"]["code"]
            prev_y = ctx["previous"]["year"]
            prev_q_t = ctx["previous"]["code"]
        except Exception:
            curr_y, curr_q_t = year, "t2"
            prev_y, prev_q_t = year, "t1"

        expected_dir = f"{bank_code}/{curr_y}_{curr_q_t}_vs_{prev_y}_{prev_q_t}"
        filtered_options: list[dict[str, Any]] = []
        for option in self.list_comparison_options():
            value = str(option.get("value", "") or "")
            if not value.startswith(expected_dir):
                continue
            payload = self.load_dash_payload(
                value,
                source="analyse_enregistree",
                source_label="Analyse enregistrée",
            )
            timestamp_label = "Analyse enregistrée"
            if payload is not None:
                created_at = str(payload["raw_data"].get("created_at", "") or "")
                if created_at and "T" in created_at:
                    date_part, time_part = created_at.split("T", 1)
                    time_part = time_part[:5]
                    day_parts = date_part.split("-")
                    if len(day_parts) == 3:
                        timestamp_label = (
                            f"Le {day_parts[2]}/{day_parts[1]} à {time_part}"
                        )

            compare_path = self.resolve_path(value)
            run_dir = compare_path.parent
            has_pdfs = (run_dir / "previous_report.pdf").exists() and (
                run_dir / "current_report.pdf"
            ).exists()
            pdf_icon = "✅" if has_pdfs else "⚠️"
            try:
                relative_parent = compare_path.relative_to(self.root_dir).parent.as_posix()
            except ValueError:
                relative_parent = compare_path.parent.as_posix()
            pretty_parent = relative_parent.replace("_", " ").replace("vs", " vs ")
            label = (
                f"{pdf_icon} {bank_code.upper()} - "
                f"{pretty_parent} "
                f"({timestamp_label})"
            )
            filtered_options.append({"label": label, "value": value})

        filtered_options.sort(key=lambda item: str(item["value"]), reverse=True)
        return filtered_options

    def resolve_path(self, target: str | Path) -> Path:
        candidate = Path(target)
        if candidate.is_absolute():
            return candidate
        return self.root_dir / candidate

    def load_dash_payload(
        self,
        target: str | Path,
        *,
        source: str,
        source_label: str,
    ) -> ComparisonPayloadBundle | None:
        compare_path = self.resolve_path(target)
        raw_data = load_comparison_result(compare_path)
        if not isinstance(raw_data, dict):
            return None

        if raw_data.get("result_type") == "metier_tableaux":
            pdf_paths = _normalize_pdf_paths_store(
                _pdf_paths_from_comparison_meta(
                    {
                        "compare_path": str(compare_path),
                        "source": source,
                        "source_label": source_label,
                    },
                    raw_data,
                )
            )
            indicator_result = raw_data
            indicator_meta = {
                "compare_path": str(compare_path),
                "source": source,
                "source_label": source_label,
                "storage_backend": "fichier_local",
                "pdf_paths": pdf_paths,
            }
            warning = _missing_pdf_warning(pdf_paths)
            return {
                "compare_path": str(compare_path),
                "raw_data": raw_data,
                "indicator_result": indicator_result,
                "indicator_meta": indicator_meta,
                "pdf_paths": pdf_paths,
                "warning": warning,
            }

        canonical = to_canonical_payload(raw_data)
        if not is_canonical_comparison(canonical):
            canonical = raw_data

        indicator_meta = (
            dict(canonical.get("meta", {})) if isinstance(canonical, dict) else {}
        )
        indicator_meta["compare_path"] = str(compare_path)
        indicator_meta["source"] = source
        indicator_meta["source_label"] = source_label
        indicator_meta["storage_backend"] = "fichier_local"
        pdf_paths = _normalize_pdf_paths_store(
            _pdf_paths_from_comparison_meta(indicator_meta, canonical)
        )
        indicator_meta["pdf_paths"] = pdf_paths
        warning = _missing_pdf_warning(pdf_paths)
        return {
            "compare_path": str(compare_path),
            "raw_data": raw_data,
            "indicator_result": canonical,
            "indicator_meta": indicator_meta,
            "pdf_paths": pdf_paths,
            "warning": warning,
        }

    def load_review_state(
        self, compare_path: str | Path | None
    ) -> dict[str, Any] | None:
        return load_review_state(compare_path)

    def save_review_state(
        self,
        compare_path: str | Path | None,
        **kwargs: Any,
    ) -> Path | None:
        return save_review_state(compare_path, **kwargs)


def build_file_comparison_store(
    *, root_dir: str | Path | None = None
) -> FileComparisonStore:
    """Return the local file-backed store used by Dash."""
    return FileComparisonStore(root_dir=root_dir)
