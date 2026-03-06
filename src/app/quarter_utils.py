"""Quarter pairing helpers for current-vs-previous comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_QUARTER_RE = re.compile(
    r"(?i)\b([qt])\s*([1-4])(?:\s*[-_/ ]\s*((?:19|20)\d{2}))?\b"
)


@dataclass(frozen=True, slots=True)
class QuarterRef:
    quarter: int
    year: int

    @property
    def code(self) -> str:
        return f"t{self.quarter}"

    @property
    def label(self) -> str:
        return f"Q{self.quarter}-{self.year}"


def parse_quarter_ref(value: str | None, *, year: int | str | None = None) -> QuarterRef:
    """Parse a quarter reference from formats like ``Q2-2025``, ``T2_2025`` or ``Q2``."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("Quarter value is required.")

    match = _QUARTER_RE.search(text)
    if not match:
        raise ValueError(f"Unsupported quarter format: {text!r}")

    quarter_num = int(match.group(2))
    parsed_year = match.group(3)
    effective_year = int(parsed_year) if parsed_year else int(year) if year is not None else None
    if effective_year is None:
        raise ValueError(f"Quarter year is required for {text!r}")
    return QuarterRef(quarter=quarter_num, year=int(effective_year))


def previous_comparable_quarter(current: QuarterRef) -> QuarterRef:
    """Return the previous comparable quarter.

    Business rule:
    - Q2 -> Q1 (same year)
    - Q3 -> Q2 (same year)
    - Q1 -> Q3 (previous year)
    - Q4 -> Q3 (same year, graceful fallback if Q4 is ever used)
    """
    if current.quarter == 1:
        return QuarterRef(quarter=3, year=current.year - 1)
    if current.quarter == 4:
        return QuarterRef(quarter=3, year=current.year)
    return QuarterRef(quarter=current.quarter - 1, year=current.year)


def build_quarter_context(
    current_quarter: str | None,
    *,
    year: int | str | None = None,
    previous_quarter: str | None = None,
) -> dict[str, Any]:
    """Build a canonical current/previous quarter context."""
    current_ref = parse_quarter_ref(current_quarter, year=year)
    previous_ref = (
        parse_quarter_ref(previous_quarter)
        if previous_quarter is not None and str(previous_quarter).strip()
        else previous_comparable_quarter(current_ref)
    )
    return {
        "current": {
            "quarter": current_ref.quarter,
            "year": current_ref.year,
            "code": current_ref.code,
            "label": current_ref.label,
        },
        "previous": {
            "quarter": previous_ref.quarter,
            "year": previous_ref.year,
            "code": previous_ref.code,
            "label": previous_ref.label,
        },
        "comparison_direction": "current_vs_previous",
        "comparison_label": f"{current_ref.label} vs {previous_ref.label}",
    }


def get_payload_quarter_context(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Best-effort quarter context from a comparison payload."""
    data = payload or {}
    meta = data.get("meta") if isinstance(data, dict) else {}
    if isinstance(meta, dict):
        ctx = meta.get("quarter_context")
        if isinstance(ctx, dict) and isinstance(ctx.get("current"), dict) and isinstance(
            ctx.get("previous"), dict
        ):
            return ctx

    current_label = str(data.get("current_quarter", "") or data.get("quarter_to", "")).strip()
    previous_label = str(data.get("previous_quarter", "") or data.get("quarter_from", "")).strip()
    fallback_year = data.get("year")
    if current_label:
        try:
            return build_quarter_context(
                current_label,
                year=fallback_year,
                previous_quarter=previous_label or None,
            )
        except Exception:
            pass
    return {
        "current": {"label": "Trimestre courant", "code": "", "quarter": None, "year": fallback_year},
        "previous": {
            "label": "Trimestre precedent",
            "code": "",
            "quarter": None,
            "year": fallback_year,
        },
        "comparison_direction": "current_vs_previous",
        "comparison_label": "Trimestre courant vs trimestre precedent",
    }


def quarter_label_from_payload(payload: dict[str, Any] | None, role: str) -> str:
    """Return ``current`` or ``previous`` quarter label for display."""
    ctx = get_payload_quarter_context(payload)
    if role == "current":
        return str(ctx.get("current", {}).get("label") or "Trimestre courant")
    if role == "previous":
        return str(ctx.get("previous", {}).get("label") or "Trimestre precedent")
    raise ValueError(f"Unsupported quarter role: {role!r}")
