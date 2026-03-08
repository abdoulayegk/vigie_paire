"""Core table models used by extraction, storage, and comparison.

Naming guide:
- ``first_column_indicators_raw``: raw first-column labels extracted by GPT-4o Vision.
- ``first_column_indicators``: deterministic normalized labels used for comparison.
- ``TableArtifact``: canonical in-memory table object for the comparison engine.

The repo still exposes the legacy field names above for compatibility, but
the properties added below make the intended roles explicit at call sites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Canonical footnote format: [{"id": str, "text": str}, ...]
# Legacy list[str] payloads are normalized at ingestion boundaries.
FootnoteList = list[dict[str, str]]

VISION_CONTENT_SOURCE = "vision_gpt4o"
UNKNOWN_CONTENT_SOURCE = "unknown"
NON_VISION_FATAL_BLOCKERS = frozenset(
    {
        "missing_vision_indicators",
        "non_vision_content_source",
    }
)


def infer_content_source(extraction_method: str | None, explicit: str | None = None) -> str:
    value = str(explicit or "").strip()
    if value:
        return value

    method = str(extraction_method or "").strip().lower()
    if method == "vision_full_gpt4o":
        return VISION_CONTENT_SOURCE
    if method.startswith("vision_"):
        return VISION_CONTENT_SOURCE
    return UNKNOWN_CONTENT_SOURCE


def derive_comparison_blockers(
    *,
    content_source: str,
    first_column_indicators_raw: list[str] | None,
    footnotes: FootnoteList | None,
) -> list[str]:
    blockers: list[str] = []
    if content_source != VISION_CONTENT_SOURCE:
        blockers.append("non_vision_content_source")
    if not list(first_column_indicators_raw or []):
        blockers.append("missing_vision_indicators")
    if footnotes is None:
        blockers.append("footnotes_unavailable")
    return blockers


def is_comparison_eligible(
    blockers: list[str] | None,
    *,
    content_source: str,
) -> bool:
    if content_source != VISION_CONTENT_SOURCE:
        return False
    blocker_set = {str(item).strip() for item in blockers or [] if str(item).strip()}
    return not bool(blocker_set & NON_VISION_FATAL_BLOCKERS)


def get_vision_raw_indicators(table: Any) -> list[str]:
    """Return Vision raw first-column labels from a table-like object."""
    if table is None:
        return []
    values = getattr(table, "vision_raw_indicators", None)
    if values is None:
        values = getattr(table, "first_column_indicators_raw", None)
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def get_comparison_indicators(table: Any) -> list[str]:
    """Return normalized first-column labels used by the comparator."""
    if table is None:
        return []
    values = getattr(table, "comparison_normalized_indicators", None)
    if values is None:
        values = getattr(table, "first_column_indicators", None)
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def get_canonical_footnotes(table: Any) -> FootnoteList:
    """Return canonical footnotes from a table-like object."""
    if table is None:
        return []
    values = getattr(table, "canonical_footnotes", None)
    if values is None:
        values = getattr(table, "footnotes", None)
    if not values:
        return []
    from vigilance.utils.footnotes_utils import normalize_footnotes_to_canonical

    return normalize_footnotes_to_canonical(list(values))


@dataclass(slots=True)
class TableArtifact:
    """Canonical in-memory representation of one extracted table."""

    bank_code: str
    section: str
    page_pdf: int
    table_id: str
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    first_column_indicators: list[str]
    extraction_method: str
    title_clean: str | None = None  # Cleaned title (no amounts); use for display/pairing when set
    title_raw: str | None = None  # Original title for traceability
    table_number: str | None = None
    bbox: dict[str, Any] | list[float] | None = None
    quarter: str | None = None
    pdf_path: str | None = None
    first_column_indicators_raw: list[str] | None = None
    first_column_groups: list[str] | None = None
    hierarchical_indicator_signature: list[str] | None = None
    title_reliability: str | None = None
    footnotes: FootnoteList | None = None
    fragmentation_detected: bool = False
    debug_metrics: dict[str, Any] | None = None
    content_source: str = UNKNOWN_CONTENT_SOURCE
    comparison_eligible: bool = False
    comparison_blockers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.content_source = infer_content_source(
            self.extraction_method,
            self.content_source,
        )
        inferred = derive_comparison_blockers(
            content_source=self.content_source,
            first_column_indicators_raw=self.first_column_indicators_raw,
            footnotes=self.footnotes,
        )
        combined: list[str] = []
        seen: set[str] = set()
        for item in [*self.comparison_blockers, *inferred]:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            combined.append(value)
        self.comparison_blockers = combined
        self.comparison_eligible = is_comparison_eligible(
            self.comparison_blockers,
            content_source=self.content_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def vision_raw_indicators(self) -> list[str]:
        """Return GPT-4o Vision raw first-column labels."""
        return [str(item) for item in (self.first_column_indicators_raw or []) if str(item).strip()]

    @property
    def comparison_normalized_indicators(self) -> list[str]:
        """Return normalized first-column labels used by the comparator."""
        return [str(item) for item in (self.first_column_indicators or []) if str(item).strip()]

    @property
    def canonical_footnotes(self) -> FootnoteList:
        """Return footnotes in canonical ``[{id, text}]`` form."""
        return list(self.footnotes or [])

    @property
    def is_vision_sourced(self) -> bool:
        """Return True when table content used for comparison comes from Vision."""
        return self.content_source == VISION_CONTENT_SOURCE


# Kept for naming compatibility with downstream code.
TableCandidate = TableArtifact
