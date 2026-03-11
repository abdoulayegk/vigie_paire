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

# Extraction certification: thresholds for status derivation.
# Tables with confidence below this are considered low_extraction_confidence (blocker).
EXTRACTION_CONFIDENCE_CERTIFIED_MIN = 0.5
# Confidence above this with no blockers and no warnings => certified.
EXTRACTION_CONFIDENCE_REVIEW_MIN = 0.7
# Legacy alias for config compatibility.
EXTRACTION_CONFIDENCE_REVIEW_FLOOR = 0.35

EXTRACTION_STATUS_CERTIFIED = "certified"
EXTRACTION_STATUS_REVIEW_REQUIRED = "review_required"
EXTRACTION_STATUS_BLOCKED = "blocked"


def infer_content_source(
    extraction_method: str | None, explicit: str | None = None
) -> str:
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


def get_extraction_confidence(table: Any) -> float:
    """
    Return extraction confidence from a table's debug_metrics (0.0--1.0).
    Prefer vision_extraction_confidence when present; otherwise 0.0.
    """
    if table is None:
        return 0.0
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return 0.0
    v = dm.get("vision_extraction_confidence")
    if v is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def get_extraction_quality_flags(table: Any) -> dict[str, bool]:
    """
    Return a normalized set of quality flags from a table's debug_metrics.
    Matcher and review pipeline should use this instead of inspecting debug_metrics ad hoc.
    """
    if table is None:
        return {}
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return {}
    return {
        "appears_truncated": bool(dm.get("appears_truncated", False)),
        "recrop_attempted": bool(dm.get("recrop_attempted", False)),
        "recrop_used": bool(dm.get("recrop_used", False)),
        "recrop_failed_incomplete": bool(dm.get("recrop_failed_incomplete", False)),
        "vision_extraction_applied": bool(dm.get("vision_extraction_applied", False)),
        "crop_rejected": bool(dm.get("crop_reject_reason")),
    }


def get_extraction_quality_profile(table: Any) -> dict[str, Any]:
    """
    Return a normalized quality profile for a table (confidence + flags + bbox_sanity etc.).
    Use for matcher and observability; avoids ad hoc debug_metrics access.
    """
    if table is None:
        return {}
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return {}
    profile: dict[str, Any] = {
        "confidence": get_extraction_confidence(table),
        "flags": get_extraction_quality_flags(table),
        "bbox_sanity_profile": dm.get("bbox_sanity_profile"),
        "page_title_assist_used": dm.get("page_title_assist_used"),
        "page_title_assist_match_method": dm.get("page_title_assist_match_method"),
        "warnings": list(dm.get("warnings") or []),
    }
    return profile


def derive_extraction_blockers(
    table: Any,
    *,
    confidence_certified_min: float = EXTRACTION_CONFIDENCE_CERTIFIED_MIN,
) -> list[str]:
    """
    Return the list of extraction blocker codes for this table.
    Blockers prevent auto-comparison; presence of any => status blocked.
    """
    if table is None:
        return ["non_vision_content_source"]
    blockers: list[str] = []
    content_source = getattr(table, "content_source", None)
    if content_source != VISION_CONTENT_SOURCE:
        content_source = infer_content_source(
            getattr(table, "extraction_method", None), None
        )
    if content_source != VISION_CONTENT_SOURCE:
        blockers.append("non_vision_content_source")
    raw_indicators = get_vision_raw_indicators(table)
    if not raw_indicators:
        blockers.append("missing_vision_indicators")
    flags = get_extraction_quality_flags(table)
    if flags.get("crop_rejected"):
        blockers.append("crop_rejected")
    if flags.get("recrop_failed_incomplete"):
        blockers.append("recrop_failed_incomplete")
    if not flags.get("vision_extraction_applied", True):
        blockers.append("vision_extraction_not_applied")
    confidence = get_extraction_confidence(table)
    if confidence < confidence_certified_min:
        blockers.append("low_extraction_confidence")
    return blockers


def get_extraction_status(
    table: Any,
    *,
    confidence_certified_min: float = EXTRACTION_CONFIDENCE_CERTIFIED_MIN,
    confidence_review_min: float = EXTRACTION_CONFIDENCE_REVIEW_MIN,
) -> str:
    """
    Return extraction status: certified, review_required, or blocked.
    Only certified tables are eligible for automatic matching.
    """
    blockers = derive_extraction_blockers(
        table, confidence_certified_min=confidence_certified_min
    )
    if blockers:
        return EXTRACTION_STATUS_BLOCKED
    confidence = get_extraction_confidence(table)
    flags = get_extraction_quality_flags(table)
    if confidence < confidence_review_min:
        return EXTRACTION_STATUS_REVIEW_REQUIRED
    if flags.get("appears_truncated") or flags.get("recrop_used"):
        return EXTRACTION_STATUS_REVIEW_REQUIRED
    return EXTRACTION_STATUS_CERTIFIED


def is_auto_compare_eligible(
    table: Any,
    *,
    confidence_certified_min: float = EXTRACTION_CONFIDENCE_CERTIFIED_MIN,
) -> bool:
    """
    Return True iff the table is certified for automatic comparison.
    Uncertified tables must not enter the matcher.
    """
    return get_extraction_status(table, confidence_certified_min=confidence_certified_min) == EXTRACTION_STATUS_CERTIFIED


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
    title_clean: str | None = (
        None  # Cleaned title (no amounts); use for display/pairing when set
    )
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
        # Always recompute blockers from current state — never accumulate
        # stale blockers passed in from storage reload or fragment merge.
        self.comparison_blockers = derive_comparison_blockers(
            content_source=self.content_source,
            first_column_indicators_raw=self.first_column_indicators_raw,
            footnotes=self.footnotes,
        )
        self.comparison_eligible = is_comparison_eligible(
            self.comparison_blockers,
            content_source=self.content_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def vision_raw_indicators(self) -> list[str]:
        """Return GPT-4o Vision raw first-column labels."""
        return [
            str(item)
            for item in (self.first_column_indicators_raw or [])
            if str(item).strip()
        ]

    @property
    def comparison_normalized_indicators(self) -> list[str]:
        """Return normalized first-column labels used by the comparator."""
        return [
            str(item)
            for item in (self.first_column_indicators or [])
            if str(item).strip()
        ]

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
