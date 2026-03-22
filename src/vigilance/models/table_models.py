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
# Successful recrops with strong confidence can stay certified.
RECROP_CERTIFIED_MIN = 0.85
# Legacy alias for config compatibility.
EXTRACTION_CONFIDENCE_REVIEW_FLOOR = 0.35

EXTRACTION_STATUS_CERTIFIED = "certified"
EXTRACTION_STATUS_REVIEW_REQUIRED = "review_required"
EXTRACTION_STATUS_BLOCKED = "blocked"


def infer_content_source(
    extraction_method: str | None, explicit: str | None = None
) -> str:
    """Infer the content source from extraction method or explicit override.

    Args:
        extraction_method: The extraction method string (e.g. "vision_full_gpt4o").
        explicit: Optional explicit content source override; takes precedence.

    Returns:
        The content source string: VISION_CONTENT_SOURCE for vision-based
        extraction, UNKNOWN_CONTENT_SOURCE otherwise.
    """
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
    """Derive comparison blocker codes from content source and table metadata.

    Args:
        content_source: The content source string (e.g. VISION_CONTENT_SOURCE).
        first_column_indicators_raw: Raw first-column labels from Vision.
        footnotes: Canonical footnotes list; None means unavailable.

    Returns:
        List of blocker codes such as "non_vision_content_source",
        "missing_vision_indicators", or "footnotes_unavailable".
    """
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
    """Check whether a table is eligible for comparison.

    Args:
        blockers: List of comparison blocker codes.
        content_source: The content source string.

    Returns:
        True if content is vision-sourced and no fatal blockers
        (non_vision_content_source, missing_vision_indicators) are present.
    """
    if content_source != VISION_CONTENT_SOURCE:
        return False
    blocker_set = {str(item).strip() for item in blockers or [] if str(item).strip()}
    return not bool(blocker_set & NON_VISION_FATAL_BLOCKERS)


def get_vision_raw_indicators(table: Any) -> list[str]:
    """Return Vision raw first-column labels from a table-like object.

    Args:
        table: Table-like object with vision_raw_indicators or
            first_column_indicators_raw attribute.

    Returns:
        List of non-empty stripped indicator strings; empty list if None or missing.
    """
    if table is None:
        return []
    values = getattr(table, "vision_raw_indicators", None)
    if values is None:
        values = getattr(table, "first_column_indicators_raw", None)
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def get_comparison_indicators(table: Any) -> list[str]:
    """Return normalized first-column labels used by the comparator.

    Args:
        table: Table-like object with comparison_normalized_indicators or
            first_column_indicators attribute.

    Returns:
        List of non-empty stripped indicator strings; empty list if None or missing.
    """
    if table is None:
        return []
    values = getattr(table, "comparison_normalized_indicators", None)
    if values is None:
        values = getattr(table, "first_column_indicators", None)
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def get_canonical_footnotes(table: Any) -> FootnoteList:
    """Return canonical footnotes from a table-like object.

    Args:
        table: Table-like object with canonical_footnotes or footnotes attribute.

    Returns:
        Canonical footnote list ``[{"id": str, "text": str}, ...]``;
        legacy formats are normalized at ingestion.
    """
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
    """Return extraction confidence from a table's debug_metrics (0.0--1.0).

    Prefers vision_extraction_confidence; falls back to vision_primary_confidence.
    When metrics are missing but the table has Vision content and raw indicators,
    assumes certified-level confidence so stored extractions do not false-block.

    Args:
        table: Table-like object with debug_metrics and optional content_source.

    Returns:
        Confidence score in [0.0, 1.0]; 0.0 when table is None or metrics invalid.
    """
    if table is None:
        return 0.0
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return 0.0
    v = dm.get("vision_extraction_confidence")
    if v is None:
        v = dm.get("vision_primary_confidence")
    if v is None:
        content_source = getattr(table, "content_source", None)
        if content_source != VISION_CONTENT_SOURCE:
            content_source = infer_content_source(
                getattr(table, "extraction_method", None), None
            )
        if content_source == VISION_CONTENT_SOURCE and get_vision_raw_indicators(table):
            return max(EXTRACTION_CONFIDENCE_CERTIFIED_MIN, 0.7)
        return 0.0
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def get_extraction_quality_flags(table: Any) -> dict[str, bool]:
    """Return a normalized set of quality flags from a table's debug_metrics.

    When metrics are missing but the table has Vision content and raw indicators,
    assumes vision_extraction_applied=True so stored extractions do not false-block.

    Args:
        table: Table-like object with debug_metrics and optional content_source.

    Returns:
        Dict of flag names to booleans: appears_truncated, recrop_attempted,
        recrop_used, recrop_failed_incomplete, vision_extraction_applied,
        crop_rejected, partial_result, rows_missing_from_fallback.
    """
    if table is None:
        return {}
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return {}
    warning_codes = {
        str(code).strip()
        for code in list(dm.get("vision_warning_codes") or dm.get("warnings") or [])
        if str(code).strip()
    }
    vision_status = str(dm.get("vision_status") or "").strip().lower()
    applied = dm.get("vision_extraction_applied")
    if applied is None:
        applied = dm.get("vision_primary_applied")
    if applied is None or applied is False:
        content_source = getattr(table, "content_source", None)
        if content_source != VISION_CONTENT_SOURCE:
            content_source = infer_content_source(
                getattr(table, "extraction_method", None), None
            )
        if content_source == VISION_CONTENT_SOURCE and get_vision_raw_indicators(table):
            applied = True
    return {
        "appears_truncated": bool(dm.get("appears_truncated", False)),
        "recrop_attempted": bool(dm.get("recrop_attempted", False)),
        "recrop_used": bool(dm.get("recrop_used", False)),
        "recrop_failed_incomplete": bool(dm.get("recrop_failed_incomplete", False)),
        "vision_extraction_applied": bool(applied if applied is not None else False),
        "crop_rejected": bool(dm.get("crop_reject_reason")),
        "partial_result": vision_status == "partial",
        "rows_missing_from_fallback": "vision_rows_missing_from_fallback"
        in warning_codes,
    }


def get_extraction_quality_profile(table: Any) -> dict[str, Any]:
    """Return a normalized quality profile for a table.

    Aggregates confidence, flags, bbox_sanity_profile, and related metadata.
    Use for matcher and observability; avoids ad hoc debug_metrics access.

    Args:
        table: Table-like object with debug_metrics.

    Returns:
        Dict with keys: confidence, flags, bbox_sanity_profile,
        page_title_assist_used, page_title_assist_match_method, warnings.
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
    """Derive extraction blocker codes for a table.

    Blockers prevent auto-comparison; presence of any yields status blocked.

    Args:
        table: Table-like object with content_source, debug_metrics, etc.
        confidence_certified_min: Minimum confidence for certified status.

    Returns:
        List of blocker codes such as non_vision_content_source,
        missing_vision_indicators, crop_rejected, low_extraction_confidence.
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
    if flags.get("partial_result"):
        blockers.append("partial_vision_result")
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
    """Return extraction status: certified, review_required, or blocked.

    The recall-first engine includes review_required tables with a
    capped confidence. See is_matching_eligible for the relaxed gate
    and is_auto_compare_eligible for the strict (legacy) gate.

    Args:
        table: Table-like object with debug_metrics and content metadata.
        confidence_certified_min: Minimum confidence for certified status.
        confidence_review_min: Minimum confidence to avoid review_required.

    Returns:
        One of EXTRACTION_STATUS_CERTIFIED, EXTRACTION_STATUS_REVIEW_REQUIRED,
        or EXTRACTION_STATUS_BLOCKED.
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
    if flags.get("appears_truncated"):
        return EXTRACTION_STATUS_REVIEW_REQUIRED
    if flags.get("recrop_used") and confidence < max(
        confidence_review_min,
        RECROP_CERTIFIED_MIN,
    ):
        return EXTRACTION_STATUS_REVIEW_REQUIRED
    return EXTRACTION_STATUS_CERTIFIED


def is_auto_compare_eligible(
    table: Any,
    *,
    confidence_certified_min: float = EXTRACTION_CONFIDENCE_CERTIFIED_MIN,
) -> bool:
    """Return True iff the table is certified for automatic comparison.

    This is the strict eligibility gate used by the legacy matching engine.
    The recall-first engine uses is_matching_eligible instead, which also
    accepts review_required tables.

    Args:
        table: Table-like object with extraction metadata.
        confidence_certified_min: Minimum confidence for certified status.

    Returns:
        True if extraction status is certified; False otherwise.
    """
    return get_extraction_status(table, confidence_certified_min=confidence_certified_min) == EXTRACTION_STATUS_CERTIFIED


def is_matching_eligible(
    table: Any,
    *,
    confidence_certified_min: float = EXTRACTION_CONFIDENCE_CERTIFIED_MIN,
) -> bool:
    """Return True if the table can participate in matching (recall-first).

    Accepts both certified and review_required tables. Only blocked tables
    are excluded. Pairs including a review_required table will have their
    confidence capped by the matching engine.

    Args:
        table: Table-like object with extraction metadata.
        confidence_certified_min: Minimum confidence for certified status.

    Returns:
        True if extraction status is not blocked; False otherwise.
    """
    return get_extraction_status(
        table, confidence_certified_min=confidence_certified_min
    ) != EXTRACTION_STATUS_BLOCKED


@dataclass(slots=True)
class TableArtifact:
    """Canonical in-memory representation of one extracted table.

    Attributes:
        bank_code: Bank identifier.
        section: Document section (e.g. balance sheet, income statement).
        page_pdf: 1-based PDF page number.
        table_id: Unique table identifier.
        title: Table title; may include amounts.
        headers: Column header strings.
        rows: Table rows as list of cell lists.
        first_column_indicators: Normalized first-column labels for comparison.
        extraction_method: Method used (e.g. vision_full_gpt4o).
        title_clean: Cleaned title without amounts; used for display/pairing.
        title_raw: Original title for traceability.
        table_number: Table number if present.
        bbox: Bounding box as dict or list of floats.
        table_index_on_page: Index of this table on the page.
        tables_on_page: Total tables on the page.
        bbox_top: Top coordinate of bounding box.
        page_local_role: Role of this table on the page.
        quarter: Reporting quarter if applicable.
        pdf_path: Path to source PDF.
        first_column_indicators_raw: Raw Vision first-column labels.
        first_column_groups: Grouped indicator labels.
        hierarchical_indicator_signature: Hierarchical indicator structure.
        title_reliability: Reliability of title extraction.
        footnotes: Canonical footnote list.
        fragmentation_detected: Whether table was detected as fragmented.
        fragment_near_merge_hint: Hints for fragment merging.
        debug_metrics: Extraction metrics and quality flags.
        content_source: Content source (vision_gpt4o or unknown).
        comparison_eligible: Whether table is eligible for comparison.
        comparison_blockers: List of blocker codes if not eligible.
    """

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
    table_index_on_page: int | None = None
    tables_on_page: int | None = None
    bbox_top: float | None = None
    page_local_role: str | None = None
    quarter: str | None = None
    pdf_path: str | None = None
    first_column_indicators_raw: list[str] | None = None
    first_column_groups: list[str] | None = None
    hierarchical_indicator_signature: list[str] | None = None
    title_reliability: str | None = None
    footnotes: FootnoteList | None = None
    fragmentation_detected: bool = False
    fragment_near_merge_hint: dict[str, Any] | None = None
    debug_metrics: dict[str, Any] | None = None
    content_source: str = UNKNOWN_CONTENT_SOURCE
    comparison_eligible: bool = False
    comparison_blockers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize derived fields from extraction state.

        Sets content_source from extraction_method, recomputes comparison_blockers
        from current state (never accumulates stale blockers from storage reload),
        and updates comparison_eligible.
        """
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
        """Serialize the artifact to a plain dict for storage or JSON export.

        Returns:
            Dict with all dataclass fields as key-value pairs.
        """
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
