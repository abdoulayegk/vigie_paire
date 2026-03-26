"""Core table models used by extraction, storage, and comparison."""

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

# Confidence telemetry thresholds kept for observability/debug only.
EXTRACTION_CONFIDENCE_CERTIFIED_MIN = 0.5
EXTRACTION_CONFIDENCE_REVIEW_MIN = 0.7
RECROP_CERTIFIED_MIN = 0.85
EXTRACTION_CONFIDENCE_REVIEW_FLOOR = 0.35

TABLE_EXTRACTION_STATUS_OK = "ok"
TABLE_EXTRACTION_STATUS_RESCUED = "rescued"
TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED = "suspect_unresolved"
TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE = "confirmed_no_table"
TABLE_EXTRACTION_STATUSES = frozenset(
    {
        TABLE_EXTRACTION_STATUS_OK,
        TABLE_EXTRACTION_STATUS_RESCUED,
        TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED,
        TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE,
    }
)


def normalize_extraction_status(value: Any) -> str:
    """Return a canonical extraction_status value."""
    status = str(value or "").strip().lower()
    if status in TABLE_EXTRACTION_STATUSES:
        return status
    return TABLE_EXTRACTION_STATUS_OK


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
    extraction_status: str,
) -> list[str]:
    """Derive matching-path blocker codes from canonical extraction_status only.

    Only ``confirmed_no_table`` blocks inclusion in GPT table matching.
    ``suspect_unresolved`` is not a blocker here; quality is signaled via
    ``extraction_status`` and the extraction quality gate.

    Args:
        extraction_status: Canonical extraction status value.

    Returns:
        Non-empty list only for ``confirmed_no_table``; otherwise empty.
    """
    status = normalize_extraction_status(extraction_status)
    if status == TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE:
        return [TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE]
    return []


def is_comparison_eligible(
    extraction_status: str,
) -> bool:
    """Whether the table is eligible for the report matching path (GPT cards).

    Eligible for all statuses except ``confirmed_no_table`` (``ok``, ``rescued``,
    and ``suspect_unresolved``). Suspect tables still participate in matching;
    ``extraction_status`` remains the quality flag for review and certification.

    Args:
        extraction_status: Canonical extraction status value.

    Returns:
        False only for ``confirmed_no_table`` after normalization.
    """
    status = normalize_extraction_status(extraction_status)
    return status != TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE


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
        Dict of flag names to booleans: recrop_attempted,
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
        rows: Table rows as list of cell lists when available.
        first_column_indicators: Normalized first-column labels for comparison.
        extraction_method: Method used (e.g. vision_full_gpt4o).
        title_clean: Cleaned title without amounts; used for display/pairing.
        table_summary: Short semantic summary used as a secondary pairing signal.
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
    comparison_eligible: Eligible for GPT table matching (all except confirmed_no_table).
    comparison_blockers: Non-empty only when not a business table (confirmed_no_table).
        extraction_status: Rescue-state classification; suspect_unresolved flags quality.
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
    table_summary: str | None = None
    title_raw: str | None = None  # Original title for traceability
    row_count: int | None = None
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
    extraction_status: str = TABLE_EXTRACTION_STATUS_OK

    def __post_init__(self) -> None:
        """Initialize derived fields from extraction state.

        Sets content_source from extraction_method, normalizes extraction_status,
        recomputes comparison_blockers from the canonical status, and updates
        comparison_eligible.
        """
        self.content_source = infer_content_source(
            self.extraction_method,
            self.content_source,
        )
        self.extraction_status = normalize_extraction_status(self.extraction_status)
        if self.title_clean is None:
            self.title_clean = self.title
        if self.row_count is None:
            if self.first_column_indicators_raw:
                self.row_count = len(
                    [item for item in self.first_column_indicators_raw if str(item).strip()]
                )
            else:
                self.row_count = len(list(self.rows or []))
        # Always recompute blockers from canonical extraction_status.
        self.comparison_blockers = derive_comparison_blockers(
            extraction_status=self.extraction_status,
        )
        self.comparison_eligible = is_comparison_eligible(
            self.extraction_status,
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
