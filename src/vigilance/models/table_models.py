"""Table data models used by the vigilance facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TableArtifact:
    """Normalized representation of one extracted table."""

    bank_code: str
    section: str
    page_pdf: int
    table_id: str
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    first_column_indicators: list[str]
    extraction_method: str
    table_number: str | None = None
    bbox: dict[str, Any] | list[float] | None = None
    quarter: str | None = None
    pdf_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Kept for naming compatibility with downstream code.
TableCandidate = TableArtifact
