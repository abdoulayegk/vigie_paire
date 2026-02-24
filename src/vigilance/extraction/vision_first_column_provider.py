"""Provider interface for Vision-based first-column extraction. No implementation in this pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class VisionFirstColumnResult:
    """Result of Vision-based first-column extraction."""

    indicators_raw: list[str]
    confidence: float
    provider: str


class VisionFirstColumnProvider(Protocol):
    """Protocol for providers that extract first-column labels from a table image."""

    def extract_first_column(self, image_path: str) -> VisionFirstColumnResult:
        """Extract first-column indicator labels from a cropped table image. Returns raw strings."""
        ...
