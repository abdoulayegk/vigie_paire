"""Compatibility facade for the refactored text analysis package.

New code should import from ``vigilance.text_analysis`` or
``vigilance.text_analysis.pipeline``. This module remains only to preserve the
historic public imports used by older scripts.
"""

from __future__ import annotations

from vigilance.text_analysis import (
    TextAnalysisQualityError,
    run_text_analysis_pipeline,
    run_text_extraction_pipeline,
)

__all__ = [
    "TextAnalysisQualityError",
    "run_text_analysis_pipeline",
    "run_text_extraction_pipeline",
]
