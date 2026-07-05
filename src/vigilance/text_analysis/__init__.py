"""Package du pipeline texte decoupe en modules par responsabilite."""

from __future__ import annotations

from .pipeline import run_text_analysis_pipeline, run_text_extraction_pipeline
from .models import TextAnalysisQualityError

__all__ = [
    "TextAnalysisQualityError",
    "run_text_analysis_pipeline",
    "run_text_extraction_pipeline",
]
