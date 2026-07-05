from __future__ import annotations

from vigilance.text_analysis import (
    TextAnalysisQualityError,
    run_text_analysis_pipeline,
    run_text_extraction_pipeline,
)
from vigilance.text_analysis_pipeline import (
    TextAnalysisQualityError as LegacyTextAnalysisQualityError,
)
from vigilance.text_analysis_pipeline import (
    run_text_analysis_pipeline as legacy_run_text_analysis_pipeline,
)
from vigilance.text_analysis_pipeline import (
    run_text_extraction_pipeline as legacy_run_text_extraction_pipeline,
)


def test_legacy_text_analysis_pipeline_imports_remain_available() -> None:
    assert legacy_run_text_analysis_pipeline is run_text_analysis_pipeline
    assert legacy_run_text_extraction_pipeline is run_text_extraction_pipeline
    assert LegacyTextAnalysisQualityError is TextAnalysisQualityError
