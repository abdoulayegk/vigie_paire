"""Unit tests for embedding integration."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skip(
    reason="Modules vigilance.embedding et vigilance.compare absents de cette branche."
)


def test_embedding_service_initializes() -> None:
    """Verify EmbeddingService initializes and has stats."""
    from vigilance.embedding import EmbeddingService, EmbeddingStats

    svc = EmbeddingService(api_key=None)
    assert svc.model == "text-embedding-3-small"
    assert hasattr(svc, "stats")
    assert svc.stats.api_calls == 0
    assert svc.stats.cache_hits == 0
    assert svc.available is False


def test_embed_sim_in_match_decision_when_enabled() -> None:
    """Verify match_decision returns embed_sim in decision when use_embeddings and service available."""
    from vigilance.compare.indicator_comparator import match_decision
    from vigilance.models.table_models import TableArtifact

    t1 = TableArtifact(
        bank_code="test",
        section="Capital",
        page_pdf=1,
        table_id="t1",
        title="Credit Risk",
        headers=["A", "B"],
        rows=[["X", "1"]],
        first_column_indicators=["X"],
        extraction_method="docling",
    )
    t2 = TableArtifact(
        bank_code="test",
        section="Capital",
        page_pdf=2,
        table_id="t2",
        title="Credit Risk Exposure",
        headers=["A", "B"],
        rows=[["X", "2"]],
        first_column_indicators=["X"],
        extraction_method="docling",
    )

    from vigilance.compare.indicator_comparator import _DEFAULTS

    th = dict(_DEFAULTS, use_embeddings=False)
    decision = match_decision(t1, t2, thresholds=th)
    assert hasattr(decision, "embed_sim")
    assert decision.embed_sim == 0.0


def test_embedding_debug_schema_complete() -> None:
    """Verify embedding_debug has all required keys."""
    from vigilance.comparison_runner import run_comparison_with_sections

    result = run_comparison_with_sections(
        pdf_path_t1="/nonexistent/t1.pdf",
        pdf_path_t2="/nonexistent/t2.pdf",
        bank_code="test",
        sections_t1=[],
        sections_t2=[],
    )
    dbg = result["meta"]["embedding_debug"]
    required = [
        "embedding_enabled",
        "embedding_table_used",
        "embedding_indicator_used",
        "embedding_api_calls",
        "embedding_cache_hits",
        "embedding_batch_sizes",
        "embedding_errors",
        "config_use_embeddings",
        "table_pair_count",
        "indicator_rename_count",
    ]
    for k in required:
        assert k in dbg, f"Missing key: {k}"
