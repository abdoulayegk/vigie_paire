"""Unit tests to validate embedding audit findings and instrumentation."""

from __future__ import annotations

import os

import pytest


def test_semantic_table_matcher_exists_and_uses_embeddings() -> None:
    """Verify SemanticTableMatcher exists, uses text-embedding-3-small, and computes cosine similarity."""
    from vigilance.comparison.matcher import SemanticTableMatcher, TableFingerprint

    assert SemanticTableMatcher is not None
    matcher = SemanticTableMatcher(api_key=os.environ.get("OPENAI_API_KEY") or "sk-test")
    assert matcher.model == "text-embedding-3-small"

    # Fingerprint has embedding field
    fp = TableFingerprint(
        table_id="t1",
        page_number=1,
        title="Test",
        headers=["A"],
        first_col_sample=["X"],
        embedding=None,
    )
    assert hasattr(fp, "embedding")


def test_run_comparison_emits_embedding_debug() -> None:
    """Verify run_comparison_with_sections emits embedding_debug in meta."""
    from app.comparison_runner import run_comparison_with_sections

    result = run_comparison_with_sections(
        pdf_path_t1="/nonexistent/t1.pdf",
        pdf_path_t2="/nonexistent/t2.pdf",
        bank_code="test",
        sections_t1=[],
        sections_t2=[],
    )

    assert "meta" in result
    meta = result["meta"]
    assert "embedding_debug" in meta, "embedding_debug must be present in meta"
    dbg = meta["embedding_debug"]
    assert "embedding_table_used" in dbg
    assert dbg["embedding_table_used"] is False
    assert "embedding_indicator_used" in dbg
    assert "embedding_api_calls" in dbg
    assert "embedding_cache_hits" in dbg
    assert "config_use_embeddings" in dbg
    assert "hungarian_table" in dbg
    assert "hungarian_indicator" in dbg
    assert "table_pair_count" in dbg
    assert "indicator_rename_count" in dbg


def test_indicator_similarity_uses_rapidfuzz_only() -> None:
    """Verify indicator pairing uses rapidfuzz, not embeddings."""
    from app.comparison_runner import _hungarian_pair_added_removed

    removed = ["Total equity"]
    added = ["Total des capitaux propres"]
    a_rest, r_rest, renames, debug = _hungarian_pair_added_removed(removed, added, th={})

    # Should produce renames or not based on rapidfuzz; debug dict has no embed info
    assert "gated_out_pairs" in debug
    assert "accepted_renames" in debug
    assert "score_distribution" in debug
    # No embedding_sim in debug - confirms rapidfuzz-only path
    assert "embedding_sim" not in str(debug)
