"""Tests for match_signals: new structural indicator keys and robust vs Jaccard."""

from __future__ import annotations

from vigilance.comparison.match_signals import compute_match_signals


def test_compute_match_signals_returns_new_indicator_keys() -> None:
    t1 = {
        "title": "Table 1",
        "headers": ["Col A", "Col B"],
        "indicators": ["Cash", "Securities", "Loans"],
        "section": "capital",
        "page": 1,
    }
    t2 = {
        "title": "Table 1",
        "headers": ["Col A", "Col B"],
        "indicators": ["Cash", "Securities", "Loans"],
        "section": "capital",
        "page": 2,
    }
    signals = compute_match_signals(t1, t2, has_headers=True)
    assert "indicator_overlap" in signals
    assert "indicator_jaccard" in signals
    assert "indicator_containment_min" in signals
    assert "indicator_lcs_ratio" in signals
    assert "indicator_size_ratio" in signals
    assert "indicator_prefix_ratio" in signals


def test_prefix_only_tables_structural_signals_differ_from_jaccard() -> None:
    t1 = {
        "title": "",
        "headers": ["T1", "T0"],
        "indicators": [
            "Tresorerie et depots",
            "Valeurs mobilieres",
            "Titres du gouvernement",
            "Prets hypothecaires",
            "Titres de participation",
        ],
        "section": "unknown",
        "page": 1,
    }
    t2 = {
        "title": "",
        "headers": ["T1", "T0"],
        "indicators": [
            "Tresorerie et depots",
            "Valeurs mobilieres",
            "Titres du gouvernement",
            "Actifs liquides par entite",
            "Actifs liquides par monnaie",
        ],
        "section": "unknown",
        "page": 2,
    }
    signals = compute_match_signals(t1, t2, has_headers=True)
    jaccard = signals["indicator_overlap"]
    lcs_ratio = signals["indicator_lcs_ratio"]
    prefix_ratio = signals["indicator_prefix_ratio"]
    assert prefix_ratio >= 0.5
    assert lcs_ratio < jaccard or lcs_ratio < 0.7
