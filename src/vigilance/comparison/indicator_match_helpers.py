"""
Structural indicator signals for robust table matching.

Provides: LCS length, longest common prefix, containment, size ratio, prefix ratio.
Used to reduce false positives from shared prefixes and reused table numbers.
"""

from __future__ import annotations


def lcs_length(a: list[str], b: list[str]) -> int:
    """
    Longest common subsequence length (order preserved, gaps allowed).

    Standard DP. O(|a| * |b|).
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def longest_common_prefix_length(a: list[str], b: list[str]) -> int:
    """
    Count of leading indicators that match in order (no gaps).
    """
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def _jaccard_sets(a: list[str], b: list[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def compute_indicator_signals(ind1: list[str], ind2: list[str]) -> dict[str, float | list[float]]:
    """
    Compute structural indicator signals for two ordered normalized indicator lists.

    Returns:
        indicator_jaccard: set overlap (backward compat / audit)
        indicator_containment_min: |A inter B| / min(|A|, |B|)
        indicator_lcs_ratio: lcs_length / max(|A|, |B|)
        indicator_size_ratio: min(len) / max(len)
        indicator_prefix_ratio: lcp_length / min(len)  (0 if empty)
    """
    n1, n2 = len(ind1), len(ind2)
    if n1 == 0 and n2 == 0:
        return {
            "indicator_jaccard": 0.0,
            "indicator_containment_min": 0.0,
            "indicator_lcs_ratio": 0.0,
            "indicator_size_ratio": 1.0,
            "indicator_prefix_ratio": 0.0,
        }
    if n1 == 0 or n2 == 0:
        return {
            "indicator_jaccard": 0.0,
            "indicator_containment_min": 0.0,
            "indicator_lcs_ratio": 0.0,
            "indicator_size_ratio": 0.0,
            "indicator_prefix_ratio": 0.0,
        }

    jaccard = _jaccard_sets(ind1, ind2)
    inter = len(set(ind1) & set(ind2))
    min_len = min(n1, n2)
    max_len = max(n1, n2)
    containment_min = inter / min_len if min_len else 0.0
    lcs = lcs_length(ind1, ind2)
    lcs_ratio = lcs / max_len if max_len else 0.0
    size_ratio = min_len / max_len if max_len else 0.0
    lcp = longest_common_prefix_length(ind1, ind2)
    prefix_ratio = lcp / min_len if min_len else 0.0

    return {
        "indicator_jaccard": jaccard,
        "indicator_containment_min": containment_min,
        "indicator_lcs_ratio": lcs_ratio,
        "indicator_size_ratio": size_ratio,
        "indicator_prefix_ratio": prefix_ratio,
    }


def compute_robust_indicator_score_from_signals(signals: dict) -> float:
    """
    Single canonical robust indicator score from a signals dict.

    Used by match_decision and match_signals (count_strong_signals).
    Blends containment, LCS ratio, size ratio; penalizes prefix-only pattern.
    """
    jaccard = float(signals.get("indicator_jaccard", signals.get("indicator_overlap", 0)) or 0)
    containment = float(signals.get("indicator_containment_min", 0) or 0)
    lcs_ratio = float(signals.get("indicator_lcs_ratio", 0) or 0)
    size_ratio = float(signals.get("indicator_size_ratio", 1.0) or 1.0)
    prefix_ratio = float(signals.get("indicator_prefix_ratio", 0) or 0)

    base = 0.35 * jaccard + 0.35 * containment + 0.25 * lcs_ratio + 0.05 * size_ratio
    if prefix_ratio >= 0.7 and lcs_ratio < 0.4:
        base *= 0.6
    return min(1.0, max(0.0, base))
