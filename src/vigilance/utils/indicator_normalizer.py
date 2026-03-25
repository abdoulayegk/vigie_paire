"""Order-invariant indicator normalization for matching (canonical + token-sorted)."""

from __future__ import annotations

import re
import unicodedata

from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison

# Footnote markers (trailing) - same logic as comparison_runner for consistency
_INDICATOR_TRAILING_SUPER = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*$")
_INDICATOR_TRAILING_STARS = re.compile(r"\s*[*\u2020\u2021\u00A7]+\s*$")
_INDICATOR_TRAILING_PAREN_NUM = re.compile(r"\s*[\(\[]\d+[\)\]]\s*$")
_INDICATOR_TRAILING_NOTE_NUM = re.compile(r"\s+Note\s+\d+\s*\.?\s*$", re.IGNORECASE)
_INDICATOR_TRAILING_COMMA_NUMS = re.compile(r"\s*,\s*\d+(?:\s*,\s*\d+)*\s*$")
_INDICATOR_TRAILING_SPACE_NUMS_COMMA = re.compile(r"\s+\d+(?:\s*,\s*\d+)+\s*$")
_INDICATOR_TRAILING_NOTE_CLUSTER = re.compile(
    r"(?:\s*,?\s*(?:[\(\[]\d+[\)\]]|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|[*\u2020\u2021\u00A7]+))+[\s,;:.]*$"
)
# Letter footnote markers: (a), (b), [a], a), b) at end of label
_INDICATOR_TRAILING_PAREN_LETTER = re.compile(
    r"\s*(?:\([a-zA-Z]\)|\[[a-zA-Z]\]|[a-zA-Z]\))\s*$"
)

_STOPWORDS = frozenset(
    {
        "de",
        "du",
        "des",
        "la",
        "le",
        "les",
        "et",
        "ou",
        "and",
        "the",
        "of",
        "to",
        "en",
        "au",
        "aux",
        "a",
        "an",
    }
)
_UNIT_TOKENS = frozenset(
    {
        "%",
        "million",
        "millions",
        "milliard",
        "milliards",
        "dollars",
        "cad",
        "usd",
        "tableau",
        "table",
    }
)


def strip_footnote_markers_from_indicator(text: str) -> str:
    """Remove trailing footnote markers; preserves semantic numbers (Tier 1, CET1, etc.).

    Single shared implementation used by comparison_runner and get_canonical_text.
    """
    if not text:
        return ""
    value = (text or "").strip()
    while True:
        prev = value
        value = _INDICATOR_TRAILING_NOTE_CLUSTER.sub("", value)
        value = _INDICATOR_TRAILING_SUPER.sub("", value)
        value = _INDICATOR_TRAILING_STARS.sub("", value)
        value = _INDICATOR_TRAILING_PAREN_NUM.sub("", value)
        value = _INDICATOR_TRAILING_PAREN_LETTER.sub("", value)
        value = _INDICATOR_TRAILING_NOTE_NUM.sub("", value)
        value = _INDICATOR_TRAILING_COMMA_NUMS.sub("", value)
        value = _INDICATOR_TRAILING_SPACE_NUMS_COMMA.sub("", value)
        value = re.sub(r"\s+", " ", value).strip()
        if value == prev:
            break
    return value


def get_canonical_text(text: str) -> str:
    """
    Current normalized indicator text: strip footnote markers, then full normalization.
    Consistent with existing pipeline (lower, whitespace, strip dates/units, accents).
    """
    if not text or not (text or "").strip():
        return ""
    stripped = strip_footnote_markers_from_indicator(text)
    return normalize_indicator_for_comparison(stripped)


def _is_mostly_digit_token(token: str) -> bool:
    """Drop tokens that are mostly digits (e.g. 2025, 30); keep short semantic (1, 2, 3, 9)."""
    if not token or not token.strip():
        return True
    if len(token) <= 1 and token.isdigit():
        return False
    if token.isdigit() and len(token) >= 2:
        return True
    digit_count = sum(1 for c in token if c.isdigit())
    return (digit_count / len(token)) > 0.5


def get_token_sorted_text(text: str) -> str:
    """
    Order-invariant form: tokenize canonical_text, remove stopwords and units-only tokens,
    drop tokens with mostly digits, sort alphabetically, join with single space.
    Deterministic and French-safe (canonical already NFD/ASCII).
    """
    canonical = get_canonical_text(text)
    if not canonical:
        return ""
    normalized = re.sub(r"[-/]", " ", canonical)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    tokens: set[str] = set()
    for t in normalized.split():
        t = t.strip().lower()
        if not t:
            continue
        if t in _STOPWORDS or t in _UNIT_TOKENS:
            continue
        if _is_mostly_digit_token(t):
            continue
        tokens.add(t)
    return " ".join(sorted(tokens))


def get_normalized_forms(text: str) -> tuple[str, str]:
    """Return (canonical_text, token_sorted_text). Convenience for callers that need both."""
    return get_canonical_text(text), get_token_sorted_text(text)
