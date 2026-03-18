"""Shared base normalization for text comparison across indicators and footnotes.

Single source of truth for: accent stripping, apostrophe canonicalization,
elision+space collapse, and whitespace normalization. Used by indicator_cleaner,
footnote_comparator, row_bbox_extractor, vision_indicator_added_validator,
and footnotes_utils to eliminate false positives from character/encoding variance.
"""

from __future__ import annotations

import re
import unicodedata

# Apostrophe variants (typographic, modifier letters, fullwidth) -> ASCII '
_APOSTROPHE_VARIANTS_RE = re.compile(r"[\u2019\u2018\u0060\u00b4\u02bc\u02bb\uff07]")
# Elision: "d' impôt" -> "d'impôt" (collapse space after apostrophe for French elision)
_ELISION_SPACE_RE = re.compile(r"\b([dljscnmt])'\s+", re.IGNORECASE)


def normalize_text_base(text: str, *, lowercase: bool = True) -> str:
    """Normalize text for stable comparison: accents, apostrophes, elision, whitespace.

    When lowercase=True (default), also lowercases (matching, footnotes, bbox).
    When lowercase=False, preserves case for indicator_cleaner early pass so
    downstream regexes (dates, units, row numbers) keep expected behavior.

    Order of operations:
    1. NFD + encode ascii ignore (strip accents)
    2. Normalize apostrophe variants to ASCII '
    3. Collapse elision+space (e.g. "d' actions" -> "d'actions")
    4. Collapse all whitespace (including U+00A0) to single space, strip
    5. Lowercase if lowercase=True
    """
    if not text:
        return ""
    # NBSP and typographic apostrophes are not ASCII; normalize before ascii strip
    s = str(text).replace("\u00a0", " ")
    s = _APOSTROPHE_VARIANTS_RE.sub("'", s)
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    s = _ELISION_SPACE_RE.sub(r"\1'", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if lowercase else s
