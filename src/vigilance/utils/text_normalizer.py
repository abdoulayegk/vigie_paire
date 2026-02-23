"""Text normalizer used by legacy comparison modules."""

from __future__ import annotations

import re

from vigilance.utils.matching_normalizer import normalize_for_matching


class TextNormalizer:
    """Compatibility wrapper for legacy static normalization helpers."""

    @staticmethod
    def normalize(
        text: str,
        *,
        aggressive: bool = False,
        remove_notes: bool = False,
        lowercase: bool = True,
    ) -> str:
        value = text or ""
        if remove_notes:
            value = re.sub(r"\s*(?:\(\d+\)|\[\d+\]|\*+)\s*", " ", value)
        value = normalize_for_matching(value)
        if not lowercase:
            value = value.upper()
        if aggressive:
            value = re.sub(r"\b(?:de|des|du|la|le|les|et|ou)\b", " ", value)
            value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def normalize_indicator(text: str) -> str:
        return TextNormalizer.normalize(
            text,
            aggressive=True,
            remove_notes=True,
            lowercase=True,
        )
