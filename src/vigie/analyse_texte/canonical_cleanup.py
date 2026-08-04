"""Règles conservatrices de nettoyage du Markdown canonique."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


_SPACE_RE = re.compile(r"[ \t\f\v]+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")
_SPACE_AFTER_OPENING_RE = re.compile(r"([«(\[])\s+")
_SPACE_BEFORE_CLOSING_RE = re.compile(r"\s+([»)\]])")

_BANK_NAME_RE = (
    r"(?:"
    r"banque\s+(?:nationale\s+du\s+canada|scotia|royale\s+du\s+canada|td)"
    r"|bmo(?:\s+groupe\s+financier)?"
    r"|groupe\s+banque\s+td"
    r"|cibc|rbc|bns|bnc|td"
    r")"
)
_QUARTER_LABEL_RE = (
    r"(?:premier|deuxi[eè]me|troisi[eè]me|quatri[eè]me)\s+"
    r"trimestre(?:\s+de)?\s+20\d{2}"
)
_BANK_QUARTERLY_CHROME_RE = re.compile(
    r"^\s*\d{1,3}\s*(?:\|\s*)?"
    + _BANK_NAME_RE
    + r"\s*(?:[-–—|]\s*)?"
    + _QUARTER_LABEL_RE
    + r"(?:\s+rapport\s+de\s+gestion)?\s*$",
    flags=re.IGNORECASE,
)
_QUARTERLY_BANK_CHROME_RE = re.compile(
    r"^\s*\d{1,3}\s*(?:\|\s*)?"
    + _QUARTER_LABEL_RE
    + r"\s*(?:[-–—|]\s*)?"
    + _BANK_NAME_RE
    + r"(?:\s+rapport\s+de\s+gestion)?\s*$",
    flags=re.IGNORECASE,
)
_STANDALONE_TABLE_MARKER_DEFINITION_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?"
    r"(?:"
    r"s\.?\s*o\.?\s*[-–—:]\s*sans\s+objet"
    r"|n\.?\s*s\.?\s*[-–—:]\s*non\s+significati(?:f|ve)"
    r"|n[eé]gl\.?\s*[-–—:]\s*n[eé]gligeable"
    r")\s*[.;]?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CanonicalCleanupDecision:
    """Décision explicable appliquée à un fragment Docling."""

    keep: bool
    text: str
    reason: str = ""
    changed: bool = False


def canonicalize_surface_text(text: str) -> str:
    """Nettoie uniquement la surface sans résumer ni modifier le sens."""
    value = html.unescape(str(text or ""))
    value = value.replace("\u00a0", " ").replace("\u202f", " ").replace("\u200b", "")
    value = _SPACE_RE.sub(" ", value)
    value = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", value)
    value = _SPACE_AFTER_OPENING_RE.sub(r"\1", value)
    value = _SPACE_BEFORE_CLOSING_RE.sub(r"\1", value)
    return value.strip()


def is_quarterly_running_chrome(text: str) -> bool:
    """Détecte les pieds/en-têtes trimestriels, même sans le mot « rapport »."""
    value = canonicalize_surface_text(text)
    return bool(
        _BANK_QUARTERLY_CHROME_RE.fullmatch(value)
        or _QUARTERLY_BANK_CHROME_RE.fullmatch(value)
    )


def is_standalone_table_marker_definition(text: str) -> bool:
    """Détecte une légende autonome de tableau telle que « s. o. – sans objet »."""
    return bool(_STANDALONE_TABLE_MARKER_DEFINITION_RE.fullmatch(canonicalize_surface_text(text)))


def cleanup_canonical_fragment(
    text: str,
    *,
    is_running_chrome: bool = False,
    is_table_unit: bool = False,
    is_chart_axis: bool = False,
    is_not_applicable: bool = False,
) -> CanonicalCleanupDecision:
    """Retourne une décision de nettoyage déterministe et traçable."""
    original = str(text or "")
    cleaned = canonicalize_surface_text(original)
    if not cleaned:
        return CanonicalCleanupDecision(False, "", "empty_fragment", changed=bool(original))
    if is_running_chrome or is_quarterly_running_chrome(cleaned):
        return CanonicalCleanupDecision(False, cleaned, "running_header_footer", changed=cleaned != original)
    if is_table_unit:
        return CanonicalCleanupDecision(False, cleaned, "table_unit_label", changed=cleaned != original)
    if is_chart_axis:
        return CanonicalCleanupDecision(False, cleaned, "chart_axis_labels", changed=cleaned != original)
    if is_not_applicable:
        return CanonicalCleanupDecision(False, cleaned, "standalone_not_applicable", changed=cleaned != original)
    if is_standalone_table_marker_definition(cleaned):
        return CanonicalCleanupDecision(False, cleaned, "table_marker_definition", changed=cleaned != original)
    return CanonicalCleanupDecision(True, cleaned, changed=cleaned != original)


def adjacent_duplicate_key(text: str) -> str:
    """Construit une clé stricte pour les doublons adjacents après nettoyage."""
    value = canonicalize_surface_text(text).casefold()
    return re.sub(r"\s+", " ", value).strip()
