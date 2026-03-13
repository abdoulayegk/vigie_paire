"""Text normalization utilities for table matching."""

from __future__ import annotations

import re
import unicodedata

_TEMPORAL_PATTERNS_BASE = [
    re.compile(r"\b(?:as at|au|au\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", re.IGNORECASE),
    re.compile(
        r"\b(?:for the quarter ended|quarter ended|trimestre termine le)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:trimestre|quarter)\s*(?:[1-4]|i{1,3}|iv)\b", re.IGNORECASE),
    re.compile(r"\b(?:t|q)\s*[1-4]\b", re.IGNORECASE),
    re.compile(r"\b(?:s|h)\s*[12]\b", re.IGNORECASE),
    re.compile(
        r"\b(?:1er|premier|deuxieme|troisieme|quatrieme)\s+trimestre\b", re.IGNORECASE
    ),
]

_TEMPORAL_PATTERNS_AGGRESSIVE = [
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"),
    re.compile(
        r"\b(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
        re.IGNORECASE,
    ),
]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return normalized.encode("ascii", "ignore").decode("utf-8")


def normalize_for_matching(text: str, target: str = "generic") -> str:
    """Normalize text into a stable comparison-friendly representation.

    Used by normalize_indicator_variants in indicator_cleaner. The canonical key
    for indicator diff is normalize_indicator_for_comparison (elision, impôt/impôts,
    guillemets are applied there before/after this step).
    """
    normalized = _strip_accents(text).lower()

    # Keep short date tokens for headers/titles, but still remove punctuation noise.
    normalized = re.sub(r"[_\-–—/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_label(text: str) -> str:
    """Alias used by legacy comparison code."""
    return normalize_for_matching(text, target="indicator")


def strip_temporal_expressions(
    text: str, target: str = "title", aggressive: bool = True
) -> str:
    """Remove date/quarter fragments to keep semantic table title content."""
    value = _strip_accents(text or "")
    if not value:
        return ""

    cleaned = value
    for pattern in _TEMPORAL_PATTERNS_BASE:
        cleaned = pattern.sub(" ", cleaned)

    if aggressive or target in {"title", "header"}:
        for pattern in _TEMPORAL_PATTERNS_AGGRESSIVE:
            cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,./")
    return cleaned


_DATE_ONLY_PATTERNS = [
    # Periodes : "Pour les trois mois", "Pour la periode", "Pour l'exercice"
    re.compile(r"^pour\s+les?\s+trois\s+mois(?:\s+.*)?$", re.IGNORECASE),
    re.compile(r"^pour\s+la\s+periode(?:\s+.*)?$", re.IGNORECASE),
    re.compile(r"^pour\s+l[' ]exercice(?:\s+.*)?$", re.IGNORECASE),
    # Dates precises : "Au 31 octobre", "Au 30 avril 2025", "Au30avril2025 (...)"
    re.compile(
        r"^au\s*\d{1,2}\s*[a-z\u00e0-\u00ff]+(\s*\d{4})?\s*(\(.*\))?\s*$", re.IGNORECASE
    ),
    # Dates sans prefixe : "31 octobre", "30 avril 2025"
    re.compile(r"^\d{1,2}\s+[a-z\u00e0-\u00ff]+(\s+\d{4})?$", re.IGNORECASE),
    # Dates numeriques seules : "31/01/2025", "2025-01-31"
    re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"),
    re.compile(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$"),
    # Fin de trimestre : "Trimestre termine le ...", "Trimestre clos le ..."
    re.compile(r"^trimestre\s+termin[eé](?:\s+le)?(?:\s+.*)?$", re.IGNORECASE),
    re.compile(r"^trimestre\s+clos\s+le(?:\s+.*)?$", re.IGNORECASE),
    re.compile(
        r"^pour\s+le\s+trimestre\s+termin[eé](?:\s+le)?(?:\s+.*)?$", re.IGNORECASE
    ),
    re.compile(r"^pour\s+le\s+trimestre\s+clos(?:\s+le)?(?:\s+.*)?$", re.IGNORECASE),
    # Annees seules : "2024", "2025"
    re.compile(r"^20\d{2}$"),
    # Trimestres seuls : "T2 2025", "Q1 2024"
    re.compile(r"^[tq][1-4]\s+20\d{2}$", re.IGNORECASE),
    # Anglais : "As at April 30, 2025", "For the quarter ended ..."
    re.compile(r"^as\s+at\s+.+$", re.IGNORECASE),
    re.compile(r"^for\s+the\s+quarter\s+ended(?:\s+.*)?$", re.IGNORECASE),
    re.compile(r"^for\s+the\s+three\s+months(?:\s+.*)?$", re.IGNORECASE),
    re.compile(r"^for\s+the\s+period(?:\s+.*)?$", re.IGNORECASE),
]

_UNIT_HEADER_RE = re.compile(
    r"^\(?(?:\s*en\s+(?:millions?|milliards?|milliers?)\s+de\s+dollars"
    r"|\s*in\s+(?:millions?|billions?|thousands?)(?:\s+of\s+(?:Canadian\s+)?dollars)?)"
    r"(?:\s+canadiens)?"
    r"(?:\s*,?\s*(?:sauf\s+indication\s+contraire|except\s+(?:as|where|otherwise)\s+\w+))?"
    r"\s*\)?\s*",
    re.IGNORECASE,
)

# Variante "en milliers d'actions / de parts" (frequente dans rapports TD)
_UNIT_MILLIERS_ACTIONS_RE = re.compile(
    r"^\(?\s*en\s+milliers?\s+(?:d\s*'?\s*actions?|de\s+parts?)",
    re.IGNORECASE,
)

# Lignes de notes de bas de tableau: (1), [2], 1), a), b), Note 1, Note 2., etc.
_NOTE_OR_UNIT_LINE_RE = re.compile(
    r"^\s*(?:[\(\[]\d+[\)\]]|\d+\)|[a-z]\)|note\s*\d+\s*[.:\-–—]?)\s*",
    re.IGNORECASE,
)


def _is_footnote_definition_line(text: str) -> bool:
    """True when text starts with a footnote marker like (1), [2], a), Note 1."""
    if not text or not text.strip():
        return False
    stripped = (text or "").strip()
    return bool(_NOTE_OR_UNIT_LINE_RE.match(stripped))


def _is_unit_header_line(text: str) -> bool:
    """True when *text* is a unit-of-measure header, optionally followed by a date.

    Matches lines that *start* with ``(en millions de dollars ...)``,
    ``(en milliers d'actions...)``, or ``(in millions ...)``.  If real indicator
    text precedes the unit mention (e.g. ``Facteurs de risque (en millions...)``),
    returns False.
    """
    if _UNIT_MILLIERS_ACTIONS_RE.match(text.strip()):
        return True
    m = _UNIT_HEADER_RE.match(text)
    if not m:
        return False
    rest = text[m.end() :].strip()
    if not rest:
        return True
    rest_clean = re.sub(r"[\d\)\*\s]+$", "", rest).strip()
    if not rest_clean:
        return True
    if re.match(r"^(?:au|as\s+at|pour|for)\b", rest_clean, re.IGNORECASE):
        return True
    return False


_PURE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_TOTAL_PREFIX_RE = re.compile(
    r"^\s*total\s+(?:du|des|des\s+elements?\s+hors\s+bilan|du\s+passif)\b",
    re.IGNORECASE,
)
# Regulatory indicators that start with "Total des/du" but must NOT be excluded.
# These are real indicators in Basel III / TLAC reports, not balance-sheet subtotals.
_REGULATORY_TOTAL_ALLOWLIST_RE = re.compile(
    r"^\s*total\s+(?:"
    r"des\s+fonds\s+propres"
    r"|des\s+actifs\s+ponder"
    r"|du\s+capital"
    r"|de\s+la\s+capacite"
    r"|tlac"
    r"|des\s+expositions?"
    r"|des\s+provisions?"
    r"|des\s+prets?"
    r"|des\s+depots?"
    r"|des\s+revenus?"
    r"|des\s+charges?"
    r"|des\s+passifs?"
    r")\b",
    re.IGNORECASE,
)
_TOTAL_PASSIF_CAPITAUX_RE = re.compile(
    r"^\s*total\s+du\s+passif\s+et\s+des\s+capitaux?\s+propres\s*$",
    re.IGNORECASE,
)
_TOTAL_ELEMENTS_HORS_BILAN_RE = re.compile(
    r"^\s*total\s+des\s+elements?\s+hors\s+bilan\s*$", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Section header detection: labels ending with ':' after footnote stripping
# e.g. "Levier(4):", "Ratio de fonds propres (en pourcentage)(4):"
# These are section headers, not data indicators.
# ---------------------------------------------------------------------------
_FOOTNOTE_TRAILING_RE = re.compile(
    r"\s*(?:[\(\[]\d+[\)\]]|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|[*\u2020\u2021\u00A7]+"
    r"|,\s*\d+(?:\s*,\s*\d+)*)*\s*$"
)
# Legitimate indicators that end with ':' but have sub-lines (not section headers)
_SECTION_HEADER_ALLOWLIST_RE = re.compile(
    r"^\s*(?:"
    r"titres?"
    r"|depots?\s+(?:stables?|non\s+stables?|operationnels?|de\s+particuliers?)"
    r"|financement\s+(?:non\s+garanti|garanti|de\s+gros)"
    r"|prets?\s+class"
    r"|titres?\s+liquides?"
    r"|sorties?\s+de\s+tresorerie"
    r"|entrees?\s+de\s+tresorerie"
    r"|engagements?\s+hors\s+bilan"
    r"|actifs?"
    r"|passifs?\s+et\s+(?:capitaux|interdependants?)"
    r"|passifs?\s+(?:et\s+)?autres"
    r"|ventilation"
    r"|autres\s+passifs?"
    r"|depots?"
    r"|fonds\s+propres"
    r")",
    re.IGNORECASE,
)
_UNIT_ONLY_RE = re.compile(
    r"^\s*(?:en\s+millions?\s+de\s+dollars?|sorties|ca|\$\s*ca|%)\s*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Aggressive unit / period header patterns (false-positive suppression)
# CIBC/TD: "En millions", "En millions de dollars", "sauf indication contraire",
# "Au 31 janvier 2025", combined unit+date lines.
# ---------------------------------------------------------------------------
# Truncated or short unit lines (must be checked early)
_UNIT_TRUNCATED_RE = re.compile(r"^\s*en\s*$", re.IGNORECASE)
# ^\s*en\s+(millions?|milliards?|milliers?)\b and any continuation
_UNIT_EN_MILLIONS_RE = re.compile(
    r"^\s*en\s+(?:millions?|milliards?|milliers?)\b",
    re.IGNORECASE,
)
# (en )?(millions?|milliards?)\s*(de\s+)?(dollars?|$|cad|usd)? and optional "sauf indication contraire"
_UNIT_MILLIONS_DOLLARS_RE = re.compile(
    r"^\s*\(?\s*(?:en\s+)?(?:millions?|milliards?|milliers?)\s*(?:de\s+)?(?:dollars?|\$|cad|usd)?\b",
    re.IGNORECASE,
)
_UNIT_OR_PERIOD_HEADER_PATTERNS = [
    _UNIT_TRUNCATED_RE,
    re.compile(
        r"^\s*en\s+(?:millions?|milliards?|milliers?)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\(?\s*en\s+(?:millions?|milliards?|milliers?)\s+de\s+\w+.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*in\s+(?:millions?|billions?|thousands?)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*moyenne\s+du\s+trimestre\s+(?:clos|termin[eé])\b.*$",
        re.IGNORECASE,
    ),
]

# Date with trailing footnote markers: "31 janvier 2025 1, 2"
_DATE_WITH_TRAILING_NOTES_RE = re.compile(
    r"^\s*\d{1,2}\s+(?:janvier|fevrier|f[eé]vrier|mars|avril|mai|juin|juillet|"
    r"ao[uû]t|aout|septembre|octobre|novembre|decembre)"
    r"(?:\s+\d{4})?"
    r"(?:\s+[\d,\s*]+)?\s*$",
    re.IGNORECASE,
)

# "au DD mois YYYY" with trailing noise
_AU_DATE_TRAILING_NOISE_RE = re.compile(
    r"^\s*au\s+\d{1,2}\s+[a-z\u00e0-\u00ff]+(?:\s+\d{4})?"
    r"(?:\s*[\d,\s\)\*]+)?\s*$",
    re.IGNORECASE,
)
# Explicit "Au 31 janvier 2025" (FR) - no trailing required
_DATE_HEADER_AU_DD_MOIS_YYYY_RE = re.compile(
    r"^\s*au\s+\d{1,2}\s+[a-z\u00e0-\u00ff]+\s+\d{4}\b.*$",
    re.IGNORECASE,
)
# English "As of April 30, 2025" / "At December 31, 2024"
_DATE_HEADER_AS_OF_RE = re.compile(
    r"^\s*(?:as\s+of|at)\s+\w+\s+\d{1,2},\s*\d{4}\b.*$",
    re.IGNORECASE,
)


def is_date_only_line(text: str) -> bool:
    """Return True when *text* is purely a date/temporal expression, a
    unit-of-measure header, or a footnote definition that should never
    appear as an indicator change.

    Covers date-only lines, period headers, monetary unit headers such as
    ``(en millions de dollars canadiens) Au 31 janvier 2025``, and footnote
    definition lines like ``(1) Definition`` or ``[2] Note``.
    """
    stripped = _strip_accents((text or "").strip())
    if not stripped:
        return False
    for pattern in _DATE_ONLY_PATTERNS:
        if pattern.match(stripped):
            return True
    if _is_unit_header_line(stripped):
        return True
    for pat in _UNIT_OR_PERIOD_HEADER_PATTERNS:
        if pat.match(stripped):
            return True
    if _DATE_HEADER_AU_DD_MOIS_YYYY_RE.match(stripped):
        return True
    if _DATE_HEADER_AS_OF_RE.match(stripped):
        return True
    if _DATE_WITH_TRAILING_NOTES_RE.match(stripped):
        return True
    if _AU_DATE_TRAILING_NOISE_RE.match(stripped):
        return True
    if _is_footnote_definition_line(stripped):
        return True
    return False


# Known section headers (no colon): normalized form for _classify_excluded_line.
_SECTION_HEADER_ALLOWLIST_NORMALIZED = frozenset({
    "general",
    "liquidite",
    "risque de credit",
    "autres risques",
    "risque de marche",
    "risque operationnel",
})


def _is_section_header_line(text: str) -> bool:
    """True when label is a section header ending with ':' (after stripping footnotes).

    Detects patterns like 'Levier(4):', 'Ratio de fonds propres (en pourcentage)(4):'
    which are structural group headers, not data indicators.
    Returns False for legitimate indicators with sub-lines (e.g. 'Titres :', 'Dépôts stables :').
    """
    if not text or not text.strip():
        return False
    stripped = _strip_accents(text.strip())
    # Strip trailing footnote markers to find the real ending
    core = _FOOTNOTE_TRAILING_RE.sub("", stripped).rstrip()
    if not core.endswith(":"):
        return False
    # Remove the trailing ':' and check the remaining label
    label = core.rstrip(":").strip()
    if not label:
        return False
    # Short labels (≤ 8 words) ending with ':' are likely section headers
    # unless they match the allowlist of legitimate indicators with sub-lines
    # or contain "dont"/"including"/"of which" (breakdown indicators, not headers)
    if len(label.split()) <= 8 and not _SECTION_HEADER_ALLOWLIST_RE.match(label):
        label_lower = label.lower()
        if (
            "dont" in label_lower
            or "including" in label_lower
            or "of which" in label_lower
        ):
            return False
        return True
    return False


def _classify_excluded_line(text: str) -> str | None:
    """Return exclusion type: 'total', 'unit', 'date', 'number', 'footnote', 'section_header', or None if not excluded."""
    if not text or not (text or "").strip():
        return None
    stripped = _strip_accents((text or "").strip())
    if not stripped:
        return None
    if _PURE_NUMBER_RE.match(stripped):
        return "number"
    if _TOTAL_PASSIF_CAPITAUX_RE.match(stripped):
        return "total"
    if _TOTAL_ELEMENTS_HORS_BILAN_RE.match(stripped):
        return "total"
    if _TOTAL_PREFIX_RE.match(stripped) and len(stripped.split()) <= 6:
        if not _REGULATORY_TOTAL_ALLOWLIST_RE.match(stripped):
            return "total"
    if _is_section_header_line(text):
        return "section_header"
    if _UNIT_ONLY_RE.match(stripped):
        return "unit"
    if _UNIT_TRUNCATED_RE.match(stripped):
        return "unit"
    if _UNIT_EN_MILLIONS_RE.match(stripped):
        return "unit"
    if _UNIT_MILLIONS_DOLLARS_RE.match(stripped) and len(stripped.split()) <= 12:
        return "unit"
    if _is_unit_header_line(stripped):
        return "unit"
    for pat in _UNIT_OR_PERIOD_HEADER_PATTERNS:
        if pat.match(stripped):
            return "unit"
    if _is_footnote_definition_line(stripped):
        return "footnote"
    for pattern in _DATE_ONLY_PATTERNS:
        if pattern.match(stripped):
            return "date"
    if _DATE_HEADER_AU_DD_MOIS_YYYY_RE.match(stripped):
        return "date"
    if _DATE_HEADER_AS_OF_RE.match(stripped):
        return "date"
    if _DATE_WITH_TRAILING_NOTES_RE.match(stripped):
        return "date"
    if _AU_DATE_TRAILING_NOISE_RE.match(stripped):
        return "date"
    normalized_label = re.sub(r"\s+", " ", stripped).strip().lower()
    if normalized_label in _SECTION_HEADER_ALLOWLIST_NORMALIZED:
        return "section_header"
    return None


def is_non_indicator_line(text: str) -> bool:
    """Return True when line should not be treated as an indicator (totals, units, numbers)."""
    return _classify_excluded_line(text) is not None


_DATE_HEADER_RE = re.compile(
    r"\b(?:au|as at|ended|termine|clos|trimestre|quarter|q[1-4]|t[1-4]|"
    r"janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_YEAR_HEADER_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_CURRENCY_HEADER_RE = re.compile(
    r"\b(?:dollars?|cad|usd|eur|en\s+millions?|en\s+milliards?|in\s+millions?|in\s+billions?)\b",
    re.IGNORECASE,
)
_PERCENT_HEADER_RE = re.compile(r"(?:%|pourcentage|percent)", re.IGNORECASE)
_NUMBER_HEADER_RE = re.compile(r"\d")


def infer_header_schema_type(value: str) -> str:
    """Infer a coarse semantic type for one header cell."""
    text = normalize_for_matching(value or "", target="header")
    if not text:
        return "EMPTY"
    if _PERCENT_HEADER_RE.search(text):
        return "PERCENT"
    if _CURRENCY_HEADER_RE.search(text):
        return "CURRENCY"
    if _DATE_HEADER_RE.search(text):
        return "DATE"
    if _YEAR_HEADER_RE.search(text):
        return "YEAR"
    if _NUMBER_HEADER_RE.search(text):
        return "NUMBER"
    return "TEXT"


def infer_header_schema(headers: list[str] | tuple[str, ...] | None) -> list[str]:
    """Map a header row to coarse semantic types used for robust matching."""
    if not headers:
        return []
    return [infer_header_schema_type(str(item)) for item in headers]


def header_schema_similarity(
    headers1: list[str] | tuple[str, ...] | None,
    headers2: list[str] | tuple[str, ...] | None,
) -> float:
    """Compare header schemas (DATE/YEAR/CURRENCY/PERCENT/TEXT) with position tolerance."""
    schema1 = infer_header_schema(headers1)
    schema2 = infer_header_schema(headers2)
    if not schema1 or not schema2:
        return 0.0

    max_len = max(len(schema1), len(schema2), 1)
    same_pos = 0
    for idx in range(min(len(schema1), len(schema2))):
        if schema1[idx] == schema2[idx]:
            same_pos += 1

    set_score = len(set(schema1) & set(schema2)) / len(set(schema1) | set(schema2))
    pos_score = same_pos / max_len
    return max(0.0, min(1.0, (0.6 * pos_score) + (0.4 * set_score)))


def header_literal_fingerprint(
    headers1: list[str] | tuple[str, ...] | None,
    headers2: list[str] | tuple[str, ...] | None,
) -> float:
    """Jaccard on normalized literal column header texts.

    Complements :func:`header_schema_similarity` (which maps to abstract types)
    with an exact-text comparison that preserves discriminative column names.
    Useful for disambiguating tables that share first-column indicators but have
    different column layouts (e.g. VaR variants).
    """
    if not headers1 or not headers2:
        return 0.0
    norm1 = [normalize_for_matching(str(h), target="header") for h in headers1 if str(h or "").strip()]
    norm2 = [normalize_for_matching(str(h), target="header") for h in headers2 if str(h or "").strip()]
    s1 = {v for v in norm1 if v}
    s2 = {v for v in norm2 if v}
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def is_generic_title(
    title: str, generic_titles: set[str] | frozenset[str] | None = None
) -> bool:
    """Return True when title is too generic to be a strong identifier."""
    value = normalize_for_matching(title, target="title")
    if not value:
        return True

    if generic_titles:
        normalized_set = {
            normalize_for_matching(item, target="title") for item in generic_titles
        }
        if value in normalized_set:
            return True

    default_generic = {
        "tableau",
        "table",
        "resultats",
        "informations",
        "donnees",
        "annexe",
        "tableau principal",
        "details",
        "autres",
        "total",
    }
    return value in default_generic
