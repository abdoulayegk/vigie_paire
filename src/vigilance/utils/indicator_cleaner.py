"""Helpers to clean indicator labels and table titles."""

from __future__ import annotations

import logging
import re
import unicodedata

from vigilance.utils.matching_normalizer import (
    _UNIT_HEADER_RE,
    _UNIT_MILLIERS_ACTIONS_RE,
    is_date_only_line,
    is_non_indicator_line,
    normalize_label,
    strip_temporal_expressions,
)

logger = logging.getLogger(__name__)

_TRAILING_NOTE_RE = re.compile(r"\s*(?:\(\d+\)|\[\d+\]|\*+)\s*$")
_TRAILING_NUM_RE = re.compile(r"\s+\d{1,4}(?:[.,]\d+)?\s*$")
_DATE_IN_TITLE_RE = re.compile(
    r"\b(?:au|as at|for the quarter ended|trimestre termine le)\b.*$", re.IGNORECASE
)
# CIBC-style dates at start or end: "31 janvier 2025", "30 avril 2025 1, 2"
_DATE_STANDALONE_RE = re.compile(
    r"^\s*\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
    r"septembre|octobre|novembre|decembre)(?:\s+\d{4})?\s*$",
    re.IGNORECASE,
)
_LEADING_DATE_RE = re.compile(
    r"^\s*\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
    r"septembre|octobre|novembre|decembre)(?:\s+\d{4})?\s+",
    re.IGNORECASE,
)
_TRAILING_DATE_RE = re.compile(
    r"\s+\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
    r"septembre|octobre|novembre|decembre)(?:\s+\d{4})?\s*$",
    re.IGNORECASE,
)

# RBC header/footer: "24 Banque Royale du Canada Premier trimestre de 2025"
_HEADER_FOOTER_RBC_RE = re.compile(
    r"(?:\d+\s+)?Banque\s+Royale\s+du\s+Canada\s+"
    r"(?:Premier|Deuxieme|Deuxième|Troisieme|Troisième|Quatrieme|Quatrième)\s+trimestre",
    re.IGNORECASE,
)
_HEADER_FOOTER_PAGE_BANK_RE = re.compile(
    r"^\s*\d+\s+[A-Za-z].*(?:trimestre|quarter)\b", re.IGNORECASE
)

# --- Patterns pour strip_dates_from_indicator_label ---
_MONTHS_FR = (
    r"janv(?:ier)?|fevr(?:ier)?|f[eé]vr(?:ier)?|mars|avr(?:il)?|mai|juin|juill(?:et)?|"
    r"ao[uû]t|aout|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|d[eé]c(?:embre)?"
)
_MONTHS_EN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# Prefixes temporels: au, a (preposition seule), as at, for the period ended, etc.
# "a" doit etre suivi d'espace pour eviter de matcher "Actifs", "Actifs greves", etc.
_INDICATOR_DATE_PREFIX_RE = re.compile(
    r"^\s*(?:au\b|a\s|as\s+at|as\s+of|for\s+the\s+(?:period|quarter)\s+ended|"
    r"trimestre\s+termin[eé]\s+le|pour\s+la\s+periode\s+close\s+le)\s*",
    re.IGNORECASE,
)
# Suffixe date: " au 30 avril 2025", " - 31/01/2025", " (30 avril 2025)"
_INDICATOR_DATE_SUFFIX_RE = re.compile(
    r"(?:[\s\-–—,;]+(?:au|a|as\s+at|as\s+of|le|du)\s+)?"
    r"(?:\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4}"
    r"|\d{4}[\s./-]\d{1,2}[\s./-]\d{1,2}"
    r"|\d{1,2}\s+(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*(?:\d{2,4})?"
    r"|(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*\d{2,4}"
    r"|[tq][1-4]\s+20\d{2}"
    r"|(?:1er|premier|deuxieme|troisieme|quatrieme)\s+trimestre\s+20\d{2}"
    r"|exercice\s+20\d{2})\s*$",
    re.IGNORECASE,
)
# Ligne entiere = date seule (pour early return "")
_INDICATOR_DATE_STANDALONE_RE = re.compile(
    r"^\s*(?:au|a|as\s+at|as\s+of|for\s+the\s+(?:period|quarter)\s+ended)\s+"
    r"(?:\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4}|\d{4}[\s./-]\d{1,2}[\s./-]\d{1,2}|"
    r"\d{1,2}\s+(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*\d{4}?|"
    r"(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*\d{4})\s*$",
    re.IGNORECASE,
)

# --- Patterns pour strip_units_currency_from_indicator_label ---
# \b ensures we don't match "en" inside words (e.g. "canadiens")
_UNITS_IN_INDICATOR_PHRASE_RE = re.compile(
    r"[\s\-–—]+(?:"
    r"(?:en\s+)?(?:millions?|milliards?|milliers?)\s+de\s+dollars?\s+canadiens?"
    r"|(?:en\s+)?(?:millions?|milliards?|milliers?)\s+de\s+dollars?"
    r"|(?:en\s+)?(?:millions?|milliards?|milliers?)\s+dollars?"
    r"|en\s+milliers?\s+(?:d\s*['']?\s*actions?|de\s+parts?)"
    r"|en\s+\$|en\s+G\s*\$|en\s+M\s*\$|en\s+k\s*\$"
    r"|(?:en\s+)?pourcentage|points?\s+de\s+base|\bpb\b|\bbps\b"
    r")\s*$",
    re.IGNORECASE,
)
_UNITS_IN_INDICATOR_PAREN_RE = re.compile(
    r"\s*\(\s*(?:"
    r"(?:en\s+)?(?:millions?|milliards?|milliers?)\s*(?:de\s+dollars?)?"
    r"|M\s*\$?|G\s*\$?|k\s*\$?"
    r"|%|CAD|USD|EUR"
    r")\s*\)\s*",
    re.IGNORECASE,
)


def strip_dates_from_indicator_label(text: str) -> str:
    """
    Remove date/temporal fragments from indicator labels (first column).

    Handles: standalone dates ("Au 30 avril 2025"), suffix ("Prets garantis au 30 avril 2025"),
    prefix, and multiple formats (FR, ISO, numeric, abbreviations, periods).
    Returns empty string when the label is purely a date expression.
    """
    if not text or not (text or "").strip():
        return text or ""
    value = (text or "").strip()
    if _INDICATOR_DATE_STANDALONE_RE.match(value):
        return ""
    value = _INDICATOR_DATE_PREFIX_RE.sub("", value)
    value = _INDICATOR_DATE_SUFFIX_RE.sub("", value)
    value = strip_temporal_expressions(value, target="title", aggressive=True)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def strip_units_currency_from_indicator_label(text: str) -> str:
    """
    Remove unit/currency phrases from indicator labels (first column).

    Strips: "en millions de dollars", "(M$)", "(CAD)", "(%)", "en milliers",
    "points de base", etc. Preserves semantic core (e.g. "Ratio CET1" from "Ratio CET1 (%)").
    """
    if not text or not (text or "").strip():
        return text or ""
    value = (text or "").strip()
    value = _UNITS_IN_INDICATOR_PAREN_RE.sub("", value)
    value = _UNITS_IN_INDICATOR_PHRASE_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_header_footer_table_title(title: str, bank_code: str | None = None) -> bool:
    """Return True if title is a header/footer (page num + bank + quarter), not semantic."""
    if not title or not (title or "").strip():
        return False
    t = (title or "").strip()
    if _HEADER_FOOTER_RBC_RE.search(t):
        return True
    if (bank_code or "").lower() == "rbc" and _HEADER_FOOTER_PAGE_BANK_RE.match(t):
        return True
    return bool(_HEADER_FOOTER_PAGE_BANK_RE.match(t))


def strip_trailing_note_or_column_value(text: str) -> str:
    """Remove trailing note markers and numeric column residue."""
    value = text or ""
    value = _TRAILING_NOTE_RE.sub("", value)
    value = _TRAILING_NUM_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def is_trailing_number_semantic(text: str) -> bool:
    """Detect cases where trailing numbers are semantically relevant."""
    value = normalize_label(text)
    if not value:
        return False
    return bool(
        re.search(r"\b(?:pilier|pillar|niveau|tier|phase|etape)\s+[0-9]+\b", value)
        or re.search(r"\b(?:ratio|note|scenario)\s+[0-9]+\b", value)
        or re.search(r"\b(?:serie|series|tranche|classe|class)\s+[0-9]+\b", value)
        or re.search(r"\b(?:categorie|category)\s+[0-9]+\b", value)
    )


def strip_dates_from_table_title(title: str) -> str:
    """Remove date-like fragments from table titles (CIBC, RBC).

    Handles: "au 31 octobre...", "31 janvier 2025", "30 avril 2025" at start/end.
    """
    value = title or ""
    value = _DATE_IN_TITLE_RE.sub("", value)
    if _DATE_STANDALONE_RE.match(value.strip()):
        return ""
    value = _LEADING_DATE_RE.sub("", value)
    value = _TRAILING_DATE_RE.sub("", value)
    value = strip_temporal_expressions(value, target="title", aggressive=True)
    return re.sub(r"\s+", " ", value).strip()


def strip_note_refs_from_title(title: str) -> str:
    """Remove trailing note references from table titles (e.g. NOTATIONS DE CREDIT1)."""
    value = title or ""
    # Trailing (1), [2], 1), or bare digit 1, 2 when likely a ref
    value = re.sub(r"\s*[\(\[]\d+[\)\]]\s*$", "", value)
    value = re.sub(r"\s*\d+\)\s*$", "", value)
    value = re.sub(r"([a-zA-ZÀ-ÿ])\d+\s*$", r"\1", value)  # CREDIT1 -> CREDIT, garde "Tableau 12"
    value = re.sub(r"\s+\d+\s*,\s*\d+(?:\s*,\s*\d+)*\s*$", "", value)  # " 1, 2", " 1, 2, 3" (virgule requise)
    return re.sub(r"\s+", " ", value).strip()


_UNITS_IN_TITLE_RE = re.compile(
    r"\s*[\(\[]?\s*"
    r"(?:en\s+)?(?:millions?|milliards?)\s*(?:de\s+dollars?|\s*\$|dollars?)?"
    r"|[\(\[]?\s*en\s+\$\s*[\)\]]?"
    r"|[\(\[]?\s*(?:en\s+)?dollars?\s*[\)\]]?"
    r"|[\(\[]?\s*%\s*[\)\]]?"
    r"\s*[\)\]]?\s*",
    re.IGNORECASE,
)


def strip_units_from_table_title(title: str, bank_code: str | None = None) -> str:
    """Remove unit phrases from table titles for semantic comparison (e.g. RBC).

    Strips leading/trailing fragments like '(en millions)', 'en $', '(en millions de dollars)'.
    """
    value = title or ""
    value = _UNITS_IN_TITLE_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if bank_code and (bank_code or "").strip().lower() == "rbc":
        # RBC-specific: trim trailing " - long subtitle" beyond first separator
        parts = value.split(" - ", 1)
        if len(parts) == 2 and len(parts[1].split()) > 12:
            value = (parts[0] + " - " + " ".join(parts[1].split()[:12])).strip()
    return value


def normalize_indicator_variants(text: str) -> str:
    """Normalize indicator variants into a canonical string."""
    value = text or ""
    value = _TRAILING_NOTE_RE.sub("", value)
    if not is_trailing_number_semantic(value):
        value = _TRAILING_NUM_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return normalize_label(value)


def _word_char_count(s: str) -> int:
    """Count alphanumeric characters for fallback logic."""
    return len(re.findall(r"\w", s or ""))


# --- PART A: camelCase / concatenated-token boundary splitter ---
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")


def split_camel_case_concatenation(text: str) -> tuple[str, bool]:
    """Insert space at lowercase-to-uppercase boundaries.

    Handles concatenated tokens like "impotAJOUTactions" or "fondsPropresTier1".
    Returns (cleaned_text, True if any split was applied).
    """
    if not text:
        return text or "", False
    result = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", text)
    return result, result != text


# --- PART B: space after change-tag prefix ---
_CHANGE_TAG_PREFIX_RE = re.compile(
    r"^(AJOUT|SUPPRESSION|RENOMMAGE)(?=[a-zA-ZÀ-ÿ])", re.UNICODE
)


def insert_space_after_change_tag(text: str) -> tuple[str, bool]:
    """Insert space after AJOUT/SUPPRESSION/RENOMMAGE if directly followed by a letter.

    Handles indicators like "AJOUTactions ordinaires" -> "AJOUT actions ordinaires".
    Returns (cleaned_text, True if correction was applied).
    """
    if not text:
        return text or "", False
    result = _CHANGE_TAG_PREFIX_RE.sub(r"\1 ", text)
    return result, result != text


def post_normalize_indicator(text: str) -> tuple[str, bool, bool]:
    """Apply camelCase split and change-tag space fix after canonical normalization.

    Returns (text, camel_split_triggered, tag_space_triggered).
    Must be called on the *cleaned* indicator value, not raw.
    """
    if not text:
        return "", False, False
    result, camel = split_camel_case_concatenation(text)
    result, tag = insert_space_after_change_tag(result)
    return result, camel, tag


def normalize_indicator_for_comparison(text: str) -> str:
    """
    Single canonical key for indicator (first column) comparison.

    Used by comparison_runner and structural_comparator so that the same
    semantic label (e.g. "Metaux precieux" vs "Metaux precieux :") produces
    one key and avoids false additions/deletions in the change detail.

    - Unicode NFD and normalize all whitespace (including U+00A0) to space
    - Strip dates and units/currency from label (e.g. "au 30 avril 2025", "(M$)")
    - Variants (CET-1, Tier-1, trailing notes)
    - Strip accents, note refs (1) [2] a) *, exposants
    - Normalize punctuation and spaces, lowercase
    """
    if not text:
        return ""

    # Normalize Unicode and collapse all whitespace (including U+00A0) to space
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Lines that are purely date/unit/note/footnote should yield empty (idempotent with extraction filter)
    if is_date_only_line(text) or is_non_indicator_line(text):
        return ""

    # Strip dates and units so "Prets garantis au 30 avril 2025" = "Prets garantis (M$)" = "Prets garantis"
    raw_for_log = text
    dates_stripped = strip_dates_from_indicator_label(text)
    if not dates_stripped or not dates_stripped.strip():
        return ""
    units_stripped = strip_units_currency_from_indicator_label(dates_stripped)
    if _word_char_count(units_stripped) >= 2:
        text = units_stripped
    elif _word_char_count(dates_stripped) >= 2:
        text = dates_stripped
    else:
        text = dates_stripped
    if logger.isEnabledFor(logging.DEBUG) and raw_for_log != text:
        logger.debug("indicator raw -> cleaned: %r -> %r", raw_for_log[:60], text[:60])

    # Strip leading unit-of-measure prefix so "En millions de dollars - X" and "X" share the same key
    for _re in (_UNIT_HEADER_RE, _UNIT_MILLIERS_ACTIONS_RE):
        m = _re.match(text)
        if m:
            rest = text[m.end() :].strip()
            if rest:
                text = rest
            break

    # Variantes frequentes (tirets, separateurs, CET-1, Tier-1)
    text = normalize_indicator_variants(text)

    # Supprimer les accents (NFD + ASCII)
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    # References de notes: (1), (2), [1], [2], etc.
    text = re.sub(r"\s*[\(\[]\d+[\)\]]\s*", " ", text)

    # Appels de notes lettres: a), b), a,b, etc. (Spec Basel III)
    text = re.sub(r"\s*[a-z]\)\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[a-z],\s*[a-z]\s*", " ", text, flags=re.IGNORECASE)

    # Patterns numeriques sans parenthese ouvrante: 2), 3)
    text = re.sub(r"\s*\d+\)\s*", " ", text)

    # Exposants et asterisques
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", text)
    text = re.sub(r"\*+", "", text)

    # Chiffres isolés en fin (notes ou colonnes), sauf semantique (pilier, tier, serie, etc.)
    if not is_trailing_number_semantic(text):
        text = strip_trailing_note_or_column_value(text)

    # Normaliser la ponctuation
    text = re.sub(r"[:\-–—]", " ", text)

    # Variantes linguistiques: "des taux" <-> "de taux" -> meme cle
    text = re.sub(r"\bdes\s+taux\b", "de taux", text, flags=re.IGNORECASE)

    # Espaces et minuscules
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text
