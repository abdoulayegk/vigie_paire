"""Helpers to clean indicator labels and table titles."""

from __future__ import annotations

import re
import unicodedata

from vigilance.utils.matching_normalizer import normalize_label, strip_temporal_expressions

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


def normalize_indicator_for_comparison(text: str) -> str:
    """
    Single canonical key for indicator (first column) comparison.

    Used by comparison_runner and structural_comparator so that the same
    semantic label (e.g. "Metaux precieux" vs "Metaux precieux :") produces
    one key and avoids false additions/deletions in the change detail.

    - Unicode NFD and normalize all whitespace (including U+00A0) to space
    - Variants (CET-1, Tier-1, trailing notes)
    - Strip accents, note refs (1) [2] a) *, exposants
    - Normalize punctuation and spaces, lowercase
    """
    if not text:
        return ""

    # Normalize Unicode and collapse all whitespace (including U+00A0) to space
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"\s+", " ", text).strip()

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
