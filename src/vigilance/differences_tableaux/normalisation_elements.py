"""Normalisation des indicateurs, notes et contextes de tableaux."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalize_footnotes(raw: Any) -> list[dict[str, str]]:
    """Normalise une liste brute de notes de bas de page en dicts ``{id, text}``."""
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not fid and not text:
            continue
        normalized.append({"id": fid, "text": text})
    return normalized


_FOOTNOTE_MARKER_RE = re.compile(r"\s*[\(\[]\d{1,2}[\)\]]\s*")


_SUPERSCRIPT_DIGITS = str.maketrans("", "", "\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070")


def _normalize_indicator_text(name: str) -> str:
    """Normalise un nom d'indicateur pour la comparaison deterministe par ensemble."""
    text = str(name or "").strip()
    text = _FOOTNOTE_MARKER_RE.sub("", text)
    text = text.translate(_SUPERSCRIPT_DIGITS)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _token_overlap_ratio(a: str, b: str) -> float:
    """Retourne le ratio de chevauchement de tokens (type Jaccard) entre deux chaines normalisees."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


_DATE_QUARTER_RE = re.compile(
    r"\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4}"
    r"|T[1-4]\s*[-–]?\s*\d{4}"
    r"|\d{4}\s*[-–]?\s*T[1-4]"
    r"|(?:premier|deuxième|troisième|quatrième)\s+trimestre\s+\d{4}",
    re.IGNORECASE,
)


_PAGE_REF_RE_DET = re.compile(r"pages?\s+\d+\s*[àa]\s*\d+", re.IGNORECASE)


def _normalize_footnote_text(text: str) -> str:
    """Normalise le texte d'une note de bas de page pour comparaison deterministe (retire dates/pages/espaces)."""
    text = str(text or "").strip()
    text = _DATE_QUARTER_RE.sub("__DATE__", text)
    text = _PAGE_REF_RE_DET.sub("__PAGE__", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


_FOOTNOTE_PAREN_RE = re.compile(r"\s*[\(\[]\d{1,2}[\)\]]\s*$")


_SUPERSCRIPT_STRIP = str.maketrans("", "", "\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070")


_DASH_RE = re.compile(r"\s*[–—−‐]\s*")


_BLOC_SUFFIX_RE = re.compile(r"\s*\(bloc\s+\d+\)\s*$", re.IGNORECASE)


_DATE_PREFIX_RE = re.compile(
    r"^(?:Au\s+)?\d{1,2}\s+"
    r"(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
    r"\s+\d{4}\s*[–—−\-]\s*",
    re.IGNORECASE,
)


_STANDALONE_DATE_RE = re.compile(
    r"^(?:Au\s+)?\d{1,2}\s+"
    r"(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
    r"\s+\d{4}\s*$",
    re.IGNORECASE,
)


def _normalize_for_diff(name: str) -> str:
    """Normalize an indicator name for matching: strip footnotes, superscripts, punctuation noise."""
    text = str(name or "").strip()
    # Strip trailing footnote markers: (1), [2], etc.
    text = _FOOTNOTE_PAREN_RE.sub("", text)
    # Strip superscript digits: ¹²³⁴⁵⁶⁷⁸⁹⁰
    text = text.translate(_SUPERSCRIPT_STRIP)
    # Strip disambiguation: (bloc N), date prefixes
    text = _BLOC_SUFFIX_RE.sub("", text)
    text = _DATE_PREFIX_RE.sub("", text)
    # Normalize dashes to simple hyphen
    text = _DASH_RE.sub(" - ", text)
    # Strip trailing colon/punctuation
    text = text.rstrip(":;,. ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _enrich_indicators_with_normalized(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add a ``normalized`` field to each indicator, stripping hierarchical prefixes.

    If indicator[i] starts with indicator[j].name + " – " (for some j < i that looks
    like a section header), the normalized form keeps only the suffix.
    """
    enriched: list[dict[str, Any]] = []
    # Build a set of potential section headers (names that appear as prefixes of later items)
    names = [item["name"] for item in items]
    norm_names = [_normalize_for_diff(n) for n in names]

    # Detect which indices are section headers used as prefixes
    header_norms: set[str] = set()
    for i, nn in enumerate(norm_names):
        if not nn:
            continue
        for j in range(i + 1, min(i + 8, len(norm_names))):
            nj = norm_names[j]
            prefix_dash = nn + " - "
            if nj.startswith(prefix_dash) and len(nj) > len(prefix_dash):
                header_norms.add(nn)
                break

    for idx, item in enumerate(items):
        norm = _normalize_for_diff(item["name"])
        # Try to strip hierarchical prefix from section headers above
        stripped = norm
        for hi in range(idx - 1, max(idx - 8, -1), -1):
            h_norm = norm_names[hi] if hi < len(norm_names) else ""
            if h_norm in header_norms:
                prefix_dash = h_norm + " - "
                if stripped.startswith(prefix_dash) and len(stripped) > len(prefix_dash):
                    stripped = stripped[len(prefix_dash) :]
                    break

        entry: dict[str, Any] = {"pos": item["pos"], "name": item["name"]}
        if stripped != item["name"]:
            entry["normalized"] = stripped
        enriched.append(entry)

    return enriched


def _table_context(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrait le contexte normalise d'un tableau pour les prompts GPT."""
    indicators = list(entry.get("indicators", []) or [])
    raw_items = [
        {"pos": i, "name": unicodedata.normalize("NFC", str(value).strip())}
        for i, value in enumerate(indicators)
        if str(value).strip() and not _STANDALONE_DATE_RE.match(str(value).strip())
    ]
    enriched = _enrich_indicators_with_normalized(raw_items)
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": entry.get("page"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [str(value).strip() for value in list(entry.get("headers", []) or []) if str(value).strip()],
        "indicators": enriched,
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }
