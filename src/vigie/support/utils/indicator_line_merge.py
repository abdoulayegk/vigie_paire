"""Fusion deterministe de lignes d'indicateurs fragmentees (heuristiques OCR/Docling securitaires)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRONG_PUNCTUATION_END_RE = re.compile(r"[.!?;:]\s*$")
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.)]\s+")
_LETTERED_ITEM_RE = re.compile(r"^\s*[A-Z]\.\s+")
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9\u00b9\u00b2\u00b3\u2070-\u209f]+", re.UNICODE)

_MARKER_CHARS = set("¹²³⁴⁵⁶⁷⁸⁹⁰*†‡[](){}<>.,;:-–—/\\")


@dataclass(frozen=True)
class IndicatorLineMergeConfig:
    """Configuration pour la fusion deterministe de lignes."""

    max_next_tokens: int = 6
    max_combined_length: int = 120
    uppercase_long_tokens_threshold: int = 4
    uppercase_long_char_threshold: int = 20
    marker_ratio_threshold: float = 0.70


_NEW_ITEM_HEADERS = {
    "total",
    "region",
    "région",
    "region:",
    "région:",
    "passifs",
    "actifs",
    "capitaux propres",
    "capital",
    "resultat",
    "résultat",
}

_CONTINUATION_WORDS = {
    "a",
    "à",
    "au",
    "aux",
    "avec",
    "category",
    "catégorie",
    "categories",
    "catégories",
    "de",
    "des",
    "du",
    "en",
    "et",
    "fonds",
    "for",
    "of",
    "or",
    "ordinaires",
    "par",
    "pour",
    "propres",
    "supplementaire",
    "supplementaires",
    "supplémentaire",
    "supplémentaires",
    "sur",
    "termes",
    "the",
}


def _normalize_spaces(text: str) -> str:
    """Normalise les espaces multiples en un seul espace."""
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> list[str]:
    """Decoupe le texte en jetons alphanumeriques."""
    return _TOKEN_RE.findall(text or "")


def _starts_with_lowercase(text: str) -> bool:
    """Retourne True si le texte commence par une minuscule."""
    value = (text or "").strip()
    return bool(value) and value[0].islower()


def _is_mostly_markers_or_superscripts(text: str, *, ratio_threshold: float) -> bool:
    """Retourne True si le texte est majoritairement compose de marqueurs ou exposants."""
    value = (text or "").strip()
    if not value:
        return False
    non_space = [ch for ch in value if not ch.isspace()]
    if not non_space:
        return False
    marker_count = sum(1 for ch in non_space if ch in _MARKER_CHARS)
    return (marker_count / len(non_space)) >= ratio_threshold


def _looks_like_new_item(text: str, *, config: IndicatorLineMergeConfig) -> bool:
    """Retourne True si le texte ressemble au debut d'un nouvel indicateur."""
    value = _normalize_spaces(text)
    if not value:
        return False
    if _NUMBERED_ITEM_RE.match(value) or _LETTERED_ITEM_RE.match(value):
        return True

    lowered = value.lower()
    if lowered in _NEW_ITEM_HEADERS:
        return True

    tokens = _tokenize(value)
    if (
        value[0].isupper()
        and len(tokens) >= config.uppercase_long_tokens_threshold
        and len(value) >= config.uppercase_long_char_threshold
    ):
        return True
    return False


def _looks_like_continuation_word_list(text: str) -> bool:
    """Retourne True si tous les jetons sont des mots de continuation."""
    tokens = [t.lower() for t in _tokenize(text)]
    if not tokens:
        return False
    return all(t in _CONTINUATION_WORDS for t in tokens)


def _can_merge(
    previous: str, current: str, *, config: IndicatorLineMergeConfig
) -> bool:
    """Determine si la ligne courante peut etre fusionnee avec la precedente."""
    prev = _normalize_spaces(previous)
    cur = _normalize_spaces(current)
    if not prev or not cur:
        return False
    if _looks_like_new_item(cur, config=config):
        return False
    if _STRONG_PUNCTUATION_END_RE.search(prev):
        return False

    starts_like_continuation = _starts_with_lowercase(
        cur
    ) or _is_mostly_markers_or_superscripts(
        cur,
        ratio_threshold=config.marker_ratio_threshold,
    )
    if not starts_like_continuation:
        return False

    token_count = len(_tokenize(cur))
    if token_count <= config.max_next_tokens or _looks_like_continuation_word_list(cur):
        merged = _normalize_spaces(f"{prev} {cur}")
        if config.max_combined_length > 0 and len(merged) > config.max_combined_length:
            return False
        return True
    return False


def merge_indicator_lines(
    indicators: list[str], *, config: IndicatorLineMergeConfig | None = None
) -> tuple[list[str], int]:
    """Fusionne les lignes d'indicateurs fragmentees avec des regles deterministes et conservatrices.

    Args:
        indicators: Liste de libelles d'indicateurs potentiellement fragmentes.
        config: Configuration de fusion ; utilise les valeurs par defaut si None.

    Returns:
        Tuple ``(liste_fusionnee, nombre_fusions)``.
    """
    if not indicators:
        return [], 0
    if len(indicators) < 2:
        cleaned = [_normalize_spaces(x) for x in indicators if _normalize_spaces(x)]
        return cleaned, 0

    cfg = config or IndicatorLineMergeConfig()
    result: list[str] = []
    merge_count = 0
    index = 0

    while index < len(indicators):
        current = _normalize_spaces(indicators[index])
        if not current:
            index += 1
            continue

        next_index = index + 1
        while next_index < len(indicators):
            nxt = _normalize_spaces(indicators[next_index])
            if not nxt:
                next_index += 1
                continue
            if not _can_merge(current, nxt, config=cfg):
                break
            current = _normalize_spaces(f"{current} {nxt}")
            merge_count += 1
            next_index += 1

        result.append(current)
        index = next_index

    return result, merge_count
