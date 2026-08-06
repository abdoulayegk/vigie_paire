"""Heuristiques de qualite sur les indicateurs extraits (faibles, narratifs, contamines).

Extrait de ``vision_full_extractor.py`` sans modification.
"""

from __future__ import annotations

import re
from typing import Any

from .result import VisionFullResult

_GENERIC_PAGE_TITLES = {
    "rapport de gestion",
    "management's discussion and analysis",
    "management discussion and analysis",
    "rapport annuel",
    "shareholders report",
    "rapport aux actionnaires",
}

_WEAK_INDICATOR_EXACT = {
    "total",
    "totaux",
    "autres",
    "autre",
    "other",
    "others",
    "canada",
    "united states",
    "états-unis",
    "etats-unis",
}

_WEAK_INDICATOR_TOKENS = {
    "total",
    "totaux",
    "autres",
    "autre",
    "other",
    "others",
    "canada",
    "united",
    "states",
    "états",
    "etats",
    "unis",
}

_NARRATIVE_INDICATOR_PHRASES = (
    "texte narratif",
    "rapport de gestion",
    "cette section",
    "ce tableau",
    "comprend",
    "comprennent",
    "inclut",
    "incluent",
    "présente",
    "presente",
    "présentent",
    "presentent",
    "représente",
    "represente",
    "décrit",
    "decrit",
    "description",
)

_DATE_RE = re.compile(
    r"^\s*(au\s+\d{1,2}\s+\w+\s+\d{4}|\d{1,2}\s+\w+\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|[tTqQ][1-4]\s*\d{4}|\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4})\s*$",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"^\s*[\(\-]?[\d\s.,]+[\)%]?\s*$")

_SUPERSCRIPT_FOOTNOTE_TRANSLATION = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹",
    "0123456789",
)


def _extract_native_text_indicators(reference_text: str) -> list[str]:
    """Extrait les candidats indicateurs du texte natif (Docling).

    Ces candidats servent de troisieme source de vote dans ``_select_consensus``.
    Ils ne peuvent PAS introduire de nouveaux libelles — ils ne font que renforcer
    les libelles deja identifies par Vision.

    Args:
        reference_text: Texte brut extrait par Docling pour ce tableau.

    Returns:
        Liste de candidats indicateurs (au plus 200), dans l'ordre visuel.
    """
    candidates: list[str] = []
    for line in reference_text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        # Reject purely numeric lines
        if re.match(r"^[\d\s\.,\-\(\)%]+$", line):
            continue
        # Reject date/period-like patterns
        if _is_period_like_indicator(line):
            continue
        # Reject short all-caps tokens (likely headers or metadata)
        if line.isupper() and len(line) <= 8:
            continue
        # Reject weak/generic standalone tokens
        if _is_weak_indicator(line):
            continue
        # Reject narrative sentence fragments
        if _looks_narrative_indicator(line):
            continue
        candidates.append(line)
        if len(candidates) >= 200:
            break
    return candidates


def _is_generic_page_title(value: str) -> bool:
    """Verifie si le titre correspond a un titre de page generique."""
    title = " ".join(str(value or "").strip().lower().split())
    return title in _GENERIC_PAGE_TITLES


def _bbox_area(bbox_norm: list[float] | None) -> float:
    """Calcule l'aire normalisee d'une bounding box."""
    if not bbox_norm or len(bbox_norm) < 4:
        return 0.0
    try:
        left, top, right, bottom = [float(v) for v in bbox_norm[:4]]
    except (TypeError, ValueError):
        return 0.0
    if right <= left or bottom <= top:
        return 0.0
    return max(0.0, (right - left) * (bottom - top))


def _is_trivial_result(
    result: VisionFullResult | None,
    *,
    bbox_norm: list[float] | None = None,
) -> bool:
    """Determine si le resultat d'extraction est trivial ou vide."""
    if result is None:
        return True
    indicators = [str(v).strip() for v in list(result.indicators or []) if str(v).strip()]
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    summary = str(result.table_summary or "").strip()
    title = str(result.table_title or "").strip()
    if result.no_table_detected and not indicators and not headers:
        return True
    if not indicators and not headers:
        return True
    if indicators:
        return False
    if headers or summary:
        return False
    if _is_generic_page_title(title):
        return True
    return _bbox_area(bbox_norm) >= 0.12 or not title


def _has_extracted_data(result: VisionFullResult) -> bool:
    """True si le resultat contient au moins 1 indicateur ou 1 header non-vide.

    Sert de garde-fou pour eviter de classer en ``confirmed_no_table``
    des tables dont l'extraction a produit des donnees exploitables.
    """
    indicators = [s for s in (str(v).strip() for v in (result.indicators or [])) if s]
    headers = [s for s in (str(v).strip() for v in (result.headers or [])) if s]
    return bool(indicators or headers)


def _normalized_signal_text(value: str) -> str:
    """Normalise un texte signal en minuscules avec espaces simples."""
    return " ".join(str(value or "").strip().lower().split())


def _is_period_like_indicator(text: str) -> bool:
    """Verifie si l'indicateur ressemble a une date ou periode."""
    normalized = _normalized_signal_text(text)
    if not normalized:
        return True
    if re.fullmatch(r"(?:q|t)\s*[1-4](?:\s*\d{4})?", normalized):
        return True
    if re.fullmatch(r"\d{4}", normalized):
        return True
    return False


def _is_weak_indicator(text: str) -> bool:
    """Verifie si l'indicateur est un libelle faible ou generique."""
    normalized = _normalized_signal_text(text)
    if not normalized:
        return True
    if normalized in _WEAK_INDICATOR_EXACT:
        return True
    tokens = normalized.split()
    if 0 < len(tokens) <= 3 and all(token in _WEAK_INDICATOR_TOKENS for token in tokens):
        return True
    if _is_period_like_indicator(normalized):
        return True
    return False


def _looks_narrative_indicator(text: str) -> bool:
    """Verifie si l'indicateur ressemble a du texte narratif plutot qu'un libelle de tableau."""
    normalized = _normalized_signal_text(text)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _NARRATIVE_INDICATOR_PHRASES):
        return True
    if normalized.endswith("."):
        return True
    tokens = normalized.split()
    return len(tokens) >= 6 and any(
        phrase in normalized for phrase in (" la ", " le ", " les ", " des ", " du ", " au ", " aux ")
    )


def _structural_indicator_count(result: VisionFullResult | None) -> int:
    """Compter les lignes reelles, meme lorsqu'elles sont peu discriminantes.

    Des libelles comme ``Canada``, ``Autres`` ou ``Total`` sont faibles pour
    identifier un tableau entre deux trimestres, mais restent de vraies lignes
    et ne doivent pas faire passer une extraction non vide pour un artefact.
    """
    if result is None:
        return 0
    count = 0
    period_count = 0
    for raw in list(result.indicators or []):
        text = str(raw or "").strip()
        if not text:
            continue
        if _is_generic_page_title(text):
            continue
        if not any(char.isalpha() for char in text):
            continue
        if _is_period_like_indicator(text):
            period_count += 1
            continue
        if _looks_narrative_indicator(text):
            continue
        count += 1
    headers = [str(value or "").strip() for value in list(result.headers or []) if str(value or "").strip()]
    if period_count >= 2 and len(headers) >= 2:
        count += period_count
    return count


def _looks_compact_textual_header(text: str) -> bool:
    """Verifie si le texte ressemble a un en-tete de colonne textuel compact."""
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    words = normalized.split()
    if len(words) == 0 or len(words) > 4:
        return False
    if len(normalized) > 28:
        return False
    return any(char.isalpha() for char in normalized)


def _has_multi_textual_headers(result: VisionFullResult | None) -> bool:
    """Verifie si le resultat contient plusieurs en-tetes textuels."""
    if result is None:
        return False
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    if len(headers) < 3:
        return False
    return all(_looks_compact_textual_header(header) for header in headers[:3])


def _token_count(text: str) -> int:
    """Compte le nombre de tokens (mots) dans un texte."""
    return len(re.findall(r"\w+", str(text or ""), flags=re.UNICODE))


def _looks_like_right_column_bleed_indicator(text: str) -> bool:
    """Verifie si l'indicateur semble provenir d'un debordement de colonne droite."""
    normalized = _normalized_signal_text(text)
    if not normalized:
        return False
    token_count = _token_count(normalized)
    if token_count >= 10:
        return True
    if token_count >= 7 and any(char.isdigit() for char in normalized):
        return True
    if token_count >= 7 and "(" in text and ")" in text:
        return True
    if token_count >= 8 and normalized.count(" de ") + normalized.count(" du ") >= 2:
        return True
    return False


def _right_column_bleed_score(
    result: VisionFullResult | None,
    *,
    baseline_result: VisionFullResult | None = None,
) -> int:
    """Calcule un score de contamination par debordement de colonne droite."""
    if result is None or baseline_result is None:
        return 0
    if not _has_multi_textual_headers(result):
        return 0
    indicators = [str(v).strip() for v in list(result.indicators or []) if str(v).strip()]
    baseline_indicators = [str(v).strip() for v in list(baseline_result.indicators or []) if str(v).strip()]
    if len(indicators) <= len(baseline_indicators) or not baseline_indicators:
        return 0
    if indicators[: len(baseline_indicators)] != baseline_indicators:
        return 0
    tail = indicators[len(baseline_indicators) :]
    if len(tail) < 2:
        return 0
    suspicious_tail = [item for item in tail if _looks_like_right_column_bleed_indicator(item)]
    if not suspicious_tail:
        return 0
    baseline_avg_tokens = sum(_token_count(item) for item in baseline_indicators) / len(baseline_indicators)
    tail_avg_tokens = sum(_token_count(item) for item in tail) / len(tail)
    if tail_avg_tokens < baseline_avg_tokens + 2.5:
        return 0
    return len(suspicious_tail) + max(0, len(tail) - 1)


def _viable_indicator_count(result: VisionFullResult | None) -> int:
    """Compte les indicateurs viables (ni faibles, ni narratifs, ni generiques)."""
    if result is None:
        return 0
    count = 0
    for raw in list(result.indicators or []):
        text = str(raw or "").strip()
        if not text:
            continue
        if _is_generic_page_title(text):
            continue
        if not any(char.isalpha() for char in text):
            continue
        if _is_weak_indicator(text):
            continue
        if _looks_narrative_indicator(text):
            continue
        count += 1
    return count


def _weak_indicator_count(result: VisionFullResult | None) -> int:
    """Compte les indicateurs consideres comme faibles."""
    if result is None:
        return 0
    return sum(
        1
        for raw in list(result.indicators or [])
        if str(raw or "").strip() and _is_weak_indicator(str(raw or "").strip())
    )


def _narrative_indicator_count(result: VisionFullResult | None) -> int:
    """Compte les indicateurs de type narratif."""
    if result is None:
        return 0
    return sum(
        1
        for raw in list(result.indicators or [])
        if str(raw or "").strip() and _looks_narrative_indicator(str(raw or "").strip())
    )


def _contamination_score(result: VisionFullResult | None) -> int:
    """Calcule un score de contamination reel (titre generique et narration).

    Les indicateurs faibles restent peu discriminants pour le matching, mais
    ne constituent pas une contamination de l'extraction.
    """
    if result is None:
        return 99
    title = str(result.table_title or "").strip()
    score = 0
    if title and _is_generic_page_title(title):
        score += 2
    score += 2 * _narrative_indicator_count(result)
    return score


def _has_dominant_contamination(result: VisionFullResult | None) -> bool:
    """Verifie si la contamination domine les indicateurs viables."""
    if result is None:
        return True
    structural_count = _structural_indicator_count(result)
    score = _contamination_score(result)
    return structural_count <= 0 or score >= (structural_count + 1)


def _has_generic_title_without_support(result: VisionFullResult | None) -> bool:
    """Verifie si le resultat a un titre generique sans en-tetes ni notes de support."""
    if result is None:
        return True
    title = str(result.table_title or "").strip()
    if not title or not _is_generic_page_title(title):
        return False
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    footnotes = [
        item
        for item in list(result.footnotes_content or [])
        if isinstance(item, dict) and (str(item.get("id") or "").strip() or str(item.get("text") or "").strip())
    ]
    return len(headers) < 2 and not footnotes


def _has_strong_non_summary_signals(result: VisionFullResult | None) -> bool:
    """Verifie la presence de signaux forts (indicateurs, en-tetes, notes) hors resume."""
    if result is None:
        return False
    if _has_dominant_contamination(result):
        return False
    structural_indicators = _structural_indicator_count(result)
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    footnotes = [
        item
        for item in list(result.footnotes_content or [])
        if isinstance(item, dict) and (str(item.get("id") or "").strip() or str(item.get("text") or "").strip())
    ]
    title = str(result.table_title or "").strip()
    if structural_indicators >= 3:
        return True
    if structural_indicators >= 2 and (headers or footnotes or (title and not _is_generic_page_title(title))):
        return True
    if structural_indicators >= 1 and len(headers) >= 2 and bool(footnotes):
        return True
    return False


def _is_viable_result(result: VisionFullResult | None) -> bool:
    """Verifie si le resultat contient des lignes reelles et peu de contamination."""
    if result is None:
        return False
    if result.no_table_detected:
        return False
    return _structural_indicator_count(result) > 0 and not _has_dominant_contamination(result)


_DATE_RE = re.compile(
    r"^\s*(au\s+\d{1,2}\s+\w+\s+\d{4}|\d{1,2}\s+\w+\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|[tTqQ][1-4]\s*\d{4}|\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4})\s*$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"^\s*[\(\-]?[\d\s.,]+[\)%]?\s*$")


def _count_real_indicators(indicators: list[Any]) -> int:
    """Compte les indicateurs qui sont de vrais libelles metier (ni dates, ni nombres, ni vides)."""
    count = 0
    for ind in indicators:
        text = str(ind).strip() if isinstance(ind, str) else str(ind).strip()
        if not text:
            continue
        if _DATE_RE.match(text):
            continue
        if _NUMBER_RE.match(text):
            continue
        count += 1
    return count


_SUPERSCRIPT_FOOTNOTE_TRANSLATION = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹",
    "0123456789",
)


def _normalize_footnote_marker_id(value: Any) -> str:
    """Normalise un identifiant de note vers sa forme canonique sans decoration."""
    text = str(value or "").strip().translate(_SUPERSCRIPT_FOOTNOTE_TRANSLATION)
    while len(text) >= 2 and (text[0], text[-1]) in {
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
    }:
        text = text[1:-1].strip()
    return text.lower()


def _extract_footnote_marker_ids(values: list[Any]) -> set[str]:
    """Extrait les marqueurs visibles dans des titres, en-tetes ou indicateurs."""
    markers: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for match in re.finditer(r"[⁰¹²³⁴-⁹*†‡]+", text):
            marker = _normalize_footnote_marker_id(match.group(0))
            if marker:
                markers.add(marker)
        for match in re.finditer(r"\(\s*([0-9a-zA-Z]{1,2})\s*\)", text):
            marker = _normalize_footnote_marker_id(match.group(1))
            if marker:
                markers.add(marker)
    return markers
