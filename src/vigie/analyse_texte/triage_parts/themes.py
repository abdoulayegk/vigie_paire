"""Derivation et normalisation des themes AMF associes a un changement.

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

from vigie.comparaison.triage.amf_taxonomy import THEMES_AMF_ANALYST_SUBJECTS, THEMES_AMF_DESCRIPTIONS

from .constants import (
    _COMPACT_THEME_CANDIDATE_LIMIT,
    _THEME_STOPWORDS,
    _THEME_TOKEN_RE,
    _WHITESPACE_RE,
)

logger = logging.getLogger("vigie.analyse_texte.triage")

def _normalize_for_cosmetic(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _theme_tokens(value: str) -> set[str]:
    normalized = _normalize_for_cosmetic(value)
    return {
        token
        for token in _THEME_TOKEN_RE.findall(normalized)
        if len(token) >= 3 and token not in _THEME_STOPWORDS
    }


def _candidate_themes_for_change(
    change: dict[str, Any],
    *,
    section_key: str,
    limit: int = _COMPACT_THEME_CANDIDATE_LIMIT,
) -> list[dict[str, str]]:
    """Sélectionne localement une courte liste de thèmes AMF plausibles."""
    corpus = " ".join(
        str(value or "")
        for value in (
            change.get("change_summary"),
            change.get("source_text_t1"),
            change.get("source_text_t2"),
            change.get("semantic_text_t1"),
            change.get("semantic_text_t2"),
            change.get("subsection_heading"),
            section_key,
        )
    )
    corpus_tokens = _theme_tokens(corpus)
    scored: list[tuple[float, str]] = []
    for code, description in THEMES_AMF_DESCRIPTIONS.items():
        theme_text = f"{THEMES_AMF_ANALYST_SUBJECTS.get(code, '')} {description}"
        theme_tokens = _theme_tokens(theme_text)
        overlap = len(corpus_tokens & theme_tokens)
        coverage = overlap / max(len(theme_tokens), 1)
        score = float(overlap) + coverage
        if overlap:
            scored.append((score, code))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    diff_type = str(change.get("diff_type") or "").lower()
    forced = {
        "added": "DIVULGATION_AJOUT",
        "removed": "DIVULGATION_RETRAIT",
        "renamed": "STRUCTURE_RAPPORT",
    }.get(diff_type)
    if forced:
        selected.append(forced)

    for _, code in scored:
        if code not in selected:
            selected.append(code)
        if len(selected) >= max(limit - 1, 1):
            break

    if len(selected) < max(limit - 1, 1):
        section_fallbacks = {
            "gestion_capital": (
                "CAPITAL_REGLEMENTAIRE",
                "FONDS_PROPRES_REGLEMENTAIRES",
                "RATIOS_REGLEMENTAIRES",
                "EXIGENCES_REGLEMENTAIRES",
            ),
            "gestion_reglementation": (
                "NOUVELLE_MENTION_REGLEMENTAIRE",
                "EXIGENCES_REGLEMENTAIRES",
                "CONTROLE_CONFORMITE",
            ),
            "gestion_risques": (
                "MODIFICATION_TEXTE_RISQUE",
                "FACTEUR_RISQUE_CHANGEMENT",
                "GOUVERNANCE_RISQUES",
                "RISQUE_EMERGENT",
            ),
        }
        for code in section_fallbacks.get(section_key, ()):
            if code not in selected:
                selected.append(code)
            if len(selected) >= max(limit - 1, 1):
                break

    if "SUJET_EMERGENT_HORS_GRILLE" not in selected:
        selected.append("SUJET_EMERGENT_HORS_GRILLE")
    selected = selected[:limit]
    return [
        {
            "code": code,
            "label": THEMES_AMF_ANALYST_SUBJECTS[code],
            "description": THEMES_AMF_DESCRIPTIONS[code],
        }
        for code in selected
    ]


def _normalize_themes_amf(themes: list[str]) -> list[str]:
    """Accepte tout code de la taxonomie AMF ; remap les inconnus vers hors grille."""
    allowed = set(THEMES_AMF_DESCRIPTIONS)
    normalized: list[str] = []
    for theme in themes:
        code = str(theme or "").strip().upper()
        if not code:
            continue
        if code in allowed:
            if code not in normalized:
                normalized.append(code)
        elif "SUJET_EMERGENT_HORS_GRILLE" not in normalized:
            normalized.append("SUJET_EMERGENT_HORS_GRILLE")
            logger.debug(
                "theme_clamped unknown=%s -> SUJET_EMERGENT_HORS_GRILLE",
                code,
            )
    return normalized
