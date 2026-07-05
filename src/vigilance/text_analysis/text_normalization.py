"""Normalisation textuelle reutilisable pour comparaison et matching."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any


logger = logging.getLogger(__name__)

from .constants import (
    _BPS_RE,
    _MATCH_STOPWORDS,
    _MATCH_TOKEN_RE,
    _MULTISPACE_RE,
    _NUMERIC_TOKEN_RE,
    _PERCENT_RE,
    _PUNCT_SPACING_RE,
    _REGULATORY_REF_RE,
    _ROMAN_NUMERAL_RE,
    _SEMANTIC_REPLACEMENTS,
)

def _json_dumps(data: Any) -> str:
    """Sérialise ``data`` en JSON indenté avec support complet des caractères UTF-8."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def _sanitize_semantic_text(text: str) -> str:
    """Normalise un texte pour la comparaison sémantique inter-trimestrielle.

    Supprime les éléments non sémantiques — chiffres, pourcentages, points de base,
    références réglementaires, numéros romains — afin que deux paragraphes exprimant
    la même idée avec des valeurs différentes soient reconnus comme identiques.
    Utilisée pour peupler ``semantic_text_t1`` / ``semantic_text_t2`` dans les changements.
    """
    value = (text or "").strip()
    if not value:
        return ""
    for pattern, replacement in _SEMANTIC_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    value = _REGULATORY_REF_RE.sub("", value)
    value = _NUMERIC_TOKEN_RE.sub("", value)
    value = _ROMAN_NUMERAL_RE.sub("", value)
    value = _PERCENT_RE.sub("", value)
    value = _BPS_RE.sub("", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\([^)]*\d[^)]*\)", "", value)
    value = re.sub(r"\s*[-–—]\s*", " ", value)
    value = re.sub(r"\b(?:Le|La|Les)\s+a\b", "La banque a", value)
    value = re.sub(r"\bLa Banque\b", "La banque", value)
    value = re.sub(r"\bLe Groupe\b", "La banque", value)
    value = re.sub(r"\bConseil d'administration\b", "gouvernance", value, flags=re.IGNORECASE)
    value = _PUNCT_SPACING_RE.sub(r"\1", value)
    value = _MULTISPACE_RE.sub(" ", value).strip(" ,;:.")
    return value.strip()


def _normalized_block_text(text: str) -> str:
    """Normalise un texte pour les comparaisons de correspondance (matching).

    Passe en minuscules, réduit les espaces multiples, retire la ponctuation et
    les caractères spéciaux. Conserve lettres accentuées et chiffres.
    Utilisée pour détecter les doublons, les en-têtes déjà vus et pour
    retrouver la page exacte d'un fragment GPT dans les blocs PDF.
    """
    value = (text or "").lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9 ]+", "", value)
    return value.strip()


def _sanitize_explanation(text: str) -> str:
    """Nettoie et tronque une explication GPT à 1 200 caractères maximum."""
    value = _sanitize_semantic_text(text)
    return value[:1200]


def _normalize_heading(heading: str) -> str:
    """Normalise un heading ### pour le pairing T1/T2 (insensible à la casse et aux préfixes de tableaux)."""
    h = heading.lower()
    h = re.sub(r"\b[tT]\d{2,3}\b\s*", "", h)  # strip T22, T25, T125, etc.
    h = re.sub(r"[^\w\s]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _normalize_match_text(text: str) -> str:
    """Normalise agressivement un texte pour les heuristiques d'alignement."""
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _matching_tokens(text: str) -> set[str]:
    """Retourne les tokens informatifs utilisés pour le matching local."""
    normalized = _normalize_match_text(text)
    return {tok for tok in _MATCH_TOKEN_RE.findall(normalized) if len(tok) > 2 and tok not in _MATCH_STOPWORDS}


def _word_count(text: str) -> int:
    """Compte approximativement les mots d'une unité narrative."""
    return len(re.findall(r"\b\w+\b", text or ""))


def _jaccard(left: set[str], right: set[str]) -> float:
    """Similarité Jaccard robuste aux ensembles vides."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clamp_confidence(value: float) -> float:
    """Garde les scores d'alignement dans l'intervalle lisible 0..1."""
    return max(0.0, min(1.0, float(value)))
