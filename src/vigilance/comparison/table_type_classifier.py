"""
Classification du type de tableau (requirements vs observed_results vs unknown).

Utilise pour le Hard Negative: rejeter les matchs entre tableaux de types differents
(ex: Capital reglementaire valeurs vs Exigences reglementaires).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from vigilance.utils.matching_normalizer import normalize_for_matching

# Mots-cles pour "requirements" (exigences reglementaires)
REQUIREMENTS_KEYWORDS = frozenset(
    {
        "exigence",
        "exigences",
        "minimum",
        "reserve",
        "reserves",
        "surcharge",
        "surcharges",
        "bsif",
        "ccbc",
        "buffer",
        "buffers",
        "stabilite",
        "stability",
        "tlac minimum",
        "requirements",
        "requirement",
        "regulatory",
        "capital conservation",
        "conservation",
    }
)

# Mots-cles pour "observed_results" (valeurs observees)
OBSERVED_RESULTS_KEYWORDS = frozenset(
    {
        "au 31",
        "as at",
        "en millions",
        "in millions",
        "en milliards",
        "in billions",
        "fonds propres",
        "equity",
        "exposition",
        "exposure",
        "actif pondere",
        "risk weighted",
        "actifs ponderes",
        "valeur",
        "value",
        "montant",
        "amount",
        "ratio",
        "ratios",
        "disponible",
        "available",
        "2024",
        "2025",
        "millions",
        "milliards",
    }
)

# Patterns pour dates (au 31 janvier, etc.)
_DATE_PATTERN = re.compile(r"au\s+31|as\s+at\s+\d|31\s+(?:jan|fev|mar|avr|mai|jun|jul|aou|sep|oct|nov|dec)", re.IGNORECASE)


@dataclass
class TableTypeResult:
    """Resultat de la classification."""

    table_type: str  # "requirements" | "observed_results" | "unknown"
    keywords_detected: list[str]
    source: str  # "title" | "headers" | "context"


def _text_to_tokens(text: str) -> set[str]:
    """Convertit un texte en ensemble de tokens normalises."""
    if not text:
        return set()
    normalized = normalize_for_matching(text, target="indicator")
    return set(normalized.split())


def _find_keywords_in_text(text: str, keywords: frozenset[str]) -> list[str]:
    """Trouve les mots-cles presents dans le texte (normalise)."""
    if not text:
        return []
    tokens = _text_to_tokens(text)
    lower_text = text.lower()
    found = []
    for kw in keywords:
        if kw in tokens:
            found.append(kw)
        elif " " in kw and kw in lower_text:
            found.append(kw)
    return list(set(found))


def classify_table_type(
    title: Optional[str] = None,
    headers: Optional[list[str]] = None,
    context_before: Optional[str] = None,
    context_after: Optional[str] = None,
    bank_code: Optional[str] = None,
) -> TableTypeResult:
    """
    Classifie un tableau en requirements, observed_results ou unknown.

    Args:
        title: Titre du tableau
        headers: En-tetes de colonnes
        context_before: 1-2 lignes au-dessus du tableau
        context_after: 1-2 lignes en-dessous
        bank_code: Code banque (pour surcharges futures)

    Returns:
        TableTypeResult avec table_type et keywords_detected
    """
    combined = " ".join(
        filter(
            None,
            [
                title or "",
                " ".join(headers or []),
                context_before or "",
                context_after or "",
            ]
        )
    )
    if not combined.strip():
        return TableTypeResult(
            table_type="unknown",
            keywords_detected=[],
            source="empty",
        )

    combined_lower = combined.lower()

    req_found = _find_keywords_in_text(combined, REQUIREMENTS_KEYWORDS)
    obs_found = _find_keywords_in_text(combined, OBSERVED_RESULTS_KEYWORDS)

    if _DATE_PATTERN.search(combined_lower):
        obs_found.append("date_pattern")

    if req_found and not obs_found:
        return TableTypeResult(
            table_type="requirements",
            keywords_detected=req_found,
            source="title" if title else "headers" if headers else "context",
        )
    if obs_found and not req_found:
        return TableTypeResult(
            table_type="observed_results",
            keywords_detected=obs_found,
            source="title" if title else "headers" if headers else "context",
        )
    if req_found and obs_found:
        req_count = len(req_found)
        obs_count = len(obs_found)
        if req_count >= 2 and obs_count < req_count:
            return TableTypeResult(
                table_type="requirements",
                keywords_detected=req_found,
                source="title" if title else "headers" if headers else "context",
            )
        if obs_count >= 2 and req_count < obs_count:
            return TableTypeResult(
                table_type="observed_results",
                keywords_detected=obs_found,
                source="title" if title else "headers" if headers else "context",
            )

    return TableTypeResult(
        table_type="unknown",
        keywords_detected=[],
        source="ambiguous",
    )
