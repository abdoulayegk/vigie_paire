"""Règles de mise en forme et filtres de reformulations non-substantives pour le triage.

Ce module isole la logique de détection des changements purement typographiques,
des ajustements de dates et des filtres d'immunité pour la gouvernance et la réglementation.
"""

from __future__ import annotations

import re

_COSMETIC_SEQUENCE_THRESHOLD = 0.985
_BANK_NOISE_SEQUENCE_THRESHOLD = 0.92

_ISOLATED_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
    flags=re.IGNORECASE,
)
_VOLATILE_TOKEN_RE = re.compile(
    r"(?:"
    r"\b\d{1,2}\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+\d{4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:t|q)\s*[1-4]\s*[\-/–]?\s*\d{2,4}\b|"
    r"\bexercice\s+\d{4}\b|"
    r"\btrimestre\s+(?:de\s+)?\d{4}\b|"
    r"\b\d{4}\b|"
    r"\d[\d\s\u00a0.,]*\s*(?:%|m\$|g\$|mds?|millions?|milliards?)?\b"
    r")",
    flags=re.IGNORECASE,
)
_BANK_OPERATION_RE = re.compile(
    r"\b(?:"
    r"acquisition|acquérir|rachet|rachat|émission|émettre|dividende|"
    r"fusion|achat\s+d['’]actions|billets?\s+à\s+moyen\s+terme|"
    r"cwb|canadian\s+western\s+bank|transaction\s+d['’]entreprise|"
    r"offre\s+publique\s+d['’]achat|opa\b|spin[- ]?off"
    r")\b",
    flags=re.IGNORECASE,
)
_CALENDAR_UPDATE_RE = re.compile(
    r"\b(?:"
    r"jusqu['’]à\s+nouvel\s+ordre|report(?:é|er|ait)?|report|"
    r"calendrier|échéanc|exercice\s+\d{4}|à\s+compter|"
    r"progressiv|coefficient\s+de\s+plancher|plancher\s+de\s+fonds"
    r")\b",
    flags=re.IGNORECASE,
)
_METHODOLOGY_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"méthodolog|trimestriellement|périodiquement|mensuellement|"
    r"approche\s+standard|approche\s+interne|airb|modèle\s+interne|"
    r"sensibilités\s+standard|calcul(?:é|er)?\s+selon"
    r")\b",
    flags=re.IGNORECASE,
)
_PROCESS_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"processus|proc[ée]dure|flux\s+de\s+travail|workflow|"
    r"cha[iî]ne\s+de\s+traitement|mode\s+op[ée]ratoire"
    r")\b",
    flags=re.IGNORECASE,
)
_NEW_REGULATORY_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"b-15|ligne\s+directrice|tlac|bâle\s+iii|nouvelle\s+exigence|"
    r"entrée\s+en\s+vigueur|exigence\s+additionnelle|"
    r"cadre\s+réglementaire|avis\s+du\s+bsif"
    r")\b",
    flags=re.IGNORECASE,
)
_GOVERNANCE_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"gouvernance|comit[ée]s?|conseil\s+d['’]administration|mandat|"
    r"lignes?\s+de\s+d[ée]fense|responsabilit[ée]s?|supervision|"
    r"reddition\s+de\s+comptes|escalade|autorit[ée]\s+d[ée]cisionnelle|"
    r"droits?\s+d['’]approbation|culture\s+de\s+risque|"
    r"r[ée]mun[ée]ration|app[ée]tit\s+(?:pour\s+le|au)\s+risque|"
    r"imp[ôo]t|fiscalit[ée]|d[ée]sinterm[ée]diation|donn[ée]es?|technologies?|"
    r"cyber|blanchiment|sanctions?|bsif|b[âa]le|tarifs?|commerciale?|"
    r"cryptos?|climatique?|environnemental|mod[èe]les?"
    r")\b",
    flags=re.IGNORECASE,
)


def is_governance_protected_edit(text: str) -> bool:
    """Vérifie si un texte contient un signal réglementaire ou de gouvernance protégé."""
    if not text:
        return False
    return bool(_GOVERNANCE_SIGNAL_RE.search(text) or _NEW_REGULATORY_SIGNAL_RE.search(text))
