"""Constantes et contrats partagés du pipeline d'analyse textuelle.

Ce module centralise la version du schéma, la taxonomie des sections, les
cibles propres aux banques ainsi que les expressions et seuils communs. Il ne
réalise aucune orchestration ni entrée-sortie.
"""

from __future__ import annotations

import re

UNIFIED_TEXT_SCHEMA_VERSION = 3


_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}


_CANONICAL_TO_TEXT_KEY: dict[str, str] = {
    "capital_management": "gestion_capital",
    "capital": "gestion_capital",
    "risk_management": "gestion_risques",
    "risk": "gestion_risques",
    "regulatory_updates": "gestion_reglementation",
    "regulatory": "gestion_reglementation",
}


_THEME_BY_SECTION: dict[str, str] = {
    "gestion_capital": "capital",
    "gestion_risques": "risque",
    "gestion_reglementation": "changement",
}


_TARGET_SECTIONS_BY_BANK: dict[str, set[str]] = {
    "bnc": {"gestion_capital", "gestion_risques"},
    "rbc": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
    "bns": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
    "scotia": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
    "cibc": {"gestion_capital", "gestion_risques"},
    "td": {"gestion_capital", "gestion_risques"},
    "bmo": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
}


_T4_TEXT_TARGET_SECTIONS = {"gestion_capital", "gestion_risques"}


_MODEL_MAX_OUTPUT_TOKENS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^gpt-4o(?:$|-)", flags=re.IGNORECASE), 16_384),
    (re.compile(r"^gpt-4\.1(?:$|-)", flags=re.IGNORECASE), 32_768),
    (re.compile(r"^gpt-4(?:$|-)", flags=re.IGNORECASE), 8_192),
]


_OPENAI_TIMEOUT_SECONDS = 300.0


_TRIAGE_BATCH_SIZE = 1


_TRIAGE_TRANSPORT_RETRIES = 2


_TRIAGE_SEMANTIC_TEXT_LIMIT = 1200


_TRIAGE_SOURCE_SNIPPET_LIMIT = 400


_TRIAGE_LENGTH_RETRIES = 1


_REGULATORY_REF_RE = re.compile(
    r"\b(?:OSFI|BSIF|Bâle|Basel|TLAC|LCR|NSFR|CET1|Tier\s*1|Tier\s*2|Pilier\s*[123]|IFRS|IAS|NIIF|BISM|VaR)\b",
    flags=re.IGNORECASE,
)


_NUMERIC_TOKEN_RE = re.compile(r"\b\S*\d\S*\b")


_ROMAN_NUMERAL_RE = re.compile(r"\b[IVX]{1,4}\b")


_PERCENT_RE = re.compile(r"[%‰]+")


_BPS_RE = re.compile(r"\b(?:pb|pbs|bp|bps|point(?:s)?\s+de\s+base)\b", flags=re.IGNORECASE)


_PUNCT_SPACING_RE = re.compile(r"\s+([,;:.])")


_MULTISPACE_RE = re.compile(r"\s+")


_TABLE_VALUE_RE = re.compile(r"\b\d+(?:[\s.,]\d+)*(?:\s*[%$])?\b")


_TABLE_ROW_MARKER_RE = re.compile(
    r"\b(?:tableau|table|total|totaux|s[ée]rie|series|moody's|s&p|fitch|dbrs|en millions de dollars|"
    r"en milliers de dollars|valeur pond[ée]r[ée]e|valeur non pond[ée]r[ée]e|aux 31|au 31)\b",
    flags=re.IGNORECASE,
)


_FOOTNOTE_MARKER_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:\(?\d+\)|\d+\)|[¹²³⁴⁵⁶⁷⁸⁹]+|\*{1,3}|[a-z]\))|(?:^|\n)\s*(?:note|source)\b)",
    flags=re.IGNORECASE,
)


_TABLE_HEADING_RE = re.compile(r"^\s*(?:tableau|table)\b", flags=re.IGNORECASE)


_SUBSECTION_SPLIT_RE = re.compile(r"^### (.+)$", re.MULTILINE)


_SEMANTIC_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bcadre de capacité totale d[’']absorption des pertes\b", flags=re.IGNORECASE),
        "un cadre renforcé d'absorption des pertes",
    ),
    (
        re.compile(r"\bligne directrice sur le levier\b", flags=re.IGNORECASE),
        "des exigences de levier",
    ),
    (
        re.compile(r"\bréformes de\s+[IVX]{1,4}\b", flags=re.IGNORECASE),
        "des réformes prudentielles",
    ),
    (
        re.compile(r"\bexigences?\s+réglementaires?\b", flags=re.IGNORECASE),
        "des exigences prudentielles",
    ),
    (
        re.compile(r"\bexigence\s+réglementaire\s+minimale\b", flags=re.IGNORECASE),
        "exigence minimale",
    ),
    (
        re.compile(r"\bBISM\b", flags=re.IGNORECASE),
        "les banques d'importance systémique",
    ),
    (
        re.compile(r"\bVaR\b", flags=re.IGNORECASE),
        "la mesure de risque de marché",
    ),
]


_OUT_OF_SCOPE_ACCOUNTING_HEADING_PATTERN_SOURCES = [
    r"consolidation\s+des\s+entit",
    r"instruments\s+financiers",
    r"transactions?\s+entre\s+parties\s+li",
    r"questions?\s+en\s+mati[eè]re\s+de\s+comptabilit",
    r"questions?\s+comptables?",
    r"m[eé]thodes?\s+(et\s+)?estimations?\s+comptables?",
    r"m[eé]thodes?\s+comptables?",
    r"normes?\s+et\s+m[eé]thodes?\s+comptables?",
    r"[eé]tats?\s+financiers?",
    r"notes?\s+aux\s+[eé]tats?\s+financiers?",
    r"convention\s+sur\s+les\s+comptes",
    r"jugements?,?\s+estimations?\s+et\s+hypoth",
    r"[eé]valuations?\s+de\s+la\s+juste\s+valeur",
    r"changements\s+comptables?",
    r"pr[eé]sentation\s+des\s+[eé]tats?\s+financiers?",
]


_OUT_OF_SCOPE_ACCOUNTING_HEADING_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in _OUT_OF_SCOPE_ACCOUNTING_HEADING_PATTERN_SOURCES
]
