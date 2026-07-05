"""Constantes, regex et mappings du pipeline texte."""

from __future__ import annotations

import re
from typing import Any

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
_STRONG_AMF_THEMES_FOR_MODERE_RETENTION: frozenset[str] = frozenset(
    {
        "MODIFICATION_METHODOLOGIE",
        "NOUVELLE_MENTION_REGLEMENTAIRE",
        "EXIGENCES_REGLEMENTAIRES",
        "FACTEUR_RISQUE_CHANGEMENT",
        "RISQUE_EMERGENT",
        "RISQUE_DONNEES",
        "RISQUE_TIERS_CLOUD",
        "RISQUE_MACRO_GEOPOLITIQUE",
    }
)

_MODEL_MAX_OUTPUT_TOKENS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^gpt-4o(?:$|-)", flags=re.IGNORECASE), 16_384),
    (re.compile(r"^gpt-4\.1(?:$|-)", flags=re.IGNORECASE), 32_768),
    (re.compile(r"^gpt-4(?:$|-)", flags=re.IGNORECASE), 8_192),
]
_OPENAI_TIMEOUT_SECONDS = 300.0
_COMPARE_BATCH_SIZE = 8
_COMPARE_BATCH_TEXT_CHAR_LIMIT = 1_600
_TRIAGE_BATCH_SIZE = 8
_TRIAGE_TRANSPORT_RETRIES = 2
_TRIAGE_SEMANTIC_TEXT_LIMIT = 1200
_TRIAGE_SOURCE_SNIPPET_LIMIT = 400
_TRIAGE_LENGTH_RETRIES = 1
_NARRATIVE_UNIT_MIN_CHARS = 120
_NARRATIVE_UNIT_TARGET_MIN_CHARS = 400
_NARRATIVE_UNIT_TARGET_MAX_CHARS = 900
_NARRATIVE_UNIT_LONG_CHARS = 1100
_NARRATIVE_UNIT_LONG_WORDS = 160

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
_PAGE_MARKER_RE = re.compile(r"^\[p\.(\d+)(?:\s*\|\s*pdf\.(\d+))?\]\s*$", flags=re.MULTILINE)
_SUBSECTION_SPLIT_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•‣▪]|\(?\d+\)|[a-z]\))\s+", flags=re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-ZÀÂÇÉÈÊËÎÏÔÙÛÜŸ]|L[’']|D[’']|Il\b|Elle\b|La\b|Le\b|Les\b|Une?\b))"
)
_MATCH_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MATCH_STOPWORDS = {
    "afin",
    "ainsi",
    "alors",
    "avec",
    "aux",
    "avoir",
    "ces",
    "cet",
    "cette",
    "comme",
    "dans",
    "des",
    "donc",
    "dont",
    "elle",
    "elles",
    "entre",
    "est",
    "etre",
    "font",
    "hors",
    "ils",
    "les",
    "leur",
    "leurs",
    "mais",
    "meme",
    "notamment",
    "nous",
    "par",
    "pas",
    "plus",
    "pour",
    "que",
    "qui",
    "sans",
    "selon",
    "ses",
    "son",
    "sont",
    "sous",
    "sur",
    "tous",
    "tout",
    "une",
    "vers",
}
_VIGIE_OBJECTIVES: tuple[dict[str, Any], ...] = (
    {
        "tag": "appetit_risque",
        "label": "Appétit pour le risque",
        "objective": "Vérifier les éléments divulgués dans la section appétit pour le risque.",
        "keywords": (
            "appétit pour le risque",
            "appetit pour le risque",
            "limite de risque",
            "limites de risque",
            "risk appetite",
        ),
    },
    {
        "tag": "risques_esg",
        "label": "Risques ESG",
        "objective": "Vérifier ce qui est inclus dans la section Risques ESG.",
        "keywords": ("esg", "environnemental", "social", "durabilité", "durabilite"),
    },
    {
        "tag": "edtf_ifi",
        "label": "EDTF et importance systémique",
        "objective": "Vérifier les textes des pairs par rapport à EDTF et aux banques d'importance systémique (IFI).",
        "keywords": (
            "edtf",
            "importance systémique",
            "importance systemique",
            "ifi",
            "bisn",
            "bism",
            "banque d'importance systémique nationale",
            "banque d importance systemique nationale",
            "banques d'importance systémique nationale",
            "banques d importance systemique nationale",
            "g-sib",
            "d-sib",
        ),
    },
    {
        "tag": "ro_calcul_fonds_propres",
        "label": "RO (calcul des fonds propres)",
        "objective": "Vérifier le niveau de détails que donnent les pairs quant au calcul.",
        "keywords": (
            "fonds propres",
            "capital réglementaire",
            "capital reglementaire",
            "actifs pondérés",
            "actifs ponderes",
            "approche ni",
            "approche standard",
            "calcul des fonds propres",
            "calcul du capital",
            "ratio cet1",
            "tlac",
        ),
    },
    {
        "tag": "climat_credit",
        "label": "Impact risque climatique sur risque de crédit",
        "objective": "Vérifier si les pairs abordent le sujet des impacts climatiques sur le risque de crédit.",
        "keywords": ("climatique", "climat", "risque de crédit", "risque de credit", "b-15", "transition climatique"),
    },
    {
        "tag": "ia",
        "label": "IA",
        "objective": "Vérifier le contenu du texte sur l'IA.",
        "keywords": ("intelligence artificielle", " ia ", "ia générative", "ia generative", "ai ", "générative"),
    },
    {
        "tag": "politique_monetaire",
        "label": "Politique monétaire",
        "objective": (
            "Vérifier si les politiques monétaires sont traitées dans une section dédiée "
            "ou intégrées dans une autre section."
        ),
        "keywords": ("politique monétaire", "politique monetaire", "politiques monétaires", "taux d'intérêt"),
    },
    {
        "tag": "endettement_menages",
        "label": "Endettement des ménages",
        "objective": "Vérifier que les pairs abordent le sujet et le contenu du texte.",
        "keywords": ("endettement des ménages", "endettement des menages", "ménages", "menages", "consommateurs"),
    },
    {
        "tag": "capacite_recruter",
        "label": "Capacité à recruter",
        "objective": "Vérifier que les pairs abordent le sujet et le contenu du texte.",
        "keywords": ("recruter", "recrutement", "rétention", "retention", "talents", "main-d'œuvre", "main d'oeuvre"),
    },
)
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
