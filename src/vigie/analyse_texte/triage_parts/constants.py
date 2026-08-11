"""Constantes et expressions regulieres du triage.

Les seuils de formatage ont une source unique dans
``triage_formatting_rules`` et sont reexportes ici pour compatibilite.
"""

from __future__ import annotations

import re

from vigie.analyse_texte.triage_formatting_rules import (  # noqa: F401 - re-export
    _BANK_NOISE_SEQUENCE_THRESHOLD,
    _BANK_OPERATION_RE,
    _CALENDAR_UPDATE_RE,
    _COSMETIC_SEQUENCE_THRESHOLD,
    _GOVERNANCE_SIGNAL_RE,
    _ISOLATED_DATE_RE,
    _METHODOLOGY_SIGNAL_RE,
    _NEW_REGULATORY_SIGNAL_RE,
    _PROCESS_SIGNAL_RE,
    _VOLATILE_TOKEN_RE,
    is_governance_protected_edit,
)

_MAX_TRIAGE_LLM_WORKERS = 6
_SEMANTIC_ALIGNMENT_DECISIONS = frozenset({"same_disclosure", "distinct_disclosures", "moved_text", "uncertain"})
_TRIAGE_DEDUP_EMBEDDING_THRESHOLD = 0.92
_TRIAGE_EMBEDDING_TRUNCATE_CHARS = 1800
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_COMPACT_THEME_CANDIDATE_LIMIT = 6
_COMPACT_COMPLETION_BASE_TOKENS = 350
_COMPACT_COMPLETION_TOKENS_PER_CHANGE = 320
_COMPACT_COMPLETION_MAX_TOKENS = 1200
_FULL_EVIDENCE_PACKET_LIMIT = 2400
# Must stay above the token equivalent of max_length=700 on factual_change /
# reason so structured completions never hit finish_reason=length.
_FULL_EVIDENCE_FACT_MAX_TOKENS = 500
_FULL_EVIDENCE_VERIFICATION_MAX_TOKENS = 500
_SEMANTIC_REASON_FIELDS = (
    "changement_constate",
    "signification_metier",
    "comparaison_interbanques",
    "limite_interpretation",
    "motif_non_pertinence",
)
_ANALYST_FIELD_END_RE = re.compile(r"[.!?]+[\u00bb\u201d\"')\]]*$")
_CALENDAR_SUBJECT_RE = re.compile(
    r"(?:"
    r"coefficient\s+de\s+plancher|plancher\s+des?\s+fonds\s+propres|"
    r"entrée\s+en\s+vigueur|report\s+des?\s+exigences|"
    r"calendrier\s+d['’]application|jusqu['’]à\s+nouvel\s+ordre"
    r")",
    flags=re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_THEME_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", flags=re.IGNORECASE)
_THEME_STOPWORDS = frozenset(
    {
        "ajout",
        "changement",
        "dans",
        "des",
        "dune",
        "dun",
        "est",
        "les",
        "lié",
        "liée",
        "modification",
        "nouvelle",
        "pour",
        "rapport",
        "retrait",
        "risque",
        "une",
    }
)
