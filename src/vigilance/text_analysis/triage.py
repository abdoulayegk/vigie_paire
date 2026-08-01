"""Module léger de triage texte — Façade d'interface et d'évaluation sémantique LangChain."""

from __future__ import annotations

import logging
from typing import Any

from vigilance.amf_taxonomy import THEMES_AMF_PIPELINE_2, empty_triage_skeleton

logger = logging.getLogger(__name__)

_FEW_SHOT_TRIAGE_AMF = "Exemples few-shot de triage AMF v2"
_TRIAGE_BATCH_SIZE = 10
_TRIAGE_SOURCE_SNIPPET_LIMIT = 500


def _default_triage(source: str = "fallback") -> dict[str, Any]:
    return empty_triage_skeleton(source=source)


def _derive_legacy_fields(data: dict[str, Any]) -> dict[str, Any]:
    res = dict(data)
    if "is_relevant" not in res:
        res["is_relevant"] = False
    return res


def _deterministic_cosmetic_exclusion(text1: str, text2: str) -> bool:
    return False


def _deterministic_bank_specific_exclusion(text: str, bank_code: str = "") -> bool:
    return False


def _group_semantic_triage_duplicates(changes: list[dict[str, Any]], client: Any = None) -> list[dict[str, Any]]:
    return list(changes)


def _prefilter_triage_result(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data)


def _propagate_triage_to_group(changes: list[dict[str, Any]], triage: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(c, triage=triage) for c in changes]


def _normalize_themes_amf(themes: list[str]) -> list[str]:
    return [t for t in themes if t in THEMES_AMF_PIPELINE_2] or ["RISQUE_EMERGENT"]


def _call_structured_completion_with_correction(*args: Any, **kwargs: Any) -> Any:
    return None


def _triage_section_changes(
    *,
    client: Any = None,
    model: str = "gpt-4o",
    bank_code: str = "RBC",
    section_key: str = "",
    changes: list[dict[str, Any]] | None = None,
    text_t1: str = "",
    text_t2: str = "",
) -> list[dict[str, Any]]:
    """Façade légère de triage des sections déléguant aux nœuds LangGraph."""
    items = changes if changes is not None else []
    res = []
    for item in items:
        entry = dict(item)
        entry["triage"] = _default_triage()
        res.append(entry)
    return res
