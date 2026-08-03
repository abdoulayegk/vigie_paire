"""Orchestration des differences pour une paire de tableaux."""

from __future__ import annotations

from typing import Any, Callable

from vigilance.differences_tableaux.comparaison_llm import (
    diff_footnotes_pair_gpt,
    diff_indicators_pair_gpt,
)
from vigilance.differences_tableaux.filtrage_artefacts import (
    _inspect_diff_artifacts_gpt,
)
from vigilance.differences_tableaux.normalisation_elements import _normalize_footnotes


def _compose_pair_reason(
    *,
    indicator_reason: str,
    footnote_reason: str,
    has_indicator_changes: bool,
    has_footnote_changes: bool,
) -> str:
    """Compose la raison globale d'une paire a partir des raisons indicateurs et notes."""
    indicator_reason = str(indicator_reason or "").strip()
    footnote_reason = str(footnote_reason or "").strip()
    if has_indicator_changes and has_footnote_changes:
        parts = [part for part in (indicator_reason, footnote_reason) if part]
        if parts:
            if len(parts) == 2 and parts[0] == parts[1]:
                return parts[0]
            return " ".join(parts)
        return "Des changements sémantiques ont été détectés sur les indicateurs et les notes de bas de page."
    if has_indicator_changes:
        return indicator_reason or "Des changements sémantiques ont été détectés sur les indicateurs."
    if has_footnote_changes:
        return footnote_reason or "Des changements sémantiques ont été détectés sur les notes de bas de page."
    return indicator_reason or footnote_reason or "Aucun changement sémantique détecté."


def diff_table_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    """Orchestre le diff complet (indicateurs + notes + inspection) d'une paire de tableaux.

    Enchaine le diff d'indicateurs, le diff de notes de bas de page et
    l'inspection des artefacts pour produire le ``technical_diff`` final.

    Args:
        previous_table: Tableau du trimestre precedent.
        current_table: Tableau du trimestre courant.
        model: Identifiant du modele OpenAI.
        call_openai_json: Fonction injectee pour l'appel OpenAI.
        usage_recorder: Accumulateur optionnel de metriques d'utilisation.
        max_validation_attempts: Nombre maximal de tentatives de validation.

    Returns:
        Dictionnaire contenant ``technical_diff``, ``reason``, ``diff_mode``
        et ``diff_calls_total``.
    """
    indicator_diff = diff_indicators_pair_gpt(
        previous_table,
        current_table,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )
    footnote_diff = diff_footnotes_pair_gpt(
        previous_table,
        current_table,
        indicator_diff=indicator_diff,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )

    # --- Post-diff GPT Inspector (artifact filter) ---
    pre_inspect_adds = list(indicator_diff.get("indicators_added", []) or [])
    pre_inspect_removes = list(indicator_diff.get("indicators_removed", []) or [])
    pre_inspect_renames = list(indicator_diff.get("indicators_renamed", []) or [])
    inspector_called = bool(pre_inspect_adds or pre_inspect_removes or pre_inspect_renames)
    indicator_diff = _inspect_diff_artifacts_gpt(
        indicator_diff,
        previous_table,
        current_table,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )

    previous_footnotes = _normalize_footnotes(previous_table.get("footnotes", []))
    current_footnotes = _normalize_footnotes(current_table.get("footnotes", []))
    footnote_gpt_called = bool(previous_footnotes and current_footnotes)

    technical_diff: dict[str, Any] = {
        "indicators_added": indicator_diff["indicators_added"],
        "indicators_removed": indicator_diff["indicators_removed"],
        "indicators_renamed": indicator_diff["indicators_renamed"],
        "footnotes_added": footnote_diff["footnotes_added"],
        "footnotes_removed": footnote_diff["footnotes_removed"],
        "footnotes_renamed": footnote_diff["footnotes_renamed"],
    }
    has_changes = any(technical_diff.values())
    technical_diff["table_level_change"] = "modifie" if has_changes else "inchange"
    reason = _compose_pair_reason(
        indicator_reason=str(indicator_diff.get("reason", "") or ""),
        footnote_reason=str(footnote_diff.get("reason", "") or ""),
        has_indicator_changes=any(
            technical_diff[field]
            for field in (
                "indicators_added",
                "indicators_removed",
                "indicators_renamed",
            )
        ),
        has_footnote_changes=any(
            technical_diff[field]
            for field in (
                "footnotes_added",
                "footnotes_removed",
                "footnotes_renamed",
            )
        ),
    )
    return {
        "technical_diff": technical_diff,
        "reason": reason,
        "diff_mode": "gpt",
        "diff_calls_total": ((2 if footnote_gpt_called else 1) + (1 if inspector_called else 0)),
    }
