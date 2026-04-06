"""Assemblage du dict text_comparison.json à partir des résultats de Pass 2 + Pass 3.

Construit la structure complète text_comparison.json, incluant section_comparisons,
block_comparisons avec résolution des block_id T1/T2, et le global_summary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

TEXT_COMPARISON_SCHEMA_VERSION = 1

_SECTION_DISPLAY: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}


# ---------------------------------------------------------------------------
# Block resolution helpers
# ---------------------------------------------------------------------------


def _find_block_by_text(
    text: str,
    blocks: list[dict[str, Any]],
    threshold: float = 0.85,
) -> dict[str, Any] | None:
    """Trouve le bloc dont le texte correspond le mieux (difflib ratio).

    Retourne le bloc avec le meilleur ratio si ≥ threshold, sinon None.
    """
    if not text or not blocks:
        return None

    try:
        from difflib import SequenceMatcher
    except ImportError:
        return None

    best_ratio = 0.0
    best_block: dict[str, Any] | None = None

    for block in blocks:
        block_text = str(block.get("text", ""))
        ratio = SequenceMatcher(None, text[:500], block_text[:500]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_block = block

    if best_ratio >= threshold:
        return best_block
    return None


def _resolve_block_refs(
    change: dict[str, Any],
    blocks_t1_by_section: dict[str, list[dict[str, Any]]],
    blocks_t2_by_section: dict[str, list[dict[str, Any]]],
    section_key: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Résout les block_id T1 et T2 pour un changement en cherchant par texte."""
    blocks_t1 = blocks_t1_by_section.get(section_key, [])
    blocks_t2 = blocks_t2_by_section.get(section_key, [])

    text_t1 = change.get("text_t1", "")
    text_t2 = change.get("text_t2", "")

    block_t1 = _find_block_by_text(text_t1, blocks_t1) if text_t1 else None
    block_t2 = _find_block_by_text(text_t2, blocks_t2) if text_t2 else None

    return block_t1, block_t2


# ---------------------------------------------------------------------------
# Section summary builder
# ---------------------------------------------------------------------------


def _compute_section_summary(changes: list[dict[str, Any]]) -> dict[str, int]:
    """Calcule les counts added/removed/modified/unchanged pour une section."""
    counts: dict[str, int] = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
    for change in changes:
        ct = change.get("change_type", "")
        if ct in counts:
            counts[ct] += 1
    return counts


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_text_comparison(
    bank_code: str,
    quarter_t1: str,
    quarter_t2: str,
    year_t1: int,
    year_t2: int,
    extraction_t1: dict[str, Any],
    extraction_t2: dict[str, Any],
    triage_result: dict[str, Any],
) -> dict[str, Any]:
    """Construit le dictionnaire complet text_comparison.json.

    Args:
        bank_code: Code banque (ex. "bns").
        quarter_t1: Trimestre précédent normalisé (ex. "t1").
        quarter_t2: Trimestre courant normalisé (ex. "t2").
        year_t1: Année du trimestre précédent.
        year_t2: Année du trimestre courant.
        extraction_t1: Contenu de text_extraction.json T1.
        extraction_t2: Contenu de text_extraction.json T2.
        triage_result: Résultat de run_text_triage() contenant
            section_diffs_enriched et global_summary.

    Returns:
        Dictionnaire prêt à être sérialisé en text_comparison.json.
    """
    # Index blocks by section for block_id resolution
    blocks_t1_by_section: dict[str, list[dict[str, Any]]] = {}
    for block in extraction_t1.get("blocks", []):
        sec = str(block.get("section", ""))
        if sec:
            blocks_t1_by_section.setdefault(sec, []).append(block)

    blocks_t2_by_section: dict[str, list[dict[str, Any]]] = {}
    for block in extraction_t2.get("blocks", []):
        sec = str(block.get("section", ""))
        if sec:
            blocks_t2_by_section.setdefault(sec, []).append(block)

    enriched: dict[str, list[dict[str, Any]]] = triage_result.get("section_diffs_enriched", {})
    global_summary = triage_result.get("global_summary", {})

    section_comparisons: list[dict[str, Any]] = []

    for section_key, changes in enriched.items():
        block_comparisons: list[dict[str, Any]] = []

        for i, change in enumerate(changes, 1):
            change_type = change.get("change_type", "")
            genai_triage = change.get("genai_triage") or {}

            # Resolve block references
            block_t1_ref, block_t2_ref = _resolve_block_refs(
                change, blocks_t1_by_section, blocks_t2_by_section, section_key
            )

            # Build block_t1 and block_t2 entries
            def _make_block_entry(
                ref: dict[str, Any] | None,
                fallback_text: str,
            ) -> dict[str, Any] | None:
                if ref:
                    return {
                        "block_id": ref.get("block_id", ""),
                        "text": ref.get("text", fallback_text),
                        "page": ref.get("page", 0),
                    }
                if fallback_text:
                    return {
                        "block_id": "",
                        "text": fallback_text,
                        "page": 0,
                    }
                return None

            block_t1_entry = _make_block_entry(block_t1_ref, change.get("text_t1", ""))
            block_t2_entry = _make_block_entry(block_t2_ref, change.get("text_t2", ""))

            # Canonical change_id
            section_short = section_key[:3]
            change_id = f"{section_short}_chg_{i:03d}"

            # Signals from triage
            signals = genai_triage.get("signals") or {
                "regulatory_reference_added": False,
                "methodology_change": False,
                "tone_change": False,
                "forward_looking": False,
                "quantitative_change": False,
            }

            block_comp: dict[str, Any] = {
                "change_id": change_id,
                "diff_type": change_type,
                "change_summary": change.get("change_summary", ""),
                "signals": signals,
                "genai_triage": {
                    "is_relevant": genai_triage.get("is_relevant", False),
                    "category": genai_triage.get("category", "INCONNU"),
                    "impact_level": genai_triage.get("impact_level", "MINEUR"),
                    "risk_type": genai_triage.get("risk_type", "autre"),
                    "relevance_score": genai_triage.get("relevance_score", "FAIBLE"),
                    "risk_level": genai_triage.get("risk_level", "FAIBLE"),
                    "explanation": genai_triage.get("explanation", ""),
                    "impact_description": genai_triage.get("impact_description", ""),
                    "action_requise": genai_triage.get("action_requise", "aucune"),
                    "reference_reglementaire": genai_triage.get("reference_reglementaire", ""),
                    "confidence": genai_triage.get("confidence", 0.0),
                    "source": genai_triage.get("source", "heuristic"),
                },
                "block_t1": block_t1_entry,
                "block_t2": block_t2_entry,
            }

            block_comparisons.append(block_comp)

        section_summary = _compute_section_summary(changes)

        section_comparisons.append(
            {
                "section_key": section_key,
                "section_title": _SECTION_DISPLAY.get(section_key, section_key),
                "summary": section_summary,
                "block_comparisons": block_comparisons,
            }
        )

    # Build global_summary counts
    counts = global_summary.get("counts") or {}

    payload: dict[str, Any] = {
        "schema_version": TEXT_COMPARISON_SCHEMA_VERSION,
        "bank_code": bank_code.lower(),
        "quarter_previous": f"{year_t1}_{quarter_t1}",
        "quarter_current": f"{year_t2}_{quarter_t2}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "global_summary": {
            "executive_overview": global_summary.get("executive_overview", ""),
            "key_highlights": global_summary.get("key_highlights", []),
            "pertinence_globale": global_summary.get("pertinence_globale", "FAIBLE"),
            "counts": counts,
        },
        "section_comparisons": section_comparisons,
    }

    total_changes = sum(len(sc.get("block_comparisons", [])) for sc in section_comparisons)
    logger.info(
        "text_comparison_builder: %s %s_vs_%s — %d sections, %d changements",
        bank_code,
        f"{year_t2}_{quarter_t2}",
        f"{year_t1}_{quarter_t1}",
        len(section_comparisons),
        total_changes,
    )

    return payload
