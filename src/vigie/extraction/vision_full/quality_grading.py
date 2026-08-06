"""Notation d'une extraction et selection du meilleur candidat entre tentatives.

Extrait de ``vision_full_extractor.py`` sans modification.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .quality_heuristics import (
    _contamination_score,
    _extract_footnote_marker_ids,
    _has_dominant_contamination,
    _has_generic_title_without_support,
    _has_strong_non_summary_signals,
    _is_generic_page_title,
    _is_viable_result,
    _narrative_indicator_count,
    _normalize_footnote_marker_id,
    _right_column_bleed_score,
    _structural_indicator_count,
    _viable_indicator_count,
)
from .result import VisionFullResult


def _grade_extraction_quality(result: VisionFullResult | None) -> list[str]:
    """Evalue la qualite de l'extraction et retourne une liste de critiques."""
    if result is None:
        return []

    critiques: list[str] = []
    indicators: list[Any] = result.indicators or []
    footnotes: list[Any] = result.footnotes_content or []

    # 1. Flat Indentation Check — DISABLED
    # Indentation is cosmetic only. Comparison normalizer strips leading spaces before
    # matching so indentation never affects comparison accuracy. Enforcing it caused
    # infinite self-healing loops (17+ retries per table) with zero data benefit.

    # 2. Orphaned Footnote Marker Check
    found_footnote_ids = {
        _normalize_footnote_marker_id(fn.get("id"))
        for fn in footnotes
        if isinstance(fn, dict) and _normalize_footnote_marker_id(fn.get("id"))
    }
    referenced_markers = _extract_footnote_marker_ids(indicators)

    missing_footnotes = referenced_markers - found_footnote_ids
    if missing_footnotes and len(missing_footnotes) <= 4:
        missing_str = ", ".join(f"({m})" for m in missing_footnotes)
        critiques.append(
            f"Vous avez extrait des renvois de notes de bas de page dans les indicateurs [{missing_str}], "
            "mais vous avez OUBLIÉ d'inclure le texte de ces notes dans le champ 'footnotes_content'. "
            "Lisez le bas du tableau et ajoutez obligatoirement ces notes manquantes."
        )

    # 3. Missing Headers Check — DISABLED
    # Headers are SECONDARY priority. Retrying for empty headers wastes API calls
    # with no impact on indicator or footnote completeness.

    # 4. Le nombre de colonnes ne permet pas d'inferer le nombre de lignes.
    # Un tableau financier de 2 lignes et 4 colonnes peut etre parfaitement
    # complet. Les vrais signaux de troncature sont traites par la geometrie,
    # la densite verticale et l'inspection visuelle du crop.

    return critiques


def _collect_incompleteness_reasons(
    result: VisionFullResult | None,
    *,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
) -> list[str]:
    """Collecte les raisons d'incompletude d'un resultat d'extraction."""
    if result is None:
        return ["missing_result"]
    reasons: list[str] = []
    expected_ids = {
        marker for value in (expected_footnote_ids or set()) if (marker := _normalize_footnote_marker_id(value))
    }
    title = str(result.table_title or "").strip()
    summary = str(result.table_summary or "").strip()
    if "output_budget_truncated" in set(result.retry_reasons or []):
        reasons.append("output_budget_truncated")
    structural_count = _structural_indicator_count(result)
    if structural_count == 0:
        reasons.append("no_viable_indicators")
        headers = [str(value or "").strip() for value in list(result.headers or []) if str(value or "").strip()]
        if len(headers) >= 2 and (title or summary):
            reasons.append("missing_body_row_labels")
    if not summary:
        reasons.append("missing_table_summary")
    if title and _is_generic_page_title(title):
        reasons.append("generic_page_title")
    if _has_generic_title_without_support(result):
        reasons.append("generic_title_without_support")
        reasons.append("dominant_contamination")
    if _narrative_indicator_count(result) > 0:
        reasons.append("narrative_indicator_contamination")
    if _has_dominant_contamination(result):
        reasons.append("dominant_contamination")
    if bbox_norm and len(bbox_norm) >= 4 and bbox_norm[1] < 0.15 and not title:
        reasons.append("top_context_missing_title")

    if bbox_norm and len(bbox_norm) >= 4:
        bbox_height = max(0.0, float(bbox_norm[3]) - float(bbox_norm[1]))
        if bbox_height > 0.25 and structural_count <= 2:
            reasons.append("low_density_vertical")

    if bbox_norm and len(bbox_norm) >= 4 and bbox_norm[3] < 0.92 and expected_ids:
        found_ids = {
            _normalize_footnote_marker_id(item.get("id"))
            for item in list(result.footnotes_content or [])
            if isinstance(item, dict) and _normalize_footnote_marker_id(item.get("id"))
        }
        if not expected_ids.issubset(found_ids):
            reasons.append("missing_expected_footnotes")
    return reasons


def _select_targeted_rescue_variant(
    rejection_reasons: list[str],
    quality_critiques: list[str] | None = None,
    qa_missing_elements: str = "",
) -> str:
    """Choisir un seul recadrage de secours a partir du signal d'echec.

    La priorite va aux notes de bas de page, puis au contexte haut, a la
    contamination et enfin a la densite du corps du tableau. Un probleme
    purement semantique (resume absent, critique QA generique) conserve le
    recadrage initial et renforce seulement l'instruction d'extraction.
    """
    reasons = {str(value or "").strip() for value in rejection_reasons if str(value or "").strip()}
    diagnostic_text = " ".join([*(quality_critiques or []), qa_missing_elements]).casefold()

    footnote_signal = bool(
        "missing_expected_footnotes" in reasons
        or "footnote" in diagnostic_text
        or "note de bas de page" in diagnostic_text
        or "notes de bas de page" in diagnostic_text
        or "renvois de notes" in diagnostic_text
    )
    if footnote_signal:
        return "bottom_extended"

    top_context_signal = bool(
        "top_context_missing_title" in reasons
        or "titre manquant" in diagnostic_text
        or "missing title" in diagnostic_text
        or "en-tête manquant" in diagnostic_text
        or "en-têtes manquants" in diagnostic_text
        or "missing header" in diagnostic_text
    )
    if top_context_signal:
        return "top_extended"

    if "generic_page_title" in reasons:
        return "top_trim"

    contamination_reasons = {
        "generic_title_without_support",
        "dominant_contamination",
        "narrative_indicator_contamination",
    }
    if reasons & contamination_reasons:
        return "tight_body"

    body_reasons = {
        "missing_result",
        "output_budget_truncated",
        "no_viable_indicators",
        "missing_body_row_labels",
        "weak_indicator_only",
        "low_density_vertical",
    }
    if reasons & body_reasons:
        return "body_expanded"

    return "same_crop_rescue"


def _candidate_quality_score(
    result: VisionFullResult | None,
    *,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
    baseline_result: VisionFullResult | None = None,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Calcule un tuple de score de qualite pour classer les candidats d'extraction."""
    if result is None:
        return (0, 0, 0, 0, 0, 0, 0, 0)
    viable_indicators = _viable_indicator_count(result)
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    title = str(result.table_title or "").strip()
    summary = str(result.table_summary or "").strip()
    found_footnotes = {
        _normalize_footnote_marker_id(item.get("id"))
        for item in list(result.footnotes_content or [])
        if isinstance(item, dict) and _normalize_footnote_marker_id(item.get("id"))
    }
    expected_ids = {
        marker for value in (expected_footnote_ids or set()) if (marker := _normalize_footnote_marker_id(value))
    }
    right_column_bleed = _right_column_bleed_score(
        result,
        baseline_result=baseline_result,
    )
    return (
        1 if _is_viable_result(result) else 0,
        -right_column_bleed,
        -_contamination_score(result),
        1 if summary else 0,
        viable_indicators,
        len(found_footnotes & expected_ids),
        len(headers),
        0 if _is_generic_page_title(title) else (1 if title else 0),
    )


def _finalize_selected_candidate(
    selected: VisionFullResult,
    *,
    best_name: str,
    initial_rejection_reasons: list[str],
    no_table_evidence_count: int,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
) -> VisionFullResult | None:
    """Finalise le candidat selectionne en evaluant son acceptabilite."""
    selected_rejection_reasons = _collect_incompleteness_reasons(
        selected,
        bbox_norm=bbox_norm,
        expected_footnote_ids=expected_footnote_ids,
    )
    combined_rejection_reasons = list(dict.fromkeys(initial_rejection_reasons + selected_rejection_reasons))
    summary_present = bool(str(selected.table_summary or "").strip())
    contamination_is_low = not _has_dominant_contamination(selected)
    structural_count = _structural_indicator_count(selected)
    headers = [str(v).strip() for v in list(selected.headers or []) if str(v).strip()]
    footnotes = [
        item
        for item in list(selected.footnotes_content or [])
        if isinstance(item, dict) and (str(item.get("id") or "").strip() or str(item.get("text") or "").strip())
    ]
    if _has_generic_title_without_support(selected):
        combined_rejection_reasons = list(
            dict.fromkeys(combined_rejection_reasons + ["generic_title_without_support", "dominant_contamination"])
        )
        contamination_is_low = False

    if summary_present and contamination_is_low and structural_count > 0:
        acceptance_reason = (
            "initial_complete"
            if best_name == "initial" and not initial_rejection_reasons
            else "rescued_summary_recovered"
        )
        return _build_result_debug_metadata(
            selected,
            acceptance_reason=acceptance_reason,
            rejection_reasons=combined_rejection_reasons,
            selected_candidate_name=best_name,
            no_table_evidence_count=no_table_evidence_count,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_footnote_ids,
        )

    if (
        not summary_present
        and contamination_is_low
        and structural_count >= 3
        and (len(headers) >= 2 or bool(footnotes))
        and _has_strong_non_summary_signals(selected)
    ):
        return _build_result_debug_metadata(
            replace(selected, extraction_status="rescued"),
            acceptance_reason="rescued_without_summary_strong_structure",
            rejection_reasons=combined_rejection_reasons,
            selected_candidate_name=best_name,
            no_table_evidence_count=no_table_evidence_count,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_footnote_ids,
        )
    return None


def _build_result_debug_metadata(
    result: VisionFullResult,
    *,
    acceptance_reason: str,
    rejection_reasons: list[str],
    selected_candidate_name: str,
    no_table_evidence_count: int,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
) -> VisionFullResult:
    """Enrichit le resultat avec les metadonnees de debug (raisons, scores, compteurs)."""
    quality_rank = list(
        _candidate_quality_score(
            result,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_footnote_ids,
            baseline_result=result,
        )
    )
    return replace(
        result,
        acceptance_reason=acceptance_reason,
        rejection_reasons=list(dict.fromkeys(rejection_reasons)),
        selected_candidate_name=selected_candidate_name,
        no_table_evidence_count=max(0, int(no_table_evidence_count)),
        summary_present=bool(str(result.table_summary or "").strip()),
        indicator_count=len([str(v).strip() for v in list(result.indicators or []) if str(v).strip()]),
        candidate_quality_rank=quality_rank,
    )
