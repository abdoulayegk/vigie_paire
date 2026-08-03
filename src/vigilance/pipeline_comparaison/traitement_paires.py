"""Comparaison, verification visuelle et ancrage des paires."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from vigilance.comparison_analyst import build_analyst_assessment
from vigilance.comparison_io import _coerce_int
from vigilance.comparison_noise_filter import (
    _filter_noise_from_diff,
    recompute_table_level_change,
)
from vigilance.pipeline_comparaison.ancrages_visuels import _visual_sanity_meta
from vigilance.pipeline_comparaison.evenements_tableaux import _pire_statut_rendu


logger = logging.getLogger(__name__)


def _rendre_preuves_paire(
    previous_table_snapshot: dict[str, Any],
    current_table_snapshot: dict[str, Any],
    *,
    source_pdf_previous: str | None,
    source_pdf_current: str | None,
    renderer: Callable[..., tuple[bytes | None, str]],
) -> tuple[bytes | None, bytes | None, str]:
    """Rendre les preuves visuelles T1 et T2 d'une paire appariee."""
    previous_render, previous_status = renderer(
        source_pdf_previous,
        page=previous_table_snapshot.get("page"),
        bbox=previous_table_snapshot.get("bbox"),
    )
    current_render, current_status = renderer(
        source_pdf_current,
        page=current_table_snapshot.get("page"),
        bbox=current_table_snapshot.get("bbox"),
    )
    return (
        previous_render,
        current_render,
        _pire_statut_rendu([previous_status, current_status]),
    )


def traiter_paires(
    matched_pairs: list[dict[str, Any]],
    *,
    previous_lookup: dict[str, dict[str, Any]],
    current_lookup: dict[str, dict[str, Any]],
    previous_snapshots: dict[str, dict[str, Any]],
    current_snapshots: dict[str, dict[str, Any]],
    source_pdf_previous: str | None,
    source_pdf_current: str | None,
    model_name: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_records: list[dict[str, Any]],
    max_validation_attempts: int,
    differ: Callable[..., dict[str, Any]],
    renderer: Callable[..., tuple[bytes | None, str]],
    verifier: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Calculer, filtrer et verifier les differences de toutes les paires."""
    pair_comparisons: list[dict[str, Any]] = []
    diff_calls_total = 0
    sanity_check_enabled = bool(source_pdf_previous and source_pdf_current)

    for pair in matched_pairs:
        previous_table_id = pair["previous_table_id"]
        current_table_id = pair["current_table_id"]
        diff = differ(
            previous_lookup[previous_table_id],
            current_lookup[current_table_id],
            model=model_name,
            call_openai_json=call_openai_json,
            usage_recorder=usage_records,
            max_validation_attempts=max_validation_attempts,
        )
        diff_calls_total += _coerce_int(diff.get("diff_calls_total"))

        diff.setdefault("visual_sanity_scope", ["indicators", "footnotes", "tables"])
        diff.setdefault("visual_sanity_render_mode", "full")
        diff.setdefault("visual_sanity_applied", False)
        diff.setdefault("visual_sanity_rejected_count", 0)
        diff.setdefault("visual_sanity_render_status", "ok")
        if sanity_check_enabled and any(
            diff.get("technical_diff", {}).get(key)
            for key in (
                "indicators_added",
                "indicators_removed",
                "indicators_renamed",
                "footnotes_added",
                "footnotes_removed",
                "footnotes_renamed",
            )
        ):
            previous_render, current_render, render_status = _rendre_preuves_paire(
                previous_snapshots[previous_table_id],
                current_snapshots[current_table_id],
                source_pdf_previous=source_pdf_previous,
                source_pdf_current=source_pdf_current,
                renderer=renderer,
            )
            if render_status == "ok":
                diff = verifier(
                    previous_render,
                    current_render,
                    diff,
                    model=model_name,
                    call_openai_json=call_openai_json,
                    usage_recorder=usage_records,
                )
            else:
                diff.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                    )
                )

        filtered_diff = _filter_noise_from_diff(diff["technical_diff"])
        filtered_diff["table_level_change"] = recompute_table_level_change(filtered_diff)
        pair_comparisons.append(
            {
                "previous_table_id": previous_table_id,
                "current_table_id": current_table_id,
                "match_confidence": pair["match_confidence"],
                "match_reason": pair.get("reason", ""),
                "diff_mode": str(diff.get("diff_mode", "") or ""),
                "previous_table": previous_snapshots[previous_table_id],
                "current_table": current_snapshots[current_table_id],
                "technical_diff": filtered_diff,
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[current_table_id],
                    technical_diff=filtered_diff,
                    change_kind="modifie",
                ),
                "reason": diff["reason"],
                "visual_sanity_applied": bool(diff.get("visual_sanity_applied", False)),
                "visual_sanity_rejected_count": _coerce_int(
                    diff.get("visual_sanity_rejected_count")
                ),
                "visual_sanity_scope": list(diff.get("visual_sanity_scope") or []),
                "visual_sanity_render_mode": str(
                    diff.get("visual_sanity_render_mode", "") or ""
                ),
                "visual_sanity_render_status": str(
                    diff.get("visual_sanity_render_status", "") or ""
                ),
            }
        )
    return pair_comparisons, diff_calls_total


def appliquer_ancrage_t1(pair_comparisons: list[dict[str, Any]]) -> None:
    """Signaler les derives de lignes probablement dues a l'extraction T2."""
    try:
        from vigilance.config.loader import load_config

        anchor_cfg = load_config("configs/bank_profiles.yaml")
        vision_cfg = anchor_cfg.get("vision_extraction", {})
        anchor_enabled = bool(vision_cfg.get("vision_t1_anchor_enabled", False))
        anchor_threshold = float(vision_cfg.get("vision_t1_anchor_diff_threshold", 0.20))
    except Exception:
        anchor_enabled = False
        anchor_threshold = 0.20

    if not anchor_enabled:
        return
    try:
        from vigilance.extraction.vision_t1_anchor import anchor_against_previous

        for pair_comp in pair_comparisons:
            prev_table = pair_comp.get("previous_table", {})
            curr_table = pair_comp.get("current_table", {})
            prev_indicators = [
                str(item)
                if isinstance(item, str)
                else str(item.get("label", item.get("name", "")))
                for item in (prev_table.get("indicators") or [])
            ]
            curr_indicators = [
                str(item)
                if isinstance(item, str)
                else str(item.get("label", item.get("name", "")))
                for item in (curr_table.get("indicators") or [])
            ]
            anchor_result = anchor_against_previous(
                table_id=str(curr_table.get("table_id", "")),
                table_title=str(curr_table.get("title", "")),
                current_indicators=curr_indicators,
                previous_indicators=prev_indicators,
                diff_threshold=anchor_threshold,
            )
            if anchor_result.skipped:
                continue
            pair_comp["t1_anchor"] = {
                "likely_extraction_error": anchor_result.likely_extraction_error,
                "explanation": anchor_result.explanation,
                "current_count": anchor_result.current_count,
                "previous_count": anchor_result.previous_count,
                "diff_ratio": anchor_result.diff_ratio,
            }
            if anchor_result.likely_extraction_error:
                logger.warning(
                    "T-1 anchor: table %s flagged as likely extraction error "
                    "(prev=%d, curr=%d, diff=%.0f%%)",
                    anchor_result.table_id,
                    anchor_result.previous_count,
                    anchor_result.current_count,
                    anchor_result.diff_ratio * 100,
                )
    except Exception as exc:
        logger.warning("T-1 anchoring failed (non-fatal): %s", exc)
