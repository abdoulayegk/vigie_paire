"""Pipeline de comparaison GPT-4o sur les artefacts canoniques tables.json."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vigilance.comparison_analyst import build_analyst_assessment
from vigilance.comparison_devil_advocate import (
    _devil_advocate_review,
)
from vigilance.comparison_diff_gpt import diff_table_pair_gpt
from vigilance.comparison_io import (
    _atomic_write_json,
    _coerce_int,
    _coerce_pathlike,
    _extract_usage_metrics,
    _load_tables_payload,
    _make_run_id,
    _merge_extraction_suspect_side,
    _partition_tables_by_status,
    _table_card,
    _table_detail,
    _table_snapshot,
    normalize_quarter,  # noqa: F401 — re-exported public API
    resolve_reference_period,  # noqa: F401 — re-exported public API
)
from vigilance.comparison_matching import (
    _MATCHING_VALIDATION_ATTEMPTS,
    _run_table_matching,
)
from vigilance.comparison_metrics import (
    _build_run_metrics,
    _count_high_priority_items,
    _count_pair_changes,
)
from vigilance.comparison_noise_filter import (
    _filter_noise_from_diff,
    recompute_table_level_change,
)
from vigilance.comparison_visual_sanity import (
    render_visual_sanity_proof,
    visual_sanity_check,
    visual_sanity_check_table_event,
)
from vigilance.config import resolve_openai_model
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.utils.genai import get_openai_api_key
from vigilance.utils.matching_normalizer import (
    normalize_for_matching,
    strip_temporal_expressions,
)
from vigilance.utils.proof_rendering import normalize_proof_bbox

logger = logging.getLogger(__name__)

MATCH_PROMPT_VERSION = "table_match_v8"
DIFF_PROMPT_VERSION = "table_diff_v4"
COMPARISON_SCHEMA_VERSION = 3


REFERENCE_RESOLUTION_RULE = "t2->t1 meme annee; t3->t2 meme annee; t1->t3 annee precedente; t4->t4 annee precedente"


def _archive_source_pdf(source: str | Path | None, target: Path) -> str:
    """Copier un PDF source dans le repertoire du run pour la portabilite inter-OS.

    Retourne le chemin de la copie archivee en cas de succes ; sinon retourne
    le chemin source original (ou ``""`` si absent). Les echecs sont logges mais
    non fatals : la comparaison reste utilisable sur la machine d'origine via le
    chemin absolu, et Dash sait retomber sur le voisin archive lorsqu'il existe.
    """
    raw = str(source or "").strip()
    if not raw:
        return ""
    src_path = Path(raw)
    if not src_path.exists():
        logger.warning("PDF source introuvable pour archivage: %s", raw)
        return raw
    if target.exists():
        try:
            if src_path.samefile(target):
                return str(target)
        except OSError:
            pass
    try:
        shutil.copy2(src_path, target)
        return str(target)
    except OSError as exc:
        logger.warning(
            "Echec de l'archivage du PDF %s -> %s: %s", src_path, target, exc
        )
        return raw


def _visual_sanity_meta(
    *,
    applied: bool,
    rejected_count: int,
    render_status: str,
) -> dict[str, Any]:
    """Construire le bloc de metadonnees de la verification visuelle."""
    return {
        "visual_sanity_applied": applied,
        "visual_sanity_rejected_count": int(rejected_count),
        "visual_sanity_scope": ["indicators", "footnotes", "tables"],
        "visual_sanity_render_mode": "full",
        "visual_sanity_render_status": render_status,
    }


def _normalize_table_anchor_section(value: Any) -> str:
    """Normaliser le nom de section pour l'ancrage visuel d'une table."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        normalized = canonicalize_section(raw)
    except Exception:
        normalized = raw
    return str(normalized or "").strip()


def _normalize_table_anchor_title(value: Any) -> str:
    """Normaliser le titre d'une table pour l'ancrage visuel."""
    raw = strip_temporal_expressions(str(value or ""), target="title", aggressive=True)
    return normalize_for_matching(raw, target="title")


def _call_openai_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int | None = None,
    temperature: float = 0.0,
    api_retry_max: int = 2,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison",
    response_model: type | None = None,
) -> dict[str, Any]:
    """Appeler l'API OpenAI avec sortie JSON.

    Quand *response_model* est une sous-classe de ``pydantic.BaseModel``, l'appel
    utilise les **Structured Outputs** OpenAI pour garantir la conformite au schema.
    Le modele valide est reconverti en dict pour que les appelants gardent une
    interface identique.

    ``max_completion_tokens=None`` (defaut) laisse le modele s'arreter naturellement
    sans plafond artificiel — privilegier la qualite complete plutot que la vitesse.
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    last_error: Exception | None = None
    use_structured = response_model is not None
    for attempt in range(api_retry_max + 1):
        if attempt > 0:
            time.sleep(1.5 * (2 ** (attempt - 1)))
        try:
            if use_structured:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": response_model,
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = client.beta.chat.completions.parse(**kwargs)
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Structured Output parsing returned None")
                data = parsed.model_dump()
            else:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("OpenAI response is not a JSON object")
            if usage_recorder is not None:
                prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
                usage_recorder.append(
                    {
                        "model": model,
                        "call_kind": call_kind,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
            return data
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = (
                "rate" in message
                and "limit" in message
                or "timeout" in message
                or "timed out" in message
                or "connection" in message
                or "connect" in message
            )
            if not retryable or attempt >= api_retry_max:
                break
    raise RuntimeError(f"OpenAI comparison call failed: {last_error}")


def compare_reports_gpt4o(
    previous_dir: Path | str,
    current_dir: Path | str,
    out_root: Path | str,
    *,
    model: str | None = None,
    config_path: str | None = None,
    reference_resolution: dict[str, Any] | None = None,
    source_pdf_previous: str | None = None,
    source_pdf_current: str | None = None,
    runtime_extraction_sec: float | None = None,
    extraction_run_metrics: dict[str, Any] | None = None,
) -> Path:
    """Executer le pipeline complet de comparaison rapport-a-rapport et ecrire l'artefact.

    Point d'entree public utilise par le CLI et l'application Dash. Charge les
    artefacts canoniques ``tables.json`` des deux trimestres, enrichit les tables
    pour le matching, execute le matcher multicouche, calcule les diffs semantiques
    par paire, agregue les resumes et metriques, puis ecrit ``comparison.json``
    dans un repertoire de sortie.

    Args:
        previous_dir: Repertoire d'extraction du trimestre de reference.
        current_dir: Repertoire d'extraction du trimestre courant.
        out_root: Repertoire racine ou le dossier de comparaison est cree.
        model: Surcharge optionnelle du modele OpenAI.
        config_path: Chemin optionnel de la configuration des modeles.
        reference_resolution: Metadonnees optionnelles decrivant la resolution
            du trimestre de reference.
        source_pdf_previous: Chemin PDF optionnel du rapport precedent.
        source_pdf_current: Chemin PDF optionnel du rapport courant.
        runtime_extraction_sec: Temps d'extraction optionnel propage dans les
            metriques finales.
        extraction_run_metrics: Metriques d'extraction optionnelles fusionnees
            dans les metriques finales.

    Returns:
        Chemin vers l'artefact ``comparison.json`` genere.
    """
    comparison_started_at = time.monotonic()
    previous_dir_path = _coerce_pathlike(previous_dir, "previous_dir")
    current_dir_path = _coerce_pathlike(current_dir, "current_dir")
    out_root_path = _coerce_pathlike(out_root, "out_root")

    previous_payload = _load_tables_payload(previous_dir_path)
    current_payload = _load_tables_payload(current_dir_path)

    previous_tables = [entry for entry in list(previous_payload.get("tables", []) or []) if isinstance(entry, dict)]
    current_tables = [entry for entry in list(current_payload.get("tables", []) or []) if isinstance(entry, dict)]
    (
        previous_business_tables,
        previous_artifact_refs,
        previous_suspect_refs,
    ) = _partition_tables_by_status(previous_tables)
    (
        current_business_tables,
        current_artifact_refs,
        current_suspect_refs,
    ) = _partition_tables_by_status(current_tables)

    def _build_views() -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        return (
            [_table_card(entry) for entry in previous_business_tables],
            [_table_card(entry) for entry in current_business_tables],
            {entry["table_id"]: _table_detail(entry) for entry in previous_business_tables},
            {entry["table_id"]: _table_detail(entry) for entry in current_business_tables},
            {entry["table_id"]: _table_snapshot(entry) for entry in previous_tables},
            {entry["table_id"]: _table_snapshot(entry) for entry in current_tables},
        )

    (
        previous_cards,
        current_cards,
        previous_lookup,
        current_lookup,
        previous_snapshots,
        current_snapshots,
    ) = _build_views()

    bank_code = str(current_payload.get("bank_code") or previous_payload.get("bank_code") or "")
    if not bank_code:
        raise ValueError("Missing bank_code in tables.json payloads")
    year_previous = int(previous_payload.get("year", 0) or 0)
    year_current = int(current_payload.get("year", 0) or 0)
    quarter_previous = str(previous_payload.get("quarter", "") or "")
    quarter_current = str(current_payload.get("quarter", "") or "")
    model_name = str(model or resolve_openai_model("default_genai", config_path=config_path))
    usage_records: list[dict[str, Any]] = []

    match_result = _run_table_matching(
        previous_cards,
        current_cards,
        model=model_name,
        call_openai_json=_call_openai_json,
        usage_recorder=usage_records,
    )
    tables_added: list[dict[str, Any]] = []
    for item in match_result["tables_added"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "ajoute",
        }
        tables_added.append(
            {
                **item,
                **current_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="ajoute",
                ),
            }
        )

    tables_removed: list[dict[str, Any]] = []
    for item in match_result["tables_removed"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "supprime",
        }
        tables_removed.append(
            {
                **item,
                **previous_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=previous_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="supprime",
                ),
            }
        )

    # --- Devil's Advocate: second-opinion review on unmatched / low-confidence ---
    low_confidence_pairs = [
        p for p in match_result.get("matched_pairs", []) if float(p.get("match_confidence", 1.0)) < 0.90
    ]
    da_added_cards = [
        _table_card(entry)
        for entry in current_business_tables
        if any(a.get("table_id") == entry.get("table_id") for a in match_result.get("tables_added", []))
    ]
    da_removed_cards = [
        _table_card(entry)
        for entry in previous_business_tables
        if any(r.get("table_id") == entry.get("table_id") for r in match_result.get("tables_removed", []))
    ]
    da_result = _devil_advocate_review(
        da_added_cards,
        da_removed_cards,
        low_confidence_pairs,
        model=model_name,
        call_openai_json=_call_openai_json,
        usage_recorder=usage_records,
    )
    # Promote new matches found by Devil's Advocate
    for new_match in da_result.get("new_matches", []):
        prev_id = str(new_match.get("previous_table_id", "") or "").strip()
        cur_id = str(new_match.get("current_table_id", "") or "").strip()
        if not prev_id or not cur_id:
            continue
        if prev_id not in previous_snapshots or cur_id not in current_snapshots:
            logger.warning("Devil's Advocate: skipping invalid match %s <-> %s", prev_id, cur_id)
            continue
        # Add to matched_pairs
        match_result["matched_pairs"].append(
            {
                "previous_table_id": prev_id,
                "current_table_id": cur_id,
                "match_confidence": float(new_match.get("match_confidence", 0.75)),
                "reason": str(new_match.get("reason", "")),
                "source": "devil_advocate",
            }
        )
        # Remove from tables_added / tables_removed
        tables_added = [t for t in tables_added if t.get("table_id") != cur_id]
        tables_removed = [t for t in tables_removed if t.get("table_id") != prev_id]
        logger.info(
            "Devil's Advocate promoted match: %s <-> %s (conf=%.2f)",
            prev_id,
            cur_id,
            float(new_match.get("match_confidence", 0.75)),
        )
    # Mark contested pairs for review
    for contested in da_result.get("contested_pairs", []):
        prev_id = str(contested.get("previous_table_id", "") or "").strip()
        cur_id = str(contested.get("current_table_id", "") or "").strip()
        for pair in match_result["matched_pairs"]:
            if pair.get("previous_table_id") == prev_id and pair.get("current_table_id") == cur_id:
                pair["review_required"] = True
                pair["devil_advocate_reason"] = str(contested.get("reason", ""))
                logger.info("Devil's Advocate contested pair: %s <-> %s", prev_id, cur_id)

    artifacts_confirmed_previous: list[dict[str, Any]] = []
    for item in previous_artifact_refs:
        table_id = item["table_id"]
        artifacts_confirmed_previous.append({**item, **previous_snapshots[table_id]})

    artifacts_confirmed_current: list[dict[str, Any]] = []
    for item in current_artifact_refs:
        table_id = item["table_id"]
        artifacts_confirmed_current.append({**item, **current_snapshots[table_id]})

    extraction_suspects_previous = _merge_extraction_suspect_side(
        previous_tables,
        previous_suspect_refs,
        previous_snapshots,
    )
    extraction_suspects_current = _merge_extraction_suspect_side(
        current_tables,
        current_suspect_refs,
        current_snapshots,
    )

    pair_comparisons: list[dict[str, Any]] = []
    diff_calls_total = 0
    _sanity_check_enabled = bool(source_pdf_previous and source_pdf_current)

    def _worst_render_status(statuses: list[str]) -> str:
        for candidate in (
            "skipped_missing_pdf",
            "skipped_missing_anchor",
            "skipped_missing_bbox",
            "skipped_render_failed",
        ):
            if candidate in statuses:
                return candidate
        return "ok"

    def _snapshot_has_render_anchor(snapshot: dict[str, Any]) -> bool:
        try:
            page = int(snapshot.get("page") or 0)
        except (TypeError, ValueError):
            return False
        return page > 0 and normalize_proof_bbox(snapshot.get("bbox")) is not None

    def _resolve_opposite_table_anchor(
        event_snapshot: dict[str, Any],
        opposite_snapshots: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        normalized_section = _normalize_table_anchor_section(event_snapshot.get("section"))
        normalized_title = _normalize_table_anchor_title(event_snapshot.get("title"))
        if not normalized_section or not normalized_title:
            return None

        candidates = [
            snapshot
            for snapshot in opposite_snapshots.values()
            if _snapshot_has_render_anchor(snapshot)
            and _normalize_table_anchor_section(snapshot.get("section")) == normalized_section
            and _normalize_table_anchor_title(snapshot.get("title")) == normalized_title
        ]
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _render_pair_proofs(
        previous_table_snapshot: dict[str, Any],
        current_table_snapshot: dict[str, Any],
    ) -> tuple[bytes | None, bytes | None, str]:
        previous_render, previous_status = render_visual_sanity_proof(
            source_pdf_previous,
            page=previous_table_snapshot.get("page"),
            bbox=previous_table_snapshot.get("bbox"),
        )
        current_render, current_status = render_visual_sanity_proof(
            source_pdf_current,
            page=current_table_snapshot.get("page"),
            bbox=current_table_snapshot.get("bbox"),
        )
        return (
            previous_render,
            current_render,
            _worst_render_status([previous_status, current_status]),
        )

    def _render_table_event_proofs(
        *,
        event_type: str,
        event_snapshot: dict[str, Any],
    ) -> tuple[bytes | None, bytes | None, str]:
        normalized_event_type = str(event_type or "").strip().lower()
        if normalized_event_type == "table_added":
            opposite_anchor = _resolve_opposite_table_anchor(
                event_snapshot,
                previous_snapshots,
            )
            if opposite_anchor is None:
                return None, None, "skipped_missing_anchor"
            previous_render, previous_status = render_visual_sanity_proof(
                source_pdf_previous,
                page=opposite_anchor.get("page"),
                bbox=opposite_anchor.get("bbox"),
            )
            current_render, current_status = render_visual_sanity_proof(
                source_pdf_current,
                page=event_snapshot.get("page"),
                bbox=event_snapshot.get("bbox"),
            )
        else:
            opposite_anchor = _resolve_opposite_table_anchor(
                event_snapshot,
                current_snapshots,
            )
            if opposite_anchor is None:
                return None, None, "skipped_missing_anchor"
            previous_render, previous_status = render_visual_sanity_proof(
                source_pdf_previous,
                page=event_snapshot.get("page"),
                bbox=event_snapshot.get("bbox"),
            )
            current_render, current_status = render_visual_sanity_proof(
                source_pdf_current,
                page=opposite_anchor.get("page"),
                bbox=opposite_anchor.get("bbox"),
            )
        return (
            previous_render,
            current_render,
            _worst_render_status([previous_status, current_status]),
        )

    for pair in match_result["matched_pairs"]:
        previous_table_id = pair["previous_table_id"]
        current_table_id = pair["current_table_id"]
        diff = diff_table_pair_gpt(
            previous_lookup[previous_table_id],
            current_lookup[current_table_id],
            model=model_name,
            call_openai_json=_call_openai_json,
            usage_recorder=usage_records,
            max_validation_attempts=_MATCHING_VALIDATION_ATTEMPTS,
        )
        diff_calls_total += _coerce_int(diff.get("diff_calls_total"))

        # --- Visual Sanity Check (post-diff) ---
        diff.setdefault(
            "visual_sanity_scope",
            ["indicators", "footnotes", "tables"],
        )
        diff.setdefault("visual_sanity_render_mode", "full")
        diff.setdefault("visual_sanity_applied", False)
        diff.setdefault("visual_sanity_rejected_count", 0)
        diff.setdefault("visual_sanity_render_status", "ok")
        if _sanity_check_enabled and any(
            diff.get("technical_diff", {}).get(k)
            for k in (
                "indicators_added",
                "indicators_removed",
                "indicators_renamed",
                "footnotes_added",
                "footnotes_removed",
                "footnotes_renamed",
            )
        ):
            prev_render, curr_render, render_status = _render_pair_proofs(
                previous_snapshots[previous_table_id],
                current_snapshots[current_table_id],
            )
            if render_status == "ok":
                diff = visual_sanity_check(
                    prev_render,
                    curr_render,
                    diff,
                    model=model_name,
                    call_openai_json=_call_openai_json,
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
                "visual_sanity_rejected_count": _coerce_int(diff.get("visual_sanity_rejected_count")),
                "visual_sanity_scope": list(diff.get("visual_sanity_scope") or []),
                "visual_sanity_render_mode": str(diff.get("visual_sanity_render_mode", "") or ""),
                "visual_sanity_render_status": str(diff.get("visual_sanity_render_status", "") or ""),
            }
        )

    if _sanity_check_enabled:
        filtered_tables_added: list[dict[str, Any]] = []
        for item in tables_added:
            previous_render, current_render, render_status = _render_table_event_proofs(
                event_type="table_added",
                event_snapshot=item,
            )
            if render_status != "ok":
                item.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                    )
                )
                filtered_tables_added.append(item)
                continue
            verdict = visual_sanity_check_table_event(
                previous_render,
                current_render,
                event_type="table_added",
                table_id=str(item.get("table_id", "") or ""),
                table_title=str(item.get("title", "") or ""),
                model=model_name,
                call_openai_json=_call_openai_json,
                usage_recorder=usage_records,
            )
            item.update({key: value for key, value in verdict.items() if key != "confirmed"})
            if verdict.get("confirmed", True):
                filtered_tables_added.append(item)
        tables_added = filtered_tables_added

        filtered_tables_removed: list[dict[str, Any]] = []
        for item in tables_removed:
            previous_render, current_render, render_status = _render_table_event_proofs(
                event_type="table_removed",
                event_snapshot=item,
            )
            if render_status != "ok":
                item.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                    )
                )
                filtered_tables_removed.append(item)
                continue
            verdict = visual_sanity_check_table_event(
                previous_render,
                current_render,
                event_type="table_removed",
                table_id=str(item.get("table_id", "") or ""),
                table_title=str(item.get("title", "") or ""),
                model=model_name,
                call_openai_json=_call_openai_json,
                usage_recorder=usage_records,
            )
            item.update({key: value for key, value in verdict.items() if key != "confirmed"})
            if verdict.get("confirmed", True):
                filtered_tables_removed.append(item)
        tables_removed = filtered_tables_removed

    # --- T-1 Anchoring: flag likely extraction errors based on row count drift ---
    try:
        from vigilance.config.loader import load_config

        _anchor_cfg = load_config("configs/bank_profiles.yaml")
        _vision_cfg = _anchor_cfg.get("vision_extraction", {})
        _t1_anchor_enabled = bool(_vision_cfg.get("vision_t1_anchor_enabled", False))
        _t1_anchor_threshold = float(_vision_cfg.get("vision_t1_anchor_diff_threshold", 0.20))
    except Exception:
        _t1_anchor_enabled = False
        _t1_anchor_threshold = 0.20

    if _t1_anchor_enabled:
        try:
            from vigilance.extraction.vision_t1_anchor import anchor_against_previous as _anchor_check

            for pair_comp in pair_comparisons:
                prev_table = pair_comp.get("previous_table", {})
                curr_table = pair_comp.get("current_table", {})
                prev_indicators = [
                    str(i) if isinstance(i, str) else str(i.get("label", i.get("name", "")))
                    for i in (prev_table.get("indicators") or [])
                ]
                curr_indicators = [
                    str(i) if isinstance(i, str) else str(i.get("label", i.get("name", "")))
                    for i in (curr_table.get("indicators") or [])
                ]

                anchor_result = _anchor_check(
                    table_id=str(curr_table.get("table_id", "")),
                    table_title=str(curr_table.get("title", "")),
                    current_indicators=curr_indicators,
                    previous_indicators=prev_indicators,
                    diff_threshold=_t1_anchor_threshold,
                )

                if not anchor_result.skipped:
                    pair_comp["t1_anchor"] = {
                        "likely_extraction_error": anchor_result.likely_extraction_error,
                        "explanation": anchor_result.explanation,
                        "current_count": anchor_result.current_count,
                        "previous_count": anchor_result.previous_count,
                        "diff_ratio": anchor_result.diff_ratio,
                    }
                    if anchor_result.likely_extraction_error:
                        logger.warning(
                            "T-1 anchor: table %s flagged as likely extraction error (prev=%d, curr=%d, diff=%.0f%%)",
                            anchor_result.table_id,
                            anchor_result.previous_count,
                            anchor_result.current_count,
                            anchor_result.diff_ratio * 100,
                        )
        except Exception as _t1_exc:
            logger.warning("T-1 anchoring failed (non-fatal): %s", _t1_exc)
    # --- End T-1 Anchoring ---

    indicator_changes_total, footnote_changes_total = _count_pair_changes(pair_comparisons)
    high_priority_items_total = _count_high_priority_items(
        pair_comparisons,
        tables_added,
        tables_removed,
    )
    comparison_runtime_sec = round(max(0.0, time.monotonic() - comparison_started_at), 3)
    run_metrics = _build_run_metrics(
        usage_records=usage_records,
        match_result=match_result,
        diff_calls_total=diff_calls_total,
        comparison_runtime_sec=comparison_runtime_sec,
        model_name=model_name,
        extraction_run_metrics=extraction_run_metrics,
        runtime_extraction_sec=float(runtime_extraction_sec or 0.0),
    )

    out_dir = out_root_path / bank_code / f"{year_current}_{quarter_current}_vs_{year_previous}_{quarter_previous}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id()
    # Archive PDFs inside the run directory for cross-OS portability: Dash falls
    # back to previous_report.pdf / current_report.pdf when absolute paths stored
    # in the JSON become invalid (e.g. run produced on macOS then opened on Windows).
    archived_pdf_previous = _archive_source_pdf(
        source_pdf_previous, out_dir / "previous_report.pdf"
    )
    archived_pdf_current = _archive_source_pdf(
        source_pdf_current, out_dir / "current_report.pdf"
    )
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "artifact_type": "report_comparison",
        "run_id": run_id,
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": quarter_previous,
        "year_current": year_current,
        "quarter_current": quarter_current,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_pdf_previous": str(source_pdf_previous or "").strip(),
        "source_pdf_current": str(source_pdf_current or "").strip(),
        "archived_pdf_previous": archived_pdf_previous,
        "archived_pdf_current": archived_pdf_current,
        "model_version": model_name,
        "prompt_version_match": MATCH_PROMPT_VERSION,
        "prompt_version_diff": DIFF_PROMPT_VERSION,
        "reference_resolution": (
            dict(reference_resolution)
            if isinstance(reference_resolution, dict)
            else {
                "mode": "automatique",
                "year_previous": year_previous,
                "quarter_previous": quarter_previous,
                "rule": REFERENCE_RESOLUTION_RULE,
            }
        ),
        "matching": {
            "matched_pairs": match_result["matched_pairs"],
            "tables_added": tables_added,
            "tables_removed": tables_removed,
            "artifacts_confirmed_previous": artifacts_confirmed_previous,
            "artifacts_confirmed_current": artifacts_confirmed_current,
            "extraction_suspects_previous": extraction_suspects_previous,
            "extraction_suspects_current": extraction_suspects_current,
        },
        "pair_comparisons": pair_comparisons,
        "run_metrics": run_metrics,
        "summary": {
            "matched_pairs_total": len(match_result["matched_pairs"]),
            "tables_added_total": len(tables_added),
            "tables_removed_total": len(tables_removed),
            "artifacts_confirmed_previous_total": len(artifacts_confirmed_previous),
            "artifacts_confirmed_current_total": len(artifacts_confirmed_current),
            "extraction_suspects_previous_total": len(extraction_suspects_previous),
            "extraction_suspects_current_total": len(extraction_suspects_current),
            "indicator_changes_total": indicator_changes_total,
            "footnote_changes_total": footnote_changes_total,
            "high_priority_items_total": high_priority_items_total,
        },
    }
    return _atomic_write_json(out_dir / "comparison.json", payload)
