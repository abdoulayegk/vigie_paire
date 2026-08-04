"""Enchainement de bout en bout du pipeline de comparaison."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from vigie.comparaison.devil_advocate import _devil_advocate_review
from vigie.comparaison.io import (
    _clean_title_for_bank,
    _coerce_pathlike,
    _load_tables_payload,
    _partition_tables_by_status,
    _table_card,
    _table_detail,
    _table_snapshot,
)
from vigie.comparaison.differences.comparaison_paire import diff_table_pair_gpt
from vigie.comparaison.rapprochement.contrats import _MATCHING_VALIDATION_ATTEMPTS
from vigie.comparaison.rapprochement.moteur_rapprochement import _run_table_matching
from vigie.comparaison.visual_sanity import (
    render_visual_sanity_proof,
    visual_sanity_check,
    visual_sanity_check_table_event,
)
from vigie.support.config import get_matching_thresholds, resolve_openai_model
from vigie.comparaison.pipeline.client_openai import (
    _call_openai_embeddings,
    _call_openai_json,
)
from vigie.comparaison.pipeline.construction_resultat import (
    ecrire_resultat_comparaison,
)
from vigie.comparaison.pipeline.evenements_tableaux import (
    appliquer_revue_avocat_diable,
    construire_etats_extraction,
    construire_evenements_non_apparies,
    valider_evenements_visuellement,
)
from vigie.comparaison.pipeline.traitement_paires import (
    appliquer_ancrage_t1,
    traiter_paires,
)

logger = logging.getLogger(__name__)


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
        """Construit les vues (cards, details, snapshots) pour T1 et T2."""
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

    if bank_code.strip().lower() == "rbc":
        for card in previous_cards:
            card["title"] = _clean_title_for_bank(card.get("title", ""), bank_code=bank_code)
        for card in current_cards:
            card["title"] = _clean_title_for_bank(card.get("title", ""), bank_code=bank_code)
    year_previous = int(previous_payload.get("year", 0) or 0)
    year_current = int(current_payload.get("year", 0) or 0)
    quarter_previous = str(previous_payload.get("quarter", "") or "")
    quarter_current = str(current_payload.get("quarter", "") or "")
    model_name = str(model or resolve_openai_model("default_genai", config_path=config_path))
    usage_records: list[dict[str, Any]] = []
    matching_settings = get_matching_thresholds(
        config_path or "configs/bank_profiles.yaml",
        bank_code=bank_code,
    )
    configured_hybrid_quarters = {
        str(value or "").strip().lower()
        for value in list(matching_settings.get("hybrid_embedding_quarters", []) or [])
        if str(value or "").strip()
    }
    hybrid_recovery_enabled = (
        bank_code.strip().lower() == "rbc"
        and bool(matching_settings.get("hybrid_embedding_recovery_enabled", False))
        and quarter_current.strip().lower() in configured_hybrid_quarters
    )

    match_result = _run_table_matching(
        previous_cards,
        current_cards,
        model=model_name,
        call_openai_json=_call_openai_json,
        usage_recorder=usage_records,
        hybrid_recovery_enabled=hybrid_recovery_enabled,
        hybrid_embedding_model=str(
            matching_settings.get("hybrid_embedding_model", "text-embedding-3-large")
        ),
        hybrid_top_k=max(1, int(matching_settings.get("hybrid_embedding_top_k", 5) or 5)),
        hybrid_min_confidence=float(matching_settings.get("hybrid_min_confidence", 0.75) or 0.75),
        call_openai_embeddings=_call_openai_embeddings,
    )

    (
        tables_added,
        tables_removed,
        boundary_scope_exclusions_previous,
        boundary_scope_exclusions_current,
    ) = construire_evenements_non_apparies(
        match_result,
        previous_snapshots=previous_snapshots,
        current_snapshots=current_snapshots,
        previous_lookup=previous_lookup,
        current_lookup=current_lookup,
    )
    tables_added, tables_removed = appliquer_revue_avocat_diable(
        match_result,
        tables_added,
        tables_removed,
        previous_business_tables=previous_business_tables,
        current_business_tables=current_business_tables,
        previous_snapshots=previous_snapshots,
        current_snapshots=current_snapshots,
        hybrid_recovery_enabled=hybrid_recovery_enabled,
        model_name=model_name,
        call_openai_json=_call_openai_json,
        usage_records=usage_records,
        reviewer=_devil_advocate_review,
    )
    (
        artifacts_confirmed_previous,
        artifacts_confirmed_current,
        extraction_suspects_previous,
        extraction_suspects_current,
    ) = construire_etats_extraction(
        previous_tables=previous_tables,
        current_tables=current_tables,
        previous_artifact_refs=previous_artifact_refs,
        current_artifact_refs=current_artifact_refs,
        previous_suspect_refs=previous_suspect_refs,
        current_suspect_refs=current_suspect_refs,
        previous_snapshots=previous_snapshots,
        current_snapshots=current_snapshots,
    )

    pair_comparisons, diff_calls_total = traiter_paires(
        match_result["matched_pairs"],
        previous_lookup=previous_lookup,
        current_lookup=current_lookup,
        previous_snapshots=previous_snapshots,
        current_snapshots=current_snapshots,
        source_pdf_previous=source_pdf_previous,
        source_pdf_current=source_pdf_current,
        model_name=model_name,
        call_openai_json=_call_openai_json,
        usage_records=usage_records,
        max_validation_attempts=_MATCHING_VALIDATION_ATTEMPTS,
        differ=diff_table_pair_gpt,
        renderer=render_visual_sanity_proof,
        verifier=visual_sanity_check,
    )
    if source_pdf_previous and source_pdf_current:
        tables_added, tables_removed = valider_evenements_visuellement(
            tables_added,
            tables_removed,
            match_result=match_result,
            previous_snapshots=previous_snapshots,
            current_snapshots=current_snapshots,
            source_pdf_previous=source_pdf_previous,
            source_pdf_current=source_pdf_current,
            model_name=model_name,
            call_openai_json=_call_openai_json,
            usage_records=usage_records,
            renderer=render_visual_sanity_proof,
            verifier=visual_sanity_check_table_event,
        )
    appliquer_ancrage_t1(pair_comparisons)

    return ecrire_resultat_comparaison(
        out_root_path=out_root_path,
        bank_code=bank_code,
        year_previous=year_previous,
        quarter_previous=quarter_previous,
        year_current=year_current,
        quarter_current=quarter_current,
        model_name=model_name,
        source_pdf_previous=source_pdf_previous,
        source_pdf_current=source_pdf_current,
        reference_resolution=reference_resolution,
        match_result=match_result,
        tables_added=tables_added,
        tables_removed=tables_removed,
        artifacts_confirmed_previous=artifacts_confirmed_previous,
        artifacts_confirmed_current=artifacts_confirmed_current,
        extraction_suspects_previous=extraction_suspects_previous,
        extraction_suspects_current=extraction_suspects_current,
        boundary_scope_exclusions_previous=boundary_scope_exclusions_previous,
        boundary_scope_exclusions_current=boundary_scope_exclusions_current,
        pair_comparisons=pair_comparisons,
        usage_records=usage_records,
        diff_calls_total=diff_calls_total,
        comparison_started_at=comparison_started_at,
        extraction_run_metrics=extraction_run_metrics,
        runtime_extraction_sec=runtime_extraction_sec,
    )
