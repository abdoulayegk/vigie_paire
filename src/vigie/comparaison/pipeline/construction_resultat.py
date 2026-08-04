"""Construction du resultat canonique et archivage des sources."""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vigie.comparaison.io import _atomic_write_json, _make_run_id
from vigie.comparaison.metrics import (
    _build_run_metrics,
    _count_high_priority_items,
    _count_pair_changes,
)
from vigie.comparaison.pipeline.resultat_models import (
    ComparisonRunResult,
    ComparisonSummary,
    MatchingBlock,
    ReferenceResolution,
)


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


def ecrire_resultat_comparaison(
    *,
    out_root_path: Path,
    bank_code: str,
    year_previous: int,
    quarter_previous: str,
    year_current: int,
    quarter_current: str,
    model_name: str,
    source_pdf_previous: str | None,
    source_pdf_current: str | None,
    reference_resolution: dict[str, Any] | None,
    match_result: dict[str, Any],
    tables_added: list[dict[str, Any]],
    tables_removed: list[dict[str, Any]],
    artifacts_confirmed_previous: list[dict[str, Any]],
    artifacts_confirmed_current: list[dict[str, Any]],
    extraction_suspects_previous: list[dict[str, Any]],
    extraction_suspects_current: list[dict[str, Any]],
    boundary_scope_exclusions_previous: list[dict[str, Any]],
    boundary_scope_exclusions_current: list[dict[str, Any]],
    pair_comparisons: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
    diff_calls_total: int,
    comparison_started_at: float,
    extraction_run_metrics: dict[str, Any] | None,
    runtime_extraction_sec: float | None,
) -> Path:
    """Construire les metriques, archiver les sources et ecrire le JSON final."""
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

    out_dir = (
        out_root_path
        / bank_code
        / f"{year_current}_{quarter_current}_vs_{year_previous}_{quarter_previous}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    archived_pdf_previous = _archive_source_pdf(
        source_pdf_previous,
        out_dir / "previous_report.pdf",
    )
    archived_pdf_current = _archive_source_pdf(
        source_pdf_current,
        out_dir / "current_report.pdf",
    )
    reference = (
        ReferenceResolution.model_validate(reference_resolution)
        if isinstance(reference_resolution, dict)
        else ReferenceResolution(
            mode="automatique",
            year_previous=year_previous,
            quarter_previous=quarter_previous,
            rule=REFERENCE_RESOLUTION_RULE,
        )
    )
    result = ComparisonRunResult(
        schema_version=COMPARISON_SCHEMA_VERSION,
        artifact_type="report_comparison",
        run_id=_make_run_id(),
        bank_code=bank_code,
        year_previous=year_previous,
        quarter_previous=quarter_previous,
        year_current=year_current,
        quarter_current=quarter_current,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_pdf_previous=str(source_pdf_previous or "").strip(),
        source_pdf_current=str(source_pdf_current or "").strip(),
        archived_pdf_previous=archived_pdf_previous,
        archived_pdf_current=archived_pdf_current,
        model_version=model_name,
        prompt_version_match=MATCH_PROMPT_VERSION,
        prompt_version_diff=DIFF_PROMPT_VERSION,
        reference_resolution=reference,
        matching=MatchingBlock(
            matched_pairs=match_result["matched_pairs"],
            tables_added=tables_added,
            tables_removed=tables_removed,
            artifacts_confirmed_previous=artifacts_confirmed_previous,
            artifacts_confirmed_current=artifacts_confirmed_current,
            extraction_suspects_previous=extraction_suspects_previous,
            extraction_suspects_current=extraction_suspects_current,
            boundary_scope_exclusions_previous=boundary_scope_exclusions_previous,
            boundary_scope_exclusions_current=boundary_scope_exclusions_current,
        ),
        pair_comparisons=pair_comparisons,
        run_metrics=run_metrics,
        summary=ComparisonSummary(
            matched_pairs_total=len(match_result["matched_pairs"]),
            tables_added_total=len(tables_added),
            tables_removed_total=len(tables_removed),
            artifacts_confirmed_previous_total=len(artifacts_confirmed_previous),
            artifacts_confirmed_current_total=len(artifacts_confirmed_current),
            extraction_suspects_previous_total=len(extraction_suspects_previous),
            extraction_suspects_current_total=len(extraction_suspects_current),
            boundary_scope_exclusions_previous_total=len(
                boundary_scope_exclusions_previous
            ),
            boundary_scope_exclusions_current_total=len(
                boundary_scope_exclusions_current
            ),
            indicator_changes_total=indicator_changes_total,
            footnote_changes_total=footnote_changes_total,
            high_priority_items_total=high_priority_items_total,
        ),
    )
    return _atomic_write_json(out_dir / "comparison.json", result.to_json_dict())
