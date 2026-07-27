"""Composants modulaires du pipeline texte."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vigilance.cli.quarter_logic import normalize_quarter, resolve_previous_quarter
from vigilance.comparison_io import _atomic_write_json
from vigilance.config import get_text_extraction_config
from vigilance.text_analysis.comparison import _compare_section_texts
from vigilance.text_analysis.constants import UNIFIED_TEXT_SCHEMA_VERSION, _SECTION_LABELS, _T4_TEXT_TARGET_SECTIONS
from vigilance.text_analysis.extraction import _extract_audits_for_pdf
from vigilance.text_analysis.global_reconciliation import reconcile_global_change_fragments
from vigilance.text_analysis.markdown import (
    _build_block_page_index,
    _build_text_extraction_markdown,
    _extract_section_text_from_markdown,
    _find_page_for_fragment,
    _parse_page_index_from_markdown,
    _rewrite_page_markers_for_display,
    _section_page_range_from_index,
)
from vigilance.text_analysis.models import TextAnalysisQualityError
from vigilance.text_analysis.openai_client import (
    _build_openai_client,
    _embed_texts,
)
from vigilance.text_analysis.precedent_memory import PrecedentMemory
from vigilance.text_analysis.sections import _allowed_target_sections, _resolve_sections
from vigilance.text_analysis.summary import _build_global_summary, _is_non_cosmetic_change, _retained_change_sort_key
from vigilance.text_analysis.summary import _build_semantic_quality_metrics
from vigilance.text_analysis.triage import (
    _evaluate_consolidated_dossier_materiality,
    _triage_section_changes,
)
from vigilance.text_comparison.text_comparison_writer import get_text_comparison_path, write_text_comparison
from vigilance.text_extraction.text_extraction_markdown_writer import (
    get_canonical_text_extraction_md_path,
    get_raw_docling_markdown_path,
    get_text_extraction_markdown_path,
    has_current_text_extraction_cache_schema,
    stamp_text_extraction_cache_schema,
    write_text_extraction_markdown,
)
from vigilance.text_extraction.text_extraction_audit_writer import (
    TEXT_EXTRACTION_AUDIT_SCHEMA_VERSION,
    get_canonical_text_extraction_audit_path,
    write_text_extraction_audit,
)
from vigilance.text_analysis.vision_boundary_validator import build_text_boundary_validator

logger = logging.getLogger(__name__)

_TEXT_ANALYSIS_CHECKPOINT_SCHEMA = "text_analysis_checkpoint_v1"
_TEXT_COMPARISON_LOGIC_VERSION = "canonical_comparison_reconciliation_v1"


def _emit_pipeline_progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(message)
    except Exception as exc:  # noqa: BLE001
        logger.debug("text pipeline progress callback ignored error=%s", exc)


def _text_analysis_checkpoint_key(
    *,
    bank_code: str,
    model: str,
    year_previous: int,
    quarter_previous: str,
    year_current: int,
    quarter_current: str,
    section_keys: list[str],
    markdown_previous: str,
    markdown_current: str,
) -> str:
    payload = {
        "checkpoint_schema": _TEXT_ANALYSIS_CHECKPOINT_SCHEMA,
        "comparison_logic_version": _TEXT_COMPARISON_LOGIC_VERSION,
        "bank_code": bank_code,
        "model": model,
        "year_previous": year_previous,
        "quarter_previous": quarter_previous,
        "year_current": year_current,
        "quarter_current": quarter_current,
        "section_keys": section_keys,
        "markdown_previous": markdown_previous,
        "markdown_current": markdown_current,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_text_analysis_checkpoint(
    path: Path,
    *,
    checkpoint_key: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            "text analysis checkpoint ignored path=%s error=%s",
            path,
            exc,
        )
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_schema")
        != _TEXT_ANALYSIS_CHECKPOINT_SCHEMA
        or payload.get("checkpoint_key") != checkpoint_key
    ):
        return {}
    return payload


def _store_text_analysis_checkpoint(
    path: Path,
    *,
    checkpoint_key: str,
    provisional_by_section: dict[str, list[dict[str, Any]]],
    reconciled_changes: list[dict[str, Any]] | None = None,
    reconciliation_audit: list[dict[str, Any]] | None = None,
) -> None:
    _atomic_write_json(
        path,
        {
            "checkpoint_schema": _TEXT_ANALYSIS_CHECKPOINT_SCHEMA,
            "checkpoint_key": checkpoint_key,
            "comparison_logic_version": _TEXT_COMPARISON_LOGIC_VERSION,
            "provisional_by_section": provisional_by_section,
            "reconciled_changes": reconciled_changes,
            "reconciliation_audit": reconciliation_audit,
        },
    )


def _prepare_period_extraction(
    *,
    pdf_path: Path,
    bank_code: str,
    year: int,
    quarter: str,
    project_root: Path,
    allowed_section_keys: set[str] | None,
    role: str,
    force_extraction: bool = False,
) -> tuple[str, dict[str, list[tuple[int, str]]], dict[str, tuple[int | None, int | None]], set[str]]:
    """Retourne (md, page_idx_by_key, section_range_by_key, available_keys).

    Si le .md canonique existe et est parsable, il est relu et Docling est sauté.
    Sinon, Docling s'exécute, construit le .md et l'écrit au chemin canonique
    ``outputs/text_extractions/{bank}/{year}/{q}/text_extraction.md``.

    Avec ``force_extraction=True``, le .md existant est ignoré et réécrit.
    """
    md_path = get_canonical_text_extraction_md_path(project_root, bank_code, year, quarter)

    if md_path.exists() and not force_extraction:
        try:
            original_md = md_path.read_text(encoding="utf-8")
            if not has_current_text_extraction_cache_schema(original_md):
                logger.info(".md canonique d'un schéma obsolète — ré-extraction: %s", md_path)
            else:
                md = _rewrite_page_markers_for_display(original_md)
                if md != original_md:
                    md_path.write_text(md, encoding="utf-8")
                    logger.info("Migration des marqueurs de pages du .md canonique: %s", md_path)
                page_idx_by_key, section_start_pages = _parse_page_index_from_markdown(md)
                if any(page_idx_by_key.values()):
                    section_range_by_key = {
                        key: _section_page_range_from_index(idx, start_page_hint=section_start_pages.get(key))
                        for key, idx in page_idx_by_key.items()
                    }
                    available_keys = set(page_idx_by_key.keys())
                    if allowed_section_keys is not None:
                        available_keys &= allowed_section_keys
                        page_idx_by_key = {k: v for k, v in page_idx_by_key.items() if k in available_keys}
                        section_range_by_key = {k: v for k, v in section_range_by_key.items() if k in available_keys}
                    logger.info("Réutilisation du .md canonique: %s", md_path)
                    return md, page_idx_by_key, section_range_by_key, available_keys
                logger.warning(".md canonique vide — ré-extraction: %s", md_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(".md canonique illisible (%s) — ré-extraction: %s", exc, md_path)
    elif md_path.exists():
        logger.info("Ré-extraction forcée, .md canonique ignoré: %s", md_path)

    sections = _resolve_sections(
        pdf_path,
        bank_code,
        quarter=quarter,
        year=year,
    )
    filtered_sections = (
        {k: v for k, v in sections.items() if k in allowed_section_keys}
        if allowed_section_keys is not None
        else sections
    )
    raw_docling_markdown_path = get_raw_docling_markdown_path(
        project_root,
        bank_code,
        year,
        quarter,
        role,
    )
    audits, raw_docling_markdown = _extract_audits_for_pdf(
        pdf_path=pdf_path,
        sections=filtered_sections,
        raw_docling_markdown_path=raw_docling_markdown_path,
    )
    audit_events: list[dict[str, Any]] = []
    boundary_validator = build_text_boundary_validator(
        pdf_path=pdf_path,
        project_root=project_root,
        config=get_text_extraction_config(bank_code=bank_code),
    )
    md = stamp_text_extraction_cache_schema(
        _build_text_extraction_markdown(
            audits,
            raw_docling_markdown=raw_docling_markdown,
            boundary_validator=boundary_validator,
            audit_events=audit_events,
        )
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    logger.info("Écriture du .md canonique: %s", md_path)
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for event in audit_events:
        action = str(event.get("action") or "unknown")
        reason = str(event.get("reason") or "unspecified")
        action_counts[action] = action_counts.get(action, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    audit_path = get_canonical_text_extraction_audit_path(md_path)
    write_text_extraction_audit(
        {
            "schema_version": TEXT_EXTRACTION_AUDIT_SCHEMA_VERSION,
            "artifact_type": "canonical_text_extraction_audit",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bank_code": bank_code,
            "year": year,
            "quarter": quarter.lower(),
            "source_pdf": str(pdf_path),
            "canonical_markdown": str(md_path),
            "summary": {
                "total_events": len(audit_events),
                "actions": action_counts,
                "reasons": reason_counts,
                "vision_calls": int(getattr(boundary_validator, "calls_made", 0)),
            },
            "events": audit_events,
        },
        audit_path,
    )
    logger.info("Écriture de l'audit du Markdown canonique: %s", audit_path)

    page_idx_by_key = {a.section_key: _build_block_page_index(a) for a in audits}
    section_range_by_key = {a.section_key: (a.start_page, a.end_page) for a in audits}
    available_keys = {a.section_key for a in audits}
    return md, page_idx_by_key, section_range_by_key, available_keys


def _resolve_text_project_root(out_root: Path) -> Path:
    """Déduit la racine projet à partir du répertoire de sortie texte."""
    if out_root.name == "resultats" and out_root.parent.name == "outputs":
        return out_root.parent.parent
    if out_root.name == "outputs":
        return out_root.parent
    return out_root.parent


def _effective_text_allowed_sections(
    bank_code: str,
    quarter_current: str,
    allowed_section_keys: set[str] | None,
) -> set[str] | None:
    effective_allowed = allowed_section_keys
    if effective_allowed is None:
        effective_allowed = _allowed_target_sections(bank_code) or None
    if quarter_current == "t4":
        effective_allowed = set(effective_allowed or set(_SECTION_LABELS)) & _T4_TEXT_TARGET_SECTIONS
    return effective_allowed


def run_text_extraction_pipeline(
    *,
    bank_code: str,
    year_current: int,
    quarter_current: str,
    pdf_previous: Path,
    pdf_current: Path,
    out_root: Path,
    allowed_section_keys: set[str] | None = None,
    project_root: Path | None = None,
    force_extraction: bool = False,
) -> dict[str, Any]:
    """Exécute seulement l'extraction texte et écrit les artefacts markdown."""
    quarter_current = normalize_quarter(quarter_current)
    year_previous, quarter_previous = resolve_previous_quarter(year_current, quarter_current)
    bank_code = bank_code.lower()

    if project_root is None:
        project_root = _resolve_text_project_root(out_root)

    effective_allowed = _effective_text_allowed_sections(
        bank_code,
        quarter_current,
        allowed_section_keys,
    )

    md_previous, _page_idx_prev, _range_prev, keys_prev = _prepare_period_extraction(
        pdf_path=pdf_previous,
        bank_code=bank_code,
        year=year_previous,
        quarter=quarter_previous,
        project_root=project_root,
        allowed_section_keys=effective_allowed,
        role="previous",
        force_extraction=force_extraction,
    )
    md_current, _page_idx_curr, _range_curr, keys_curr = _prepare_period_extraction(
        pdf_path=pdf_current,
        bank_code=bank_code,
        year=year_current,
        quarter=quarter_current,
        project_root=project_root,
        allowed_section_keys=effective_allowed,
        role="current",
        force_extraction=force_extraction,
    )

    if not (keys_prev | keys_curr):
        raise TextAnalysisQualityError("Aucune section texte ciblée localisée dans les rapports.")

    out_path = get_text_comparison_path(
        out_root=out_root,
        bank_code=bank_code,
        year_t2=year_current,
        quarter_t2=quarter_current,
        year_t1=year_previous,
        quarter_t1=quarter_previous,
    )
    out_dir = out_path.parent
    previous_artifact = get_text_extraction_markdown_path(
        out_dir,
        f"{year_previous}_{quarter_previous}",
    )
    current_artifact = get_text_extraction_markdown_path(
        out_dir,
        f"{year_current}_{quarter_current}",
    )
    write_text_extraction_markdown(md_previous, previous_artifact)
    write_text_extraction_markdown(md_current, current_artifact)

    return {
        "schema_version": UNIFIED_TEXT_SCHEMA_VERSION,
        "artifact_type": "text_extraction",
        "pipeline": "gpt4o_markdown_source_of_truth",
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": f"{year_previous}_{quarter_previous}",
        "year_current": year_current,
        "quarter_current": f"{year_current}_{quarter_current}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "force_extraction": force_extraction,
        "sections_previous": sorted(keys_prev),
        "sections_current": sorted(keys_curr),
        "extraction_artifact_t1": str(previous_artifact),
        "extraction_artifact_t2": str(current_artifact),
        "canonical_extraction_t1": str(
            get_canonical_text_extraction_md_path(
                project_root,
                bank_code,
                year_previous,
                quarter_previous,
            )
        ),
        "canonical_extraction_t2": str(
            get_canonical_text_extraction_md_path(
                project_root,
                bank_code,
                year_current,
                quarter_current,
            )
        ),
        "canonical_extraction_audit_t1": str(
            get_canonical_text_extraction_audit_path(
                get_canonical_text_extraction_md_path(
                    project_root,
                    bank_code,
                    year_previous,
                    quarter_previous,
                )
            )
        ),
        "canonical_extraction_audit_t2": str(
            get_canonical_text_extraction_audit_path(
                get_canonical_text_extraction_md_path(
                    project_root,
                    bank_code,
                    year_current,
                    quarter_current,
                )
            )
        ),
    }


def run_text_analysis_pipeline(
    *,
    bank_code: str,
    year_current: int,
    quarter_current: str,
    pdf_previous: Path,
    pdf_current: Path,
    out_root: Path,
    model: str = "gpt-4o",
    allowed_section_keys: set[str] | None = None,
    project_root: Path | None = None,
    force_extraction: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Exécute le pipeline texte complet avec le markdown comme source de vérité.

    Déroulement :

    1. Pour chaque période, si le .md canonique existe sur disque, relecture et parse ; sinon Docling + construction du .md + écriture canonique pour réutilisation.
    2. Envoi des sections .md à GPT-4o pour comparaison T1 vs T2 (ajouts, modifications, retraits).
    3. Triage et retenue uniquement des changements substantiels, triés par impact métier.

    Notes opérationnelles :

    * Le .md canonique vit dans ``outputs/text_extractions/{bank}/{year}/{q}/text_extraction.md`` et sert d'artefact d'audit ET d'entrée réutilisable pour les runs suivants.
    * ``force_extraction=True`` ignore ce cache et force une ré-extraction Docling.
    * Les marqueurs ``[p.N]`` présents dans le .md sont strippés avant tout appel GPT (gatekeeper unique : ``_extract_section_text_from_markdown``).
    * Les appels GPT du flux texte n'envoient pas de limite de complétion explicite par défaut, afin de privilégier l'arrêt naturel du modèle.
    """
    quarter_current = normalize_quarter(quarter_current)
    year_previous, quarter_previous = resolve_previous_quarter(year_current, quarter_current)
    bank_code = bank_code.lower()

    if project_root is None:
        project_root = _resolve_text_project_root(out_root)

    out_path = get_text_comparison_path(
        out_root=out_root,
        bank_code=bank_code,
        year_t2=year_current,
        quarter_t2=quarter_current,
        year_t1=year_previous,
        quarter_t1=quarter_previous,
    )
    triage_cache_dir = out_path.parent / ".text_triage_cache"

    effective_allowed = _effective_text_allowed_sections(
        bank_code,
        quarter_current,
        allowed_section_keys,
    )

    md_previous, page_idx_prev, range_prev, keys_prev = _prepare_period_extraction(
        pdf_path=pdf_previous,
        bank_code=bank_code,
        year=year_previous,
        quarter=quarter_previous,
        project_root=project_root,
        allowed_section_keys=effective_allowed,
        role="previous",
        force_extraction=force_extraction,
    )
    md_current, page_idx_curr, range_curr, keys_curr = _prepare_period_extraction(
        pdf_path=pdf_current,
        bank_code=bank_code,
        year=year_current,
        quarter=quarter_current,
        project_root=project_root,
        allowed_section_keys=effective_allowed,
        role="current",
        force_extraction=force_extraction,
    )

    section_keys = sorted(keys_prev | keys_curr)
    if not section_keys:
        raise TextAnalysisQualityError("Aucune section texte ciblée localisée dans les rapports.")

    # Validate: at least one period must have content per section
    for section_key in section_keys:
        text_t1 = _extract_section_text_from_markdown(md_previous, section_key)
        text_t2 = _extract_section_text_from_markdown(md_current, section_key)
        if not text_t1.strip() and not text_t2.strip():
            raise TextAnalysisQualityError(f"Section vide dans les deux rapports: {section_key}")

    # First GPT pass: local comparison on .md sections.  Do not triage yet:
    # a later global pass can reconcile content that was moved or resegmented
    # between different subsections (or even target sections).
    client = _build_openai_client()
    checkpoint_path = out_path.parent / ".text_analysis_checkpoint.json"
    checkpoint_key = _text_analysis_checkpoint_key(
        bank_code=bank_code,
        model=model,
        year_previous=year_previous,
        quarter_previous=quarter_previous,
        year_current=year_current,
        quarter_current=quarter_current,
        section_keys=section_keys,
        markdown_previous=md_previous,
        markdown_current=md_current,
    )
    checkpoint = _load_text_analysis_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
    )
    raw_provisional = checkpoint.get("provisional_by_section")
    provisional_by_section: dict[str, list[dict[str, Any]]] = (
        {
            str(section_key): [
                dict(change)
                for change in changes
                if isinstance(change, dict)
            ]
            for section_key, changes in raw_provisional.items()
            if (
                str(section_key) in section_keys
                and isinstance(changes, list)
            )
        }
        if isinstance(raw_provisional, dict)
        else {}
    )
    for section_index, section_key in enumerate(section_keys, start=1):
        if section_key in provisional_by_section:
            _emit_pipeline_progress(
                progress_callback,
                (
                    f"Comparaison {section_index}/{len(section_keys)} : "
                    f"{_SECTION_LABELS.get(section_key, section_key)} "
                    "reprise du cache."
                ),
            )
            continue
        _emit_pipeline_progress(
            progress_callback,
            (
                f"Comparaison {section_index}/{len(section_keys)} : "
                f"{_SECTION_LABELS.get(section_key, section_key)}."
            ),
        )
        text_t1 = _extract_section_text_from_markdown(md_previous, section_key)
        text_t2 = _extract_section_text_from_markdown(md_current, section_key)
        changes = _compare_section_texts(
            client=client,
            model=model,
            bank_code=bank_code,
            section_key=section_key,
            text_t1=text_t1,
            text_t2=text_t2,
        )
        # Inject exact page numbers by matching GPT text fragments back to block index.
        # Falls back to section start_page if no block match is found.
        _block_idx_t1 = page_idx_prev.get(section_key, [])
        _block_idx_t2 = page_idx_curr.get(section_key, [])
        _fallback_t1 = range_prev.get(section_key, (None, None))[0]
        _fallback_t2 = range_curr.get(section_key, (None, None))[0]
        for _ch in changes:
            _dt = _ch.get("diff_type", "")
            _txt_t1 = _ch.get("source_text_t1") or ""
            _txt_t2 = _ch.get("source_text_t2") or ""
            if _dt in {"modified", "removed", "unchanged", "renamed"} and _txt_t1:
                _pg = _find_page_for_fragment(_txt_t1, _block_idx_t1)
                _p_t1: list[int] = [_pg] if _pg else ([_fallback_t1] if _fallback_t1 else [])
            else:
                _p_t1 = []
            if _dt in {"modified", "added", "unchanged", "renamed"} and _txt_t2:
                _pg = _find_page_for_fragment(_txt_t2, _block_idx_t2)
                _p_t2: list[int] = [_pg] if _pg else ([_fallback_t2] if _fallback_t2 else [])
            else:
                _p_t2 = []
            _ch["pages_t1"] = _p_t1
            _ch["pages_t2"] = _p_t2
            if isinstance(_ch.get("evidence_t1"), dict):
                _ch["evidence_t1"]["pages"] = _p_t1
            if isinstance(_ch.get("evidence_t2"), dict):
                _ch["evidence_t2"]["pages"] = _p_t2

        provisional_by_section[section_key] = changes
        _store_text_analysis_checkpoint(
            checkpoint_path,
            checkpoint_key=checkpoint_key,
            provisional_by_section=provisional_by_section,
        )

    raw_reconciled = checkpoint.get("reconciled_changes")
    raw_reconciliation_audit = checkpoint.get("reconciliation_audit")
    if isinstance(raw_reconciled, list) and isinstance(
        raw_reconciliation_audit,
        list,
    ):
        reconciled_changes = [
            dict(change)
            for change in raw_reconciled
            if isinstance(change, dict)
        ]
        reconciliation_audit = [
            dict(row)
            for row in raw_reconciliation_audit
            if isinstance(row, dict)
        ]
        _emit_pipeline_progress(
            progress_callback,
            "Réconciliation globale reprise du cache.",
        )
    else:
        provisional_changes = [
            change
            for section_key in section_keys
            for change in provisional_by_section.get(section_key, [])
        ]
        reconciled_changes, reconciliation_audit = (
            reconcile_global_change_fragments(
                client=client,
                model=model,
                changes=provisional_changes,
            )
        )
        _store_text_analysis_checkpoint(
            checkpoint_path,
            checkpoint_key=checkpoint_key,
            provisional_by_section=provisional_by_section,
            reconciled_changes=reconciled_changes,
            reconciliation_audit=reconciliation_audit,
        )
    reconciled_by_section: dict[str, list[dict[str, Any]]] = {section_key: [] for section_key in section_keys}
    for change in reconciled_changes:
        section_key = str(change.get("section_key") or "")
        if section_key in reconciled_by_section:
            reconciled_by_section[section_key].append(change)

    # Second GPT pass: AMF triage only after global reconciliation has removed
    # false added/removed records created by a split or a move.
    precedent_memory = PrecedentMemory.from_paths(
        out_root,
        embedding_engine=lambda texts: _embed_texts(
            client,
            list(texts),
            model="text-embedding-3-small",
        ),
    )
    logger.info(
        "Mémoire de précédents analystes: %d cas validés, %d fichier(s) lus",
        len(precedent_memory.precedents),
        precedent_memory.load_report.files_seen,
    )
    section_comparisons: list[dict[str, Any]] = []
    for section_index, section_key in enumerate(section_keys, start=1):
        changes = reconciled_by_section.get(section_key, [])
        non_unchanged = [change for change in changes if change.get("diff_type") != "unchanged"]
        _emit_pipeline_progress(
            progress_callback,
            (
                f"Triage {section_index}/{len(section_keys)} : "
                f"{_SECTION_LABELS.get(section_key, section_key)}, "
                f"{len(non_unchanged)} changements candidats."
            ),
        )
        enriched = _triage_section_changes(
            client=client,
            model=model,
            bank_code=bank_code,
            section_key=section_key,
            changes=non_unchanged,
            precedent_memory=precedent_memory,
            cache_dir=triage_cache_dir,
            progress_callback=progress_callback,
        )
        enriched = _evaluate_consolidated_dossier_materiality(
            client=client,
            model=model,
            bank_code=bank_code,
            section_key=section_key,
            changes=enriched,
        )
        all_changes = [dict(change) for change in enriched]
        all_changes.sort(key=_retained_change_sort_key)
        retained = [change for change in enriched if _is_non_cosmetic_change(change["genai_triage"])]
        retained.sort(key=_retained_change_sort_key)
        _prev_start, _prev_end = range_prev.get(section_key, (None, None))
        _curr_start, _curr_end = range_curr.get(section_key, (None, None))
        section_comparisons.append(
            {
                "section_key": section_key,
                "section_title": _SECTION_LABELS.get(section_key, section_key),
                "block_comparisons": retained,
                "all_block_comparisons": all_changes,
                "summary": {
                    "retained_changes": len(retained),
                    "all_changes": len(all_changes),
                    "pages_previous": [p for p in (_prev_start, _prev_end) if p is not None],
                    "pages_current": [p for p in (_curr_start, _curr_end) if p is not None],
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": UNIFIED_TEXT_SCHEMA_VERSION,
        "artifact_type": "text_comparison",
        "pipeline": "gpt4o_markdown_source_of_truth",
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": f"{year_previous}_{quarter_previous}",
        "year_current": year_current,
        "quarter_current": f"{year_current}_{quarter_current}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "extraction_artifact_t1": f"text_extraction_{year_previous}_{quarter_previous}.md",
        "extraction_artifact_t2": f"text_extraction_{year_current}_{quarter_current}.md",
        "section_comparisons": section_comparisons,
        "global_reconciliation_audit": reconciliation_audit,
        "semantic_quality_metrics": _build_semantic_quality_metrics(
            section_comparisons=section_comparisons,
            reconciliation_audit=reconciliation_audit,
        ),
        "precedent_memory": {
            "schema_version": "analyst_precedent_v1",
            "precedent_count": len(precedent_memory.precedents),
            "files_seen": precedent_memory.load_report.files_seen,
            "records_seen": precedent_memory.load_report.records_seen,
            "accepted_records": precedent_memory.load_report.accepted_records,
            "rejected_records": precedent_memory.load_report.rejected_records,
            "duplicate_records": precedent_memory.load_report.duplicate_records,
            "errors": list(precedent_memory.load_report.errors),
        },
    }
    payload["global_summary"] = _build_global_summary(
        section_comparisons,
        bank_code=bank_code,
    )
    payload["all_changes_summary"] = _build_global_summary(
        [
            {
                "section_key": section["section_key"],
                "block_comparisons": section.get("all_block_comparisons") or [],
            }
            for section in section_comparisons
        ],
        bank_code=bank_code,
    )

    out_dir = out_path.parent
    write_text_extraction_markdown(
        md_previous,
        get_text_extraction_markdown_path(out_dir, f"{year_previous}_{quarter_previous}"),
    )
    write_text_extraction_markdown(
        md_current,
        get_text_extraction_markdown_path(out_dir, f"{year_current}_{quarter_current}"),
    )
    write_text_comparison(payload, out_path)
    return payload, out_path
