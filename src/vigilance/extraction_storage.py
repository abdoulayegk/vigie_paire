"""Persistance des extractions canoniques ``TableArtifact`` avec ``tables.json`` comme source de verite."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.models.table_models import TableArtifact, infer_content_source
from vigilance.utils.indicator_cleaner import (
    normalize_indicator_for_comparison,
    post_normalize_indicator,
)
from vigilance.utils.table_page_structure import derive_page_local_structure

logger = logging.getLogger(__name__)

STORAGE_SCHEMA_VERSION = 7
TABLES_FILENAME = "tables.json"
REPORT_INDICATORS_FILENAME = "indicators.json"
REPORT_FOOTNOTES_FILENAME = "footnotes.json"

def _normalize_storage_quarter(quarter: str) -> str:
    """Normaliser les libelles de trimestre en ``t1``..``t4`` pour le stockage."""
    value = str(quarter or "").strip().lower()
    match = None
    if value:
        import re

        match = re.search(r"([qt])\s*([1-4])", value, flags=re.IGNORECASE)
    if match:
        return f"t{match.group(2)}"
    return value or "t1"


def _extraction_dir(base_dir: Path, bank_code: str, year: int, quarter: str) -> Path:
    """Retourner le repertoire d'un rapport : {base_dir}/{bank}/{year}/{quarter}/."""
    return base_dir / str(bank_code) / str(year) / str(quarter).lower()


def get_extraction_artifact_paths(
    bank_code: str,
    year: int,
    quarter: str,
    base_dir: Path,
) -> dict[str, Path]:
    """Retourner les chemins canoniques du systeme de fichiers pour une extraction stockee."""
    quarter_norm = _normalize_storage_quarter(quarter)
    target_dir = _extraction_dir(base_dir, bank_code, year, quarter_norm)
    return {
        "dir": target_dir,
        "tables": target_dir / TABLES_FILENAME,
        "indicators": target_dir / REPORT_INDICATORS_FILENAME,
        "footnotes": target_dir / REPORT_FOOTNOTES_FILENAME,
    }


def _footnote_to_canonical(item: dict[str, Any]) -> dict[str, str]:
    """Normaliser un element de note de bas de page vers la forme canonique stockee."""
    fid = (item.get("id") or item.get("marker") or "").strip()
    text = str(item.get("text") or "").strip()
    return {"id": fid, "text": text}


def _backfill_page_local_structure(tables: list[TableArtifact]) -> None:
    """Recalculer et assigner la structure locale de page sur chaque table apres rechargement."""
    structure = derive_page_local_structure(tables)
    for t in tables:
        key = (t.table_id, t.page_pdf)
        if key not in structure:
            continue
        info = structure[key]
        t.table_index_on_page = info.get("table_index_on_page")
        t.tables_on_page = info.get("tables_on_page")
        t.bbox_top = info.get("bbox_top")
        t.page_local_role = info.get("page_local_role")


def table_artifact_from_dict(d: dict[str, Any]) -> TableArtifact:
    """Reconstruire un ``TableArtifact`` canonique depuis le JSON minimal stocke."""
    bank_code = str(d.get("bank_code", "") or "")
    section = str(d.get("section", "") or "")
    page_pdf = int(d.get("page_pdf", d.get("page", 0)) or 0)
    table_id = str(d.get("table_id", "") or "")
    title = d.get("title")
    table_summary = str(d.get("table_summary", "") or "") or None
    headers = list(d.get("headers") or [])
    rows: list[list[str]] = []
    extraction_method = str(d.get("extraction_method") or "vision_minimal")
    content_source = infer_content_source(
        extraction_method,
        d.get("content_source"),
    )
    indicators_source = (
        d.get("indicators")
        if d.get("indicators") is not None
        else d.get("first_column_indicators_raw")
    )
    first_column_indicators_raw = [
        str(ind or "").strip()
        for ind in list(indicators_source or [])
        if str(ind or "").strip()
    ]
    first_column_indicators: list[str] = []
    for ind in first_column_indicators_raw:
        normalized = normalize_indicator_for_comparison(ind)
        if not normalized:
            continue
        fixed, _, _ = post_normalize_indicator(normalized)
        candidate = normalize_indicator_for_comparison(fixed or normalized)
        if candidate:
            first_column_indicators.append(candidate)
    title_clean = d.get("title")
    title_raw = d.get("title")
    row_count = int(d.get("row_count", len(first_column_indicators_raw)) or 0)
    bbox = d.get("bbox")
    quarter = d.get("quarter")
    pdf_path = d.get("pdf_path")
    footnotes = d.get("footnotes")
    if isinstance(footnotes, dict):
        footnotes = [
            {"id": str(k), "text": str(v)}
            for k, v in footnotes.items()  # preserves insertion order (Python 3.7+)
        ]
    elif footnotes is not None and not isinstance(footnotes, list):
        footnotes = None
    debug_metrics = d.get("debug_metrics")
    if debug_metrics is not None and not isinstance(debug_metrics, dict):
        debug_metrics = None
    bbox_provenance = d.get("bbox_provenance")
    if isinstance(bbox_provenance, dict):
        debug_metrics = {
            **(debug_metrics or {}),
            **bbox_provenance,
        }
    extraction_status = str(d.get("extraction_status") or "ok").strip() or "ok"
    return TableArtifact(
        bank_code=bank_code,
        section=section,
        page_pdf=page_pdf,
        table_id=table_id,
        title=title,
        headers=headers,
        rows=rows,
        first_column_indicators=first_column_indicators,
        extraction_method=extraction_method,
        title_clean=title_clean,
        table_summary=table_summary,
        title_raw=title_raw,
        row_count=row_count,
        bbox=bbox,
        quarter=quarter,
        pdf_path=pdf_path,
        first_column_indicators_raw=first_column_indicators_raw,
        footnotes=footnotes,
        debug_metrics=debug_metrics,
        content_source=content_source,
        extraction_status=extraction_status,
    )


def _ensure_projection_artifacts(target_dir: Path) -> None:
    """S'assurer que indicators.json et footnotes.json existent comme projections de tables.json."""
    tables_json_path = target_dir / TABLES_FILENAME
    if not tables_json_path.exists():
        return
    indicators_path = target_dir / REPORT_INDICATORS_FILENAME
    footnotes_path = target_dir / REPORT_FOOTNOTES_FILENAME
    if indicators_path.exists() and footnotes_path.exists():
        return
    from vigilance.extraction.vision_extraction_writer import (
        write_compact_footnotes_json,
        write_compact_indicators_json,
    )

    if not indicators_path.exists():
        write_compact_indicators_json(tables_json_path, target_dir)
    if not footnotes_path.exists():
        write_compact_footnotes_json(tables_json_path, target_dir)


def save_extraction(
    bank_code: str,
    year: int,
    quarter: str,
    tables: list[TableArtifact],
    meta: dict[str, Any],
    base_dir: Path,
) -> Path:
    """Sauvegarder l'extraction d'un rapport.

    Ecrit le fichier canonique ``tables.json``, puis derive ``indicators.json``
    et ``footnotes.json`` a partir de celui-ci.
    Arborescence : ``outputs/extractions/{bank_code}/{year}/{quarter}/``.

    Args:
        bank_code: Code identifiant la banque.
        year: Annee du rapport.
        quarter: Libelle du trimestre (ex. ``t1``).
        tables: Liste des artefacts de tableaux extraits.
        meta: Metadonnees d'extraction (model_version, etc.).
        base_dir: Repertoire racine des extractions.

    Returns:
        Chemin du repertoire contenant les artefacts sauvegardes.
    """
    quarter_norm = _normalize_storage_quarter(quarter)
    target_dir = _extraction_dir(base_dir, bank_code, year, quarter_norm)
    target_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = dict(meta or {})
    meta_payload.setdefault("schema_version", STORAGE_SCHEMA_VERSION)
    meta_payload.setdefault("bank_code", bank_code)
    meta_payload.setdefault("year", year)
    meta_payload.setdefault("quarter", quarter_norm)
    meta_payload.setdefault("extracted_at", datetime.now(timezone.utc).isoformat())
    meta_payload.setdefault("created_at", meta_payload.get("extracted_at"))
    from vigilance.extraction.vision_extraction_writer import write_compact_report_artifacts

    write_compact_report_artifacts(
        tables=tables,
        out_dir=target_dir,
        bank_code=bank_code,
        year=year,
        quarter=quarter_norm,
        meta=meta_payload,
    )

    logger.info(
        "Extraction enregistree: %s/%s/%s (%d tableaux)",
        bank_code,
        year,
        quarter_norm,
        len(tables),
    )
    return target_dir


def load_extraction(
    bank_code: str,
    year: int,
    quarter: str,
    base_dir: Path,
) -> tuple[list[TableArtifact], dict[str, Any]] | None:
    """Charger l'extraction d'un rapport.

    Lit uniquement le fichier canonique ``tables.json``. Verifie la version
    du schema et reconstruit les ``TableArtifact`` depuis les donnees stockees.

    Args:
        bank_code: Code identifiant la banque.
        year: Annee du rapport.
        quarter: Libelle du trimestre (ex. ``t1``).
        base_dir: Repertoire racine des extractions.

    Returns:
        Tuple (tables, meta) ou ``None`` si le fichier est absent, invalide
        ou d'une version de schema incompatible.
    """
    quarter_norm = _normalize_storage_quarter(quarter)
    paths = get_extraction_artifact_paths(bank_code, year, quarter_norm, base_dir)
    target_dir = paths["dir"]
    tables_path = paths["tables"]

    if not tables_path.exists():
        return None
    try:
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
        stored_version = payload.get("schema_version")
        try:
            if int(stored_version) != STORAGE_SCHEMA_VERSION:
                logger.debug(
                    "load_extraction rejected schema_version bank=%s year=%s quarter=%s stored=%s current=%s",
                    bank_code,
                    year,
                    quarter_norm,
                    stored_version,
                    STORAGE_SCHEMA_VERSION,
                )
                return None
        except (TypeError, ValueError):
            return None
        tables_data = payload.get("tables", [])
        if not isinstance(tables_data, list):
            return None
        tables = [table_artifact_from_dict(t) for t in tables_data]
        if not tables:
            logger.debug(
                "load_extraction empty tables bank=%s year=%s quarter=%s",
                bank_code,
                year,
                quarter_norm,
            )
            return None
        _backfill_page_local_structure(tables)
        meta: dict[str, Any] = {
            key: payload.get(key)
            for key in ("bank_code", "year", "quarter", "created_at", "schema_version")
        }
        _ensure_projection_artifacts(target_dir)
        return (tables, meta)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug("Load extraction failed %s: %s", target_dir, e)
        return None


def load_stored_extractions(
    bank_code: str,
    year: int,
    base_dir: Path,
) -> (
    tuple[list[TableArtifact], list[TableArtifact], dict[str, Any], dict[str, Any]]
    | None
):
    """Charger les extractions T1 et T2 si elles existent toutes les deux.

    Args:
        bank_code: Code identifiant la banque.
        year: Annee du rapport.
        base_dir: Repertoire racine des extractions.

    Returns:
        Tuple (tables_t1, tables_t2, meta_t1, meta_t2) ou ``None`` si l'une
        des deux est absente.
    """
    t1 = load_extraction(bank_code, year, "t1", base_dir)
    t2 = load_extraction(bank_code, year, "t2", base_dir)
    if t1 is None or t2 is None:
        return None
    tables_t1, meta_t1 = t1
    tables_t2, meta_t2 = t2
    return (tables_t1, tables_t2, meta_t1, meta_t2)
