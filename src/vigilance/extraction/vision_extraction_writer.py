"""Ecrivains pour le contrat d'extraction canonique et les artefacts de revue derives."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..extraction.section_taxonomy import canonicalize_section
from ..models.table_models import (
    get_canonical_footnotes,
    get_comparison_indicators,
    get_vision_raw_indicators,
)
from ..utils.footnotes_utils import (
    count_stringified_dict_suspects,
    footnotes_list_to_dict,
)

logger = logging.getLogger(__name__)

COMPACT_REPORT_SCHEMA_VERSION = 7

def _compact_table_section(table: Any) -> str:
    """Retourner la valeur de section canonique compacte avec fallback defensif."""
    raw = str(getattr(table, "section", "") or "").strip()
    if not raw:
        return "unknown_section"
    try:
        normalized = canonicalize_section(raw)
    except Exception:
        normalized = raw
    return str(normalized or "unknown_section").strip() or "unknown_section"

def _table_page(table: Any) -> int:
    """Retourner le numero de page PDF pour un objet de type tableau."""
    return int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)


def _compact_table_title(table: Any) -> str:
    """Retourner la valeur de titre compacte ; chaine vide si aucun titre n'existe."""
    title = getattr(table, "title_clean", None)
    if title is None:
        title = getattr(table, "title", None)
    return str(title or "")


def _compact_created_at(meta: dict[str, Any] | None = None) -> str:
    """Retourner le timestamp de creation depuis les metadonnees ou l'horodatage courant."""
    metadata = dict(meta or {})
    return str(
        metadata.get("created_at")
        or metadata.get("extracted_at")
        or datetime.now().isoformat(timespec="seconds")
    )


def _compact_top_level(
    *,
    bank_code: str,
    year: int,
    quarter: str,
    created_at: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construire le dictionnaire de niveau superieur pour le JSON compact."""
    return {
        "bank_code": str(bank_code),
        "year": int(year),
        "quarter": str(quarter),
        "created_at": str(created_at),
        "schema_version": COMPACT_REPORT_SCHEMA_VERSION,
    }


def _stable_sort_tables(tables: list[Any]) -> list[Any]:
    """Trier les tableaux de maniere stable par page puis par index sur la page."""
    indexed = list(enumerate(tables))
    indexed.sort(
        key=lambda item: (
            _table_page(item[1]),
            int(getattr(item[1], "table_index_on_page", 10**9) or 10**9),
            item[0],
        )
    )
    return [table for _, table in indexed]


def _validate_unique_table_ids(tables: list[Any]) -> None:
    """Valider l'unicite des ``table_id`` ; lever ``ValueError`` en cas de doublon."""
    seen: dict[str, tuple[int, int]] = {}
    for index, table in enumerate(tables):
        table_id = str(getattr(table, "table_id", "") or "").strip()
        page = _table_page(table)
        if not table_id:
            raise ValueError(
                f"Empty table_id at tables[{index}] on page {page}; table_id must be non-empty and unique within tables.json"
            )
        previous = seen.get(table_id)
        if previous is not None:
            previous_index, previous_page = previous
            raise ValueError(
                "Duplicate table_id in tables.json: "
                f"{table_id!r} at tables[{previous_index}] page {previous_page} "
                f"and tables[{index}] page {page}. "
                "Multiple tables may share a page, but table_id must remain unique."
            )
        seen[table_id] = (index, page)


def _compact_footnotes(table: Any) -> list[dict[str, str]]:
    """Extraire les notes de bas de tableau sous forme compacte."""
    return [
        {
            "id": str(item.get("id") or "").strip(),
            "text": str(item.get("text") or "").strip(),
        }
        for item in get_canonical_footnotes(table)
        if str(item.get("id") or "").strip() and str(item.get("text") or "").strip()
    ]


def _compact_indicator_rows(table: Any) -> list[str]:
    """Retourner les lignes d'indicateurs bruts minimaux pour les exports compacts."""
    indicators = get_vision_raw_indicators(table)
    if not indicators:
        indicators = get_comparison_indicators(table)
    return [str(value).strip() for value in indicators if str(value).strip()]


def _compact_table_bbox(table: Any) -> list[float] | None:
    """Retourner le bbox normalise ``[l, t, r, b]`` si valide, sinon ``None``."""
    bbox = getattr(table, "bbox", None)
    try:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            normalized = [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            ]
        elif isinstance(bbox, dict):
            if {"l", "t", "r", "b"}.issubset(bbox):
                normalized = [
                    float(bbox["l"]),
                    float(bbox["t"]),
                    float(bbox["r"]),
                    float(bbox["b"]),
                ]
            elif {"x0", "y0", "x1"}.issubset(bbox):
                y1 = bbox.get("y1")
                if y1 is None:
                    return None
                normalized = [
                    float(bbox["x0"]),
                    float(bbox["y0"]),
                    float(bbox["x1"]),
                    float(y1),
                ]
            else:
                return None
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None

    l_, top, r, bottom = normalized
    if not (
        0.0 <= l_ <= 1.0
        and 0.0 <= top <= 1.0
        and 0.0 <= r <= 1.0
        and 0.0 <= bottom <= 1.0
    ):
        return None
    if r <= l_ or bottom <= top:
        return None
    return normalized


def _compact_table_common_entry(table: Any) -> dict[str, Any]:
    """Construire l'entree commune (id, page, section, titre, bbox) d'un tableau."""
    return {
        "table_id": str(getattr(table, "table_id", "") or ""),
        "page": _table_page(table),
        "section": _compact_table_section(table),
        "title": _compact_table_title(table),
        "bbox": _compact_table_bbox(table),
    }


def _compact_table_entry(table: Any) -> dict[str, Any]:
    """Construire l'entree complete d'un tableau pour ``tables.json``."""
    entry = _compact_table_common_entry(table)
    entry["extraction_status"] = str(
        getattr(table, "extraction_status", "") or "ok"
    ).strip() or "ok"
    entry["headers"] = [
        str(value) for value in list(getattr(table, "headers", []) or [])
    ]
    indicators = _compact_indicator_rows(table)
    entry["table_summary"] = str(getattr(table, "table_summary", "") or "")
    entry["row_count"] = len(indicators)
    entry["indicators"] = indicators
    entry["footnotes"] = _compact_footnotes(table)
    debug_metrics = getattr(table, "debug_metrics", None)
    if isinstance(debug_metrics, dict):
        provenance_keys = (
            "bbox_original",
            "bbox_final",
            "bbox_source",
            "bbox_confidence",
            "bbox_verified",
            "page_context_title",
            "page_context_continuation",
            "page_context_table_count",
            "locator_merge_collapsed",
            "locator_merged_table_ids",
            "locator_original_bboxes",
            "bbox_verification_reason",
        )
        bbox_provenance = {
            key: debug_metrics[key]
            for key in provenance_keys
            if key in debug_metrics and debug_metrics[key] is not None
        }
        if bbox_provenance:
            entry["bbox_provenance"] = bbox_provenance
    return entry


def _compact_indicator_entry(table_entry: dict[str, Any]) -> dict[str, Any]:
    """Construire l'entree indicateurs a partir d'une entree de tableau."""
    return {
        "table_id": str(table_entry.get("table_id", "") or ""),
        "page": int(table_entry.get("page", 0) or 0),
        "section": str(table_entry.get("section", "") or "unknown_section"),
        "title": str(table_entry.get("title", "") or ""),
        "indicators": [
            str(value).strip()
            for value in list(table_entry.get("indicators", []) or [])
            if str(value).strip()
        ],
    }


def _compact_footnote_entry(table_entry: dict[str, Any]) -> dict[str, Any]:
    """Construire l'entree notes de bas de page a partir d'une entree de tableau."""
    return {
        "table_id": str(table_entry.get("table_id", "") or ""),
        "page": int(table_entry.get("page", 0) or 0),
        "section": str(table_entry.get("section", "") or "unknown_section"),
        "footnotes": [
            {
                "id": str(item.get("id", "") or "").strip(),
                "text": str(item.get("text", "") or "").strip(),
            }
            for item in list(table_entry.get("footnotes", []) or [])
            if isinstance(item, dict)
            and (
                str(item.get("id", "") or "").strip()
                or str(item.get("text", "") or "").strip()
            )
        ],
    }


def _atomic_write_json(out_path: Path, payload: dict[str, Any]) -> Path:
    """Ecrire un fichier JSON de maniere atomique (via fichier temporaire)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(out_path)
    return out_path

def _table_entry_indicators(
    table: Any,
    source: str,
) -> dict[str, Any]:
    """Construire l'entree indicateurs pour un tableau."""
    table_id = str(getattr(table, "table_id", "") or "")
    title = getattr(table, "title_clean", None) or getattr(table, "title", None) or ""
    page = int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)
    indicators = get_comparison_indicators(table)
    indicators_raw = get_vision_raw_indicators(table) or indicators
    unit_context = getattr(table, "unit_context", None) or ""

    sections: list[dict[str, Any]] = []
    if indicators_raw:
        sections.append(
            {
                "section": title or "Indicateurs",
                "indicators": [
                    str(x).strip() for x in indicators_raw if str(x).strip()
                ],
            }
        )

    return {
        "table_id": table_id,
        "title": str(title),
        "date_reference": str(unit_context),
        "page": page,
        "source": source,
        "sections": sections,
    }


def _table_entry_footnotes(
    table: Any,
    source: str,
) -> dict[str, Any]:
    """Construire l'entree notes de bas de page pour un tableau."""
    table_id = str(getattr(table, "table_id", "") or "")
    title = getattr(table, "title_clean", None) or getattr(table, "title", None) or ""
    page = int(getattr(table, "page_pdf", 0) or getattr(table, "page_number", 0) or 0)
    footnotes_source = getattr(table, "footnotes", None) or []
    footnotes_raw = get_canonical_footnotes(table)
    fn_dict = footnotes_list_to_dict(footnotes_raw)
    footnote_markers = list(fn_dict.keys())
    repr_suspects = count_stringified_dict_suspects(footnotes_source)

    return {
        "table_id": table_id,
        "title": str(title),
        "page": page,
        "source": source,
        "has_footnotes": bool(fn_dict),
        "footnote_markers": footnote_markers,
        "footnotes_content": fn_dict,
        "_repr_suspect_count": int(repr_suspects),
    }


def write_indicators_json(
    tables_t1: list[Any],
    tables_t2: list[Any],
    out_dir: Path,
    bank_code: str,
    run_id: str,
) -> Path:
    """Ecrire ``indicators.json`` pour l'audit.

    Chaque entree contient : table_id, title, date_reference, page, source
    (T1/T2), sections.

    Args:
        tables_t1: Tableaux du trimestre T1.
        tables_t2: Tableaux du trimestre T2.
        out_dir: Repertoire de sortie.
        bank_code: Code de la banque.
        run_id: Identifiant de l'execution.

    Returns:
        Chemin du fichier ``indicators.json`` ecrit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for t in tables_t1:
        entries.append(_table_entry_indicators(t, "t1"))
    for t in tables_t2:
        entries.append(_table_entry_indicators(t, "t2"))

    payload: dict[str, Any] = {
        "bank_code": bank_code,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "artifact_role": "audit_only",
        "authoritative_source": "table_artifacts_for_comparison",
        "tables": entries,
    }

    out_path = out_dir / "indicators.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Wrote indicators.json to %s", out_path)
    return out_path


def write_footnotes_json(
    tables_t1: list[Any],
    tables_t2: list[Any],
    out_dir: Path,
    bank_code: str,
    run_id: str,
) -> Path:
    """Ecrire ``footnotes.json`` pour l'audit.

    Chaque entree contient : table_id, title, page, source (T1/T2),
    has_footnotes, footnote_markers, footnotes_content.

    Args:
        tables_t1: Tableaux du trimestre T1.
        tables_t2: Tableaux du trimestre T2.
        out_dir: Repertoire de sortie.
        bank_code: Code de la banque.
        run_id: Identifiant de l'execution.

    Returns:
        Chemin du fichier ``footnotes.json`` ecrit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for t in tables_t1:
        entries.append(_table_entry_footnotes(t, "t1"))
    for t in tables_t2:
        entries.append(_table_entry_footnotes(t, "t2"))

    tables_total = len(entries)
    tables_with_footnotes = sum(1 for e in entries if bool(e.get("has_footnotes")))
    footnote_entries_total = sum(
        len(e.get("footnotes_content", {}) or {}) for e in entries
    )
    repr_suspect_count = sum(int(e.get("_repr_suspect_count", 0) or 0) for e in entries)

    for entry in entries:
        entry.pop("_repr_suspect_count", None)

    payload: dict[str, Any] = {
        "bank_code": bank_code,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "artifact_role": "audit_only",
        "authoritative_source": "table_artifacts_for_comparison",
        "meta": {
            "tables_total": tables_total,
            "tables_with_footnotes": tables_with_footnotes,
            "footnote_entries_total": footnote_entries_total,
            "repr_suspect_count": repr_suspect_count,
        },
        "tables": entries,
    }
    if repr_suspect_count > 0:
        payload["warnings"] = [
            {
                "code": "repr_suspect_detected",
                "message": "Stringified-dict footnotes detected in source payload.",
                "count": repr_suspect_count,
            }
        ]

    out_path = out_dir / "footnotes.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Wrote footnotes.json to %s", out_path)
    return out_path


def write_compact_tables_json(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Ecrire le ``tables.json`` canonique compact utilise par le flux d'extraction minimal.

    Args:
        tables: Liste d'objets tableaux.
        out_dir: Repertoire de sortie.
        bank_code: Code de la banque.
        year: Annee du rapport.
        quarter: Trimestre (ex. ``"t2"``).
        meta: Metadonnees optionnelles (``created_at``, etc.).

    Returns:
        Chemin du fichier ``tables.json`` ecrit.
    """
    created_at = _compact_created_at(meta)
    ordered_tables = _stable_sort_tables(tables)
    _validate_unique_table_ids(ordered_tables)
    payload: dict[str, Any] = {
        **_compact_top_level(
            bank_code=bank_code,
            year=year,
            quarter=quarter,
            created_at=created_at,
            meta=meta,
        ),
        "tables": [_compact_table_entry(table) for table in ordered_tables],
    }
    out_path = out_dir / "tables.json"
    _atomic_write_json(out_path, payload)
    logger.debug("Wrote compact tables.json to %s", out_path)
    return out_path


def _load_tables_payload(tables_json_path: Path) -> dict[str, Any]:
    """Charger le payload ``tables.json`` depuis le disque."""
    payload = json.loads(tables_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tables.json payload at {tables_json_path}")
    return payload


def _projection_top_level_from_tables(payload: dict[str, Any]) -> dict[str, Any]:
    """Extraire le dictionnaire de niveau superieur depuis un payload ``tables.json``."""
    return {
        "bank_code": str(payload.get("bank_code", "") or ""),
        "year": int(payload.get("year", 0) or 0),
        "quarter": str(payload.get("quarter", "") or ""),
        "created_at": str(payload.get("created_at", "") or ""),
        "schema_version": int(payload.get("schema_version", COMPACT_REPORT_SCHEMA_VERSION) or COMPACT_REPORT_SCHEMA_VERSION),
    }


def write_compact_indicators_json(
    tables_json_path: Path,
    out_dir: Path,
) -> Path:
    """Ecrire ``indicators.json`` comme projection directe de ``tables.json``.

    Args:
        tables_json_path: Chemin vers le fichier ``tables.json`` source.
        out_dir: Repertoire de sortie.

    Returns:
        Chemin du fichier ``indicators.json`` ecrit.
    """
    tables_payload = _load_tables_payload(tables_json_path)
    payload: dict[str, Any] = {
        **_projection_top_level_from_tables(tables_payload),
        "tables": [
            _compact_indicator_entry(table)
            for table in list(tables_payload.get("tables", []) or [])
            if isinstance(table, dict)
        ],
    }
    out_path = out_dir / "indicators.json"
    _atomic_write_json(out_path, payload)
    logger.debug("Wrote compact indicators.json to %s", out_path)
    return out_path


def write_compact_footnotes_json(
    tables_json_path: Path,
    out_dir: Path,
) -> Path:
    """Ecrire ``footnotes.json`` comme projection directe de ``tables.json``.

    Args:
        tables_json_path: Chemin vers le fichier ``tables.json`` source.
        out_dir: Repertoire de sortie.

    Returns:
        Chemin du fichier ``footnotes.json`` ecrit.
    """
    tables_payload = _load_tables_payload(tables_json_path)
    payload: dict[str, Any] = {
        **_projection_top_level_from_tables(tables_payload),
        "tables": [
            _compact_footnote_entry(table)
            for table in list(tables_payload.get("tables", []) or [])
            if isinstance(table, dict)
        ],
    }
    out_path = out_dir / "footnotes.json"
    _atomic_write_json(out_path, payload)
    logger.debug("Wrote compact footnotes.json to %s", out_path)
    return out_path


def write_compact_report_artifacts(
    tables: list[Any],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Ecrire ``tables.json`` puis en deriver ``indicators.json`` et ``footnotes.json``.

    Args:
        tables: Liste d'objets tableaux.
        out_dir: Repertoire de sortie.
        bank_code: Code de la banque.
        year: Annee du rapport.
        quarter: Trimestre (ex. ``"t2"``).
        meta: Metadonnees optionnelles.

    Returns:
        Dictionnaire ``{"tables": Path, "indicators": Path, "footnotes": Path}``.
    """
    metadata = dict(meta or {})
    metadata["created_at"] = _compact_created_at(metadata)
    tables_path = write_compact_tables_json(
        tables, out_dir, bank_code, year, quarter, meta=metadata
    )
    return {
        "tables": tables_path,
        "indicators": write_compact_indicators_json(tables_path, out_dir),
        "footnotes": write_compact_footnotes_json(tables_path, out_dir),
    }
