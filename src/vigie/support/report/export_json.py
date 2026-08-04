"""Utilitaires d'export JSON pour les plages de sections et les sorties d'extraction de tableaux."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigie.support.models.section_models import SectionRangesResult
from vigie.support.models.table_models import TableArtifact


def _utc_now_iso() -> str:
    """Retourne l'horodatage UTC courant au format ISO."""
    return datetime.now(timezone.utc).isoformat()


def _prepare_out_dir(out_dir: str | Path) -> Path:
    """Cree le repertoire de sortie s'il n'existe pas et retourne son chemin."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_section_ranges(out_dir: str | Path, result: SectionRangesResult) -> Path:
    """Ecrit ``section_ranges.json`` dans *out_dir* et retourne son chemin.

    Args:
        out_dir: Repertoire de sortie.
        result: Resultat de detection des plages de sections.

    Returns:
        Chemin du fichier JSON ecrit.
    """
    target_dir = _prepare_out_dir(out_dir)
    out_path = target_dir / "section_ranges.json"
    payload: dict[str, Any] = {
        "metadata": {
            "bank_code": result.bank_code,
            "quarter": result.quarter,
            "pdf_path": result.pdf_path,
            "created_at": _utc_now_iso(),
        },
        "ranges": [item.to_dict() for item in result.ranges],
        "skipped_pages": result.skipped_pages,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def write_tables_docling(out_dir: str | Path, tables: list[TableArtifact]) -> Path:
    """Ecrit ``tables_docling.json`` dans *out_dir* et retourne son chemin.

    Args:
        out_dir: Repertoire de sortie.
        tables: Liste d'artefacts de tableau a serialiser.

    Returns:
        Chemin du fichier JSON ecrit.
    """
    target_dir = _prepare_out_dir(out_dir)
    out_path = target_dir / "tables_docling.json"
    first = tables[0] if tables else None
    payload: dict[str, Any] = {
        "metadata": {
            "bank_code": first.bank_code if first else "",
            "quarter": first.quarter if first else "",
            "pdf_path": first.pdf_path if first else "",
            "created_at": _utc_now_iso(),
        },
        "tables": [table.to_dict() for table in tables],
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path
