"""Utilitaires de construction de l'arborescence de sortie standardisee.

Architecture « Extract Once, Reference Everywhere » :
- Les extractions vivent dans ``outputs/extractions/{bank}/{year}/{quarter}/``.
- Les comparaisons vivent dans ``outputs/comparisons/{bank}/{label}/``.
- Le manifest contient des references vers les extractions et les PDFs sources.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_run_dir(
    out_root: Path,
    bank: str,
    year_current: int,
    quarter_current: str,
    year_previous: int,
    quarter_previous: str,
) -> Path:
    """Creer et retourner le repertoire d'execution (Run).

    Arborescence simplifiee (pas de sous-dossiers par trimestre)::

        {out_root}/{BANK}_{YEAR}{Q_CUR}_vs_{YEAR}{Q_PREV}/
            comparison.json
            manifest.json
    """
    run_name = f"{bank.upper()}_{year_current}{quarter_current.upper()}_vs_{year_previous}{quarter_previous.upper()}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def split_audit_files(tables_json_path: Path, target_dir: Path) -> dict[str, Path]:
    """Lire un ``tables.json`` et le fractionner en fichiers d'audit separes.

    Produit :
    - ``indicators.json`` -- liste plate des indicateurs par table
    - ``footnotes.json`` -- liste plate des notes de bas de page par table

    Returns:
        Dictionnaire associant le type de fichier au chemin ecrit.
    """
    with open(tables_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    tables: list[dict[str, Any]] = []
    if isinstance(data, dict):
        tables = data.get("tables", data.get("extracted_tables", []))
    elif isinstance(data, list):
        tables = data

    indicators_out: list[dict[str, Any]] = []
    footnotes_out: list[dict[str, Any]] = []

    for tbl in tables:
        table_id = tbl.get("table_id", tbl.get("id", "unknown"))
        title = tbl.get("table_title", tbl.get("title", ""))

        inds = tbl.get("indicators", [])
        indicators_out.append(
            {
                "table_id": table_id,
                "title": title,
                "indicators": inds,
            }
        )

        fns = tbl.get("footnotes_content", tbl.get("footnotes", []))
        footnotes_out.append(
            {
                "table_id": table_id,
                "title": title,
                "footnotes": fns,
            }
        )

    paths: dict[str, Path] = {}

    ind_path = target_dir / "indicators.json"
    with open(ind_path, "w", encoding="utf-8") as fh:
        json.dump(indicators_out, fh, ensure_ascii=False, indent=2)
    paths["indicators"] = ind_path

    fn_path = target_dir / "footnotes.json"
    with open(fn_path, "w", encoding="utf-8") as fh:
        json.dump(footnotes_out, fh, ensure_ascii=False, indent=2)
    paths["footnotes"] = fn_path

    return paths


def write_run_manifest(
    run_dir: Path,
    *,
    bank: str,
    year_current: int,
    quarter_current: str,
    year_previous: int,
    quarter_previous: str,
    status: str = "completed",
    extraction_dir_current: str | Path | None = None,
    extraction_dir_previous: str | Path | None = None,
    pdf_path_current: str | Path | None = None,
    pdf_path_previous: str | Path | None = None,
) -> Path:
    """Ecrire un fichier ``manifest.json`` a la racine du repertoire d'execution.

    Le manifest reference les extractions et PDFs sources au lieu de les copier.
    """
    manifest: dict[str, Any] = {
        "bank": bank.upper(),
        "current": {
            "year": year_current,
            "quarter": quarter_current.upper(),
            "extraction_path": str(extraction_dir_current or ""),
            "pdf_path": str(pdf_path_current or ""),
        },
        "previous": {
            "year": year_previous,
            "quarter": quarter_previous.upper(),
            "extraction_path": str(extraction_dir_previous or ""),
            "pdf_path": str(pdf_path_previous or ""),
        },
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return path
