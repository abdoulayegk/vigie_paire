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
