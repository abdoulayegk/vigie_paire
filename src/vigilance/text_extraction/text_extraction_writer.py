"""Lecture et écriture du fichier text_extraction.json."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .text_extractor import TextBlock

logger = logging.getLogger(__name__)

TEXT_EXTRACTION_SCHEMA_VERSION = 1


def get_text_extraction_path(
    out_root: Path,
    bank_code: str,
    year: int,
    quarter: str,
) -> Path:
    """Retourne le chemin canonique du fichier text_extraction.json.

    Pattern : out_root/{bank}/{year}/{quarter}/text_extraction.json
    Exemple  : outputs/text_extractions/bns/2025/t2/text_extraction.json
    """
    return out_root / bank_code.lower() / str(year) / quarter.lower() / "text_extraction.json"


def write_text_extraction(
    blocks: list[TextBlock],
    out_dir: Path,
    bank_code: str,
    year: int,
    quarter: str,
    source_pdf: str,
) -> Path:
    """Sérialise et écrit text_extraction.json dans out_dir.

    Args:
        blocks: Liste de TextBlock extraits.
        out_dir: Répertoire de sortie (créé si nécessaire).
        bank_code: Code banque (ex. "bns").
        year: Année du rapport.
        quarter: Trimestre normalisé (ex. "t2").
        source_pdf: Chemin du PDF source.

    Returns:
        Path du fichier écrit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "text_extraction.json"

    payload: dict[str, Any] = {
        "schema_version": TEXT_EXTRACTION_SCHEMA_VERSION,
        "bank_code": bank_code.lower(),
        "year": year,
        "quarter": quarter.lower(),
        "source_pdf": str(source_pdf),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "total_blocks": len(blocks),
        "blocks": [b.to_dict() for b in blocks],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("text_extraction.json écrit : %s (%d blocs)", out_path, len(blocks))
    return out_path


def load_text_extraction(extraction_path: Path) -> dict[str, Any]:
    """Charge text_extraction.json et valide le schema_version.

    Args:
        extraction_path: Chemin vers text_extraction.json.

    Returns:
        Dictionnaire chargé.

    Raises:
        FileNotFoundError: Si le fichier est absent.
        ValueError: Si le schema_version est incompatible.
    """
    if not extraction_path.exists():
        raise FileNotFoundError(f"text_extraction.json introuvable : {extraction_path}")

    data = json.loads(extraction_path.read_text(encoding="utf-8"))

    version = data.get("schema_version")
    if version != TEXT_EXTRACTION_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version incompatible : attendu {TEXT_EXTRACTION_SCHEMA_VERSION}, trouvé {version} "
            f"dans {extraction_path}"
        )
    return data
