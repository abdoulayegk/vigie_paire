"""Lecture et écriture du fichier text_comparison.json."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEXT_COMPARISON_SCHEMA_VERSION = 3
_ACCEPTED_TEXT_COMPARISON_SCHEMAS = {TEXT_COMPARISON_SCHEMA_VERSION}
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text_value(value: Any) -> Any:
    """Retire les caractères de contrôle illégaux pour JSON/Dash/Excel."""
    if isinstance(value, str):
        return _CONTROL_CHAR_RE.sub("", value)
    if isinstance(value, list):
        return [_sanitize_text_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_text_value(item) for key, item in value.items()}
    return value


def get_text_comparison_path(
    out_root: Path,
    bank_code: str,
    year_t2: int,
    quarter_t2: str,
    year_t1: int,
    quarter_t1: str,
) -> Path:
    """Retourne le chemin canonique du fichier text_comparison.json.

    Pattern : out_root/{bank}/{year_t2}_{qt2}_vs_{year_t1}_{qt1}/text_comparison.json
    Exemple  : outputs/resultats/bns/2025_t2_vs_2025_t1/text_comparison.json
    """
    folder = f"{year_t2}_{quarter_t2.lower()}_vs_{year_t1}_{quarter_t1.lower()}"
    return out_root / bank_code.lower() / folder / "text_comparison.json"


def write_text_comparison(
    payload: dict[str, Any],
    out_path: Path,
) -> Path:
    """Sérialise et écrit text_comparison.json.

    Args:
        payload: Dictionnaire texte canonique.
        out_path: Chemin complet du fichier de sortie.

    Returns:
        Path du fichier écrit.
    """
    sanitized_payload = _sanitize_text_value(payload)
    if isinstance(sanitized_payload, dict):
        payload.clear()
        payload.update(sanitized_payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total_changes = sum(
        len(sc.get("block_comparisons", []))
        for sc in payload.get("section_comparisons", [])
    )
    logger.info(
        "text_comparison.json écrit : %s (%d sections, %d changements)",
        out_path,
        len(payload.get("section_comparisons", [])),
        total_changes,
    )
    return out_path


def load_text_comparison(comparison_path: Path) -> dict[str, Any]:
    """Charge text_comparison.json et valide le schema_version.

    Args:
        comparison_path: Chemin vers text_comparison.json.

    Returns:
        Dictionnaire chargé.

    Raises:
        FileNotFoundError: Si le fichier est absent.
        ValueError: Si le schema_version est incompatible.
    """
    if not comparison_path.exists():
        raise FileNotFoundError(f"text_comparison.json introuvable : {comparison_path}")

    data = json.loads(comparison_path.read_text(encoding="utf-8"))

    version = data.get("schema_version")
    if version not in _ACCEPTED_TEXT_COMPARISON_SCHEMAS:
        raise ValueError(
            f"schema_version incompatible : accepté {_ACCEPTED_TEXT_COMPARISON_SCHEMAS}, "
            f"trouvé {version} dans {comparison_path}"
        )
    return data
