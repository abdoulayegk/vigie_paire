"""Lecture et écriture des artefacts d'audit d'extraction texte."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEXT_EXTRACTION_AUDIT_SCHEMA_VERSION = 2


def get_canonical_text_extraction_audit_path(canonical_markdown_path: Path) -> Path:
    """Retourne l'audit JSON placé à côté du Markdown canonique."""
    return canonical_markdown_path.with_name("text_extraction.audit.json")


def write_text_extraction_audit(
    payload: dict[str, Any],
    out_path: Path,
) -> Path:
    """Écrit l'artefact d'audit d'extraction texte."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
