"""Écriture du fichier markdown d'extraction textuelle (source de vérité auditée)."""

from __future__ import annotations

import re
from pathlib import Path

CANONICAL_TEXT_EXTRACTIONS_DIR = "text_extractions"
TEXT_EXTRACTION_CACHE_SCHEMA_VERSION = 9
_CACHE_MARKER_PREFIX = "<!-- vigie-text-extraction-schema:"
_CACHE_MARKER_PATTERN = re.compile(
    r"^<!-- [a-z][a-z0-9-]*-text-extraction-schema:\s*(\d+)\s*-->",
    flags=re.IGNORECASE,
)


def _cache_marker() -> str:
    """Retourne le marqueur de version du Markdown canonique."""
    return f"{_CACHE_MARKER_PREFIX} {TEXT_EXTRACTION_CACHE_SCHEMA_VERSION} -->"


def has_current_text_extraction_cache_schema(content: str) -> bool:
    """Vérifie que le Markdown canonique correspond au schéma courant."""
    match = _CACHE_MARKER_PATTERN.match(str(content or "").lstrip())
    return bool(match and int(match.group(1)) == TEXT_EXTRACTION_CACHE_SCHEMA_VERSION)


def stamp_text_extraction_cache_schema(content: str) -> str:
    """Ajoute le marqueur de schéma au Markdown canonique réutilisable."""
    value = str(content or "").lstrip()
    match = _CACHE_MARKER_PATTERN.match(value)
    if match and int(match.group(1)) == TEXT_EXTRACTION_CACHE_SCHEMA_VERSION:
        body = value[match.end() :].lstrip("\r\n")
        return f"{_cache_marker()}\n\n{body}"
    return f"{_cache_marker()}\n\n{value}"


def get_text_extraction_markdown_path(out_dir: Path, quarter_label: str) -> Path:
    """Retourne le chemin du fichier markdown d'extraction pour un trimestre donné."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"text_extraction_{quarter_label.lower()}.md"


def get_canonical_text_extraction_md_path(
    project_root: Path,
    bank_code: str,
    year: int,
    quarter: str,
) -> Path:
    """Retourne le chemin canonique du .md d'extraction pour une période.

    Le .md vit dans ``outputs/text_extractions/{bank}/{year}/{q}/text_extraction.md``
    et est réutilisé par les runs suivants pour éviter de relancer Docling.
    Pour forcer une ré-extraction, supprimer ce fichier.
    """
    return (
        project_root
        / "outputs"
        / CANONICAL_TEXT_EXTRACTIONS_DIR
        / bank_code.lower()
        / str(year)
        / quarter.lower()
        / "text_extraction.md"
    )


def get_raw_docling_markdown_path(
    project_root: Path,
    bank_code: str,
    year: int,
    quarter: str,
    role: str,
) -> Path:
    """Retourne le chemin du markdown brut exporté directement par Docling."""
    normalized_role = str(role or "").strip().lower()
    if normalized_role not in {"current", "previous"}:
        raise ValueError("role must be 'current' or 'previous'")
    bank = bank_code.lower()
    quarter_norm = quarter.lower()
    return (
        project_root
        / "outputs"
        / CANONICAL_TEXT_EXTRACTIONS_DIR
        / bank
        / str(year)
        / quarter_norm
        / f"{bank}_{normalized_role}_{year}_{quarter_norm}.md"
    )


def write_text_extraction_markdown(content: str, out_path: Path) -> Path:
    """Écrit le contenu markdown dans le fichier de sortie et retourne le chemin."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path
