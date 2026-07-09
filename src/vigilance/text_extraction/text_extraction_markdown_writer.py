"""Écriture du fichier markdown d'extraction textuelle (source de vérité auditée)."""

from __future__ import annotations

from pathlib import Path

CANONICAL_TEXT_EXTRACTIONS_DIR = "text_extractions"


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
