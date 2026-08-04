"""Utilitaires d'entree/sortie pour les workflows Dash."""

from __future__ import annotations

import json
from pathlib import Path

from vigie.interface.ui_config import INDICATOR_COMPARISON_DIR


def save_pdfs_to_temp(
    pdf_bytes_t1: bytes, pdf_bytes_t2: bytes, *, temp_dir: Path
) -> tuple[str, str]:
    """Persiste les bytes PDF uploades dans un repertoire temporaire.

    Args:
        pdf_bytes_t1: Contenu binaire du PDF du trimestre precedent.
        pdf_bytes_t2: Contenu binaire du PDF du trimestre courant.
        temp_dir: Repertoire temporaire de travail.

    Returns:
        Tuple ``(chemin_t1, chemin_t2)`` des fichiers sauvegardes.

    Raises:
        ValueError: Si l'un des deux PDF est vide.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_bytes_t1 or not pdf_bytes_t2:
        raise ValueError(
            "Les deux fichiers PDF (trimestre precedent et trimestre courant) sont requis."
        )

    path_t1 = temp_dir / "previous_upload.pdf"
    path_t2 = temp_dir / "current_upload.pdf"

    path_t1.write_bytes(pdf_bytes_t1)
    path_t2.write_bytes(pdf_bytes_t2)

    return str(path_t1), str(path_t2)


def load_comparison_result(path: str | Path) -> dict | None:
    """Charge un payload JSON de comparaison depuis le disque.

    Args:
        path: Chemin du fichier JSON.

    Returns:
        Dictionnaire du payload, ou ``None`` si absent/invalide.
    """
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_available_indicator_comparison_options() -> list[dict[str, str]]:
    """Retourne les options de dropdown pour les fichiers JSON de comparaison disponibles."""
    if not INDICATOR_COMPARISON_DIR.exists():
        return []

    files = sorted(
        INDICATOR_COMPARISON_DIR.glob("**/*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    options: list[dict[str, str]] = []
    for path in files:
        if path.name != "comparison.json":
            continue
        try:
            rel = path.relative_to(INDICATOR_COMPARISON_DIR)
        except ValueError:
            rel = path
        rel_value = rel.as_posix()
        options.append({"label": rel_value, "value": rel_value})
    return options
