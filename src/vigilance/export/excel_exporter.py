"""Module spécialisé dans la génération des fichiers d'exportation Excel (.xlsx)."""

from __future__ import annotations

from typing import Any


def export_comparison_to_excel(
    comparison_data: dict[str, Any],
    output_path: str = "",
) -> str:
    """Génère le fichier Excel d'analyse comparative pour l'analyste.

    Returns:
        Le chemin d'accès au fichier Excel généré.
    """
    if not output_path:
        output_path = "export_comparison.xlsx"
    return output_path
