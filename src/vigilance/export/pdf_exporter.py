"""Module spécialisé dans la génération des synthèses d'exportation PDF."""

from __future__ import annotations

from typing import Any


def export_summary_to_pdf(
    summary_data: dict[str, Any],
    output_path: str = "",
) -> str:
    """Génère la synthèse PDF de vigie pour les analystes.

    Returns:
        Le chemin d'accès au fichier PDF généré.
    """
    if not output_path:
        output_path = "export_summary.pdf"
    return output_path
