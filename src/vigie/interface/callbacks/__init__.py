"""Modules de callbacks pour l'application Dash."""

from __future__ import annotations


def register_all_callbacks() -> None:  # noqa: D401
    """Force l'import de chaque module de callback pour que Dash enregistre les decorateurs."""
    from vigie.interface.callbacks import (  # noqa: F401, PLC0415 - enregistrement tardif des decorateurs Dash
        changements_communs_flow,
        dashboard_flow,
        export_flow,
        load_flow,
        proof_flow,
        review_flow,
        text_flow,
        upload_flow,
        utility_flow,
    )
