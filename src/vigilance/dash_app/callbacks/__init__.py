"""Callback modules for the Dash application."""

from __future__ import annotations


def register_all_callbacks() -> None:  # noqa: D401
    """Force-import every callback module so Dash picks up the decorators."""
    from vigilance.dash_app.callbacks import (  # noqa: F401
        dashboard_flow,
        export_flow,
        load_flow,
        proof_flow,
        review_flow,
        upload_flow,
        utility_flow,
    )
