"""Package des sous-composants réutilisables de détails Dash (proof badges, proof cards)."""

from __future__ import annotations

from vigilance.dash_app.components.detail_widgets.proof_badge_builder import build_proof_badges
from vigilance.dash_app.components.detail_widgets.proof_card_builder import build_comparison_proof_card

__all__ = [
    "build_proof_badges",
    "build_comparison_proof_card",
]
