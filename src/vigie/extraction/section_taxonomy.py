"""Fonctions utilitaires pour la taxonomie canonique des sections."""

from __future__ import annotations

import re
import unicodedata


def _normalize_text(raw: str) -> str:
    """Normaliser le texte brut (accents, casse, caracteres speciaux)."""
    text = unicodedata.normalize("NFD", raw or "")
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split()).strip()


def _to_snake_case(raw: str) -> str:
    """Convertir le texte normalise en snake_case."""
    normalized = _normalize_text(raw)
    if not normalized:
        return ""
    return normalized.replace(" ", "_")


def canonicalize_section(raw: str) -> str:
    """Associer les libelles de section a leurs valeurs canoniques.

    Args:
        raw: Libelle brut de la section.

    Returns:
        Valeur canonique (ex. ``"capital_management"``, ``"risk_management"``).
    """
    snake = _to_snake_case(raw)
    spaced = snake.replace("_", " ")

    if snake in {"capital_management", "risk_management", "regulatory_updates"}:
        return snake

    if (
        any(
            token in spaced
            for token in (
                "gestion capital",
                "gestion du capital",
                "gestion fonds propres",
                "gestion des fonds propres",
                "situation des fonds propres",
                "fonds propres",
                "capitalisation",
            )
        )
        or snake == "gestion_capital"
    ):
        return "capital_management"

    if (
        any(
            token in spaced
            for token in (
                "gestion risques",
                "gestion des risques",
                "gestion du risque",
                "risk management",
                "risque",
            )
        )
        or snake == "gestion_risques"
    ):
        return "risk_management"

    if (
        any(
            token in spaced
            for token in (
                "faits nouveaux en matiere de reglementation",
                "reglementation",
                "regulatory updates",
                "regulatory update",
            )
        )
        or snake == "gestion_reglementation"
    ):
        return "regulatory_updates"

    return snake
