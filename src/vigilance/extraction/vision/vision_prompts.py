"""Module spécialisé dans la construction des prompts pour GPT-4o Vision."""

from __future__ import annotations

from typing import Any


def build_vision_system_prompt() -> str:
    """Construit le prompt système pour l'extraction de tableaux bancaires par Vision."""
    return (
        "Vous êtes un expert en extraction de données financières à partir d'images de documents PDF. "
        "Votre rôle est d'extraire fidèlement le titre, les en-têtes, les indicateurs chiffrés et les notes de bas de page."
    )


def build_vision_user_prompt(bank_code: str = "", context: str = "") -> str:
    """Construit le prompt utilisateur personnalisé pour l'image fournie."""
    prompt = "Extraire toutes les données structurées du tableau sur cette image."
    if bank_code:
        prompt += f" Document de la banque: {bank_code.upper()}."
    if context:
        prompt += f" Contexte: {context}."
    return prompt
