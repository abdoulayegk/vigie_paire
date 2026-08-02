"""Génération des prompts et modèles Pydantic pour l'extraction Vision GPT-4o.

Ce module centralise les instructions système OpenAI Vision, la modélisation Pydantic
des structures d'indicateurs et la construction des requêtes d'extraction d'images.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VisionFootnote(BaseModel):
    """Note de bas de page extraite par Vision."""

    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(description="Symbole ou numéro de la note (ex: '(1)', '*')")
    text: str = Field(description="Texte complet de la note de bas de page")


class VisionIndicator(BaseModel):
    """Indicateur (ligne de tableau) extrait par Vision."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Libellé exact de l'indicateur ou du poste")
    values: list[str] = Field(default_factory=list, description="Valeurs chiffrées par colonne")


class VisionTableExtraction(BaseModel):
    """Structure d'extraction complète d'un tableau bancaire par Vision."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(description="Titre exact du tableau")
    section: str = Field(default="inconnue", description="Nom de la section ou sous-section")
    headers: list[str] = Field(default_factory=list, description="En-têtes de colonnes")
    indicators: list[VisionIndicator] = Field(default_factory=list, description="Liste des indicateurs")
    footnotes: list[VisionFootnote] = Field(default_factory=list, description="Notes de bas de page")


_PROMPT_BASE = """
You are a financial table extraction engine for Canadian bank quarterly reports (French language).
Extract the structured table content from the provided image crop accurately and completely.
"""


def build_vision_user_prompt(
    *,
    page_number: int,
    section_context: str = "",
    reference_text: str = "",
) -> str:
    """Construit le prompt utilisateur pour l'extraction Vision."""
    parts = [f"Page number: {page_number}"]
    if section_context:
        parts.append(f"Section context: {section_context}")
    if reference_text:
        parts.append(f"Reference OCR text:\n{reference_text}")
    return "\n".join(parts)
