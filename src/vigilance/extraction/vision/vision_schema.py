"""Module spécialisé dans la définition des schémas Pydantic pour les réponses GPT-4o Vision."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class VisionIndicatorItem(BaseModel):
    """Représente une ligne / indicateur chiffré extrait d'une image de tableau."""

    label: str = Field(default="", description="Intitulé de l'indicateur")
    value: str = Field(default="", description="Valeur numérique ou textuelle")


class VisionFootnoteItem(BaseModel):
    """Représente une note explicative de bas de page."""

    id: str = Field(default="1", description="Identifiant ou symbole de la note")
    text: str = Field(default="", description="Texte explicatif complet")


class VisionTableResponse(BaseModel):
    """Schéma global de réponse pour l'extraction d'un tableau par Vision."""

    title: str = Field(default="", description="Titre du tableau")
    indicators: list[VisionIndicatorItem] = Field(default_factory=list)
    footnotes: list[VisionFootnoteItem] = Field(default_factory=list)
