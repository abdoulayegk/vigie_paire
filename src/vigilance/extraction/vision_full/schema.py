"""Schema Pydantic de la reponse Vision et contrat OpenAI Structured Outputs.

Extrait de ``vision_full_extractor.py`` sans modification.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...utils.openai_schema import (
    build_strict_openai_response_format,
    validate_strict_openai_response_format,
)

class VisionFootnoteItem(BaseModel):
    """Schema strict pour une entree de note de bas de page."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Marqueur de note (ex: 1, (1), a)")
    text: str = Field(description="Texte de la note")

    @field_validator("id", "text", mode="before")
    @classmethod
    def _coerce_non_empty_str(cls, v: Any) -> str:
        """Convertit en chaine non vide ou leve une erreur."""
        s = str(v or "").strip()
        if not s:
            raise ValueError("must be non-empty")
        return s


class VisionResponseCommonSchema(BaseModel):
    """Schema commun pour les reponses Vision completes et de repli."""

    model_config = ConfigDict(extra="forbid")

    table_title: str = Field(
        default="",
        description="Titre complet et visible du tableau, incluant le numéro (ex: 'Tableau 1') s'il est au-dessus. Chaine vide si aucun titre visible.",
    )
    table_summary: str = Field(
        default="",
        description="Résumé métier du tableau en 15 mots maximum, sans chiffres inventés ni interprétation.",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="En-tetes de colonnes du tableau, dans l'ordre",
    )
    indicators: list[str] = Field(
        description="Libelles logiques de la premiere colonne, ordre visuel haut vers bas. Fusionner les retours a la ligne d'un meme libelle en un seul element.",
    )
    footnotes_content: list[VisionFootnoteItem] = Field(
        description="Liste ORDONNEE de notes structurees [{id, text}] — ordre visuel strict",
        default_factory=list,
    )
    no_table_detected: bool = Field(
        default=False,
        description="True only when no real tabular structure is visible in the crop.",
    )

    @field_validator("indicators", mode="before")
    @classmethod
    def _coerce_indicators(cls, v: Any) -> list[str]:
        """Accepte les formats d'indicateurs chaine et objet legacy.

        Utilise ``.rstrip()`` au lieu de ``.strip()`` pour que les espaces de tete
        encodant l'indentation visuelle (profondeur hierarchique) soient preserves.
        """
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                text = item.rstrip()
                if text:
                    result.append(text)
            elif isinstance(item, dict):
                text = str(item.get("text") or "").rstrip()
                if text:
                    result.append(text)
        return result

    @field_validator("table_summary", mode="after")
    @classmethod
    def _normalize_table_summary(cls, v: str) -> str:
        """Normalise le resume du tableau en le tronquant a 15 mots."""
        words = [word for word in str(v or "").split() if word.strip()]
        return " ".join(words[:15]).strip()

    @field_validator("headers", mode="after")
    @classmethod
    def _normalize_headers(cls, v: list[str]) -> list[str]:
        """Normalise les en-tetes en supprimant les espaces superflus."""
        return [str(x).strip() for x in v]

    @field_validator("footnotes_content", mode="before")
    @classmethod
    def _coerce_footnotes_content(cls, v: Any) -> list[dict[str, str]]:
        """Convertit les formats legacy de notes de bas de page en liste ordonnee."""
        # Migration shim: accept legacy dict marker->text and normalize to ordered list.
        # The dict form loses visual order — items are added in insertion order.
        if isinstance(v, dict):
            out: list[dict[str, str]] = []
            for k, val in v.items():
                marker = str(k).strip()
                text = str(val).strip()
                if marker and text:
                    out.append({"id": marker, "text": text})
            return out
        if isinstance(v, list):
            out = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                marker = str(item.get("id") or item.get("marker") or item.get("ref") or "").strip()
                text = str(item.get("text") or item.get("value") or "").strip()
                if marker and text:
                    out.append({"id": marker, "text": text})
            return out
        return []


class VisionFullResponseSchema(VisionResponseCommonSchema):
    """Schema strict pour la sortie d'extraction Vision normale."""


class VisionSchemaContractError(RuntimeError):
    """Levee lorsque le contrat de schema OpenAI Structured Outputs est invalide."""


def _build_openai_json_schema() -> dict[str, Any]:
    """Construit le format json_schema OpenAI a partir du modele Pydantic pour Structured Outputs (schema complet uniquement)."""
    return build_strict_openai_response_format(
        VisionFullResponseSchema,
        name="vision_full_extraction",
        error_cls=VisionSchemaContractError,
    )


def _validate_openai_strict_schema_contract(schema: dict[str, Any]) -> None:
    """Valide le contrat strict Structured Outputs en local avant l'appel API."""
    validate_strict_openai_response_format(
        schema,
        error_cls=VisionSchemaContractError,
    )
