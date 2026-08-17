"""Conversion Vision d'une figure PDF en texte narratif comparable."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Literal

import openai
from pydantic import BaseModel, ConfigDict, Field

from vigie.support.config import resolve_openai_model
from vigie.support.utils.genai import get_openai_api_key
from vigie.support.utils.pdf_crop import crop_table_region_to_bytes

logger = logging.getLogger(__name__)


class FigureNarrative(BaseModel):
    """Description factuelle à insérer dans le Markdown canonique."""

    model_config = ConfigDict(extra="forbid")

    visual_type: Literal[
        "chart",
        "org_chart",
        "process_diagram",
        "conceptual_figure",
        "decorative",
        "unreadable",
    ]
    title: str = ""
    summary: str = ""
    elements: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    explicit_values: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _clean_items(values: list[str]) -> list[str]:
    return [_clean_text(value) for value in values if _clean_text(value)]


def render_figure_narrative(
    narrative: FigureNarrative,
    *,
    page: int,
    preceding_heading: bool,
) -> list[str]:
    """Rend une description stable, lisible et directement comparable."""
    title = _clean_text(narrative.title) or "Contenu visuel"
    lines = (
        ["**Description du contenu visuel :**", ""] if preceding_heading else [f"### Figure — {title} [pdf.{page}]", ""]
    )
    summary = _clean_text(narrative.summary)
    if summary:
        lines.extend([summary, ""])
    groups = (
        ("Éléments visibles", narrative.elements),
        ("Relations visibles", narrative.relationships),
        ("Valeurs explicitement lisibles", narrative.explicit_values),
        ("Tendances visibles", narrative.trends),
    )
    for label, raw_values in groups:
        values = _clean_items(raw_values)
        if not values:
            continue
        lines.append(f"{label} :")
        lines.extend(f"- {value}" for value in values)
        lines.append("")
    return lines


class OpenAIFigureNarrator:
    """Décrit les figures informatives détectées par Docling, sans sidecar."""

    def __init__(
        self,
        *,
        pdf_path: Path,
        client: Any,
        model: str,
        confidence_threshold: float,
        max_calls: int,
        render_dpi: int,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.client = client
        self.model = str(model)
        self.confidence_threshold = float(confidence_threshold)
        self.max_calls = max(int(max_calls), 0)
        self.render_dpi = max(int(render_dpi), 96)
        self.calls_made = 0

    def describe(
        self,
        *,
        page: int,
        bbox_norm: list[float],
        section_title: str,
        context_before: str,
        context_after: str,
    ) -> FigureNarrative | None:
        """Retourne uniquement une description informative et suffisamment fiable."""
        if self.calls_made >= self.max_calls:
            return None
        crop = crop_table_region_to_bytes(
            str(self.pdf_path),
            int(page),
            list(bbox_norm),
            dpi=self.render_dpi,
            horizontal_padding=0.01,
        )
        if not crop:
            return None

        self.calls_made += 1
        encoded = base64.standard_b64encode(crop).decode("ascii")
        prompt = (
            "Décris uniquement le contenu informatif visible dans cette figure d'un rapport bancaire. "
            "La sortie sera insérée dans le texte du rapport puis comparée à celle d'une autre période. "
            "Pour un organigramme ou un processus, transcris les entités et les relations visibles. "
            "Pour un graphique, transcris les axes, unités, séries et tendances non ambiguës. "
            "Une valeur est explicit_values uniquement si elle est imprimée et parfaitement lisible; "
            "n'estime jamais un nombre depuis la hauteur d'une barre ou la position d'une courbe. "
            "Le contexte aide à nommer la figure mais ne doit jamais créer un élément absent de l'image. "
            "Classe les images sans information métier comme decorative et toute image illisible comme unreadable.\n\n"
            f"Section : {section_title}\n"
            f"Texte précédent : {context_before}\n"
            f"Texte suivant : {context_after}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu extrais fidèlement les graphiques, organigrammes et diagrammes de rapports bancaires. "
                    "Réponds strictement selon le schéma, en français, sans interprétation spéculative."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
                    },
                ],
            },
        ]
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=FigureNarrative,
                temperature=0.0,
            )
            narrative = response.choices[0].message.parsed
        except Exception as exc:  # noqa: BLE001 - une figure ne doit jamais bloquer le texte
            logger.warning("Description Vision de la figure pdf.%s impossible: %s", page, exc)
            return None
        if narrative is None:
            return None
        if narrative.visual_type in {"decorative", "unreadable"}:
            return None
        if narrative.confidence < self.confidence_threshold:
            return None
        return narrative


def build_figure_narrator(
    *,
    pdf_path: Path,
    config: dict[str, Any],
    client: Any | None = None,
) -> OpenAIFigureNarrator | None:
    """Construit l'analyseur optionnel; l'absence de clé conserve le flux texte."""
    if not bool(config.get("figure_vision_enabled", True)):
        return None
    api_key = get_openai_api_key()
    if client is None and not api_key:
        logger.info("Description Vision des figures désactivée: OPENAI_API_KEY absente.")
        return None
    if client is None:
        client = openai.OpenAI(api_key=api_key, timeout=float(config.get("figure_vision_timeout_sec", 120)))
    return OpenAIFigureNarrator(
        pdf_path=pdf_path,
        client=client,
        model=str(config.get("figure_vision_model") or resolve_openai_model("default_genai")),
        confidence_threshold=float(config.get("figure_vision_confidence_min", 0.75)),
        max_calls=int(config.get("figure_vision_max_calls_per_report", 20)),
        render_dpi=int(config.get("figure_vision_dpi", 220)),
    )
