"""Page-Level Assist for Table Titles.

Lightweight Vision pre-pass that extracts candidate table titles from a full page image.
Used as a fallback when the per-table red-box extraction misses or truncates the title.

This module does NOT extract indicators, rows, or footnotes — only titles.
The per-table VisionFullExtractor remains the single source of truth for table content.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

_PAGE_TITLE_EXTRACTION_METHOD = "page_level_assist_gpt4o"

_PAGE_TITLE_PROMPT = """
Tu es un expert en extraction de titres de tableaux à partir de rapports financiers bancaires canadiens.

TÂCHE
On te fournit l'image d'une page complète d'un rapport financier.
Cette page peut contenir UN ou PLUSIEURS tableaux.

Ta mission est UNIQUEMENT d'identifier et de lister les titres des tableaux visibles sur cette page.

Pour chaque tableau visible :
1. Trouve le numéro du tableau (ex: "Tableau 1", "Table 5", "T3a") s'il est visible.
2. Trouve le titre sémantique complet du tableau (ex: "Bilan consolidé condensé").
3. Estime la position verticale approximative du titre sur la page (bbox normalisée 0-1).

RÈGLES :
- NE PAS extraire le contenu des tableaux (ni indicateurs, ni données, ni notes).
- NE PAS inventer de titres — uniquement ce qui est visible.
- Si un tableau n'a pas de titre visible, ne pas l'inclure.
- Si le numéro et le nom sont sur deux lignes séparées, les combiner dans title_full.
- Respecter l'ordre visuel (haut vers bas).

RÉPONSE JSON STRICTE :

{
  "page_table_titles": [
    {
      "table_number": "1",
      "title_full": "Tableau 1 - Bilan consolidé condensé",
      "title_semantic": "Bilan consolidé condensé",
      "bbox_title": [0.05, 0.10, 0.90, 0.13],
      "confidence": 0.95
    }
  ]
}

DÉFINITIONS :
- table_number : numéro extrait (ex: "1", "5a", "28"). Chaîne vide si absent.
- title_full : titre complet incluant le numéro et le nom.
- title_semantic : titre sans le numéro (partie sémantique uniquement).
- bbox_title : [x_min, y_min, x_max, y_max] normalisé 0-1 de la position du titre.
- confidence : score 0.0-1.0 de confiance pour ce titre.
"""


class PageTitleCandidate(BaseModel):
    """Schema for one title candidate extracted from a page."""

    model_config = ConfigDict(extra="forbid")

    table_number: str = Field(
        default="",
        description="Numéro du tableau (ex: '1', '5a', '28'). Vide si absent.",
    )
    title_full: str = Field(
        default="",
        description="Titre complet incluant le numéro.",
    )
    title_semantic: str = Field(
        default="",
        description="Titre sémantique sans le numéro.",
    )
    bbox_title: list[float] | None = Field(
        default=None,
        description="Position normalisée [x_min, y_min, x_max, y_max].",
    )
    confidence: float = Field(
        default=0.0,
        description="Score de confiance 0.0-1.0.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("table_number", "title_full", "title_semantic", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        return str(v or "").strip()

    @field_validator("bbox_title", mode="before")
    @classmethod
    def _coerce_bbox(cls, v: Any) -> list[float] | None:
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 4:
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                return None
        return None


class PageTitleResponse(BaseModel):
    """Schema for the full page-level title extraction response."""

    model_config = ConfigDict(extra="forbid")

    page_table_titles: list[PageTitleCandidate] = Field(
        default_factory=list,
        description="Liste ordonnée des titres de tableaux détectés sur la page.",
    )


@dataclass
class PageTitleResult:
    """Result of page-level title extraction."""

    page_number: int
    candidates: list[dict[str, Any]] = field(default_factory=list)
    extraction_method: str = _PAGE_TITLE_EXTRACTION_METHOD

    def get_candidate_by_number(self, table_number: str) -> dict[str, Any] | None:
        """Find candidate by table number (exact match)."""
        number = str(table_number).strip()
        if not number:
            return None
        for c in self.candidates:
            if str(c.get("table_number", "")).strip() == number:
                return c
        return None

    def get_candidate_by_bbox_proximity(
        self, table_bbox: list[float], max_vertical_distance: float = 0.15
    ) -> dict[str, Any] | None:
        """Find the closest candidate title that is above the table bbox.

        Uses vertical proximity: the title bbox bottom (y_max) should be near
        the table bbox top (y_min), and the title should be ABOVE the table.
        """
        if not table_bbox or len(table_bbox) < 4:
            return None
        table_top = table_bbox[1]  # y_min of the table

        best: dict[str, Any] | None = None
        best_distance = float("inf")

        for c in self.candidates:
            bbox = c.get("bbox_title")
            if not bbox or len(bbox) < 4:
                continue
            title_bottom = bbox[3]  # y_max of the title
            # Title must be above the table
            if title_bottom > table_top + 0.02:
                continue
            distance = abs(table_top - title_bottom)
            if distance < best_distance and distance <= max_vertical_distance:
                best_distance = distance
                best = c

        return best


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def _parse_page_title_response(raw: str | dict[str, Any]) -> PageTitleResult | None:
    """Parse and validate the page-level title response."""
    try:
        if isinstance(raw, str):
            cleaned = _strip_markdown_fences(raw)
            data = json.loads(cleaned)
        else:
            data = raw

        if not isinstance(data, dict):
            return None

        validated = PageTitleResponse.model_validate(data)
        candidates = []
        for item in validated.page_table_titles:
            if not item.title_full and not item.title_semantic:
                continue
            candidates.append(
                {
                    "table_number": item.table_number,
                    "title_full": item.title_full,
                    "title_semantic": item.title_semantic,
                    "bbox_title": item.bbox_title,
                    "confidence": item.confidence,
                }
            )
        # Return with page_number=0 — caller sets the real page number.
        return PageTitleResult(page_number=0, candidates=candidates)
    except Exception as e:
        logger.debug("Page title response validation failed: %s", e)
        return None


class PageTitleAssistant:
    """Extract candidate table titles from a full-page image via GPT-4o.

    Lightweight, read-only pre-pass. Does not extract table content.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        min_confidence: float = 0.7,
        max_candidates: int = 10,
    ):
        from ..utils.genai import get_openai_api_key

        self._api_key = api_key or get_openai_api_key()
        self._model = model
        self._min_confidence = min_confidence
        self._max_candidates = max_candidates
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY required for page title assistant")
        self._client = OpenAI(api_key=self._api_key)

    def extract_page_titles(
        self,
        page_image_bytes: bytes,
        page_number: int,
    ) -> PageTitleResult | None:
        """Extract candidate table titles from a full page image.

        Args:
            page_image_bytes: PNG bytes of the full page.
            page_number: 1-based page number.

        Returns:
            PageTitleResult with candidates, or None on failure.
        """
        try:
            self._ensure_client()
        except (ImportError, ValueError) as e:
            logger.warning("PageTitleAssistant: client init failed: %s", e)
            return None

        client = self._client
        if client is None:
            return None

        image_b64 = base64.standard_b64encode(page_image_bytes).decode("ascii")

        content: list[Any] = [
            {"type": "text", "text": _PAGE_TITLE_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "low",
                },
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=2048,
            )
            raw_content = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(
                "Page title extraction API error (page %s): %s", page_number, e
            )
            return None

        result = _parse_page_title_response(raw_content)
        if result is None:
            logger.debug(
                "Page title extraction: invalid response for page %s", page_number
            )
            return None

        result.page_number = page_number

        # Filter by min_confidence and limit max_candidates
        result.candidates = [
            c
            for c in result.candidates
            if c.get("confidence", 0.0) >= self._min_confidence
        ][: self._max_candidates]

        return result
