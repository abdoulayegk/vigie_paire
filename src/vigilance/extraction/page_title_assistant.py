"""Assistance au niveau de la page pour les titres de tableaux.

Pre-passe Vision legere qui extrait les titres candidats de tableaux a partir
d'une image de page complete. Utilisee en fallback lorsque l'extraction
per-table (red-box) manque ou tronque le titre.

Ce module n'extrait PAS d'indicateurs, de lignes ni de notes de bas de page
-- uniquement les titres. Le ``VisionFullExtractor`` per-table reste la source
de verite pour le contenu des tableaux.
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
Tu es un expert en extraction de titres de tableaux dans des rapports financiers bancaires canadiens (RBC, TD, CIBC, BNS, BMO, BNC).

## TÂCHE
On te fournit l'image d'une page complète d'un rapport financier.
Cette page peut contenir UN ou PLUSIEURS tableaux.
Ta mission : identifier et lister UNIQUEMENT les titres des tableaux visibles.

---

## CE QU'EST UN TITRE DE TABLEAU

Un titre de tableau est un texte qui :
- Apparaît DIRECTEMENT au-dessus du tableau (typiquement 0 à 4 lignes avant la première ligne du tableau)
- Décrit le contenu du tableau (ex : "Fonds propres réglementaires", "Ratio de levier")
- Peut être précédé d'un numéro (ex : "Tableau 12", "T3a", "Table 5")
- Peut contenir une unité ou une période entre parenthèses sur la ligne suivante

#### Patterns visuels à reconnaître par banque

RBC :
- Titre en gras.
- Phrase descriptive complète.
- Pas toujours de numéro de tableau.
- Peut contenir des notes (1).
- Une phrase explicative peut apparaître entre le titre et le tableau.

TD :
- Titre commence souvent par "TABLEAU XX :".
- Texte souvent en majuscules.
- Peut contenir un tiret ou deux-points.

CIBC :
- Titre peut être une phrase descriptive longue.
- Parfois sans numéro de tableau.
- Peut ressembler à un titre de section mais il est immédiatement suivi d’un tableau.

BNS (Scotia) :
- Format fréquent : "T20 Titre".
- Code court suivi du nom du tableau.

BMO :
- Titre en bleu.
- Numéro du tableau sur la ligne suivante (ex : TABLEAU 23).
- Les deux lignes doivent être combinées.

BNC :
- Phrase descriptive longue.
- Contient souvent des notes (1) (2).
- Souvent suivi d’une ligne indiquant l’unité.

Le titre doit toujours être associé au tableau le plus proche situé en dessous.
Si la distance verticale entre le texte et le tableau est trop grande,
ce texte n'est probablement pas un titre de tableau.

### Titres sur plusieurs lignes
Si le titre est réparti sur 2-3 lignes consécutives directement au-dessus du tableau
(ex: ligne 1 = "Tableau 5", ligne 2 = "Exposition au risque de crédit", ligne 3 = "(en millions de dollars)"),
COMBINER toutes ces lignes dans `title_full`, et exclure la ligne d'unité de `title_semantic`.

---

## CE QUI N'EST PAS UN TITRE DE TABLEAU

- **Titres de section ou sous-section** : titres de paragraphe qui introduisent du texte narratif (pas un tableau)
- **Phrases introductives** : "Le tableau ci-après présente..." ou "Les données suivantes illustrent..."
  → Ces phrases peuvent PRÉCÉDER un tableau — le vrai titre est le texte en gras/majuscule juste au-dessus du tableau lui-même
- **En-têtes de colonnes** : première ligne à l'intérieur du tableau (ex : "T1 2025 | T4 2024 | T1 2024")
- **Notes de bas de tableau** : lignes commençant par (1), ¹, *, ou "Note :"
- **Étiquettes de graphique** : légendes ou titres de figures/graphiques

---

## RÈGLES D'EXTRACTION

1. **NE PAS** extraire le contenu des tableaux (données, indicateurs, notes).
2. **NE PAS** inventer de titres — uniquement ce qui est visuellement présent.
3. Si un tableau n'a **aucun titre visible** directement au-dessus de lui, **ne pas l'inclure**.
4. Respecter l'**ordre visuel** (haut de page → bas de page).
5. Si le numéro et le titre sont sur deux lignes séparées, les **combiner** dans `title_full`.
6. `title_semantic` = titre sans le numéro ET sans la mention d'unité/période.
7. Pour `bbox_title` : coordonnées [x_min, y_min, x_max, y_max] normalisées 0.0–1.0
   représentant la position du bloc-titre (toutes les lignes du titre incluses).

---

## FORMAT DE RÉPONSE — JSON STRICT UNIQUEMENT
```json
{
  "page_table_titles": [
    {
      "table_number": "12",
      "title_full": "Tableau 12 – Fonds propres réglementaires (en millions de dollars)",
      "title_semantic": "Fonds propres réglementaires",
      "bbox_title": [0.04, 0.08, 0.92, 0.12],
      "confidence": 0.95
    }
  ]
}
```

Si aucun titre de tableau n'est trouvé sur la page :
```json
{ "page_table_titles": [] }
```

---

## DÉFINITIONS DES CHAMPS

| Champ | Type | Description |
|---|---|---|
| `table_number` | string | Numéro extrait (ex: "1", "5a", "28"). Chaîne vide `""` si absent. |
| `title_full` | string | Titre complet tel qu'il apparaît : numéro + nom + unité si présente. |
| `title_semantic` | string | Titre sans le numéro et sans la mention d'unité ou de période. |
| `bbox_title` | float[4] | [x_min, y_min, x_max, y_max] normalisé 0–1. Couvre toutes les lignes du titre. |
| `confidence` | float | Score 0.0–1.0. Mettre < 0.7 si le titre est ambigu ou partiellement visible. |

---

## EXEMPLES DE CAS LIMITES

- *"Risque de crédit"* seul en gras dans un paragraphe narratif → **NE PAS extraire** (titre de section)
- *"Tableau 3\nExposition au risque de crédit\n(en millions $)"* au-dessus d'un tableau → **EXTRAIRE**, `title_semantic` = "Exposition au risque de crédit"
- Une phrase *"Le tableau suivant présente les ratios..."* suivie d'un tableau sans titre en gras → **NE PAS extraire** (pas de titre formel)
- Titre partiellement coupé en haut de page → extraire avec `confidence` ≤ 0.6
"""


class PageTitleCandidate(BaseModel):
    """Schema d'un titre candidat extrait d'une page."""

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
        """Convertir en chaine et retirer les espaces superflus."""
        return str(v or "").strip()

    @field_validator("bbox_title", mode="before")
    @classmethod
    def _coerce_bbox(cls, v: Any) -> list[float] | None:
        """Convertir le bbox en liste de 4 flottants ou ``None``."""
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 4:
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                return None
        return None


class PageTitleResponse(BaseModel):
    """Schema de la reponse complete d'extraction de titres au niveau de la page."""

    model_config = ConfigDict(extra="forbid")

    page_table_titles: list[PageTitleCandidate] = Field(
        default_factory=list,
        description="Liste ordonnée des titres de tableaux détectés sur la page.",
    )


@dataclass
class PageTitleResult:
    """Resultat de l'extraction de titres au niveau de la page."""

    page_number: int
    candidates: list[dict[str, Any]] = field(default_factory=list)
    extraction_method: str = _PAGE_TITLE_EXTRACTION_METHOD

    def get_candidate_by_number(self, table_number: str) -> dict[str, Any] | None:
        """Trouver un candidat par numero de tableau (correspondance exacte)."""
        number = str(table_number).strip()
        if not number:
            return None
        for c in self.candidates:
            if str(c.get("table_number", "")).strip() == number:
                return c
        return None

    def get_candidate_by_bbox_proximity(
        self,
        table_bbox: list[float],
        max_vertical_distance: float = 0.15,
        other_table_bboxes: list[list[float]] | None = None,
    ) -> dict[str, Any] | None:
        """Trouver le titre candidat le plus proche situe au-dessus du tableau.

        Utilise la proximite verticale : le bas du bbox du titre (y_max) doit
        etre proche du haut du bbox du tableau (y_min), et le titre doit etre
        AU-DESSUS du tableau.

        Lorsque *other_table_bboxes* est fourni (page multi-tableaux), un
        candidat est rejete si le bbox d'un autre tableau se situe verticalement
        entre le titre et ce tableau (evite d'attribuer un titre de section /
        tableau superieur a un tableau inferieur).

        Args:
            table_bbox: Bbox normalise du tableau ``[x_min, y_min, x_max, y_max]``.
            max_vertical_distance: Distance verticale maximale autorisee.
            other_table_bboxes: Bboxes des autres tableaux sur la meme page.

        Returns:
            Dictionnaire du candidat le plus proche ou ``None``.
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
            # On multi-table pages: reject if any other table sits between title and this table
            if other_table_bboxes:
                blocked = False
                for other in other_table_bboxes:
                    if not other or len(other) < 4:
                        continue
                    other_top = other[1]
                    other_bottom = other[3]
                    if title_bottom < other_top and other_bottom < table_top:
                        blocked = True
                        break
                if blocked:
                    continue
            distance = abs(table_top - title_bottom)
            if distance < best_distance and distance <= max_vertical_distance:
                best_distance = distance
                best = c

        return best


def _strip_markdown_fences(text: str) -> str:
    """Retirer les clotures Markdown (````` ```json ... ``` `````) d'une chaine."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def _parse_page_title_response(raw: str | dict[str, Any]) -> PageTitleResult | None:
    """Parser et valider la reponse d'extraction de titres au niveau de la page."""
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
    """Extraire les titres candidats de tableaux depuis une image de page via GPT-4o.

    Pre-passe legere en lecture seule. N'extrait pas le contenu des tableaux.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        min_confidence: float = 0.7,
        max_candidates: int = 10,
        api_retry_max: int = 3,
        api_retry_backoff_ms: float = 1000,
    ):
        """Initialiser l'assistant d'extraction de titres de page.

        Args:
            api_key: Cle API OpenAI (ou via ``OPENAI_API_KEY``).
            model: Modele a utiliser (defaut : ``gpt-4o``).
            min_confidence: Score de confiance minimal pour conserver un candidat.
            max_candidates: Nombre maximal de candidats renvoyes.
            api_retry_max: Nombre maximal de tentatives en cas d'erreur API.
            api_retry_backoff_ms: Delai de base entre les tentatives (ms).
        """
        from ..utils.genai import get_openai_api_key

        self._api_key = api_key or get_openai_api_key()
        self._model = model
        self._min_confidence = min_confidence
        self._max_candidates = max_candidates
        self._api_retry_max = max(0, api_retry_max)
        self._api_retry_backoff_ms = max(0.0, api_retry_backoff_ms)
        self._client: Any = None

    def _ensure_client(self) -> None:
        """Initialiser le client OpenAI si necessaire."""
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
        """Extraire les titres candidats de tableaux depuis une image de page complete.

        Args:
            page_image_bytes: Octets PNG de la page complete.
            page_number: Numero de page (1-indexed).

        Returns:
            ``PageTitleResult`` avec les candidats, ou ``None`` en cas d'echec.
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

        import time

        raw_content = ""
        last_error: Exception | None = None
        for attempt in range(self._api_retry_max + 1):
            if attempt > 0:
                backoff_sec = (self._api_retry_backoff_ms / 1000.0) * (
                    2 ** (attempt - 1)
                )
                logger.info(
                    "Page title extraction: retry %s/%s after %.1fs (page %s)",
                    attempt,
                    self._api_retry_max,
                    backoff_sec,
                    page_number,
                )
                time.sleep(backoff_sec)
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_completion_tokens=2048,
                )
                raw_content = response.choices[0].message.content or ""
                last_error = None
                break
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                retryable = (
                    "rate" in msg
                    and "limit" in msg
                    or "timeout" in msg
                    or "timed out" in msg
                    or "connection" in msg
                    or "connect" in msg
                )
                if not retryable or attempt >= self._api_retry_max:
                    logger.warning(
                        "Page title extraction API error (page %s): %s",
                        page_number,
                        e,
                    )
                    return None
        if last_error is not None:
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
