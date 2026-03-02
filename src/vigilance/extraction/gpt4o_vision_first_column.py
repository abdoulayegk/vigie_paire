"""GPT-4o Vision provider for first-column indicator extraction."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from ..utils.genai import get_openai_api_key
from .vision_first_column_provider import VisionFirstColumnResult

logger = logging.getLogger(__name__)

EXTRACT_FIRST_COLUMN_PROMPT = """Tu es un expert en extraction de donnees de rapports bancaires canadiens (BNC, BMO, CIBC, TD, RBC, BNS).

TACHE: Extraire UNIQUEMENT les libelles de la premiere colonne (indicateurs financiers) de cette image de tableau.

INSTRUCTIONS:
1. Retourne les libelles dans l'ordre visuel de haut en bas
2. Fusionne les libelles multi-lignes en une seule chaine
3. Conserve la hierarchie : les sous-lignes indentees doivent etre incluses avec leurs espaces en debut
4. Inclure TOUTES les lignes : sous-totaux, totaux, lignes indentees
5. Conserver les references de notes (1), (2), *, etc. dans le texte
6. NE PAS inclure les en-tetes de colonnes (dates, periodes, etc.)
7. NE PAS inclure les lignes de notes de bas de page
8. NE PAS inclure les lignes d'unites (en millions, en milliers, etc.)
9. Conserver le texte EXACTEMENT tel qu'il apparait (pas de normalisation)
10. Ne pas inferer ni resumer

FORMAT DE REPONSE (JSON strict):
{
  "indicators": ["Libelle 1", "  Sous-libelle 1a", "Total", "..."],
  "confidence": 0.0
}

REGLES:
- indicators = TOUS les libelles de la premiere colonne, dans l'ordre visuel
- confidence entre 0.0 et 1.0 selon la lisibilite de l'image
"""


class GPT4oVisionFirstColumnProvider:
    """GPT-4o Vision implementation of VisionFirstColumnProvider."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_openai_api_key()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI

            if not self._api_key:
                raise ValueError("OPENAI_API_KEY required")
            self._client = OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def extract_first_column(self, image_path: str) -> VisionFirstColumnResult:
        """Extract first-column labels from a cropped table image. Returns raw strings."""
        try:
            self._ensure_client()
        except (ImportError, ValueError):
            return VisionFirstColumnResult(
                indicators_raw=[],
                confidence=0.0,
                provider="gpt-4o",
            )

        path = Path(image_path)
        if not path.exists():
            return VisionFirstColumnResult(
                indicators_raw=[],
                confidence=0.0,
                provider="gpt-4o",
            )

        try:
            from .vision_image_preprocessor import preprocess_for_vision

            processed = preprocess_for_vision(path.read_bytes())
            image_b64 = base64.standard_b64encode(processed).decode("ascii")
        except Exception:
            return VisionFirstColumnResult(
                indicators_raw=[],
                confidence=0.0,
                provider="gpt-4o",
            )

        try:
            client = self._client
            if client is None:
                return VisionFirstColumnResult(
                    indicators_raw=[],
                    confidence=0.0,
                    provider="gpt-4o",
                )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": EXTRACT_FIRST_COLUMN_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ]},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=4096,
            )
            raw_content = response.choices[0].message.content or ""
        except Exception as e:
            logger.debug("GPT-4o first-column extraction failed: %s", e)
            return VisionFirstColumnResult(
                indicators_raw=[],
                confidence=0.0,
                provider="gpt-4o",
            )

        result = self._parse_response(raw_content)
        if not result.indicators_raw:
            logger.warning(
                "GPT-4o Vision returned 0 indicators. Raw response (first 200 chars): %s",
                repr(raw_content[:200]),
            )
        else:
            logger.info(
                "GPT-4o Vision extracted %d indicators (confidence=%.2f)",
                len(result.indicators_raw),
                result.confidence,
            )
        return result

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences that GPT-4o sometimes wraps around JSON."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1:]
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
        return stripped

    def _parse_response(self, raw: str) -> VisionFirstColumnResult:
        try:
            cleaned = self._strip_markdown_fences(raw)
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                return VisionFirstColumnResult(
                    indicators_raw=[],
                    confidence=0.0,
                    provider="gpt-4o",
                )
            indicators = data.get("indicators", [])
            if not isinstance(indicators, list):
                indicators = []
            indicators = [str(x).strip() for x in indicators if x]
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            return VisionFirstColumnResult(
                indicators_raw=indicators,
                confidence=confidence,
                provider="gpt-4o",
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return VisionFirstColumnResult(
                indicators_raw=[],
                confidence=0.0,
                provider="gpt-4o",
            )
