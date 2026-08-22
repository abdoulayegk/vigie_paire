"""Module d'inspection QA approfondie des extractions Vision.

Ce module implemente le "Second Cerveau" (pipeline QA dual). Il prend un crop
d'image et le JSON genere par l'extracteur Vision primaire, puis les croise
afin de garantir une exhaustivite absolue (zero ligne oubliee, zero note
de bas de page manquante).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vigie.llm import get_client, resolve_model, structured_completions_parse

logger = logging.getLogger(__name__)
OPENAI_VISION_QA_TIMEOUT_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QAResult(BaseModel):
    """Resultat d'une inspection QA approfondie sur un tableau extrait."""

    model_config = ConfigDict(extra="forbid")

    is_perfect: bool = Field(
        ...,
        description="True if ALL textual rows from the image are present in the JSON. False if anything is missing.",
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="List of specific text snippets (e.g. sub-indicators, footnotes) missing from the JSON.",
    )
    justification: str = Field(
        ...,
        description="Brief explanation of the QA outcome.",
    )


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class VisionTableInspector:
    """Inspecteur QA intransigeant pour les extractions Vision GPT."""

    def __init__(self, model: str | None = None) -> None:
        """Initialiser l'inspecteur QA.

        Args:
            model: Modele OpenAI a utiliser. Par defaut, le modele chat resolu
                depuis la configuration (gpt-5.4).
        """
        self.model = str(model or "").strip() or resolve_model("chat")
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = get_client(timeout=OPENAI_VISION_QA_TIMEOUT_SECONDS, max_retries=1)
        return self._client

    def _build_qa_prompt(self, extracted_json: str) -> str:
        """Construire le prompt systeme pour l'audit QA."""
        return f"""You are an intransigent, highly meticulous financial data auditor.
Your ONLY job is to compare an image of a financial table against a provided JSON payload that claims to contain 100% of the required extracted fields.

MISSION RULES:
1. Audit the JSON against the extraction contract, field by field.
2. For "indicators", verify ONLY the logical row labels from the FIRST / LEFTMOST COLUMN of the table body, in top-to-bottom order.
3. If columns 2+ contain textual content, those cells are NOT indicators and MUST be ignored when auditing indicator completeness.
4. If a visual line has no visible first-column / leftmost cell, it must NOT create an indicator.
5. For "headers", verify the visible column headers from the header row only.
6. For "footnotes_content", verify only notes located BELOW the table body.
7. Ignore numeric column alignment, values, and any textual content that belongs to non-leftmost columns when judging indicators.
8. If a required first-column indicator row or footnote is visible in the image but missing from the JSON, you MUST return is_perfect: false, and list EXACTLY what the primary extractor forgot.

CRITICAL MULTI-COLUMN RULE:
- In tables such as "Canada | États-Unis | Europe", only the Canada / leftmost column produces indicators.
- Text appearing under "États-Unis", "Europe", or any other non-leftmost header must NEVER be reported as a missing indicator.
- Do not ask the extractor to include all text visible in the image. Audit only the contracted fields.

PROVIDED JSON TO AUDIT:
```json
{extracted_json}
```

Respond STRICTLY using the required JSON schema format."""

    def inspect_extraction(self, image_bytes: bytes, extracted_json: dict[str, Any]) -> QAResult:
        """Inspecter l'extraction pour detecter les elements manquants.

        Args:
            image_bytes: Octets bruts de l'image recadree du tableau.
            extracted_json: Sortie JSON brute de l'extracteur primaire.

        Returns:
            Resultat de la verification QA rigoureuse.

        Raises:
            Exception: Si l'appel OpenAI echoue apres toutes les tentatives.
        """
        json_str = json.dumps(extracted_json, ensure_ascii=False, indent=2)
        system_prompt = self._build_qa_prompt(json_str)
        base64_img = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Audit this image against the JSON payload provided in your system instructions.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    },
                ],
            },
        ]

        logger.debug("Executing Deep QA Inspector on table crop using model %s", self.model)

        client = self._ensure_client()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                parsed = structured_completions_parse(
                    client,
                    model=self.model,
                    messages=messages,
                    response_format=QAResult,
                    profile="extraction",
                )
                if not parsed.is_perfect:
                    logger.info(
                        "QA INSPECTOR ALERT: Missing elements detected: %s",
                        parsed.missing_elements,
                    )
                else:
                    logger.debug("QA INSPECTOR PASSED: Extraction is 100%% complete.")
                return parsed
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    raise
                logger.warning(
                    "QA Inspector OpenAI call failed on attempt %s/3: %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(0.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]
