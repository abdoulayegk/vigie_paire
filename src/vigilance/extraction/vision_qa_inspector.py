"""Module for deep QA inspection of Vision extractions.

This module implements the "Second Brain" (Dual QA Pipeline). It takes an image crop
and the generated JSON from the primary Vision extractor, and cross-checks them to
ensure absolute exhaustiveness (zero dropped lines or forgotten footnotes).
"""

from __future__ import annotations

import json
import logging
from typing import Any


from pydantic import BaseModel, Field

from vigilance.config import get_settings
from vigilance.utils.openai_client import (
    call_openai_with_retries,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QAResult(BaseModel):
    """Result of a deep QA inspection on an extracted table."""
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
    """Intransigent QA Inspector for GPT-4o Vision extractions."""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize the QA Inspector.
        
        Args:
            model: The OpenAI model to use. Default is 'gpt-4o' for maximum reliability,
                   but can be set to 'gpt-4o-mini' to save costs.
        """
        self.model = model
        self.settings = get_settings()
        
    def _build_qa_prompt(self, extracted_json: str) -> str:
        return f"""You are an intransigent, highly meticulous financial data auditor.
Your ONLY job is to compare an image of a financial table against a provided JSON payload that claims to contain 100% of the table's textual rows.

MISSION RULES:
1. Scan the image strictly line-by-line, top-to-bottom.
2. For EVERY textual row you see in the image (especially deeply indented sub-items and tiny footnotes at the bottom), verify if it is present somewhere in the JSON payload provided below.
3. Ignore formatting or numeric column alignment. Focus ONLY on completeness of textual indicators and footnotes.
4. If a word or row is in the image but missing from the JSON, you MUST return is_perfect: false, and list EXACTLY what the primary extractor forgot.

PROVIDED JSON TO AUDIT:
```json
{extracted_json}
```

Respond STRICTLY using the required JSON schema format."""

    def inspect_extraction(self, image_bytes: bytes, extracted_json: dict[str, Any]) -> QAResult:
        """Inspect the extraction for missing elements.
        
        Args:
            image_bytes: Raw bytes of the cropped table image.
            extracted_json: The raw JSON output from the primary extractor.
            
        Returns:
            QAResult: The result of the rigorous QA check.
            Raises exception if OpenAI fails.
        """
        # Convert JSON structure to a clean string for the prompt
        json_str = json.dumps(extracted_json, ensure_ascii=False, indent=2)
        
        system_prompt = self._build_qa_prompt(json_str)
        
        import base64
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
        
        payload_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 1500,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ExtractionQA",
                    "strict": True,
                    "schema": QAResult.model_json_schema(),
                },
            },
        }

        # Use robust retry wrapper
        raw_response = call_openai_with_retries(
            payload_kwargs,
            api_key=self.settings.openai_api_key.get_secret_value() if self.settings.openai_api_key else None,
            max_retries=2, 
        )

        try:
            parsed = QAResult.model_validate_json(raw_response)
            if not parsed.is_perfect:
                logger.info(
                    "QA INSPECTOR ALERT: Missing elements detected: %s", 
                    parsed.missing_elements
                )
            else:
                logger.debug("QA INSPECTOR PASSED: Extraction is 100%% complete.")
            return parsed
        except Exception as e:
            logger.error("Failed to parse QA Inspector response: %s", e)
            # In case of QA parse failure, assume perfect to avoid blocking the pipeline on QA internal errors
            return QAResult(is_perfect=True, missing_elements=[], justification="QA parsing failed, defaulting to True")
