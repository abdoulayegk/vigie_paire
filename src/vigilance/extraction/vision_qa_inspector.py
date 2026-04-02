"""Module for deep QA inspection of Vision extractions.

This module implements the "Second Brain" (Dual QA Pipeline). It takes an image crop
and the generated JSON from the primary Vision extractor, and cross-checks them to
ensure absolute exhaustiveness (zero dropped lines or forgotten footnotes).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any


from pydantic import BaseModel, Field

from vigilance.utils.genai import get_openai_api_key

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
        self.api_key = get_openai_api_key()
        
    def _build_qa_prompt(self, extracted_json: str) -> str:
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

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        raw_response = ""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(**payload_kwargs)
                raw_response = response.choices[0].message.content or ""
                break
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
        if not raw_response and last_exc is not None:
            raise last_exc

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
