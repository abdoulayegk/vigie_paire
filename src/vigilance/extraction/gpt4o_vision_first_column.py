"""GPT-4o Vision provider for first-column indicator extraction."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from ..utils.genai import get_openai_api_key
from .vision_first_column_provider import VisionFirstColumnResult

logger = logging.getLogger(__name__)

EXTRACT_FIRST_COLUMN_PROMPT = """You are extracting structured financial table data from a cropped table image.

Task:
Extract ONLY the first column row labels (financial indicators).

Rules:
- Return labels in visual top-to-bottom order.
- Merge multi-line labels into one single string.
- Do NOT include column headers.
- Do NOT include footnote rows.
- Do NOT include unit lines.
- Keep exact text as written (no normalization).
- Do not infer or summarize.
- Output strict JSON:
{
  "indicators": ["...", "...", "..."],
  "confidence": 0.0
}
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
            image_b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
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

        return self._parse_response(raw_content)

    def _parse_response(self, raw: str) -> VisionFirstColumnResult:
        try:
            data = json.loads(raw)
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
