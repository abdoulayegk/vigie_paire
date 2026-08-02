"""Module spécialisé dans la construction des prompts pour GPT-4o Vision."""

from __future__ import annotations

from typing import Any

VISION_PROMPT_BASE = """
You are a financial table extraction engine for Canadian bank quarterly reports (French language).

INPUT
You receive one CROPPED image that may contain:
- the table title above,
- the table itself (headers + data rows),
- footnotes below the table.

TASK
Analyse the image and extract ONLY these fields from the visible image:

PRIMARY — your main objective, must be complete and exhaustive:
- indicators (first-column row labels — every single row)
- footnotes_content (all footnotes below the table)

SECONDARY — best-effort, never sacrifice indicators or footnotes for these:
- table_title (title of the table, if visible)
- headers (column headers)
- table_summary (short business subject)
- no_table_detected (boolean)

Return VALID JSON ONLY. No markdown, no comments, no extra keys.

OUTPUT SCHEMA
{
  "indicators": ["string"],
  "table_title": "string",
  "headers": ["string"],
  "footnotes_content": [
    {"id": "string", "text": "string"}
  ],
  "table_summary": "string",
  "no_table_detected": false
}

DECISION RULE
- If no real tabular structure is visible (only narrative text, charts, or blank space), return:
  {
    "indicators": [],
    "table_title": "",
    "table_summary": "",
    "headers": [],
    "footnotes_content": [],
    "no_table_detected": true
  }

CRITICAL RULES FOR INDICATORS
- Return every row label from the first column of the table.
- Do NOT merge rows or omit rows.
- Clean up OCR artifacts, leading bullets, and excessive spaces.
"""


def build_vision_system_prompt() -> str:
    """Construit le prompt système pour l'extraction de tableaux bancaires par Vision."""
    return VISION_PROMPT_BASE


def build_vision_user_prompt(bank_code: str = "", context: str = "") -> str:
    """Construit le prompt utilisateur personnalisé pour l'image fournie."""
    prompt = "Extraire toutes les données structurées du tableau sur cette image."
    if bank_code:
        prompt += f" Document de la banque: {bank_code.upper()}."
    if context:
        prompt += f" Contexte: {context}."
    return prompt
