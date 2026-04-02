"""Vision-based minimal extraction: title, summary, headers, indicators, and footnotes."""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import resolve_openai_model
from ..utils.genai import get_openai_api_key
from .vision_cache import (
    cache_get,
    cache_put,
    get_vision_cache_dir,
    make_cache_key,
)

logger = logging.getLogger(__name__)

_EXTRACTION_METHOD = "vision_full_gpt4o"

_RECROP_EXTENSION_INCREMENT = 0.06
_DEFAULT_MAX_COMPLETION_TOKENS = 65536
# Current extraction routing uses 64k by default and allows a 128k rescue pass when truncation is detected.
_MAX_COMPLETION_TOKENS_API_LIMIT = 128000
_RESCUE_MAX_COMPLETION_TOKENS = 128000
_MAX_COMPLETION_TOKENS_SAFE_FALLBACK = 16384
_DEFAULT_REFERENCE_TEXT_MAX_CHARS = 6000
_MODEL_ROLE = "extraction_primary"
_GENERIC_PAGE_TITLES = {
    "rapport de gestion",
    "management's discussion and analysis",
    "management discussion and analysis",
    "rapport annuel",
    "shareholders report",
    "rapport aux actionnaires",
}
_WEAK_INDICATOR_EXACT = {
    "total",
    "totaux",
    "autres",
    "autre",
    "other",
    "others",
    "canada",
    "united states",
    "états-unis",
    "etats-unis",
}
_WEAK_INDICATOR_TOKENS = {
    "total",
    "totaux",
    "autres",
    "autre",
    "other",
    "others",
    "canada",
    "united",
    "states",
    "états",
    "etats",
    "unis",
}
_NARRATIVE_INDICATOR_PHRASES = (
    "texte narratif",
    "rapport de gestion",
    "cette section",
    "ce tableau",
    "comprend",
    "comprennent",
    "inclut",
    "incluent",
    "présente",
    "presente",
    "présentent",
    "presentent",
    "représente",
    "represente",
    "décrit",
    "decrit",
    "description",
)

_PROMPT_BASE = """

You are a precision-first financial table extraction engine for Canadian bank quarterly reports (French language).

INPUT
You receive one CROPPED image that may contain:
- the table title above,
- the table itself (headers + data rows),
- footnotes below the table.

TASK
Analyse the image and extract ONLY these fields from the visible image:
- indicators (first-column row labels)
- footnotes_content (footnotes below the table)
- headers (column headers)
- table_title (title of the table)
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
- Otherwise extract all visible fields and set "no_table_detected": false.

═══════════════════════════════════════════
FIELD 1: indicators (HIGHEST PRIORITY)
═══════════════════════════════════════════

Extract ALL logical row labels from the FIRST COLUMN of the table, in strict visual order top → bottom.
An indicator comes ONLY from the LEFTMOST visible cell of each logical row in the table body.

An indicator is any text that functions as a row label:
- normal row label
- indented sub-row
- group heading / section heading within the table
- subtotal or total row
- maturity bucket or period bucket used as a row label
- a label with an attached footnote marker such as (1), *, †, ¹

PRESERVE:
- exact wording (French accents, hyphens, special characters)
- attached footnote markers that are part of the label
- visual top-to-bottom order

DO NOT:
- translate, normalize, summarize, correct spelling, or reorder
- include column headers (these go in the "headers" field)
- include pure numeric values, units-only cells, or isolated footnote markers
- include narrative paragraphs or explanatory text blocks
- include free text below the table (these may be footnotes)
- include text from columns 2+ even when those cells contain meaningful business phrases

CRITICAL: MULTI-COLUMN TEXT TABLES
If the table has multiple textual columns (for example headers like "Canada", "États-Unis", "Europe"):
- ONLY the text in the LEFTMOST / FIRST COLUMN is eligible for "indicators"
- text visible under "États-Unis", "Europe", or any other non-leftmost header must NEVER appear in "indicators"
- if a logical row has no visible first-column / leftmost cell, it must NOT produce an indicator
- exhaustiveness means "all first-column row labels", NOT "all text visible anywhere in the image"

CRITICAL: FORCED EXHAUSTIVENESS (ANTI-SUMMARIZATION)
You MUST extract EVERY SINGLE ROW containing financial data. Do NOT summarize or group rows.
- Highly indented sub-items and child rows are VALID indicators. Extract them all.
- Never drop a row just because it looks like a subordinate breakdown. Read the table line-by-line and extract everything.

CRITICAL: MULTI-LINE MERGE RULE
If one indicator wraps onto multiple visual lines in the first column, merge them into ONE indicator string.
Merge ONLY when:
- same left alignment and indentation level
- the next line clearly continues the same business label
- the next line does NOT begin a new row with its own data values
- the merged text forms one natural row label

Do NOT merge when:
- the second line is a new row with its own values in other columns
- the second line is a subtotal, total, or new category
- the first line is a group heading and the next line is a distinct sub-row

CRITICAL: DISAMBIGUATION RULE FOR REPEATED LABELS
When the EXACT SAME label text appears in multiple rows of the same table (because the table has repeating sub-sections), you MUST disambiguate by prepending the nearest visible GROUP HEADING or SECTION HEADING:
- Format: "Group Heading – repeated_label"
- Example: a table has section "Fonds propres CET1" containing "Solde au début" AND section "Fonds propres catégorie 1" also containing "Solde au début".
  Output: ["Fonds propres CET1 – Solde au début", ..., "Fonds propres catégorie 1 – Solde au début"]
- If no group heading exists above the repeated label, keep the original label unchanged.
- Apply this ONLY to labels that would otherwise appear as exact duplicates in the output list.
- Every indicator in the final output list MUST be unique.
  If disambiguation via group heading is not possible, append a position marker: " (bloc 2)" to the second occurrence.

CRITICAL: EXCLUDE DATE/PERIOD IDENTIFIERS
Do NOT extract as indicators:
- date identifiers that function as row-group delimiters or column sub-headers:
  "Au 30 avril 2025", "Au 31 janvier 2025", "Au 31 octobre 2024"
- period identifiers:
  "Trimestre clos le ...", "Pour l'exercice clos le ...", "Exercice terminé le ...",
  "Semestre terminé le ..."
- unit descriptors spanning the table width:
  "En millions de dollars", "En milliers de dollars", "(en millions de dollars)"
- quarter labels used as sub-headers: "T1 2025", "Q1 2025", "2024", "2025"
These are temporal/unit metadata, NOT business indicators.

HIERARCHY PRESERVATION
Use leading spaces to indicate the nesting depth as visible in the table:
- Top-level labels: no leading spaces
- First-level sub-items (visually indented once): prefix with "  " (2 spaces)
- Second-level sub-items (indented twice): prefix with "    " (4 spaces)
Reflect the visual hierarchy faithfully. This is essential for downstream accuracy.

═══════════════════════════════════════════
FIELD 2: footnotes_content (SECOND PRIORITY)
═══════════════════════════════════════════

Extract ONLY footnotes located BELOW the table body, inside the cropped image.

For each footnote return:
- id: normalized marker → "1", "2", "3", "*", "†", "a" (strip parentheses: "(1)" → "1")
- text: exact full footnote text

Detect ALL marker styles:
- superscript digits: ¹ ² ³ → normalize to "1", "2", "3"
- parenthetical: (1) (2) (3) → normalize to "1", "2", "3"
- symbols: *, †, ‡ → keep as-is

Rules:
- preserve exact wording of each footnote
- preserve visual top-to-bottom order (do NOT sort by id)
- do not merge two separate footnotes into one
- do not invent missing text
- do not extract narrative paragraphs, body text, or discussion as footnotes
- if no footnotes are visible below the table, return []

═══════════════════════════════════════════
FIELD 3: headers (THIRD PRIORITY)
═══════════════════════════════════════════

Return the visible column headers in left-to-right order.
- If a header spans multiple visual lines, merge into one string.
- Do NOT include row labels from the first column unless the first column has its own explicit header text.
- Do NOT include empty strings in the list.
- If no column headers are visible, return [].

═══════════════════════════════════════════
FIELD 4: table_title
═══════════════════════════════════════════

Return the full visible title of the table, including the table number if present (e.g., "Tableau 12 – Titre du tableau").

TITLE RULES:
- A table title is a heading DIRECTLY ABOVE the table's header row or first data row.
- It typically starts with "Tableau XX", "Table XX", or "TABLEAU XX", or is bold/larger text.
- Do NOT use page running headers, section headings, or chapter titles as table_title.
  Examples of page furniture to IGNORE: "Rapport de gestion", "Management's Discussion and Analysis", "Rapport aux actionnaires", "Rapport annuel"
- If the title is partially cut off at the crop boundary, extract the visible portion.
- If no real table title is visible directly above the table, return "".
- NEVER invent or guess a title.

═══════════════════════════════════════════
FIELD 5: table_summary
═══════════════════════════════════════════

Return a short noun phrase (max 15 words) describing the business subject of the table.
- Base it ONLY on the visible title and content.
- Do NOT add analysis, numbers, trends, or conclusions.
- If the image is a continuation of a table from a previous page (no title, indicators continue mid-sequence), prefix with "Suite: " then the subject.
- If unclear, return "".

═══════════════════════════════════════════
FIELD 6: no_table_detected
═══════════════════════════════════════════

Set to true ONLY if NO real tabular structure (rows + columns with data) is visible in the crop.
If even a partial table is visible, set to false.

═══════════════════════════════════════════
GENERAL PRIORITY RULES
═══════════════════════════════════════════

1. PRECISION over recall. Never invent content.
2. Extract ONLY what is visible in the cropped image.
3. Keep row order and footnote order exactly as visually seen.
4. If uncertain whether text is an indicator or narrative → EXCLUDE from indicators.
5. If uncertain whether bottom text is a footnote → EXCLUDE from footnotes_content.
6. If uncertain whether there is a real table → set no_table_detected to true.

FINAL REQUIREMENT
Return one JSON object only, with exactly these 6 keys:
indicators, footnotes_content, table_title, headers, table_summary, no_table_detected
"""

_PROMPT_JSON_STRICT = """
STRICT JSON RESPONSE.
Return valid JSON only. No text before or after.

The JSON object must strictly follow this structure:

{
"indicators": ["Group A – Label 1", "  Sub-label", "Group A – Total", "Group B – Label 1"],
"footnotes_content": [
  {"id": "1", "text": "footnote text 1"},
  {"id": "2", "text": "footnote text 2"}
],
"headers": ["Column 1", "Column 2", "Column 3"],
"table_summary": "Business subject of the table in 15 words maximum",
"table_title": "Tableau 1 – Full title as visible above the table",
"no_table_detected": false
}

VALIDATION CHECKLIST (apply before returning):
1. indicators: every element is UNIQUE — if duplicates exist, disambiguate with group heading prefix ("Group – label")
2. indicators: no date/period identifiers ("Au 30 avril 2025", "Trimestre clos le...", "En millions de dollars")
3. indicators: no pure numeric values, no column headers, no narrative paragraphs, no text from columns 2+
4. indicators: multi-line labels merged into single strings
5. indicators: hierarchy preserved via leading spaces (2-space increments per nesting level)
6. footnotes_content: visual order preserved (top → bottom), NOT sorted by id
7. footnotes_content: marker ids normalized (strip parentheses, convert superscripts to digits)
8. headers: no empty strings, left-to-right order
9. table_title: not a page running header or section heading — must be directly above the table
10. table_summary: ≤ 15 words, noun phrase only, prefix "Suite: " for continuation tables
11. no_table_detected: true ONLY if zero tabular structure is visible
"""
_PROMPT_RESCUE_SUFFIX = """

RESCUE MODE — The previous extraction was empty, partial, or contaminated.
Apply these overrides:
- IGNORE page titles, running headers, section headings, and any non-tabular page furniture.
- Focus ONLY on the real table visible in the crop.
- If both a page title and a real table are visible, the page title must NOT be used as table_title.
- If a real table is visible, extract it as completely and precisely as possible.
- Apply the disambiguation rule for repeated labels (prepend group heading).
- Apply the date/period exclusion rule.
- Apply hierarchy preservation via leading spaces.
- In multi-column textual tables, re-check that ONLY the leftmost column populates "indicators".
- Use no_table_detected = true only if absolutely no tabular structure is visible.
"""


class VisionFootnoteItem(BaseModel):
    """Strict item schema for one footnote entry."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Marqueur de note (ex: 1, (1), a)")
    text: str = Field(description="Texte de la note")

    @field_validator("id", "text", mode="before")
    @classmethod
    def _coerce_non_empty_str(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("must be non-empty")
        return s


class VisionResponseCommonSchema(BaseModel):
    """Common schema for full and fallback Vision responses."""

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
        """Accept both string and legacy object indicator formats.

        Uses ``.rstrip()`` instead of ``.strip()`` so that leading whitespace
        encoding visual indentation (hierarchy depth) is preserved.
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
        words = [word for word in str(v or "").split() if word.strip()]
        return " ".join(words[:15]).strip()

    @field_validator("headers", mode="after")
    @classmethod
    def _normalize_headers(cls, v: list[str]) -> list[str]:
        return [str(x).strip() for x in v]

    @field_validator("footnotes_content", mode="before")
    @classmethod
    def _coerce_footnotes_content(cls, v: Any) -> list[dict[str, str]]:
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
                marker = str(
                    item.get("id") or item.get("marker") or item.get("ref") or ""
                ).strip()
                text = str(item.get("text") or item.get("value") or "").strip()
                if marker and text:
                    out.append({"id": marker, "text": text})
            return out
        return []


class VisionFullResponseSchema(VisionResponseCommonSchema):
    """Strict schema for normal Vision extraction output."""


class VisionSchemaContractError(RuntimeError):
    """Raised when OpenAI Structured Outputs schema contract is invalid."""


def _build_openai_json_schema() -> dict[str, Any]:
    """Build OpenAI json_schema format from Pydantic model for Structured Outputs (full schema only)."""
    schema = VisionFullResponseSchema.model_json_schema()
    props = schema.get("properties", {})
    defs = schema.get("$defs", {})
    required = list(props.keys())
    strict_schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
    if isinstance(defs, dict) and defs:
        # OpenAI strict mode: every object must have required == all properties.
        # Pydantic omits fields with defaults from required — fix that recursively.
        for def_name, def_schema in defs.items():
            if isinstance(def_schema, dict) and def_schema.get("type") == "object":
                def_props = def_schema.get("properties", {})
                if isinstance(def_props, dict):
                    def_schema["required"] = list(def_props.keys())
        strict_schema["$defs"] = defs

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vision_full_extraction",
            "strict": True,
            "schema": strict_schema,
        },
    }


def _validate_openai_strict_schema_contract(schema: dict[str, Any]) -> None:
    """Validate local Structured Outputs strict contract before API call."""
    try:
        if schema.get("type") != "json_schema":
            raise VisionSchemaContractError("schema.type must be 'json_schema'")
        json_schema = schema["json_schema"]
        block = json_schema["schema"]
        properties = block["properties"]
        required = block["required"]
    except Exception as exc:
        raise VisionSchemaContractError(
            f"schema malformed for Structured Outputs: {exc}"
        ) from exc

    if not isinstance(properties, dict):
        raise VisionSchemaContractError("schema.properties must be a dict")
    if not isinstance(required, list):
        raise VisionSchemaContractError("schema.required must be a list")

    prop_keys = set(properties.keys())
    req_keys = {str(k) for k in required}
    if prop_keys != req_keys:
        missing = sorted(prop_keys - req_keys)
        extra = sorted(req_keys - prop_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing_in_required={missing}")
        if extra:
            details.append(f"unknown_in_required={extra}")
        joined = ", ".join(details) if details else "required/properties mismatch"
        raise VisionSchemaContractError(
            "Structured Outputs strict contract invalid: "
            f"required must exactly match properties ({joined})"
        )
    _validate_no_map_like_objects(block, path="$")


def _validate_no_map_like_objects(node: Any, path: str) -> None:
    if not isinstance(node, dict):
        return
    node_type = node.get("type")
    if node_type == "object":
        # OpenAI strict schema handling is fragile with map-like additionalProperties objects.
        if "additionalProperties" in node and node.get("additionalProperties") not in (
            False,
            None,
        ):
            raise VisionSchemaContractError(
                f"Structured Outputs strict contract invalid: map-like object not allowed at {path}"
            )
        props = node.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                _validate_no_map_like_objects(sub, f"{path}.properties.{key}")
    if node_type == "array":
        _validate_no_map_like_objects(node.get("items"), f"{path}.items")
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for idx, sub in enumerate(variants):
                _validate_no_map_like_objects(sub, f"{path}.{key}[{idx}]")
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for key, sub in defs.items():
            _validate_no_map_like_objects(sub, f"{path}.$defs.{key}")


def _classify_openai_error(exc: Exception) -> str:
    """Classify OpenAI API error to choose deterministic handling."""
    msg = str(exc).lower()
    if (
        "could not parse the json body of your request" in msg
        or "what was sent was not valid json" in msg
        or ("json body" in msg and "not valid json" in msg)
    ):
        return "request_body_invalid"
    if "invalid schema for response_format" in msg or (
        "missing '" in msg and "response_format" in msg and "required" in msg
    ):
        return "schema_contract_invalid"
    if (
        "json_schema" in msg
        and "response_format" in msg
        and ("not supported" in msg or "unsupported" in msg)
    ):
        return "structured_output_unsupported"
    if "rate" in msg and "limit" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "connect" in msg or "network" in msg:
        return "connection"
    if "max_tokens" in msg and ("too large" in msg or "at most" in msg):
        return "max_tokens_too_large"
    return "other"


@dataclass
class VisionFullResult:
    """Result of Vision minimal extraction."""

    table_title: str
    table_summary: str
    headers: list[str]
    indicators: list[str]
    footnotes_content: list[dict[str, str]]
    no_table_detected: bool = False
    extraction_method: str = _EXTRACTION_METHOD
    vision_status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    retry_reasons: list[str] = field(default_factory=list)
    requested_max_completion_tokens: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    rescue_used: bool = False
    recrop_attempted: bool = False
    recrop_used: bool = False
    recrop_failed_incomplete: bool = False
    extraction_status: str = "ok"
    acceptance_reason: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    selected_candidate_name: str | None = None
    no_table_evidence_count: int = 0
    summary_present: bool = False
    indicator_count: int = 0
    candidate_quality_rank: list[int] = field(default_factory=list)
    qa_inspected: bool = False

    def to_footnotes_list(self) -> list[dict[str, str]]:
        return list(self.footnotes_content)


def _build_prompt(
    bank_code: str,
    vision_cfg: dict[str, Any],
    reference_text: str | None = None,
    *,
    rescue_mode: bool = False,
) -> str:
    """Build prompt with bank-specific footnote marker hints and OCR reference text (always injected when provided)."""
    marker_type = str(vision_cfg.get("footnote_marker_type", "")).strip().lower()
    expected = vision_cfg.get("expected_markers")
    hints = []
    if marker_type == "parenthetical":
        hints.append("Format attendu: parenthesique (1), (2), (3)")
    elif marker_type == "superscript":
        hints.append("Format attendu: superscript ou chiffres 1, 2, 3 (ou 1 2 3 4 5)")
    if expected and isinstance(expected, list):
        hints.append(f"Marqueurs possibles: {expected[:5]}")
    suffix = "\n".join(hints) if hints else ""

    # Multimodal Grounding: always inject OCR reference text when provided (precision for indicators)
    reference_section = ""
    reference_text_max_chars = int(
        vision_cfg.get(
            "vision_reference_text_max_chars", _DEFAULT_REFERENCE_TEXT_MAX_CHARS
        )
    )
    if (
        reference_text
        and len(reference_text.strip()) > 20
        and reference_text_max_chars > 0
    ):
        truncated = reference_text.strip()[:reference_text_max_chars]
        reference_section = (
            "\n\n=== DICTIONNAIRE DE RÉFÉRENCE (Texte OCR du tableau) ===\n"
            f"{truncated}\n"
            "=== FIN DICTIONNAIRE ===\n\n"
            "CONSIGNE : Utilise l'image pour l'ordre visuel et la structure du tableau. "
            "Utilise le Dictionnaire de Référence ci-dessus UNIQUEMENT pour VÉRIFIER L'ORTHOGRAPHE EXACTE "
            "des libellés d'indicateurs, en-têtes et notes de bas de page APRÈS avoir identifié les bonnes cellules dans l'image. "
            "Ne traite jamais ce dictionnaire comme une liste d'indicateurs à recopier. "
            "Dans un tableau multi-colonnes textuel, si le dictionnaire contient du texte des colonnes 2+, ce texte doit être ignoré pour le champ indicators. "
            "Transcris un libellé d'indicateur à l'identique du dictionnaire seulement s'il provient réellement de la cellule la plus à gauche dans l'image ; "
            "ne modifie pas la casse, la ponctuation ni les espaces. "
            "Si un libellé long est renvoyé sur 2 ou plusieurs lignes visuelles, reconstruis-le comme un seul indicateur logique et ne le coupe jamais en deux indicateurs distincts, car cela crée des faux positifs en aval. "
            "En cas de conflit entre l'image et le dictionnaire, privilégie l'orthographe du dictionnaire.\n"
        )
    else:
        logger.debug(
            "Vision: no reference text provided or too short; extraction without dictionary."
        )
        reference_section = (
            "\n\nCONSIGNE (pas de dictionnaire OCR disponible) : "
            "Transcris EXACTEMENT ce que tu vois dans l'image. "
            "Ne corrige PAS l'orthographe, la casse, la ponctuation ni les espaces des libelles. "
            "Si un libellé long occupe 2 ou plusieurs lignes visuelles, retourne un seul indicateur logique et ne le scinde jamais en deux indicateurs. "
            "Conserve les accents, les tirets et les caracteres speciaux tels quels.\n"
        )

    prompt = (
        _PROMPT_BASE
        + (f"\n{suffix}\n" if suffix else "")
        + reference_section
        + _PROMPT_JSON_STRICT
    )
    if rescue_mode:
        prompt = prompt + _PROMPT_RESCUE_SUFFIX
    return prompt


def _build_content(prompt: str, image_b64: str) -> list[Any]:
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
                "detail": "high",
            },
        },
    ]


def _build_repair_prompt(base_prompt: str, raw_content: str) -> str:
    return (
        "Le contenu precedent n'etait pas exploitable. "
        "Retourne UNIQUEMENT un objet JSON valide conforme au schema demande, "
        "sans markdown, sans commentaire, sans texte avant ou apres.\n\n"
        f"{base_prompt}\n\n"
        "Reponse precedente a corriger:\n"
        f"{raw_content[:2000]}"
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from GPT response and find JSON boundaries."""
    stripped = text.strip()

    # Étape 1 : Retirer les balises markdown si présentes
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    # Étape 2 : Chercher l'objet JSON (accolades)
    # L'API Vision / JSON mode retourne toujours un objet (dictionnaire) dans ce contexte
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]

    return stripped


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """Parse JSON from response. Returns None on failure."""
    try:
        cleaned = _strip_markdown_fences(raw)
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _preview_response_text(raw: str, limit: int = 500) -> str:
    """Return a compact response preview suitable for logs."""
    text = (raw or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head} ... {tail}"


def _extract_usage_metrics(response: Any) -> tuple[int | None, int | None, int | None]:
    """Best-effort extraction of token usage from OpenAI responses."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return (
        _coerce_int(getattr(usage, "prompt_tokens", None)),
        _coerce_int(getattr(usage, "completion_tokens", None)),
        _coerce_int(getattr(usage, "total_tokens", None)),
    )


def _with_attempt_metadata(
    result: VisionFullResult,
    *,
    requested_max_completion_tokens: int,
    finish_reason: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    rescue_used: bool = False,
) -> VisionFullResult:
    """Return a result carrying per-attempt metadata."""
    return replace(
        result,
        requested_max_completion_tokens=requested_max_completion_tokens,
        finish_reason=finish_reason or None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        rescue_used=rescue_used,
    )


def _make_truncated_placeholder_result(
    *,
    requested_max_completion_tokens: int,
    finish_reason: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> VisionFullResult:
    """Return a minimal partial result so higher-level rescue logic can retry with more budget."""
    return VisionFullResult(
        table_title="",
        table_summary="",
        headers=[],
        indicators=[],
        footnotes_content=[],
        extraction_method=_EXTRACTION_METHOD,
        vision_status="partial",
        warnings=["vision_truncated"],
        retry_reasons=["output_budget_truncated"],
        requested_max_completion_tokens=requested_max_completion_tokens,
        finish_reason=finish_reason or None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


_FULL_RESPONSE_KEYS = frozenset(
    {
        "table_title",
        "table_summary",
        "headers",
        "indicators",
        "footnotes_content",
        "no_table_detected",
    }
)
_FULL_REQUIRED_KEYS = frozenset({"table_summary", "indicators"})


def _extract_embedded_schema_candidate(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return the most likely nested payload matching the expected Vision full schema."""
    if not isinstance(raw, dict):
        return None
    response_keys = _FULL_RESPONSE_KEYS
    required_keys = _FULL_REQUIRED_KEYS
    raw_keys = set(raw.keys())
    if required_keys.issubset(raw_keys):
        return raw

    best_candidate: dict[str, Any] | None = None
    best_score = 0
    queue: list[dict[str, Any]] = [raw]
    seen_ids: set[int] = set()

    while queue:
        candidate = queue.pop(0)
        obj_id = id(candidate)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        keys = set(candidate.keys())
        score = len(keys & response_keys)
        if required_keys.issubset(keys):
            return candidate
        if score > best_score:
            best_candidate = candidate
            best_score = score

        for value in candidate.values():
            if isinstance(value, dict):
                queue.append(value)

    if best_candidate is not None and best_score >= 2:
        return best_candidate
    return None


def _parse_vision_result(
    raw: str | dict[str, Any],
) -> VisionFullResult | None:
    """Parse and validate JSON into VisionFullResult via Pydantic. Returns None on validation error."""
    try:
        if isinstance(raw, dict):
            validated = VisionFullResponseSchema.model_validate(raw)
        else:
            validated = VisionFullResponseSchema.model_validate_json(raw)
    except Exception as e:
        if isinstance(raw, dict):
            candidate = _extract_embedded_schema_candidate(raw)
            if candidate is not None and candidate is not raw:
                try:
                    validated = VisionFullResponseSchema.model_validate(candidate)
                    logger.info(
                        "Vision response recovered from nested wrapper keys: %s",
                        list(raw.keys())[:3],
                    )
                except Exception:
                    logger.warning(
                        "Vision response validation failed (Pydantic schema error): %s",
                        e,
                    )
                    return None
            else:
                logger.warning(
                    "Vision response validation failed (Pydantic schema error): %s", e
                )
                return None
        else:
            logger.warning(
                "Vision response validation failed (Pydantic schema error): %s", e
            )
            return None

    footnotes_ordered: list[dict[str, str]] = [
        {"id": str(item.id).strip(), "text": str(item.text).strip()}
        for item in validated.footnotes_content
        if str(item.id).strip() and str(item.text).strip()
    ]
    indicators_ordered = [
        str(item).rstrip() for item in validated.indicators if str(item).rstrip()
    ]

    return VisionFullResult(
        table_title=validated.table_title or "",
        table_summary=validated.table_summary or "",
        headers=validated.headers or [],
        indicators=indicators_ordered,
        footnotes_content=footnotes_ordered,
        no_table_detected=bool(validated.no_table_detected),
        extraction_method=_EXTRACTION_METHOD,
        vision_status="ok",
        warnings=[],
    )


def _try_parse_truncated_result(raw_content: str) -> VisionFullResult | None:
    """Best-effort parse of truncated JSON for the minimal extraction contract."""
    data = _parse_json_response(raw_content)
    if not data or not isinstance(data, dict):
        return None
    indicators_raw = data.get("indicators")
    if indicators_raw is None:
        return None
    if not isinstance(indicators_raw, list):
        return None
    indicators_ordered: list[str] = [
        str(item).strip() for item in indicators_raw if str(item).strip()
    ]
    footnotes_content: list[dict[str, str]] = []
    try:
        fn_raw = data.get("footnotes_content")
        if isinstance(fn_raw, list):
            for item in fn_raw:
                if not isinstance(item, dict):
                    continue
                marker = str(
                    item.get("id") or item.get("marker") or item.get("ref") or ""
                ).strip()
                text = str(item.get("text") or item.get("value") or "").strip()
                if marker and text:
                    footnotes_content.append({"id": marker, "text": text})
        elif isinstance(fn_raw, dict):
            for k, v in fn_raw.items():
                marker = str(k).strip()
                text = str(v).strip()
                if marker and text:
                    footnotes_content.append({"id": marker, "text": text})
    except Exception:
        footnotes_content = []

    return VisionFullResult(
        table_title=str(data.get("table_title") or "").strip(),
        table_summary=str(data.get("table_summary") or "").strip(),
        headers=[str(x).strip() for x in data.get("headers") or []],
        indicators=indicators_ordered,
        footnotes_content=footnotes_content,
        no_table_detected=bool(data.get("no_table_detected", False)),
        extraction_method=_EXTRACTION_METHOD,
        vision_status="partial",
        warnings=["vision_truncated"],
        retry_reasons=["output_budget_truncated"],
    )


def _is_generic_page_title(value: str) -> bool:
    title = " ".join(str(value or "").strip().lower().split())
    return title in _GENERIC_PAGE_TITLES


def _bbox_area(bbox_norm: list[float] | None) -> float:
    if not bbox_norm or len(bbox_norm) < 4:
        return 0.0
    try:
        left, top, right, bottom = [float(v) for v in bbox_norm[:4]]
    except (TypeError, ValueError):
        return 0.0
    if right <= left or bottom <= top:
        return 0.0
    return max(0.0, (right - left) * (bottom - top))


def _is_trivial_result(
    result: VisionFullResult | None,
    *,
    bbox_norm: list[float] | None = None,
) -> bool:
    if result is None:
        return True
    indicators = [
        str(v).strip() for v in list(result.indicators or []) if str(v).strip()
    ]
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    summary = str(result.table_summary or "").strip()
    title = str(result.table_title or "").strip()
    if result.no_table_detected and not indicators and not headers:
        return True
    if indicators:
        return False
    if headers or summary:
        return False
    if _is_generic_page_title(title):
        return True
    return _bbox_area(bbox_norm) >= 0.12 or not title


def _normalized_signal_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_period_like_indicator(text: str) -> bool:
    normalized = _normalized_signal_text(text)
    if not normalized:
        return True
    if re.fullmatch(r"(?:q|t)\s*[1-4](?:\s*\d{4})?", normalized):
        return True
    if re.fullmatch(r"\d{4}", normalized):
        return True
    return False


def _is_weak_indicator(text: str) -> bool:
    normalized = _normalized_signal_text(text)
    if not normalized:
        return True
    if normalized in _WEAK_INDICATOR_EXACT:
        return True
    tokens = normalized.split()
    if 0 < len(tokens) <= 3 and all(
        token in _WEAK_INDICATOR_TOKENS for token in tokens
    ):
        return True
    if _is_period_like_indicator(normalized):
        return True
    return False


def _looks_narrative_indicator(text: str) -> bool:
    normalized = _normalized_signal_text(text)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _NARRATIVE_INDICATOR_PHRASES):
        return True
    if normalized.endswith("."):
        return True
    tokens = normalized.split()
    return len(tokens) >= 6 and any(
        phrase in normalized
        for phrase in (" la ", " le ", " les ", " des ", " du ", " au ", " aux ")
    )


def _looks_compact_textual_header(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    words = normalized.split()
    if len(words) == 0 or len(words) > 4:
        return False
    if len(normalized) > 28:
        return False
    return any(char.isalpha() for char in normalized)


def _has_multi_textual_headers(result: VisionFullResult | None) -> bool:
    if result is None:
        return False
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    if len(headers) < 3:
        return False
    return all(_looks_compact_textual_header(header) for header in headers[:3])


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+", str(text or ""), flags=re.UNICODE))


def _looks_like_right_column_bleed_indicator(text: str) -> bool:
    normalized = _normalized_signal_text(text)
    if not normalized:
        return False
    token_count = _token_count(normalized)
    if token_count >= 10:
        return True
    if token_count >= 7 and any(char.isdigit() for char in normalized):
        return True
    if token_count >= 7 and "(" in text and ")" in text:
        return True
    if token_count >= 8 and normalized.count(" de ") + normalized.count(" du ") >= 2:
        return True
    return False


def _right_column_bleed_score(
    result: VisionFullResult | None,
    *,
    baseline_result: VisionFullResult | None = None,
) -> int:
    if result is None or baseline_result is None:
        return 0
    if not _has_multi_textual_headers(result):
        return 0
    indicators = [
        str(v).strip() for v in list(result.indicators or []) if str(v).strip()
    ]
    baseline_indicators = [
        str(v).strip() for v in list(baseline_result.indicators or []) if str(v).strip()
    ]
    if len(indicators) <= len(baseline_indicators) or not baseline_indicators:
        return 0
    if indicators[: len(baseline_indicators)] != baseline_indicators:
        return 0
    tail = indicators[len(baseline_indicators) :]
    if len(tail) < 2:
        return 0
    suspicious_tail = [
        item for item in tail if _looks_like_right_column_bleed_indicator(item)
    ]
    if not suspicious_tail:
        return 0
    baseline_avg_tokens = sum(_token_count(item) for item in baseline_indicators) / len(
        baseline_indicators
    )
    tail_avg_tokens = sum(_token_count(item) for item in tail) / len(tail)
    if tail_avg_tokens < baseline_avg_tokens + 2.5:
        return 0
    return len(suspicious_tail) + max(0, len(tail) - 1)


def _viable_indicator_count(result: VisionFullResult | None) -> int:
    if result is None:
        return 0
    count = 0
    for raw in list(result.indicators or []):
        text = str(raw or "").strip()
        if not text:
            continue
        if _is_generic_page_title(text):
            continue
        if not any(char.isalpha() for char in text):
            continue
        if _is_weak_indicator(text):
            continue
        if _looks_narrative_indicator(text):
            continue
        count += 1
    return count


def _weak_indicator_count(result: VisionFullResult | None) -> int:
    if result is None:
        return 0
    return sum(
        1
        for raw in list(result.indicators or [])
        if str(raw or "").strip() and _is_weak_indicator(str(raw or "").strip())
    )


def _narrative_indicator_count(result: VisionFullResult | None) -> int:
    if result is None:
        return 0
    return sum(
        1
        for raw in list(result.indicators or [])
        if str(raw or "").strip() and _looks_narrative_indicator(str(raw or "").strip())
    )


def _contamination_score(result: VisionFullResult | None) -> int:
    if result is None:
        return 99
    title = str(result.table_title or "").strip()
    score = 0
    if title and _is_generic_page_title(title):
        score += 2
    score += _weak_indicator_count(result)
    score += 2 * _narrative_indicator_count(result)
    return score


def _has_dominant_contamination(result: VisionFullResult | None) -> bool:
    if result is None:
        return True
    viable_count = _viable_indicator_count(result)
    score = _contamination_score(result)
    return viable_count <= 0 or score >= (viable_count + 1)


def _has_generic_title_without_support(result: VisionFullResult | None) -> bool:
    if result is None:
        return True
    title = str(result.table_title or "").strip()
    if not title or not _is_generic_page_title(title):
        return False
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    footnotes = [
        item
        for item in list(result.footnotes_content or [])
        if isinstance(item, dict)
        and (str(item.get("id") or "").strip() or str(item.get("text") or "").strip())
    ]
    return len(headers) < 2 and not footnotes


def _has_strong_non_summary_signals(result: VisionFullResult | None) -> bool:
    if result is None:
        return False
    if _has_dominant_contamination(result):
        return False
    viable_indicators = _viable_indicator_count(result)
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    footnotes = [
        item
        for item in list(result.footnotes_content or [])
        if isinstance(item, dict)
        and (str(item.get("id") or "").strip() or str(item.get("text") or "").strip())
    ]
    title = str(result.table_title or "").strip()
    if viable_indicators >= 3:
        return True
    if viable_indicators >= 2 and (
        headers or footnotes or (title and not _is_generic_page_title(title))
    ):
        return True
    if viable_indicators >= 1 and len(headers) >= 2 and bool(footnotes):
        return True
    return False


def _is_viable_result(result: VisionFullResult | None) -> bool:
    if result is None:
        return False
    if result.no_table_detected:
        return False
    return _viable_indicator_count(result) > 0 and not _has_dominant_contamination(
        result
    )


_DATE_RE = re.compile(
    r"^\s*(au\s+\d{1,2}\s+\w+\s+\d{4}|\d{1,2}\s+\w+\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|[tTqQ][1-4]\s*\d{4}|\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4})\s*$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"^\s*[\(\-]?[\d\s.,]+[\)%]?\s*$")


def _count_real_indicators(indicators: list[Any]) -> int:
    """Count indicators that are real business labels (not dates, numbers, or blanks)."""
    count = 0
    for ind in indicators:
        text = str(ind).strip() if isinstance(ind, str) else str(ind).strip()
        if not text:
            continue
        if _DATE_RE.match(text):
            continue
        if _NUMBER_RE.match(text):
            continue
        count += 1
    return count


def _grade_extraction_quality(result: VisionFullResult | None) -> list[str]:
    import re

    if result is None:
        return []

    critiques: list[str] = []
    indicators: list[Any] = result.indicators or []
    footnotes: list[Any] = result.footnotes_content or []

    # 1. Flat Indentation Check
    # Only fire when the table is large enough AND contains "Total" lines
    # (which strongly imply a parent-child hierarchy that was flattened).
    if len(indicators) > 15:
        has_indent = any(
            str(ind).startswith(" ") for ind in indicators if isinstance(ind, str)
        )
        has_total_lines = any(
            "total" in str(ind).lower() for ind in indicators if isinstance(ind, str)
        )
        if not has_indent and has_total_lines:
            critiques.append(
                "Vous avez complètement perdu la hiérarchie et l'indentation. "
                "Chaque sous-catégorie DOIT obligatoirement commencer par des espaces de tête (ex: '  Hypothèques résidentielles'). "
                "Ne mettez jamais tous les indicateurs à niveau."
            )

    # 2. Orphaned Footnote Marker Check
    found_footnote_ids = {
        str(fn.get("id")).strip()
        for fn in footnotes
        if isinstance(fn, dict) and str(fn.get("id", "")).strip()
    }

    referenced_markers = set()
    for ind in indicators:
        if isinstance(ind, str):
            matches = re.findall(r"\(([a-zA-Z0-9]{1,2})\)", ind)
            for m in matches:
                referenced_markers.add(m.strip())

    missing_footnotes = referenced_markers - found_footnote_ids
    if missing_footnotes and len(missing_footnotes) <= 4:
        missing_str = ", ".join(f"({m})" for m in missing_footnotes)
        critiques.append(
            f"Vous avez extrait des renvois de notes de bas de page dans les indicateurs [{missing_str}], "
            "mais vous avez OUBLIÉ d'inclure le texte de ces notes dans le champ 'footnotes_content'. "
            "Lisez le bas du tableau et ajoutez obligatoirement ces notes manquantes."
        )

    # 3. Missing Headers Check
    if len(result.headers or []) == 0 and len(indicators) > 3:
        title = str(result.table_title or "").lower()
        if "au 31" in title or "trimestre" in title or "202" in title:
            critiques.append(
                "Ce tableau semble contenir des colonnes de dates ou de périodes, mais le champ 'headers' est vide. "
                "Vous devez absolument renseigner les en-têtes de colonnes."
            )

    # 4. Truncated Extraction Check (many headers, few real indicators)
    headers = result.headers or []
    real_indicator_count = _count_real_indicators(indicators)
    if len(headers) >= 3 and real_indicator_count <= 2:
        critiques.append(
            f"Vous avez extrait {len(headers)} headers de colonnes mais seulement "
            f"{real_indicator_count} indicateur(s) réel(s). Un tableau avec {len(headers)} colonnes "
            "a forcément beaucoup plus de lignes de données. "
            "Relisez l'image ENTIÈREMENT du haut vers le bas et extrayez TOUTES les lignes."
        )

    return critiques


def _collect_incompleteness_reasons(
    result: VisionFullResult | None,
    *,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
) -> list[str]:
    if result is None:
        return ["missing_result"]
    reasons: list[str] = []
    expected_ids = expected_footnote_ids or set()
    title = str(result.table_title or "").strip()
    summary = str(result.table_summary or "").strip()
    if "output_budget_truncated" in set(result.retry_reasons or []):
        reasons.append("output_budget_truncated")
    if _viable_indicator_count(result) == 0:
        reasons.append("no_viable_indicators")
    if not summary:
        reasons.append("missing_table_summary")
    if title and _is_generic_page_title(title):
        reasons.append("generic_page_title")
    if _has_generic_title_without_support(result):
        reasons.append("generic_title_without_support")
        reasons.append("dominant_contamination")
    if _narrative_indicator_count(result) > 0:
        reasons.append("narrative_indicator_contamination")
    if _weak_indicator_count(result) > 0 and _viable_indicator_count(result) == 0:
        reasons.append("weak_indicator_only")
    if _has_dominant_contamination(result):
        reasons.append("dominant_contamination")
    if bbox_norm and len(bbox_norm) >= 4 and bbox_norm[1] < 0.15 and not title:
        reasons.append("top_context_missing_title")

    if bbox_norm and len(bbox_norm) >= 4:
        bbox_height = max(0.0, float(bbox_norm[3]) - float(bbox_norm[1]))
        viable = _viable_indicator_count(result)
        if bbox_height > 0.25 and viable <= 2:
            reasons.append("low_density_vertical")

    if bbox_norm and len(bbox_norm) >= 4 and bbox_norm[3] < 0.92 and expected_ids:
        found_ids = {
            str(item.get("id") or "").strip()
            for item in list(result.footnotes_content or [])
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        if not expected_ids.issubset(found_ids):
            reasons.append("missing_expected_footnotes")
    return reasons


def _candidate_quality_score(
    result: VisionFullResult | None,
    *,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
    baseline_result: VisionFullResult | None = None,
) -> tuple[int, int, int, int, int, int, int, int]:
    if result is None:
        return (0, 0, 0, 0, 0, 0, 0, 0)
    viable_indicators = _viable_indicator_count(result)
    headers = [str(v).strip() for v in list(result.headers or []) if str(v).strip()]
    title = str(result.table_title or "").strip()
    summary = str(result.table_summary or "").strip()
    found_footnotes = {
        str(item.get("id") or "").strip()
        for item in list(result.footnotes_content or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    expected_ids = expected_footnote_ids or set()
    right_column_bleed = _right_column_bleed_score(
        result,
        baseline_result=baseline_result,
    )
    return (
        1 if _is_viable_result(result) else 0,
        -right_column_bleed,
        -_contamination_score(result),
        1 if summary else 0,
        viable_indicators,
        len(found_footnotes & expected_ids),
        len(headers),
        0 if _is_generic_page_title(title) else (1 if title else 0),
    )


def _finalize_selected_candidate(
    selected: VisionFullResult,
    *,
    best_name: str,
    initial_rejection_reasons: list[str],
    no_table_evidence_count: int,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
) -> VisionFullResult | None:
    selected_rejection_reasons = _collect_incompleteness_reasons(
        selected,
        bbox_norm=bbox_norm,
        expected_footnote_ids=expected_footnote_ids,
    )
    combined_rejection_reasons = list(
        dict.fromkeys(initial_rejection_reasons + selected_rejection_reasons)
    )
    summary_present = bool(str(selected.table_summary or "").strip())
    contamination_is_low = not _has_dominant_contamination(selected)
    viable_count = _viable_indicator_count(selected)
    headers = [str(v).strip() for v in list(selected.headers or []) if str(v).strip()]
    footnotes = [
        item
        for item in list(selected.footnotes_content or [])
        if isinstance(item, dict)
        and (str(item.get("id") or "").strip() or str(item.get("text") or "").strip())
    ]
    if _has_generic_title_without_support(selected):
        combined_rejection_reasons = list(
            dict.fromkeys(
                combined_rejection_reasons
                + ["generic_title_without_support", "dominant_contamination"]
            )
        )
        contamination_is_low = False

    if summary_present and contamination_is_low and viable_count > 0:
        acceptance_reason = (
            "initial_complete"
            if best_name == "initial" and not initial_rejection_reasons
            else "rescued_summary_recovered"
        )
        return _build_result_debug_metadata(
            selected,
            acceptance_reason=acceptance_reason,
            rejection_reasons=combined_rejection_reasons,
            selected_candidate_name=best_name,
            no_table_evidence_count=no_table_evidence_count,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_footnote_ids,
        )

    if (
        not summary_present
        and contamination_is_low
        and viable_count >= 3
        and (len(headers) >= 2 or bool(footnotes))
        and _has_strong_non_summary_signals(selected)
    ):
        return _build_result_debug_metadata(
            replace(selected, extraction_status="rescued"),
            acceptance_reason="rescued_without_summary_strong_structure",
            rejection_reasons=combined_rejection_reasons,
            selected_candidate_name=best_name,
            no_table_evidence_count=no_table_evidence_count,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_footnote_ids,
        )
    return None


def _build_result_debug_metadata(
    result: VisionFullResult,
    *,
    acceptance_reason: str,
    rejection_reasons: list[str],
    selected_candidate_name: str,
    no_table_evidence_count: int,
    bbox_norm: list[float] | None = None,
    expected_footnote_ids: set[str] | None = None,
) -> VisionFullResult:
    quality_rank = list(
        _candidate_quality_score(
            result,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_footnote_ids,
            baseline_result=result,
        )
    )
    return replace(
        result,
        acceptance_reason=acceptance_reason,
        rejection_reasons=list(dict.fromkeys(rejection_reasons)),
        selected_candidate_name=selected_candidate_name,
        no_table_evidence_count=max(0, int(no_table_evidence_count)),
        summary_present=bool(str(result.table_summary or "").strip()),
        indicator_count=len(
            [str(v).strip() for v in list(result.indicators or []) if str(v).strip()]
        ),
        candidate_quality_rank=quality_rank,
    )


def _cache_payload_from_result(result: VisionFullResult) -> dict[str, Any]:
    return {
        "table_title": result.table_title,
        "table_summary": result.table_summary,
        "headers": result.headers,
        "indicators": result.indicators,
        "footnotes_content": result.footnotes_content,
        "no_table_detected": result.no_table_detected,
        "vision_status": result.vision_status,
        "warnings": result.warnings,
        "retry_reasons": result.retry_reasons,
        "requested_max_completion_tokens": result.requested_max_completion_tokens,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "rescue_used": result.rescue_used,
        "extraction_status": result.extraction_status,
        "acceptance_reason": result.acceptance_reason,
        "rejection_reasons": result.rejection_reasons,
        "selected_candidate_name": result.selected_candidate_name,
        "no_table_evidence_count": result.no_table_evidence_count,
        "summary_present": result.summary_present,
        "indicator_count": result.indicator_count,
        "candidate_quality_rank": result.candidate_quality_rank,
    }


class VisionFullExtractor:
    """
    Extract indicators + footnotes from a table crop image via OpenAI Vision.

    One call per table minimum. Supports retry on invalid JSON, deterministic recrop,
    and cache via vision_cache (pdf_sha + page + bbox).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries_json: int = 2,
        use_cache: bool = False,
    ):
        self._api_key = api_key or get_openai_api_key()
        self._model = str(model or "").strip() or resolve_openai_model(_MODEL_ROLE)
        self._max_retries_json = max_retries_json
        self._use_cache = use_cache
        self._client: Any = None
        self._disabled_reason: str | None = None
        self._schema_contract_checked: set[str] = set()
        self._schema_contract_error_logged = False

    @property
    def model_name(self) -> str:
        """Return the resolved extraction model name."""
        return self._model

    @property
    def model_role(self) -> str:
        """Return the logical role used for model resolution."""
        return _MODEL_ROLE

    def _ensure_schema_validated(self, schema: dict[str, Any] | None = None) -> None:
        """Validate schema once and mark as checked. Raises VisionSchemaContractError if invalid."""
        if "full" in self._schema_contract_checked:
            return
        schema = schema if schema is not None else _build_openai_json_schema()
        try:
            _validate_openai_strict_schema_contract(schema)
            self._schema_contract_checked.add("full")
        except VisionSchemaContractError as exc:
            self._schema_contract_checked.add("full")
            self._disabled_reason = str(exc)
            if not self._schema_contract_error_logged:
                logger.error(
                    "Vision schema contract invalid (local validation): %s",
                    exc,
                )
                self._schema_contract_error_logged = True
            raise

    def validate_schema(self) -> None:
        """Pre-validate OpenAI Structured Outputs schema. Raises VisionSchemaContractError if invalid."""
        self._ensure_schema_validated()

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI

            if not self._api_key:
                raise ValueError("OPENAI_API_KEY required for Vision extraction")
            self._client = OpenAI(api_key=self._api_key)
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

    def extract(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        bottom_extension_used: float = 0.0,
        reference_text: str | None = None,
        max_completion_tokens_override: int | None = None,
        rescue_mode: bool = False,
        rescue_instruction: str = "",
        temperature: float = 0.0,
    ) -> VisionFullResult | None:
        """
        Extract indicators and footnotes from a table crop.

        Args:
            crop_bytes: PNG bytes of the cropped table image
            bank_code: Bank code for prompt hints (bnc, rbc, td, etc.)
            pdf_sha: PDF hash for cache key (optional)
            page_number: Page number for cache key (optional)
            bbox_norm: Normalized bbox for cache key (optional)
            vision_cfg: Config overrides (footnote_marker_type, expected_markers)
            bottom_extension_used: Bottom extension already applied (for cache key variant)
            max_completion_tokens_override: Explicit output budget for this extraction pass

        Returns:
            VisionFullResult or None on failure
        """
        if self._disabled_reason:
            raise VisionSchemaContractError(self._disabled_reason)

        vision_cfg = vision_cfg or {}
        cache_key = ""
        configured_max_completion_tokens = max(
            1,
            int(
                max_completion_tokens_override
                if max_completion_tokens_override is not None
                else vision_cfg.get(
                    "vision_max_completion_tokens",
                    vision_cfg.get(
                        "vision_max_completion_tokens_full",
                        _DEFAULT_MAX_COMPLETION_TOKENS,
                    ),
                )
            ),
        )
        configured_max_completion_tokens = min(
            configured_max_completion_tokens, _MAX_COMPLETION_TOKENS_API_LIMIT
        )

        if (
            self._use_cache
            and not rescue_mode
            and pdf_sha
            and page_number
            and bbox_norm
            and len(bbox_norm) == 4
        ):
            bbox_with_ext = list(bbox_norm)
            if len(bbox_with_ext) >= 4:
                bbox_with_ext[3] = min(1.0, bbox_with_ext[3] + bottom_extension_used)
            cache_key = make_cache_key(
                pdf_sha,
                page_number,
                bbox_with_ext,
                max_completion_tokens=configured_max_completion_tokens,
            )
            if cache_key:
                cache_dir = get_vision_cache_dir()
                cached = cache_get(cache_dir, cache_key)
                if cached:
                    indicators = cached.get("indicators")
                    fn_content_raw = cached.get("footnotes_content", [])
                    retry_reasons = [
                        str(value).strip()
                        for value in list(cached.get("retry_reasons", []) or [])
                        if str(value).strip()
                    ]
                    if isinstance(indicators, list):
                        migrated_indicators: list[str] = []
                        for item in indicators:
                            if isinstance(item, str) and item.strip():
                                migrated_indicators.append(item.strip())
                            elif isinstance(item, dict) and item.get("text"):
                                migrated_indicators.append(
                                    str(item.get("text") or "").strip()
                                )
                        indicators = migrated_indicators
                    else:
                        indicators = None
                    if isinstance(indicators, list):
                        if isinstance(fn_content_raw, dict):
                            fn_content: list[dict[str, str]] = [
                                {"id": str(k), "text": str(v)}
                                for k, v in fn_content_raw.items()
                                if str(k).strip() and str(v).strip()
                            ]
                        elif isinstance(fn_content_raw, list):
                            fn_content = [
                                item
                                for item in fn_content_raw
                                if isinstance(item, dict)
                                and (item.get("id") or item.get("marker"))
                                and item.get("text")
                            ]
                            fn_content = [
                                {
                                    "id": str(
                                        item.get("id") or item.get("marker") or ""
                                    ).strip(),
                                    "text": str(item.get("text", "")).strip(),
                                }
                                for item in fn_content
                            ]
                        else:
                            fn_content = []
                        logger.info(
                            "VisionFull cache hit: %d indicators",
                            len(indicators),
                        )
                        return VisionFullResult(
                            table_title=str(cached.get("table_title") or ""),
                            table_summary=str(cached.get("table_summary") or ""),
                            headers=list(cached.get("headers") or []),
                            indicators=indicators,
                            footnotes_content=fn_content,
                            no_table_detected=bool(
                                cached.get("no_table_detected", False)
                            ),
                            extraction_method=_EXTRACTION_METHOD,
                            vision_status=str(cached.get("vision_status") or "ok"),
                            warnings=list(cached.get("warnings") or []),
                            retry_reasons=retry_reasons,
                            requested_max_completion_tokens=(
                                int(cached.get("requested_max_completion_tokens"))
                                if cached.get("requested_max_completion_tokens")
                                is not None
                                else configured_max_completion_tokens
                            ),
                            finish_reason=(
                                str(cached.get("finish_reason"))
                                if cached.get("finish_reason") is not None
                                else None
                            ),
                            prompt_tokens=(
                                int(cached.get("prompt_tokens"))
                                if cached.get("prompt_tokens") is not None
                                else None
                            ),
                            completion_tokens=(
                                int(cached.get("completion_tokens"))
                                if cached.get("completion_tokens") is not None
                                else None
                            ),
                            total_tokens=(
                                int(cached.get("total_tokens"))
                                if cached.get("total_tokens") is not None
                                else None
                            ),
                            rescue_used=bool(cached.get("rescue_used", False)),
                            extraction_status=str(
                                cached.get("extraction_status") or "ok"
                            ).strip()
                            or "ok",
                            acceptance_reason=(
                                str(cached.get("acceptance_reason")).strip()
                                if cached.get("acceptance_reason") is not None
                                else None
                            ),
                            rejection_reasons=[
                                str(value).strip()
                                for value in list(
                                    cached.get("rejection_reasons", []) or []
                                )
                                if str(value).strip()
                            ],
                            selected_candidate_name=(
                                str(cached.get("selected_candidate_name")).strip()
                                if cached.get("selected_candidate_name") is not None
                                else None
                            ),
                            no_table_evidence_count=int(
                                cached.get("no_table_evidence_count") or 0
                            ),
                            summary_present=bool(cached.get("summary_present", False)),
                            indicator_count=int(cached.get("indicator_count") or 0),
                            candidate_quality_rank=[
                                int(value)
                                for value in list(
                                    cached.get("candidate_quality_rank", []) or []
                                )
                                if isinstance(value, (int, float))
                            ],
                        )

        try:
            self._ensure_client()
        except (ImportError, ValueError) as e:
            logger.warning("VisionFullExtractor: client init failed: %s", e)
            return None

        client = self._client
        if client is None:
            return None

        try:
            from .vision_image_preprocessor import preprocess_for_vision

            preprocess_flag = vision_cfg.get("vision_preprocess")
            preprocess_enabled: bool | None = None
            if preprocess_flag is not None:
                preprocess_enabled = str(preprocess_flag).strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
            processed = preprocess_for_vision(crop_bytes, enabled=preprocess_enabled)
            image_b64 = base64.standard_b64encode(processed).decode("ascii")
        except Exception as e:
            logger.debug("Vision preprocessing failed, using raw: %s", e)
            image_b64 = base64.standard_b64encode(crop_bytes).decode("ascii")

        prompt = _build_prompt(
            bank_code,
            vision_cfg,
            reference_text=reference_text,
            rescue_mode=rescue_mode,
        )
        max_completion_tokens = configured_max_completion_tokens
        openai_schema_full = _build_openai_json_schema()
        self._ensure_schema_validated(openai_schema_full)

        api_retry_max = int(vision_cfg.get("api_retry_max", 3))
        api_retry_backoff_ms = float(vision_cfg.get("api_retry_backoff_ms", 1000))

        def _issue_request(
            prompt_text: str,
            *,
            structured: bool,
            max_completion_tokens: int,
            label: str,
        ) -> tuple[str, str, bool, int, int | None, int | None, int | None] | None:
            local_use_structured = structured
            effective_max = max_completion_tokens
            transport_attempt = 0
            while transport_attempt <= api_retry_max:
                if transport_attempt > 0:
                    backoff_sec = (api_retry_backoff_ms / 1000.0) * (
                        2 ** (transport_attempt - 1)
                    )
                    logger.info(
                        "Vision %s: transport retry %s/%s after %.1fs backoff",
                        label,
                        transport_attempt,
                        api_retry_max,
                        backoff_sec,
                    )
                    time.sleep(backoff_sec)
                try:
                    response_format: dict[str, Any] = (
                        openai_schema_full
                        if local_use_structured
                        else {"type": "json_object"}
                    )
                    response = client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {
                                "role": "user",
                                "content": _build_content(prompt_text, image_b64),
                            }
                        ],
                        response_format=response_format,
                        temperature=temperature,
                        max_completion_tokens=effective_max,
                    )
                    prompt_tokens, completion_tokens, total_tokens = (
                        _extract_usage_metrics(response)
                    )
                    return (
                        response.choices[0].message.content or "",
                        str(getattr(response.choices[0], "finish_reason", "") or ""),
                        local_use_structured,
                        effective_max,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    )
                except Exception as e:
                    err_kind = _classify_openai_error(e)
                    if err_kind == "schema_contract_invalid":
                        self._disabled_reason = f"Vision schema contract invalid: {e}"
                        if not self._schema_contract_error_logged:
                            logger.error("%s", self._disabled_reason)
                            self._schema_contract_error_logged = True
                        raise VisionSchemaContractError(self._disabled_reason) from e
                    if (
                        err_kind == "max_tokens_too_large"
                        and effective_max > _MAX_COMPLETION_TOKENS_SAFE_FALLBACK
                    ):
                        logger.warning(
                            "Vision %s: model limits max_completion_tokens to %s; retrying with %s",
                            label,
                            effective_max,
                            _MAX_COMPLETION_TOKENS_SAFE_FALLBACK,
                        )
                        effective_max = _MAX_COMPLETION_TOKENS_SAFE_FALLBACK
                        continue
                    if local_use_structured and err_kind in (
                        "structured_output_unsupported",
                        "request_body_invalid",
                    ):
                        logger.debug(
                            "Structured Outputs unavailable for %s, falling back to json_object: %s",
                            label,
                            e,
                        )
                        local_use_structured = False
                        continue
                    if err_kind in ("rate_limit", "timeout", "connection"):
                        transport_attempt += 1
                        if transport_attempt <= api_retry_max:
                            continue
                        logger.warning(
                            "Vision %s API error (retries exhausted): %s",
                            label,
                            e,
                        )
                        return None
                    logger.warning("Vision %s API error: %s", label, e)
                    return None
            logger.warning("Vision %s: no successful API response", label)
            return None

        failure_causes: list[str] = []

        current_prompt = prompt
        max_self_healing_attempts = 2
        prev_real_indicator_count: int | None = None  # Quorum tracking

        for attempt in range(max_self_healing_attempts):
            issued = _issue_request(
                current_prompt,
                structured=True,
                max_completion_tokens=max_completion_tokens,
                label=f"full_pass_attempt_{attempt + 1}",
            )
            if issued is None:
                return None

            (
                raw_content,
                finish_reason,
                used_structured,
                effective_max_completion_tokens,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ) = issued

            if finish_reason == "length":
                failure_causes.append("vision_truncated")
                logger.warning(
                    "Vision full: response truncated (raw_len=%d)",
                    len(raw_content),
                )
                partial_result = _try_parse_truncated_result(raw_content)
                if partial_result is not None:
                    return _with_attempt_metadata(
                        partial_result,
                        requested_max_completion_tokens=effective_max_completion_tokens,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                return _make_truncated_placeholder_result(
                    requested_max_completion_tokens=effective_max_completion_tokens,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            data = _parse_json_response(raw_content)
            if data is None:
                failure_causes.append("vision_invalid_json")
                logger.info(
                    "Vision full: JSON parse failed (raw_len=%d, preview=%r)",
                    len(raw_content),
                    _preview_response_text(raw_content),
                )
                if self._max_retries_json >= 1:
                    repair_prompt = _build_repair_prompt(current_prompt, raw_content)
                    retry_issued = _issue_request(
                        repair_prompt,
                        structured=False,
                        max_completion_tokens=max_completion_tokens,
                        label="retry-json",
                    )
                    if retry_issued is not None:
                        (
                            retry_raw,
                            retry_reason,
                            _,
                            retry_effective_max_completion_tokens,
                            retry_prompt_tokens,
                            retry_completion_tokens,
                            retry_total_tokens,
                        ) = retry_issued
                        if retry_reason != "length":
                            retry_data = _parse_json_response(retry_raw)
                            if retry_data is not None:
                                result = _parse_vision_result(retry_data)
                                if result is not None:
                                    critiques = _grade_extraction_quality(result)
                                    if (
                                        critiques
                                        and attempt < max_self_healing_attempts - 1
                                    ):
                                        logger.warning(
                                            "Vision full: self-healing triggered after json-retry: %s",
                                            critiques,
                                        )
                                        current_prompt = (
                                            prompt
                                            + "\n\n### CRÉTIQUES DE LA TENTATIVE PRÉCÉDENTE, CORRIGEZ ###\n- "
                                            + "\n- ".join(critiques)
                                        )
                                        continue

                                    result = replace(
                                        result,
                                        vision_status="partial",
                                        warnings=list(
                                            dict.fromkeys(
                                                failure_causes
                                                + ["vision_structured_output_fallback"]
                                                + (
                                                    ["self_healing_failed"]
                                                    if critiques
                                                    else []
                                                )
                                            )
                                        ),
                                    )
                                    result = _with_attempt_metadata(
                                        result,
                                        requested_max_completion_tokens=(
                                            retry_effective_max_completion_tokens
                                        ),
                                        finish_reason=retry_reason,
                                        prompt_tokens=retry_prompt_tokens,
                                        completion_tokens=retry_completion_tokens,
                                        total_tokens=retry_total_tokens,
                                    )
                                    if self._use_cache and cache_key:
                                        cache_dir = get_vision_cache_dir()
                                        cache_put(
                                            cache_dir,
                                            cache_key,
                                            _cache_payload_from_result(result),
                                        )
                                    return result
                logger.warning(
                    "Vision full extraction: invalid content after retry (%s)",
                    ", ".join(
                        dict.fromkeys(failure_causes + ["vision_retry_exhausted"])
                    ),
                )
                return None

            result = _parse_vision_result(data)
            if result is None:
                failure_causes.append("vision_schema_validation_failed")
                logger.info(
                    "Vision full: schema validation failed (raw_len=%d, keys=%s)",
                    len(raw_content),
                    sorted(data.keys()),
                )
                if self._max_retries_json >= 1:
                    repair_prompt = _build_repair_prompt(current_prompt, raw_content)
                    retry_issued = _issue_request(
                        repair_prompt,
                        structured=False,
                        max_completion_tokens=max_completion_tokens,
                        label="retry-json",
                    )
                    if retry_issued is not None:
                        (
                            retry_raw,
                            retry_reason,
                            _,
                            retry_effective_max_completion_tokens,
                            retry_prompt_tokens,
                            retry_completion_tokens,
                            retry_total_tokens,
                        ) = retry_issued
                        if retry_reason != "length":
                            retry_data = _parse_json_response(retry_raw)
                            if retry_data is not None:
                                result = _parse_vision_result(retry_data)
                                if result is not None:
                                    critiques = _grade_extraction_quality(result)
                                    if (
                                        critiques
                                        and attempt < max_self_healing_attempts - 1
                                    ):
                                        logger.warning(
                                            "Vision full: self-healing triggered after json-retry: %s",
                                            critiques,
                                        )
                                        current_prompt = (
                                            prompt
                                            + "\n\n### CRÉTIQUES DE LA TENTATIVE PRÉCÉDENTE, CORRIGEZ ###\n- "
                                            + "\n- ".join(critiques)
                                        )
                                        continue

                                    result = replace(
                                        result,
                                        vision_status="partial",
                                        warnings=list(
                                            dict.fromkeys(
                                                failure_causes
                                                + ["vision_structured_output_fallback"]
                                                + (
                                                    ["self_healing_failed"]
                                                    if critiques
                                                    else []
                                                )
                                            )
                                        ),
                                    )
                                    result = _with_attempt_metadata(
                                        result,
                                        requested_max_completion_tokens=(
                                            retry_effective_max_completion_tokens
                                        ),
                                        finish_reason=retry_reason,
                                        prompt_tokens=retry_prompt_tokens,
                                        completion_tokens=retry_completion_tokens,
                                        total_tokens=retry_total_tokens,
                                    )
                                    if self._use_cache and cache_key:
                                        cache_dir = get_vision_cache_dir()
                                        cache_put(
                                            cache_dir,
                                            cache_key,
                                            _cache_payload_from_result(result),
                                        )
                                    return result
                logger.warning(
                    "Vision full extraction: invalid content after retry (%s)",
                    ", ".join(
                        dict.fromkeys(failure_causes + ["vision_retry_exhausted"])
                    ),
                )
                return None

            # --- Self-Healing Check ---
            critiques = _grade_extraction_quality(result)
            if critiques and attempt < max_self_healing_attempts - 1:
                # --- Quorum: if same indicator count as previous attempt, accept ---
                current_real_count = _count_real_indicators(result.indicators or [])
                if (
                    prev_real_indicator_count is not None
                    and abs(current_real_count - prev_real_indicator_count) <= 1
                ):
                    logger.info(
                        "Vision full: Quorum reached (attempt %d: %d real inds ≈ prev %d). "
                        "Accepting result despite critiques: %s",
                        attempt + 1,
                        current_real_count,
                        prev_real_indicator_count,
                        critiques,
                    )
                else:
                    prev_real_indicator_count = current_real_count
                    logger.warning("Vision full: self-healing triggered: %s", critiques)
                    current_prompt = (
                        prompt
                        + "\n\n### CRÉTIQUES DE LA TENTATIVE PRÉCÉDENTE, CORRIGEZ ###\n- "
                        + "\n- ".join(critiques)
                    )
                    continue

            if critiques:
                failure_causes.append("self_healing_failed")

            if used_structured is False:
                failure_causes.append("vision_structured_output_fallback")
            retry_reasons = list(result.retry_reasons or [])
            if (
                "vision_truncated" in failure_causes
                and "output_budget_truncated" not in retry_reasons
            ):
                retry_reasons.append("output_budget_truncated")
            if retry_reasons or failure_causes:
                result = replace(result, vision_status="partial")
            result = replace(
                result,
                warnings=list(dict.fromkeys(failure_causes)),
                retry_reasons=list(dict.fromkeys(retry_reasons)),
            )
            result = _with_attempt_metadata(
                result,
                requested_max_completion_tokens=effective_max_completion_tokens,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            if self._use_cache and cache_key:
                cache_dir = get_vision_cache_dir()
                cache_put(
                    cache_dir,
                    cache_key,
                    _cache_payload_from_result(result),
                )
            return result

        return None

    # ------------------------------------------------------------------
    # Multi-Shot Consensus extraction
    # ------------------------------------------------------------------

    _CONSENSUS_TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4)

    def extract_with_consensus(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        bottom_extension_used: float = 0.0,
        reference_text: str | None = None,
        max_completion_tokens_override: int | None = None,
        rescue_mode: bool = False,
        rescue_instruction: str = "",
        temperatures: tuple[float, ...] | None = None,
    ) -> VisionFullResult | None:
        """Multi-shot extraction with consensus voting.

        Dispatches ``len(temperatures)`` parallel extractions at different
        temperatures, then selects the result with the best consensus on
        indicator count and label overlap.

        Falls back to a single temperature-0 extraction when the consensus
        is unanimous or only one shot succeeds.
        """
        temps = temperatures or self._CONSENSUS_TEMPERATURES
        if len(temps) <= 1:
            return self.extract(
                crop_bytes=crop_bytes,
                bank_code=bank_code,
                pdf_sha=pdf_sha,
                page_number=page_number,
                bbox_norm=bbox_norm,
                vision_cfg=vision_cfg,
                bottom_extension_used=bottom_extension_used,
                reference_text=reference_text,
                max_completion_tokens_override=max_completion_tokens_override,
                rescue_mode=rescue_mode,
                rescue_instruction=rescue_instruction,
                temperature=temps[0],
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _shot(temp: float) -> VisionFullResult | None:
            return self.extract(
                crop_bytes=crop_bytes,
                bank_code=bank_code,
                pdf_sha=pdf_sha,
                page_number=page_number,
                bbox_norm=bbox_norm,
                vision_cfg=vision_cfg,
                bottom_extension_used=bottom_extension_used,
                reference_text=reference_text,
                max_completion_tokens_override=max_completion_tokens_override,
                rescue_mode=rescue_mode,
                rescue_instruction=rescue_instruction,
                temperature=temp,
            )

        results: list[VisionFullResult] = []
        with ThreadPoolExecutor(max_workers=len(temps)) as pool:
            futures = {pool.submit(_shot, t): t for t in temps}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r is not None and r.indicators:
                        results.append(r)
                except Exception as exc:
                    logger.debug(
                        "Consensus shot at temp=%.1f failed: %s",
                        futures[future],
                        exc,
                    )

        if not results:
            return None
        if len(results) == 1:
            return results[0]

        # --- Consensus voting ---
        return self._select_consensus(results, bbox_norm=bbox_norm)

    @staticmethod
    def _select_consensus(
        results: list[VisionFullResult],
        *,
        bbox_norm: list[float] | None = None,
    ) -> VisionFullResult:
        """Pick the result with the best consensus on indicator labels.

        Strategy:
        1. Compute *median* indicator count across all shots.
        2. Build a superset of normalised indicator labels from all shots and
           compute per-label vote counts.
        3. Score each result by: (a) closeness to median count, (b) sum of
           vote counts for its indicators (label popularity).
        4. Return the result with the highest composite score.
        """
        import statistics

        counts = [_count_real_indicators(r.indicators or []) for r in results]
        median_count = statistics.median(counts)

        # Normalise labels for comparison
        def _norm(label: str) -> str:
            return label.strip().lower()

        # Collect vote counts per normalised label
        label_votes: dict[str, int] = {}
        for r in results:
            for ind in r.indicators or []:
                key = _norm(str(ind))
                if key:
                    label_votes[key] = label_votes.get(key, 0) + 1

        def _score(r: VisionFullResult) -> float:
            real_count = _count_real_indicators(r.indicators or [])
            # Penalty for deviating from the median count
            count_penalty = abs(real_count - median_count)
            # Popularity: sum of votes for this result's labels
            popularity = sum(
                label_votes.get(_norm(str(ind)), 0) for ind in (r.indicators or [])
            )
            # Quality bonus from existing scoring function (tuple of ints)
            quality_tuple = _candidate_quality_score(
                r,
                bbox_norm=bbox_norm,
                expected_footnote_ids=set(),
                baseline_result=None,
            )
            return popularity - count_penalty * 3 + sum(quality_tuple) * 0.5

        best = max(results, key=_score)

        logger.info(
            "Consensus: selected result with %d indicators (median=%s, from %d shots)",
            _count_real_indicators(best.indicators or []),
            median_count,
            len(results),
        )
        return best

    def extract_with_quality_pass(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        initial_bottom_extension: float = 0.0,
        get_recrop_fn: Any = None,
        get_variant_crop_fn: Any = None,
        reference_text: str | None = None,
    ) -> VisionFullResult | None:
        """
        Extract with a deterministic quality pass.

        Flow:
        - initial crop at configured budget (64k by default)
        - same-crop rescue at 128k only when truncation is detected
        - optional recrop at configured budget
        - recrop rescue at 128k only when truncation is detected

        get_recrop_fn(bottom_extension: float) -> bytes | None
        """
        vision_cfg = vision_cfg or {}
        expected_markers = vision_cfg.get("expected_markers")
        if isinstance(expected_markers, list):
            expected_set = {str(m).strip() for m in expected_markers[:10]}
        else:
            expected_set = set()
        base_max_completion_tokens = max(
            1,
            min(
                int(
                    vision_cfg.get(
                        "vision_max_completion_tokens",
                        vision_cfg.get(
                            "vision_max_completion_tokens_full",
                            _DEFAULT_MAX_COMPLETION_TOKENS,
                        ),
                    )
                ),
                _MAX_COMPLETION_TOKENS_API_LIMIT,
            ),
        )
        rescue_enabled = bool(
            vision_cfg.get("vision_max_completion_tokens_rescue_enabled", False)
        )
        rescue_max_completion_tokens = max(
            1,
            min(
                int(
                    vision_cfg.get(
                        "vision_max_completion_tokens_rescue",
                        _RESCUE_MAX_COMPLETION_TOKENS,
                    )
                ),
                _MAX_COMPLETION_TOKENS_API_LIMIT,
            ),
        )

        def _footnote_ids(r: VisionFullResult) -> set[str]:
            return {
                str(item.get("id") or "").strip()
                for item in list(r.footnotes_content or [])
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            }

        def _extract_markers_from_indicators(indicators: list[str]) -> set[str]:
            import re

            markers = set()
            for text in indicators:
                text_lower = text.lower()
                for m in re.finditer(r"[²³*†‡]", text):
                    val = m.group(0)
                    if val == "²":
                        markers.add("2")
                    elif val == "³":
                        markers.add("3")
                    else:
                        markers.add(val)
                for m in re.finditer(r"\(\s*([0-9a-z])\s*\)", text_lower):
                    if m.start() <= 1:
                        prefix = text[: m.start()]
                        if not any(c.isalnum() for c in prefix):
                            continue
                    markers.add(m.group(1))
            return markers

        def _needs_recrop(result: VisionFullResult | None) -> bool:
            return bool(
                _collect_incompleteness_reasons(
                    result,
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                )
            )

        def _has_truncation_signal(result: VisionFullResult | None) -> bool:
            if result is None:
                return False
            if str(result.finish_reason or "").strip().lower() == "length":
                return True
            if "output_budget_truncated" in set(result.retry_reasons or []):
                return True
            return "vision_truncated" in {str(w).strip() for w in result.warnings or []}

        consensus_enabled = bool(vision_cfg.get("vision_consensus_enabled", False))

        def _run_pass(
            *,
            crop_bytes_for_pass: bytes,
            bottom_extension_used: float,
            rescue_mode: bool = False,
            rescue_instruction: str = "",
        ) -> VisionFullResult | None:
            if consensus_enabled and not rescue_mode:
                primary = self.extract_with_consensus(
                    crop_bytes=crop_bytes_for_pass,
                    bank_code=bank_code,
                    pdf_sha=pdf_sha,
                    page_number=page_number,
                    bbox_norm=bbox_norm,
                    vision_cfg=vision_cfg,
                    bottom_extension_used=bottom_extension_used,
                    reference_text=reference_text,
                    max_completion_tokens_override=base_max_completion_tokens,
                    rescue_mode=rescue_mode,
                    rescue_instruction=rescue_instruction,
                )
            else:
                primary = self.extract(
                    crop_bytes=crop_bytes_for_pass,
                    bank_code=bank_code,
                    pdf_sha=pdf_sha,
                    page_number=page_number,
                    bbox_norm=bbox_norm,
                    vision_cfg=vision_cfg,
                    bottom_extension_used=bottom_extension_used,
                    reference_text=reference_text,
                    max_completion_tokens_override=base_max_completion_tokens,
                    rescue_mode=rescue_mode,
                    rescue_instruction=rescue_instruction,
                )
            if (
                rescue_enabled
                and rescue_max_completion_tokens > base_max_completion_tokens
                and _has_truncation_signal(primary)
            ):
                rescue = self.extract(
                    crop_bytes=crop_bytes_for_pass,
                    bank_code=bank_code,
                    pdf_sha=pdf_sha,
                    page_number=page_number,
                    bbox_norm=bbox_norm,
                    vision_cfg=vision_cfg,
                    bottom_extension_used=bottom_extension_used,
                    reference_text=reference_text,
                    max_completion_tokens_override=rescue_max_completion_tokens,
                    rescue_mode=rescue_mode,
                    rescue_instruction=rescue_instruction,
                )
                if rescue is not None:
                    return replace(rescue, rescue_used=True)
            return primary

        first = _run_pass(
            crop_bytes_for_pass=crop_bytes,
            bottom_extension_used=initial_bottom_extension,
        )

        if first is not None and first.indicators:
            found_markers = _extract_markers_from_indicators(
                [str(item).strip() for item in list(first.indicators)]
            )
            if found_markers:
                expected_set.update(found_markers)

        initial_is_suspect = _is_trivial_result(first, bbox_norm=bbox_norm)
        initial_rejection_reasons = _collect_incompleteness_reasons(
            first,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_set,
        )

        # --- Dual LLM QA Inspector (Priority 1) ---
        qa_missing_str = ""
        passed_qa = False
        if (
            first is not None
            and not initial_is_suspect
            and not initial_rejection_reasons
        ):
            try:
                import dataclasses

                from vigilance.extraction.vision_qa_inspector import (
                    VisionTableInspector,
                )

                first_dict = dataclasses.asdict(first)

                inspector = VisionTableInspector(model="gpt-4o")
                qa_result = inspector.inspect_extraction(crop_bytes, first_dict)

                if not qa_result.is_perfect:
                    initial_rejection_reasons.append("qa_inspector_failed")
                    qa_missing_str = ", ".join(qa_result.missing_elements)
                else:
                    passed_qa = True
            except Exception as e:
                logger.error("Failed to execute VisionTableInspector: %s", e)
        # ------------------------------------------

        if not initial_is_suspect and not initial_rejection_reasons:
            assert first is not None
            return _build_result_debug_metadata(
                replace(first, extraction_status="ok", qa_inspected=passed_qa),
                acceptance_reason="initial_complete",
                rejection_reasons=[],
                selected_candidate_name="initial",
                no_table_evidence_count=0,
                bbox_norm=bbox_norm,
                expected_footnote_ids=expected_set,
            )

        candidates: list[tuple[str, VisionFullResult | None]] = [("initial", first)]
        no_table_evidence = 0
        if (
            first is not None
            and first.no_table_detected
            and _is_trivial_result(first, bbox_norm=bbox_norm)
        ):
            no_table_evidence += 1

        base_rescue_instruction = ""
        if "qa_inspector_failed" in initial_rejection_reasons:
            base_rescue_instruction = (
                f"CRITICAL WARNING: The rigid QA Inspector found you missed required first-column row labels or footnotes in the image: [{qa_missing_str}].\n"
                "You MUST execute the extraction again and GUARANTEE these missing FIRST-COLUMN row labels or footnotes are included. "
                "Do NOT add text from non-leftmost columns. Reread carefully line-by-line."
            )
        elif "low_density_vertical" in initial_rejection_reasons:
            base_rescue_instruction = (
                "WARNING: You failed to extract the full table height in the previous pass. "
                "Reread the entire image, line-by-line, and extract EVERY first-column row label including heavily indented sub-items. "
                "Do NOT convert text from columns 2+ into indicators."
            )

        same_crop_rescue = _run_pass(
            crop_bytes_for_pass=crop_bytes,
            bottom_extension_used=initial_bottom_extension,
            rescue_mode=True,
            rescue_instruction=base_rescue_instruction,
        )
        candidates.append(("same_crop_rescue", same_crop_rescue))
        if (
            same_crop_rescue is not None
            and same_crop_rescue.no_table_detected
            and _is_trivial_result(same_crop_rescue, bbox_norm=bbox_norm)
        ):
            no_table_evidence += 1

        def _append_candidate(
            name: str,
            crop_bytes_for_pass: bytes | None,
            *,
            bottom_extension_used: float,
            candidate_bbox: list[float] | None = None,
            custom_rescue_instr: str | None = None,
        ) -> None:
            nonlocal no_table_evidence
            if not crop_bytes_for_pass:
                return
            result = _run_pass(
                crop_bytes_for_pass=crop_bytes_for_pass,
                bottom_extension_used=bottom_extension_used,
                rescue_mode=True,
                rescue_instruction=custom_rescue_instr
                if custom_rescue_instr is not None
                else base_rescue_instruction,
            )
            candidates.append((name, result))
            if (
                result is not None
                and result.no_table_detected
                and _is_trivial_result(result, bbox_norm=candidate_bbox or bbox_norm)
            ):
                no_table_evidence += 1

        if get_variant_crop_fn is not None and bbox_norm and len(bbox_norm) >= 4:
            left, top, right, bottom = [float(v) for v in bbox_norm[:4]]
            height = max(0.0, bottom - top)
            width = max(0.0, right - left)

            top_trim_bbox = [
                left,
                min(bottom, top + min(height * 0.12, 0.06)),
                right,
                bottom,
            ]
            _append_candidate(
                "top_trim",
                get_variant_crop_fn(
                    bbox_override=top_trim_bbox,
                    bottom_extension=initial_bottom_extension,
                ),
                bottom_extension_used=initial_bottom_extension,
                candidate_bbox=top_trim_bbox,
            )

            recrop_ext = initial_bottom_extension + _RECROP_EXTENSION_INCREMENT
            _append_candidate(
                "bottom_extended",
                get_variant_crop_fn(bottom_extension=recrop_ext),
                bottom_extension_used=recrop_ext,
            )

            tight_bbox = [
                max(0.0, left + min(width * 0.015, 0.01)),
                min(bottom, top + min(height * 0.08, 0.04)),
                min(1.0, right - min(width * 0.015, 0.01)),
                bottom,
            ]
            if tight_bbox[2] > tight_bbox[0] and tight_bbox[3] > tight_bbox[1]:
                _append_candidate(
                    "tight_body",
                    get_variant_crop_fn(
                        bbox_override=tight_bbox,
                        bottom_extension=max(0.0, initial_bottom_extension * 0.5),
                    ),
                    bottom_extension_used=max(0.0, initial_bottom_extension * 0.5),
                    candidate_bbox=tight_bbox,
                )
        elif get_recrop_fn is not None:
            recrop_ext = initial_bottom_extension + _RECROP_EXTENSION_INCREMENT
            _append_candidate(
                "bottom_extended",
                get_recrop_fn(recrop_ext),
                bottom_extension_used=recrop_ext,
            )

        usable_candidates = [
            item
            for item in candidates
            if item[1] is not None and _is_viable_result(item[1])
        ]
        if usable_candidates:
            best_name, best_result = max(
                usable_candidates,
                key=lambda item: _candidate_quality_score(
                    item[1],
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                    baseline_result=first,
                ),
            )
            assert best_result is not None
            final_status = (
                "ok"
                if best_name == "initial" and not initial_rejection_reasons
                else "rescued"
            )
            selected = replace(
                best_result,
                rescue_used=best_name != "initial" or best_result.rescue_used,
                recrop_attempted=len(candidates) > 1,
                recrop_used=best_name != "initial",
                recrop_failed_incomplete=False,
                extraction_status=final_status,
            )
            finalized = _finalize_selected_candidate(
                selected,
                best_name=best_name,
                initial_rejection_reasons=initial_rejection_reasons,
                no_table_evidence_count=no_table_evidence,
                bbox_norm=bbox_norm,
                expected_footnote_ids=expected_set,
            )
            if finalized is not None:
                return finalized

        if no_table_evidence >= 2:
            fallback = (
                same_crop_rescue
                or first
                or VisionFullResult(
                    table_title="",
                    table_summary="",
                    headers=[],
                    indicators=[],
                    footnotes_content=[],
                    no_table_detected=True,
                )
            )
            return _build_result_debug_metadata(
                replace(
                    fallback,
                    no_table_detected=True,
                    recrop_attempted=len(candidates) > 1,
                    recrop_used=False,
                    recrop_failed_incomplete=False,
                    extraction_status="confirmed_no_table",
                ),
                acceptance_reason="confirmed_no_table_after_repeated_no_table_evidence",
                rejection_reasons=initial_rejection_reasons or ["no_table_evidence"],
                selected_candidate_name="same_crop_rescue"
                if same_crop_rescue
                else "initial",
                no_table_evidence_count=no_table_evidence,
                bbox_norm=bbox_norm,
                expected_footnote_ids=expected_set,
            )

        fallback = same_crop_rescue or first
        if fallback is None:
            fallback = VisionFullResult(
                table_title="",
                table_summary="",
                headers=[],
                indicators=[],
                footnotes_content=[],
            )
        fallback_rejection_reasons = list(
            dict.fromkeys(
                initial_rejection_reasons
                + _collect_incompleteness_reasons(
                    fallback,
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                )
            )
        )
        return _build_result_debug_metadata(
            replace(
                fallback,
                recrop_attempted=len(candidates) > 1,
                recrop_used=False,
                recrop_failed_incomplete=True,
                extraction_status="suspect_unresolved",
            ),
            acceptance_reason="suspect_unresolved_after_rescue_exhaustion",
            rejection_reasons=fallback_rejection_reasons or ["rescue_exhausted"],
            selected_candidate_name="same_crop_rescue"
            if same_crop_rescue
            else "initial",
            no_table_evidence_count=no_table_evidence,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_set,
        )
