"""Prompts Vision et leur assemblage.

Extrait de ``vision_full_extractor.py`` sans modification : les quatre prompts,
les identifiants de variante et les fonctions qui assemblent le prompt final,
le contenu multimodal et le prompt de reparation.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_REFERENCE_TEXT_MAX_CHARS = 6000

_PROMPT_BASE = """

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

CRITICAL: FORCED EXHAUSTIVENESS — ZERO TOLERANCE FOR OMISSIONS
You MUST extract EVERY SINGLE ROW visible in the first column of the table. No row is ever skipped, ignored, or forgotten.
- Every section heading is an indicator. Extract it.
- Every sub-row, child row, and indented item is an indicator. Extract it.
- Every subtotal and total row is an indicator. Extract it.
- DO NOT group, summarize, or collapse rows under any circumstances. Each row = one indicator.
- DO NOT skip a row because it looks like a heading, a total, a duplicate, or a continuation.
- DO NOT skip a row because it has no visible numbers on the right side.
- DO NOT skip a row because you are uncertain. Uncertainty is never a reason to omit.
- Read the first column line-by-line from top to bottom. Miss nothing.

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

CRITICAL: DATE/PERIOD IDENTIFIERS — CLASSIFY BY GEOMETRIC ROLE
INCLUDE a date or period as an indicator when it is the LEFTMOST cell of a
body data row and the same horizontal row contains data values in columns 2+.
Examples that MUST be included in that body-row role:
"Au 30 avril 2025", "Au 31 janvier 2025", "Au 31 octobre 2024", "T1 2025".

EXCLUDE a date or period only when it functions as a row-group delimiter,
spanning label, column sub-header, or table-wide temporal caption.
Also exclude unit descriptors spanning the table width:
"En millions de dollars", "En milliers de dollars", "(en millions de dollars)".

The visible geometry decides: body data row = indicator; delimiter/header = metadata.

HIERARCHY PRESERVATION
Use leading spaces to indicate the nesting depth as visible in the table:
- Top-level labels: no leading spaces
- First-level sub-items (visually indented once): prefix with "  " (2 spaces)
- Second-level sub-items (indented twice): prefix with "    " (4 spaces)
Reflect the visual hierarchy faithfully. This is essential for downstream accuracy.

═══════════════════════════════════════════
FIELD 2: footnotes_content (SECOND PRIORITY)
═══════════════════════════════════════════

Extract ALL footnotes located BELOW the table body, inside the cropped image.
A footnote is any line below the table that BEGINS with one of the known marker styles listed below,
regardless of whether that marker also appears in the first column or anywhere else in the table.

KNOWN MARKER STYLES — recognize and normalize all of these:
- parenthetical full:  (1) (2) (3) ... (10)  → normalize id to "1", "2", "3" ... "10"
- parenthetical half:  1) 2) 3) ... 10)       → normalize id to "1", "2", "3" ... "10"
- bare digit at line start: 1  2  3 ... 10    → normalize id to "1", "2" ... "10"
- superscript digit:   ¹ ² ³ ...              → normalize id to "1", "2", "3"
- asterisk:            *                       → keep id as "*"
- s. o.               (sans objet)             → keep id as "s. o."

For each footnote return:
- id: normalized marker as described above (strip parentheses, convert superscripts to digits)
- text: exact full footnote text, preserving French accents and punctuation

NOISE GUARD — do NOT extract as a footnote:
- a lone number with no following text (e.g. a bare "3" or "10" alone on a line)
- narrative paragraph text that does NOT start with a recognized marker
- table body data rows that happen to appear at the bottom of the visible crop
- page running headers or section titles
- blank lines or lines shorter than 4 characters

Rules:
- preserve exact wording of each footnote
- preserve visual top-to-bottom order (do NOT sort by id)
- do not merge two separate footnotes into one
- do not invent missing text
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

1. indicators and footnotes_content are your ONLY mission. You MUST complete them fully. Zero omissions allowed.
2. Extract ONLY what is visible in the cropped image. Never invent text.
3. Keep row order and footnote order exactly as visually seen.
4. NEVER skip a first-column label for any reason. Every text in the first column is an indicator — include it unconditionally.
5. NEVER skip a footnote below the table. Every line starting with a known marker is a footnote — include it unconditionally.
6. If uncertain whether there is a real table → set no_table_detected to true.

MANDATORY SELF-CHECK BEFORE RETURNING
Before writing your final JSON, scan the image one more time:
- Count the visible rows in the table. Does your indicators list match that count? If not, find the missing rows and add them now.
- Count the footnote markers visible below the table. Does your footnotes_content match that count? If not, find the missing ones and add them now.
Only return your JSON after this check passes.

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
2. indicators: no date/period delimiters or column sub-headers; KEEP dates/periods that label body data rows
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
- Apply the date/period geometric-role rule: KEEP a date or period when it is
  the leftmost label of a body data row with values across that same row.
- Apply hierarchy preservation via leading spaces.
- In multi-column textual tables, re-check that ONLY the leftmost column populates "indicators".
- Use no_table_detected = true only if absolutely no tabular structure is visible.
"""

_PROMPT_BASE_PRECISION = """

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
FIELD 1: indicators (HIGHEST PRIORITY — PRECISION MODE)
═══════════════════════════════════════════

Extract ONLY first-column row labels that you are 100% certain belong to the table's left column.
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

WHEN IN DOUBT, OMIT — A missed indicator is less harmful than a hallucinated one.
If you are not certain that a text element belongs to the first column, do not include it.

CRITICAL: MULTI-COLUMN TEXT TABLES
If the table has multiple textual columns (for example headers like "Canada", "États-Unis", "Europe"):
- ONLY the text in the LEFTMOST / FIRST COLUMN is eligible for "indicators"
- text visible under "États-Unis", "Europe", or any other non-leftmost header must NEVER appear in "indicators"
- if a logical row has no visible first-column / leftmost cell, it must NOT produce an indicator
- exhaustiveness means "all first-column row labels", NOT "all text visible anywhere in the image"

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
- If no group heading exists above the repeated label, keep the original label unchanged.
- Apply this ONLY to labels that would otherwise appear as exact duplicates in the output list.
- Every indicator in the final output list MUST be unique.
  If disambiguation via group heading is not possible, append a position marker: " (bloc 2)" to the second occurrence.

CRITICAL: DATE/PERIOD IDENTIFIERS — CLASSIFY BY GEOMETRIC ROLE
INCLUDE a date or period as an indicator when it is the LEFTMOST cell of a
body data row and the same horizontal row contains data values in columns 2+.
Examples that MUST be included in that body-row role:
"Au 30 avril 2025", "Au 31 janvier 2025", "Au 31 octobre 2024", "T1 2025".

EXCLUDE a date or period only when it functions as a row-group delimiter,
spanning label, column sub-header, or table-wide temporal caption.
Also exclude unit descriptors spanning the table width:
"En millions de dollars", "En milliers de dollars", "(en millions de dollars)".

The visible geometry decides: body data row = indicator; delimiter/header = metadata.

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

# Prompt variant identifiers for dual-prompt consensus
_PROMPT_VARIANT_EXHAUSTIVE = "exhaustive"
_PROMPT_VARIANT_PRECISION = "precision"
_CONSENSUS_PROMPT_VARIANTS: tuple[str, str] = (_PROMPT_VARIANT_EXHAUSTIVE, _PROMPT_VARIANT_PRECISION)


def _build_prompt(
    bank_code: str,
    vision_cfg: dict[str, Any],
    reference_text: str | None = None,
    *,
    rescue_mode: bool = False,
    rescue_instruction: str = "",
) -> str:
    """Construit le prompt avec les indices de marqueurs de notes specifiques a la banque et le texte de reference OCR (toujours injecte lorsque fourni)."""
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
    reference_text_max_chars = int(vision_cfg.get("vision_reference_text_max_chars", _DEFAULT_REFERENCE_TEXT_MAX_CHARS))
    if reference_text and len(reference_text.strip()) > 20 and reference_text_max_chars > 0:
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
        logger.debug("Vision: no reference text provided or too short; extraction without dictionary.")
        reference_section = (
            "\n\nCONSIGNE (pas de dictionnaire OCR disponible) : "
            "Transcris EXACTEMENT ce que tu vois dans l'image. "
            "Ne corrige PAS l'orthographe, la casse, la ponctuation ni les espaces des libelles. "
            "Si un libellé long occupe 2 ou plusieurs lignes visuelles, retourne un seul indicateur logique et ne le scinde jamais en deux indicateurs. "
            "Conserve les accents, les tirets et les caracteres speciaux tels quels.\n"
        )

    prompt = _PROMPT_BASE + (f"\n{suffix}\n" if suffix else "") + reference_section + _PROMPT_JSON_STRICT
    if rescue_mode:
        prompt = prompt + _PROMPT_RESCUE_SUFFIX
        if rescue_instruction:
            prompt = prompt + "\n\n### RESCUE INSTRUCTIONS ###\n" + rescue_instruction
    return prompt


def _build_precision_prompt(
    bank_code: str,
    vision_cfg: dict[str, Any],
    reference_text: str | None = None,
) -> str:
    """Construit le prompt variante 'precision' (omission en cas de doute).

    Identique a ``_build_prompt`` mais utilise ``_PROMPT_BASE_PRECISION``
    comme base. N'ajoute jamais le suffixe de sauvetage — la variante precision
    est uniquement utilisee pour le premier tir du consensus.
    """
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

    reference_section = ""
    reference_text_max_chars = int(vision_cfg.get("vision_reference_text_max_chars", _DEFAULT_REFERENCE_TEXT_MAX_CHARS))
    if reference_text and len(reference_text.strip()) > 20 and reference_text_max_chars > 0:
        truncated = reference_text.strip()[:reference_text_max_chars]
        reference_section = (
            "\n\n=== DICTIONNAIRE DE RÉFÉRENCE (Texte OCR du tableau) ===\n"
            f"{truncated}\n"
            "=== FIN DICTIONNAIRE ===\n\n"
            "CONSIGNE : Utilise l'image pour l'ordre visuel et la structure du tableau. "
            "Utilise le Dictionnaire de Référence ci-dessus UNIQUEMENT pour VÉRIFIER L'ORTHOGRAPHE EXACTE "
            "des libellés que tu as DÉJÀ identifiés avec certitude dans l'image. "
            "Ne recopie pas le dictionnaire. Si tu n'es pas certain qu'un libellé vient de la colonne gauche, OMETS-LE.\n"
        )
    else:
        reference_section = (
            "\n\nCONSIGNE (pas de dictionnaire OCR disponible) : "
            "Transcris EXACTEMENT ce que tu vois dans la première colonne. "
            "Si tu n'es pas certain qu'un texte appartient à la première colonne, OMETS-LE.\n"
        )

    return _PROMPT_BASE_PRECISION + (f"\n{suffix}\n" if suffix else "") + reference_section + _PROMPT_JSON_STRICT


def _build_content(prompt: str, image_b64: str) -> list[Any]:
    """Construit le contenu multimodal (texte + image) pour l'appel API."""
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
    """Construit un prompt de reparation a partir de la reponse precedente invalide."""
    return (
        "Le contenu precedent n'etait pas exploitable. "
        "Retourne UNIQUEMENT un objet JSON valide conforme au schema demande, "
        "sans markdown, sans commentaire, sans texte avant ou apres.\n\n"
        f"{base_prompt}\n\n"
        "Reponse precedente a corriger:\n"
        f"{raw_content[:2000]}"
    )
