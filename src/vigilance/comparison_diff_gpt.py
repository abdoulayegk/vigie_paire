"""Diff semantique via GPT pour des tableaux financiers canoniques deja apparies."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Callable

from vigilance.models.comparison_models import (
    FootnoteDiffResponse,
    IndicatorDiffResponse,
    InspectorResponse,
)

logger = logging.getLogger(__name__)


INDICATOR_DIFF_SYSTEM_PROMPT = """
You are a precision-first banking table indicator diff engine.

You compare two already-matched canonical banking tables from adjacent quarterly reports.

Your task is to report only meaningful semantic indicator changes:
- indicators_added
- indicators_removed
- indicators_renamed

Rules:
- The table pair is already matched. Do not question the pairing.
- Compare only indicator meaning and role in the table.
- Ignore numeric values, dates, periods, formatting, OCR noise, row order changes, and line wrapping.
- IGNORE footnote marker changes: if two indicators differ ONLY by a footnote reference like (1), (2), (3), (4) being added, removed, or changed, they are THE SAME indicator. Do NOT classify this as renamed. Example: 'catégorie 1 (4)' and 'catégorie 1' are identical — ignore this.
- IGNORE extraction disambiguation labels: tables with repeated sub-sections may be extracted with
  disambiguation tags like '(bloc 2)', '(bloc 3)' appended, or date prefixes like '31 octobre 2025 – '
  prepended. These are NOT part of the real indicator name. When matching, strip these prefixes/suffixes
  and compare the core business name. Examples:
    • 'Total (bloc 2)' and '31 octobre 2025 – Total' → SAME indicator (core = 'Total').
    • 'Total (libellé en dollars canadiens) (bloc 2)' and '31 octobre 2025 – Total (libellé en dollars canadiens)' → SAME.
  Treat these as UNCHANGED — do not report them as added, removed, or renamed.
- Indicator present only in current = indicators_added.
- Indicator present only in previous = indicators_removed.
- Classify indicators_renamed only when the previous and current indicators clearly represent the exact same business concept with the same scope and the same role in the table.
- If the change could instead be explained by addition, removal, scope change, row split, or row merge, do NOT classify it as renamed.
- When unsure between rename and add/remove, prefer add/remove.
- If one previous indicator appears split into multiple current indicators, treat the new rows as additions rather than rename.
- If multiple previous indicators appear merged into one current indicator, do not classify as rename unless the scope is clearly identical.
- Be conservative and report only clear semantic differences.
- For each change (added, removed, renamed), you MUST act as a Senior Risk Analyst and provide an 'analyst_assessment'.
- The 'analyst_assessment' MUST include:
  1. A 'relevance_level' (integer: 1 for Critical/Regulatory, 2 for High/Structural, 3 for Low/Cosmetic).
  2. A 'justification' (A clear, articulate, and complete descriptive sentence explaining the business impact and exactly WHY this change matters to guide the analyst).

Output must be valid JSON following the response_schema.
"""


FOOTNOTE_DIFF_SYSTEM_PROMPT = """
You are a precision-first banking table footnote diff engine.

You compare footnotes from two already-matched canonical banking tables from adjacent quarterly reports.

Your task is to report only meaningful footnote changes:
- footnotes_added
- footnotes_removed
- footnotes_renamed

Rules:
- The table pair is already matched. Do not question the pairing.
- Compare only semantic footnote meaning, not numbering or formatting.
- Ignore pure footnote renumbering when the meaning is unchanged.
- Ignore changes caused only by dates, quarter references, formatting, punctuation, or minor drafting changes that do not alter meaning.
- IGNORE page number changes: if two footnotes differ ONLY by page references (e.g. 'pages 6 à 10' vs 'pages 6 à 12'), they are THE SAME footnote. Do NOT classify this as renamed.
- IGNORE quarter/date reference updates: if a footnote text changes only the quarter or date reference (e.g. '31 janvier 2025' vs '30 avril 2025'), this is NOT a meaningful change.
- Footnote present only in current = footnotes_added.
- Footnote present only in previous = footnotes_removed.
- Footnote with the same semantic meaning but materially revised wording = footnotes_renamed.
- Compare footnotes within the logical scope of the same table and in the context of the already-matched pair.
- Be conservative and report only clear semantic differences.
- For each change (added, removed, renamed), you MUST act as a Senior Risk Analyst and provide an 'analyst_assessment'.
- The 'analyst_assessment' MUST include:
  1. A 'relevance_level' (integer: 1 for Critical/Regulatory, 2 for High/Structural, 3 for Low/Cosmetic).
  2. A 'justification' (A clear, articulate, and complete descriptive sentence explaining the business impact and exactly WHY this change matters to guide the analyst).

Output must be valid JSON following the response_schema.
"""


INSPECTOR_SYSTEM_PROMPT = """
You are a Senior Risk Analyst quality-control inspector for a banking table diff pipeline.

You receive a diff produced by a first-pass GPT model that compared two already-matched
canonical banking tables from adjacent quarterly reports. Your ONLY job is to classify
each indicator flagged as "added" or "removed" as either a REAL semantic change or an
extraction ARTIFACT that should be suppressed.

CRITICAL PRINCIPLE — artifact vs real:
An indicator should ONLY be classified as "artifact" when you can clearly identify its
counterpart on the opposite side (added↔removed) and the ONLY difference between them
is extraction noise (footnote markers, unicode, OCR, whitespace). If a removed indicator
has NO plausible counterpart among added indicators (and vice versa), it MUST be "real".

Common artifact patterns you MUST catch:
1. Footnote marker variance: 'Goodwill³' vs 'Goodwill', 'catégorie 1 (4)' vs 'catégorie 1'.
2. Footnote reference shift on the SAME business concept: 'Série 3⁶' vs 'Série 3⁴' — same
   indicator name, only the superscript reference number changed. But 'Série 5' vs 'Série 7'
   are DIFFERENT business entities — never pair them.
3. Disambiguation label / date-prefix / bloc-suffix variance (CRITICAL):
   Tables with repeated sub-sections are often extracted with disambiguation tags.
   One quarter may use '(bloc 2)', '(bloc 3)' suffixes; another may use date prefixes
   like '31 octobre 2025 – ', '31 janvier 2025 – ', etc. To detect these artifacts:
     a) Strip any leading date pattern (e.g. '31 octobre 2025 – ', '30 avril 2025 – ').
     b) Strip any trailing '(bloc N)' tag.
     c) Compare the remaining core indicator name.
   If the core names match, mark BOTH as artifact and pair them.
   Examples:
     • removed='Total (bloc 2)' + added='31 octobre 2025 – Total' → artifact pair (core='Total').
     • removed='Total (libellé en dollars canadiens) (bloc 2)' +
       added='31 octobre 2025 – Total (libellé en dollars canadiens)' → artifact pair.
     • removed='Total (non libellé en dollars canadiens) (bloc 2)' +
       added='31 octobre 2025 – Total (non libellé en dollars canadiens)' → artifact pair.
4. Unicode apostrophe / quote variance: '\u2019' (U+2019) vs "'" (U+0027), « » vs ".
5. Minor OCR / whitespace / punctuation noise: trailing dots, extra spaces, accented char variance.

Rules:
- For each indicator in added_indicators and removed_indicators, output a verdict.
- verdict MUST be one of: "real" or "artifact".
- An indicator can ONLY be "artifact" if it has a clear extraction-noise counterpart on the
  other side. A removed indicator with no matching added indicator is ALWAYS "real".
  An added indicator with no matching removed indicator is ALWAYS "real".
- If an added indicator and a removed indicator are the SAME business concept (just extraction noise),
  mark BOTH as "artifact" and pair them in artifact_pairs.
- DATE-PREFIXED INDICATORS: When an indicator starts with a date in any format
  (e.g. '31 octobre 2025 – ...', '2025-01-31 – ...', 'October 31 2025 – ...'),
  always look for a counterpart without the date prefix (or with a '(bloc N)' suffix instead).
  The date prefix is extraction noise, not part of the real indicator name.
- CONTEXTUAL CROSS-CHECK: When unsure whether an added/removed pair are the same indicator,
  look at their NEIGHBORS in the previous_table and current_table indicator lists. If the
  indicators directly above and below are the same on both sides, it strongly confirms
  the flagged indicators occupy the same structural position and are the same business concept.
- When in doubt, mark as "real". Missing a genuine change is worse than surfacing a false positive.
- Do NOT re-assess renamed indicators — they are already handled.
- Cross-check against the provided previous_table and current_table indicator lists. If an
  indicator genuinely does not appear in the other quarter's list, it is "real".

Output must be valid JSON following the response_schema.
"""


def _normalize_footnotes(raw: Any) -> list[dict[str, str]]:
    """Normalise une liste brute de notes de bas de page en dicts ``{id, text}``."""
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not fid and not text:
            continue
        normalized.append({"id": fid, "text": text})
    return normalized


# ---------------------------------------------------------------------------
# Deterministic diff helpers (safety net)
# ---------------------------------------------------------------------------

_FOOTNOTE_MARKER_RE = re.compile(r"\s*[\(\[]\d{1,2}[\)\]]\s*")
_SUPERSCRIPT_DIGITS = str.maketrans("", "", "\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070")


def _normalize_indicator_text(name: str) -> str:
    """Normalise un nom d'indicateur pour la comparaison deterministe par ensemble."""
    text = str(name or "").strip()
    text = _FOOTNOTE_MARKER_RE.sub("", text)
    text = text.translate(_SUPERSCRIPT_DIGITS)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _token_overlap_ratio(a: str, b: str) -> float:
    """Retourne le ratio de chevauchement de tokens (type Jaccard) entre deux chaines normalisees."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _deterministic_indicator_diff(
    prev_indicators: list[str],
    curr_indicators: list[str],
    *,
    fuzzy_threshold: float = 0.80,
) -> dict[str, Any]:
    """Calcule le diff d'indicateurs par ensemble avant l'appel GPT."""
    prev_norm = {_normalize_indicator_text(ind): ind for ind in prev_indicators}
    curr_norm = {_normalize_indicator_text(ind): ind for ind in curr_indicators}

    prev_keys = set(prev_norm.keys())
    curr_keys = set(curr_norm.keys())

    only_prev = prev_keys - curr_keys
    only_curr = curr_keys - prev_keys

    # Attempt fuzzy matching between the unmatched sets
    det_renamed: list[dict[str, str]] = []
    matched_prev: set[str] = set()
    matched_curr: set[str] = set()
    for pkey in sorted(only_prev):
        best_score = 0.0
        best_ckey = ""
        for ckey in sorted(only_curr):
            if ckey in matched_curr:
                continue
            score = _token_overlap_ratio(pkey, ckey)
            if score > best_score:
                best_score = score
                best_ckey = ckey
        if best_score >= fuzzy_threshold and best_ckey:
            det_renamed.append({"previous": prev_norm[pkey], "current": curr_norm[best_ckey]})
            matched_prev.add(pkey)
            matched_curr.add(best_ckey)

    det_removed = [prev_norm[k] for k in sorted(only_prev - matched_prev)]
    det_added = [curr_norm[k] for k in sorted(only_curr - matched_curr)]

    return {
        "det_added": det_added,
        "det_removed": det_removed,
        "det_renamed": det_renamed,
    }


_DATE_QUARTER_RE = re.compile(
    r"\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4}"
    r"|T[1-4]\s*[-–]?\s*\d{4}"
    r"|\d{4}\s*[-–]?\s*T[1-4]"
    r"|(?:premier|deuxième|troisième|quatrième)\s+trimestre\s+\d{4}",
    re.IGNORECASE,
)
_PAGE_REF_RE_DET = re.compile(r"pages?\s+\d+\s*[àa]\s*\d+", re.IGNORECASE)


def _normalize_footnote_text(text: str) -> str:
    """Normalise le texte d'une note de bas de page pour comparaison deterministe (retire dates/pages/espaces)."""
    text = str(text or "").strip()
    text = _DATE_QUARTER_RE.sub("__DATE__", text)
    text = _PAGE_REF_RE_DET.sub("__PAGE__", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _deterministic_footnote_diff(
    prev_footnotes: list[dict[str, str]],
    curr_footnotes: list[dict[str, str]],
) -> dict[str, Any]:
    """Calcule le diff de notes de bas de page par ensemble avant l'appel GPT."""
    prev_by_id: dict[str, dict[str, str]] = {}
    for fn in prev_footnotes:
        fid = str(fn.get("id", "") or "").strip()
        if fid:
            prev_by_id[fid] = fn

    curr_by_id: dict[str, dict[str, str]] = {}
    for fn in curr_footnotes:
        fid = str(fn.get("id", "") or "").strip()
        if fid:
            curr_by_id[fid] = fn

    prev_ids = set(prev_by_id.keys())
    curr_ids = set(curr_by_id.keys())

    det_added = [curr_by_id[fid] for fid in sorted(curr_ids - prev_ids)]
    det_removed = [prev_by_id[fid] for fid in sorted(prev_ids - curr_ids)]

    # For IDs present in both, check if text changed materially
    det_modified: list[dict[str, Any]] = []
    for fid in sorted(prev_ids & curr_ids):
        prev_text = _normalize_footnote_text(prev_by_id[fid].get("text", ""))
        curr_text = _normalize_footnote_text(curr_by_id[fid].get("text", ""))
        if prev_text != curr_text:
            det_modified.append(
                {
                    "previous_id": fid,
                    "current_id": fid,
                    "previous_text": prev_by_id[fid].get("text", ""),
                    "current_text": curr_by_id[fid].get("text", ""),
                }
            )

    # Cross-match removed/added by text similarity (re-numbered footnotes)
    unmatched_removed = list(det_removed)
    unmatched_added = list(det_added)
    cross_renamed: list[dict[str, Any]] = []
    still_removed: list[dict[str, str]] = []
    for rfn in unmatched_removed:
        r_text = _normalize_footnote_text(rfn.get("text", ""))
        best_idx = -1
        best_match = False
        for idx, afn in enumerate(unmatched_added):
            a_text = _normalize_footnote_text(afn.get("text", ""))
            if r_text == a_text:
                best_idx = idx
                best_match = True
                break
        if best_match:
            afn = unmatched_added.pop(best_idx)
            # Same text, different ID → pure renumbering, not a real change
        else:
            still_removed.append(rfn)
    det_removed = still_removed
    det_added = unmatched_added

    return {
        "det_added": det_added,
        "det_removed": det_removed,
        "det_modified": det_modified,
    }


def _table_context(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrait le contexte normalise d'un tableau pour les prompts GPT."""
    indicators = list(entry.get("indicators", []) or [])
    return {
        "table_id": str(entry.get("table_id", "") or ""),
        "section": str(entry.get("section", "") or "unknown_section"),
        "title": str(entry.get("title", "") or ""),
        "table_summary": str(entry.get("table_summary", "") or ""),
        "page": entry.get("page"),
        "row_count": int(entry.get("row_count", len(indicators)) or 0),
        "headers": [str(value).strip() for value in list(entry.get("headers", []) or []) if str(value).strip()],
        "indicators": [unicodedata.normalize("NFC", str(value).strip()) for value in indicators if str(value).strip()],
        "footnotes": _normalize_footnotes(entry.get("footnotes", [])),
    }


def _call_validated_diff_json(
    *,
    system_prompt: str,
    prompt: dict[str, Any],
    required_list_fields: tuple[str, ...],
    model: str,
    call_kind: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None,
    max_validation_attempts: int,
    response_model: type | None = None,
) -> dict[str, Any]:
    """Appelle GPT et valide que la reponse contient les champs liste requis, avec re-essais.

    Args:
        system_prompt: Prompt systeme a envoyer a GPT.
        prompt: Corps du prompt utilisateur (serialise en JSON).
        required_list_fields: Noms des champs qui doivent etre des listes dans la reponse.
        model: Identifiant du modele OpenAI.
        call_kind: Etiquette pour le suivi d'utilisation.
        call_openai_json: Fonction injectee pour l'appel OpenAI.
        usage_recorder: Accumulateur optionnel de metriques d'utilisation.
        max_validation_attempts: Nombre maximal de tentatives de validation.
        response_model: Modele Pydantic optionnel pour la validation structuree.

    Returns:
        Dictionnaire JSON valide retourne par GPT.

    Raises:
        RuntimeError: Si la reponse reste structurellement invalide apres les re-essais.
    """
    validation_feedback = ""
    data: dict[str, Any] | None = None
    for attempt in range(max_validation_attempts):
        request_prompt = dict(prompt)
        if validation_feedback:
            request_prompt["validation_feedback"] = validation_feedback
            request_prompt["rules"] = list(prompt["rules"]) + [
                "Your previous response was structurally invalid. Fix the validation issue and return corrected JSON."
            ]
        data = call_openai_json(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(request_prompt, ensure_ascii=False),
                },
            ],
            usage_recorder=usage_recorder,
            call_kind=call_kind,
            response_model=response_model,
        )
        if all(isinstance(data.get(field, []), list) for field in required_list_fields):
            return data
        validation_feedback = "Diff response must contain list-valued fields for: " + ", ".join(required_list_fields)
        if attempt + 1 >= max_validation_attempts:
            raise RuntimeError(f"GPT {call_kind} output remained structurally invalid after retries.")
    raise RuntimeError("Unreachable diff validation loop")


def diff_indicators_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    """Compare les indicateurs de deux tableaux apparies via GPT.

    Effectue un pre-diff deterministe puis envoie le contexte a GPT pour
    une analyse semantique des ajouts, suppressions et renommages d'indicateurs.

    Args:
        previous_table: Tableau du trimestre precedent.
        current_table: Tableau du trimestre courant.
        model: Identifiant du modele OpenAI.
        call_openai_json: Fonction injectee pour l'appel OpenAI.
        usage_recorder: Accumulateur optionnel de metriques d'utilisation.
        max_validation_attempts: Nombre maximal de tentatives de validation.

    Returns:
        Dictionnaire contenant les listes ``indicators_added``,
        ``indicators_removed``, ``indicators_renamed`` et ``reason``.
    """
    prev_ctx = _table_context(previous_table)
    curr_ctx = _table_context(current_table)

    # --- Deterministic pre-diff (safety net) ---
    det_diff = _deterministic_indicator_diff(
        prev_ctx["indicators"],
        curr_ctx["indicators"],
    )
    det_hints: dict[str, Any] = {}
    if det_diff["det_removed"] or det_diff["det_added"] or det_diff["det_renamed"]:
        det_hints = {
            "deterministic_analysis": {
                "mechanically_absent_from_current": det_diff["det_removed"],
                "mechanically_absent_from_previous": det_diff["det_added"],
                "potential_renames_by_similarity": [
                    {"previous": r["previous"], "current": r["current"]} for r in det_diff["det_renamed"]
                ],
            }
        }

    rules = [
        "Return JSON only and strictly follow the response_schema.",
        "The two tables are already matched. Do not question the pairing.",
        "Compare only the canonical indicators.",
        "Ignore numeric values, dates, periods, formatting differences, OCR noise, row order changes, and line wrapping.",
        "Indicator present only in current = indicators_added.",
        "Indicator present only in previous = indicators_removed.",
        "Classify indicators_renamed only when the concept, scope, and role are clearly identical.",
        "When unsure between rename and add/remove, prefer add/remove.",
        "Do not treat row splits or row merges as renamed indicators unless the scope is clearly identical.",
    ]
    if det_hints:
        rules.append(
            "A deterministic set analysis is provided in 'deterministic_analysis'. "
            "You MUST account for every indicator listed there: classify each as truly "
            "added/removed/renamed, or explain why it is OCR noise / footnote marker difference."
        )

    prompt: dict[str, Any] = {
        "task": ("Compare two already-matched banking tables and report only meaningful semantic indicator changes."),
        "rules": rules,
        "response_schema": {
            "indicators_added": [
                {
                    "value": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "indicators_removed": [
                {
                    "value": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "indicators_renamed": [
                {
                    "previous": "string",
                    "current": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "reason": "string",
        },
        "previous_table": {key: value for key, value in prev_ctx.items() if key != "footnotes"},
        "current_table": {key: value for key, value in curr_ctx.items() if key != "footnotes"},
    }
    if det_hints:
        prompt.update(det_hints)

    data = _call_validated_diff_json(
        system_prompt=INDICATOR_DIFF_SYSTEM_PROMPT,
        prompt=prompt,
        required_list_fields=(
            "indicators_added",
            "indicators_removed",
            "indicators_renamed",
        ),
        model=model,
        call_kind="diff_indicators",
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
        response_model=IndicatorDiffResponse,
    )
    return {
        "indicators_added": data["indicators_added"],
        "indicators_removed": data["indicators_removed"],
        "indicators_renamed": data["indicators_renamed"],
        "reason": data["reason"],
    }


def diff_footnotes_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    indicator_diff: dict[str, Any],
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    """Compare les notes de bas de page de deux tableaux apparies via GPT.

    Gere les cas triviaux (aucune note, notes uniquement dans un trimestre)
    de facon deterministe et delegue les cas complexes a GPT.

    Args:
        previous_table: Tableau du trimestre precedent.
        current_table: Tableau du trimestre courant.
        indicator_diff: Resultat du diff d'indicateurs (pour contexte).
        model: Identifiant du modele OpenAI.
        call_openai_json: Fonction injectee pour l'appel OpenAI.
        usage_recorder: Accumulateur optionnel de metriques d'utilisation.
        max_validation_attempts: Nombre maximal de tentatives de validation.

    Returns:
        Dictionnaire contenant les listes ``footnotes_added``,
        ``footnotes_removed``, ``footnotes_renamed`` et ``reason``.
    """
    previous_footnotes = _normalize_footnotes(previous_table.get("footnotes", []))
    current_footnotes = _normalize_footnotes(current_table.get("footnotes", []))

    if not previous_footnotes and not current_footnotes:
        return {
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "",
        }

    if not previous_footnotes and current_footnotes:
        return {
            "footnotes_added": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "reason": "Footnote present only in current table.",
                    "analyst_assessment": {
                        "relevance_level": 3,
                        "justification": "L'ajout d'une nouvelle note de bas de page sans contexte détaillé nécessite une vérification manuelle pour confirmer son impact.",
                    },
                }
                for item in current_footnotes
            ],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "reason": "Current table contains footnotes while previous table had none.",
        }

    if previous_footnotes and not current_footnotes:
        return {
            "footnotes_added": [],
            "footnotes_removed": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "reason": "Footnote present only in previous table.",
                    "analyst_assessment": {
                        "relevance_level": 3,
                        "justification": "La suppression d'une ancienne note de bas de page sans contexte détaillé nécessite une vérification manuelle pour confirmer son impact.",
                    },
                }
                for item in previous_footnotes
            ],
            "footnotes_renamed": [],
            "reason": "Previous table contains footnotes while current table has none.",
        }

    # --- Deterministic footnote pre-diff (safety net) ---
    det_fn_diff = _deterministic_footnote_diff(previous_footnotes, current_footnotes)
    det_fn_hints: dict[str, Any] = {}
    if det_fn_diff["det_added"] or det_fn_diff["det_removed"] or det_fn_diff["det_modified"]:
        det_fn_hints = {
            "deterministic_footnote_analysis": {
                "footnotes_only_in_current": [
                    {"id": fn.get("id", ""), "text": fn.get("text", "")} for fn in det_fn_diff["det_added"]
                ],
                "footnotes_only_in_previous": [
                    {"id": fn.get("id", ""), "text": fn.get("text", "")} for fn in det_fn_diff["det_removed"]
                ],
                "footnotes_with_text_changes": [
                    {
                        "id": fn["previous_id"],
                        "previous_text": fn["previous_text"],
                        "current_text": fn["current_text"],
                    }
                    for fn in det_fn_diff["det_modified"]
                ],
            }
        }

    fn_rules = [
        "Return JSON only and strictly follow the response_schema.",
        "The two tables are already matched. Do not question the pairing.",
        "Ignore pure footnote renumbering when meaning is unchanged.",
        "Ignore changes caused only by dates, quarter references, formatting, punctuation, or minor drafting changes that do not alter meaning.",
        "Footnote present only in current = footnotes_added.",
        "Footnote present only in previous = footnotes_removed.",
        "Same semantic note with materially revised wording = footnotes_renamed.",
        "Be conservative and report only clear semantic differences.",
    ]
    if det_fn_hints:
        fn_rules.append(
            "A deterministic footnote analysis is provided in 'deterministic_footnote_analysis'. "
            "You MUST account for every footnote listed there: classify each change as truly "
            "added/removed/renamed, or explain why it is pure renumbering / date-only change."
        )

    prompt: dict[str, Any] = {
        "task": (
            "Compare footnotes for two already-matched banking tables and report only meaningful semantic footnote changes."
        ),
        "rules": fn_rules,
        "response_schema": {
            "footnotes_added": [
                {
                    "id": "string",
                    "text": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "footnotes_removed": [
                {
                    "id": "string",
                    "text": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "footnotes_renamed": [
                {
                    "previous_id": "string",
                    "current_id": "string",
                    "previous_text": "string",
                    "current_text": "string",
                    "reason": "string",
                    "analyst_assessment": {
                        "relevance_level": "integer",
                        "justification": "string",
                    },
                }
            ],
            "reason": "string",
        },
        "examples": [
            {
                "description": "Pure footnote renumbering should not produce a published change.",
                "previous_footnotes": [{"id": "7", "text": "Comprennent les engagements de la Banque."}],
                "current_footnotes": [{"id": "8", "text": "Comprennent les engagements de la Banque."}],
                "expected_output": {
                    "footnotes_added": [],
                    "footnotes_removed": [],
                    "footnotes_renamed": [],
                },
            },
            {
                "description": "Material semantic wording update should be exposed as footnotes_renamed.",
                "previous_footnotes": [{"id": "7", "text": "Comprennent les engagements de la Banque."}],
                "current_footnotes": [
                    {
                        "id": "8",
                        "text": "Comprennent aussi les engagements de la Banque.",
                    }
                ],
                "expected_output": {
                    "footnotes_added": [],
                    "footnotes_removed": [],
                    "footnotes_renamed": [
                        {
                            "previous_id": "7",
                            "current_id": "8",
                            "previous_text": "Comprennent les engagements de la Banque.",
                            "current_text": "Comprennent aussi les engagements de la Banque.",
                            "reason": "Same note with materially revised wording.",
                            "analyst_assessment": {
                                "relevance_level": 2,
                                "justification": "La clarification de la portée des engagements élargit le périmètre d'inclusion comptable, ce qui justifie une révision analytique.",
                            },
                        }
                    ],
                },
            },
        ],
        "pair_context": {
            "previous_table": _table_context(previous_table),
            "current_table": _table_context(current_table),
            "indicator_diff": {
                "indicators_added": indicator_diff["indicators_added"],
                "indicators_removed": indicator_diff["indicators_removed"],
                "indicators_renamed": indicator_diff["indicators_renamed"],
            },
        },
    }
    if det_fn_hints:
        prompt.update(det_fn_hints)

    data = _call_validated_diff_json(
        system_prompt=FOOTNOTE_DIFF_SYSTEM_PROMPT,
        prompt=prompt,
        required_list_fields=(
            "footnotes_added",
            "footnotes_removed",
            "footnotes_renamed",
        ),
        model=model,
        call_kind="diff_footnotes",
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
        response_model=FootnoteDiffResponse,
    )
    return {
        "footnotes_added": data["footnotes_added"],
        "footnotes_removed": data["footnotes_removed"],
        "footnotes_renamed": data["footnotes_renamed"],
        "reason": data["reason"],
    }


# ---------------------------------------------------------------------------
# Post-diff GPT Inspector (artifact filter)
# ---------------------------------------------------------------------------


def _inspect_diff_artifacts_gpt(
    indicator_diff: dict[str, Any],
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Filtre les artefacts d'extraction du diff d'indicateurs via un appel GPT Inspector.

    Returns:
        Copie nettoyee de *indicator_diff* dont les artefacts ont ete retires.
    """
    added = list(indicator_diff.get("indicators_added", []) or [])
    removed = list(indicator_diff.get("indicators_removed", []) or [])

    # Nothing to inspect — skip the call entirely
    if not added and not removed:
        return indicator_diff

    prev_ctx = _table_context(previous_table)
    curr_ctx = _table_context(current_table)

    prompt: dict[str, Any] = {
        "task": (
            "Inspect each added/removed indicator from a first-pass diff and classify "
            "it as a REAL semantic change or an extraction ARTIFACT."
        ),
        "rules": [
            "Return JSON only and strictly follow the response_schema.",
            "For each entry in added_indicators and removed_indicators, provide a verdict: 'real' or 'artifact'.",
            "If an added and a removed indicator refer to the same business concept (extraction noise), "
            "mark BOTH as 'artifact' and list them in artifact_pairs.",
            "Be strict: when in doubt, prefer 'artifact'. False positives are worse than missing a real change.",
        ],
        "response_schema": {
            "added_verdicts": [
                {
                    "value": "string",
                    "verdict": "'real' or 'artifact'",
                    "reason": "string",
                }
            ],
            "removed_verdicts": [
                {
                    "value": "string",
                    "verdict": "'real' or 'artifact'",
                    "reason": "string",
                }
            ],
            "artifact_pairs": [{"removed": "string", "added": "string", "reason": "string"}],
        },
        "added_indicators": [item.get("value", "") for item in added],
        "removed_indicators": [item.get("value", "") for item in removed],
        "previous_table": {
            "title": prev_ctx["title"],
            "indicators": prev_ctx["indicators"],
        },
        "current_table": {
            "title": curr_ctx["title"],
            "indicators": curr_ctx["indicators"],
        },
    }

    # Also pass existing renames for context
    renamed = list(indicator_diff.get("indicators_renamed", []) or [])
    if renamed:
        prompt["already_renamed"] = [
            {"previous": r.get("previous", ""), "current": r.get("current", "")} for r in renamed
        ]

    data = call_openai_json(
        model=model,
        messages=[
            {"role": "system", "content": INSPECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        usage_recorder=usage_recorder,
        call_kind="inspect_artifacts",
        response_model=InspectorResponse,
    )

    # --- Parse verdicts and filter ---
    try:
        artifact_added = {
            v["value"] for v in data.get("added_verdicts", []) if v.get("verdict", "").lower() == "artifact"
        }
        artifact_removed = {
            v["value"] for v in data.get("removed_verdicts", []) if v.get("verdict", "").lower() == "artifact"
        }
    except (KeyError, TypeError, AttributeError):
        logger.warning("Inspector returned malformed response — skipping artifact filtering.")
        return indicator_diff

    filtered_added = [item for item in added if str(item.get("value", "")).strip() not in artifact_added]
    filtered_removed = [item for item in removed if str(item.get("value", "")).strip() not in artifact_removed]

    n_filtered = (len(added) - len(filtered_added)) + (len(removed) - len(filtered_removed))
    if n_filtered:
        logger.info(
            "Inspector filtered %d artifact(s) from indicator diff.",
            n_filtered,
        )

    return {
        **indicator_diff,
        "indicators_added": filtered_added,
        "indicators_removed": filtered_removed,
    }


def _compose_pair_reason(
    *,
    indicator_reason: str,
    footnote_reason: str,
    has_indicator_changes: bool,
    has_footnote_changes: bool,
) -> str:
    """Compose la raison globale d'une paire a partir des raisons indicateurs et notes."""
    indicator_reason = str(indicator_reason or "").strip()
    footnote_reason = str(footnote_reason or "").strip()
    if has_indicator_changes and has_footnote_changes:
        parts = [part for part in (indicator_reason, footnote_reason) if part]
        if parts:
            if len(parts) == 2 and parts[0] == parts[1]:
                return parts[0]
            return " ".join(parts)
        return "Des changements sémantiques ont été détectés sur les indicateurs et les notes de bas de page."
    if has_indicator_changes:
        return indicator_reason or "Des changements sémantiques ont été détectés sur les indicateurs."
    if has_footnote_changes:
        return footnote_reason or "Des changements sémantiques ont été détectés sur les notes de bas de page."
    return indicator_reason or footnote_reason or "Aucun changement sémantique détecté."


def diff_table_pair_gpt(
    previous_table: dict[str, Any],
    current_table: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    max_validation_attempts: int = 3,
) -> dict[str, Any]:
    """Orchestre le diff complet (indicateurs + notes + inspection) d'une paire de tableaux.

    Enchaine le diff d'indicateurs, le diff de notes de bas de page et
    l'inspection des artefacts pour produire le ``technical_diff`` final.

    Args:
        previous_table: Tableau du trimestre precedent.
        current_table: Tableau du trimestre courant.
        model: Identifiant du modele OpenAI.
        call_openai_json: Fonction injectee pour l'appel OpenAI.
        usage_recorder: Accumulateur optionnel de metriques d'utilisation.
        max_validation_attempts: Nombre maximal de tentatives de validation.

    Returns:
        Dictionnaire contenant ``technical_diff``, ``reason``, ``diff_mode``
        et ``diff_calls_total``.
    """
    indicator_diff = diff_indicators_pair_gpt(
        previous_table,
        current_table,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )
    footnote_diff = diff_footnotes_pair_gpt(
        previous_table,
        current_table,
        indicator_diff=indicator_diff,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
        max_validation_attempts=max_validation_attempts,
    )

    # --- Post-diff GPT Inspector (artifact filter) ---
    pre_inspect_adds = list(indicator_diff.get("indicators_added", []) or [])
    pre_inspect_removes = list(indicator_diff.get("indicators_removed", []) or [])
    inspector_called = bool(pre_inspect_adds or pre_inspect_removes)
    indicator_diff = _inspect_diff_artifacts_gpt(
        indicator_diff,
        previous_table,
        current_table,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )

    previous_footnotes = _normalize_footnotes(previous_table.get("footnotes", []))
    current_footnotes = _normalize_footnotes(current_table.get("footnotes", []))
    footnote_gpt_called = bool(previous_footnotes and current_footnotes)

    technical_diff: dict[str, Any] = {
        "indicators_added": indicator_diff["indicators_added"],
        "indicators_removed": indicator_diff["indicators_removed"],
        "indicators_renamed": indicator_diff["indicators_renamed"],
        "footnotes_added": footnote_diff["footnotes_added"],
        "footnotes_removed": footnote_diff["footnotes_removed"],
        "footnotes_renamed": footnote_diff["footnotes_renamed"],
    }
    has_changes = any(technical_diff.values())
    technical_diff["table_level_change"] = "modifie" if has_changes else "inchange"
    reason = _compose_pair_reason(
        indicator_reason=str(indicator_diff.get("reason", "") or ""),
        footnote_reason=str(footnote_diff.get("reason", "") or ""),
        has_indicator_changes=any(
            technical_diff[field]
            for field in (
                "indicators_added",
                "indicators_removed",
                "indicators_renamed",
            )
        ),
        has_footnote_changes=any(
            technical_diff[field]
            for field in (
                "footnotes_added",
                "footnotes_removed",
                "footnotes_renamed",
            )
        ),
    )
    return {
        "technical_diff": technical_diff,
        "reason": reason,
        "diff_mode": "gpt",
        "diff_calls_total": ((2 if footnote_gpt_called else 1) + (1 if inspector_called else 0)),
    }
