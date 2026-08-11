"""Comparaison LLM des indicateurs et notes de bas de page."""

from __future__ import annotations

import json
from typing import Any, Callable

from vigie.comparaison.differences.comparaison_deterministe import (
    _deterministic_footnote_diff,
)
from vigie.comparaison.differences.normalisation_elements import (
    _normalize_footnotes,
    _table_context,
)
from vigie.support.models.comparison_models import (
    FootnoteDiffResponse,
    IndicatorDiffResponse,
)

INDICATOR_DIFF_SYSTEM_PROMPT = """
You are a precision-first banking table indicator diff engine.

You compare two already-matched canonical banking tables from adjacent quarterly reports.
Each indicator is provided with its structural position ("pos") and original name ("name").
Some indicators also have a "normalized" field — a cleaned version with footnote markers,
hierarchical section prefixes, and punctuation noise removed.

Your task is to report only meaningful semantic indicator changes:
- indicators_added
- indicators_removed
- indicators_renamed

MATCHING STRATEGY:
1. First, match indicators using the "normalized" field when present, falling back to "name".
2. Two indicators that share the same "normalized" value (or whose normalized forms match)
   at the same or nearby position are THE SAME indicator — do not report them as changed.
3. Always report the original "name" in your output, never the normalized form.

Rules:
- The table pair is already matched. Do not question the pairing.
- Compare only indicator meaning and role in the table.
- Ignore numeric values, dates, periods, formatting, OCR noise, row order changes, and line wrapping.
- IGNORE footnote marker changes: if two indicators differ ONLY by a footnote reference like (1), (2), (3), (4) being added, removed, or changed, they are THE SAME indicator. Do NOT classify this as renamed. Example: 'catégorie 1 (4)' and 'catégorie 1' are identical — ignore this.
- IGNORE extraction disambiguation labels: tables with repeated sub-sections may be extracted with
  disambiguation tags like '(bloc 2)', '(bloc 3)' appended, or date prefixes like '31 octobre 2025 – '
  prepended. These are NOT part of the real indicator name. When matching, strip these prefixes/suffixes
  and compare the core business name.
- POSITIONAL ALIGNMENT: Use the structural position ("pos") of indicators as a strong matching signal.
  When two indicators occupy the SAME position in their respective tables and share overlapping
  meaning (even if names differ due to extraction inconsistencies), they are very likely the SAME indicator.
  Look at surrounding indicators (same neighbors above/below) as confirmation.
- HIERARCHICAL SUB-INDICATORS (CRITICAL): Banking tables often use group headers
  (e.g. "Ratios des fonds propres") followed by sub-indicators (e.g. "CET1", "catégorie 1").
  One quarter's extraction may fully qualify sub-indicators with the group header prefix
  (e.g. "Ratios des fonds propres – CET1") while another quarter extracts them as bare names
  (e.g. "CET1"). These are the SAME indicator. The "normalized" field already strips these prefixes
  for you — use it as the primary matching key.
- DATE AND HEADER ARTIFACTS: Some extractions include date headers (e.g. "Au 31 janvier 2026")
  or section titles as indicators. These are extraction artifacts, not real indicator additions or removals.
  Ignore them if they appear on only one side.
- PUNCTUATION-ONLY DIFFERENCES: Changes like "–" vs "-", trailing colons, commas added/removed,
  or spacing variations are NEVER meaningful renames. Ignore them entirely.
- SUPERSCRIPT vs PARENTHESIZED footnotes: '²' and '(2)' are the same footnote marker format —
  differences between them are NOT meaningful.
- Indicator present only in current = indicators_added.
- Indicator present only in previous = indicators_removed.
- Classify indicators_renamed only when the previous and current indicators clearly represent the exact same business concept with the same scope and the same role in the table AND the name change itself is semantically meaningful (not just extraction noise).
- If the change could instead be explained by addition, removal, scope change, row split, or row merge, do NOT classify it as renamed.
- When unsure between rename and add/remove, prefer add/remove.
- If one previous indicator appears split into multiple current indicators, treat the new rows as additions rather than rename.
- CONSOLIDATION (N→1) — STRICT RULE: If 2 or more previous indicators could map to a single current indicator (e.g., "Obligations du gouvernement du Canada", "Obligations d'agences fédérales des États-Unis", "Obligations de gouvernements provinciaux" all potentially mapping to a new "Obligations du gouvernement, d'organismes fédéraux, d'entités du secteur public..."), you MUST treat ALL previous indicators as indicators_removed and the new current indicator as indicators_added. Do NOT pick one previous indicator arbitrarily as "renamed" and silently absorb the others. A renamed pair requires a strict 1:1 semantic match — one previous indicator maps to exactly one current indicator with no other previous indicator as a plausible alternative.
- Be conservative and report only clear semantic differences.
- For each change (added, removed, renamed), you MUST act as a Senior Risk Analyst and provide an 'analyst_assessment'.
- The 'analyst_assessment' MUST include:
  1. A 'relevance_level' (integer: 1 for Critical/Regulatory, 2 for High/Structural, 3 for Low/Cosmetic).
  2. A 'justification' : un texte fluide et naturel d'au moins 3 paragraphes EN FRANÇAIS (separés par \n\n). NE PAS écrire « Paragraphe 1 » ou « Paragraphe 2 » ni aucune étiquette de structure -- le texte doit se lire comme une analyse rédigée par un analyste senior.
     Le contenu doit couvrir, dans l'ordre, ces trois aspects :
     (a) Décrire précisément le changement constaté (quel indicateur, quelle valeur, quel tableau, quel contexte).
     (b) Expliquer l'impact métier ou réglementaire concret (lien avec Bâle III, BSIF, ratios prudentiels, divulgation, méthodologie de calcul, etc.). Mentionner les références réglementaires applicables si pertinent.
     (c) Conclure par une lecture de vigie qui explique le signal métier à retenir, sans demander à l'analyste de valider, accepter ou rejeter le changement.
     Par exemple : « L'indicateur "Ratio CET1" a été ajouté dans le tableau de synthèse des fonds propres au T2 2025, alors qu'il était absent au T1 2025.\n\nCet ajout répond aux exigences de divulgation Bâle III sur les ratios de fonds propres de catégorie 1 (CET1), conformément aux lignes directrices du BSIF. La présence de ce ratio dans le tableau de synthèse renforce la transparence envers les parties prenantes et s'inscrit dans le cadre des exigences TLAC.\n\nIl s'agit d'une nouvelle divulgation réglementaire. Le signal de vigie à retenir est que la banque rend plus visible un indicateur prudentiel utile pour lire le capital réglementaire et la comparabilité inter-pairs. »

IMPORTANT : Toutes les justifications doivent être rédigées exclusivement en français, avec au minimum 3 paragraphes distincts. Ne jamais inclure d'étiquettes comme « Paragraphe 1 » dans le texte.

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
- Footnote present only in current quarter = footnotes_added. A note absent from the previous table but present in the current table is an addition.
- Footnote present only in previous quarter = footnotes_removed. A note present in the previous table but absent from the current table is a REMOVAL — never classify it as footnotes_added.
- CRITICAL DIRECTION RULE: The direction (added vs removed) is determined solely by which quarter contains the note. A note that existed in the previous quarter and is GONE in the current quarter is ALWAYS footnotes_removed, regardless of its semantic content.
- Footnote with the same semantic meaning but materially revised wording = footnotes_renamed.
- Compare footnotes within the logical scope of the same table and in the context of the already-matched pair.
- Be conservative and report only clear semantic differences.
- For each change (added, removed, renamed), you MUST act as a Senior Risk Analyst and provide an 'analyst_assessment'.
- The 'analyst_assessment' MUST include:
  1. A 'relevance_level' (integer: 1 for Critical/Regulatory, 2 for High/Structural, 3 for Low/Cosmetic).
  2. A 'justification' : un texte fluide et naturel d'au moins 3 paragraphes EN FRANÇAIS (separés par \n\n). NE PAS écrire « Paragraphe 1 » ou « Paragraphe 2 » ni aucune étiquette de structure -- le texte doit se lire comme une analyse rédigée par un analyste senior.
     Le contenu doit couvrir, dans l'ordre, ces trois aspects :
     (a) Décrire précisément le changement constaté (quelle note, quel texte modifié, quel tableau, quel contexte).
     (b) Expliquer l'impact métier ou réglementaire concret (lien avec Bâle III, BSIF, ratios prudentiels, divulgation, méthodologie de calcul, etc.). Mentionner les références réglementaires applicables si pertinent.
     (c) Conclure par une lecture de vigie qui explique le signal métier à retenir, sans demander à l'analyste de valider, accepter ou rejeter le changement.
     Par exemple : « La note de bas de page n-3 a été ajoutée dans le tableau des provisions pour pertes de crédit attendues (ECL) au T2 2025.\n\nCette note précise un changement méthodologique dans le calcul des provisions de stade 2, en introduisant une pondération macroéconomique révisée conforme aux recommandations du BSIF sur les modèles IFRS 9.\n\nIl s'agit d'un changement méthodologique significatif. Le signal de vigie à retenir est que la banque rend plus explicite une méthode de provisionnement qui peut influencer la lecture du risque de crédit et la comparaison avec les pairs. »

IMPORTANT : Toutes les justifications doivent être rédigées exclusivement en français, avec au minimum 3 paragraphes distincts. Ne jamais inclure d'étiquettes comme « Paragraphe 1 » dans le texte.

Output must be valid JSON following the response_schema.
"""


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

    Envoie le contexte complet (avec position structurelle) a GPT pour
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

    rules = [
        "Return JSON only and strictly follow the response_schema.",
        "The two tables are already matched. Do not question the pairing.",
        "Compare only the canonical indicators using their name and structural position (pos).",
        "Ignore numeric values, dates, periods, formatting differences, OCR noise, row order changes, and line wrapping.",
        "Indicator present only in current = indicators_added.",
        "Indicator present only in previous = indicators_removed.",
        "Classify indicators_renamed only when the concept, scope, and role are clearly identical.",
        "When unsure between rename and add/remove, prefer add/remove.",
        "Do not treat row splits or row merges as renamed indicators unless the scope is clearly identical.",
    ]

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
                "direction_note": (
                    "footnotes_only_in_current → must become footnotes_added (absent in previous, new in current). "
                    "footnotes_only_in_previous → must become footnotes_removed (existed in previous, gone from current). "
                    "Do NOT swap these directions."
                ),
                "footnotes_only_in_current__must_be_ADDED": [
                    {"id": fn.get("id", ""), "text": fn.get("text", "")} for fn in det_fn_diff["det_added"]
                ],
                "footnotes_only_in_previous__must_be_REMOVED": [
                    {"id": fn.get("id", ""), "text": fn.get("text", "")} for fn in det_fn_diff["det_removed"]
                ],
                "footnotes_with_text_changes__must_be_RENAMED": [
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
            "CRITICAL — the direction is already established by the field names: "
            "'footnotes_only_in_current__must_be_ADDED' entries MUST go into footnotes_added (or be dismissed as pure renumbering/date-only); "
            "'footnotes_only_in_previous__must_be_REMOVED' entries MUST go into footnotes_removed (or be dismissed as pure renumbering/date-only); "
            "'footnotes_with_text_changes__must_be_RENAMED' entries MUST go into footnotes_renamed (or be dismissed). "
            "Do NOT reclassify the direction — a note that is only in the previous table can NEVER be footnotes_added."
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
