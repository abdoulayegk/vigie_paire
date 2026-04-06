"""Pass 3 — Triage réglementaire par changement de paragraphe.

Pour chaque changement non-unchanged issu du diff sémantique (Pass 2),
un appel GPT-4o classifie la pertinence, l'impact et le type de risque.
Les blocs unchanged sont marqués source="skip" sans appel LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset(
    {
        "REGLEMENTAIRE",
        "RISQUE",
        "CAPITAL",
        "STRUCTURE",
        "COSMETIQUE",
        "INCONNU",
    }
)

VALID_IMPACT_LEVELS = frozenset({"MAJEUR", "MODERE", "MINEUR"})

VALID_RELEVANCE = frozenset({"ELEVEE", "MOYENNE", "FAIBLE"})

VALID_RISK_LEVELS = frozenset({"ELEVE", "MODERE", "FAIBLE"})

VALID_RISK_TYPES = frozenset({"credit", "marche", "operationnel", "liquidite", "capital", "autre"})

VALID_ACTIONS = frozenset({"escalade", "investigation", "confirmation", "information", "aucune"})

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM_PROMPT = """\
Tu es un analyste senior en réglementation bancaire canadienne (BSIF/OSFI). \
On te soumet un changement de paragraphe détecté entre deux trimestres consécutifs \
d'un rapport bancaire (section gestion du capital, gestion des risques, ou \
faits nouveaux en matière de réglementation).

Ton rôle : évaluer si ce changement est pertinent pour la veille réglementaire \
et le classifier précisément.

RÈGLES STRICTES D'EXCLUSION — marque comme NON PERTINENT :
- Les simples mises à jour de dates ou de trimestres (ex : T1 2025 → T2 2025).
- Les variations de valeurs numériques propres à la banque (chiffre d'affaires, \
  rendements, nombre d'actions) qui ne sont pas des seuils réglementaires.
- Les changements purement cosmétiques (reformulations sans changement de sens).

RÈGLES STRICTES D'INCLUSION — marque comme PERTINENT :
- Ajout ou suppression de paragraphes liés à la réglementation (Bâle III/IV, \
  TLAC, LCR, NSFR, CET1, coussins de fonds propres, BSIF, OSFI, etc.).
- Reformulations substantielles modifiant l'engagement, la méthodologie ou \
  les risques identifiés.
- Changements touchant les ratios prudentiels, seuils réglementaires, ou \
  définitions de capital.
- Ajout ou suppression de risques identifiés ou de facteurs d'atténuation.
- Changements de politique interne liés à la conformité réglementaire.

RÉPONDRE UNIQUEMENT en JSON valide, sans markdown, selon ce schéma exact :
{
  "is_relevant": true | false,
  "category": "REGLEMENTAIRE" | "RISQUE" | "CAPITAL" | "STRUCTURE" | "COSMETIQUE" | "INCONNU",
  "impact_level": "MAJEUR" | "MODERE" | "MINEUR",
  "risk_type": "credit" | "marche" | "operationnel" | "liquidite" | "capital" | "autre",
  "relevance_score": "ELEVEE" | "MOYENNE" | "FAIBLE",
  "risk_level": "ELEVE" | "MODERE" | "FAIBLE",
  "confidence": 0.0 à 1.0,
  "explanation": "<1-2 phrases en français expliquant la pertinence>",
  "impact_description": "<1 phrase décrivant l'impact concret du changement>",
  "action_requise": "escalade" | "investigation" | "confirmation" | "information" | "aucune",
  "reference_reglementaire": "<référence si applicable, ex: BSIF — Ligne directrice A-1, sinon chaîne vide>",
  "signals": {
    "regulatory_reference_added": true | false,
    "methodology_change": true | false,
    "tone_change": true | false,
    "forward_looking": true | false,
    "quantitative_change": true | false
  }
}

GUIDE pour impact_level :
- MAJEUR : changement critique affectant les ratios prudentiels ou la conformité réglementaire.
- MODERE : changement méthodologique ou structurel à surveiller.
- MINEUR : changement mineur de formulation ou de portée limitée.

GUIDE pour risk_type :
- credit : risque de crédit, provisionnement, pertes sur prêts.
- marche : risque de marché, VaR, expositions de trading.
- operationnel : risque opérationnel, continuité, fraude, cyber.
- liquidite : risque de liquidité, LCR, NSFR, financement.
- capital : adéquation du capital, ratios CET1, Tier 1, TLAC.
- autre : si aucun type ci-dessus ne s'applique.

GUIDE pour signals :
- regulatory_reference_added : mention explicite d'une nouvelle directive BSIF/Bâle/loi.
- methodology_change : changement de méthode de calcul, de périmètre ou de définition.
- tone_change : passage d'un ton neutre à préoccupé (ou inversement).
- forward_looking : nouveau langage prospectif (ex : "nous prévoyons", "risque émergent").
- quantitative_change : nouveau seuil, ratio ou valeur numérique réglementaire.

GUIDE pour action_requise :
- escalade : changement critique nécessitant une revue immédiate par un senior.
- investigation : anomalie ou surprise nécessitant une analyse approfondie.
- confirmation : changement attendu à confirmer comme normal.
- information : changement mineur, pour information seulement.
- aucune : non pertinent, aucune action.
"""

_SUMMARY_SYSTEM_PROMPT = """\
Tu es un analyste senior en réglementation bancaire canadienne (BSIF/OSFI). \
On te soumet la liste des changements de paragraphes jugés pertinents dans \
un rapport bancaire entre deux trimestres consécutifs.

Produis un résumé exécutif structuré.

RÉPONDRE UNIQUEMENT en JSON valide, sans markdown, selon ce schéma exact :
{
  "executive_overview": "<2-4 phrases résumant les changements les plus importants>",
  "key_highlights": ["<point clé 1>", "<point clé 2>", ...],
  "pertinence_globale": "ELEVEE" | "MOYENNE" | "FAIBLE"
}
"""

# ---------------------------------------------------------------------------
# Helpers — prompt builders
# ---------------------------------------------------------------------------

_SECTION_DISPLAY: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}


def _build_change_prompt(
    section_key: str,
    change: dict[str, Any],
) -> str:
    """Construit le prompt utilisateur pour un changement de paragraphe."""
    section_title = _SECTION_DISPLAY.get(section_key, section_key)
    change_type = change.get("change_type", "")
    text_t1 = (change.get("text_t1") or "").strip()
    text_t2 = (change.get("text_t2") or "").strip()
    summary = (change.get("change_summary") or "").strip()

    lines = [
        f"Section : {section_title}",
        f"Type de changement : {change_type.upper()}",
        "",
    ]

    if summary:
        lines += [f"Résumé du changement : {summary}", ""]

    if change_type == "added":
        lines += [
            "=== PARAGRAPHE AJOUTÉ (absent du trimestre précédent) ===",
            text_t2 or "(vide)",
        ]
    elif change_type == "removed":
        lines += [
            "=== PARAGRAPHE SUPPRIMÉ (absent du trimestre courant) ===",
            text_t1 or "(vide)",
        ]
    else:  # modified
        lines += [
            "=== VERSION PRÉCÉDENTE ===",
            text_t1 or "(vide)",
            "",
            "=== VERSION COURANTE ===",
            text_t2 or "(vide)",
        ]

    return "\n".join(lines)


def _build_summary_prompt(relevant_changes: list[dict[str, Any]]) -> str:
    """Construit le prompt pour le résumé global."""
    parts = []
    for i, c in enumerate(relevant_changes, 1):
        section = c.get("_section", "")
        change_type = c.get("_change_type", "")
        explanation = c.get("explanation", "")
        impact = c.get("impact_description", "")
        parts.append(
            f"[{i}] Section={section}, Type={change_type}, "
            f"Impact={c.get('impact_level', '')}\n"
            f"  Explication : {explanation}\n"
            f"  Impact : {impact}"
        )
    return "Changements pertinents détectés :\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers — async LLM
# ---------------------------------------------------------------------------


async def _call_openai_json_async(
    client: Any,
    *,
    system: str,
    user: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 600,
) -> dict[str, Any] | None:
    """Appel asynchrone à OpenAI retournant du JSON parsé."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return json.loads(raw)
    except Exception as exc:
        logger.warning("text_triage: appel LLM échoué — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers — validation
# ---------------------------------------------------------------------------


def _validate_triage_response(data: dict[str, Any] | None) -> dict[str, Any]:
    """Valide et normalise une réponse LLM de triage individuelle."""
    default = {
        "is_relevant": False,
        "category": "INCONNU",
        "impact_level": "MINEUR",
        "risk_type": "autre",
        "relevance_score": "FAIBLE",
        "risk_level": "FAIBLE",
        "confidence": 0.0,
        "explanation": "",
        "impact_description": "",
        "action_requise": "aucune",
        "reference_reglementaire": "",
        "signals": {
            "regulatory_reference_added": False,
            "methodology_change": False,
            "tone_change": False,
            "forward_looking": False,
            "quantitative_change": False,
        },
        "source": "heuristic",
    }

    if not data or not isinstance(data, dict):
        return default

    is_relevant = bool(data.get("is_relevant", False))

    category = str(data.get("category") or "INCONNU").upper()
    if category not in VALID_CATEGORIES:
        category = "INCONNU"

    impact_level = str(data.get("impact_level") or "MINEUR").upper()
    if impact_level not in VALID_IMPACT_LEVELS:
        impact_level = "MINEUR"

    risk_type = str(data.get("risk_type") or "autre").lower()
    if risk_type not in VALID_RISK_TYPES:
        risk_type = "autre"

    relevance = str(data.get("relevance_score") or "FAIBLE").upper()
    if relevance not in VALID_RELEVANCE:
        relevance = "FAIBLE"

    risk_level = str(data.get("risk_level") or "FAIBLE").upper()
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "FAIBLE"

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    action_requise = str(data.get("action_requise") or "aucune").lower()
    if action_requise not in VALID_ACTIONS:
        action_requise = "aucune"

    # Validate signals
    raw_signals = data.get("signals") or {}
    if not isinstance(raw_signals, dict):
        raw_signals = {}
    signals = {
        "regulatory_reference_added": bool(raw_signals.get("regulatory_reference_added", False)),
        "methodology_change": bool(raw_signals.get("methodology_change", False)),
        "tone_change": bool(raw_signals.get("tone_change", False)),
        "forward_looking": bool(raw_signals.get("forward_looking", False)),
        "quantitative_change": bool(raw_signals.get("quantitative_change", False)),
    }

    return {
        "is_relevant": is_relevant,
        "category": category,
        "impact_level": impact_level,
        "risk_type": risk_type,
        "relevance_score": relevance,
        "risk_level": risk_level,
        "confidence": confidence,
        "explanation": str(data.get("explanation") or "")[:500],
        "impact_description": str(data.get("impact_description") or "")[:500],
        "action_requise": action_requise,
        "reference_reglementaire": str(data.get("reference_reglementaire") or "")[:200],
        "signals": signals,
        "source": "llm",
    }


def _validate_summary_response(data: dict[str, Any] | None) -> dict[str, Any]:
    """Valide et normalise la réponse LLM du résumé global."""
    if not data or not isinstance(data, dict):
        return {
            "executive_overview": "",
            "key_highlights": [],
            "pertinence_globale": "FAIBLE",
            "source": "heuristic",
        }

    highlights = data.get("key_highlights")
    if not isinstance(highlights, list):
        highlights = []
    highlights = [str(h)[:300] for h in highlights if h][:10]

    overall = str(data.get("pertinence_globale") or "FAIBLE").upper()
    if overall not in VALID_RELEVANCE:
        overall = "FAIBLE"

    return {
        "executive_overview": str(data.get("executive_overview") or "")[:2000],
        "key_highlights": highlights,
        "pertinence_globale": overall,
        "source": "llm",
    }


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------


async def _triage_section_changes(
    client: Any,
    semaphore: asyncio.Semaphore,
    section_key: str,
    changes: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    """Triage LLM de tous les changements d'une section.

    Les paragraphes ``unchanged`` sont marqués directement (source=skip).
    Pour les autres, un appel GPT est créé par changement, limité par
    le *semaphore* partagé. Chaque réponse est validée par
    ``_validate_triage_response``.

    Args:
        client: Instance AsyncOpenAI.
        semaphore: Sémaphore de contrôle de concurrence.
        section_key: Clé de section (ex. ``gestion_capital``).
        changes: Liste mutable de dicts changement ; enrichis *in place*
            avec la clé ``genai_triage``.
        model: Modèle OpenAI.

    Returns:
        La même liste *changes*, enrichie du champ ``genai_triage`` sur
        chaque élément.
    """
    tasks: list[tuple[int, asyncio.Task]] = []

    for idx, change in enumerate(changes):
        if change.get("change_type") == "unchanged":
            change["genai_triage"] = {
                "is_relevant": False,
                "category": "COSMETIQUE",
                "impact_level": "MINEUR",
                "risk_type": "autre",
                "relevance_score": "FAIBLE",
                "risk_level": "FAIBLE",
                "confidence": 1.0,
                "explanation": "Paragraphe inchangé.",
                "impact_description": "",
                "action_requise": "aucune",
                "reference_reglementaire": "",
                "signals": {
                    "regulatory_reference_added": False,
                    "methodology_change": False,
                    "tone_change": False,
                    "forward_looking": False,
                    "quantitative_change": False,
                },
                "source": "skip",
            }
            continue

        prompt = _build_change_prompt(section_key, change)

        async def _run(p: str = prompt) -> dict[str, Any] | None:
            async with semaphore:
                return await _call_openai_json_async(client, system=_TRIAGE_SYSTEM_PROMPT, user=p, model=model)

        task = asyncio.create_task(_run())
        tasks.append((idx, task))

    if tasks:
        await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)

    for idx, task in tasks:
        try:
            raw = task.result()
        except Exception:
            raw = None
        changes[idx]["genai_triage"] = _validate_triage_response(raw)

    return changes


async def _run_triage_async(
    section_diffs: dict[str, list[dict[str, Any]]],
    model: str = "gpt-4o",
    max_concurrency: int = 20,
) -> dict[str, Any]:
    """Orchestre le triage réglementaire sur toutes les sections en parallèle.

    Crée un client AsyncOpenAI, lance le triage par section via
    ``_triage_section_changes``, puis génère un résumé exécutif global
    pour les changements jugés pertinents. Retombe sur
    ``_fallback_triage`` si la clé API est absente.

    Args:
        section_diffs: ``{section_key: [change_dicts]}`` depuis
            ``run_section_diff()``.
        model: Modèle OpenAI (défaut gpt-4o).
        max_concurrency: Nombre maximal d'appels LLM simultanés.

    Returns:
        ``{"section_diffs_enriched": {...}, "global_summary": {...}}``.
    """
    from openai import AsyncOpenAI

    from vigilance.utils.genai import get_openai_api_key

    api_key = get_openai_api_key()
    if not api_key:
        logger.warning("text_triage: OPENAI_API_KEY non définie — triage ignoré.")
        return _fallback_triage(section_diffs)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    # Triage par section (en parallèle entre sections)
    section_tasks: list[tuple[str, asyncio.Task]] = []
    for section_key, changes in section_diffs.items():
        if not changes:
            continue
        task = asyncio.create_task(_triage_section_changes(client, semaphore, section_key, changes, model))
        section_tasks.append((section_key, task))

    if section_tasks:
        await asyncio.gather(*(t[1] for t in section_tasks), return_exceptions=True)

    enriched: dict[str, list[dict[str, Any]]] = {}
    for section_key, changes in section_diffs.items():
        enriched[section_key] = changes  # mutated in place

    # For tasks that ran, retrieve results (already mutated in place)
    for section_key, task in section_tasks:
        try:
            enriched[section_key] = task.result()
        except Exception as exc:
            logger.warning("text_triage: erreur récupération section %s — %s", section_key, exc)

    # Collect relevant changes for global summary
    relevant: list[dict[str, Any]] = []
    for section_key, changes in enriched.items():
        for change in changes:
            triage = change.get("genai_triage") or {}
            if triage.get("is_relevant"):
                entry = dict(triage)
                entry["_section"] = section_key
                entry["_change_type"] = change.get("change_type", "")
                relevant.append(entry)

    # Global summary
    if relevant:
        summary_prompt = _build_summary_prompt(relevant)
        summary_raw = await _call_openai_json_async(
            client,
            system=_SUMMARY_SYSTEM_PROMPT,
            user=summary_prompt,
            model=model,
            max_tokens=1500,
        )
        global_summary = _validate_summary_response(summary_raw)
    else:
        global_summary = {
            "executive_overview": "Aucun changement réglementaire ou structurel significatif détecté dans les paragraphes de ce trimestre.",
            "key_highlights": [],
            "pertinence_globale": "FAIBLE",
            "source": "heuristic",
        }

    # Compute counts
    total = sum(len(v) for v in enriched.values())
    by_impact: dict[str, int] = {"MAJEUR": 0, "MODERE": 0, "MINEUR": 0}
    by_category: dict[str, int] = {}
    by_action: dict[str, int] = {}

    for changes in enriched.values():
        for change in changes:
            triage = change.get("genai_triage") or {}
            if triage.get("source") == "skip":
                continue
            imp = triage.get("impact_level", "MINEUR")
            by_impact[imp] = by_impact.get(imp, 0) + 1
            cat = triage.get("category", "INCONNU")
            by_category[cat] = by_category.get(cat, 0) + 1
            action = triage.get("action_requise", "aucune")
            by_action[action] = by_action.get(action, 0) + 1

    global_summary["counts"] = {
        "total": total,
        "total_relevant": len(relevant),
        "by_impact": by_impact,
        "by_category": by_category,
        "by_action": by_action,
    }

    return {
        "section_diffs_enriched": enriched,
        "global_summary": global_summary,
    }


def _fallback_triage(
    section_diffs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Triage heuristique de repli utilisé quand la clé API est absente.

    Marque les paragraphes ``unchanged`` comme non pertinents et tous les
    autres comme pertinents avec un impact modéré et une confiance nulle.

    Args:
        section_diffs: ``{section_key: [change_dicts]}`` — enrichis *in place*.

    Returns:
        Même structure que ``_run_triage_async`` avec ``source=heuristic``.
    """
    for changes in section_diffs.values():
        for change in changes:
            if change.get("change_type") == "unchanged":
                change["genai_triage"] = {
                    "is_relevant": False,
                    "source": "skip",
                    "category": "COSMETIQUE",
                    "impact_level": "MINEUR",
                    "risk_type": "autre",
                    "relevance_score": "FAIBLE",
                    "risk_level": "FAIBLE",
                    "confidence": 1.0,
                    "explanation": "Paragraphe inchangé.",
                    "impact_description": "",
                    "action_requise": "aucune",
                    "reference_reglementaire": "",
                    "signals": {
                        k: False
                        for k in (
                            "regulatory_reference_added",
                            "methodology_change",
                            "tone_change",
                            "forward_looking",
                            "quantitative_change",
                        )
                    },
                }
            else:
                change["genai_triage"] = {
                    "is_relevant": True,
                    "source": "heuristic",
                    "category": "INCONNU",
                    "impact_level": "MODERE",
                    "risk_type": "autre",
                    "relevance_score": "MOYENNE",
                    "risk_level": "MODERE",
                    "confidence": 0.0,
                    "explanation": "",
                    "impact_description": "",
                    "action_requise": "information",
                    "reference_reglementaire": "",
                    "signals": {
                        k: False
                        for k in (
                            "regulatory_reference_added",
                            "methodology_change",
                            "tone_change",
                            "forward_looking",
                            "quantitative_change",
                        )
                    },
                }
    return {
        "section_diffs_enriched": section_diffs,
        "global_summary": {
            "executive_overview": "",
            "key_highlights": [],
            "pertinence_globale": "FAIBLE",
            "source": "heuristic",
            "counts": {"total": 0, "total_relevant": 0, "by_impact": {}, "by_category": {}, "by_action": {}},
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_text_triage(
    section_diffs: dict[str, list[dict[str, Any]]],
    model: str = "gpt-4o",
    max_concurrency: int = 20,
) -> dict[str, Any]:
    """Lance le triage réglementaire sur tous les changements (synchrone).

    Args:
        section_diffs: {section_key: [change_dicts]} depuis run_section_diff().
        model: Modèle OpenAI (défaut gpt-4o).
        max_concurrency: Max appels LLM en parallèle.

    Returns:
        {
            "section_diffs_enriched": {section_key: [change_with_genai_triage]},
            "global_summary": {...}
        }
    """
    if not section_diffs:
        return {
            "section_diffs_enriched": {},
            "global_summary": {
                "executive_overview": "Aucune section à analyser.",
                "key_highlights": [],
                "pertinence_globale": "FAIBLE",
                "source": "heuristic",
                "counts": {"total": 0, "total_relevant": 0, "by_impact": {}, "by_category": {}, "by_action": {}},
            },
        }

    return asyncio.run(
        _run_triage_async(
            section_diffs=section_diffs,
            model=model,
            max_concurrency=max_concurrency,
        )
    )
