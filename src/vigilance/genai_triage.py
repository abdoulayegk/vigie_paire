"""Triage GenAI par lots -- analyse de pertinence LLM hors-ligne pour les resultats de comparaison.

Ce module s'execute **apres** le diff technique et **avant** le deploiement vers Dash.
Il enrichit chaque pair_comparison / tableau ajoute / tableau supprime avec une
classification de pertinence et une explication generees par LLM, puis produit un
resume executif global.

Tous les appels LLM ont lieu ici en mode batch (pipeline de nuit). Dash n'appelle
jamais ce module -- il ne lit que les champs pre-calcules dans comparison.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
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

VALID_RELEVANCE = frozenset({"ELEVEE", "MOYENNE", "FAIBLE"})

VALID_RISK_LEVELS = frozenset({"ELEVE", "MODERE", "FAIBLE"})

VALID_IMPACT_TYPES = frozenset({"structurel", "contenu", "methodologique", "cosmetique"})

VALID_PROJECT_PHASES = frozenset({"rapport_gestion", "pilier_3", "ifc", "autre"})

VALID_ACTIONS = frozenset({"escalade", "investigation", "confirmation", "information", "aucune"})

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM_PROMPT = """\
Tu es un analyste senior en réglementation bancaire canadienne (BSIF/OSFI). \
On te soumet un changement détecté entre deux trimestres consécutifs d'un \
rapport bancaire (rapport de gestion, Pilier 3, ou IFC).

Ton rôle : déterminer si ce changement est pertinent pour la veille \
réglementaire et l'expliquer brièvement.

RÈGLES STRICTES D'EXCLUSION — marque comme NON PERTINENT :
- Les simples mises à jour de dates ou de trimestres (ex: T1 2025 → T2 2025).
- Les déplacements de texte sans modification de fond.
- Les variations de valeurs numériques propres à la banque (chiffre d'affaires, \
  rendement, nombre d'actions, etc.) qui ne sont pas des seuils réglementaires.
- Les changements purement cosmétiques (mise en forme, renumérotation).

RÈGLES STRICTES D'INCLUSION — marque comme PERTINENT :
- Ajout ou suppression de lignes/colonnes liées à la réglementation \
  (Bâle III, TLAC, LCR, NSFR, CET1, coussin de fonds propres, etc.).
- Modifications de notes de bas de tableau signalant un changement \
  méthodologique ou réglementaire.
- Nouveaux tableaux de risque ou suppression de tableaux existants.
- Montants liés à la réglementation (seuils, ratios prudentiels).
- Changements structurels dans les sections gestion du capital ou gestion \
  des risques.

RÉPONDRE UNIQUEMENT en JSON valide, sans markdown, selon ce schéma exact :
{
  "is_relevant": true | false,
  "category": "REGLEMENTAIRE" | "RISQUE" | "CAPITAL" | "STRUCTURE" | "COSMETIQUE" | "INCONNU",
  "relevance_score": "ELEVEE" | "MOYENNE" | "FAIBLE",
  "risk_level": "ELEVE" | "MODERE" | "FAIBLE",
  "confidence": 0.0 à 1.0,
  "explanation": "<3-5 phrases en français structurées ainsi : (1) indiquer clairement si le changement est pertinent ou non et pourquoi ; (2) décrire l'impact réglementaire ou métier concret ; (3) préciser s'il s'agit d'une nouvelle divulgation, d'un changement méthodologique, ou d'une mise à jour ordinaire>",
  "impact_type": "structurel" | "contenu" | "methodologique" | "cosmetique",
  "project_phase": "rapport_gestion" | "pilier_3" | "ifc" | "autre",
  "action_requise": "escalade" | "investigation" | "confirmation" | "information" | "aucune",
  "reference_reglementaire": "<référence si applicable, ex: Bâle III — CET1, sinon chaîne vide>",
  "impact_description": "<1 phrase décrivant l'impact concret du changement>"
}

GUIDE pour risk_level :
- ELEVE : impact direct sur les ratios prudentiels, seuils réglementaires, ou conformité.
- MODERE : changement méthodologique ou structurel à surveiller.
- FAIBLE : changement mineur ou cosmétique.

GUIDE pour impact_type :
- structurel : ajout/suppression de lignes, colonnes, tableaux entiers.
- contenu : modification de valeurs, seuils, descriptions réglementaires.
- methodologique : changement de méthode de calcul, de périmètre, de définition.
- cosmetique : renommage, reformulation, mise en forme sans impact de fond.

GUIDE pour project_phase :
- rapport_gestion : sections gestion du capital, gestion des risques, texte risque.
- pilier_3 : tableaux de divulgation Pilier 3 (BSIF).
- ifc : rapports intermédiaires/condensés.
- autre : si la section ne correspond à aucune phase ci-dessus.

GUIDE pour action_requise :
- escalade : changement critique nécessitant une revue immédiate par un senior.
- investigation : anomalie ou surprise nécessitant une analyse approfondie.
- confirmation : changement attendu à confirmer comme normal.
- information : changement mineur, pour information seulement.
- aucune : non pertinent, aucune action.
"""

_SUMMARY_SYSTEM_PROMPT = """\
Tu es un analyste senior en réglementation bancaire canadienne. \
On te fournit la liste de tous les changements jugés pertinents entre \
deux trimestres d'un rapport bancaire.

Produis un résumé exécutif concis en français, structuré ainsi :

RÉPONDRE UNIQUEMENT en JSON valide, sans markdown :
{
  "executive_overview": "<paragraphe de 3-5 phrases résumant les évolutions clés>",
  "key_highlights": ["<point clé 1>", "<point clé 2>", "..."],
  "pertinence_globale": "ELEVEE" | "MOYENNE" | "FAIBLE",
  "par_phase": {
    "rapport_gestion": {"count": N, "resume": "<1 phrase>"},
    "pilier_3": {"count": N, "resume": "<1 phrase>"},
    "ifc": {"count": N, "resume": "<1 phrase>"},
    "autre": {"count": N, "resume": "<1 phrase>"}
  },
  "par_action": {
    "escalade": N,
    "investigation": N,
    "confirmation": N,
    "information": N,
    "aucune": N
  }
}

IMPORTANT :
- par_phase : ventiler les changements pertinents par phase projet.
- par_action : compter les changements par type d'action requise.
- Si une phase ou action n'a aucun changement, mettre count/N à 0.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_change_prompt(change: dict[str, Any], change_type: str) -> str:
    """Construit un prompt utilisateur decrivant un changement detecte pour le LLM.

    Args:
        change: Dictionnaire du changement (paire, tableau ajoute ou supprime).
        change_type: Type de changement : ``"pair"``, ``"added"`` ou ``"removed"``.

    Returns:
        Texte du prompt formate pour l'appel LLM de triage.
    """
    parts: list[str] = []
    section = (
        change.get("section")
        or (change.get("current_table") or {}).get("section")
        or (change.get("previous_table") or {}).get("section")
        or "inconnue"
    )

    if change_type == "pair":
        title_t1 = (change.get("previous_table") or {}).get("title", "")
        title_t2 = (change.get("current_table") or {}).get("title", "")
        title = title_t2 or title_t1 or "(sans titre)"
        parts.append(f"Section : {section}")
        parts.append(f"Tableau : {title}")

        diff = change.get("technical_diff") or {}
        added = diff.get("indicators_added") or []
        removed = diff.get("indicators_removed") or []
        renamed = diff.get("indicators_renamed") or []
        fn_added = diff.get("footnotes_added") or []
        fn_removed = diff.get("footnotes_removed") or []
        fn_renamed = diff.get("footnotes_renamed") or []
        status = diff.get("table_level_change", "inchange")

        parts.append(f"Statut du tableau : {status}")

        if added:
            names = [_indicator_label(i) for i in added[:15]]
            parts.append(f"Indicateurs ajoutés : {', '.join(names)}")
        if removed:
            names = [_indicator_label(i) for i in removed[:15]]
            parts.append(f"Indicateurs supprimés : {', '.join(names)}")
        if renamed:
            renames = [f"{r.get('previous', '')} → {r.get('current', '')}" for r in renamed[:10]]
            parts.append(f"Indicateurs renommés : {', '.join(renames)}")
        if fn_added:
            texts = [_footnote_text(f) for f in fn_added[:5]]
            parts.append(f"Notes ajoutées : {'; '.join(texts)}")
        if fn_removed:
            texts = [_footnote_text(f) for f in fn_removed[:5]]
            parts.append(f"Notes supprimées : {'; '.join(texts)}")
        if fn_renamed:
            renames = [
                f"'{r.get('previous_text', '')[:80]}' → '{r.get('current_text', '')[:80]}'" for r in fn_renamed[:5]
            ]
            parts.append(f"Notes modifiées : {'; '.join(renames)}")

    elif change_type == "added":
        title = change.get("title") or "(sans titre)"
        indicators = change.get("indicators") or []
        parts.append(f"Section : {section}")
        parts.append(f"NOUVEAU TABLEAU ajouté : {title}")
        if indicators:
            names = [_indicator_label(i) for i in indicators[:15]]
            parts.append(f"Indicateurs : {', '.join(names)}")

    elif change_type == "removed":
        title = change.get("title") or "(sans titre)"
        indicators = change.get("indicators") or []
        parts.append(f"Section : {section}")
        parts.append(f"TABLEAU SUPPRIMÉ : {title}")
        if indicators:
            names = [_indicator_label(i) for i in indicators[:15]]
            parts.append(f"Indicateurs : {', '.join(names)}")

    return "\n".join(parts)


def _indicator_label(ind: Any) -> str:
    """Extrait le libelle textuel d'un indicateur."""
    if isinstance(ind, dict):
        return str(ind.get("value") or ind.get("name") or ind.get("text") or "")[:80]
    return str(ind)[:80]


def _footnote_text(fn: Any) -> str:
    """Extrait le texte d'une note de bas de page."""
    if isinstance(fn, dict):
        return str(fn.get("text") or fn.get("value") or "")[:120]
    return str(fn)[:120]


def _build_summary_prompt(relevant_changes: list[dict[str, Any]]) -> str:
    """Construit le prompt utilisateur pour l'appel LLM de resume global.

    Args:
        relevant_changes: Liste des changements juges pertinents par le triage.

    Returns:
        Texte du prompt formate pour l'appel LLM de synthese.
    """
    parts: list[str] = []
    parts.append(f"Nombre total de changements pertinents : {len(relevant_changes)}\n")
    for i, item in enumerate(relevant_changes[:60], 1):
        cat = item.get("category", "INCONNU")
        expl = item.get("explanation", "")
        title = item.get("_title", "")
        section = item.get("_section", "")
        phase = item.get("project_phase", "autre")
        action = item.get("action_requise", "aucune")
        parts.append(f"{i}. [{cat}] {title} (section: {section}, phase: {phase}, action: {action}) \u2014 {expl}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI async helpers
# ---------------------------------------------------------------------------


async def _call_openai_json_async(
    client: Any,
    *,
    system: str,
    user: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Appel asynchrone unique a OpenAI retournant du JSON parse.

    Args:
        client: Instance ``AsyncOpenAI``.
        system: Contenu du message systeme.
        user: Contenu du message utilisateur.
        model: Identifiant du modele OpenAI.
        temperature: Temperature d'echantillonnage.
        max_tokens: Nombre maximal de tokens de completion. ``None`` laisse le
            modele s'arreter naturellement — preferer la qualite complete.

    Returns:
        Dictionnaire JSON parse ou ``None`` en cas d'echec.
    """
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        return json.loads(raw)
    except Exception as exc:
        logger.warning("GenAI triage call failed: %s", exc)
        return None


def _validate_triage_response(data: dict[str, Any] | None) -> dict[str, Any]:
    """Valide et normalise une reponse LLM de triage individuelle.

    Args:
        data: Dictionnaire brut retourne par le LLM, ou ``None``.

    Returns:
        Dictionnaire valide avec toutes les cles attendues et des valeurs par defaut.
    """
    if not data or not isinstance(data, dict):
        return {
            "is_relevant": False,
            "category": "INCONNU",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "confidence": 0.0,
            "explanation": "",
            "impact_type": "cosmetique",
            "project_phase": "autre",
            "action_requise": "aucune",
            "reference_reglementaire": "",
            "impact_description": "",
            "source": "heuristic",
        }

    is_relevant = bool(data.get("is_relevant", False))

    category = str(data.get("category") or "INCONNU").upper()
    if category not in VALID_CATEGORIES:
        category = "INCONNU"

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

    explanation = str(data.get("explanation") or "")[:1200]

    impact_type = str(data.get("impact_type") or "cosmetique").lower()
    if impact_type not in VALID_IMPACT_TYPES:
        impact_type = "cosmetique"

    project_phase = str(data.get("project_phase") or "autre").lower()
    if project_phase not in VALID_PROJECT_PHASES:
        project_phase = "autre"

    action_requise = str(data.get("action_requise") or "aucune").lower()
    if action_requise not in VALID_ACTIONS:
        action_requise = "aucune"

    reference_reglementaire = str(data.get("reference_reglementaire") or "")[:200]
    impact_description = str(data.get("impact_description") or "")[:500]

    return {
        "is_relevant": is_relevant,
        "category": category,
        "relevance_score": relevance,
        "risk_level": risk_level,
        "confidence": confidence,
        "explanation": explanation,
        "impact_type": impact_type,
        "project_phase": project_phase,
        "action_requise": action_requise,
        "reference_reglementaire": reference_reglementaire,
        "impact_description": impact_description,
        "source": "llm",
    }


def _validate_summary_response(data: dict[str, Any] | None) -> dict[str, Any]:
    """Valide et normalise la reponse LLM du resume global.

    Args:
        data: Dictionnaire brut retourne par le LLM, ou ``None``.

    Returns:
        Dictionnaire valide avec les cles ``executive_overview``,
        ``key_highlights``, ``pertinence_globale``, ``par_phase``, ``par_action``.
    """
    if not data or not isinstance(data, dict):
        return {
            "executive_overview": "",
            "key_highlights": [],
            "pertinence_globale": "FAIBLE",
            "par_phase": {},
            "par_action": {},
            "source": "heuristic",
        }

    overview = str(data.get("executive_overview") or "")[:2000]

    highlights = data.get("key_highlights")
    if not isinstance(highlights, list):
        highlights = []
    highlights = [str(h)[:300] for h in highlights if h][:10]

    overall = str(data.get("pertinence_globale") or "FAIBLE").upper()
    if overall not in VALID_RELEVANCE:
        overall = "FAIBLE"

    # Validate par_phase
    par_phase_raw = data.get("par_phase")
    par_phase: dict[str, Any] = {}
    if isinstance(par_phase_raw, dict):
        for phase in ("rapport_gestion", "pilier_3", "ifc", "autre"):
            entry = par_phase_raw.get(phase)
            if isinstance(entry, dict):
                try:
                    count = int(entry.get("count", 0))
                except (TypeError, ValueError):
                    count = 0
                par_phase[phase] = {
                    "count": count,
                    "resume": str(entry.get("resume") or "")[:300],
                }
            else:
                par_phase[phase] = {"count": 0, "resume": ""}

    # Validate par_action
    par_action_raw = data.get("par_action")
    par_action: dict[str, int] = {}
    if isinstance(par_action_raw, dict):
        for action in (
            "escalade",
            "investigation",
            "confirmation",
            "information",
            "aucune",
        ):
            try:
                par_action[action] = int(par_action_raw.get(action, 0))
            except (TypeError, ValueError):
                par_action[action] = 0

    return {
        "executive_overview": overview,
        "key_highlights": highlights,
        "pertinence_globale": overall,
        "par_phase": par_phase,
        "par_action": par_action,
        "source": "llm",
    }


# ---------------------------------------------------------------------------
# Core triage logic
# ---------------------------------------------------------------------------


def _has_meaningful_diff(pair: dict[str, Any]) -> bool:
    """Retourne ``True`` si une paire de comparaison contient un changement reel a analyser."""
    diff = pair.get("technical_diff") or {}
    status = diff.get("table_level_change", "inchange")
    if status not in ("inchange", "stable"):
        return True
    for key in (
        "indicators_added",
        "indicators_removed",
        "indicators_renamed",
        "footnotes_added",
        "footnotes_removed",
        "footnotes_renamed",
    ):
        if diff.get(key):
            return True
    return False


async def _triage_all_changes(
    comparison: dict[str, Any],
    *,
    model: str = "gpt-4o",
    max_concurrency: int = 20,
) -> dict[str, Any]:
    """Execute le triage LLM sur chaque changement de la comparaison, enrichit en place et retourne le resume."""
    from openai import AsyncOpenAI

    from vigilance.utils.genai import get_openai_api_key

    api_key = get_openai_api_key()
    if not api_key:
        logger.warning("GenAI triage: OPENAI_API_KEY non définie, passage au mode heuristique.")
        return _fallback_enrich(comparison)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    # -- Collect all tasks ------------------------------------------------
    tasks: list[tuple[str, int, str, asyncio.Task[dict[str, Any] | None]]] = []

    pair_comparisons = comparison.get("pair_comparisons") or []
    tables_added = comparison.get("matching", {}).get("tables_added") or []
    tables_removed = comparison.get("matching", {}).get("tables_removed") or []

    for idx, pair in enumerate(pair_comparisons):
        if not _has_meaningful_diff(pair):
            pair["genai_triage"] = {
                "is_relevant": False,
                "category": "COSMETIQUE",
                "relevance_score": "FAIBLE",
                "risk_level": "FAIBLE",
                "confidence": 1.0,
                "explanation": "Aucun changement sémantique détecté.",
                "impact_type": "cosmetique",
                "project_phase": "autre",
                "action_requise": "aucune",
                "reference_reglementaire": "",
                "impact_description": "",
                "source": "skip",
            }
            continue

        prompt = _build_change_prompt(pair, "pair")

        async def _run(p: str = prompt) -> dict[str, Any] | None:
            async with semaphore:
                return await _call_openai_json_async(
                    client,
                    system=_TRIAGE_SYSTEM_PROMPT,
                    user=p,
                    model=model,
                )

        task = asyncio.create_task(_run())
        tasks.append(("pair", idx, prompt, task))

    for idx, tbl in enumerate(tables_added):
        prompt = _build_change_prompt(tbl, "added")

        async def _run_added(p: str = prompt) -> dict[str, Any] | None:
            async with semaphore:
                return await _call_openai_json_async(
                    client,
                    system=_TRIAGE_SYSTEM_PROMPT,
                    user=p,
                    model=model,
                )

        task = asyncio.create_task(_run_added())
        tasks.append(("added", idx, prompt, task))

    for idx, tbl in enumerate(tables_removed):
        prompt = _build_change_prompt(tbl, "removed")

        async def _run_removed(p: str = prompt) -> dict[str, Any] | None:
            async with semaphore:
                return await _call_openai_json_async(
                    client,
                    system=_TRIAGE_SYSTEM_PROMPT,
                    user=p,
                    model=model,
                )

        task = asyncio.create_task(_run_removed())
        tasks.append(("removed", idx, prompt, task))

    # -- Gather results ---------------------------------------------------
    if tasks:
        await asyncio.gather(*(t[3] for t in tasks), return_exceptions=True)

    for kind, idx, _, task in tasks:
        try:
            raw = task.result()
        except Exception:
            raw = None
        validated = _validate_triage_response(raw)

        if kind == "pair":
            pair_comparisons[idx]["genai_triage"] = validated
        elif kind == "added":
            tables_added[idx]["genai_triage"] = validated
        elif kind == "removed":
            tables_removed[idx]["genai_triage"] = validated

    # -- Global summary ---------------------------------------------------
    relevant: list[dict[str, Any]] = []
    for pair in pair_comparisons:
        triage = pair.get("genai_triage") or {}
        if triage.get("is_relevant"):
            entry = dict(triage)
            prev_t = pair.get("previous_table") or {}
            cur_t = pair.get("current_table") or {}
            entry["_title"] = cur_t.get("title") or prev_t.get("title") or ""
            entry["_section"] = cur_t.get("section") or prev_t.get("section") or ""
            relevant.append(entry)
    for tbl in tables_added:
        triage = tbl.get("genai_triage") or {}
        if triage.get("is_relevant"):
            entry = dict(triage)
            entry["_title"] = tbl.get("title") or ""
            entry["_section"] = tbl.get("section") or ""
            relevant.append(entry)
    for tbl in tables_removed:
        triage = tbl.get("genai_triage") or {}
        if triage.get("is_relevant"):
            entry = dict(triage)
            entry["_title"] = tbl.get("title") or ""
            entry["_section"] = tbl.get("section") or ""
            relevant.append(entry)

    global_summary: dict[str, Any]
    if relevant:
        summary_prompt = _build_summary_prompt(relevant)
        summary_raw = await _call_openai_json_async(
            client,
            system=_SUMMARY_SYSTEM_PROMPT,
            user=summary_prompt,
            model=model,
            max_tokens=2000,
        )
        global_summary = _validate_summary_response(summary_raw)
    else:
        global_summary = {
            "executive_overview": "Aucun changement réglementaire ou structurel significatif détecté ce trimestre.",
            "key_highlights": [],
            "pertinence_globale": "FAIBLE",
            "par_phase": {},
            "par_action": {},
            "source": "heuristic",
        }

    global_summary["total_changes_analysed"] = len(pair_comparisons) + len(tables_added) + len(tables_removed)
    global_summary["total_relevant"] = len(relevant)

    comparison["global_summary"] = global_summary
    return comparison


def _fallback_enrich(comparison: dict[str, Any]) -> dict[str, Any]:
    """Enrichissement heuristique de repli lorsqu'aucune cle API n'est disponible.

    Args:
        comparison: Dictionnaire de comparaison a enrichir en place.

    Returns:
        Le meme dictionnaire ``comparison``, enrichi avec des valeurs heuristiques.
    """
    for pair in comparison.get("pair_comparisons") or []:
        if not pair.get("genai_triage"):
            meaningful = _has_meaningful_diff(pair)
            pair["genai_triage"] = {
                "is_relevant": meaningful,
                "category": "INCONNU",
                "relevance_score": "MOYENNE" if meaningful else "FAIBLE",
                "risk_level": "MODERE" if meaningful else "FAIBLE",
                "confidence": 0.0,
                "explanation": "",
                "impact_type": "cosmetique",
                "project_phase": "autre",
                "action_requise": "aucune",
                "reference_reglementaire": "",
                "impact_description": "",
                "source": "heuristic",
            }
    for tbl in comparison.get("matching", {}).get("tables_added") or []:
        if not tbl.get("genai_triage"):
            tbl["genai_triage"] = {
                "is_relevant": True,
                "category": "INCONNU",
                "relevance_score": "MOYENNE",
                "risk_level": "MODERE",
                "confidence": 0.0,
                "explanation": "Nouveau tableau détecté.",
                "impact_type": "structurel",
                "project_phase": "autre",
                "action_requise": "investigation",
                "reference_reglementaire": "",
                "impact_description": "Nouveau tableau ajouté dans le rapport.",
                "source": "heuristic",
            }
    for tbl in comparison.get("matching", {}).get("tables_removed") or []:
        if not tbl.get("genai_triage"):
            tbl["genai_triage"] = {
                "is_relevant": True,
                "category": "INCONNU",
                "relevance_score": "MOYENNE",
                "risk_level": "MODERE",
                "confidence": 0.0,
                "explanation": "Tableau supprimé.",
                "impact_type": "structurel",
                "project_phase": "autre",
                "action_requise": "investigation",
                "reference_reglementaire": "",
                "impact_description": "Tableau supprimé du rapport.",
                "source": "heuristic",
            }
    comparison["global_summary"] = {
        "executive_overview": "",
        "key_highlights": [],
        "pertinence_globale": "FAIBLE",
        "par_phase": {},
        "par_action": {},
        "source": "heuristic",
        "total_changes_analysed": 0,
        "total_relevant": 0,
    }
    return comparison


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_comparison_with_genai_triage(
    comparison_path: str | Path,
    *,
    model: str = "gpt-4o",
    max_concurrency: int = 20,
) -> Path:
    """Lit un comparison.json, l'enrichit avec le triage GenAI et le reecrit.

    Point d'entree principal appele par ``run_pipeline.py``.

    Args:
        comparison_path: Chemin vers le fichier comparison.json sur disque.
        model: Modele OpenAI a utiliser pour les appels de triage.
        max_concurrency: Nombre maximal de requetes LLM en parallele.

    Returns:
        Chemin vers le comparison.json enrichi (meme fichier, reecrit).
    """
    path = Path(comparison_path)
    logger.info("GenAI triage: lecture de %s", path)
    comparison = json.loads(path.read_text(encoding="utf-8"))

    t0 = time.monotonic()
    comparison = asyncio.run(_triage_all_changes(comparison, model=model, max_concurrency=max_concurrency))
    elapsed = time.monotonic() - t0

    summary = comparison.get("global_summary") or {}
    logger.info(
        "GenAI triage terminé en %.1fs — %d changements analysés, %d pertinents",
        elapsed,
        summary.get("total_changes_analysed", 0),
        summary.get("total_relevant", 0),
    )

    path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def inject_llm_resume_metier(comparison: dict) -> dict:
    """Injecte la justification LLM dans genai_analysis['resume_metier'] pour chaque changement/tableau."""
    # Pour chaque changement pair_comparisons
    for pair in comparison.get("pair_comparisons", []):
        triage = pair.get("genai_triage", {})
        if triage.get("explanation"):
            ga = pair.setdefault("genai_analysis", {})
            ga["resume_metier"] = triage["explanation"]

    # Pour chaque tableau ajouté
    for tbl in comparison.get("matching", {}).get("tables_added", []):
        triage = tbl.get("genai_triage", {})
        if triage.get("explanation"):
            ga = tbl.setdefault("genai_analysis", {})
            ga["resume_metier"] = triage["explanation"]

    # Pour chaque tableau supprimé
    for tbl in comparison.get("matching", {}).get("tables_removed", []):
        triage = tbl.get("genai_triage", {})
        if triage.get("explanation"):
            ga = tbl.setdefault("genai_analysis", {})
            ga["resume_metier"] = triage["explanation"]

    return comparison
