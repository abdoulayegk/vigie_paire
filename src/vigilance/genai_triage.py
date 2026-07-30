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
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vigilance.analyst_change_presentation import bank_subject
from vigilance.amf_taxonomy import (
    IMPACT_IT_DETAIL_LABELS,
    POSTURE_DETAIL_LABELS,
    format_theme_subjects_for_prompt,
    format_themes_for_prompt,
    missing_labeled_analysis_sections,
)
from vigilance.triage_prompts import (
    _SUMMARY_SYSTEM_PROMPT,
    _TRIAGE_SYSTEM_PROMPT,
    _build_change_prompt,
    _build_summary_user_prompt,
)
from vigilance.triage_validation import (
    VALID_ACTIONS,
    VALID_CATEGORIES,
    VALID_CHANGEMENTS_POSTURE,
    VALID_CONFIANCES_POSTURE,
    VALID_IMPACT_IT,
    VALID_IMPACT_TYPES,
    VALID_PROJECT_PHASES,
    VALID_RELEVANCE,
    VALID_RISK_LEVELS,
    VALID_STATUTS_MISE_EN_OEUVRE,
    VALID_THEMES_AMF,
    _empty_triage_skeleton,
    _validate_triage_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Format strict pour la justification GPT : commence par OUI ou NON suivi
# d'un séparateur (tiret ou virgule), au moins 3 phrases substantives.
_JUSTIFICATION_MIN_SENTENCES = 3
_JUSTIFICATION_MIN_SENTENCE_LENGTH = 20
_JUSTIFICATION_MIN_TOTAL_LENGTH = 200
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+")
_REQUIRED_JUSTIFICATION_SECTIONS = (
    "Nouvel élément à surveiller :",
    "Sujet détecté :",
    "Ce qui change :",
    "Pertinence métier :",
    "Point de surveillance :",
)
_LEGACY_SURVEILLANCE_SECTION = "Lecture de vigie :"


def _count_substantive_sentences(text: str) -> int:
    """Compte les phrases substantives dans ``text`` (≥ 15 caractères chacune)."""
    if not text:
        return 0
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return sum(
        1 for part in parts if len(part.strip()) >= _JUSTIFICATION_MIN_SENTENCE_LENGTH
    )


def _missing_justification_sections(text: str) -> list[str]:
    """Retourne les rubriques obligatoires absentes de la justification GPT."""
    missing: list[str] = []
    for section in _REQUIRED_JUSTIFICATION_SECTIONS:
        if section in text:
            continue
        if section == "Point de surveillance :" and _LEGACY_SURVEILLANCE_SECTION in text:
            continue
        missing.append(section)
    return missing


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


        changement_posture = "INDETERMINE"
        justification_posture = ""
        statut_mise_en_oeuvre = "INDETERMINE"
        confiance_posture = "INDETERMINE"

    if not is_relevant:
        impact_it = "INDETERMINE"
        impact_it_justification = ""
        changement_posture = "AUCUN"
        justification_posture = ""
        statut_mise_en_oeuvre = "INDETERMINE"
        confiance_posture = "INDETERMINE"

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    explanation = str(data.get("explanation") or "")[:1200]

    impact_type = str(data.get("impact_type") or "non_substantif").lower()
    if impact_type not in VALID_IMPACT_TYPES:
        impact_type = "non_substantif"

    project_phase = str(data.get("project_phase") or "autre").lower()
    if project_phase not in VALID_PROJECT_PHASES:
        project_phase = "autre"

    action_requise = str(data.get("action_requise") or "aucune").lower()
    if action_requise not in VALID_ACTIONS:
        action_requise = "aucune"

    reference_reglementaire = str(data.get("reference_reglementaire") or "")[:200]
    impact_description = str(data.get("impact_description") or "")[:500]

    invariant_error = _validate_amf_invariants(
        is_relevant=is_relevant,
        themes_amf=themes_amf,
        category=category,
        nouvelle_idee=nouvelle_idee,
        nouvelle_idee_justification=nouvelle_idee_justification,
        action_requise=action_requise,
    )
    if invariant_error:
        logger.warning(
            "Invariants AMF violés dans la sortie LLM (%s) — triage forcé en NON_PERTINENT",
            invariant_error,
        )
        return _empty_triage_skeleton(source="invariant_violation")

    return {
        "is_relevant": is_relevant,
        "themes_amf": themes_amf,
        "nouvelle_idee": nouvelle_idee,
        "nouvelle_idee_justification": nouvelle_idee_justification,
        "impact_level": impact_level,
        "impact_it": impact_it,
        "impact_it_justification": impact_it_justification,
        "changement_posture": changement_posture,
        "justification_posture": justification_posture,
        "statut_mise_en_oeuvre": statut_mise_en_oeuvre,
        "confiance_posture": confiance_posture,
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


def _empty_triage_skeleton(*, source: str = "heuristic") -> dict[str, Any]:
    """Squelette de triage non pertinent (aligné Pipeline 2 - empty_triage_skeleton).

    Utilisé quand GPT n'a pas répondu, quand la réponse est inutilisable, ou
    quand les invariants AMF sont violés. Volontairement neutre pour ne pas
    polluer le rapport avec des classifications fictives.
    """
    return {
        "is_relevant": False,
        "themes_amf": [],
        "nouvelle_idee": False,
        "nouvelle_idee_justification": (
            "NON — Nouvel élément à surveiller : Non.\n\n"
            "Sujet détecté : Élément non classifié par l'analyse automatisée.\n\n"
            "Ce qui change : Aucun triage AMF exploitable n'a été produit par "
            "GPT-4o pour ce changement. Le système ne dispose donc pas d'une "
            "lecture fiable du contenu T1/T2 pour qualifier cette ligne.\n\n"
            "Pertinence métier : Ce cas ne constitue pas une nouvelle idée "
            "métier détectée par la vigie, car aucun thème AMF, risque, "
            "méthode, conformité ou divulgation substantielle n'a pu être "
            "rattaché au changement de façon fiable.\n\n"
            "Point de surveillance : Élément non classifié — La ligne ne porte "
            "pas de signal métier exploitable dans le résumé de surveillance "
            "automatisé."
        ),
        "impact_level": "MINEUR",
        "impact_it": "INDETERMINE",
        "impact_it_justification": "",
        "changement_posture": "AUCUN",
        "justification_posture": "",
        "statut_mise_en_oeuvre": "INDETERMINE",
        "confiance_posture": "INDETERMINE",
        "category": "NON_PERTINENT",
        "relevance_score": "FAIBLE",
        "risk_level": "FAIBLE",
        "confidence": 0.0,
        "explanation": "",
        "impact_type": "non_substantif",
        "project_phase": "autre",
        "action_requise": "aucune",
        "reference_reglementaire": "",
        "impact_description": "",
        "source": source,
    }


def _validate_amf_invariants(
    *,
    is_relevant: bool,
    themes_amf: list[str],
    category: str,
    nouvelle_idee: bool,
    nouvelle_idee_justification: str,
    action_requise: str,
) -> str | None:
    """Vérifie les invariants AMF transversaux (mêmes règles que Pipeline 2).

    La ``nouvelle_idee_justification`` est OBLIGATOIRE et SUBSTANTIELLE quel
    que soit ``is_relevant`` — l'analyste doit toujours comprendre la décision
    GPT (≥ 3 phrases complètes, ≥ 200 caractères, préfixée par OUI ou NON).

    Returns:
        ``None`` si tous les invariants sont satisfaits, sinon un message
        décrivant la première violation détectée.
    """
    justification = nouvelle_idee_justification.strip()
    if _count_substantive_sentences(justification) < _JUSTIFICATION_MIN_SENTENCES:
        return (
            f"nouvelle_idee_justification exige ≥ {_JUSTIFICATION_MIN_SENTENCES} "
            f"phrases complètes (≥ {_JUSTIFICATION_MIN_SENTENCE_LENGTH} chars chacune)"
        )
    if len(justification) < _JUSTIFICATION_MIN_TOTAL_LENGTH:
        return (
            f"nouvelle_idee_justification exige ≥ {_JUSTIFICATION_MIN_TOTAL_LENGTH} "
            "caractères au total"
        )
    expected_prefix = "OUI" if nouvelle_idee else "NON"
    if not justification.upper().startswith(expected_prefix):
        return (
            f"nouvelle_idee_justification doit commencer par '{expected_prefix}' "
            f"quand nouvelle_idee={nouvelle_idee}"
        )
    missing_sections = _missing_justification_sections(justification)
    if missing_sections:
        return (
            "nouvelle_idee_justification doit contenir les rubriques "
            f"obligatoires : {', '.join(missing_sections)}"
        )

    if is_relevant:
        if not themes_amf:
            return "is_relevant=True exige themes_amf non vide"
    else:
        if themes_amf:
            return "is_relevant=False interdit themes_amf non vide"
        if nouvelle_idee:
            return "is_relevant=False interdit nouvelle_idee=True"
        if action_requise != "aucune":
            return "is_relevant=False exige action_requise='aucune'"
    return None


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
            "revue_prioritaire",
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
    bank_code = str(comparison.get("bank_code") or "")

    # -- Collect all tasks ------------------------------------------------
    tasks: list[tuple[str, int, str, asyncio.Task[dict[str, Any] | None]]] = []

    pair_comparisons = comparison.get("pair_comparisons") or []
    tables_added = comparison.get("matching", {}).get("tables_added") or []
    tables_removed = comparison.get("matching", {}).get("tables_removed") or []

    for idx, pair in enumerate(pair_comparisons):
        if not _has_meaningful_diff(pair):
            pair["genai_triage"] = _empty_triage_skeleton(source="skip")
            continue

        prompt = _build_change_prompt(pair, "pair", bank_code=bank_code)

        async def _run(p: str = prompt) -> dict[str, Any] | None:
            """Tâche async qui appelle le triage GPT pour un changement de paire."""
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
        prompt = _build_change_prompt(tbl, "added", bank_code=bank_code)

        async def _run_added(p: str = prompt) -> dict[str, Any] | None:
            """Tâche async qui appelle le triage GPT pour un tableau ajouté."""
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
        prompt = _build_change_prompt(tbl, "removed", bank_code=bank_code)

        async def _run_removed(p: str = prompt) -> dict[str, Any] | None:
            """Tâche async qui appelle le triage GPT pour un tableau retiré."""
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
        pbar = tqdm(
            total=len(tasks),
            desc="Triage GenAI Tableaux",
            unit="tableau",
        )
        for fut in asyncio.as_completed([t[3] for t in tasks]):
            try:
                await fut
            except Exception:
                pass
            pbar.update(1)
        pbar.close()

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
        summary_prompt = _build_summary_user_prompt(relevant)
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
            pair["genai_triage"] = _empty_triage_skeleton(source="heuristic")
    for tbl in comparison.get("matching", {}).get("tables_added") or []:
        if not tbl.get("genai_triage"):
            tbl["genai_triage"] = _empty_triage_skeleton(source="heuristic")
    for tbl in comparison.get("matching", {}).get("tables_removed") or []:
        if not tbl.get("genai_triage"):
            tbl["genai_triage"] = _empty_triage_skeleton(source="heuristic")
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
