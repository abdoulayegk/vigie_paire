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

from vigilance.analyst_change_presentation import bank_subject
from vigilance.amf_taxonomy import (
    IMPACT_IT_DETAIL_LABELS,
    POSTURE_DETAIL_LABELS,
    THEMES_AMF_PIPELINE_2,
    format_theme_subjects_for_prompt,
    format_themes_for_prompt,
    missing_labeled_analysis_sections,
)

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
        "NON_PERTINENT",
        "INCONNU",
    }
)

VALID_RELEVANCE = frozenset({"ELEVEE", "MOYENNE", "FAIBLE"})

VALID_RISK_LEVELS = frozenset({"ELEVE", "MODERE", "FAIBLE"})

VALID_IMPACT_TYPES = frozenset({"structurel", "contenu", "methodologique", "non_substantif"})

VALID_PROJECT_PHASES = frozenset({"rapport_gestion", "pilier_3", "ifc", "autre"})

VALID_ACTIONS = frozenset({"revue_prioritaire", "investigation", "confirmation", "information", "aucune"})

VALID_IMPACT_IT = frozenset({"ELEVE", "MOYEN", "FAIBLE", "INDETERMINE"})

VALID_CHANGEMENTS_POSTURE = frozenset(
    {
        "RENFORCEMENT",
        "ALLEGEMENT",
        "NOUVEAU_DISPOSITIF",
        "RETRAIT_DISPOSITIF",
        "AUCUN",
        "INDETERMINE",
    }
)

VALID_STATUTS_MISE_EN_OEUVRE = frozenset(
    {"ANNONCE", "PLANIFIE", "EN_COURS", "MIS_EN_OEUVRE", "INDETERMINE"}
)

VALID_CONFIANCES_POSTURE = frozenset(
    {"ELEVEE", "MOYENNE", "FAIBLE", "INDETERMINE"}
)

# Réutilise la taxonomie AMF unifiée définie dans amf_taxonomy.py
# (mêmes codes que Pipeline 2, partagés pour permettre des filtres transverses).
VALID_THEMES_AMF = frozenset(THEMES_AMF_PIPELINE_2)

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
# System Prompts
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM_PROMPT = (
    "Tu es un analyste senior en gestion intégrée des risques, spécialisé "
    "dans la vigie de pairs des banques canadiennes alignée sur les attentes "
    "de l'AMF (Autorité des marchés financiers du Québec) et du BSIF.\n\n"
    "On te soumet un changement détecté dans un TABLEAU ou une FOOTNOTE de "
    "tableau entre deux rapports d'une même banque comparés pair-à-pair :\n"
    "- T1 = rapport PRÉCÉDENT dans la paire\n"
    "- T2 = rapport COURANT dans la paire\n\n"
    "Ton rôle : qualifier ce changement contre la taxonomie AMF unifiée, "
    "trancher si c'est une NOUVELLE IDÉE, et produire une justification "
    "ancrée sur le contenu réel du rapport.\n\n"
    "RÈGLES STRICTES D'EXCLUSION — mettre is_relevant=false :\n"
    "- Variations chiffrées propres à la banque (valeur d'une cellule, montant "
    "d'actif, exposition chiffrée propre) — PAS un seuil réglementaire.\n"
    "- Reformulation d'un libellé d'indicateur ou de footnote sans nouveau fond "
    "(synonyme, ordre des mots).\n"
    "- Déplacement de texte sans modification de contenu.\n"
    "- Formatage visuel pur (gras, italique, ponctuation, casse, espacement).\n"
    "- Mises à jour de dates pures (ex : « janvier » → « avril »).\n\n"
    "RÈGLES STRICTES D'INCLUSION — mettre is_relevant=true :\n"
    "- Ajout / retrait d'un tableau entier, d'une ligne (indicateur), ou d'une "
    "footnote substantive.\n"
    "- Renommage d'un indicateur signalant un changement méthodologique "
    "(ex : « Tier 1 » → « Tier 1 ex-AT1 »).\n"
    "- Footnote nouvelle ou modifiée citant une nouvelle ligne directrice "
    "(BSIF, BCBS, Bâle, AMF).\n"
    "- Montants RÉGLEMENTAIRES (seuils prudentiels, planchers Bâle, "
    "exigences pilier 2) — INCLUS dans le scope, marqués via "
    "MONTANT_REGLEMENTAIRE.\n"
    "- Indicateurs de risques émergents (cyber, IA, IA générative, fraude "
    "numérique) — PRIORITAIRES, impact_level minimum MODERE.\n\n"
    "COUVERTURE DONNÉES / TIERS / CLOUD :\n"
    "- RISQUE_DONNEES couvre la gouvernance, la qualité, l'intégrité, la "
    "protection, la localisation, la traçabilité et le cycle de vie des données.\n"
    "- RISQUE_TIERS_CLOUD couvre les fournisseurs critiques, l'impartition, "
    "l'infonuagique, la concentration, la résilience et les stratégies de sortie.\n"
    "- Une simple occurrence des mots données, tiers ou fournisseur ne suffit "
    "pas : le changement doit modifier la substance de la divulgation.\n\n"
    "CHANGEMENT DE POSTURE : déterminer si la banque renforce ou allège ses "
    "contrôles, crée ou retire un comité, un cadre, une responsabilité, une "
    "diligence, une exigence contractuelle ou une stratégie de sortie. Une "
    "simple mention de risque n'est pas un changement de posture.\n\n"
    "TAXONOMIE AMF (utilise UNIQUEMENT ces codes dans themes_amf, multi-label "
    "autorisé et encouragé) :\n"
    f"{format_themes_for_prompt()}\n\n"
    "LIBELLÉS ANALYSTE À UTILISER dans `Sujet détecté` et la justification "
    "(ne pas laisser seulement les codes AMF techniques) :\n"
    f"{format_theme_subjects_for_prompt()}\n\n"
    "DÉFINITION DE `nouvelle_idee` — 3 conditions cumulatives :\n"
    "(a) SUBSTANTIELLE : modifie la SUBSTANCE de la divulgation (concept, "
    "indicateur, mention réglementaire, méthodologie) — PAS une variation "
    "chiffrée propre à la banque ni une reformulation.\n"
    "(b) NOUVEAUTÉ INFORMATIONNELLE : ajoute (présent T2 absent T1), retire "
    "(présent T1 absent T2), OU modifie substantiellement la posture sur "
    "un thème AMF.\n"
    "(c) ADOSSÉE À UN THÈME AMF : au moins un code dans themes_amf.\n"
    "Si UNE des 3 conditions est violée → nouvelle_idee=false.\n\n"
    "FORMAT STRICT pour `nouvelle_idee_justification` (TOUJOURS obligatoire, "
    "y compris pour les changements jugés non pertinents) :\n"
    "- Commencer par 'OUI' (si nouvelle_idee=true) ou 'NON' (sinon), suivi "
    "d'un tiret ou d'une virgule.\n"
    "- Rédiger une NOTE D'ANALYSTE structurée avec ces rubriques EXACTES, "
    "dans cet ordre, séparées par \\n\\n :\n"
    "  1) 'OUI — Nouvel élément à surveiller : Oui' ou "
    "'NON — Nouvel élément à surveiller : Non'.\n"
    "  2) 'Sujet détecté : ...' avec des mots simples tirés des libellés "
    "analyste ci-dessus (ex : IA, cybersécurité, risque climatique, "
    "conformité, capital, liquidité, méthode de calcul).\n"
    "  3) 'Ce qui change : ...' avec l'élément exact ajouté, retiré ou "
    "modifié entre T1 et T2. Le prompt utilisateur fournit la banque analysée; "
    "commencer cette rubrique par son nom court suivi d'un verbe d'action direct. "
    "T1, T2, rapport courant et rapport précédent restent du contexte technique "
    "et ne doivent jamais être le sujet de cette phrase.\n"
    "  4) 'Pertinence métier : ...' avec une explication longue, concrète et "
    "formulée comme un analyste de vigie : commencer idéalement par "
    "'Ce changement met l'accent sur ...' ou 'Ce changement met en évidence ...'. "
    "Relier le changement au sujet détecté (IA, cyber, climat, conformité, "
    "capital, méthode, divulgation), aux attentes prudentielles, à la conformité, "
    "aux contrôles, à la comparabilité entre pairs et à son importance pour une "
    "banque.\n"
    "  5) 'Point de surveillance : ...' avec le point de surveillance à retenir, "
    "sans demander à l'analyste de vérifier, accepter ou rejeter le changement.\n"
    "- Au moins 3 phrases complètes (≥ 20 caractères chacune, ponctuation "
    "finale) ET ≥ 200 caractères au total — l'analyste doit avoir une "
    "explication détaillée et claire, pas un résumé.\n"
    "- Citer l'ÉLÉMENT SPÉCIFIQUE du rapport : nom exact d'un indicateur, "
    "titre du tableau, libellé de footnote — adossé au contenu réel des "
    "rapports aux actionnaires traités.\n"
    "- Si is_relevant=true : mentionner les thèmes AMF concernés en langage "
    "naturel dans 'Sujet détecté' et expliquer pourquoi c'est une nouveauté "
    "métier pour la banque.\n"
    "- Si is_relevant=false : expliquer en LANGAGE MÉTIER pourquoi ce "
    "changement n'est PAS une nouvelle idée AMF (variation chiffrée propre, "
    "reformulation sans nouveau fond, formatage, déplacement). L'analyste "
    "doit comprendre la raison de l'exclusion sans avoir à interpréter le "
    "code d'exclusion.\n\n"
    "Ne jamais remplacer l'analyse par une simple liste de codes AMF ou par "
    "une phrase générique du type 'ce changement affecte les thèmes AMF'. Les "
    "codes peuvent apparaître, mais la justification doit expliquer le "
    "raisonnement métier. Ne pas utiliser de formules de tâche comme "
    "'vérifier si', 'accepter', 'rejeter', 'à confirmer par l'analyste' dans "
    "la justification : Dash affiche déjà la preuve et l'analyste prend la "
    "décision finale.\n\n"
    "RÉPONDRE UNIQUEMENT en JSON valide, sans markdown, selon ce schéma exact :\n"
    "{\n"
    '  "is_relevant": true | false,\n'
    '  "themes_amf": ["<code AMF>", ...],   // multi-label, vide si is_relevant=false\n'
    '  "nouvelle_idee": true | false,\n'
    '  "nouvelle_idee_justification": "<OUI/NON — rubriques obligatoires : Nouvel élément à surveiller, Sujet détecté, Ce qui change, Pertinence métier, Point de surveillance>",\n'
    '  "category": "REGLEMENTAIRE" | "RISQUE" | "CAPITAL" | "STRUCTURE" | "NON_PERTINENT" | "INCONNU",\n'
    '  "relevance_score": "ELEVEE" | "MOYENNE" | "FAIBLE",\n'
    '  "risk_level": "ELEVE" | "MODERE" | "FAIBLE",\n'
    '  "impact_it": "ELEVE" | "MOYEN" | "FAIBLE" | "INDETERMINE",\n'
    '  "impact_it_justification": "<rubriques exactes : Éléments observés, Conséquence probable, Limite de l\'analyse; vide si INDETERMINE>",\n'
    '  "changement_posture": "RENFORCEMENT" | "ALLEGEMENT" | "NOUVEAU_DISPOSITIF" | "RETRAIT_DISPOSITIF" | "AUCUN" | "INDETERMINE",\n'
    '  "justification_posture": "<rubriques exactes : Preuve, Effet sur la gestion du risque, Justification du statut, Justification de la confiance; vide si AUCUN ou INDETERMINE>",\n'
    '  "statut_mise_en_oeuvre": "ANNONCE" | "PLANIFIE" | "EN_COURS" | "MIS_EN_OEUVRE" | "INDETERMINE",\n'
    '  "confiance_posture": "ELEVEE" | "MOYENNE" | "FAIBLE" | "INDETERMINE",\n'
    '  "confidence": 0.0 à 1.0,\n'
    '  "explanation": "<3 paragraphes français séparés par \\n\\n>",\n'
    '  "impact_type": "structurel" | "contenu" | "methodologique" | "non_substantif",\n'
    '  "project_phase": "rapport_gestion" | "pilier_3" | "ifc" | "autre",\n'
    '  "action_requise": "revue_prioritaire" | "investigation" | "confirmation" | "information" | "aucune",\n'
    '  "reference_reglementaire": "<référence si applicable, ex: Bâle III — CET1, sinon chaîne vide>",\n'
    '  "impact_description": "<1 phrase décrivant l\'impact concret>"\n'
    "}\n\n"
    "GUIDE pour `risk_level` :\n"
    "- ELEVE : impact direct sur les ratios prudentiels, seuils réglementaires, ou conformité.\n"
    "- MODERE : changement méthodologique ou structurel à surveiller.\n"
    "- FAIBLE : changement modeste ou non substantiel.\n\n"
    "GUIDE pour `changement_posture` :\n"
    "- RENFORCEMENT : contrôles, surveillance, diligence ou exigences renforcés.\n"
    "- ALLEGEMENT : encadrement ou niveau de contrôle réduit.\n"
    "- NOUVEAU_DISPOSITIF : nouveau comité, cadre, responsabilité, stratégie de sortie ou contrôle.\n"
    "- RETRAIT_DISPOSITIF : suppression d'un dispositif de gestion existant.\n"
    "- AUCUN : aucune évolution de la façon de gérer le risque.\n"
    "- INDETERMINE : le texte ne permet pas de conclure.\n"
    "- justification_posture utilise exactement quatre rubriques séparées par "
    "\\n\\n : Preuve, Effet sur la gestion du risque, Justification du statut, "
    "Justification de la confiance. Elle est vide pour AUCUN ou INDETERMINE.\n"
    "- statut_mise_en_oeuvre distingue une ANNONCE, une mesure PLANIFIEE, "
    "EN_COURS ou MIS_EN_OEUVRE. Utilise INDETERMINE sans preuve temporelle.\n"
    "- confiance_posture évalue uniquement la solidité de cette classification.\n\n"
    "GUIDE pour `impact_it`, distinct de l'impact métier :\n"
    "- ELEVE : architecture, migration, fournisseur remplacé, localisation des données ou contrôles majeurs.\n"
    "- MOYEN : nouveaux processus, inventaires, surveillance, rapports ou exigences contractuelles.\n"
    "- FAIBLE : clarification ou ajustement limité avec un effet IT identifiable, sans transformation technologique apparente.\n"
    "- INDETERMINE : information insuffisante ou aucun lien IT crédible démontré. Ne jamais présenter une intention comme une mise en œuvre réalisée; FAIBLE ne signifie pas absence d'impact IT.\n"
    "- impact_it_justification utilise exactement trois rubriques séparées par "
    "\\n\\n : Éléments observés, Conséquence probable, Limite de l'analyse.\n\n"
    "GUIDE pour `impact_type` :\n"
    "- structurel : ajout/suppression de lignes, colonnes, tableaux entiers.\n"
    "- contenu : modification de valeurs, seuils, descriptions réglementaires.\n"
    "- methodologique : changement de méthode de calcul, de périmètre, de définition.\n"
    "- non_substantif : renommage, reformulation, mise en forme sans impact de fond.\n\n"
    "GUIDE pour `project_phase` :\n"
    "- rapport_gestion : sections gestion du capital, gestion des risques, texte risque.\n"
    "- pilier_3 : tableaux de divulgation Pilier 3 (BSIF).\n"
    "- ifc : rapports intermédiaires/condensés.\n"
    "- autre : si la section ne correspond à aucune phase ci-dessus.\n\n"
    "GUIDE pour `action_requise` :\n"
    "- revue_prioritaire : changement critique MAJEUR nécessitant une revue immédiate par un senior.\n"
    "- investigation : anomalie ou surprise nécessitant une analyse approfondie.\n"
    "- confirmation : changement attendu à confirmer comme normal.\n"
    "- information : changement mineur, pour information seulement.\n"
    "- aucune : non pertinent, aucune action.\n\n"
    "INVARIANTS STRICTS (toute violation rejette la réponse) :\n"
    "- nouvelle_idee_justification est TOUJOURS OBLIGATOIRE (≥ 3 phrases, "
    "≥ 200 chars), commençant par 'OUI' ou 'NON' selon nouvelle_idee, "
    "et contenant les rubriques exactes : Nouvel élément à surveiller, "
    "Sujet détecté, Ce qui change, Pertinence métier, Point de surveillance.\n"
    "- is_relevant=true → themes_amf NON VIDE.\n"
    "- is_relevant=false → themes_amf=[], category='NON_PERTINENT', "
    "nouvelle_idee=false, action_requise='aucune'. La justification reste "
    "obligatoire pour expliquer à l'analyste pourquoi le changement n'est "
    "pas une nouvelle idée AMF.\n"
)

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
    "revue_prioritaire": N,
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


def _build_change_prompt(
    change: dict[str, Any],
    change_type: str,
    *,
    bank_code: str = "",
) -> str:
    """Construit un prompt utilisateur decrivant un changement detecte pour le LLM.

    Args:
        change: Dictionnaire du changement (paire, tableau ajoute ou supprime).
        change_type: Type de changement : ``"pair"``, ``"added"`` ou ``"removed"``.
        bank_code: Code court de la banque analysée.

    Returns:
        Texte du prompt formate pour l'appel LLM de triage.
    """
    subject = bank_subject(bank_code)
    parts: list[str] = [
        f"Banque analysée : {subject}",
        (
            "Règle de rédaction : dans « Ce qui change », commencer exactement "
            f"par « {subject} » suivi d'un verbe d'action direct."
        ),
    ]
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

    Applique la taxonomie AMF unifiée (themes_amf multi-label) et les
    invariants nouvelle_idee + nouvelle_idee_justification définis avec
    Pipeline 2. En cas de violation des invariants stricts, le triage est
    forcé en non pertinent (squelette neutre) et l'incident est journalisé.

    Args:
        data: Dictionnaire brut retourne par le LLM, ou ``None``.

    Returns:
        Dictionnaire valide avec toutes les cles attendues et des valeurs par defaut.
    """
    if not data or not isinstance(data, dict):
        return _empty_triage_skeleton(source="heuristic")

    is_relevant = bool(data.get("is_relevant", False))

    raw_themes = data.get("themes_amf") or []
    themes_amf: list[str] = []
    if isinstance(raw_themes, list):
        seen: set[str] = set()
        for theme in raw_themes:
            code = str(theme or "").upper()
            if code in VALID_THEMES_AMF and code not in seen:
                seen.add(code)
                themes_amf.append(code)

    nouvelle_idee = bool(data.get("nouvelle_idee", False))
    nouvelle_idee_justification = str(data.get("nouvelle_idee_justification") or "").strip()

    category = str(data.get("category") or "INCONNU").upper()
    if category not in VALID_CATEGORIES:
        category = "INCONNU"

    relevance = str(data.get("relevance_score") or "FAIBLE").upper()
    if relevance not in VALID_RELEVANCE:
        relevance = "FAIBLE"

    risk_level = str(data.get("risk_level") or "FAIBLE").upper()
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "FAIBLE"

    # Dérivation du champ AMF v2 ``impact_level`` (MAJEUR/MODERE/MINEUR)
    # à partir du ``risk_level`` legacy (ELEVE/MODERE/FAIBLE) — assure la
    # cohérence avec Pipeline 2 et l'affichage UI Dash.
    _RISK_TO_IMPACT = {"ELEVE": "MAJEUR", "MODERE": "MODERE", "FAIBLE": "MINEUR"}
    impact_level = _RISK_TO_IMPACT.get(risk_level, "MINEUR")

    impact_it = str(data.get("impact_it") or "INDETERMINE").upper()
    if impact_it not in VALID_IMPACT_IT:
        impact_it = "INDETERMINE"
    impact_it_justification = str(
        data.get("impact_it_justification") or ""
    ).strip()[:500]
    if impact_it == "INDETERMINE":
        impact_it_justification = ""
    elif (
        len(impact_it_justification) < 20
        or missing_labeled_analysis_sections(
            impact_it_justification,
            IMPACT_IT_DETAIL_LABELS,
        )
    ):
        impact_it = "INDETERMINE"
        impact_it_justification = ""

    changement_posture = str(
        data.get("changement_posture") or "INDETERMINE"
    ).upper()
    if changement_posture not in VALID_CHANGEMENTS_POSTURE:
        changement_posture = "INDETERMINE"

    justification_posture = str(
        data.get("justification_posture") or ""
    ).strip()[:500]
    statut_mise_en_oeuvre = str(
        data.get("statut_mise_en_oeuvre") or "INDETERMINE"
    ).upper()
    if statut_mise_en_oeuvre not in VALID_STATUTS_MISE_EN_OEUVRE:
        statut_mise_en_oeuvre = "INDETERMINE"
    confiance_posture = str(
        data.get("confiance_posture") or "INDETERMINE"
    ).upper()
    if confiance_posture not in VALID_CONFIANCES_POSTURE:
        confiance_posture = "INDETERMINE"

    posture_evaluee = changement_posture in {
        "RENFORCEMENT",
        "ALLEGEMENT",
        "NOUVEAU_DISPOSITIF",
        "RETRAIT_DISPOSITIF",
    }
    if not posture_evaluee:
        justification_posture = ""
        statut_mise_en_oeuvre = "INDETERMINE"
        confiance_posture = "INDETERMINE"
    elif (
        len(justification_posture) < 20
        or confiance_posture == "INDETERMINE"
        or missing_labeled_analysis_sections(
            justification_posture,
            POSTURE_DETAIL_LABELS,
        )
    ):
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
