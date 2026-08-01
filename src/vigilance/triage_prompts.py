"""Prompts système et utilitaires de génération de prompts pour le triage GenAI.

Ce module centralise la définition du prompt système de triage AMF v2,
des consignes de qualification du niveau d'impact (Few-Shot) et de la
génération des prompts utilisateur.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from vigilance.analyst_change_presentation import bank_subject
from vigilance.amf_taxonomy import (
    format_theme_subjects_for_prompt,
    format_themes_for_prompt,
)


def get_triage_chat_prompt_template() -> ChatPromptTemplate:
    """Retourne l'objet ChatPromptTemplate LangChain pour le triage AMF v2."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=_TRIAGE_SYSTEM_PROMPT),
            ("human", "{change_description}"),
        ]
    )


def _indicator_label(ind: Any) -> str:
    """Extraie un libelle lisible pour un indicateur."""
    if isinstance(ind, dict):
        return str(ind.get("value") or ind.get("name") or ind.get("label") or ind.get("key") or "(sans nom)")[:80]
    return str(ind)[:80]


def _footnote_text(fn: dict[str, Any]) -> str:
    """Extraie le texte d'une note de bas de page."""
    return str(fn.get("text") or fn.get("content") or "(note sans texte)")[:120]


_TRIAGE_SYSTEM_PROMPT = (
    "Tu es un analyste senior en réglementation bancaire canadienne (BSIF / AMF). "
    "On te fournit un changement détecté entre deux trimestres d'un rapport bancaire "
    "(tableau ajouté, supprimé ou modifié avec ses indicateurs et footnotes, ou passage textuel).\n\n"
    "Ton rôle est d'analyser la PERTINENCE RÉGLEMENTAIRE ET PRUDENTIELLE de ce changement "
    "et de fournir une CLASSIFICATION STRUCTURÉE complète et rigoureuse.\n\n"
    "CADRE D'ANALYSE — PORTÉE ET SCOPE PRUDENTIEL :\n"
    "- Le périmètre d'analyse couvre TOUTES les divulgations financières et de risques "
    "du rapport aux actionnaires / rapport de gestion.\n"
    "- Sont PERTINENTS (is_relevant=true) : tout changement modifiant les risques "
    "(crédit, marché, liquidité, opérationnel, cyber, IA, climat), la gouvernance, "
    "les fonds propres (CET1, levier, TLAC), la modélisation ou la conformité réglementaire "
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
    "  4) 'Pertinence métier : ...' avec une explication détaillée, concrète et "
    "rédigée sous la forme d'un paragraphe continu (3 à 4 phrases fluides) : commencer par "
    "'Ce changement met l'accent sur ...' ou 'Ce changement met en évidence ...'. "
    "Expliquer de façon complète l'impact du changement sur le sujet détecté (ex: cybermenaces, "
    "tensions géopolitiques, climat, conformité, capital, liquidité, méthode), sa portée sur la "
    "lecture prudentielle et la transparence de la banque, son incidence sur la résilience face aux "
    "risques externes ou chocs, ainsi que la comparabilité de la gestion des risques de la banque avec ses pairs.\n"
    "  5) 'Point de surveillance : ...' avec le point de surveillance à retenir, "
    "sans demander à l'analyste de vérifier, accepter ou rejeter le changement.\n"
    "- Au moins 3 phrases complètes (≥ 20 caractères chacune, ponctuation "
    "finale) ET ≥ 200 caractères au total — l'analyste doit avoir une "
    "explication détaillée et claire, pas un résumé.\n"
    "- Citer l'ÉLÉMENT SPÉCISIFIQUE du rapport : nom exact d'un indicateur, "
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
    "- ELEVE : impact direct sur les ratios prudentiels, réformes BSIF, seuils réglementaires, conformité, climat/ESG, cyber/IA ou réputation.\n"
    "- MODERE : tout changement touchant la gouvernance, la rémunération incitative, le rôle des comités, la périodicité de reporting, ou la culture du risque (quelle que soit la longueur du texte modifié).\n"
    "- FAIBLE : uniquement les corrections typographiques, la ponctuation, ou les renommages purement cosmétiques sans aucun impact de gouvernance ni de risque.\n\n"
    "EXEMPLES FEW-SHOT DE QUALIFICATION D'IMPACT :\n"
    "1. Exemple Rémunération : 'Ajout de et à encourager les comportements attendus dans la rémunération incitative' -> nouvelle_idee: true, risk_level: 'MODERE' (impact métier sur la culture de risque et gouvernance).\n"
    "2. Exemple Reporting : 'Précision sur la présentation du rapport sur une base trimestrielle' -> nouvelle_idee: true, risk_level: 'MODERE' (modification de fréquence de suivi prudentiel).\n"
    "3. Exemple BSIF : 'Ajout de la ligne directrice E-23 BSIF relative à la gestion du risque de modélisation' -> nouvelle_idee: true, risk_level: 'ELEVE'.\n"
    "4. Exemple Cosmétique : 'Remplacement de la virgule par un point-virgule' -> nouvelle_idee: false, risk_level: 'FAIBLE'.\n\n"
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


def _build_change_prompt(
    change: dict[str, Any],
    change_type: str,
    *,
    bank_code: str = "",
) -> str:
    """Construit un prompt utilisateur decrivant un changement detecte pour le LLM."""
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
            parts.append(f"Indicateurs qu'il contenait : {', '.join(names)}")

    return "\n".join(parts)


def _build_summary_user_prompt(relevant_changes: list[dict[str, Any]]) -> str:
    """Construit le prompt utilisateur pour la synthese executive globale."""
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
