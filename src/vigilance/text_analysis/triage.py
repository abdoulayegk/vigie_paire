"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Any

from pydantic import ValidationError

from vigilance.amf_taxonomy import (
    TRIAGE_SOURCE_VERSION,
    TriageAMFLLMBatch,
    TriageValidationError,
    empty_triage_skeleton,
    format_exclusion_reasons_for_prompt,
    format_theme_subjects_for_prompt,
    format_themes_for_prompt,
)
from vigilance.text_analysis.constants import (
    _TRIAGE_BATCH_SIZE,
    _TRIAGE_SEMANTIC_TEXT_LIMIT,
    _TRIAGE_SOURCE_SNIPPET_LIMIT,
)
from vigilance.text_analysis.normalization import _json_dumps, _sanitize_explanation
from vigilance.text_analysis.openai_client import _call_structured_completion_with_correction, _truncate_prompt_text
from vigilance.text_comparison.change_segments import build_change_segments

logger = logging.getLogger(__name__)

_MAX_TRIAGE_LLM_WORKERS = 6


def _default_triage() -> dict[str, Any]:
    """Retourne un triage par défaut conservateur (non pertinent).

    Schéma cible AMF v2 (``themes_amf``, ``exclusion_reason``) **plus** les
    champs hérités (``category``, ``signals``, ``confidence``, ...) maintenus
    avec valeurs par défaut pour préserver la compatibilité avec les
    consommateurs aval (review_export, review_models_v2, review_queue_normalizer)
    non encore migrés.
    """
    triage = empty_triage_skeleton()
    triage["source"] = TRIAGE_SOURCE_VERSION
    triage.update(
        {
            "category": "NON_PERTINENT",
            "risk_type": "autre",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "impact_description": "",
            "reference_reglementaire": "",
            "confidence": 0.0,
            "signals": {
                "regulatory_reference_added": False,
                "methodology_change": False,
                "tone_changed": False,
                "forward_looking": False,
                "quantitative_changed": False,
            },
        }
    )
    return triage


_FEW_SHOT_TRIAGE_AMF = """\
Exemples à imiter strictement. Le champ nouvelle_idee_justification doit être une note d'analyste en français, préfixée OUI/NON, avec les rubriques exactes : Nouvel élément à surveiller, Sujet détecté, Ce qui change, Pertinence métier, Point de surveillance. Les codes AMF servent à choisir les sujets, mais la justification doit expliquer la pertinence métier avec un vocabulaire naturel d'analyste de vigie.

Exemple 1 — Risque émergent IA (added, MAJEUR)
Input : diff_type="added", T1="", T2="La Banque a établi un cadre de gouvernance pour l'utilisation responsable de l'intelligence artificielle générative dans ses activités."
Output : {"is_relevant": true, "themes_amf": ["RISQUE_EMERGENT", "GOUVERNANCE_RISQUES", "DIVULGATION_AJOUT"], "impact_level": "MAJEUR", "nouvelle_idee": true, "action_requise": "revue_prioritaire", "exclusion_reason": null, "explanation": "Au T2, la banque introduit un cadre formel de gouvernance pour l'IA générative, absent au T1. Ce changement relève des risques émergents et de la gouvernance des risques selon les attentes AMF. Il ajoute une information substantielle sur les contrôles et responsabilités associés à une technologie émergente.", "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui.\n\nSujet détecté : Intelligence artificielle, risque émergent, gouvernance des risques, information ajoutée.\n\nCe qui change : Le T2 ajoute un cadre de gouvernance pour l'utilisation responsable de l'intelligence artificielle générative dans les activités de la banque. Cette information était absente du T1 et introduit un sujet de risque technologique explicite dans la divulgation.\n\nPertinence métier : Ce changement met l'accent sur l'encadrement d'un risque émergent qui peut toucher la gestion des modèles, les fournisseurs technologiques, les contrôles internes, la conformité, la protection des données et la gouvernance des risques. La mention d'un cadre de gouvernance ne décrit pas seulement l'utilisation d'un outil; elle rend visible la manière dont la banque structure ses responsabilités et ses contrôles autour d'une technologie qui devient comparable entre pairs.\n\nPoint de surveillance : Intelligence artificielle / gouvernance des risques — Le changement indique que la banque formalise davantage l'encadrement de l'IA générative. Ce point permet de suivre la maturité des contrôles liés aux modèles, aux fournisseurs technologiques, à la protection des données et à la comparabilité des pratiques de gouvernance IA entre pairs."}

Exemple 2 — Retrait de facteur de risque cyber (removed, MAJEUR)
Input : diff_type="removed", T1="Les risques liés aux cybermenaces incluent les attaques par déni de service et les ransomwares.", T2=""
Output : {"is_relevant": true, "themes_amf": ["FACTEUR_RISQUE_CHANGEMENT", "RISQUE_EMERGENT", "DIVULGATION_RETRAIT"], "impact_level": "MAJEUR", "nouvelle_idee": true, "action_requise": "revue_prioritaire", "exclusion_reason": null, "explanation": "Au T1, la banque listait explicitement les attaques par déni de service et les ransomwares comme cybermenaces, mais cette mention disparaît au T2. Ce retrait touche un facteur de risque et un risque émergent prioritaire. Il modifie le niveau de détail de la divulgation cyber.", "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui.\n\nSujet détecté : Cybersécurité, risque émergent, facteur de risque modifié, information retirée.\n\nCe qui change : Le T2 retire la mention explicite des attaques par déni de service et des ransomwares comme cybermenaces. Ces risques étaient nommés directement au T1 et ne sont plus présentés avec le même niveau de précision dans le passage comparé.\n\nPertinence métier : Ce changement met l'accent sur la précision de la divulgation relative aux cyberrisques. Dans un rapport bancaire, l'ajout ou le retrait de menaces cyber précises peut modifier la lecture de l'exposition au risque, de la transparence et de la comparabilité avec les pairs, surtout lorsque les cybermenaces constituent un sujet de surveillance prioritaire.\n\nPoint de surveillance : Cyberrisque — Le changement modifie le niveau de détail fourni sur les menaces cyber, notamment les attaques par déni de service et les ransomwares. Ce point permet de suivre la transparence de la banque sur son exposition aux risques technologiques, la précision de ses facteurs de risque et la comparabilité de sa divulgation avec les autres institutions."}

Exemple 3 — Nouvelle mention BSIF climatique (added, MAJEUR)
Input : diff_type="added", T1="", T2="Conformément aux nouvelles attentes du BSIF en matière de risques climatiques (Ligne directrice B-15), nous avons mis en place un comité dédié."
Output : {"is_relevant": true, "themes_amf": ["NOUVELLE_MENTION_REGLEMENTAIRE", "ESG_CLIMATIQUE", "GOUVERNANCE_RISQUES", "DIVULGATION_AJOUT"], "impact_level": "MAJEUR", "nouvelle_idee": true, "action_requise": "revue_prioritaire", "exclusion_reason": null, "explanation": "Au T2, la banque mentionne pour la première fois la Ligne directrice B-15 du BSIF et la création d'un comité ESG ou climatique. Ce changement croise nouvelle mention réglementaire, divulgation ESG et gouvernance des risques. Il rend plus explicite l'arrimage de la banque aux attentes prudentielles climatiques.", "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui.\n\nSujet détecté : Risque climatique, ESG, nouvelle mention réglementaire, gouvernance des risques.\n\nCe qui change : Le T2 ajoute une référence aux attentes du BSIF en matière de risques climatiques, soit la Ligne directrice B-15, ainsi qu'un comité dédié. Cette mention n'apparaissait pas au T1 et introduit une articulation plus explicite entre réglementation climatique et gouvernance interne.\n\nPertinence métier : Ce changement met l'accent sur l'évolution du cadre réglementaire applicable aux divulgations climatiques des institutions financières. La mise à jour de la ligne directrice B-15 par le BSIF peut influencer la manière dont les banques planifient leur conformité, structurent leurs contrôles ESG et communiquent leurs risques climatiques. Ce point est important à suivre, car il permet d'évaluer l'évolution des attentes prudentielles, la comparabilité des pratiques de divulgation entre pairs et le niveau de préparation des banques face aux exigences climatiques.\n\nPoint de surveillance : Risque climatique / ESG — Le changement indique que la banque rend plus explicite son positionnement face aux exigences climatiques du BSIF. Ce point permet de suivre l'adaptation aux attentes prudentielles climatiques, le niveau de préparation ESG et la comparabilité des pratiques de divulgation entre pairs."}

Exemple 4 — Variation chiffrée propre à la banque (EXCLU)
Input : diff_type="modified", T1="Notre portefeuille de prêts hypothécaires s'élève à 287 G$.", T2="Notre portefeuille de prêts hypothécaires s'élève à 294 G$."
Output : {"is_relevant": false, "themes_amf": [], "impact_level": "MINEUR", "nouvelle_idee": false, "action_requise": "aucune", "exclusion_reason": "variation_numerique_propre_banque", "explanation": "", "nouvelle_idee_justification": "NON — Nouvel élément à surveiller : Non.\n\nSujet détecté : Mise à jour quantitative propre à la banque.\n\nCe qui change : Le T2 met à jour le montant du portefeuille de prêts hypothécaires, qui passe de 287 G$ à 294 G$. L'indicateur existait déjà au T1 et le changement porte uniquement sur la valeur publiée.\n\nPertinence métier : Cette variation ne constitue pas une nouvelle idée à surveiller, car elle reflète l'évolution normale d'un chiffre propre à la banque. Elle ne modifie aucun seuil prudentiel, aucune règle BSIF ou AMF, aucune méthode de calcul et aucune posture de risque qui changerait la lecture réglementaire ou la comparabilité métier.\n\nPoint de surveillance : Mise à jour quantitative — La substance de la divulgation demeure stable. Ce point peut être écarté du suivi prioritaire, sauf si une autre information indique un changement de seuil prudentiel, de méthode, de conformité ou de posture de risque."}

Exemple 5 — Montant réglementaire (seuil prudentiel)
Input : diff_type="modified", T1="Le seuil prudentiel CET1 minimal applicable est de 4,5 %.", T2="Le seuil prudentiel CET1 minimal applicable est de 5,0 %, conformément aux nouvelles exigences pilier 2 du BSIF."
Output : {"is_relevant": true, "themes_amf": ["RATIOS_REGLEMENTAIRES", "EXIGENCES_REGLEMENTAIRES", "MONTANT_REGLEMENTAIRE", "NOUVELLE_MENTION_REGLEMENTAIRE"], "impact_level": "MAJEUR", "nouvelle_idee": true, "action_requise": "revue_prioritaire", "exclusion_reason": null, "explanation": "Au T2, le seuil prudentiel CET1 minimal applicable passe de 4,5 % à 5,0 % en lien avec les nouvelles exigences pilier 2 du BSIF. Ce changement porte sur un seuil réglementaire, pas sur une variation propre à la banque. Il modifie la lecture du cadre de capital applicable.", "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui.\n\nSujet détecté : Ratio prudentiel, seuil réglementaire, exigence réglementaire, capital.\n\nCe qui change : Le T2 modifie le seuil prudentiel CET1 minimal applicable, qui passe de 4,5 % à 5,0 %, et rattache ce changement aux nouvelles exigences pilier 2 du BSIF. Il s'agit d'un seuil réglementaire, pas d'une simple variation du ratio publié par la banque.\n\nPertinence métier : Ce changement met en évidence une évolution du cadre prudentiel applicable aux ratios réglementaires et aux exigences de capital. Un changement de seuil peut modifier l'interprétation de la marge de gestion du capital, la comparaison entre banques et la lecture du niveau de contrainte réglementaire applicable.\n\nPoint de surveillance : Capital réglementaire — La variation observée ne doit pas être lue uniquement comme un mouvement de chiffre. Ce point permet de suivre l'évolution du cadre prudentiel présenté par la banque, la contrainte réglementaire applicable et la comparabilité des ratios de capital entre institutions."}

Exemple 6 — Ajout d'un risque tarifaire / commercial (added, MAJEUR)
Input : diff_type="added", T1="", T2="L'application de nouveaux tarifs douaniers et de mesures de représailles accroît l'incertitude économique, perturbe les chaînes d'approvisionnement et amplifie la volatilité des marchés ainsi que le risque de crédit."
Output : {"is_relevant": true, "themes_amf": ["RISQUE_MACRO_GEOPOLITIQUE", "FACTEUR_RISQUE_CHANGEMENT", "DIVULGATION_AJOUT"], "impact_level": "MAJEUR", "nouvelle_idee": true, "action_requise": "revue_prioritaire", "exclusion_reason": null, "explanation": "Au T2, la banque ajoute une divulgation sur l'incidence des nouveaux tarifs douaniers et des mesures de représailles, absente au T1. Ce déclencheur macroéconomique et commercial se transmet au risque de crédit, au risque de marché et aux chaînes d'approvisionnement. Il introduit un facteur de risque externe explicite dans la divulgation.", "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui.\n\nSujet détecté : Risque commercial et géopolitique, tarifs douaniers, facteur de risque, information ajoutée.\n\nCe qui change : Le T2 ajoute une divulgation sur les nouveaux tarifs douaniers et les mesures de représailles, ainsi que leurs effets sur l'incertitude économique, les chaînes d'approvisionnement, la volatilité des marchés et le risque de crédit. Cette information était absente du T1.\n\nPertinence métier : Ce changement met l'accent sur un déclencheur macroéconomique et commercial externe qui se transmet aux risques bancaires classiques. L'ajout d'une divulgation tarifaire rend visible la manière dont la banque relie un choc commercial à son exposition au crédit, au marché et au financement, ce qui modifie la lecture de son profil de risque.\n\nPoint de surveillance : Risque commercial et géopolitique — Le changement indique que la banque divulgue désormais explicitement l'incidence des tarifs douaniers. Ce point permet de suivre la transmission de ce déclencheur externe au risque de crédit et de marché, ainsi que l'évolution de la transparence de la banque sur les chocs commerciaux."}

Exemple 7 — Retrait d'un risque tarifaire / commercial (removed, MAJEUR)
Input : diff_type="removed", T1="Les nouveaux tarifs douaniers pourraient avoir une incidence sur les clients de détail et commerciaux, qui pourraient être touchés par la hausse du chômage et voir leur capacité à rembourser leurs prêts réduite.", T2=""
Output : {"is_relevant": true, "themes_amf": ["RISQUE_MACRO_GEOPOLITIQUE", "FACTEUR_RISQUE_CHANGEMENT", "DIVULGATION_RETRAIT"], "impact_level": "MAJEUR", "nouvelle_idee": true, "action_requise": "revue_prioritaire", "exclusion_reason": null, "explanation": "Au T1, la banque divulguait l'incidence des tarifs douaniers sur ses clients et sur leur capacité de remboursement, mais cette mention disparaît au T2. Ce retrait touche un déclencheur macroéconomique et commercial relié au risque de crédit. Il réduit le niveau de détail de la divulgation sur un risque externe important.", "nouvelle_idee_justification": "OUI — Nouvel élément à surveiller : Oui.\n\nSujet détecté : Risque commercial et géopolitique, tarifs douaniers, facteur de risque, information retirée.\n\nCe qui change : Le T2 retire la divulgation, présente au T1, sur l'incidence des tarifs douaniers sur les clients de détail et commerciaux et sur leur capacité à rembourser leurs prêts. Ce lien explicite entre tarifs et risque de crédit n'apparaît plus.\n\nPertinence métier : Ce changement met l'accent sur le fait que la banque atténue sa communication sur un déclencheur macroéconomique et commercial relié au risque de crédit. Le retrait d'une divulgation sur les tarifs n'est pas neutre : il modifie la lecture de l'exposition de la banque et de sa transparence sur un risque externe, et mérite la même attention qu'un ajout.\n\nPoint de surveillance : Risque commercial et géopolitique — Le changement indique que la banque retire une divulgation tarifaire reliée au risque de crédit. Ce point permet de suivre l'évolution de la transparence de la banque sur les chocs commerciaux et la cohérence de sa divulgation du risque externe dans le temps."}
"""


def _derive_legacy_fields(triage_amf: dict[str, Any]) -> dict[str, Any]:
    """Dérive les champs hérités (category, signals, ...) depuis le schéma AMF v2.

    Permet aux consommateurs aval (review_export, review_models_v2, ...) qui
    lisent encore l'ancien schéma de continuer à fonctionner sans modification.
    À retirer une fois ces consommateurs migrés vers ``themes_amf``.
    """
    if not triage_amf.get("is_relevant"):
        return {
            "category": "NON_PERTINENT",
            "risk_type": "autre",
            "relevance_score": "FAIBLE",
            "risk_level": "FAIBLE",
            "impact_description": "",
            "reference_reglementaire": "",
            "confidence": 0.0,
            "signals": {
                "regulatory_reference_added": False,
                "methodology_change": False,
                "tone_changed": False,
                "forward_looking": False,
                "quantitative_changed": False,
            },
        }

    themes = set(triage_amf.get("themes_amf") or [])
    impact = str(triage_amf.get("impact_level") or "MINEUR").upper()

    if themes & {"CAPITAL_REGLEMENTAIRE", "FONDS_PROPRES_REGLEMENTAIRES", "RATIOS_REGLEMENTAIRES"}:
        category = "CAPITAL"
        risk_type = "capital"
    elif "LIQUIDITE" in themes:
        category = "REGLEMENTAIRE"
        risk_type = "liquidite"
    elif themes & {"EXIGENCES_REGLEMENTAIRES", "NOUVELLE_MENTION_REGLEMENTAIRE"}:
        category = "REGLEMENTAIRE"
        risk_type = "conformite"
    elif themes & {"MODIFICATION_TEXTE_RISQUE", "FACTEUR_RISQUE_CHANGEMENT", "HYPOTHESES_EXPLICATIONS_RISQUES"}:
        category = "RISQUE"
        risk_type = "credit"
    elif themes & {"RISQUE_EMERGENT", "RISQUE_DONNEES", "RISQUE_TIERS_CLOUD"}:
        category = "RISQUE"
        risk_type = "autre"
    elif "ESG_CLIMATIQUE" in themes:
        category = "RISQUE"
        risk_type = "autre"
    elif themes & {"GOUVERNANCE_RISQUES", "CONTROLE_CONFORMITE"}:
        category = "STRUCTURE"
        risk_type = "conformite"
    elif "STRUCTURE_RAPPORT" in themes:
        category = "STRUCTURE"
        risk_type = "autre"
    else:
        category = "STRUCTURE"
        risk_type = "autre"

    severity_map = {"MAJEUR": "ELEVEE", "MODERE": "MOYENNE", "MINEUR": "FAIBLE"}
    return {
        "category": category,
        "risk_type": risk_type,
        "relevance_score": severity_map.get(impact, "FAIBLE"),
        "risk_level": severity_map.get(impact, "FAIBLE"),
        "impact_description": "",
        "reference_reglementaire": "",
        "confidence": 0.85,
        "signals": {
            "regulatory_reference_added": "NOUVELLE_MENTION_REGLEMENTAIRE" in themes,
            "methodology_change": "MODIFICATION_METHODOLOGIE" in themes,
            "tone_changed": False,
            "forward_looking": False,
            "quantitative_changed": "MONTANT_REGLEMENTAIRE" in themes,
        },
    }


def _triage_section_changes(
    *,
    client: Any,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Qualifie metier les changements detectes et fusionne le triage.

    Le triage ne recalcule pas la diff textuelle: il prend les changements deja
    identifies, demande une qualification selective au modele, puis rattache le
    resultat a chaque changement pour la retention finale et le resume global.

    Aligne sur la taxonomie AMF appliquee au suivi prudentiel canadien. Le modèle produit
    le schéma AMF v2 (themes_amf multi-label, exclusion_reason, ...) ; les
    champs hérités (category, signals, ...) sont dérivés localement pour
    préserver la compatibilité aval.
    """
    if not changes:
        return []

    if len(changes) > _TRIAGE_BATCH_SIZE:
        chunks = [
            changes[start : start + _TRIAGE_BATCH_SIZE]
            for start in range(0, len(changes), _TRIAGE_BATCH_SIZE)
        ]
        max_workers = min(_MAX_TRIAGE_LLM_WORKERS, len(chunks))
        results_by_index: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _triage_section_changes,
                    client=client,
                    model=model,
                    section_key=section_key,
                    changes=chunk,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results_by_index[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Section triage failed for {section_key}/batch t{index:02d}: {exc}"
                    ) from exc

        enriched_batches: list[dict[str, Any]] = []
        for index in range(len(chunks)):
            enriched_batches.extend(results_by_index.get(index, []))
        return enriched_batches

    triage_inputs = []
    for idx, change in enumerate(changes, start=1):
        triage_inputs.append(
            {
                "change_index": idx,
                "diff_type": change["diff_type"],
                "semantic_text_t1": _truncate_prompt_text(
                    change.get("semantic_text_t1", ""),
                    _TRIAGE_SEMANTIC_TEXT_LIMIT,
                ),
                "semantic_text_t2": _truncate_prompt_text(
                    change.get("semantic_text_t2", ""),
                    _TRIAGE_SEMANTIC_TEXT_LIMIT,
                ),
                "source_snippet_t1": _truncate_prompt_text(
                    change.get("source_text_t1") or "",
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "source_snippet_t2": _truncate_prompt_text(
                    change.get("source_text_t2") or "",
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "change_summary": change.get("change_summary", ""),
            }
        )

    system_prompt = (
        "Tu es un analyste senior en gestion intégrée des risques, spécialisé "
        "dans la vigie de pairs des banques canadiennes alignée sur les "
        "attentes de l'AMF (Autorité des marchés financiers du Québec) et "
        "du BSIF.\n\n"
        "Tu analyses chaque changement détecté entre deux rapports d'une "
        "même banque comparés pair-à-pair :\n"
        "- T1 = rapport PRÉCÉDENT dans la paire\n"
        "- T2 = rapport COURANT dans la paire\n"
        "Les paires possibles sont : T2 vs T1, T3 vs T2, T1 N+1 vs T3 N "
        "(passage d'année), T4 N+1 vs T4 N (rapports annuels). Le suffixe "
        "T1/T2 ne désigne PAS forcément un trimestre.\n\n"
        "Tu utilises uniquement la taxonomie AMF fournie ci-dessous, en "
        "multi-label si plusieurs thèmes s'appliquent."
    )

    user_prompt = (
        "Pour chaque changement de la liste ci-dessous, produis un triage AMF "
        "dans le batch de sortie en réutilisant le même change_index. Le "
        "schéma de sortie est imposé par l'API ; tu ne dois pas en dévier.\n\n"
        "Ne produis pas de segments de surlignage ni de preuve verbatim : "
        "ces preuves sont calculées localement à partir des textes T1/T2. "
        "Concentre-toi uniquement sur le triage métier AMF.\n\n"
        "Taxonomie AMF (utilise UNIQUEMENT ces codes pour themes_amf, "
        "multi-label autorisé et encouragé) :\n"
        f"{format_themes_for_prompt()}\n\n"
        "Libellés analyste à utiliser dans `Sujet détecté` et dans "
        "`nouvelle_idee_justification` (ne pas laisser seulement les codes "
        "AMF techniques) :\n"
        f"{format_theme_subjects_for_prompt()}\n\n"
        "Raisons d'exclusion (à utiliser quand is_relevant=false) :\n"
        f"{format_exclusion_reasons_for_prompt()}\n\n"
        "Règles métier :\n"
        "1. MULTI-LABEL : un changement peut combiner plusieurs thèmes "
        "(ex : modification méthodologique qui touche la gouvernance → "
        '["MODIFICATION_METHODOLOGIE", "GOUVERNANCE_RISQUES"]).\n\n'
        "1b. RENOMMAGE NARRATIF : si diff_type='renamed', il s'agit d'un "
        "titre ou d'une sous-section renommée entre T1 et T2. Classer avec "
        "'STRUCTURE_RAPPORT' et, si le nouveau libellé change la lecture du "
        "risque, ajouter les thèmes métier applicables. Ne pas assimiler un "
        "renommage à une simple reformulation lorsque le libellé oriente "
        "différemment l'analyse de vigie.\n\n"
        "2. EXCLUSIONS DURES — mettre is_relevant=false avec exclusion_reason :\n"
        "   - Variations chiffrées PROPRES à la banque (taille du portefeuille, "
        "exposition, profits, montants d'actifs, ratios chiffrés de la banque) "
        "→ 'variation_numerique_propre_banque'.\n"
        "   - Reformulation sans nouveau fond (synonymes, ordre des mots, "
        "tournure équivalente) → 'reformulation_mineure'.\n"
        "   - Texte déplacé sans modification → 'deplacement_texte'.\n"
        "   - Formatage visuel (gras, italique, ponctuation, casse, espacement) "
        "→ 'formatage_visuel'.\n"
        "   IMPORTANT : ces changements restent dans le batch de sortie. Tu les "
        "classes comme non pertinents, mais tu ne les supprimes jamais : "
        "la décision finale appartient à l'analyste.\n\n"
        "2b. DEPLACEMENT DE TEXTE — nuance obligatoire :\n"
        "   - Classer en 'deplacement_texte' UNIQUEMENT si le passage conserve "
        "le même sens, le même niveau de détail et un contexte équivalent.\n"
        "   - Si le déplacement change la section, le titre, la visibilité, "
        "le rattachement à un thème AMF, la posture de risque ou le niveau de "
        "mise en évidence, NE PAS l'exclure comme simple déplacement. Classer "
        "plutôt avec 'STRUCTURE_RAPPORT' et les thèmes métier applicables "
        "(ex. RISQUE_DONNEES, RISQUE_TIERS_CLOUD, EXIGENCES_REGLEMENTAIRES, "
        "FACTEUR_RISQUE_CHANGEMENT).\n"
        "   - Un paragraphe déplacé d'une rubrique générale vers une rubrique "
        "dédiée aux risques, à la réglementation, aux données, aux tiers, à "
        "l'infonuagique ou à la cybersécurité peut être une nouvelle idée si "
        "ce nouveau contexte change la lecture analyste.\n\n"
        "3. INCLUSION EXPLICITE — les MONTANTS RÉGLEMENTAIRES (seuils "
        "prudentiels, planchers Bâle, exigences pilier 2, lignes directrices "
        "BSIF chiffrées) sont EN scope. Ajouter le marqueur 'MONTANT_REGLEMENTAIRE' "
        "aux thèmes principaux quand la divulgation porte sur un seuil "
        "réglementaire chiffré (PAS un chiffre propre à la banque).\n\n"
        "4. RISQUE_EMERGENT (cyberrisque, IA, IA générative, fraude numérique, "
        "ransomware, modèles tiers) est PRIORITAIRE : impact_level minimum MODERE.\n\n"
        "4a. RISQUE_DONNEES et RISQUE_TIERS_CLOUD sont des axes distincts. "
        "Ne classe pas une simple occurrence du mot 'données' ou 'tiers'. "
        "Retiens ces thèmes lorsque la divulgation traite de gouvernance, "
        "qualité, protection, localisation ou cycle de vie des données, ou de "
        "fournisseurs critiques, impartition, concentration, infonuagique, "
        "résilience et stratégie de sortie.\n\n"
        "4b. RISQUE_MACRO_GEOPOLITIQUE (tarifs douaniers, guerre commerciale, "
        "sanctions, conflits, incertitude des politiques commerciales) est un "
        "déclencheur externe qui se transmet au crédit, au marché et au "
        "financement : PRIORITAIRE, impact_level minimum MODERE. Un AJOUT comme "
        "un RETRAIT significatif de cette divulgation est MAJEUR — un retrait "
        "signale que la banque atténue sa communication sur ce risque, ce qui "
        "est aussi important qu'un ajout.\n\n"
        "5. nouvelle_idee = true SI ET SEULEMENT SI les 3 conditions cumulatives sont vraies :\n"
        "   (a) SUBSTANTIELLE : modifie la SUBSTANCE de la divulgation (concept, "
        "facteur de risque, mention réglementaire, méthodologie, indicateur "
        "prudentiel) — PAS une variation chiffrée propre à la banque ni une "
        "reformulation.\n"
        "   (b) NOUVEAUTÉ INFORMATIONNELLE : ajoute un élément absent au T1, OU "
        "retire un élément présent au T1, OU modifie substantiellement la "
        "posture de la banque sur un thème AMF.\n"
        "   (c) ADOSSÉE À UN THÈME AMF : au moins un code dans themes_amf "
        "(sinon hors scope vigie).\n"
        "   Si UNE des 3 conditions est violée → nouvelle_idee = false.\n\n"
        "5b. changement_posture : détermine si le changement modifie la façon "
        "de gérer le risque. Utilise RENFORCEMENT pour des contrôles ou une "
        "surveillance renforcés, ALLEGEMENT pour un encadrement réduit, "
        "NOUVEAU_DISPOSITIF pour un nouveau comité, cadre, responsabilité, "
        "diligence, exigence contractuelle ou stratégie de sortie, "
        "RETRAIT_DISPOSITIF pour leur suppression, AUCUN si la gestion ne "
        "change pas, et INDETERMINE si le texte ne permet pas de conclure. "
        "Une simple mention de risque n'est pas un changement de posture. "
        "Pour une posture autre que AUCUN ou INDETERMINE, renseigne "
        "justification_posture avec exactement quatre rubriques séparées par "
        "\\n\\n : Preuve, Effet sur la gestion du risque, Justification du "
        "statut, Justification de la confiance. Renseigne aussi "
        "confiance_posture (ELEVEE, MOYENNE ou FAIBLE).\n\n"
        "5c. statut_mise_en_oeuvre décrit le niveau réellement démontré par le "
        "rapport : ANNONCE, PLANIFIE, EN_COURS, MIS_EN_OEUVRE ou INDETERMINE. "
        "Ne transforme jamais un futur, un projet ou une intention en mesure "
        "déjà mise en œuvre.\n\n"
        "6. impact_level :\n"
        "   - MAJEUR : modification méthodologique substantielle, retrait/ajout "
        "significatif de divulgation, nouvelle exigence réglementaire, risque "
        "émergent introduit ou retiré.\n"
        "   - MODERE : modification de posture, croisement multi-thèmes notable.\n"
        "   - MINEUR : changement modeste mais substantif.\n\n"
        "6b. impact_it est un axe distinct de impact_level et doit rester "
        "INDETERMINE par défaut :\n"
        "   - Règle de prudence : ne JAMAIS inférer un impact IT indirectement. "
        "Évalue impact_it seulement si le changement contient un signal explicite "
        "lié aux systèmes, à la technologie, aux données, aux fournisseurs, au cloud, "
        "à la cybersécurité, à l'IA, aux modèles, à l'automatisation, à l'infrastructure, "
        "à une migration ou à des contrôles technologiques.\n"
        "   - ELEVE : migration ou changement d'architecture, remplacement ou "
        "sortie d'un fournisseur, contrôles technologiques majeurs, localisation "
        "ou déplacement de données, continuité ou résilience structurante.\n"
        "   - MOYEN : nouveaux processus, inventaires, surveillance, rapports, "
        "diligence ou exigences contractuelles nécessitant un effort IT.\n"
        "   - FAIBLE : clarification ou ajustement limité sans transformation "
        "technologique apparente, mais avec un effet IT identifiable.\n"
        "   - INDETERMINE : information insuffisante ou absence de signal IT explicite. "
        "Utilise INDETERMINE pour les changements de capital, ratio, crédit, liquidité, "
        "réglementation ou gouvernance générale qui ne mentionnent pas clairement "
        "une dimension technologique. Ne déduis jamais qu'un changement IT est réalisé "
        "si le rapport décrit seulement une intention. FAIBLE ne signifie pas absence "
        "d'impact IT : si le lien IT n'est pas explicite, choisis INDETERMINE. "
        "Renseigne impact_it_justification avec exactement trois rubriques "
        "séparées par \\n\\n : Éléments observés, Conséquence probable, Limite "
        "de l'analyse. Laisse ce champ vide si impact_it=INDETERMINE.\n\n"
        "7. action_requise : 'revue_prioritaire' UNIQUEMENT pour les changements MAJEUR "
        "(invariant strict — revue_prioritaire exige impact_level=MAJEUR) ; 'investigation' "
        "pour MODERE ou MAJEUR sans revue_prioritaire ; 'confirmation' à valider avec "
        "source ; 'information' pertinent non actionnable ; 'aucune' uniquement "
        "si is_relevant=false.\n\n"
        "8. INVARIANTS STRICTS (toute violation rejette la réponse) :\n"
        "   - is_relevant=true → themes_amf NON VIDE, exclusion_reason=null, "
        "explanation ≥ 50 caractères (3 phrases pleines), nouvelle_idee_justification "
        "≥ 3 phrases complètes, entre 220 et 700 caractères au total, commençant par 'OUI' "
        "et contenant les rubriques exactes : Nouvel élément à surveiller, "
        "Sujet détecté, Ce qui change, Pertinence métier, Point de surveillance.\n"
        "   - is_relevant=false → themes_amf=[], exclusion_reason renseigné, "
        "nouvelle_idee=false, impact_level=MINEUR, action_requise='aucune', "
        "changement_posture=AUCUN, impact_it=INDETERMINE, "
        "justification_posture vide, statut_mise_en_oeuvre=INDETERMINE, "
        "confiance_posture=INDETERMINE, impact_it_justification vide, "
        "explanation vide. "
        "nouvelle_idee_justification "
        "OBLIGATOIRE : ≥ 3 phrases complètes, entre 220 et 700 caractères, commençant "
        "par 'NON', contenant les mêmes rubriques exactes, et expliquant "
        "clairement POURQUOI ce changement n'est pas une nouvelle idée AMF "
        "(citer la raison d'exclusion en langage métier, pas seulement le code).\n\n"
        "Exigence pour `explanation` (3 phrases obligatoires si is_relevant=true, "
        "chaîne vide sinon) :\n"
        "1. Ce qui a changé concrètement entre T1 (précédent) et T2 (courant).\n"
        "2. Pourquoi ce changement relève des thèmes AMF identifiés (et non "
        "d'une simple reformulation ou variation chiffrée propre à la banque).\n"
        "3. Ce que cela implique pour la surveillance de cette banque.\n\n"
        "Exigence pour `nouvelle_idee_justification` (TOUJOURS obligatoire, "
        "y compris pour les is_relevant=false) :\n"
        "- Format STRICT : commencer par 'OUI' (si nouvelle_idee=true) ou 'NON' "
        "(si nouvelle_idee=false), suivi d'un tiret '—' ou '-'.\n"
        "- LONGUEUR STRICTE : rester entre 220 et 700 caractères au total. "
        "Une phrase courte par rubrique suffit; ne jamais produire de note longue.\n"
        "- Rédiger une NOTE D'ANALYSTE avec ces rubriques EXACTES, dans cet "
        "ordre, séparées par \\n\\n :\n"
        "  1) 'OUI — Nouvel élément à surveiller : Oui' ou "
        "'NON — Nouvel élément à surveiller : Non'.\n"
        "  2) 'Sujet détecté : ...' avec des mots simples liés aux codes AMF "
        "(IA, cybersécurité, risque climatique, conformité, capital, liquidité, "
        "méthode de calcul, information ajoutée ou retirée).\n"
        "  3) 'Ce qui change : ...' avec l'élément exact ajouté, retiré ou "
        "modifié entre T1 et T2.\n"
        "  4) 'Pertinence métier : ...' avec une explication concise, concrète "
        "et formulée comme un analyste de vigie : commencer idéalement par "
        "'Ce changement met l'accent sur ...' ou 'Ce changement met en évidence ...'. "
        "Relier le changement au sujet détecté, aux attentes prudentielles, "
        "à la conformité, aux contrôles, à la comparabilité entre pairs et à "
        "son importance pour une banque. Éviter la formule qui associe "
        "directement les mots vigie et bancaire.\n"
        "  5) 'Point de surveillance : ...' avec le point de surveillance à retenir, "
        "sans demander à l'analyste de vérifier, accepter ou rejeter le changement.\n"
        "- Au moins 3 phrases complètes (ponctuation finale, ≥ 20 chars chacune) "
        "et ≥ 200 caractères au total — l'analyste doit avoir une explication "
        "claire et concise, pas une note exhaustive.\n"
        "- Citer l'élément SPÉCIFIQUE du rapport : nom exact d'un indicateur, "
        "fragment de phrase, libellé de footnote, titre de tableau — adossé au "
        "contenu réel des rapports aux actionnaires traités.\n"
        "- Si is_relevant=true : mentionner explicitement le ou les sujets AMF "
        "concernés en langage naturel dans 'Sujet détecté' et expliquer en quoi le changement "
        "constitue une nouveauté.\n"
        "- Si is_relevant=false : expliquer en LANGAGE MÉTIER pourquoi ce "
        "changement n'est PAS une nouvelle idée AMF (variation chiffrée propre "
        "à la banque, reformulation sans nouveau fond, déplacement de texte, "
        "etc.). L'analyste doit comprendre la raison de l'exclusion sans avoir "
        "à interpréter le code d'exclusion.\n"
        "- Ne pas produire une justification de type gabarit qui se contente de "
        "dire 'ce changement affecte les thèmes AMF ...'. Les codes AMF peuvent "
        "être mentionnés, mais ils ne remplacent jamais l'explication métier.\n"
        "- Ne pas utiliser de formules de tâche comme 'vérifier si', 'accepter', "
        "'rejeter' ou 'à confirmer par l'analyste' : Dash affiche déjà la "
        "preuve et l'analyste prend la décision finale.\n"
        "- Adossé aux règles AMF appliquées sur le contenu réel (pas de "
        "généralités, pas de paraphrase de la règle abstraite).\n\n"
        f"{_FEW_SHOT_TRIAGE_AMF}\n"
        f"Section: {section_key}\n"
        f"Changements à trier:\n{_json_dumps(triage_inputs)}"
    )

    try:
        batch = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TriageAMFLLMBatch,
            max_retries=1,
        )
    except ValidationError as exc:
        raise TriageValidationError(
            section_key=section_key,
            change_index=None,
            raw_payload=None,
            validation_error=exc,
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Section triage failed for {section_key}: {exc}") from exc

    triage_map: dict[int, dict[str, Any]] = {}
    relevant_count = 0
    nouvelle_idee_count = 0
    for triage_obj in batch.triages:
        triage_dict = triage_obj.model_dump(exclude={"change_index"})
        source_change = changes[triage_obj.change_index - 1]
        triage_dict["change_segments"] = (
            build_change_segments(source_change) if triage_dict.get("is_relevant") else []
        )
        triage_dict["explanation"] = _sanitize_explanation(triage_dict["explanation"])
        legacy_fields = _derive_legacy_fields(triage_dict)
        triage = {**triage_dict, **legacy_fields, "source": TRIAGE_SOURCE_VERSION}
        triage_map[triage_obj.change_index] = triage
        if triage_obj.is_relevant:
            relevant_count += 1
        if triage_obj.nouvelle_idee:
            nouvelle_idee_count += 1
        logger.info(
            "triage validated section=%s change_index=%d is_relevant=%s themes=%s impact=%s nouvelle_idee=%s action=%s",
            section_key,
            triage_obj.change_index,
            triage_obj.is_relevant,
            triage_obj.themes_amf,
            triage_obj.impact_level,
            triage_obj.nouvelle_idee,
            triage_obj.action_requise,
        )

    logger.info(
        "triage section summary section=%s total=%d relevant=%d nouvelles_idees=%d",
        section_key,
        len(batch.triages),
        relevant_count,
        nouvelle_idee_count,
    )

    enriched: list[dict[str, Any]] = []
    for idx, change in enumerate(changes, start=1):
        triage = triage_map.get(idx, _default_triage())
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return enriched
