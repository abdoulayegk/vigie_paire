"""Regroupement des changements semantiquement equivalents et propagation du triage.

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

import logging
from typing import Any

from vigie.analyse_texte.openai_client import _embed_texts

from .constants import (
    _DEFAULT_EMBEDDING_MODEL,
    _TRIAGE_DEDUP_EMBEDDING_THRESHOLD,
    _TRIAGE_EMBEDDING_TRUNCATE_CHARS,
)
from .results import _default_triage

logger = logging.getLogger("vigie.analyse_texte.triage")


def _triage_retrieval_text(change: dict[str, Any]) -> str:
    parts = [
        str(change.get("diff_type") or ""),
        str(change.get("subsection_heading") or ""),
        str(change.get("change_summary") or ""),
        str(change.get("source_text_t1") or "")[:600],
        str(change.get("source_text_t2") or "")[:600],
    ]
    text = " | ".join(part.strip() for part in parts if str(part or "").strip())
    if len(text) <= _TRIAGE_EMBEDDING_TRUNCATE_CHARS:
        return text
    return text[:_TRIAGE_EMBEDDING_TRUNCATE_CHARS]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _changes_compatible_for_dedup(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Never merge when nature, decision or evidence shape diverge."""
    if str(left.get("diff_type") or "") != str(right.get("diff_type") or ""):
        return False
    if str(left.get("alignment_decision") or "") != str(right.get("alignment_decision") or ""):
        return False
    left_has_t1 = bool(str(left.get("source_text_t1") or "").strip())
    right_has_t1 = bool(str(right.get("source_text_t1") or "").strip())
    left_has_t2 = bool(str(left.get("source_text_t2") or "").strip())
    right_has_t2 = bool(str(right.get("source_text_t2") or "").strip())
    if left_has_t1 != right_has_t1 or left_has_t2 != right_has_t2:
        return False
    return True


def _group_semantic_triage_duplicates(
    changes: list[dict[str, Any]],
    *,
    client: Any,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> list[list[int]]:
    """Regroupe les quasi-doublons et retourne leurs indices dans ``changes``."""
    if len(changes) <= 1:
        return [[index] for index in range(len(changes))]

    try:
        embeddings = _embed_texts(
            client,
            [_triage_retrieval_text(change) for change in changes],
            model=embedding_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Déduplication triage embeddings indisponible: %s", exc)
        return [[index] for index in range(len(changes))]

    parents = list(range(len(changes)))

    def find(index: int) -> int:
        """Retourne la racine d'un groupe avec compression de chemin."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        """Fusionne les groupes de deux changements jugés équivalents."""
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    for left_index in range(len(changes)):
        for right_index in range(left_index + 1, len(changes)):
            if not _changes_compatible_for_dedup(changes[left_index], changes[right_index]):
                continue
            score = _cosine_similarity(embeddings[left_index], embeddings[right_index])
            if score >= _TRIAGE_DEDUP_EMBEDDING_THRESHOLD:
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(changes)):
        grouped.setdefault(find(index), []).append(index)
    return [sorted(members) for members in grouped.values()]


def _propagate_triage_to_group(
    *,
    representative: dict[str, Any],
    members: list[dict[str, Any]],
    group_id: str,
    bank_code: str = "",
) -> list[dict[str, Any]]:
    triage = dict(representative.get("genai_triage") or _default_triage(bank_code))
    member_ids = [str(change.get("change_id") or "") for change in members]
    propagated: list[dict[str, Any]] = []
    for change in members:
        enriched = dict(change)
        member_triage = dict(triage)
        member_triage["triage_group_id"] = group_id
        member_triage["triage_group_member_ids"] = member_ids
        member_triage["triage_group_representative_id"] = str(representative.get("change_id") or "")
        if str(change.get("change_id") or "") != str(representative.get("change_id") or ""):
            member_triage["source"] = f"{triage.get('source') or 'gpt'}_propagated"
        enriched["genai_triage"] = member_triage
        enriched["triage_dedup"] = {
            "group_id": group_id,
            "representative_change_id": str(representative.get("change_id") or ""),
            "member_change_ids": member_ids,
            "propagated": str(change.get("change_id") or "") != str(representative.get("change_id") or ""),
        }
        propagated.append(enriched)
    return propagated


_FEW_SHOT_TRIAGE_AMF = """\
Exemple 1 — ajout cyber pertinent
Input : {"bank_subject": "CIBC", "change_index": 1, "diff_type": "added", "change_summary": "Ajout d’exercices annuels de simulation de cyberattaque."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["RISQUE_EMERGENT", "CONTROLE_CONFORMITE"], "nouvelle_idee": true, "changement_constate": "CIBC ajoute des simulations annuelles de cyberattaque avec ses unités d’affaires.", "signification_metier": "Cette évolution rend explicite un mécanisme récurrent de préparation aux incidents cybernétiques.", "comparaison_interbanques": "Elle permet de comparer la fréquence, le périmètre et la participation des unités d’affaires aux exercices déclarés par les banques.", "limite_interpretation": "La divulgation ne précise toutefois ni les scénarios testés ni les résultats obtenus.", "motif_non_pertinence": ""}

Exemple 2 — variation propre à la banque non pertinente
Input : {"bank_subject": "BMO", "change_index": 1, "diff_type": "modified", "change_summary": "Le portefeuille hypothécaire passe de 287 G$ à 294 G$."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "BMO fait passer son portefeuille hypothécaire de 287 G$ à 294 G$, sans modifier la méthode de calcul ni le périmètre présenté.", "signification_metier": "", "comparaison_interbanques": "", "limite_interpretation": "", "motif_non_pertinence": "Cette variation reflète l’évolution normale des activités et n’apporte aucun nouvel élément sur les pratiques de gestion des risques à comparer entre les banques."}

Exemple 3 — calendrier d’application non pertinent
Input : {"bank_subject": "RBC", "change_index": 1, "diff_type": "modified", "change_summary": "Le BSIF reporte l’augmentation du coefficient de plancher de 2026 à 2027."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "RBC actualise uniquement le calendrier d’application du coefficient de plancher annoncé par le BSIF, sans changer la nature de l’exigence.", "signification_metier": "", "comparaison_interbanques": "", "limite_interpretation": "", "motif_non_pertinence": "Cette mise à jour d’échéances n’apporte aucun élément nouveau pour comparer les pratiques de gestion des fonds propres entre les banques."}

Exemple 3b — report indéfini et préavis pertinent
Input : {"bank_subject": "RBC", "change_index": 1, "diff_type": "modified", "change_summary": "Le 12 février 2025, le BSIF a annoncé un report indéfini du plancher, avec un préavis d’au moins deux ans."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["EXIGENCES_REGLEMENTAIRES"], "nouvelle_idee": true, "changement_constate": "RBC divulgue le report indéfini du coefficient de plancher annoncé par le BSIF et le préavis d’au moins deux ans avant toute nouvelle hausse.", "signification_metier": "Ce report indéfini et ce préavis changent le régime d’application de l’exigence, pas seulement une date.", "comparaison_interbanques": "Ils permettent de comparer le calendrier prudentiel et le préavis réglementaire entre les banques.", "limite_interpretation": "La divulgation ne précise pas si d’autres paramètres du plancher ont aussi changé.", "motif_non_pertinence": ""}

Exemple 4 — acquisition interne non pertinente
Input : {"bank_subject": "BNC", "change_index": 1, "diff_type": "added", "change_summary": "Inclusion de CWB dans le calcul du risque opérationnel à la suite de l’acquisition."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "BNC inclut CWB dans le calcul du risque opérationnel à la suite de son acquisition, sans décrire une nouvelle méthode de calcul.", "signification_metier": "", "comparaison_interbanques": "", "limite_interpretation": "", "motif_non_pertinence": "Cette opération propre à la banque n’offre aucune base comparable sur les pratiques de gestion des risques entre institutions."}

Exemple 5 — rachat d’actions non pertinent
Input : {"bank_subject": "TD", "change_index": 1, "diff_type": "modified", "change_summary": "Mise à jour des montants de rachat d’actions ordinaires au semestre."}
Output : {"change_index": 1, "is_relevant": false, "themes_amf": [], "nouvelle_idee": false, "changement_constate": "TD met à jour les montants de rachat d’actions ordinaires déjà présentés, sans modifier le cadre réglementaire associé.", "signification_metier": "", "comparaison_interbanques": "", "limite_interpretation": "", "motif_non_pertinence": "Cette transaction propre à la banque n’éclaire pas la comparabilité des pratiques prudentielles entre pairs."}

Exemple 6 — transfert de responsabilité de gouvernance pertinent et substantiel
Input : {"bank_subject": "RBC", "change_index": 1, "diff_type": "modified", "change_summary": "L’approbation de l’appétit pour le risque passe du comité de direction au conseil d’administration."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": true, "changement_constate": "RBC transfère au conseil d’administration l’approbation de l’appétit pour le risque auparavant confiée au comité de direction.", "signification_metier": "Ce transfert élève la décision au niveau de gouvernance ultime de la banque.", "comparaison_interbanques": "Il permet de comparer l’autorité d’approbation, la répartition des responsabilités et le rôle du conseil entre les banques.", "limite_interpretation": "La divulgation ne précise toutefois pas si les mécanismes de suivi ou de reddition de comptes ont également changé.", "motif_non_pertinence": ""}

Exemple 7 — comité renommé ciblé de gouvernance pertinent
Input : {"bank_subject": "CIBC", "change_index": 1, "diff_type": "modified", "change_summary": "Le Comité de gestion des risques est renommé Comité des risques et de la conformité, sans modification de son mandat."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": true, "changement_constate": "CIBC renomme le Comité de gestion des risques en Comité des risques et de la conformité, modifiant le cadrage et la visibilité de son rôle de gouvernance.", "signification_metier": "Ce renommage élargit la perception du périmètre de conformité et change la présentation de la gouvernance des risques.", "comparaison_interbanques": "Il permet de comparer la formulation, le positionnement et le niveau d’attention accordé à la conformité dans les gouvernances des banques.", "limite_interpretation": "La divulgation ne précise toutefois pas si ce renommage s’accompagne d’une modification effective du mandat ou des responsabilités.", "motif_non_pertinence": ""}
# Post-process local : is_relevant=true + nouvelle_idee=true → impact_level potential MAJEUR ou MODERE selon thèmes.

Exemple 8 — changement réel de méthodologie pertinent et substantiel
Input : {"bank_subject": "BMO", "change_index": 1, "diff_type": "modified", "change_summary": "La méthode standard de mesure du risque de crédit est remplacée par un modèle interne avancé."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["MODIFICATION_METHODOLOGIE"], "nouvelle_idee": true, "changement_constate": "BMO remplace la méthode standard de mesure du risque de crédit par un modèle interne avancé.", "signification_metier": "Cette nouvelle base méthodologique peut modifier la mesure et la sensibilité du risque déclaré.", "comparaison_interbanques": "Elle permet de comparer les approches de modélisation, les hypothèses et le recours aux modèles internes entre les banques.", "limite_interpretation": "La divulgation ne fournit toutefois pas les paramètres ni les effets quantifiés nécessaires pour mesurer l’incidence du remplacement.", "motif_non_pertinence": ""}

Exemple 9 — modification réelle de processus pertinente et substantielle
Input : {"bank_subject": "BNS", "change_index": 1, "diff_type": "modified", "change_summary": "Les alertes de conformité sont désormais validées par une deuxième équipe avant leur clôture."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["CONTROLE_CONFORMITE"], "nouvelle_idee": true, "changement_constate": "BNS ajoute une seconde validation au processus de clôture des alertes de conformité.", "signification_metier": "Cette étape supplémentaire formalise un contrôle indépendant avant la clôture des alertes.", "comparaison_interbanques": "Elle permet de comparer le nombre de validations, la séparation des responsabilités et le niveau de supervision entre les banques.", "limite_interpretation": "La divulgation ne précise toutefois ni l’identité de la deuxième équipe ni les critères utilisés pour valider la clôture.", "motif_non_pertinence": ""}

Exemple 10 — périodicité de reporting et comportements attendus (gouvernance)
Input : {"bank_subject": "BNC", "change_index": 1, "diff_type": "modified", "change_summary": "Le rendement du capital est désormais calculé trimestriellement et la rémunération incitative vise aussi les comportements attendus."}
Output : {"change_index": 1, "is_relevant": true, "themes_amf": ["GOUVERNANCE_RISQUES"], "nouvelle_idee": true, "changement_constate": "BNC précise un calcul trimestriel du rendement du capital et ajoute que la rémunération incitative vise aussi les comportements attendus.", "signification_metier": "Cette évolution formalise la fréquence de suivi prudentiel et le lien entre rémunération et culture de risque.", "comparaison_interbanques": "Elle permet de comparer la périodicité de reporting et les critères de rémunération liés au risque entre les banques.", "limite_interpretation": "La divulgation ne détaille toutefois ni les indicateurs de comportements ni les seuils associés.", "motif_non_pertinence": ""}
"""
