"""Taxonomie AMF pour le triage GPT-4o de la vigie bancaire canadienne.

Ce module centralise la liste des thèmes AMF en scope pour la Pipeline 2
(texte narratif), les raisons d'exclusion explicites, ainsi que le modèle
Pydantic qui valide la sortie GPT.

La référence métier est la spec verrouillée pour la vigie de pairs canadienne
(alignée AMF, comparabilité des divulgations). Toute évolution de la
taxonomie doit passer par ce fichier — le prompt GPT et les exports en aval
en dérivent.

Conventions de nommage :

* ``T1`` désigne le rapport précédent dans la paire comparée (avant).
* ``T2`` désigne le rapport courant dans la paire comparée (après).
* Paires supportées : T2 vs T1, T3 vs T2, T1 N+1 vs T3 N (passage d'année), T4 N+1 vs T4 N (annuel sur annuel).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

THEMES_AMF_DESCRIPTIONS: dict[str, str] = {
    "DIVULGATION_AJOUT": ("Ajout d'une nouvelle divulgation absente du rapport précédent."),
    "DIVULGATION_RETRAIT": ("Retrait d'une divulgation présente au rapport précédent."),
    "MODIFICATION_TEXTE_RISQUE": (
        "Modification substantielle d'un texte décrivant un risque (hors simple variation chiffrée propre à la banque)."
    ),
    "MODIFICATION_METHODOLOGIE": (
        "Changement de méthodologie ou d'approche de gestion des risques "
        "(modèle interne, méthode de calcul, cadre, approche standardisée → "
        "approche par modèles internes, etc.)."
    ),
    "FACTEUR_RISQUE_CHANGEMENT": (
        "Ajout ou retrait d'un facteur de risque dans la liste des risques "
        "de la banque (nouveau risque déclaré, risque qui disparaît du listing)."
    ),
    "CAPITAL_REGLEMENTAIRE": (
        "Changement lié au capital réglementaire (CET1, AT1, Tier 2, "
        "structure du capital, instruments de capital reconnus)."
    ),
    "LIQUIDITE": (
        "Changement lié à la liquidité réglementaire (LCR, NSFR, gestion du "
        "financement, sources de liquidité, plans de contingence)."
    ),
    "FONDS_PROPRES_REGLEMENTAIRES": (
        "Changement lié aux fonds propres réglementaires (composition, déductions, ajustements prudentiels)."
    ),
    "EXIGENCES_REGLEMENTAIRES": (
        "Changement dans les exigences réglementaires applicables (seuils "
        "prudentiels, planchers, calendriers réglementaires, exigences pilier 2)."
    ),
    "RATIOS_REGLEMENTAIRES": (
        "Changement dans la définition, la méthode ou la cible d'un ratio "
        "réglementaire — PAS la simple variation chiffrée du ratio de la banque."
    ),
    "STRUCTURE_RAPPORT": (
        "Changement significatif dans la structure du rapport mentionné dans "
        "le texte narratif (ajout/retrait d'un tableau, d'une note de bas de "
        "tableau, réorganisation de sections)."
    ),
    "HYPOTHESES_EXPLICATIONS_RISQUES": (
        "Changement dans les hypothèses ou explications qualitatives "
        "associées aux risques (scénarios narratifs, justifications)."
    ),
    "ESG_CLIMATIQUE": (
        "Changement lié aux divulgations ESG ou climatiques (risques "
        "climatiques, transition énergétique, gouvernance ESG, ligne "
        "directrice B-15 BSIF, attentes climatiques)."
    ),
    "RISQUE_EMERGENT": (
        "Changement lié à un risque émergent — TRÈS PRIORITAIRE : "
        "cyberrisque, cybersécurité, intelligence artificielle, IA générative, "
        "modèles tiers, fraude numérique, usurpation d'identité, ransomware, "
        "attaques sur la chaîne d'approvisionnement, dépendances technologiques."
    ),
    "RISQUE_DONNEES": (
        "Changement lié au risque et à la gouvernance des données : qualité, "
        "intégrité, disponibilité, confidentialité, protection, localisation, "
        "souveraineté, conservation, traçabilité, lignage, accès, perte ou fuite "
        "de données, y compris les données utilisées par les modèles et l'IA."
    ),
    "RISQUE_TIERS_CLOUD": (
        "Changement lié aux fournisseurs, à l'impartition et aux services "
        "infonuagiques : tiers critiques, concentration, dépendance ou verrouillage "
        "fournisseur, sous-traitants, localisation des données, continuité, "
        "résilience, stratégie de sortie, surveillance et exigences contractuelles."
    ),
    "RISQUE_MACRO_GEOPOLITIQUE": (
        "Changement lié à un risque macroéconomique, commercial ou "
        "géopolitique — PRIORITAIRE : tarifs douaniers, guerre commerciale, "
        "sanctions, conflits, incertitude des politiques commerciales. C'est un "
        "déclencheur externe qui se transmet aux risques bancaires classiques "
        "(crédit, marché, financement, macroéconomie). Couvre l'ajout, le "
        "retrait ou la modification de la divulgation de ce déclencheur, qu'il "
        "soit traité comme facteur autonome ou intégré au risque de crédit/marché."
    ),
    "GOUVERNANCE_RISQUES": (
        "Changement dans la gouvernance des risques : nom ou mandat d'un "
        "comité, autorité décisionnelle, rôles, responsabilités, supervision, "
        "reddition de comptes, lignes de défense, culture de risque, "
        "rémunération liée au risque, appétit pour le risque ou cadre de "
        "gouvernance. Un renommage explicite d'un comité reste pertinent même "
        "si son mandat demeure identique; il ne constitue toutefois pas, à lui "
        "seul, une nouvelle idée substantielle."
    ),
    "CONTROLE_CONFORMITE": (
        "Changement dans les processus de contrôle interne ou de conformité "
        "(dispositifs de contrôle, audit interne, conformité réglementaire, "
        "lutte contre le blanchiment)."
    ),
    "NOUVELLE_MENTION_REGLEMENTAIRE": (
        "Nouvelle mention d'une attente réglementaire ou d'une autorité "
        "(BSIF, AMF, OSFI, BCBS, Bâle, ligne directrice, lettre prudentielle, "
        "consultation publique)."
    ),
    "MONTANT_REGLEMENTAIRE": (
        "Marqueur à combiner avec le thème principal : la divulgation porte "
        "sur un montant ou seuil RÉGLEMENTAIRE (ex : seuil prudentiel 4.5%, "
        "plancher Bâle, exigence pilier 2). NE PAS utiliser pour les chiffres "
        "propres à la banque (portefeuille, exposition, profits) — ceux-là "
        "sont exclus via 'variation_numerique_propre_banque'."
    ),
    "SUJET_EMERGENT_HORS_GRILLE": (
        "Changement substantiel pertinent pour la vigie bancaire, mais qui "
        "ne relève directement d'aucune des 20 catégories de contrôle. "
        "À utiliser seulement lorsque le sujet, son ajout/retrait ou son effet "
        "sur le risque, la gouvernance, la stratégie ou la divulgation est "
        "clairement expliqué dans la justification."
    ),
}

THEMES_AMF_ANALYST_SUBJECTS: dict[str, str] = {
    "DIVULGATION_AJOUT": "Information ajoutée",
    "DIVULGATION_RETRAIT": "Information retirée",
    "MODIFICATION_TEXTE_RISQUE": "Texte de risque modifié",
    "MODIFICATION_METHODOLOGIE": "Méthode de calcul ou approche modifiée",
    "FACTEUR_RISQUE_CHANGEMENT": "Facteur de risque modifié",
    "CAPITAL_REGLEMENTAIRE": "Capital réglementaire",
    "LIQUIDITE": "Liquidité",
    "FONDS_PROPRES_REGLEMENTAIRES": "Fonds propres réglementaires",
    "EXIGENCES_REGLEMENTAIRES": "Exigences réglementaires ou conformité",
    "RATIOS_REGLEMENTAIRES": "Ratio ou seuil prudentiel",
    "STRUCTURE_RAPPORT": "Structure du rapport",
    "HYPOTHESES_EXPLICATIONS_RISQUES": "Hypothèses ou explications de risque",
    "ESG_CLIMATIQUE": "Risque climatique / ESG",
    "RISQUE_EMERGENT": "Risque émergent : IA, cybersécurité, fraude, cryptoactifs ou modèles tiers",
    "RISQUE_DONNEES": "Risque et gouvernance des données",
    "RISQUE_TIERS_CLOUD": "Risque de tiers, fournisseurs et services infonuagiques",
    "RISQUE_MACRO_GEOPOLITIQUE": "Risque commercial et géopolitique : tarifs douaniers, sanctions, conflits",
    "GOUVERNANCE_RISQUES": "Gouvernance des risques",
    "CONTROLE_CONFORMITE": "Contrôle interne ou conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "Nouvelle mention réglementaire",
    "MONTANT_REGLEMENTAIRE": "Montant ou seuil réglementaire",
    "SUJET_EMERGENT_HORS_GRILLE": "À qualifier — sujet émergent hors grille",
}

THEMES_AMF_PIPELINE_2: list[str] = list(THEMES_AMF_DESCRIPTIONS.keys())

ThemeAMF = Literal[
    "DIVULGATION_AJOUT",
    "DIVULGATION_RETRAIT",
    "MODIFICATION_TEXTE_RISQUE",
    "MODIFICATION_METHODOLOGIE",
    "FACTEUR_RISQUE_CHANGEMENT",
    "CAPITAL_REGLEMENTAIRE",
    "LIQUIDITE",
    "FONDS_PROPRES_REGLEMENTAIRES",
    "EXIGENCES_REGLEMENTAIRES",
    "RATIOS_REGLEMENTAIRES",
    "STRUCTURE_RAPPORT",
    "HYPOTHESES_EXPLICATIONS_RISQUES",
    "ESG_CLIMATIQUE",
    "RISQUE_EMERGENT",
    "RISQUE_DONNEES",
    "RISQUE_TIERS_CLOUD",
    "RISQUE_MACRO_GEOPOLITIQUE",
    "GOUVERNANCE_RISQUES",
    "CONTROLE_CONFORMITE",
    "NOUVELLE_MENTION_REGLEMENTAIRE",
    "MONTANT_REGLEMENTAIRE",
    "SUJET_EMERGENT_HORS_GRILLE",
]

EXCLUSION_REASONS_DESCRIPTIONS: dict[str, str] = {
    "variation_numerique_propre_banque": (
        "Variation de chiffres propres à la banque (taille du portefeuille, "
        "exposition, profits, montants d'actifs) sans dimension réglementaire."
    ),
    "operation_interne_banque": (
        "Opération spécifique à la banque (acquisition, rachat, émission, "
        "dividende, fusion ou transaction d'entreprise) sans nouveau cadre "
        "réglementaire ou méthodologique à comparer entre institutions."
    ),
    "mise_a_jour_calendrier": (
        "Mise à jour de dates ou d'échéances d'application d'une exigence "
        "déjà connue, sans nouveau fond réglementaire ni nouvelle méthode."
    ),
    "reformulation_mineure": (
        "Texte reformulé sans changement de fond (synonymes, ordre des mots, tournure équivalente)."
    ),
    "deplacement_texte": (
        "Texte déplacé d'une section à une autre sans modification de contenu, "
        "de contexte, de visibilité ou de rattachement métier."
    ),
    "formatage_visuel": (
        "Changement purement visuel (gras, italique, ponctuation, casse, espacement, retour à la ligne)."
    ),
    "non_pertinent_autre": ("Autre changement non pertinent pour la vigie AMF."),
}

ExclusionReason = Literal[
    "variation_numerique_propre_banque",
    "operation_interne_banque",
    "mise_a_jour_calendrier",
    "reformulation_mineure",
    "deplacement_texte",
    "formatage_visuel",
    "non_pertinent_autre",
]

ImpactLevel = Literal["MAJEUR", "MODERE", "MINEUR"]

ImpactIT = Literal["ELEVE", "MOYEN", "FAIBLE", "INDETERMINE"]

ChangementPosture = Literal[
    "RENFORCEMENT",
    "ALLEGEMENT",
    "NOUVEAU_DISPOSITIF",
    "RETRAIT_DISPOSITIF",
    "AUCUN",
    "INDETERMINE",
]

StatutMiseEnOeuvre = Literal[
    "ANNONCE",
    "PLANIFIE",
    "EN_COURS",
    "MIS_EN_OEUVRE",
    "INDETERMINE",
]

ConfiancePosture = Literal["ELEVEE", "MOYENNE", "FAIBLE", "INDETERMINE"]

ActionRequise = Literal[
    "revue_prioritaire",
    "investigation",
    "confirmation",
    "information",
    "aucune",
]

TRIAGE_SOURCE_VERSION = "gpt4o_triage_amf_compact_v2"


ChangeSegmentKind = Literal["added", "removed", "modified"]


class ChangeSegment(BaseModel):
    """Segment substantiel identifié par GPT comme différent entre T1 et T2.

    Citations VERBATIM depuis le texte source — ne pas paraphraser.
    Le rendu Dash recherche ces segments par ``str.find()`` pour surligner
    les portions correspondantes dans la vue side-by-side.

    Invariants :
    - ``kind="added"``    → ``text_t1=""``, ``text_t2`` non vide
    - ``kind="removed"``  → ``text_t1`` non vide, ``text_t2=""``
    - ``kind="modified"`` → ``text_t1`` non vide ET ``text_t2`` non vide
    """

    kind: ChangeSegmentKind
    text_t1: str = ""
    text_t2: str = ""

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> "ChangeSegment":
        """Vérifie que ``text_t1`` / ``text_t2`` sont cohérents avec ``kind``."""
        if self.kind == "added":
            if self.text_t1.strip():
                raise ValueError("kind='added' interdit text_t1 non vide")
            if not self.text_t2.strip():
                raise ValueError("kind='added' exige text_t2 non vide")
        elif self.kind == "removed":
            if not self.text_t1.strip():
                raise ValueError("kind='removed' exige text_t1 non vide")
            if self.text_t2.strip():
                raise ValueError("kind='removed' interdit text_t2 non vide")
        else:  # modified
            if not self.text_t1.strip():
                raise ValueError("kind='modified' exige text_t1 non vide")
            if not self.text_t2.strip():
                raise ValueError("kind='modified' exige text_t2 non vide")
        return self


_EXPLANATION_MIN_LENGTH = 50
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

IMPACT_IT_DETAIL_LABELS = (
    "Éléments observés",
    "Conséquence probable",
    "Limite de l'analyse",
)

POSTURE_DETAIL_LABELS = (
    "Preuve",
    "Effet sur la gestion du risque",
    "Justification du statut",
    "Justification de la confiance",
)


def extract_labeled_analysis(
    text: str,
    labels: tuple[str, ...],
) -> dict[str, str]:
    """Extrait des rubriques ``Libellé : contenu`` dans leur ordre attendu."""
    if not text:
        return {}

    cleaned = text.strip()
    label_pattern = "|".join(re.escape(label) for label in labels)
    matches = list(
        re.finditer(
            rf"(?m)^(?P<label>{label_pattern})\s*:\s*",
            cleaned,
        )
    )
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        sections[match.group("label")] = cleaned[match.end() : end].strip()
    return sections


def missing_labeled_analysis_sections(
    text: str,
    labels: tuple[str, ...],
    *,
    min_content_length: int = 25,
) -> list[str]:
    """Retourne les rubriques absentes ou sans contenu analytique suffisant."""
    sections = extract_labeled_analysis(text, labels)
    return [label for label in labels if len(sections.get(label, "").strip()) < min_content_length]


def _count_substantive_sentences(text: str) -> int:
    """Compte les phrases complètes dans ``text``.

    Une phrase est dite « substantive » si elle se termine par un point, un
    point d'exclamation ou un point d'interrogation, et contient au moins
    ``_JUSTIFICATION_MIN_SENTENCE_LENGTH`` caractères significatifs (hors
    espaces). Empêche un texte trivial comme « OUI. » de passer pour deux
    phrases.
    """
    if not text:
        return 0
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return sum(1 for part in parts if len(part.strip()) >= _JUSTIFICATION_MIN_SENTENCE_LENGTH)


def _missing_justification_sections(text: str) -> list[str]:
    """Retourne les rubriques obligatoires absentes de la justification."""
    missing: list[str] = []
    for section in _REQUIRED_JUSTIFICATION_SECTIONS:
        if section in text:
            continue
        if section == "Point de surveillance :" and _LEGACY_SURVEILLANCE_SECTION in text:
            continue
        missing.append(section)
    return missing


class _TriageAMFResultBase(BaseModel):
    """Sortie validée d'un triage GPT-4o pour un changement.

    Invariants garantis (toute violation lève ``pydantic.ValidationError``).

    **Cohérence pertinent / non pertinent**

    * ``is_relevant=True`` implique ``themes_amf`` non vide, ``exclusion_reason=None``, ``explanation`` ≥ 50 caractères (3 phrases attendues), ``nouvelle_idee_justification`` ≥ 3 phrases complètes commençant par ``OUI`` ou ``NON`` selon ``nouvelle_idee``.
    * ``is_relevant=False`` implique ``themes_amf=[]``, ``exclusion_reason`` renseigné, ``nouvelle_idee=False``, ``impact_level="MINEUR"``, ``impact_it="INDETERMINE"``, ``changement_posture="AUCUN"``, ``action_requise="aucune"``, ``explanation=""``, ``nouvelle_idee_justification`` détaillée.

    **Cohérence sémantique**

    * ``action_requise="revue_prioritaire"`` implique ``impact_level="MAJEUR"``.

    Pas de fallback silencieux : un triage invalide doit remonter en exception
    et être traité explicitement par l'appelant.
    """

    is_relevant: bool
    themes_amf: list[ThemeAMF] = Field(default_factory=list)
    impact_level: ImpactLevel = "MINEUR"
    impact_it: ImpactIT = "INDETERMINE"
    impact_it_justification: str = ""
    changement_posture: ChangementPosture = "INDETERMINE"
    justification_posture: str = ""
    statut_mise_en_oeuvre: StatutMiseEnOeuvre = "INDETERMINE"
    confiance_posture: ConfiancePosture = "INDETERMINE"
    nouvelle_idee: bool = False
    explanation: str = ""
    nouvelle_idee_justification: str = ""
    action_requise: ActionRequise = "aucune"
    exclusion_reason: ExclusionReason | None = None

    @field_validator("themes_amf")
    @classmethod
    def _dedupe_themes(cls, value: list[str]) -> list[str]:
        """Supprime les doublons de la liste de thèmes AMF en préservant l'ordre."""
        seen: set[str] = set()
        out: list[str] = []
        for theme in value:
            if theme not in seen:
                seen.add(theme)
                out.append(theme)
        return out

    @model_validator(mode="before")
    @classmethod
    def _repair_justification_if_needed(cls, data: object) -> object:
        """Reconstruit la justification si GPT omet les rubriques obligatoires."""
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        justification = str(normalized.get("nouvelle_idee_justification") or "").strip()
        from vigie.analyse_texte.text_comparison.justification import (
            is_structured_text_triage_justification,
            synthesize_triage_justification_from_payload,
        )

        if justification and is_structured_text_triage_justification(justification):
            return normalized

        normalized["nouvelle_idee_justification"] = synthesize_triage_justification_from_payload(normalized)
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _default_irrelevant_extended_signals(cls, data: object) -> object:
        """Préserve la compatibilité des anciens payloads non pertinents."""
        if not isinstance(data, dict) or data.get("is_relevant") is not False:
            return data

        normalized = dict(data)
        normalized.setdefault("impact_it", "INDETERMINE")
        normalized.setdefault("impact_it_justification", "")
        normalized.setdefault("changement_posture", "AUCUN")
        normalized.setdefault("justification_posture", "")
        normalized.setdefault("statut_mise_en_oeuvre", "INDETERMINE")
        normalized.setdefault("confiance_posture", "INDETERMINE")
        return normalized

    @model_validator(mode="after")
    def _check_invariants(self) -> "_TriageAMFResultBase":
        """Garantit la cohérence métier de la sortie GPT (cf. docstring de classe)."""
        # ---------- Justification : OBLIGATOIRE et SUBSTANTIELLE ----------
        # Quel que soit ``is_relevant`` (Oui ou Non), l'analyste a besoin
        # d'une explication détaillée pour comprendre la décision GPT.
        justification = self.nouvelle_idee_justification.strip()
        if _count_substantive_sentences(justification) < _JUSTIFICATION_MIN_SENTENCES:
            raise ValueError(
                f"nouvelle_idee_justification exige au moins "
                f"{_JUSTIFICATION_MIN_SENTENCES} phrases complètes "
                f"de {_JUSTIFICATION_MIN_SENTENCE_LENGTH}+ caractères chacune"
            )
        if len(justification) < _JUSTIFICATION_MIN_TOTAL_LENGTH:
            raise ValueError(
                f"nouvelle_idee_justification exige au moins {_JUSTIFICATION_MIN_TOTAL_LENGTH} caractères au total"
            )
        expected_prefix = "OUI" if self.nouvelle_idee else "NON"
        if not justification.upper().startswith(expected_prefix):
            raise ValueError(
                "nouvelle_idee_justification doit commencer par "
                f"'{expected_prefix}' quand nouvelle_idee={self.nouvelle_idee}"
            )
        missing_sections = _missing_justification_sections(justification)
        if missing_sections:
            raise ValueError(
                f"nouvelle_idee_justification doit contenir les rubriques obligatoires : {', '.join(missing_sections)}"
            )

        # ---------- Cohérence pertinent / non pertinent ----------
        if self.is_relevant:
            if not self.themes_amf:
                raise ValueError("is_relevant=True exige au moins un thème AMF dans themes_amf")
            if self.exclusion_reason is not None:
                raise ValueError("is_relevant=True interdit exclusion_reason renseigné")
            if len(self.explanation.strip()) < _EXPLANATION_MIN_LENGTH:
                raise ValueError(
                    "is_relevant=True exige une explanation d'au moins "
                    f"{_EXPLANATION_MIN_LENGTH} caractères (3 phrases attendues)"
                )
        else:
            if self.themes_amf:
                raise ValueError("is_relevant=False interdit themes_amf non vide")
            if self.exclusion_reason is None:
                raise ValueError("is_relevant=False exige exclusion_reason renseigné")
            if self.nouvelle_idee:
                raise ValueError("is_relevant=False interdit nouvelle_idee=True")
            if self.impact_level != "MINEUR":
                raise ValueError("is_relevant=False exige impact_level=MINEUR")
            if self.impact_it != "INDETERMINE":
                raise ValueError("is_relevant=False exige impact_it=INDETERMINE")
            if self.impact_it_justification.strip():
                raise ValueError("is_relevant=False exige impact_it_justification vide")
            if self.changement_posture != "AUCUN":
                raise ValueError("is_relevant=False exige changement_posture=AUCUN")
            if self.justification_posture.strip():
                raise ValueError("is_relevant=False exige justification_posture vide")
            if self.statut_mise_en_oeuvre != "INDETERMINE":
                raise ValueError("is_relevant=False exige statut_mise_en_oeuvre=INDETERMINE")
            if self.confiance_posture != "INDETERMINE":
                raise ValueError("is_relevant=False exige confiance_posture=INDETERMINE")
            if self.action_requise != "aucune":
                raise ValueError("is_relevant=False exige action_requise='aucune'")
            if self.explanation.strip():
                raise ValueError("is_relevant=False exige explanation vide")
        if self.action_requise == "revue_prioritaire" and self.impact_level != "MAJEUR":
            raise ValueError("action_requise='revue_prioritaire' exige impact_level='MAJEUR'")
        if self.impact_it != "INDETERMINE" and len(self.impact_it_justification.strip()) < 20:
            raise ValueError("impact_it_justification exige au moins 20 caractères quand impact_it est évalué")
        if self.impact_it != "INDETERMINE":
            missing_impact_sections = missing_labeled_analysis_sections(
                self.impact_it_justification,
                IMPACT_IT_DETAIL_LABELS,
            )
            if missing_impact_sections:
                raise ValueError(
                    "impact_it_justification doit contenir les rubriques "
                    f"détaillées : {', '.join(missing_impact_sections)}"
                )
        if self.impact_it == "INDETERMINE" and self.impact_it_justification.strip():
            raise ValueError("impact_it_justification doit être vide quand impact_it=INDETERMINE")
        posture_evaluee = self.changement_posture in {
            "RENFORCEMENT",
            "ALLEGEMENT",
            "NOUVEAU_DISPOSITIF",
            "RETRAIT_DISPOSITIF",
        }
        if posture_evaluee and len(self.justification_posture.strip()) < 20:
            raise ValueError(
                "justification_posture exige au moins 20 caractères quand un changement de posture est évalué"
            )
        if posture_evaluee:
            missing_posture_sections = missing_labeled_analysis_sections(
                self.justification_posture,
                POSTURE_DETAIL_LABELS,
            )
            if missing_posture_sections:
                raise ValueError(
                    "justification_posture doit contenir les rubriques "
                    f"détaillées : {', '.join(missing_posture_sections)}"
                )
        if posture_evaluee and self.confiance_posture == "INDETERMINE":
            raise ValueError("confiance_posture doit être évaluée quand un changement de posture est identifié")

        return self


class TriageAMFResult(_TriageAMFResultBase):
    """Triage AMF complet persisté dans les artefacts texte.

    ``change_segments`` est rattaché hors LLM par le pipeline, afin que les
    preuves verbatim restent déterministes et auditables.
    """

    change_segments: list[ChangeSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_change_segments_for_irrelevant(self) -> "TriageAMFResult":
        """Un changement non pertinent ne doit pas afficher de surlignage."""
        if not self.is_relevant and self.change_segments:
            raise ValueError("is_relevant=False exige change_segments vide")
        return self


class TriageAMFResultWithIndex(TriageAMFResult):
    """Triage AMF accompagné de l'index du changement dans la section.

    Utilisé comme item du batch retourné par GPT-4o via les structured outputs.
    L'index est 1-based pour rester aligné avec l'énumération transmise au
    modèle dans le prompt et permettre un mapping robuste vers les changements
    sources, indépendamment de l'ordre de la liste retournée.
    """

    change_index: int = Field(..., ge=1)


class TriageAMFLLMResultWithIndex(_TriageAMFResultBase):
    """Triage AMF demandé au LLM, sans preuve de surlignage.

    Les segments verbatim sont calculés localement depuis les textes T1/T2 pour
    éviter de mélanger jugement métier et extraction mécanique de preuve.
    """

    change_index: int = Field(..., ge=1)


class TriageAMFBatch(BaseModel):
    """Lot de triages AMF retourné par GPT-4o pour une section donnée.

    Schéma racine passé à ``client.beta.chat.completions.parse()`` comme
    ``response_format`` ; OpenAI garantit alors que la sortie respecte les
    types et énumérations du schéma. Les invariants logiques transversaux
    (cohérence is_relevant, revue_prioritaire↔MAJEUR, ...) restent vérifiés par
    Pydantic après désérialisation.
    """

    triages: list[TriageAMFResultWithIndex]


class TriageAMFLLMBatch(BaseModel):
    """Lot de triages AMF retourné par le LLM sans ``change_segments``."""

    triages: list[TriageAMFLLMResultWithIndex]


COMPACT_RELEVANT_REASON_SENTENCE_COUNT = 4
COMPACT_SECONDARY_REASON_SENTENCE_COUNT = 2
_COMPACT_SENTENCE_END_RE = re.compile(r"(?P<mark>[.!?]+)(?P<closers>[\u00bb\u201d\"')\]]*)(?=\s+|$)")
_COMPACT_NON_TERMINAL_ABBREVIATIONS = frozenset(
    {
        "art",
        "dr",
        "dre",
        "m",
        "mlle",
        "mme",
        "mmes",
        "no",
        "nos",
        "p",
        "pp",
        "pr",
        "prof",
    }
)


def _is_non_terminal_compact_period(value: str, match: re.Match[str]) -> bool:
    """Distingue une abréviation interne d'une véritable fin de phrase."""
    if match.group("mark") != "." or match.end() == len(value):
        return False
    prefix = value[: match.start()].rstrip()
    suffix = value[match.end() :].lstrip()
    token_match = re.search(r"([\wÀ-ÖØ-öø-ÿ-]+)$", prefix)
    token = token_match.group(1) if token_match else ""
    if token.casefold() in _COMPACT_NON_TERMINAL_ABBREVIATIONS:
        return True
    folded_prefix = prefix.casefold()
    folded_suffix = suffix.casefold()
    # ``s. o.`` / ``s.o.`` signifie « sans objet » dans les rapports
    # bancaires. Protéger les deux points afin qu'ils ne soient pas comptés
    # comme des fins de phrase.
    if re.search(r"(?:^|[^\w])s$", folded_prefix) and re.match(r"o\.(?:\s|$)", folded_suffix):
        return True
    if re.search(r"(?:^|[^\w])s\.\s*o$", folded_prefix):
        return True
    if folded_prefix.endswith("p. ex"):
        return True
    if folded_prefix.endswith(("c.-à-d", "c.-a-d")):
        return bool(suffix and suffix[0].islower())
    if token.casefold() == "etc":
        return bool(suffix and suffix[0].islower())
    # Une suite d'initiales telle que ``N.A.`` est une abréviation interne
    # seulement lorsque la phrase continue ensuite. Une majuscule après
    # l'abréviation signale généralement une nouvelle phrase
    # (``BMO Bank N.A. Cette reformulation...``).
    return bool(re.search(r"(?:\b[A-ZÀ-ÖØ-Þ]\.)+[A-ZÀ-ÖØ-Þ]$", prefix) and suffix and suffix[0].islower())


def _compact_complete_sentence_parts(value: str) -> list[str]:
    """Retourne les phrases terminées en conservant les abréviations internes."""
    normalized = " ".join(str(value or "").split())
    parts: list[str] = []
    start = 0
    for match in _COMPACT_SENTENCE_END_RE.finditer(normalized):
        if _is_non_terminal_compact_period(normalized, match):
            continue
        parts.append(normalized[start : match.end()].strip())
        start = match.end()
    return parts


def count_complete_sentences(value: str) -> int:
    """Compte les phrases terminées qui contiennent du contenu lexical."""
    return sum(1 for part in _compact_complete_sentence_parts(value) if re.search(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]", part))


def _has_complete_sentence_ending(value: str) -> bool:
    parts = _compact_complete_sentence_parts(value)
    return bool(parts) and value.endswith(parts[-1])


class TriageAMFCompactLLMResultWithIndex(BaseModel):
    """Décision AMF compacte avec unités analystes explicitement structurées.

    ``relevance_reason`` demeure disponible comme propriété assemblée pour les
    consommateurs historiques. Il n'est toutefois plus demandé au LLM et aucun
    invariant ne dépend du comptage de phrases dans un paragraphe fusionné.
    """

    change_index: int = Field(..., ge=1)
    is_relevant: bool
    themes_amf: list[ThemeAMF] = Field(default_factory=list, max_length=2)
    nouvelle_idee: bool = False
    changement_constate: str = Field(
        description=(
            "Constat factuel autonome, commençant par le nom canonique de la "
            "banque et décrivant directement ce qu'elle ajoute, retire, modifie "
            "ou précise."
        )
    )
    signification_metier: str = Field(
        description=("Signification métier du changement pertinent; chaîne vide lorsque is_relevant=false.")
    )
    comparaison_interbanques: str = Field(
        description=(
            "Dimensions concrètes que le changement permet de comparer entre "
            "banques; chaîne vide lorsque is_relevant=false."
        )
    )
    limite_interpretation: str = Field(
        description=(
            "Limite d'interprétation étayée par les éléments non démontrés ou "
            "non précisés; chaîne vide lorsque is_relevant=false."
        )
    )
    motif_non_pertinence: str = Field(
        description=("Motif métier expliquant la non-pertinence; chaîne vide lorsque is_relevant=true.")
    )

    @field_validator("themes_amf")
    @classmethod
    def _dedupe_compact_themes(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_relevance_reason(cls, data: object) -> object:
        """Accepte encore un ancien payload monolithique à la construction.

        Cette voie sert uniquement à la lecture d'artefacts ou aux appelants
        historiques. Le schéma exposé au LLM exige les cinq champs structurés.
        """
        if not isinstance(data, dict):
            return data
        semantic_fields = (
            "changement_constate",
            "signification_metier",
            "comparaison_interbanques",
            "limite_interpretation",
            "motif_non_pertinence",
        )
        if any(field in data for field in semantic_fields):
            return data
        legacy_reason = " ".join(str(data.get("relevance_reason") or "").split())
        if not legacy_reason:
            return data
        if not _has_complete_sentence_ending(legacy_reason):
            legacy_reason = legacy_reason.rstrip(" ,;:…") + "."

        parts = _compact_complete_sentence_parts(legacy_reason)
        migrated = dict(data)
        is_relevant = bool(data.get("is_relevant"))
        if is_relevant:
            migrated.update(
                {
                    "changement_constate": parts[0] if len(parts) >= 1 else "",
                    "signification_metier": parts[1] if len(parts) >= 2 else "",
                    "comparaison_interbanques": parts[2] if len(parts) >= 3 else "",
                    "limite_interpretation": (" ".join(parts[3:]) if len(parts) >= 4 else ""),
                    "motif_non_pertinence": "",
                }
            )
        else:
            migrated.update(
                {
                    "changement_constate": parts[0] if parts else "",
                    "signification_metier": "",
                    "comparaison_interbanques": "",
                    "limite_interpretation": "",
                    "motif_non_pertinence": (" ".join(parts[1:]) if len(parts) >= 2 else ""),
                }
            )
        return migrated

    @field_validator(
        "changement_constate",
        "signification_metier",
        "comparaison_interbanques",
        "limite_interpretation",
        "motif_non_pertinence",
        mode="before",
    )
    @classmethod
    def _normalize_semantic_field(cls, value: object) -> str:
        normalized = " ".join(str(value or "").split())
        if not normalized:
            return ""
        if not re.search(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]", normalized):
            raise ValueError("chaque champ renseigné doit contenir du contenu lexical")
        if not _has_complete_sentence_ending(normalized):
            normalized = normalized.rstrip(" ,;:…") + "."
        return normalized

    @model_validator(mode="after")
    def _check_compact_invariants(self) -> "TriageAMFCompactLLMResultWithIndex":
        if not self.changement_constate:
            raise ValueError("changement_constate est obligatoire")

        if self.is_relevant:
            if not self.themes_amf:
                raise ValueError("is_relevant=True exige au moins un thème AMF dans themes_amf")
            required_fields = {
                "signification_metier": self.signification_metier,
                "comparaison_interbanques": self.comparaison_interbanques,
                "limite_interpretation": self.limite_interpretation,
            }
            missing_fields = [field for field, value in required_fields.items() if not value]
            if missing_fields:
                raise ValueError("is_relevant=True exige les champs non vides suivants : " + ", ".join(missing_fields))
            if self.motif_non_pertinence:
                raise ValueError("is_relevant=True exige motif_non_pertinence vide")
        else:
            if self.themes_amf:
                raise ValueError("is_relevant=False interdit themes_amf non vide")
            if self.nouvelle_idee:
                raise ValueError("is_relevant=False interdit nouvelle_idee=True")
            forbidden_fields = {
                "signification_metier": self.signification_metier,
                "comparaison_interbanques": self.comparaison_interbanques,
                "limite_interpretation": self.limite_interpretation,
            }
            populated_fields = [field for field, value in forbidden_fields.items() if value]
            if populated_fields:
                raise ValueError("is_relevant=False exige les champs vides suivants : " + ", ".join(populated_fields))
            if not self.motif_non_pertinence:
                raise ValueError("is_relevant=False exige motif_non_pertinence non vide")
        return self

    @property
    def relevance_reason(self) -> str:
        """Assemble le texte historique sans analyser sa ponctuation interne."""
        if self.is_relevant:
            parts = (
                self.changement_constate,
                self.signification_metier,
                self.comparaison_interbanques,
                self.limite_interpretation,
            )
        else:
            parts = (self.changement_constate, self.motif_non_pertinence)
        return " ".join(part for part in parts if part)


class TriageAMFCompactLLMBatch(BaseModel):
    """Lot compact : une décision courte pour chaque changement demandé."""

    triages: list[TriageAMFCompactLLMResultWithIndex]


def format_themes_for_prompt() -> str:
    """Formate la taxonomie AMF pour injection dans le prompt GPT-4o."""
    return "\n".join(f"- {code} : {description}" for code, description in THEMES_AMF_DESCRIPTIONS.items())


def format_theme_subjects_for_prompt() -> str:
    """Formate les libellés analyste associés aux codes AMF."""
    return "\n".join(f"- {code} -> {subject}" for code, subject in THEMES_AMF_ANALYST_SUBJECTS.items())


def empty_triage_skeleton() -> dict:
    """Squelette de triage non pertinent pour un changement absent du batch GPT.

    Utilisé uniquement quand GPT n'a pas produit de triage pour un
    ``change_index`` particulier (cas rare, défaut structurel — PAS un
    fallback d'erreur de validation). Toute erreur de parsing ou
    d'invariant doit remonter en exception explicite via
    ``TriageValidationError``.
    """
    return TriageAMFResult(
        is_relevant=False,
        themes_amf=[],
        impact_level="MINEUR",
        impact_it="INDETERMINE",
        impact_it_justification="",
        changement_posture="AUCUN",
        justification_posture="",
        statut_mise_en_oeuvre="INDETERMINE",
        confiance_posture="INDETERMINE",
        nouvelle_idee=False,
        explanation="",
        nouvelle_idee_justification=(
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
        action_requise="aucune",
        exclusion_reason="non_pertinent_autre",
    ).model_dump()


class TriageValidationError(RuntimeError):
    """Levée quand un triage GPT viole les invariants de schéma ou métier.

    Contient le contexte d'audit nécessaire pour qu'un humain (ou un retry
    automatisé avec feedback correctif) puisse comprendre et corriger
    l'erreur sans avoir à inspecter les logs bruts.
    """

    def __init__(
        self,
        *,
        section_key: str,
        change_index: int | None,
        raw_payload: object,
        validation_error: Exception,
    ) -> None:
        """Initialise l'erreur avec le contexte de triage non conforme.

        Args:
            section_key: Section métier dans laquelle le triage a échoué.
            change_index: Indice du changement concerné, si disponible.
            raw_payload: Réponse brute reçue avant validation.
            validation_error: Erreur de validation à l'origine du rejet.
        """
        message = f"Triage AMF invalide [section={section_key}, change_index={change_index}] : {validation_error}"
        super().__init__(message)
        self.section_key = section_key
        self.change_index = change_index
        self.raw_payload = raw_payload
        self.validation_error = validation_error
