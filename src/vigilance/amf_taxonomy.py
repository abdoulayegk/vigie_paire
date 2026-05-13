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
    "DIVULGATION_AJOUT": (
        "Ajout d'une nouvelle divulgation absente du rapport précédent."
    ),
    "DIVULGATION_RETRAIT": (
        "Retrait d'une divulgation présente au rapport précédent."
    ),
    "MODIFICATION_TEXTE_RISQUE": (
        "Modification substantielle d'un texte décrivant un risque "
        "(hors simple variation chiffrée propre à la banque)."
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
        "Changement lié aux fonds propres réglementaires (composition, "
        "déductions, ajustements prudentiels)."
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
    "GOUVERNANCE_RISQUES": (
        "Changement dans la gouvernance des risques (comités, rôles, "
        "responsabilités, appétit pour le risque, cadre de gouvernance)."
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
    "GOUVERNANCE_RISQUES": "Gouvernance des risques",
    "CONTROLE_CONFORMITE": "Contrôle interne ou conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "Nouvelle mention réglementaire",
    "MONTANT_REGLEMENTAIRE": "Montant ou seuil réglementaire",
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
    "GOUVERNANCE_RISQUES",
    "CONTROLE_CONFORMITE",
    "NOUVELLE_MENTION_REGLEMENTAIRE",
    "MONTANT_REGLEMENTAIRE",
]

EXCLUSION_REASONS_DESCRIPTIONS: dict[str, str] = {
    "variation_numerique_propre_banque": (
        "Variation de chiffres propres à la banque (taille du portefeuille, "
        "exposition, profits, montants d'actifs) sans dimension réglementaire."
    ),
    "reformulation_mineure": (
        "Texte reformulé sans changement de fond (synonymes, ordre des mots, "
        "tournure équivalente)."
    ),
    "deplacement_texte": (
        "Texte déplacé d'une section à une autre sans modification de contenu."
    ),
    "formatage_visuel": (
        "Changement purement visuel (gras, italique, ponctuation, casse, "
        "espacement, retour à la ligne)."
    ),
    "non_pertinent_autre": (
        "Autre changement non pertinent pour la vigie AMF."
    ),
}

ExclusionReason = Literal[
    "variation_numerique_propre_banque",
    "reformulation_mineure",
    "deplacement_texte",
    "formatage_visuel",
    "non_pertinent_autre",
]

ImpactLevel = Literal["MAJEUR", "MODERE", "MINEUR"]

ActionRequise = Literal[
    "revue_prioritaire",
    "investigation",
    "confirmation",
    "information",
    "aucune",
]

TRIAGE_SOURCE_VERSION = "gpt4o_triage_amf_v2"


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
    return sum(
        1 for part in parts if len(part.strip()) >= _JUSTIFICATION_MIN_SENTENCE_LENGTH
    )


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


class TriageAMFResult(BaseModel):
    """Sortie validée d'un triage GPT-4o pour un changement.

    Invariants garantis (toute violation lève ``pydantic.ValidationError``).

    **Cohérence pertinent / non pertinent**

    * ``is_relevant=True`` implique ``themes_amf`` non vide, ``exclusion_reason=None``, ``explanation`` ≥ 50 caractères (3 phrases attendues), ``nouvelle_idee_justification`` ≥ 3 phrases complètes commençant par ``OUI`` ou ``NON`` selon ``nouvelle_idee``.
    * ``is_relevant=False`` implique ``themes_amf=[]``, ``exclusion_reason`` renseigné, ``nouvelle_idee=False``, ``impact_level="MINEUR"``, ``action_requise="aucune"``, ``explanation=""``, ``nouvelle_idee_justification`` détaillée.

    **Cohérence sémantique**

    * ``action_requise="revue_prioritaire"`` implique ``impact_level="MAJEUR"``.

    Pas de fallback silencieux : un triage invalide doit remonter en exception
    et être traité explicitement par l'appelant.
    """

    is_relevant: bool
    themes_amf: list[ThemeAMF] = Field(default_factory=list)
    impact_level: ImpactLevel = "MINEUR"
    nouvelle_idee: bool = False
    explanation: str = ""
    nouvelle_idee_justification: str = ""
    action_requise: ActionRequise = "aucune"
    exclusion_reason: ExclusionReason | None = None
    change_segments: list[ChangeSegment] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def _check_invariants(self) -> "TriageAMFResult":
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
                f"nouvelle_idee_justification exige au moins "
                f"{_JUSTIFICATION_MIN_TOTAL_LENGTH} caractères au total"
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
                "nouvelle_idee_justification doit contenir les rubriques "
                f"obligatoires : {', '.join(missing_sections)}"
            )

        # ---------- Cohérence pertinent / non pertinent ----------
        if self.is_relevant:
            if not self.themes_amf:
                raise ValueError(
                    "is_relevant=True exige au moins un thème AMF dans themes_amf"
                )
            if self.exclusion_reason is not None:
                raise ValueError(
                    "is_relevant=True interdit exclusion_reason renseigné"
                )
            if len(self.explanation.strip()) < _EXPLANATION_MIN_LENGTH:
                raise ValueError(
                    "is_relevant=True exige une explanation d'au moins "
                    f"{_EXPLANATION_MIN_LENGTH} caractères (3 phrases attendues)"
                )
        else:
            if self.themes_amf:
                raise ValueError(
                    "is_relevant=False interdit themes_amf non vide"
                )
            if self.exclusion_reason is None:
                raise ValueError(
                    "is_relevant=False exige exclusion_reason renseigné"
                )
            if self.nouvelle_idee:
                raise ValueError(
                    "is_relevant=False interdit nouvelle_idee=True"
                )
            if self.impact_level != "MINEUR":
                raise ValueError(
                    "is_relevant=False exige impact_level=MINEUR"
                )
            if self.action_requise != "aucune":
                raise ValueError(
                    "is_relevant=False exige action_requise='aucune'"
                )
            if self.explanation.strip():
                raise ValueError(
                    "is_relevant=False exige explanation vide"
                )
            if self.change_segments:
                raise ValueError(
                    "is_relevant=False exige change_segments vide"
                )

        if self.action_requise == "revue_prioritaire" and self.impact_level != "MAJEUR":
            raise ValueError(
                "action_requise='revue_prioritaire' exige impact_level='MAJEUR'"
            )

        return self


class TriageAMFResultWithIndex(TriageAMFResult):
    """Triage AMF accompagné de l'index du changement dans la section.

    Utilisé comme item du batch retourné par GPT-4o via les structured outputs.
    L'index est 1-based pour rester aligné avec l'énumération transmise au
    modèle dans le prompt et permettre un mapping robuste vers les changements
    sources, indépendamment de l'ordre de la liste retournée.
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


def format_themes_for_prompt() -> str:
    """Formate la taxonomie AMF pour injection dans le prompt GPT-4o."""
    return "\n".join(
        f"- {code} : {description}"
        for code, description in THEMES_AMF_DESCRIPTIONS.items()
    )


def format_theme_subjects_for_prompt() -> str:
    """Formate les libellés analyste associés aux codes AMF."""
    return "\n".join(
        f"- {code} -> {subject}"
        for code, subject in THEMES_AMF_ANALYST_SUBJECTS.items()
    )


def format_exclusion_reasons_for_prompt() -> str:
    """Formate les raisons d'exclusion pour injection dans le prompt GPT-4o."""
    return "\n".join(
        f"- {code} : {description}"
        for code, description in EXCLUSION_REASONS_DESCRIPTIONS.items()
    )


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
        message = (
            f"Triage AMF invalide [section={section_key}, "
            f"change_index={change_index}] : {validation_error}"
        )
        super().__init__(message)
        self.section_key = section_key
        self.change_index = change_index
        self.raw_payload = raw_payload
        self.validation_error = validation_error
