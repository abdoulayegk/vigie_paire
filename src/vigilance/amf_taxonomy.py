"""Taxonomie AMF pour le triage GPT-4o de la vigie bancaire canadienne.

Ce module centralise la liste des thèmes AMF en scope pour la Pipeline 2
(texte narratif), les raisons d'exclusion explicites, ainsi que le modèle
Pydantic qui valide la sortie GPT.

La référence métier est la spec verrouillée pour la vigie de pairs canadienne
(alignée AMF, comparabilité des divulgations). Toute évolution de la
taxonomie doit passer par ce fichier — le prompt GPT et les exports en aval
en dérivent.

Conventions :
- ``t1`` désigne le rapport précédent dans la paire comparée (avant)
- ``t2`` désigne le rapport courant dans la paire comparée (après)
- Les paires possibles sont : T2 vs T1, T3 vs T2, T1 N+1 vs T3 N (passage
  d'année), T4 N+1 vs T4 N (annuel sur annuel).
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
    "escalade",
    "investigation",
    "confirmation",
    "information",
    "aucune",
]

TRIAGE_SOURCE_VERSION = "gpt4o_triage_amf_v2"


_EXPLANATION_MIN_LENGTH = 50
_JUSTIFICATION_MIN_SENTENCES = 2
_JUSTIFICATION_MIN_SENTENCE_LENGTH = 15
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+")


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


class TriageAMFResult(BaseModel):
    """Sortie validée d'un triage GPT-4o pour un changement.

    Invariants garantis (toute violation lève ``pydantic.ValidationError``) :

    Cohérence pertinent / non pertinent :
    - ``is_relevant=True`` ⟹ ``themes_amf`` non vide, ``exclusion_reason=None``,
      ``explanation`` ≥ 50 caractères (3 phrases attendues),
      ``nouvelle_idee_justification`` ≥ 2 phrases complètes commençant par
      ``OUI`` ou ``NON`` selon ``nouvelle_idee``.
    - ``is_relevant=False`` ⟹ ``themes_amf=[]``, ``exclusion_reason`` renseigné,
      ``nouvelle_idee=False``, ``impact_level="MINEUR"``, ``action_requise="aucune"``,
      ``explanation=""``, ``nouvelle_idee_justification=""``.

    Cohérence sémantique :
    - ``action_requise="escalade"`` ⟹ ``impact_level="MAJEUR"``.

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

    @field_validator("themes_amf")
    @classmethod
    def _dedupe_themes(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for theme in value:
            if theme not in seen:
                seen.add(theme)
                out.append(theme)
        return out

    @model_validator(mode="after")
    def _check_invariants(self) -> "TriageAMFResult":
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
            justification = self.nouvelle_idee_justification.strip()
            if _count_substantive_sentences(justification) < _JUSTIFICATION_MIN_SENTENCES:
                raise ValueError(
                    "is_relevant=True exige nouvelle_idee_justification d'au "
                    f"moins {_JUSTIFICATION_MIN_SENTENCES} phrases complètes "
                    f"de {_JUSTIFICATION_MIN_SENTENCE_LENGTH}+ caractères chacune"
                )
            expected_prefix = "OUI" if self.nouvelle_idee else "NON"
            if not justification.upper().startswith(expected_prefix):
                raise ValueError(
                    "nouvelle_idee_justification doit commencer par "
                    f"'{expected_prefix}' quand nouvelle_idee={self.nouvelle_idee}"
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
            if self.nouvelle_idee_justification.strip():
                raise ValueError(
                    "is_relevant=False exige nouvelle_idee_justification vide"
                )

        if self.action_requise == "escalade" and self.impact_level != "MAJEUR":
            raise ValueError(
                "action_requise='escalade' exige impact_level='MAJEUR'"
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
    (cohérence is_relevant, escalade↔MAJEUR, ...) restent vérifiés par
    Pydantic après désérialisation.
    """

    triages: list[TriageAMFResultWithIndex]


def format_themes_for_prompt() -> str:
    """Formate la taxonomie AMF pour injection dans le prompt GPT-4o."""
    return "\n".join(
        f"- {code} : {description}"
        for code, description in THEMES_AMF_DESCRIPTIONS.items()
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
        message = (
            f"Triage AMF invalide [section={section_key}, "
            f"change_index={change_index}] : {validation_error}"
        )
        super().__init__(message)
        self.section_key = section_key
        self.change_index = change_index
        self.raw_payload = raw_payload
        self.validation_error = validation_error
