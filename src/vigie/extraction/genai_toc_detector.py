"""Lectures Vision de la table des matières annuelle et des transitions de chapitre.

Consommé par ``AnnualSectionBoundaryValidator`` pour confirmer les bornes des
rapports T4. Les réponses passent exclusivement par des sorties structurées
Pydantic : aucun JSON n'est parsé à la main.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import openai
import pdfplumber
from pydantic import BaseModel, ConfigDict, Field

from vigie.support.utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)

_T_StructuredModel = TypeVar("_T_StructuredModel", bound=BaseModel)


class AnnualTocEntryLLM(BaseModel):
    """Entrée TDM renvoyée par Vision en sortie structurée."""

    model_config = ConfigDict(extra="forbid")

    title: str
    page: int = Field(ge=1)
    level: int = Field(ge=0, le=5)


class AnnualTocBoundaryLLM(BaseModel):
    """Bornes de chapitre cible lues dans la TDM annuelle."""

    model_config = ConfigDict(extra="forbid")

    section_type: Literal["capital_management", "risk_management"]
    title_found: str
    start_page: int = Field(ge=1)
    successor_title: str
    successor_page: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)


class AnnualTocAnalysisLLM(BaseModel):
    """Réponse structurée Vision pour une page TDM annuelle candidate."""

    model_config = ConfigDict(extra="forbid")

    is_master_toc: bool
    confidence: float = Field(ge=0.0, le=1.0)
    entries: list[AnnualTocEntryLLM]
    boundaries: list[AnnualTocBoundaryLLM]
    warnings: list[str]


class PageTransitionLLM(BaseModel):
    """Réponse structurée Vision pour une transition de chapitre."""

    model_config = ConfigDict(extra="forbid")

    confirmed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    observed_title: str
    previous_page_belongs_to_prior_section: bool
    candidate_page_starts_expected_section: bool
    reason: str


@dataclass
class TOCBoundaryRole:
    """Repere de debut et de fin lu dans la TDM annuelle."""

    section_type: str
    title_found: str
    start_page: int
    successor_title: str
    successor_page: int
    confidence: float


@dataclass
class AnnualTOCAnalysis:
    """Lecture Vision complete d'une TDM annuelle."""

    is_master_toc: bool
    confidence: float
    page_number: int
    entries: list[dict]
    boundaries: list[TOCBoundaryRole]
    warnings: list[str]
    raw_response: str | None = None


@dataclass
class PageTransitionValidation:
    """Validation Vision d'une transition entre deux pages physiques."""

    confirmed: bool
    confidence: float
    observed_title: str
    previous_page_belongs_to_prior_section: bool
    candidate_page_starts_expected_section: bool
    reason: str
    raw_response: str | None = None


class GenAITOCDetector:
    """Lectures Vision de la TDM annuelle et des transitions de chapitre.

    Deux usages, tous deux appelés par la validation des bornes annuelles :
    ``analyze_annual_toc_page`` et ``validate_section_transition``.
    """

    ANNUAL_TOC_ANALYSIS_PROMPT = """Tu analyses une page complète d'un rapport annuel bancaire canadien.

OBJECTIF
1. Déterminer si cette page est la table des matières PRINCIPALE du rapport de gestion.
2. Extraire fidèlement toutes les entrées visibles (titre et numéro de page imprimé).
3. Identifier les limites des chapitres de gestion du capital et de gestion des risques.

Une table des matières principale couvre plusieurs grands chapitres du rapport. Ne classe pas comme
table des matières principale un sommaire interne à une section, une liste de tableaux, un index,
un glossaire, ni un tableau financier.

Pour chaque section cible :
- capital_management : gestion du capital, fonds propres ou Capital Management;
- risk_management : gestion des risques, gestion du risque ou Risk Management.

Le successeur est la première entrée au même niveau hiérarchique que la section cible, ou le premier
nouveau chapitre qui commence après sa portée complète. Ne saute pas un chapitre intermédiaire comme
« Titrisation et arrangements hors bilan », « Instruments financiers » ou « Contrôles et procédures ».
Ce n'est toutefois pas une sous-section interne comme risque de crédit, risque de marché ou risque
opérationnel.

Si un grand titre n'a aucun numéro propre, utilise le numéro de sa première sous-entrée numérotée.
N'infère jamais son numéro depuis l'entrée précédente. Préserve les titres visibles et signale toute
ambiguïté dans warnings.

Priorise les boundaries des sections cibles, puis les entrées de niveau 0-1. Évite de saturer la
sortie avec des sous-sections profondes si l'espace est limité.

Réponds strictement selon le schéma structuré fourni."""

    PAGE_TRANSITION_PROMPT = """Tu reçois deux pages physiques complètes et consécutives d'un rapport bancaire.
La PREMIÈRE IMAGE est la page précédente. La DEUXIÈME IMAGE est la page candidate.

Vérifie si la deuxième page commence réellement le grand chapitre attendu ci-dessous :
- rôle canonique : {section_type}
- titre attendu ou équivalent sémantique : {expected_title}

Un titre seulement mentionné dans une phrase, un en-tête courant, une table des matières, une note ou
un renvoi n'est pas un début de chapitre. Le titre peut toutefois varier légèrement ou être bilingue.
La page précédente doit encore appartenir au chapitre antérieur, sauf si la mise en page explique
clairement une transition sur la même page.

Réponds strictement selon le schéma structuré fourni."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        """Initialiser le détecteur GenAI.

        Args:
            api_key: Clé API OpenAI (ou depuis OPENAI_API_KEY)
            model: Modèle à utiliser (default: gpt-4o)
        """
        self.api_key = api_key or get_openai_api_key()
        self.model = model
        self._client = None

        if not self.api_key:
            logger.warning("Clé API OpenAI non configurée pour GenAITOCDetector")

    @property
    def client(self):
        """Client OpenAI (chargement paresseux)."""
        if self._client is None:
            try:
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=120.0,
                    max_retries=1,
                )
            except ImportError:
                logger.error("openai non installé")
                return None
        return self._client

    def _page_to_base64(self, pdf_path: Path, page_num: int) -> str | None:
        """Convertir une page PDF en image base64.

        Args:
            pdf_path: Chemin vers le PDF
            page_num: Numéro de page (1-indexed)

        Returns:
            Image encodée en base64 ou None
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    return None

                page = pdf.pages[page_num - 1]
                img = page.to_image(resolution=300)

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            logger.error(f"Erreur conversion page {page_num}: {e}")
            return None

    def _call_vision_structured(
        self,
        prompt: str,
        images_base64: list[str],
        *,
        response_format: type[_T_StructuredModel],
        max_completion_tokens: int = 4000,
    ) -> _T_StructuredModel | None:
        """Appeler Vision multi-images avec sortie Pydantic structurée.

        Remplace ``json_object`` + ``json.loads`` pour les lectures TDM / transitions
        annuelles. Soft-fail (``None``) en cas d'erreur, refus ou troncature.
        """
        if not self.client or not images_base64:
            return None

        content: list[dict] = [{"type": "text", "text": prompt}]
        for index, image_base64 in enumerate(images_base64, start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"IMAGE {index} sur {len(images_base64)}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ]
            )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                response_format=response_format,
                temperature=0.0,
                max_completion_tokens=max_completion_tokens,
            )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                logger.error(
                    "Erreur API Vision multi-pages: sortie structurée tronquée "
                    "(finish_reason=length, max_completion_tokens=%s)",
                    max_completion_tokens,
                )
                return None

            message = choice.message
            refusal = getattr(message, "refusal", None)
            if refusal:
                logger.error("Erreur API Vision multi-pages: refus modèle: %s", refusal)
                return None

            parsed = getattr(message, "parsed", None)
            if parsed is None:
                logger.error("Erreur API Vision multi-pages: sortie structurée vide")
                return None
            return parsed
        except Exception as e:
            logger.error("Erreur API Vision multi-pages: %s", e)
            return None

    def analyze_annual_toc_page(
        self,
        pdf_path: str | Path,
        page_num: int,
    ) -> AnnualTOCAnalysis:
        """Classifier et lire en un appel une TDM annuelle candidate."""
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            return AnnualTOCAnalysis(False, 0.0, page_num, [], [], [])
        image_b64 = self._page_to_base64(Path(raw_pdf_path), page_num)
        if not image_b64:
            return AnnualTOCAnalysis(False, 0.0, page_num, [], [], [])

        result = self._call_vision_structured(
            self.ANNUAL_TOC_ANALYSIS_PROMPT,
            [image_b64],
            response_format=AnnualTocAnalysisLLM,
            max_completion_tokens=6000,
        )
        if not result:
            return AnnualTOCAnalysis(False, 0.0, page_num, [], [], [])

        boundaries: list[TOCBoundaryRole] = []
        for raw_boundary in result.boundaries:
            if raw_boundary.successor_page <= raw_boundary.start_page:
                continue
            boundaries.append(
                TOCBoundaryRole(
                    section_type=raw_boundary.section_type,
                    title_found=raw_boundary.title_found.strip(),
                    start_page=raw_boundary.start_page,
                    successor_title=raw_boundary.successor_title.strip(),
                    successor_page=raw_boundary.successor_page,
                    confidence=raw_boundary.confidence,
                )
            )

        return AnnualTOCAnalysis(
            is_master_toc=result.is_master_toc,
            confidence=result.confidence,
            page_number=page_num,
            entries=[entry.model_dump() for entry in result.entries],
            boundaries=boundaries,
            warnings=[warning for warning in result.warnings if str(warning).strip()],
            raw_response=result.model_dump_json(),
        )

    def validate_section_transition(
        self,
        pdf_path: str | Path,
        previous_page: int,
        candidate_page: int,
        *,
        section_type: str,
        expected_title: str,
    ) -> PageTransitionValidation:
        """Valider sur pages complètes qu'un chapitre commence à la page candidate."""
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            return PageTransitionValidation(False, 0.0, "", False, False, "")
        pdf_path = Path(raw_pdf_path)
        images = [
            image
            for page in (previous_page, candidate_page)
            if page > 0 and (image := self._page_to_base64(pdf_path, page))
        ]
        if not images:
            return PageTransitionValidation(False, 0.0, "", False, False, "")

        result = self._call_vision_structured(
            self.PAGE_TRANSITION_PROMPT.format(
                section_type=section_type,
                expected_title=expected_title,
            ),
            images,
            response_format=PageTransitionLLM,
            max_completion_tokens=1200,
        )
        if not result:
            return PageTransitionValidation(False, 0.0, "", False, False, "")
        return PageTransitionValidation(
            confirmed=result.confirmed,
            confidence=result.confidence,
            observed_title=result.observed_title.strip(),
            previous_page_belongs_to_prior_section=result.previous_page_belongs_to_prior_section,
            candidate_page_starts_expected_section=result.candidate_page_starts_expected_section,
            reason=result.reason.strip(),
            raw_response=result.model_dump_json(),
        )
