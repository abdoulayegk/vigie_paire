"""Détecteur de Table des Matières utilisant GPT-4 Vision.

Ce module est utilisé en FALLBACK lorsque les méthodes déterministes
du SectionLocator échouent à détecter les sections cibles.

Inspiré du prototype "Jad" avec triple vérification Vision+Texte.
"""

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)


@dataclass
class TOCDetectionResult:
    """Résultat de la détection de Table des Matières."""

    is_toc: bool
    confidence: float
    page_number: int
    entries: list[dict]  # [{"title": str, "page": int, "level": int}]
    raw_response: str | None = None


@dataclass
class SectionDetectionResult:
    """Résultat de la détection des sections cibles."""

    section_type: str  # "gestion_capital" ou "gestion_risques"
    title_found: str
    start_page: int
    confidence: float


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
    """Détecteur GenAI de Table des Matières pour rapports bancaires.

    Utilisé comme fallback lorsque les règles déterministes échouent.
    """

    # Prompt de détection de TDM (inspiré de Jad)
    TOC_DETECTION_PROMPT = """Tu es un spécialiste en analyse de documents financiers bancaires.

TÂCHE : Analyser cette page PDF et déterminer si elle contient une Table des Matières (TDM).

INDICATEURS D'UNE TDM :
1. Titres explicites : "Table des matières", "Sommaire", "Contents"
2. Pattern répétitif : [Titre de section] ... [Numéro de page]
3. Structure hiérarchique avec indentation
4. Numéros de page généralement croissants
5. Utilisation de points de conduite (.....) entre titre et page

ATTENTION - Ce n'est PAS une TDM si :
- C'est une liste de tableaux ou figures
- C'est un index alphabétique
- C'est un glossaire
- Les numéros ne sont pas des pages mais des chapitres

Réponds en JSON :
{
    "is_toc": true ou false,
    "confidence": 0.0 à 1.0,
    "reasoning": "Explication courte"
}"""

    # Prompt d'extraction des entrées de TDM
    TOC_EXTRACTION_PROMPT = """Tu es un expert en extraction de données structurées.

TÂCHE : Extraire TOUTES les entrées de cette Table des Matières.

RÈGLES :
1. Extrais CHAQUE entrée visible (titre + numéro de page)
2. Détermine le niveau hiérarchique (0 = section principale, 1+ = sous-section)
3. Si un numéro de page semble erroné (ex: 6 au lieu de 8), signale-le
4. Préserve le texte exact des titres

ERREURS OCR COURANTES à corriger :
- 6 ↔ 8
- 3 ↔ 8
- 1 ↔ 7

Réponds en JSON :
{
    "title": "Titre de la TDM si visible",
    "entries": [
        {"title": "Nom de la section", "page": 12, "level": 0},
        {"title": "Sous-section", "page": 15, "level": 1}
    ],
    "warnings": ["Page 18 incertaine - pourrait être 16"]
}"""

    # Prompt de détection des sections cibles
    SECTION_DETECTION_PROMPT = """Tu es un expert en rapports financiers bancaires canadiens.

TÂCHE : Dans cette Table des Matières, identifier les sections suivantes :

1. **Gestion du capital** (variantes : "Gestion des fonds propres", "Capital Management")
2. **Gestion des risques** (variantes : "Gestion du risque", "Risk Management").
   Cette portée comprend aussi les sections ou sous-sections autonomes sur les
   données, la technologie, la cybersécurité, les fournisseurs et tiers,
   l'impartition, les services infonuagiques, la vie privée et la résilience
   opérationnelle. Si aucun titre global "Gestion des risques" n'existe,
   utiliser la première de ces sections autonomes comme début de
   gestion_risques.

Pour chaque section trouvée, retourne :
- Le type (gestion_capital ou gestion_risques)
- Le titre exact trouvé
- Le numéro de page de début
- Un score de confiance

Si une section n'est pas trouvée, ne l'inclus pas.

Réponds en JSON :
{
    "sections": [
        {
            "type": "gestion_capital",
            "title_found": "Gestion du capital",
            "start_page": 6,
            "confidence": 0.95
        }
    ]
}"""

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

Réponds uniquement avec cet objet JSON :
{
  "is_master_toc": true,
  "confidence": 0.0,
  "entries": [
    {"title": "Titre exact", "page": 12, "level": 0}
  ],
  "boundaries": [
    {
      "section_type": "capital_management",
      "title_found": "Titre exact de début",
      "start_page": 53,
      "successor_title": "Titre exact du chapitre suivant",
      "successor_page": 62,
      "confidence": 0.95
    }
  ],
  "warnings": []
}"""

    PAGE_TRANSITION_PROMPT = """Tu reçois deux pages physiques complètes et consécutives d'un rapport bancaire.
La PREMIÈRE IMAGE est la page précédente. La DEUXIÈME IMAGE est la page candidate.

Vérifie si la deuxième page commence réellement le grand chapitre attendu ci-dessous :
- rôle canonique : {section_type}
- titre attendu ou équivalent sémantique : {expected_title}

Un titre seulement mentionné dans une phrase, un en-tête courant, une table des matières, une note ou
un renvoi n'est pas un début de chapitre. Le titre peut toutefois varier légèrement ou être bilingue.
La page précédente doit encore appartenir au chapitre antérieur, sauf si la mise en page explique
clairement une transition sur la même page.

Réponds uniquement avec cet objet JSON :
{{
  "confirmed": true,
  "confidence": 0.0,
  "observed_title": "Titre réellement visible",
  "previous_page_belongs_to_prior_section": true,
  "candidate_page_starts_expected_section": true,
  "reason": "Justification courte"
}}"""

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
                from openai import OpenAI

                self._client = OpenAI(
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
            import pdfplumber

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

    def _call_vision_api(self, prompt: str, image_base64: str) -> dict | None:
        """Appeler l'API GPT-4 Vision avec un prompt et une image.

        Args:
            prompt: Prompt système
            image_base64: Image encodée en base64

        Returns:
            Réponse JSON parsée ou None
        """
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
                max_completion_tokens=2000,
            )

            content = response.choices[0].message.content

            # Extraire le JSON de la réponse
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group())

            return None

        except Exception as e:
            logger.error(f"Erreur API Vision: {e}")
            return None

    def _call_vision_api_images(
        self,
        prompt: str,
        images_base64: list[str],
        *,
        max_completion_tokens: int = 4000,
    ) -> dict | None:
        """Appeler Vision avec plusieurs pages complètes dans un ordre explicite."""
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or ""
            parsed = json.loads(raw_content)
            return parsed if isinstance(parsed, dict) else None
        except Exception as e:
            logger.error("Erreur API Vision multi-pages: %s", e)
            return None

    @staticmethod
    def _safe_int(value: object) -> int:
        """Convertir un numéro de page Vision sans lever d'exception."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

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

        result = self._call_vision_api_images(
            self.ANNUAL_TOC_ANALYSIS_PROMPT,
            [image_b64],
            max_completion_tokens=6000,
        )
        if not result:
            return AnnualTOCAnalysis(False, 0.0, page_num, [], [], [])

        entries = result.get("entries")
        if not isinstance(entries, list):
            entries = []

        boundaries: list[TOCBoundaryRole] = []
        for raw_boundary in result.get("boundaries", []):
            if not isinstance(raw_boundary, dict):
                continue
            section_type = str(raw_boundary.get("section_type") or "").strip()
            if section_type not in {"capital_management", "risk_management"}:
                continue
            start_page = self._safe_int(raw_boundary.get("start_page"))
            successor_page = self._safe_int(raw_boundary.get("successor_page"))
            if start_page <= 0 or successor_page <= start_page:
                continue
            boundaries.append(
                TOCBoundaryRole(
                    section_type=section_type,
                    title_found=str(raw_boundary.get("title_found") or "").strip(),
                    start_page=start_page,
                    successor_title=str(raw_boundary.get("successor_title") or "").strip(),
                    successor_page=successor_page,
                    confidence=float(raw_boundary.get("confidence") or 0.0),
                )
            )

        warnings = result.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
        return AnnualTOCAnalysis(
            is_master_toc=bool(result.get("is_master_toc")),
            confidence=float(result.get("confidence") or 0.0),
            page_number=page_num,
            entries=[entry for entry in entries if isinstance(entry, dict)],
            boundaries=boundaries,
            warnings=[str(warning) for warning in warnings if str(warning).strip()],
            raw_response=json.dumps(result, ensure_ascii=False),
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

        result = self._call_vision_api_images(
            self.PAGE_TRANSITION_PROMPT.format(
                section_type=section_type,
                expected_title=expected_title,
            ),
            images,
            max_completion_tokens=1200,
        )
        if not result:
            return PageTransitionValidation(False, 0.0, "", False, False, "")
        return PageTransitionValidation(
            confirmed=bool(result.get("confirmed")),
            confidence=float(result.get("confidence") or 0.0),
            observed_title=str(result.get("observed_title") or "").strip(),
            previous_page_belongs_to_prior_section=bool(
                result.get("previous_page_belongs_to_prior_section")
            ),
            candidate_page_starts_expected_section=bool(
                result.get("candidate_page_starts_expected_section")
            ),
            reason=str(result.get("reason") or "").strip(),
            raw_response=json.dumps(result, ensure_ascii=False),
        )

    def detect_toc_page(
        self, pdf_path: str | Path, page_num: int
    ) -> TOCDetectionResult:
        """Détecter si une page contient une Table des Matières.

        Args:
            pdf_path: Chemin vers le PDF
            page_num: Numéro de page à analyser

        Returns:
            TOCDetectionResult avec is_toc et confidence
        """
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            return TOCDetectionResult(
                is_toc=False, confidence=0.0, page_number=page_num, entries=[]
            )
        pdf_path = Path(raw_pdf_path)

        # Convertir la page en image
        image_b64 = self._page_to_base64(pdf_path, page_num)
        if not image_b64:
            return TOCDetectionResult(
                is_toc=False, confidence=0.0, page_number=page_num, entries=[]
            )

        # Appeler l'API
        result = self._call_vision_api(self.TOC_DETECTION_PROMPT, image_b64)

        if not result:
            return TOCDetectionResult(
                is_toc=False, confidence=0.0, page_number=page_num, entries=[]
            )

        return TOCDetectionResult(
            is_toc=result.get("is_toc", False),
            confidence=result.get("confidence", 0.0),
            page_number=page_num,
            entries=[],
            raw_response=json.dumps(result),
        )

    def extract_toc_entries(
        self, pdf_path: str | Path, page_num: int
    ) -> TOCDetectionResult:
        """Extraire les entrées d'une Table des Matières.

        Args:
            pdf_path: Chemin vers le PDF
            page_num: Numéro de page (doit être une TDM)

        Returns:
            TOCDetectionResult avec entries remplies
        """
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            return TOCDetectionResult(
                is_toc=False, confidence=0.0, page_number=page_num, entries=[]
            )
        pdf_path = Path(raw_pdf_path)

        image_b64 = self._page_to_base64(pdf_path, page_num)
        if not image_b64:
            return TOCDetectionResult(
                is_toc=False, confidence=0.0, page_number=page_num, entries=[]
            )

        result = self._call_vision_api(self.TOC_EXTRACTION_PROMPT, image_b64)

        if not result:
            return TOCDetectionResult(
                is_toc=False, confidence=0.0, page_number=page_num, entries=[]
            )

        entries = result.get("entries", [])

        return TOCDetectionResult(
            is_toc=True,
            confidence=0.9 if entries else 0.0,
            page_number=page_num,
            entries=entries,
            raw_response=json.dumps(result),
        )

    def detect_target_sections(
        self, pdf_path: str | Path, toc_page: int
    ) -> list[SectionDetectionResult]:
        """Détecter les sections cibles dans une TDM.

        Args:
            pdf_path: Chemin vers le PDF
            toc_page: Page contenant la TDM

        Returns:
            Liste de SectionDetectionResult
        """
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            return []
        pdf_path = Path(raw_pdf_path)

        image_b64 = self._page_to_base64(pdf_path, toc_page)
        if not image_b64:
            return []

        result = self._call_vision_api(self.SECTION_DETECTION_PROMPT, image_b64)

        if not result:
            return []

        sections = []
        for section_data in result.get("sections", []):
            sections.append(
                SectionDetectionResult(
                    section_type=section_data.get("type", ""),
                    title_found=section_data.get("title_found", ""),
                    start_page=section_data.get("start_page", 0),
                    confidence=section_data.get("confidence", 0.0),
                )
            )

        return sections

    def find_and_extract_sections(
        self, pdf_path: str | Path, search_pages: list[int] | None = None
    ) -> list[SectionDetectionResult]:
        """Processus complet : trouver la TDM et extraire les sections cibles.

        Args:
            pdf_path: Chemin vers le PDF
            search_pages: Pages à analyser (default: 2-10)

        Returns:
            Liste de SectionDetectionResult
        """
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            return []
        pdf_path = Path(raw_pdf_path)

        if search_pages is None:
            search_pages = list(range(2, 31))  # Rapports annuels: TDM souvent vers p.15

        logger.info(f"GenAI: Recherche TDM pages {search_pages[0]}-{search_pages[-1]}")

        # Étape 1: Trouver la page TDM
        toc_page = None
        for page_num in search_pages:
            detection = self.detect_toc_page(pdf_path, page_num)

            if detection.is_toc and detection.confidence >= 0.8:
                toc_page = page_num
                logger.info(
                    f"GenAI: TDM trouvée page {page_num} (conf={detection.confidence:.2f})"
                )
                break

        if toc_page is None:
            logger.warning("GenAI: Aucune TDM trouvée")
            return []

        # Étape 2: Extraire les sections cibles
        sections = self.detect_target_sections(pdf_path, toc_page)

        logger.info(f"GenAI: {len(sections)} sections détectées")
        for section in sections:
            logger.info(
                f"  - {section.section_type}: '{section.title_found}' "
                f"page {section.start_page} (conf={section.confidence:.2f})"
            )

        return sections
