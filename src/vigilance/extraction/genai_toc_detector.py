"""
Détecteur de Table des Matières utilisant GPT-4 Vision.

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


class GenAITOCDetector:
    """
    Détecteur GenAI de Table des Matières pour rapports bancaires.

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
2. **Gestion des risques** (variantes : "Gestion du risque", "Risk Management")

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

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        """
        Initialiser le détecteur GenAI.

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
        """Client OpenAI (lazy loading)."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai non installé")
                return None
        return self._client

    def _page_to_base64(self, pdf_path: Path, page_num: int) -> str | None:
        """
        Convertir une page PDF en image base64.

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
        """
        Appeler l'API GPT-4 Vision avec un prompt et une image.

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

    def detect_toc_page(
        self, pdf_path: str | Path, page_num: int
    ) -> TOCDetectionResult:
        """
        Détecter si une page contient une Table des Matières.

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
        """
        Extraire les entrées d'une Table des Matières.

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
        """
        Détecter les sections cibles dans une TDM.

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
        """
        Processus complet : trouver la TDM et extraire les sections cibles.

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
            search_pages = list(range(2, 11))  # Pages 2-10

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
