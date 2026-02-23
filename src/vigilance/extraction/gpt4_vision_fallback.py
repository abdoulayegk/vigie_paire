"""
Fallback GPT-4 Vision pour l'extraction de tableaux complexes.
Utilisé lorsque Docling et pdfplumber échouent sur des tableaux difficiles.

AMELIORATIONS:
- Retry logic pour resilience aux erreurs reseau
- Rate limiting pour eviter les erreurs 429
"""

import base64
import json
import logging
from dataclasses import dataclass

import numpy as np

from ..utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)

# Import des utilitaires de resilience
try:
    from ..utils.rate_limiter import openai_limiter
    from ..utils.retry import retry_openai_call

    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False
    logger.debug("Modules retry/rate_limiter non disponibles")


@dataclass
class VisionExtractionResult:
    """Résultat de l'extraction via GPT-4 Vision."""

    success: bool
    headers: list[str]
    rows: list[list[str]]
    footnotes: list[str]
    table_title: str | None
    confidence: float
    raw_response: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "headers": self.headers,
            "rows": self.rows,
            "footnotes": self.footnotes,
            "table_title": self.table_title,
            "confidence": self.confidence,
            "error": self.error,
        }


class GPT4VisionFallback:
    """
    Extraction de tableaux via GPT-4 Vision (gpt-4o) pour les cas complexes.

    Utilisé lorsque:
    - Docling ne peut pas parser le tableau
    - Le tableau est renversé/pivoté
    - La structure est trop complexe
    - L'image est de mauvaise qualité
    """

    # Prompt système pour l'extraction de tableaux bancaires
    SYSTEM_PROMPT = """Tu es un expert en extraction de données de rapports bancaires canadiens.
Ta tâche est d'extraire les données d'un tableau à partir d'une image.

INSTRUCTIONS:
1. Identifie le titre du tableau s'il est visible
2. Extrais les en-têtes de colonnes (peut être sur plusieurs lignes)
3. Extrais toutes les lignes de données
4. Note les notes de bas de page ou références (1), (2), *, etc.
5. Si le tableau est renversé/pivoté, redresse-le mentalement avant extraction

FORMAT DE RÉPONSE (JSON strict):
{
    "table_title": "Titre du tableau ou null",
    "headers": ["Col1", "Col2", "Col3"],
    "rows": [
        ["Valeur1", "Valeur2", "Valeur3"],
        ["Valeur4", "Valeur5", "Valeur6"]
    ],
    "footnotes": ["(1) Note explicative", "(2) Autre note"],
    "is_rotated": false,
    "confidence": 0.95
}

RÈGLES:
- Préserve les valeurs numériques exactement (avec $, %, M, G, etc.)
- Préserve les parenthèses pour les valeurs négatives
- Si une cellule est vide, utilise ""
- Fusionne les en-têtes multi-lignes avec " - "
- confidence entre 0.0 et 1.0 selon la clarté de l'image
- confidence entre 0.0 et 1.0 selon la clarté de l'image
"""

    # Prompt de fusion intelligente (Smart Merge) inspiré du prototype Jad
    MERGE_PROMPT = """Tu es un expert en réconciliation de données financières.
Tu disposes de deux extractions du MÊME tableau bancaire :
1. "DOCLING": Extraction basée sur la structure (précise pour le texte, moins pour la mise en page complexe)
2. "VISION": Extraction visuelle via GPT-4o (précise pour la structure, parfois des erreurs OCR)

TA TÂCHE : Fusionner ces deux résultats en une version unique et parfaite.

RÈGLES DE FUSION :
- Priorité TEXTE : Si un chiffre ou un texte diffère, privilégie généralement DOCLING (texte extrait du PDF) sauf s'il semble manifestement corrompu.
- Priorité STRUCTURE : Si la structure (lignes/colonnes) diffère, privilégie VISION qui "voit" la mise en page.
- Complétude : Garde les lignes présentes dans l'un mais manquantes dans l'autre (après vérification qu'elles ne sont pas des doublons).
- Validation : Vérifie que le total des lignes et la logique des colonnes sont cohérents.

FORMAT DE SORTIE (JSON strict) :
{
    "table_title": "Titre consolidé",
    "headers": ["Col1", "Col2"],
    "rows": [["Val1", "Val2"], ...],
    "footnotes": [],
    "merge_notes": ["Note sur ce qui a été fusionné ou corrigé"],
    "confidence": 0.95
}
"""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        """
        Initialiser le fallback GPT-4 Vision.

        Args:
            api_key: Clé API OpenAI (ou depuis env OPENAI_API_KEY)
            model: Modèle à utiliser (gpt-4o recommandé)
        """
        self.model = model
        self._client = None
        self._api_key = api_key

    def _ensure_client(self):
        """Initialisation différée du client OpenAI."""
        if self._client is not None:
            return

        try:
            from openai import OpenAI

            api_key = self._api_key or get_openai_api_key()
            if not api_key:
                raise ValueError(
                    "Clé API OpenAI requise. Définissez OPENAI_API_KEY ou passez api_key."
                )

            self._client = OpenAI(api_key=api_key)
            logger.info(f"Client OpenAI initialisé (modèle: {self.model})")

        except ImportError:
            raise ImportError("Package openai requis. Installez avec: pip install openai")

    def extract_table_from_image(
        self, image: np.ndarray, context: str | None = None, max_completion_tokens: int = 4096
    ) -> VisionExtractionResult:
        """
        Extraire un tableau depuis une image via GPT-4 Vision.

        Args:
            image: Image du tableau (numpy array BGR)
            context: Contexte additionnel (ex: "Tableau de ratios de capital")
            max_completion_tokens: Nombre maximum de tokens pour la réponse

        Returns:
            VisionExtractionResult avec les données extraites
        """
        self._ensure_client()

        try:
            # Encoder l'image en base64
            image_b64 = self._encode_image(image)

            # Construire le prompt utilisateur
            user_prompt = "Extrais les données de ce tableau bancaire en JSON."
            if context:
                user_prompt += f"\n\nContexte: {context}"

            # Appliquer rate limiting si disponible
            if RETRY_AVAILABLE:
                openai_limiter.acquire()

            # Appel API avec retry
            def _make_api_call():
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    max_completion_tokens=max_completion_tokens,
                    temperature=0.1,  # Faible température pour extraction précise
                    response_format={"type": "json_object"},
                )

            if RETRY_AVAILABLE:
                response = retry_openai_call(_make_api_call)
            else:
                response = _make_api_call()

            # Parser la réponse
            raw_content = response.choices[0].message.content
            result = self._parse_response(raw_content)

            logger.info(
                f"Extraction Vision réussie: {len(result.rows)} lignes, confiance {result.confidence:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Erreur extraction GPT-4 Vision: {e}")
            return VisionExtractionResult(
                success=False,
                headers=[],
                rows=[],
                footnotes=[],
                table_title=None,
                confidence=0.0,
                error=str(e),
            )

    def smart_merge_results(
        self,
        docling_data: dict,
        vision_result: VisionExtractionResult,
        context: str | None = None,
    ) -> VisionExtractionResult:
        """
        Fusionner intelligemment les résultats Docling et Vision via LLM.

        Args:
            docling_data: Données extraites par Docling (dict)
            vision_result: Résultat de l'extraction Vision
            context: Contexte optionnel

        Returns:
            VisionExtractionResult fusionné
        """
        self._ensure_client()

        # Si l'un des deux est vide/invalide, retourner l'autre
        if not docling_data or not docling_data.get("rows"):
            return vision_result

        if not vision_result.success:
            return VisionExtractionResult(
                success=True,
                headers=docling_data.get("headers", []),
                rows=docling_data.get("rows", []),
                footnotes=docling_data.get("footnotes", []),
                table_title=docling_data.get("title"),
                confidence=0.5,  # Confiance moyenne si fallback texte uniquement
                raw_response="Fallback Docling only (Vision failed)",
            )

        try:
            logger.info("Lancement du Smart Merge (Docling + Vision)...")

            # Préparer les données pour le prompt
            merge_input = {
                "DOCLING_EXTRACTION": {
                    "title": docling_data.get("title"),
                    "headers": docling_data.get("headers"),
                    "rows": docling_data.get("rows")[:50],  # Limiter pour le prompt
                    "row_count": len(docling_data.get("rows", [])),
                },
                "VISION_EXTRACTION": {
                    "title": vision_result.table_title,
                    "headers": vision_result.headers,
                    "rows": vision_result.rows[:50],
                    "row_count": len(vision_result.rows),
                },
                "CONTEXT": context or "Tableau financier",
            }

            # Appliquer rate limiting
            if RETRY_AVAILABLE:
                openai_limiter.acquire()

            # Appel API
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.MERGE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Fusionne ces deux extractions :\n{json.dumps(merge_input, indent=2, ensure_ascii=False)}",
                    },
                ],
                max_completion_tokens=4096,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # Parser le résultat
            merged_content = response.choices[0].message.content
            merged_data = json.loads(merged_content)

            # Créer le résultat fusionné
            merged_result = VisionExtractionResult(
                success=True,
                headers=merged_data.get("headers", []),
                rows=merged_data.get("rows", []),
                footnotes=merged_data.get("footnotes", []) or vision_result.footnotes,
                table_title=merged_data.get("table_title") or vision_result.table_title,
                confidence=merged_data.get("confidence", 0.9),
                raw_response=merged_content,
            )

            logger.info(
                f"Smart Merge réussi : {len(merged_result.rows)} lignes (Notes: {merged_data.get('merge_notes')})"
            )
            return merged_result

        except Exception as e:
            logger.error(f"Erreur Smart Merge: {e}")
            # En cas d'erreur de fusion, on retourne le résultat Vision (souvent plus cohérent structurellement en fallback)
            return vision_result

    def extract_table_from_pdf_region(
        self,
        pdf_path: str,
        page_number: int,
        region: tuple | None = None,
        context: str | None = None,
    ) -> VisionExtractionResult:
        """
        Extraire un tableau d'une région spécifique d'une page PDF.

        Args:
            pdf_path: Chemin vers le fichier PDF
            page_number: Numéro de page (1-indexed)
            region: Région (x, y, width, height) ou None pour page entière
            context: Contexte additionnel

        Returns:
            VisionExtractionResult avec les données extraites
        """
        try:
            # Convertir la page PDF en image
            image = self._pdf_to_image(pdf_path, page_number)

            # Extraire la région si spécifiée
            if region:
                x, y, w, h = region
                image = image[y : y + h, x : x + w]

            return self.extract_table_from_image(image, context)

        except Exception as e:
            logger.error(f"Erreur extraction PDF région: {e}")
            return VisionExtractionResult(
                success=False,
                headers=[],
                rows=[],
                footnotes=[],
                table_title=None,
                confidence=0.0,
                error=str(e),
            )

    def _encode_image(self, image: np.ndarray) -> str:
        """Encoder une image numpy RGB en base64 PNG."""
        from io import BytesIO

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow requis. Installez avec: pip install pillow")

        pil_img = Image.fromarray(image)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _pdf_to_image(self, pdf_path: str, page_number: int, dpi: int = 300) -> np.ndarray:
        """Convertir une page PDF en image numpy RGB."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF requis. Installez avec: pip install pymupdf")

        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = img[:, :, :3]
        elif pix.n == 1:
            img = np.stack([img.squeeze()] * 3, axis=-1)

        doc.close()
        return img

    def _parse_response(self, raw_content: str) -> VisionExtractionResult:
        """Parser la réponse JSON de GPT-4 Vision."""
        try:
            data = json.loads(raw_content)

            return VisionExtractionResult(
                success=True,
                headers=data.get("headers", []),
                rows=data.get("rows", []),
                footnotes=data.get("footnotes", []),
                table_title=data.get("table_title"),
                confidence=data.get("confidence", 0.8),
                raw_response=raw_content,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Réponse JSON invalide: {e}")

            # Tenter d'extraire le JSON du texte
            import re

            json_match = re.search(r"\{[\s\S]*\}", raw_content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return VisionExtractionResult(
                        success=True,
                        headers=data.get("headers", []),
                        rows=data.get("rows", []),
                        footnotes=data.get("footnotes", []),
                        table_title=data.get("table_title"),
                        confidence=data.get("confidence", 0.5),  # Confiance réduite
                        raw_response=raw_content,
                    )
                except:
                    pass

            return VisionExtractionResult(
                success=False,
                headers=[],
                rows=[],
                footnotes=[],
                table_title=None,
                confidence=0.0,
                raw_response=raw_content,
                error=f"JSON invalide: {e}",
            )


class VisionTableValidator:
    """
    Validateur pour vérifier la qualité des extractions Vision.
    """

    @staticmethod
    def validate_extraction(result: VisionExtractionResult) -> tuple[bool, list[str]]:
        """
        Valider une extraction Vision.

        Returns:
            Tuple (is_valid, list of issues)
        """
        issues = []

        if not result.success:
            issues.append(f"Extraction échouée: {result.error}")
            return False, issues

        # Vérifier les en-têtes
        if not result.headers:
            issues.append("Aucun en-tête extrait")

        # Vérifier les lignes
        if not result.rows:
            issues.append("Aucune ligne de données extraite")

        # Vérifier la cohérence (même nombre de colonnes)
        if result.headers and result.rows:
            header_count = len(result.headers)
            for i, row in enumerate(result.rows):
                if len(row) != header_count:
                    issues.append(f"Ligne {i + 1}: {len(row)} colonnes vs {header_count} en-têtes")

        # Vérifier la confiance
        if result.confidence < 0.5:
            issues.append(f"Confiance faible: {result.confidence:.2f}")

        is_valid = len(issues) == 0 or (len(issues) == 1 and "Confiance faible" in issues[0])

        return is_valid, issues

    @staticmethod
    def merge_extractions(
        docling_result: dict | None, vision_result: VisionExtractionResult
    ) -> dict:
        """
        Fusionner les résultats Docling et Vision pour obtenir le meilleur.

        Args:
            docling_result: Résultat Docling (peut être None ou incomplet)
            vision_result: Résultat GPT-4 Vision

        Returns:
            Meilleur résultat fusionné
        """
        # Si Docling a échoué complètement, utiliser Vision
        if not docling_result or not docling_result.get("headers"):
            if vision_result.success:
                return {
                    "headers": vision_result.headers,
                    "rows": vision_result.rows,
                    "footnotes": vision_result.footnotes,
                    "title": vision_result.table_title,
                    "extraction_method": "gpt4_vision",
                    "confidence": vision_result.confidence,
                }

        # Si les deux ont des résultats, préférer celui avec plus de données
        docling_rows = len(docling_result.get("rows", []))
        vision_rows = len(vision_result.rows) if vision_result.success else 0

        if vision_rows > docling_rows * 1.2:  # Vision a 20% plus de lignes
            return {
                "headers": vision_result.headers,
                "rows": vision_result.rows,
                "footnotes": vision_result.footnotes,
                "title": vision_result.table_title,
                "extraction_method": "gpt4_vision",
                "confidence": vision_result.confidence,
            }

        # Sinon garder Docling
        return {**docling_result, "extraction_method": "docling", "confidence": 1.0}


# Instance globale
_vision_fallback = None


def get_vision_fallback(api_key: str | None = None) -> GPT4VisionFallback:
    """Obtenir l'instance singleton du fallback Vision."""
    global _vision_fallback
    if _vision_fallback is None:
        _vision_fallback = GPT4VisionFallback(api_key=api_key)
    return _vision_fallback
