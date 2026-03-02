"""
VisionTableExtractor - Extracteur de tableaux "Vision-Native" base sur GPT-4o.

Architecture:
1. Sniper: Ciblage precis des pages via SectionLocator
2. Scanner: Detection visuelle des tableaux via GPT-4o (page entiere)
3. Cerveau: Extraction structuree des donnees via GPT-4o (image decoupee)
4. Garde-Fou: Validation croisee avec pdfplumber

Priorite: FIABILITE MAXIMALE (Coût non-limitant)
"""

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Detection des dependances optionnelles
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF non disponible")

try:
    from openai import OpenAI

    from ..utils.genai import get_openai_api_key

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.debug("OpenAI non disponible")

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber non disponible")


# ==============================================================================
# DATACLASSES
# ==============================================================================


@dataclass
class DetectedTable:
    """Tableau detecte par le Scanner Vision."""

    page_number: int
    bbox: tuple[float, float, float, float]  # x, y, width, height (normalized 0-1)
    confidence: float
    table_type: str  # "financial", "regulatory", "footnote", "unknown"
    description: str | None = None


@dataclass
class ExtractedTableData:
    """Donnees extraites par le Cerveau Vision."""

    table_id: str
    page_number: int
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    first_column_indicators: list[str]  # Pour le matching
    footnotes: list[str]
    confidence: float
    extraction_method: str  # "gpt4o_vision", "pdfplumber_fallback", "hybrid"
    raw_image_path: str | None = None  # Preuve visuelle


@dataclass
class ValidationResult:
    """Resultat de validation Garde-Fou."""

    is_valid: bool
    gpt4o_rows: int
    pdfplumber_rows: int
    discrepancies: list[str]
    recommendation: str  # "use_gpt4o", "use_pdfplumber", "manual_review"


@dataclass
class ExtractionPipelineResult:
    """Resultat complet du pipeline Vision-Native."""

    pdf_path: str
    section_name: str
    pages_processed: list[int]
    detected_tables: list[DetectedTable]
    extracted_tables: list[ExtractedTableData]
    validation_results: list[ValidationResult]
    errors: list[str]
    total_cost_estimate: float  # Estimation cout OpenAI


# ==============================================================================
# PROMPTS GPT-4o
# ==============================================================================

SCANNER_SYSTEM_PROMPT = """Tu es un expert en detection de tableaux dans les rapports bancaires canadiens (BNC, BMO, CIBC, TD, RBC, BNS).

TACHE: Analyser visuellement cette page et identifier TOUS les tableaux financiers.

INSTRUCTIONS:
1. Identifie chaque tableau distinct (meme sans lignes visibles)
2. DETECTE TOUS les tableaux sur la page (plusieurs tableaux possibles)
3. Fournis les coordonnees normalisees (0-1) de chaque tableau
4. Classifie le type de tableau
5. EXTRAIT le numero de tableau si present ("TABLEAU 23", "T18", etc.)
6. Ignore les graphiques, images decoratives et en-tetes de page
7. Ignore les numeros de page en bas ("Page 40", etc.)

PATTERNS DE DETECTION:
- Bordures roses/mauves (TD, CIBC) = delimiteur de tableau
- Bordures grises (BNC) = delimiteur de tableau
- "TABLEAU XX" ou "T23" dans le titre = numero de tableau
- Headers en gras avec dates (Q1 2025, 31 janvier 2025) = nouvelle section

FORMAT DE REPONSE (JSON strict):
{
    "tables_detected": [
        {
            "bbox": {"x": 0.05, "y": 0.20, "width": 0.90, "height": 0.35},
            "confidence": 0.95,
            "table_type": "financial",
            "table_number": "28",
            "description": "TABLEAU 28: Expositions brutes au risque de credit"
        }
    ],
    "page_has_tables": true,
    "tables_count": 2,
    "notes": "Deux tableaux sur cette page: T28 et T29"
}

TYPES DE TABLEAUX:
- "financial": Donnees chiffrees (bilans, resultats, ratios)
- "regulatory": Tableaux reglementaires (Bale III, CET1, LCR, NSFR, TLAC)
- "footnote": Notes et references
- "unknown": Structure tableau mais contenu non financier
"""

EXTRACTION_SYSTEM_PROMPT = """Tu es un expert en extraction de donnees de rapports bancaires canadiens (BNC, BMO, CIBC, TD, RBC, BNS).

TACHE: Extraire TOUTES les donnees de ce tableau en JSON structure.

INSTRUCTIONS CRITIQUES:
1. Lis CHAQUE ligne, meme les sous-totaux, totaux et les lignes indentees
2. Preserve les valeurs exactement (avec $, %, M, G, parentheses pour negatifs)
3. Si la premiere colonne est vide, utilise le contexte de la ligne precedente
4. Capture les notes de bas de page (*, (1), (2),Superscript ¹²³⁴⁵⁶⁷⁸⁹⁰ etc.)
5. EXTRAIT le numero de tableau ("TABLEAU 23", "T18") s'il est visible

HEADERS COMPLEXES (TRES IMPORTANT):
- Si le tableau a des headers multi-niveaux (2-4 lignes de headers), fusionne-les intelligemment
- Exemple: "31 janvier 2025" + "Standard" + "NI" = "31 janvier 2025 - Standard", "31 janvier 2025 - NI"
- Les dates en headers sont critiques: "Q1 2025", "T1 2025", "30 avr 2025"
- PRESERVE l'ordre des colonnes de gauche a droite

FORMAT DE REPONSE (JSON strict):
{
    "table_number": "28",
    "table_title": "TABLEAU 28: Expositions brutes au risque de credit",
    "headers": ["Categorie", "31 janv 2025 - Standard", "31 janv 2025 - NI", "31 janv 2025 - Total", "31 oct 2024 - Standard", "31 oct 2024 - NI", "31 oct 2024 - Total"],
    "rows": [
        ["Expositions de detail", "", "", "", "", "", ""],
        ["  Immo residentiel", "4 383 $", "543 043 $", "547 426 $", "4 163 $", "537 075 $", "541 238 $"],
        ["Total", "15 426 $", "2 301 275 $", "2 316 701 $", "15 512 $", "2 294 435 $", "2 309 947 $"]
    ],
    "first_column_labels": ["Expositions de detail", "Immo residentiel", "Renouvelables", "Autres", "Total"],
    "footnotes": ["(1) Ne tient pas compte des mesures d'attenuation"],
    "confidence": 0.95,
    "is_complex_layout": true,
    "header_levels": 3
}

REGLES:
- Si une cellule est vide, utilise ""
- confidence entre 0.0 et 1.0
- is_complex_layout = true si headers multi-niveaux ou cellules fusionnees
- header_levels = nombre de lignes d'en-tete (1, 2, 3, 4)
- first_column_labels = TOUS les libelles de la premiere colonne (pour matching)
"""

EXTRACTION_SYSTEM_PROMPT_LABELS_ONLY = """Tu es un expert en extraction de donnees de rapports bancaires canadiens (BNC, BMO, CIBC, TD, RBC, BNS).

TACHE: Extraire UNIQUEMENT le texte de la premiere colonne de ce tableau (indicateurs, categories, libelles).

INSTRUCTIONS:
1. Ne pas extraire les montants, pourcentages, totaux ni aucune valeur numerique
2. Conserver la hierarchie (indentation) et les libelles bruts tels qu'ils apparaissent
3. Preserver les references de notes (1), (2), etc. dans le texte ; le nettoyage sera fait cote code
4. EXTRAIT le numero de tableau ("TABLEAU 23", "T18") s'il est visible
5. Inclure toutes les lignes: sous-totaux, totaux, lignes indentees

FORMAT DE REPONSE (JSON strict):
{
    "table_number": "28",
    "table_title": "TABLEAU 28: Expositions brutes au risque de credit",
    "first_column_labels": [
        "Titres de participation",
        "Change",
        "Taux d'interet (1)",
        "Risque de credit specifique (2)",
        "VAR des activites de negociation",
        "VAR totale"
    ],
    "confidence": 0.95
}

REGLES:
- first_column_labels = TOUS les libelles de la premiere colonne, dans l'ordre
- confidence entre 0.0 et 1.0
"""


# ==============================================================================
# CLASSE PRINCIPALE
# ==============================================================================


class VisionTableExtractor:
    """
    Extracteur de tableaux Vision-Native utilisant GPT-4o comme moteur principal.

    Workflow:
    1. Recevoir une section ciblee (pages start-end)
    2. Scanner chaque page pour detecter les tableaux
    3. Extraire le contenu de chaque tableau detecte
    4. Valider avec pdfplumber (Garde-Fou)
    5. Retourner les donnees structurees avec preuves visuelles
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        save_proof_images: bool = True,
        output_dir: str | None = None,
        dpi: int = 300,
        labels_only: bool = False,
    ):
        """
        Initialiser l'extracteur Vision-Native.

        Args:
            api_key: Cle API OpenAI (ou depuis env OPENAI_API_KEY)
            model: Modele a utiliser (gpt-4o recommande)
            save_proof_images: Sauvegarder les images des tableaux extraits
            output_dir: Repertoire de sortie pour les images
            dpi: Resolution pour la conversion PDF -> Image
            labels_only: Extraire uniquement la premiere colonne (pas de montants)
        """
        self.model = model
        self.save_proof_images = save_proof_images
        self.output_dir = output_dir or "vision_extraction_output"
        self.dpi = dpi
        self._labels_only = labels_only
        self._client: OpenAI | None = None
        self._api_key = api_key or get_openai_api_key()

        # Compteur de cout
        self._total_tokens_used = 0
        self._api_calls_count = 0

    def _ensure_client(self):
        """Initialisation differee du client OpenAI."""
        if self._client is not None:
            return

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "Package openai requis. Installez avec: pip install openai"
            )

        if not self._api_key:
            raise ValueError("Cle API OpenAI requise. Definissez OPENAI_API_KEY.")

        self._client = OpenAI(api_key=self._api_key)
        logger.info(f"Client OpenAI initialise (modele: {self.model})")

    # --------------------------------------------------------------------------
    # PHASE 1: SNIPER (Integration avec SectionLocator)
    # --------------------------------------------------------------------------

    def extract_from_section(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
        section_name: str = "unknown",
        bank_code: str = "unknown",
    ) -> ExtractionPipelineResult:
        """
        Pipeline complet d'extraction pour une section donnee.

        Args:
            pdf_path: Chemin vers le PDF
            start_page: Page de debut (1-indexed)
            end_page: Page de fin (inclusive)
            section_name: Nom de la section (pour metadata)
            bank_code: Code banque (pour patterns)

        Returns:
            ExtractionPipelineResult avec toutes les donnees
        """
        logger.info(
            f"=== EXTRACTION VISION-NATIVE: {section_name} (Pages {start_page}-{end_page}) ==="
        )

        result = ExtractionPipelineResult(
            pdf_path=pdf_path,
            section_name=section_name,
            pages_processed=[],
            detected_tables=[],
            extracted_tables=[],
            validation_results=[],
            errors=[],
            total_cost_estimate=0.0,
        )

        # Creer le repertoire de sortie
        if self.save_proof_images:
            proof_dir = Path(self.output_dir) / f"{bank_code}_{section_name}"
            proof_dir.mkdir(parents=True, exist_ok=True)
        else:
            proof_dir = None

        try:
            self._ensure_client()

            # Phase 2: Scanner chaque page
            for page_num in range(start_page, end_page + 1):
                result.pages_processed.append(page_num)

                # Convertir la page en image
                page_image = self._pdf_page_to_image(pdf_path, page_num)
                if page_image is None:
                    result.errors.append(f"Echec conversion page {page_num}")
                    continue

                # Scanner pour detecter les tableaux
                detected = self._scan_page_for_tables(page_image, page_num)
                result.detected_tables.extend(detected)

                # Phase 3: Extraire chaque tableau detecte
                for i, table in enumerate(detected):
                    # Decouper l'image du tableau
                    table_image = self._crop_table_image(page_image, table.bbox)

                    # Sauvegarder la preuve visuelle
                    proof_path = None
                    if proof_dir:
                        proof_path = str(proof_dir / f"page{page_num}_table{i}.png")
                        self._save_image(table_image, proof_path)

                    # Extraire les donnees avec GPT-4o
                    extracted = self._extract_table_data(
                        table_image,
                        page_num,
                        table_index=i,
                        context=table.description,
                    )
                    if extracted:
                        extracted.raw_image_path = proof_path
                        result.extracted_tables.append(extracted)

                        # Phase 4: Validation Garde-Fou
                        validation = self._validate_with_pdfplumber(
                            pdf_path, page_num, table.bbox, extracted
                        )
                        result.validation_results.append(validation)

            # Estimer le cout
            result.total_cost_estimate = self._estimate_cost()

            logger.info(
                f"Extraction terminee: {len(result.extracted_tables)} tableaux, "
                f"cout estime: ${result.total_cost_estimate:.2f}"
            )

        except Exception as e:
            logger.error(f"Erreur pipeline: {e}")
            result.errors.append(str(e))

        return result

    # --------------------------------------------------------------------------
    # PHASE 2: SCANNER (Detection GPT-4o)
    # --------------------------------------------------------------------------

    def _scan_page_for_tables(
        self, page_image: bytes, page_num: int
    ) -> list[DetectedTable]:
        """Detecter tous les tableaux sur une page via GPT-4o Vision."""
        try:
            from .vision_image_preprocessor import preprocess_for_vision

            processed = preprocess_for_vision(page_image)
            image_b64 = base64.b64encode(processed).decode("utf-8")

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SCANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyse cette page {page_num} et detecte tous les tableaux.",
                            },
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
                max_completion_tokens=1000,
                temperature=0,
                response_format={"type": "json_object"},
            )

            self._api_calls_count += 1
            self._total_tokens_used += (
                response.usage.total_tokens if response.usage else 0
            )

            data = json.loads(response.choices[0].message.content)

            detected_tables = []
            for t in data.get("tables_detected", []):
                bbox = t.get("bbox", {})
                detected_tables.append(
                    DetectedTable(
                        page_number=page_num,
                        bbox=(
                            bbox.get("x", 0),
                            bbox.get("y", 0),
                            bbox.get("width", 1),
                            bbox.get("height", 1),
                        ),
                        confidence=t.get("confidence", 0.8),
                        table_type=t.get("table_type", "unknown"),
                        description=t.get("description"),
                    )
                )

            logger.info(f"Page {page_num}: {len(detected_tables)} tableaux detectes")
            return detected_tables

        except Exception as e:
            logger.error(f"Erreur scan page {page_num}: {e}")
            return []

    # --------------------------------------------------------------------------
    # PHASE 3: CERVEAU (Extraction GPT-4o)
    # --------------------------------------------------------------------------

    def _extract_table_data(
        self,
        table_image: bytes,
        page_num: int,
        table_index: int,
        context: str | None = None,
    ) -> ExtractedTableData | None:
        """Extraire les donnees structurees d'un tableau via GPT-4o Vision."""
        try:
            from .vision_image_preprocessor import preprocess_for_vision

            processed = preprocess_for_vision(table_image)
            image_b64 = base64.b64encode(processed).decode("utf-8")

            labels_only = self._labels_only
            system_prompt = (
                EXTRACTION_SYSTEM_PROMPT_LABELS_ONLY
                if labels_only
                else EXTRACTION_SYSTEM_PROMPT
            )
            user_prompt = (
                "Extrais uniquement les libelles de la premiere colonne en JSON."
                if labels_only
                else "Extrais toutes les donnees de ce tableau bancaire en JSON."
            )
            if context:
                user_prompt += f"\n\nContexte: {context}"

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
                max_completion_tokens=4096,
                temperature=0,
                response_format={"type": "json_object"},
            )

            self._api_calls_count += 1
            self._total_tokens_used += (
                response.usage.total_tokens if response.usage else 0
            )

            data = json.loads(response.choices[0].message.content)

            if labels_only:
                headers = []
                rows = []
                footnotes = []
            else:
                headers = data.get("headers", [])
                rows = data.get("rows", [])
                footnotes = data.get("footnotes", [])

            return ExtractedTableData(
                table_id=f"vision_p{page_num}_t{table_index}",
                page_number=page_num,
                title=data.get("table_title"),
                headers=headers,
                rows=rows,
                first_column_indicators=data.get("first_column_labels", []),
                footnotes=footnotes,
                confidence=data.get("confidence", 0.8),
                extraction_method="gpt4o_vision",
            )

        except Exception as e:
            logger.error(f"Erreur extraction page {page_num} table {table_index}: {e}")
            return None

    # --------------------------------------------------------------------------
    # PHASE 4: GARDE-FOU (Validation pdfplumber)
    # --------------------------------------------------------------------------

    def _validate_with_pdfplumber(
        self,
        pdf_path: str,
        page_num: int,
        bbox: tuple,
        gpt4o_result: ExtractedTableData,
    ) -> ValidationResult:
        """Valider l'extraction GPT-4o avec pdfplumber."""
        if not PDFPLUMBER_AVAILABLE:
            return ValidationResult(
                is_valid=True,
                gpt4o_rows=len(gpt4o_result.rows),
                pdfplumber_rows=0,
                discrepancies=["pdfplumber non disponible pour validation"],
                recommendation="use_gpt4o",
            )

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()

                # Trouver le tableau le plus proche de la bbox
                pdfplumber_rows = 0
                if tables:
                    # Prendre le tableau avec le plus de lignes (heuristique simple)
                    best_table = max(tables, key=lambda t: len(t) if t else 0)
                    pdfplumber_rows = len(best_table) if best_table else 0

                gpt4o_rows = len(gpt4o_result.rows)

                discrepancies = []
                if abs(gpt4o_rows - pdfplumber_rows) > 2:
                    discrepancies.append(
                        f"Difference de lignes: GPT-4o={gpt4o_rows}, pdfplumber={pdfplumber_rows}"
                    )

                # Recommandation
                if gpt4o_rows >= pdfplumber_rows:
                    recommendation = "use_gpt4o"
                elif pdfplumber_rows > gpt4o_rows * 1.5:
                    recommendation = "manual_review"
                else:
                    recommendation = "use_gpt4o"

                return ValidationResult(
                    is_valid=len(discrepancies) == 0,
                    gpt4o_rows=gpt4o_rows,
                    pdfplumber_rows=pdfplumber_rows,
                    discrepancies=discrepancies,
                    recommendation=recommendation,
                )

        except Exception as e:
            logger.warning(f"Erreur validation pdfplumber: {e}")
            return ValidationResult(
                is_valid=True,
                gpt4o_rows=len(gpt4o_result.rows),
                pdfplumber_rows=0,
                discrepancies=[f"Erreur pdfplumber: {e}"],
                recommendation="use_gpt4o",
            )

    # --------------------------------------------------------------------------
    # UTILITAIRES
    # --------------------------------------------------------------------------

    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> bytes | None:
        """Convertir une page PDF en image PNG (bytes)."""
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF requis pour la conversion PDF -> Image")
            return None

        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num - 1)

            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            png_data = pix.tobytes("png")
            doc.close()

            return png_data

        except Exception as e:
            logger.error(f"Erreur conversion page {page_num}: {e}")
            return None

    def _crop_table_image(self, page_image: bytes, bbox: tuple) -> bytes:
        """Decouper une region de l'image de page avec padding de 3%."""
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(page_image))
            width, height = img.size

            pad = 0.03
            x, y, w, h = bbox
            left = int(max(0.0, x - pad) * width)
            top = int(max(0.0, y - pad) * height)
            right = int(min(1.0, x + w + pad) * width)
            bottom = int(min(1.0, y + h + pad) * height)

            cropped = img.crop((left, top, right, bottom))

            output = io.BytesIO()
            cropped.save(output, format="PNG")
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Erreur crop image: {e}, retourne image complete")
            return page_image

    def _save_image(self, image_data: bytes, path: str):
        """Sauvegarder une image."""
        try:
            with open(path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde image: {e}")

    def _estimate_cost(self) -> float:
        """Estimer le cout OpenAI (approximatif)."""
        # Tarif approximatif GPT-4o Vision: $0.005 per 1K tokens
        return (self._total_tokens_used / 1000) * 0.005


# ==============================================================================
# FONCTION D'INTEGRATION
# ==============================================================================


def extract_tables_vision_native(
    pdf_path: str,
    start_page: int,
    end_page: int,
    section_name: str = "unknown",
    bank_code: str = "unknown",
    api_key: str | None = None,
) -> ExtractionPipelineResult:
    """
    Fonction de convenance pour l'extraction Vision-Native.

    Args:
        pdf_path: Chemin vers le PDF
        start_page: Page de debut (1-indexed)
        end_page: Page de fin (inclusive)
        section_name: Nom de la section
        bank_code: Code banque
        api_key: Cle API OpenAI (optionnel si env)

    Returns:
        ExtractionPipelineResult complet
    """
    extractor = VisionTableExtractor(api_key=api_key)
    return extractor.extract_from_section(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        section_name=section_name,
        bank_code=bank_code,
    )
