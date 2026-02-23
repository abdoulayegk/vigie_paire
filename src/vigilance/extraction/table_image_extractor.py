"""
Extracteur d'images de tableaux pour validation GPT-4 Vision.

Extrait les tableaux des PDFs comme images PNG haute resolution
pour permettre une analyse visuelle avec GPT-4 Vision.

Pipeline:
1. Detecter les zones de tableaux avec pdfplumber
2. Convertir chaque zone en image PNG (300 DPI)
3. Appliquer corrections (rotation, contraste)
4. Sauvegarder avec metadata
"""

import base64
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
_PATTERN_LOADER_IMPORT_LOGGED = False

# Imports conditionnels
try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber non disponible")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL non disponible")

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF non disponible, utilisation de pdfplumber")

# Patterns pour filtrer les dates des indicateurs de premiere colonne
# Ces patterns ne sont PAS des indicateurs metier mais des en-tetes temporels
DATE_FILTER_PATTERNS = [
    r"^\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)",
    r"^(au\s+)?\d{1,2}\s+(janv|fév|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)",
    r"^Au\s+\d",  # "Au 31 octobre" mais pas "Autres actifs"
    r"^Pour\s+(le|la|l')",  # "Pour le trimestre", "Pour la période", "Pour l'exercice"
    r"^Trimestre\b",  # "Trimestre terminé le..."
    r"^\d{4}$",  # Annee seule (ex: "2024", "2025")
    r"^T[1-4]\s+\d{4}",  # Ex: "T1 2025"
    r"^(premier|deuxieme|troisieme|quatrieme)\s+trimestre",
]

# Pattern pour detecter le debut d'un tableau financier
# Utilise pour separer plusieurs tableaux sur une meme page
TABLE_START_PATTERN = r"\(en\s+(millions|milliards)\s+de\s+dollars(\s+canadiens)?[^)]*\)"


def _load_extraction_patterns(bank_code: str | None = None):
    """Load optional pattern config without hard-failing when unavailable."""
    global _PATTERN_LOADER_IMPORT_LOGGED
    try:
        from vigilance.utils.pattern_loader import get_patterns
    except Exception as exc:
        if not _PATTERN_LOADER_IMPORT_LOGGED:
            logger.debug("Pattern loader indisponible, fallback par defaut utilise: %s", exc)
            _PATTERN_LOADER_IMPORT_LOGGED = True
        return None

    try:
        return get_patterns(bank_code=bank_code)
    except Exception as exc:
        logger.warning("Impossible de charger les patterns configurables: %s", exc)
        return None


@dataclass
class TableImage:
    """Represente une image de tableau extraite."""

    image_path: str
    image_base64: str
    page_number: int
    table_index: int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1

    # Metadata
    title: str = ""
    section: str = ""
    width: int = 0
    height: int = 0

    # Indicateurs extraits (premiere colonne uniquement, pour matching et comparaison)
    first_column_indicators: list[str] = field(default_factory=list)
    # En-tetes de colonnes (si extraits, pour match_signals header_overlap)
    headers: list[str] = field(default_factory=list)

    # Bounding boxes des lignes pour annotation precise
    # Liste de tuples (indicator, y_min, y_max) en pixels image
    row_bboxes: list[tuple[str, float, float]] = field(default_factory=list)

    # Flags
    was_rotated: bool = False
    extraction_method: str = "pdfplumber"

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "bbox": self.bbox,
            "title": self.title,
            "section": self.section,
            "width": self.width,
            "height": self.height,
            "first_column_indicators": self.first_column_indicators,
            "headers": self.headers,
            "row_bboxes": self.row_bboxes,
            "was_rotated": self.was_rotated,
            "extraction_method": self.extraction_method,
        }


@dataclass
class ExtractionResult:
    """Resultat de l'extraction d'images de tableaux."""

    pdf_path: str
    output_dir: str
    table_images: list[TableImage]
    total_pages: int
    pages_processed: int
    tables_extracted: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pdf_path": self.pdf_path,
            "output_dir": self.output_dir,
            "total_pages": self.total_pages,
            "pages_processed": self.pages_processed,
            "tables_extracted": self.tables_extracted,
            "table_images": [t.to_dict() for t in self.table_images],
            "errors": self.errors,
        }


class TableImageExtractor:
    """
    Extrait les tableaux des PDFs comme images PNG.

    Utilise PyMuPDF (fitz) pour une meilleure qualite,
    avec fallback sur pdfplumber.
    """

    # Resolution pour l'extraction (DPI)
    DEFAULT_DPI = 300
    VISION_DPI = 150  # Suffisant pour GPT-4 Vision

    # Padding autour des tableaux (pixels)
    TABLE_PADDING = 20

    def __init__(
        self,
        output_dir: str | None = None,
        dpi: int = 150,
        save_images: bool = True,
        bank_code: str | None = None,
    ):
        """
        Initialiser l'extracteur.

        Args:
            output_dir: Repertoire de sortie pour les images
            dpi: Resolution d'extraction (defaut 150 pour Vision)
            save_images: Sauvegarder les images sur disque
            bank_code: Code de la banque pour utiliser les patterns specifiques
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.save_images = save_images
        self.bank_code = bank_code

        # Charger les patterns configurables
        self.patterns = _load_extraction_patterns(bank_code=bank_code)
        if self.patterns is not None:
            logger.debug(f"Patterns charges pour banque: {bank_code or 'default'}")

        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber requis. Installez avec: pip install pdfplumber")
        if not PIL_AVAILABLE:
            raise ImportError("PIL requis. Installez avec: pip install Pillow")

    def extract_table_images(
        self, pdf_path: str, start_page: int = 1, end_page: int | None = None, section: str = ""
    ) -> ExtractionResult:
        """
        Extraire les tableaux comme images PNG.

        Args:
            pdf_path: Chemin vers le PDF
            start_page: Page de debut (1-indexed)
            end_page: Page de fin (inclusive)
            section: Nom de la section (pour metadata)

        Returns:
            ExtractionResult avec les images extraites
        """
        pdf_path = str(pdf_path)
        table_images = []
        errors = []

        # Creer le repertoire de sortie
        if self.save_images:
            output_dir = self._get_output_dir(pdf_path, section)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = ""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                end_page = end_page or total_pages

                # Valider les pages
                start_page = max(1, min(start_page, total_pages))
                end_page = max(start_page, min(end_page, total_pages))

                pages_processed = 0

                for page_num in range(start_page, end_page + 1):
                    try:
                        page = pdf.pages[page_num - 1]  # 0-indexed
                        page_images = self._extract_tables_from_page(
                            pdf_path=pdf_path,
                            page=page,
                            page_num=page_num,
                            output_dir=output_dir,
                            section=section,
                        )
                        table_images.extend(page_images)
                        pages_processed += 1

                        # Logger le nombre de tableaux trouves par page
                        if page_images:
                            logger.debug(
                                f"Page {page_num}: {len(page_images)} tableau(x) extrait(s)"
                            )
                        else:
                            logger.debug(f"Page {page_num}: aucun tableau detecte")

                    except Exception as e:
                        error_msg = f"Erreur page {page_num}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                logger.info(
                    f"Extraction terminee: {len(table_images)} tableaux "
                    f"de {pages_processed} pages (plage {start_page}-{end_page})"
                )

                if not table_images and pages_processed > 0:
                    logger.warning(
                        f"Aucun tableau extrait des pages {start_page}-{end_page}. "
                        f"Verifiez que les pages contiennent bien des tableaux detectables."
                    )

        except Exception as e:
            error_msg = f"Erreur ouverture PDF: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return ExtractionResult(
                pdf_path=pdf_path,
                output_dir=output_dir,
                table_images=[],
                total_pages=0,
                pages_processed=0,
                tables_extracted=0,
                errors=errors,
            )

        return ExtractionResult(
            pdf_path=pdf_path,
            output_dir=output_dir,
            table_images=table_images,
            total_pages=total_pages,
            pages_processed=pages_processed,
            tables_extracted=len(table_images),
            errors=errors,
        )

    def _extract_tables_from_page(
        self, pdf_path: str, page, page_num: int, output_dir: str, section: str
    ) -> list[TableImage]:
        """Extraire les tableaux d'une page."""
        table_images = []

        # Detecter les tableaux avec pdfplumber
        tables = page.find_tables()
        extraction_method = "pdfplumber"

        # Detecter les marqueurs de tableau "(en millions de dollars...)"
        # pour identifier s'il y a plusieurs tableaux sur la page
        table_markers = self._detect_table_markers(page)
        num_markers = len(table_markers)

        # Si pdfplumber ne trouve rien, essayer PyMuPDF comme fallback
        if not tables and PYMUPDF_AVAILABLE:
            logger.info(
                f"pdfplumber n'a trouve aucun tableau page {page_num}, essai avec PyMuPDF..."
            )
            pymupdf_tables = self._extract_with_pymupdf(pdf_path, page_num, output_dir, section)
            if pymupdf_tables:
                logger.info(f"PyMuPDF a trouve {len(pymupdf_tables)} tableau(x) page {page_num}")
                return pymupdf_tables
            else:
                logger.warning(f"Aucun tableau trouve page {page_num} avec pdfplumber ni PyMuPDF")
                return []

        if not tables:
            logger.debug(f"Aucun tableau trouve page {page_num} avec pdfplumber")
            return []

        # Verifier si pdfplumber a fusionne plusieurs tableaux
        if num_markers > len(tables):
            logger.warning(
                f"Page {page_num}: {num_markers} marqueurs '(en millions...)' detectes "
                f"mais seulement {len(tables)} tableau(x) trouve(s) par pdfplumber. "
                f"Possible fusion de tableaux."
            )

        # Convertir la page en image
        page_image = self._page_to_image(page)

        for table_idx, table in enumerate(tables):
            try:
                # Obtenir les coordonnees du tableau
                bbox = table.bbox  # (x0, y0, x1, y1)

                # Extraire l'image du tableau
                table_img = self._crop_table_image(
                    page_image=page_image, bbox=bbox, page_width=page.width, page_height=page.height
                )

                if table_img is None:
                    continue

                # Extraire les indicateurs (1ere colonne) et leurs positions
                indicators, row_bboxes = self._extract_indicators_with_positions(
                    table=table,
                    table_bbox=bbox,
                    page_width=page.width,
                    page_height=page.height,
                    image_width=table_img.width,
                    image_height=table_img.height,
                )

                # Extraire les en-tetes de colonnes (premiere ligne de table.extract())
                headers = self._extract_table_headers(table)

                # Detecter le titre
                title = self._detect_table_title(page, bbox)

                # Generer le nom de fichier
                filename = (
                    f"{section}_{page_num}_{table_idx}.png"
                    if section
                    else f"table_p{page_num}_{table_idx}.png"
                )
                filename = filename.replace(" ", "_").replace("/", "_")

                # Sauvegarder si demande
                image_path = ""
                if self.save_images and output_dir:
                    image_path = os.path.join(output_dir, filename)
                    table_img.save(image_path, "PNG")

                # Convertir en base64
                img_base64 = self._image_to_base64(table_img)

                table_images.append(
                    TableImage(
                        image_path=image_path,
                        image_base64=img_base64,
                        page_number=page_num,
                        table_index=table_idx,
                        bbox=bbox,
                        title=title,
                        section=section,
                        width=table_img.width,
                        height=table_img.height,
                        first_column_indicators=indicators,
                        headers=headers,
                        row_bboxes=row_bboxes,
                        was_rotated=False,
                        extraction_method=extraction_method,
                    )
                )

            except Exception as e:
                logger.warning(f"Erreur extraction table {table_idx} page {page_num}: {e}")

        return table_images

    def _page_to_image(self, page) -> Image.Image:
        """Convertir une page PDF en image PIL."""
        # Utiliser pdfplumber pour la conversion
        img = page.to_image(resolution=self.dpi)

        # Convertir en PIL Image
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return Image.open(img_buffer).convert("RGB")

    def _crop_table_image(
        self,
        page_image: Image.Image,
        bbox: tuple[float, float, float, float],
        page_width: float,
        page_height: float,
    ) -> Image.Image | None:
        """Decouper l'image du tableau avec padding."""
        try:
            # Calculer le ratio de conversion
            img_width, img_height = page_image.size
            x_ratio = img_width / page_width
            y_ratio = img_height / page_height

            # Convertir les coordonnees
            x0 = int(bbox[0] * x_ratio) - self.TABLE_PADDING
            y0 = int(bbox[1] * y_ratio) - self.TABLE_PADDING
            x1 = int(bbox[2] * x_ratio) + self.TABLE_PADDING
            y1 = int(bbox[3] * y_ratio) + self.TABLE_PADDING

            # Limiter aux bords de l'image
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(img_width, x1)
            y1 = min(img_height, y1)

            # Verifier la taille minimale
            if x1 - x0 < 50 or y1 - y0 < 30:
                return None

            return page_image.crop((x0, y0, x1, y1))

        except Exception as e:
            logger.warning(f"Erreur crop: {e}")
            return None

    def _extract_table_headers(self, table) -> list[str]:
        """Extraire les en-tetes de colonnes (premiere ligne de table.extract())."""
        try:
            extracted = table.extract() if hasattr(table, "extract") else []
            if not extracted or len(extracted) < 1:
                return []
            first_row = extracted[0]
            if isinstance(first_row, (list, tuple)):
                return [str(c).strip() for c in first_row if c is not None and str(c).strip()]
            return []
        except Exception:
            return []

    def _extract_indicators_with_positions(
        self,
        table,
        table_bbox: tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[list[str], list[tuple[str, float, float]]]:
        """
        Extraire les indicateurs avec leurs positions Y dans l'image du tableau.

        Utilise les cellules pdfplumber pour obtenir les positions reelles.
        Filtre strict pour ne garder que la PREMIERE colonne et ignorer les en-tetes.

        Args:
            table: Objet table pdfplumber
            table_bbox: Bounding box du tableau dans le PDF (x0, y0, x1, y1)
            page_width: Largeur de la page PDF
            page_height: Hauteur de la page PDF
            image_width: Largeur de l'image extraite
            image_height: Hauteur de l'image extraite

        Returns:
            Tuple (liste indicateurs, liste (indicateur, y_min, y_max) en pixels image)
        """
        import re

        indicators = []
        row_bboxes = []

        try:
            # Obtenir les cellules du tableau pour avoir les positions
            cells = table.cells if hasattr(table, "cells") else []
            extracted = table.extract()

            if not extracted:
                return [], []

            # Calculer le ratio de conversion PDF -> Image
            table_x0, table_y0, table_x1, table_y1 = table_bbox
            table_pdf_height = table_y1 - table_y0

            # Le padding ajoute dans crop
            padding = self.TABLE_PADDING

            # Regrouper les cellules par ligne (meme y0 approximativement)
            rows_cells = {}  # y0 -> liste de cellules
            for cell in cells:
                cell_x0, cell_y0, cell_x1, cell_y1 = cell
                # Arrondir y0 pour regrouper les cellules de la meme ligne
                y_key = round(cell_y0, 1)
                if y_key not in rows_cells:
                    rows_cells[y_key] = []
                rows_cells[y_key].append(cell)

            # Trier par y0 pour avoir l'ordre des lignes
            sorted_y_keys = sorted(rows_cells.keys())

            # Identifier la ligne d'en-tete d'unite (ex: "(en millions de dollars...)")
            # On ignorera tout ce qui est AVANT ou SUR cette ligne
            start_processing = True  # Par defaut on traite tout

            # Utiliser les patterns configurables si disponibles
            if self.patterns:
                header_patterns = [
                    p.pattern for p in self.patterns.table_extraction.header_patterns
                ]
            else:
                # Fallback sur les patterns par defaut
                header_patterns = [
                    r"\(?en\s+millions",
                    r"millions\s+de\s+dollars",
                    r"\(?en\s+\%\)?",
                    r"\(?en\s+\$\)?",
                    r"thousands\s+of\s+dollars",
                    r"\(?en\s+milliers",
                    r"milliers\s+de\s+dollars",
                    r"\bM\$\b",
                    r"\(?en\s+milliards",
                    r"milliards\s+de\s+dollars",
                ]

            # Determiner la colonne "Indicateurs"
            # C'est generalement la colonne la plus a gauche (x0 minimum)
            # On calcule le x0 min de toutes les cellules pour definir le "bord gauche"
            min_x0 = min(c[0] for c in cells) if cells else 0
            # Tolerance pour etre considere comme "premiere colonne" (configurable)
            col_x_tolerance = self.patterns.table_extraction.x_tolerance if self.patterns else 10.0

            # Pour chaque ligne extraite
            row_idx = 0
            header_found = False

            for y_key in sorted_y_keys:
                if row_idx >= len(extracted):
                    break

                row = extracted[row_idx]
                row_cells = rows_cells[y_key]
                row_idx += 1

                if not row:
                    continue

                # Verifier si c'est une ligne d'en-tete d'unite
                row_text = " ".join(str(c) for c in row if c).lower()
                is_header_line = any(re.search(p, row_text) for p in header_patterns)

                if is_header_line:
                    header_found = True
                    start_processing = False  # On le reactivera APRES cette ligne
                    continue  # On saute la ligne d'en-tete elle-meme

                # Si on a trouve un header et on est a la ligne suivante, on commence/reprend
                if header_found and not start_processing:
                    start_processing = True

                # Si on n'a pas encore trouve de header mais qu'on est au debut,
                # on verifie si la premiere cellule ressemble a un titre de colonne ("Montants utilises", etc.)
                # Heuristique: Si c'est en gras ou majuscules (difficile sans info de police), ou mots-cles
                first_cell_text = str(row[0]).strip() if row and row[0] else ""
                if not header_found and row_idx < 5:  # Seulement les 5 premieres lignes
                    if first_cell_text.lower() in [
                        "montants utilisés",
                        "total",
                        "particuliers",
                        "entreprises",
                    ]:
                        # Si c'est un mot qui ressemble a une categorie, c'est bon, on garde
                        pass
                    elif not first_cell_text:
                        continue  # Ligne vide

                # Trouver la cellule de la premiere colonne
                # Elle doit etre alignee a gauche (proche de min_x0)
                first_col_cells = [c for c in row_cells if abs(c[0] - min_x0) < col_x_tolerance]

                if first_col_cells:
                    cell = first_col_cells[0]
                    # Indice dans la ligne extraite (pdfplumber peut avoir fusionne ou decale)
                    # On cherche la valeur textuelle correspondante

                    # On re-extrait le texte de cette cellule specifique pour etre sur
                    # Car extracted[row_idx] peut ne pas correspondre 1:1 si cellules fusionnees
                    # Mais on n'a pas l'objet page ici facilement pour re-extraire...
                    # On va utiliser l'ordre visuel: c'est la cellule la plus a gauche

                    # Chercher quel index dans 'row' correspond a cette cellule x0/y0
                    # C'est approximatif car 'extracted' est une liste de chaînes

                    found_text = None
                    # On suppose que row[0] est la premiere colonne si elle n'est pas vide
                    if row[0] and str(row[0]).strip():
                        found_text = str(row[0]).strip()
                    elif (
                        len(row) > 1
                        and row[1]
                        and str(row[1]).strip()
                        and not first_col_cells[0][0] < min_x0 + 50
                    ):
                        # Parfois la premiere colonne est vide dans 'extracted' mais la cellule existe
                        # Ca arrive avec l'indentation
                        pass

                    # Si on n'a pas trouve via row[0], on ne peut pas deviner le texte
                    if not found_text:
                        continue

                    indicator = found_text

                    # FILTRAGE STRICT (longueur minimale configurable)
                    min_length = (
                        self.patterns.table_extraction.min_indicator_length if self.patterns else 2
                    )

                    # FILTRER LES DATES (ne sont pas des indicateurs metier)
                    # Ex: "31 janvier", "30 avril", "Au 31 octobre", "T1 2025"
                    is_date_pattern = any(
                        re.search(p, indicator, re.IGNORECASE) for p in DATE_FILTER_PATTERNS
                    )

                    if (
                        indicator
                        and not self._is_numeric_only(indicator)
                        and len(indicator) > min_length
                        and start_processing
                        and not is_date_pattern  # Exclure les dates
                    ):  # Seulement si on est dans la zone "data"
                        # Position verticale
                        cell_y0_pdf = cell[1]
                        cell_y1_pdf = cell[3]

                        # Convertir en coordonnees image
                        rel_y0 = cell_y0_pdf - table_y0
                        rel_y1 = cell_y1_pdf - table_y0

                        y_ratio = (image_height - 2 * padding) / table_pdf_height

                        img_y0 = padding + rel_y0 * y_ratio
                        img_y1 = padding + rel_y1 * y_ratio

                        indicators.append(indicator)
                        row_bboxes.append((indicator, img_y0, img_y1))

            # Si on n'a pas pu extraire les positions, fallback sur les indicateurs seuls
            if not row_bboxes and indicators:
                logger.debug("Pas de positions de cellules disponibles, utilisation fallback")

        except Exception as e:
            logger.warning(f"Erreur extraction indicateurs avec positions: {e}")
            # Fallback sur la methode simple
            return self._extract_indicators(table), []

        return indicators, row_bboxes

    def _extract_indicators(self, table) -> list[str]:
        """Extraire les indicateurs (1ere colonne) d'un tableau."""
        import re

        indicators = []

        try:
            extracted = table.extract()
            if not extracted:
                return []

            for row in extracted:
                if row and len(row) > 0 and row[0]:
                    indicator = str(row[0]).strip()
                    # Filtrer les valeurs purement numeriques
                    if indicator and not self._is_numeric_only(indicator):
                        # FILTRER LES DATES (ne sont pas des indicateurs metier)
                        is_date = any(
                            re.search(p, indicator, re.IGNORECASE) for p in DATE_FILTER_PATTERNS
                        )
                        if not is_date:
                            indicators.append(indicator)

        except Exception as e:
            logger.warning(f"Erreur extraction indicateurs: {e}")

        return indicators

    def _is_numeric_only(self, text: str) -> bool:
        """Verifier si un texte est purement numerique."""
        import re

        # Supprimer espaces, virgules, points, %, $
        cleaned = re.sub(r"[\s,$%().-]", "", text)
        return cleaned.isdigit() or not cleaned

    def _detect_table_title(self, page, bbox: tuple) -> str:
        """Detecter le titre au-dessus du tableau."""
        try:
            # Zone au-dessus du tableau
            title_bbox = (bbox[0], max(0, bbox[1] - 50), bbox[2], bbox[1])

            # Extraire le texte de cette zone
            cropped = page.within_bbox(title_bbox)
            text = cropped.extract_text() or ""

            # Nettoyer
            text = text.strip()
            if text and len(text) < 200:  # Titre raisonnable
                # Prendre la premiere ligne
                lines = text.split("\n")
                return lines[0].strip() if lines else ""

        except Exception:
            pass

        return ""

    def _detect_table_markers(self, page) -> list[tuple[float, str]]:
        """
        Detecter les marqueurs de debut de tableau sur une page.

        Cherche les occurrences de '(en millions de dollars...)' et similaires.
        Permet d'identifier combien de tableaux sont sur une page et leurs positions Y.

        Args:
            page: Page pdfplumber

        Returns:
            Liste de tuples (position_y, texte_marqueur) tries par position Y
        """
        import re

        markers = []

        try:
            # Extraire tout le texte avec positions
            words = page.extract_words()

            if not words:
                return []

            # Reconstruire les lignes de texte avec positions Y
            lines_by_y = {}
            for word in words:
                y_key = round(word["top"], 1)
                if y_key not in lines_by_y:
                    lines_by_y[y_key] = {"y": word["top"], "text": []}
                lines_by_y[y_key]["text"].append(word["text"])

            # Chercher le pattern dans chaque ligne
            for y_key, line_data in lines_by_y.items():
                line_text = " ".join(line_data["text"])
                if re.search(TABLE_START_PATTERN, line_text, re.IGNORECASE):
                    markers.append((line_data["y"], line_text))
                    logger.debug(
                        f"Marqueur de tableau trouve a Y={line_data['y']:.1f}: {line_text[:60]}..."
                    )

        except Exception as e:
            logger.debug(f"Erreur detection marqueurs de tableau: {e}")

        # Trier par position Y (haut vers bas)
        return sorted(markers, key=lambda x: x[0])

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convertir une image PIL en base64."""
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _get_output_dir(self, pdf_path: str, section: str) -> str:
        """Generer le chemin du repertoire de sortie."""
        if self.output_dir:
            return self.output_dir

        pdf_name = Path(pdf_path).stem
        base_dir = Path(pdf_path).parent / "table_images"

        if section:
            return str(base_dir / pdf_name / section.replace(" ", "_"))
        return str(base_dir / pdf_name)

    def _extract_with_pymupdf(
        self, pdf_path: str, page_num: int, output_dir: str, section: str
    ) -> list[TableImage]:
        """
        Extraire les tableaux d'une page en utilisant PyMuPDF comme fallback.

        Cette methode est utilisee quand pdfplumber ne trouve pas de tableaux.
        """
        if not PYMUPDF_AVAILABLE:
            return []

        table_images = []

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]

            # PyMuPDF peut detecter les tableaux avec find_tables()
            # Mais cette methode est moins fiable que pdfplumber
            # On va plutot essayer d'extraire toute la page comme image
            # et laisser l'utilisateur voir la page complete

            # Convertir la page en image
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Chercher des zones qui ressemblent a des tableaux
            # En utilisant la detection de texte structure
            text_dict = page.get_text("dict")

            # Essayer de trouver des blocs de texte qui pourraient etre des tableaux
            # (plusieurs lignes avec alignement similaire)
            blocks = text_dict.get("blocks", [])

            # Pour l'instant, on retourne une image de la page complete
            # comme fallback si aucun tableau n'est detecte
            # L'utilisateur pourra au moins voir le contenu

            # Generer le nom de fichier
            filename = (
                f"{section}_{page_num}_fallback.png"
                if section
                else f"table_p{page_num}_fallback.png"
            )
            filename = filename.replace(" ", "_").replace("/", "_")

            # Sauvegarder si demande
            image_path = ""
            if self.save_images and output_dir:
                image_path = os.path.join(output_dir, filename)
                page_image.save(image_path, "PNG")

            # Convertir en base64
            img_base64 = self._image_to_base64(page_image)

            # Creer un TableImage pour la page complete
            # (bbox de toute la page)
            page_width = page.rect.width
            page_height = page.rect.height
            bbox = (0, 0, page_width, page_height)

            # Essayer d'extraire du texte comme indicateurs
            indicators = []
            title = f"Page {page_num}"  # Default title
            try:
                text = page.get_text()
                lines = text.split("\n")[:30]  # Premiere 30 lignes pour chercher titre

                # Chercher un titre de tableau (TABLEAU XX, etc.)
                import re

                for line in lines[:15]:  # Chercher dans les 15 premieres lignes
                    line_stripped = line.strip()
                    # Patterns pour detecter un titre de tableau
                    if re.match(r"^TABLEAU\s*\d+", line_stripped, re.IGNORECASE):
                        title = line_stripped
                        break
                    elif re.match(r"^TABLE\s*\d+", line_stripped, re.IGNORECASE):
                        title = line_stripped
                        break
                    elif re.match(r"^T\d+\s", line_stripped):
                        title = line_stripped
                        break
                    # Titre potentiel: ligne en majuscules avec plus de 20 caracteres
                    elif len(line_stripped) > 20 and line_stripped.isupper():
                        title = line_stripped[:60]  # Limiter la longueur
                        break

                # Extraire indicateurs (1ere colonne)
                indicators = [
                    line.strip() for line in lines if line.strip() and len(line.strip()) > 3
                ]
            except:
                pass

            table_images.append(
                TableImage(
                    image_path=image_path,
                    image_base64=img_base64,
                    page_number=page_num,
                    table_index=0,
                    bbox=bbox,
                    title=title,
                    section=section,
                    width=page_image.width,
                    height=page_image.height,
                    first_column_indicators=indicators[:10],  # Limiter a 10 indicateurs
                    was_rotated=False,
                    extraction_method="pymupdf_fallback",
                )
            )

            doc.close()
            logger.info(f"Extraction PyMuPDF fallback: 1 image creee pour page {page_num}")

        except Exception as e:
            logger.error(f"Erreur extraction PyMuPDF page {page_num}: {e}")

        return table_images


class PyMuPDFTableExtractor(TableImageExtractor):
    """
    Version optimisee utilisant PyMuPDF pour une meilleure qualite.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF non disponible, utilisation de pdfplumber")

    def _page_to_image_pymupdf(self, pdf_path: str, page_num: int) -> Image.Image | None:
        """Convertir une page en image avec PyMuPDF (meilleure qualite)."""
        if not PYMUPDF_AVAILABLE:
            return None

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]

            # Matrice pour la resolution
            zoom = self.dpi / 72  # 72 DPI est la resolution PDF par defaut
            mat = fitz.Matrix(zoom, zoom)

            # Render la page
            pix = page.get_pixmap(matrix=mat)

            # Convertir en PIL
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            doc.close()
            return img

        except Exception as e:
            logger.warning(f"Erreur PyMuPDF: {e}")
            return None


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def extract_tables_as_images(
    pdf_path: str,
    start_page: int = 1,
    end_page: int | None = None,
    section: str = "",
    output_dir: str | None = None,
    dpi: int = 150,
) -> ExtractionResult:
    """
    Fonction utilitaire pour extraire les tableaux comme images.

    Args:
        pdf_path: Chemin vers le PDF
        start_page: Page de debut
        end_page: Page de fin
        section: Nom de la section
        output_dir: Repertoire de sortie
        dpi: Resolution

    Returns:
        ExtractionResult avec les images
    """
    extractor = TableImageExtractor(output_dir=output_dir, dpi=dpi, save_images=bool(output_dir))

    return extractor.extract_table_images(
        pdf_path=pdf_path, start_page=start_page, end_page=end_page, section=section
    )


def get_table_images_for_comparison(
    pdf_path_t1: str, pdf_path_t2: str, start_page: int, end_page: int, section: str = ""
) -> tuple[list[TableImage], list[TableImage]]:
    """
    Extraire les images de tableaux de deux PDFs pour comparaison.

    Returns:
        Tuple (images_t1, images_t2)
    """
    extractor = TableImageExtractor(save_images=False)

    result_t1 = extractor.extract_table_images(
        pdf_path=pdf_path_t1, start_page=start_page, end_page=end_page, section=section
    )

    result_t2 = extractor.extract_table_images(
        pdf_path=pdf_path_t2, start_page=start_page, end_page=end_page, section=section
    )

    return result_t1.table_images, result_t2.table_images
