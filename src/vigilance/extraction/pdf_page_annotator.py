"""
Annotateur de pages PDF completes pour la validation humaine.

Ce module permet d'annoter les pages PDF completes avec des marqueurs visuels
(cercles, rectangles) pour indiquer les changements detectes dans les tableaux.

Utilise PyMuPDF pour le rendu PDF et PIL pour l'annotation graphique.
"""

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import conditionnel de PyMuPDF
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF non disponible pour l'annotation PDF")

# Import conditionnel de PIL
try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL non disponible pour l'annotation")


# Couleurs pour les annotations
COLORS = {
    "red": (220, 53, 69),  # Rouge pour suppressions
    "green": (40, 167, 69),  # Vert pour ajouts
    "yellow": (255, 193, 7),  # Jaune pour nouvelles idees
    "blue": (0, 123, 255),  # Bleu pour info
}

# Epaisseur des lignes
LINE_WIDTH_CIRCLE = 4
LINE_WIDTH_RECTANGLE = 5
CIRCLE_RADIUS = 20


def render_pdf_page_to_image(
    pdf_path: str | Path, page_number: int, scale: float = 1.5
) -> Image.Image | None:
    """
    Rendre une page PDF en image PIL.

    Args:
        pdf_path: Chemin vers le PDF
        page_number: Numero de page (1-indexed)
        scale: Echelle de rendu

    Returns:
        Image PIL ou None si erreur
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        logger.error("PyMuPDF et PIL requis pour le rendu PDF")
        return None

    try:
        doc = fitz.open(str(pdf_path))
        page_idx = page_number - 1

        if page_idx < 0 or page_idx >= len(doc):
            logger.error(f"Page {page_number} hors limites (1-{len(doc)})")
            doc.close()
            return None

        page = doc[page_idx]
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)

        # Convertir pixmap en PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()
        return img

    except Exception as e:
        logger.error(f"Erreur rendu page {page_number}: {e}")
        return None


def draw_circle_on_image(
    img: Image.Image, x: int, y: int, radius: int = CIRCLE_RADIUS, color: str = "red"
) -> None:
    """
    Dessiner un cercle sur une image PIL.

    Args:
        img: Image PIL
        x: Position X du centre
        y: Position Y du centre
        radius: Rayon du cercle
        color: Couleur ("red", "green", etc.)
    """
    if not PIL_AVAILABLE:
        return

    rgb = COLORS.get(color, COLORS["red"])
    draw = ImageDraw.Draw(img)

    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], outline=rgb, width=LINE_WIDTH_CIRCLE
    )


def draw_rectangle_on_image(
    img: Image.Image, bbox: tuple[float, float, float, float], color: str = "red", padding: int = 5
) -> None:
    """
    Dessiner un rectangle autour d'une zone sur une image PIL.

    Args:
        img: Image PIL
        bbox: Bounding box (x0, y0, x1, y1) en coordonnees PDF
        color: Couleur
        padding: Marge interieure
    """
    if not PIL_AVAILABLE:
        return

    rgb = COLORS.get(color, COLORS["red"])
    draw = ImageDraw.Draw(img)

    # Convertir bbox PDF en coordonnees image
    # Note: bbox est en coordonnees PDF, on doit les convertir selon le scale
    x0, y0, x1, y1 = bbox

    # Ajuster avec padding
    x0 = max(0, int(x0) - padding)
    y0 = max(0, int(y0) - padding)
    x1 = min(img.width, int(x1) + padding)
    y1 = min(img.height, int(y1) + padding)

    draw.rectangle([x0, y0, x1, y1], outline=rgb, width=LINE_WIDTH_RECTANGLE)


def annotate_pdf_page_with_table_changes(
    pdf_path: str | Path,
    page_number: int,
    table_bboxes: list[dict[str, Any]],
    changes_to_mark: list[dict[str, Any]],
    scale: float = 1.5,
    mark_entire_tables: bool = True,
    mark_rows: bool = True,
) -> bytes | None:
    """
    Annoter une page PDF complete avec les changements detectes.

    Args:
        pdf_path: Chemin vers le PDF
        page_number: Numero de page (1-indexed)
        table_bboxes: Liste de dicts avec bbox des tableaux sur cette page
                     Format: [{"bbox": (x0, y0, x1, y1), "table_title": "...", ...}]
        changes_to_mark: Liste des changements a marquer
                        Format: [{"change_type": "ajoute"/"supprime", "phrase": "...",
                                 "table_bbox": (x0, y0, x1, y1), "row_y": float, ...}]
        scale: Echelle de rendu
        mark_entire_tables: Marquer les tableaux entiers avec rectangle
        mark_rows: Marquer les lignes avec cercles

    Returns:
        Bytes de l'image annotee (PNG) ou None si erreur
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        logger.error("PyMuPDF et PIL requis pour l'annotation")
        return None

    # Rendre la page en image
    img = render_pdf_page_to_image(pdf_path, page_number, scale)
    if img is None:
        return None

    # Obtenir les dimensions de la page PDF originale pour conversion
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        doc.close()
    except Exception as e:
        logger.warning(f"Impossible d'obtenir dimensions PDF: {e}")
        pdf_width = img.width / scale
        pdf_height = img.height / scale

    # Facteurs de conversion PDF -> Image
    x_ratio = img.width / pdf_width
    y_ratio = img.height / pdf_height

    # Grouper changements par tableau
    changes_by_table = {}
    for change in changes_to_mark:
        table_bbox = change.get("table_bbox")
        if table_bbox:
            table_key = str(table_bbox)
            if table_key not in changes_by_table:
                changes_by_table[table_key] = []
            changes_by_table[table_key].append(change)

    # Annoter chaque tableau
    for table_bbox_dict in table_bboxes:
        bbox_pdf = table_bbox_dict.get("bbox")
        if not bbox_pdf:
            continue

        # Convertir bbox PDF en coordonnees image
        x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox_pdf
        x0_img = int(x0_pdf * x_ratio)
        y0_img = int(y0_pdf * y_ratio)
        x1_img = int(x1_pdf * x_ratio)
        y1_img = int(y1_pdf * y_ratio)

        # Trouver les changements pour ce tableau
        table_key = str(bbox_pdf)
        table_changes = changes_by_table.get(table_key, [])

        # Determiner la couleur selon le type de changement
        has_additions = any(c.get("change_type") == "ajoute" for c in table_changes)
        has_removals = any(c.get("change_type") == "supprime" for c in table_changes)

        if mark_entire_tables:
            # Marquer le tableau entier
            if has_additions and has_removals:
                # Tableau avec ajouts et suppressions: jaune
                draw_rectangle_on_image(img, (x0_img, y0_img, x1_img, y1_img), "yellow")
            elif has_additions:
                # Tableau avec ajouts: vert
                draw_rectangle_on_image(img, (x0_img, y0_img, x1_img, y1_img), "green")
            elif has_removals:
                # Tableau avec suppressions: rouge
                draw_rectangle_on_image(img, (x0_img, y0_img, x1_img, y1_img), "red")

        # Marquer les lignes specifiques
        if mark_rows:
            for change in table_changes:
                change_type = change.get("change_type", "ajoute")
                row_y = change.get("row_y")

                if row_y is not None:
                    # Convertir position Y PDF -> Image
                    row_y_img = int(row_y * y_ratio)

                    # Position X: a gauche du tableau
                    marker_x = x0_img - CIRCLE_RADIUS - 10
                    marker_y = row_y_img

                    # Couleur selon type
                    color = "green" if change_type == "ajoute" else "red"
                    draw_circle_on_image(img, marker_x, marker_y, CIRCLE_RADIUS, color)

    # Convertir image en bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def find_text_position(
    pdf_path: str | Path, page_number: int, search_text: str, scale: float = 1.5
) -> tuple[int, int] | None:
    """
    Chercher un texte dans une page PDF et retourner sa position.

    Utilise plusieurs strategies de recherche pour maximiser les chances
    de trouver le texte (texte complet, mots cles, variantes).

    Args:
        pdf_path: Chemin vers le PDF
        page_number: Numero de page (1-indexed)
        search_text: Texte a chercher
        scale: Echelle appliquee a l'image

    Returns:
        Tuple (x, y) en coordonnees image ou None si non trouve
    """
    if not PYMUPDF_AVAILABLE:
        return None

    try:
        doc = fitz.open(str(pdf_path))
        page_idx = page_number - 1

        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None

        page = doc[page_idx]

        # Preparer les variantes de recherche
        search_variants = []

        # 1. Texte original (limite a 30 caracteres)
        clean_text = search_text.strip()
        search_variants.append(clean_text[:30] if len(clean_text) > 30 else clean_text)

        # 2. Premiers mots significatifs (2-3 mots)
        words = clean_text.split()
        if len(words) >= 2:
            search_variants.append(" ".join(words[:2]))
        if len(words) >= 3:
            search_variants.append(" ".join(words[:3]))

        # 3. Premier mot seul (si assez long et pas commun)
        if words and len(words[0]) >= 4:
            common_words = {"total", "pour", "avec", "dans", "cette", "autres"}
            if words[0].lower() not in common_words:
                search_variants.append(words[0])

        # 4. Sans les parentheses et numeros de reference
        import re

        clean_no_refs = re.sub(r"\s*\([^)]*\)\s*", " ", clean_text).strip()
        if clean_no_refs and clean_no_refs != clean_text:
            search_variants.append(clean_no_refs[:30])

        # Eliminer les doublons tout en preservant l'ordre
        seen = set()
        unique_variants = []
        for v in search_variants:
            if v and v not in seen:
                seen.add(v)
                unique_variants.append(v)

        # Essayer chaque variante
        for variant in unique_variants:
            text_instances = page.search_for(variant)

            if text_instances:
                # Prendre la premiere occurrence
                rect = text_instances[0]
                # Calculer le centre Y de la bbox
                y_pdf = (rect.y0 + rect.y1) / 2
                x_pdf = rect.x0

                # Convertir en coordonnees image (appliquer le scale)
                x_img = int(x_pdf * scale)
                y_img = int(y_pdf * scale)

                doc.close()
                logger.debug(f"Texte trouve avec variante '{variant}' a ({x_img}, {y_img})")
                return (x_img, y_img)

        doc.close()
        return None

    except Exception as e:
        logger.warning(f"Erreur recherche texte '{search_text}': {e}")
        return None


def annotate_pdf_page_simple(
    pdf_path: str | Path, page_number: int, changes: list[dict[str, Any]], scale: float = 1.5
) -> bytes | None:
    """
    Annoter une page avec des changements en utilisant la position exacte du texte.

    Utilise PyMuPDF pour chercher le texte de chaque indicateur et positionner
    le cercle precisement a cote de la ligne correspondante.

    Args:
        pdf_path: Chemin vers le PDF
        page_number: Numero de page
        changes: Liste des changements a marquer
                 Format: [{"change_type": "ajoute/supprime", "element": "texte..."}]
        scale: Echelle de rendu

    Returns:
        Bytes de l'image annotee
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        return None

    # Rendre la page
    img = render_pdf_page_to_image(pdf_path, page_number, scale)
    if img is None:
        return None

    num_changes = len(changes)
    if num_changes == 0:
        # Pas de changements, retourner l'image telle quelle
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    # Compteur de positions trouvees vs heuristiques
    found_count = 0
    fallback_positions = []

    for i, change in enumerate(changes):
        change_type = change.get("change_type", "ajoute")
        element = change.get("element", "")
        color = "green" if change_type == "ajoute" else "red"

        # Extraire le texte de l'indicateur (pour les renommages, prendre la partie avant ->)
        if "->" in element:
            # Format: "Ancien texte -> Nouveau texte"
            parts = element.split("->")
            indicator_text = parts[0].strip() if change_type == "supprime" else parts[-1].strip()
        else:
            indicator_text = element.strip()

        # Chercher la position exacte du texte
        position = find_text_position(pdf_path, page_number, indicator_text, scale)

        if position:
            x_img, y_img = position
            # Dessiner le cercle a gauche de la ligne trouvee
            x_pos = max(30, x_img - CIRCLE_RADIUS - 10)
            draw_circle_on_image(img, x_pos, y_img, CIRCLE_RADIUS, color)
            found_count += 1
            logger.debug(f"Position trouvee pour '{indicator_text}': ({x_pos}, {y_img})")
        else:
            # Garder pour fallback heuristique
            fallback_positions.append((i, color))
            logger.debug(f"Position non trouvee pour '{indicator_text}', fallback heuristique")

    # Fallback heuristique pour les textes non trouves
    if fallback_positions:
        section_height = img.height / (num_changes + 1)
        for idx, (i, color) in enumerate(fallback_positions):
            y_pos = int(section_height * (i + 1))
            x_pos = 30
            draw_circle_on_image(img, x_pos, y_pos, CIRCLE_RADIUS, color)

    logger.info(
        f"Annotations: {found_count}/{num_changes} positions exactes, {len(fallback_positions)} heuristiques"
    )

    # Convertir en bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def create_side_by_side_annotated_pages(
    pdf_path_t1: str | Path,
    pdf_path_t2: str | Path,
    page_number: int,
    changes_t1: list[dict[str, Any]],
    changes_t2: list[dict[str, Any]],
    table_bboxes_t1: list[dict[str, Any]] | None = None,
    table_bboxes_t2: list[dict[str, Any]] | None = None,
    scale: float = 1.5,
    gap: int = 20,
) -> bytes | None:
    """
    Creer une image cote-a-cote avec pages T1 et T2 annotees.

    Args:
        pdf_path_t1: PDF T1
        pdf_path_t2: PDF T2
        page_number: Numero de page (meme page pour T1 et T2)
        changes_t1: Changements a marquer sur T1 (suppressions)
        changes_t2: Changements a marquer sur T2 (ajouts)
        table_bboxes_t1: Bboxes des tableaux sur T1
        table_bboxes_t2: Bboxes des tableaux sur T2
        scale: Echelle de rendu
        gap: Espace entre les deux images

    Returns:
        Image combinee en bytes
    """
    if not PIL_AVAILABLE:
        return None

    # Annoter T1
    annotated_t1 = None
    if pdf_path_t1:
        if table_bboxes_t1:
            annotated_t1 = annotate_pdf_page_with_table_changes(
                pdf_path_t1, page_number, table_bboxes_t1, changes_t1, scale
            )
        else:
            annotated_t1 = annotate_pdf_page_simple(pdf_path_t1, page_number, changes_t1, scale)

    # Annoter T2
    annotated_t2 = None
    if pdf_path_t2:
        if table_bboxes_t2:
            annotated_t2 = annotate_pdf_page_with_table_changes(
                pdf_path_t2, page_number, table_bboxes_t2, changes_t2, scale
            )
        else:
            annotated_t2 = annotate_pdf_page_simple(pdf_path_t2, page_number, changes_t2, scale)

    # Combiner les images
    if annotated_t1 and annotated_t2:
        img1 = Image.open(io.BytesIO(annotated_t1))
        img2 = Image.open(io.BytesIO(annotated_t2))

        max_height = max(img1.height, img2.height)
        total_width = img1.width + gap + img2.width

        combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width + gap, 0))

        buffer = io.BytesIO()
        combined.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    elif annotated_t1:
        return annotated_t1
    elif annotated_t2:
        return annotated_t2

    return None
