"""
Annotateur d'images de tableaux pour la visualisation des changements.

Ce module permet de dessiner des annotations visuelles (cercles, rectangles)
sur les images de tableaux pour indiquer les lignes ajoutees ou supprimees.

Utilise PIL (Pillow) pour le dessin graphique.
"""

import base64
import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import conditionnel de PIL
try:
    from PIL import Image, ImageDraw, ImageFont

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
LINE_WIDTH_CIRCLE = 3
LINE_WIDTH_RECTANGLE = 5
CIRCLE_RADIUS = 15


@dataclass
class RowAnnotation:
    """Annotation pour une ligne de tableau."""

    indicator: str  # Texte de l'indicateur
    y_position: float  # Position Y estimee dans l'image
    change_type: str  # "added" ou "removed"
    verified: bool = False  # Si l'utilisateur a verifie

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "y_position": self.y_position,
            "change_type": self.change_type,
            "verified": self.verified,
        }


def image_from_base64(base64_str: str) -> Image.Image | None:
    """
    Convertir une image base64 en objet PIL Image.

    Args:
        base64_str: Image encodee en base64

    Returns:
        Image PIL ou None si erreur
    """
    if not PIL_AVAILABLE:
        logger.error("PIL non disponible")
        return None

    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Erreur decodage base64: {e}")
        return None


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """
    Convertir une image PIL en bytes.

    Args:
        img: Image PIL
        format: Format de sortie (PNG, JPEG)

    Returns:
        Bytes de l'image
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    Convertir une image PIL en base64.

    Args:
        img: Image PIL
        format: Format de sortie

    Returns:
        String base64
    """
    return base64.b64encode(image_to_bytes(img, format)).decode()


def get_row_positions_from_bboxes(
    row_bboxes: list[tuple[str, float, float]],
) -> list[tuple[str, float]]:
    """
    Convertir les bounding boxes de lignes en positions Y centrales.

    Args:
        row_bboxes: Liste de (indicateur, y_min, y_max) en pixels

    Returns:
        Liste de (indicateur, y_center)
    """
    positions = []
    for indicator, y_min, y_max in row_bboxes:
        y_center = (y_min + y_max) / 2
        positions.append((indicator, y_center))
    return positions


def estimate_row_positions(
    indicators: list[str],
    image_height: int,
    header_height: int = 50,
    row_height: int | None = None,
    row_bboxes: list[tuple[str, float, float]] | None = None,
) -> list[tuple[str, float]]:
    """
    Estimer les positions Y des lignes dans une image de tableau.

    Utilise les positions reelles si row_bboxes est fourni,
    sinon fallback sur l'heuristique.

    Args:
        indicators: Liste des indicateurs (premiere colonne)
        image_height: Hauteur totale de l'image
        header_height: Hauteur estimee de l'en-tete (pour fallback)
        row_height: Hauteur fixe par ligne (optionnel, pour fallback)
        row_bboxes: Positions reelles des lignes (indicateur, y_min, y_max)

    Returns:
        Liste de tuples (indicateur, position_y)
    """
    # Utiliser les positions reelles si disponibles
    if row_bboxes and len(row_bboxes) > 0:
        logger.debug(f"Utilisation des positions reelles pour {len(row_bboxes)} lignes")
        return get_row_positions_from_bboxes(row_bboxes)

    if not indicators:
        return []

    logger.debug(f"Fallback sur heuristique pour {len(indicators)} lignes")

    # Hauteur disponible pour les lignes de donnees
    available_height = image_height - header_height

    if row_height:
        # Utiliser hauteur fixe
        positions = []
        for i, indicator in enumerate(indicators):
            y = header_height + (i * row_height) + (row_height / 2)
            positions.append((indicator, y))
    else:
        # Diviser equitablement
        num_rows = len(indicators)
        row_h = available_height / num_rows

        positions = []
        for i, indicator in enumerate(indicators):
            y = header_height + (i * row_h) + (row_h / 2)
            positions.append((indicator, y))

    return positions


def draw_circle(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int = CIRCLE_RADIUS,
    color: str = "red",
) -> None:
    """
    Dessiner un cercle sur l'image.

    Args:
        draw: Objet ImageDraw
        x: Position X du centre
        y: Position Y du centre
        radius: Rayon du cercle
        color: Couleur ("red", "green", etc.)
    """
    rgb = COLORS.get(color, COLORS["red"])

    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=rgb,
        width=LINE_WIDTH_CIRCLE,
    )


def draw_rectangle_border(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    color: str = "red",
    padding: int = 5,
) -> None:
    """
    Dessiner un rectangle autour de l'image entiere.

    Args:
        draw: Objet ImageDraw
        width: Largeur de l'image
        height: Hauteur de l'image
        color: Couleur
        padding: Marge interieure
    """
    rgb = COLORS.get(color, COLORS["red"])

    draw.rectangle(
        [padding, padding, width - padding - 1, height - padding - 1],
        outline=rgb,
        width=LINE_WIDTH_RECTANGLE,
    )


def highlight_row(
    draw: ImageDraw.ImageDraw,
    y_position: float,
    image_width: int,
    color: str = "green",
    opacity: int = 50,
) -> None:
    """
    Surligner une ligne complete du tableau.

    Args:
        draw: Objet ImageDraw
        y_position: Position Y de la ligne
        image_width: Largeur de l'image
        color: Couleur du surlignage
        opacity: Opacite (0-255)
    """
    rgb = COLORS.get(color, COLORS["green"])

    # Dessiner une bande semi-transparente
    row_height = 25  # Hauteur estimee
    y0 = int(y_position - row_height / 2)
    y1 = int(y_position + row_height / 2)

    # Creer une couleur avec alpha
    rgba = (*rgb, opacity)

    # Note: Pour le surlignage, on dessine un rectangle
    draw.rectangle([0, y0, image_width, y1], fill=rgba[:3])


def annotate_table_image(
    image_base64: str,
    rows_to_highlight: list[str],
    all_indicators: list[str],
    color: str = "red",
    is_entire_table: bool = False,
    draw_circles: bool = True,
    highlight_rows: bool = False,
    row_bboxes: list[tuple[str, float, float]] | None = None,
    debug_mode: bool = False,
) -> bytes | None:
    """
    Annoter une image de tableau avec cercles ou surlignages.

    Args:
        image_base64: Image du tableau en base64
        rows_to_highlight: Liste des indicateurs a marquer
        all_indicators: Tous les indicateurs du tableau (pour calcul positions)
        color: Couleur des annotations ("red" ou "green")
        is_entire_table: Si True, encadre le tableau entier
        draw_circles: Dessiner des cercles
        highlight_rows: Surligner les lignes
        row_bboxes: Positions reelles des lignes (indicateur, y_min, y_max)
        debug_mode: Si True, affiche les indices de ligne pour verification

    Returns:
        Bytes de l'image annotee ou None si erreur
    """
    if not PIL_AVAILABLE:
        logger.error("PIL non disponible")
        return None

    # Charger l'image
    img = image_from_base64(image_base64)
    if img is None:
        return None

    # Convertir en RGBA pour transparence
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    draw = ImageDraw.Draw(img)
    width, height = img.size

    if is_entire_table:
        # Encadrer le tableau entier
        draw_rectangle_border(draw, width, height, color)
    else:
        # Calculer positions des lignes (utilise row_bboxes si disponible)
        positions = estimate_row_positions(
            all_indicators, height, row_bboxes=row_bboxes
        )

        # Trouver les indicateurs a marquer
        for idx, (indicator, y_pos) in enumerate(positions):
            is_highlighted = indicator in rows_to_highlight

            if is_highlighted:
                if draw_circles:
                    # Dessiner un cercle a gauche
                    draw_circle(draw, x=25, y=int(y_pos), color=color)

                if highlight_rows:
                    # Surligner la ligne
                    highlight_row(draw, y_pos, width, color)

            # Mode debug: afficher l'index de chaque ligne
            if debug_mode:
                debug_color = (
                    COLORS.get(color, COLORS["red"])
                    if is_highlighted
                    else (128, 128, 128)
                )
                try:
                    # Dessiner l'index de ligne
                    draw.text((5, int(y_pos) - 8), str(idx), fill=debug_color)
                except Exception:
                    pass

    # Reconvertir en RGB pour PNG
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    return image_to_bytes(img)


def annotate_table_with_changes(
    image_base64: str,
    rows_added: list[str],
    rows_removed: list[str],
    all_indicators_t1: list[str],
    all_indicators_t2: list[str],
    is_for_t1: bool = True,
    is_new_table: bool = False,
    is_removed_table: bool = False,
    row_bboxes_t1: list[tuple[str, float, float]] | None = None,
    row_bboxes_t2: list[tuple[str, float, float]] | None = None,
    debug_mode: bool = False,
) -> bytes | None:
    """
    Annoter une image avec tous les changements detectes.

    Pour T1: Cercles rouges sur les lignes supprimees
    Pour T2: Cercles verts sur les lignes ajoutees

    Args:
        image_base64: Image du tableau
        rows_added: Lignes ajoutees (presentes dans T2, pas T1)
        rows_removed: Lignes supprimees (presentes dans T1, pas T2)
        all_indicators_t1: Tous les indicateurs de T1
        all_indicators_t2: Tous les indicateurs de T2
        is_for_t1: True si c'est l'image de T1
        is_new_table: Tableau entierement nouveau
        is_removed_table: Tableau entierement supprime
        row_bboxes_t1: Positions reelles des lignes T1 (indicateur, y_min, y_max)
        row_bboxes_t2: Positions reelles des lignes T2 (indicateur, y_min, y_max)
        debug_mode: Si True, affiche les indices de ligne pour verification

    Returns:
        Bytes de l'image annotee
    """
    if not PIL_AVAILABLE:
        return None

    if is_for_t1:
        # Pour T1: marquer en rouge ce qui a disparu
        if is_removed_table:
            return annotate_table_image(
                image_base64, [], [], color="red", is_entire_table=True
            )
        else:
            return annotate_table_image(
                image_base64,
                rows_to_highlight=rows_removed,
                all_indicators=all_indicators_t1,
                color="red",
                draw_circles=True,
                row_bboxes=row_bboxes_t1,
                debug_mode=debug_mode,
            )
    else:
        # Pour T2: marquer en vert ce qui est nouveau
        if is_new_table:
            return annotate_table_image(
                image_base64, [], [], color="green", is_entire_table=True
            )
        else:
            return annotate_table_image(
                image_base64,
                rows_to_highlight=rows_added,
                all_indicators=all_indicators_t2,
                color="green",
                draw_circles=True,
                row_bboxes=row_bboxes_t2,
                debug_mode=debug_mode,
            )


def annotate_proof_for_review_item(
    image_bytes: bytes,
    change_type: str,
    indicator: str,
    indicators_t1: list[str],
    indicators_t2: list[str],
    is_for_t1: bool = True,
) -> bytes | None:
    """
    Annoter une image de preuve pour un ReviewItem (une seule moitie T1 ou T2).

    Args:
        image_bytes: Image en bytes
        change_type: "added", "removed" ou "renamed"
        indicator: Texte de l'indicateur (ou "old -> new" pour renamed)
        indicators_t1: Tous les indicateurs du tableau T1
        indicators_t2: Tous les indicateurs du tableau T2
        is_for_t1: True si l'image est celle de T1 (gauche), False pour T2 (droite)

    Returns:
        Bytes de l'image annotee
    """
    if not PIL_AVAILABLE:
        return None

    b64 = base64.b64encode(image_bytes).decode()
    rows_to_highlight: list[str] = []
    all_indicators: list[str] = []
    color = "red"

    if change_type == "added":
        if not is_for_t1:
            rows_to_highlight = [indicator]
            all_indicators = indicators_t2
            color = "green"
    elif change_type == "removed":
        if is_for_t1:
            rows_to_highlight = [indicator]
            all_indicators = indicators_t1
            color = "red"
    elif change_type == "renamed":
        parts = indicator.split(" -> ", 1)
        old_label = parts[0].strip() if len(parts) > 0 else ""
        new_label = parts[1].strip() if len(parts) > 1 else ""
        if is_for_t1 and old_label:
            rows_to_highlight = [old_label]
            all_indicators = indicators_t1
            color = "yellow"
        elif not is_for_t1 and new_label:
            rows_to_highlight = [new_label]
            all_indicators = indicators_t2
            color = "yellow"

    if not rows_to_highlight or not all_indicators:
        return image_bytes

    result = annotate_table_image(
        b64,
        rows_to_highlight=rows_to_highlight,
        all_indicators=all_indicators,
        color=color,
        draw_circles=True,
        row_bboxes=None,
    )
    return result if result else image_bytes


def annotate_side_by_side_proof_for_review_item(
    image_bytes: bytes,
    change_type: str,
    indicator: str,
    indicators_t1: list[str],
    indicators_t2: list[str],
    gap: int = 20,
) -> bytes | None:
    """
    Annoter une image cote-a-cote (T1|T2) pour un ReviewItem.

    Decoupe l'image en deux, annote chaque moitie, recombine.
    """
    if not PIL_AVAILABLE:
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        mid = w // 2
        left = img.crop((0, 0, mid, h))
        right = img.crop((mid, 0, w, h))

        left_bytes = image_to_bytes(left)
        right_bytes = image_to_bytes(right)

        annotated_left = annotate_proof_for_review_item(
            left_bytes,
            change_type=change_type,
            indicator=indicator,
            indicators_t1=indicators_t1,
            indicators_t2=indicators_t2,
            is_for_t1=True,
        )
        annotated_right = annotate_proof_for_review_item(
            right_bytes,
            change_type=change_type,
            indicator=indicator,
            indicators_t1=indicators_t1,
            indicators_t2=indicators_t2,
            is_for_t1=False,
        )

        if annotated_left and annotated_right:
            img_left = Image.open(io.BytesIO(annotated_left))
            img_right = Image.open(io.BytesIO(annotated_right))
            max_height = max(img_left.height, img_right.height)
            total_width = img_left.width + gap + img_right.width
            combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))
            combined.paste(img_left, (0, 0))
            combined.paste(img_right, (img_left.width + gap, 0))
            return image_to_bytes(combined)
        elif annotated_left:
            return annotated_left
        elif annotated_right:
            return annotated_right
    except Exception as e:
        logger.warning("Annotation side-by-side echouee: %s", e)

    return image_bytes


def create_side_by_side_comparison(
    image_t1_base64: str | None,
    image_t2_base64: str | None,
    rows_added: list[str],
    rows_removed: list[str],
    all_indicators_t1: list[str],
    all_indicators_t2: list[str],
    gap: int = 20,
    row_bboxes_t1: list[tuple[str, float, float]] | None = None,
    row_bboxes_t2: list[tuple[str, float, float]] | None = None,
    debug_mode: bool = False,
) -> bytes | None:
    """
    Creer une image cote-a-cote avec T1 et T2 annotes.

    Args:
        image_t1_base64: Image T1 en base64 (peut etre None)
        image_t2_base64: Image T2 en base64 (peut etre None)
        rows_added: Lignes ajoutees
        rows_removed: Lignes supprimees
        all_indicators_t1: Indicateurs T1
        all_indicators_t2: Indicateurs T2
        gap: Espace entre les deux images
        row_bboxes_t1: Positions reelles des lignes T1
        row_bboxes_t2: Positions reelles des lignes T2
        debug_mode: Afficher les indices de ligne

    Returns:
        Image combinee en bytes
    """
    if not PIL_AVAILABLE:
        return None

    # Annoter les images
    annotated_t1 = None
    annotated_t2 = None

    if image_t1_base64:
        annotated_t1 = annotate_table_with_changes(
            image_t1_base64,
            rows_added,
            rows_removed,
            all_indicators_t1,
            all_indicators_t2,
            is_for_t1=True,
            row_bboxes_t1=row_bboxes_t1,
            row_bboxes_t2=row_bboxes_t2,
            debug_mode=debug_mode,
        )

    if image_t2_base64:
        annotated_t2 = annotate_table_with_changes(
            image_t2_base64,
            rows_added,
            rows_removed,
            all_indicators_t1,
            all_indicators_t2,
            is_for_t1=False,
            row_bboxes_t1=row_bboxes_t1,
            row_bboxes_t2=row_bboxes_t2,
            debug_mode=debug_mode,
        )

    # Creer image combinee
    if annotated_t1 and annotated_t2:
        img1 = Image.open(io.BytesIO(annotated_t1))
        img2 = Image.open(io.BytesIO(annotated_t2))

        # Calculer dimensions
        max_height = max(img1.height, img2.height)
        total_width = img1.width + gap + img2.width

        # Creer image combinee
        combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width + gap, 0))

        return image_to_bytes(combined)

    elif annotated_t1:
        return annotated_t1
    elif annotated_t2:
        return annotated_t2

    return None
