"""Classification des blocs PDF en texte narratif, tableau, note ou bruit."""

from __future__ import annotations

import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

from .constants import (
    _FOOTNOTE_MARKER_RE,
    _TABLE_HEADING_RE,
    _TABLE_ROW_MARKER_RE,
    _TABLE_VALUE_RE,
)
from .models import PDFBlock, ResolvedSection, SectionAudit
from .text_normalization import _normalized_block_text

def _count_numeric_values(text: str) -> int:
    """Compte le nombre de valeurs numériques dans ``text`` (entiers, décimaux, négatifs)."""
    return len(_TABLE_VALUE_RE.findall(text or ""))


def _contains_dense_numeric_line(text: str) -> bool:
    """Retourne True si au moins une ligne contient plus de 3 valeurs numériques.

    Signe caractéristique d'une ligne de tableau financier (colonnes de chiffres).
    """
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _count_numeric_values(line) > 3:
            return True
    return False


def _looks_like_table_or_financial_grid(text: str) -> bool:
    """Détecte si un bloc ressemble à un tableau ou une grille financière.

    Utilisée pour exclure les blocs non narratifs (tableaux, grilles de notation,
    listes de chiffres) lors de la classification des blocs PDF.
    Critères : en-têtes de tableau, marqueurs de ligne + chiffres, densité numérique
    élevée, ratings, tabulations, colonnes séparées par des espaces multiples.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _TABLE_HEADING_RE.search(value):
        return True
    if _TABLE_ROW_MARKER_RE.search(value) and _count_numeric_values(value) >= 2:
        return True
    if _contains_dense_numeric_line(value) and len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)) <= 20:
        return True
    if re.search(r"\b(?:AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|Aa[123]|A[123]|Baa[123]|FPUNV)\b", value):
        return True
    if "\t" in value or "|" in value:
        return True
    if re.search(r"(?:\b\S+\s+\d+(?:[.,]\d+)?\s*){4,}", value):
        return True
    return False


def _looks_like_footnote(text: str) -> bool:
    """Détecte si un bloc ressemble à une note de bas de page.

    Reconnaît les marqueurs courants : ``(1)``, ``1)``, ``*``, exposants Unicode,
    numéros suivis d'un texte court. Ces blocs sont exclus du contenu narratif
    pour ne pas polluer la comparaison de sections textuelles.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _FOOTNOTE_MARKER_RE.search(value):
        return True
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[¹²³⁴⁵⁶⁷⁸⁹]+\s*", line):
            return True
        if re.match(r"^\d{1,2}\s+", line):
            words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", line)
            if len(words) <= 30:
                return True
    return False


def _looks_like_table_footnote_text(text: str) -> bool:
    """Détecte si un bloc est une légende ou note annexée à un tableau.

    Variante de ``_looks_like_footnote`` pour les blocs situés juste sous un
    tableau (zone inférée par ``_infer_table_footnote_bboxes``). Reconnaît en
    plus les préfixes « s.o. », « note », « source » et les courtes phrases
    mêlant quelques chiffres et parenthèses.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _looks_like_footnote(value):
        return True
    lower_value = value.lower()
    if lower_value.startswith(("s.o.", "n.s.", "sans objet", "note", "source")):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)
    return len(words) <= 30 and _count_numeric_values(value) >= 1 and "(" in value


def _looks_like_narrative_paragraph(text: str) -> bool:
    """Détecte si un bloc est un paragraphe narratif (texte continu à analyser).

    Un paragraphe narratif valide doit contenir ≥ 18 mots, ≥ 120 caractères,
    une forte proportion de lettres, peu de chiffres, et au moins un connecteur
    grammatical ou une ponctuation de fin de phrase. Les tableaux et grilles
    financières sont explicitement rejetés.
    """
    value = str(text or "").strip()
    if not value:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)
    if len(words) < 18 or len(value) < 120:
        return False
    if "\t" in value or "|" in value:
        return False
    if _TABLE_HEADING_RE.search(value):
        return False
    connectors = re.findall(
        r"\b(?:la|le|les|une|un|des|afin|ainsi|mais|et|ou|que|qui|dont|pour|avec|alors|toutefois|cependant|de plus|enfin)\b",
        value,
        flags=re.IGNORECASE,
    )
    sentence_marks = len(re.findall(r"[.;:?!]", value))
    alpha_chars = sum(1 for ch in value if ch.isalpha())
    digit_chars = sum(1 for ch in value if ch.isdigit())
    alpha_ratio = alpha_chars / max(1, len(value))
    digit_ratio = digit_chars / max(1, len(value))
    short_word_ratio = sum(1 for word in words if len(word) <= 3) / max(1, len(words))
    return (
        alpha_ratio >= 0.45
        and digit_ratio <= 0.2
        and short_word_ratio <= 0.5
        and (sentence_marks >= 1 or len(connectors) >= 4)
    )


def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    """Calcule le ratio de chevauchement de la bounding box ``a`` avec ``b``.

    Retourne l'aire de l'intersection divisée par l'aire de ``a`` (valeur entre 0 et 1).
    Un résultat de 1.0 signifie que ``a`` est entièrement contenu dans ``b``.
    Les coordonnées sont normalisées [x0, y0, x1, y1] dans l'espace [0, 1].
    """
    if len(a) < 4 or len(b) < 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area = max(1e-9, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    return inter / area


def _block_overlaps_table(block: PDFBlock, table_bboxes: list[list[float]]) -> bool:
    """Retourne True si le bloc chevauche d'au moins 5 % une des bounding boxes de tableau."""
    return any(_bbox_overlap_ratio(block.bbox_norm, bbox) >= 0.05 for bbox in table_bboxes)


def _infer_table_footnote_bboxes(
    table_bboxes_by_page: dict[int, list[list[float]]],
    *,
    max_height: float = 0.05,
) -> dict[int, list[list[float]]]:
    """Infère les zones de notes de bas de tableau à partir des bounding boxes des tableaux.

    Pour chaque tableau, génère une zone candidate juste en-dessous (hauteur ≤ ``max_height``
    en coordonnées normalisées). Ces zones sont ensuite utilisées par
    ``_classify_block_type`` pour identifier les blocs de légende ou de notes
    annexés aux tableaux et les exclure du contenu narratif.

    Args:
        table_bboxes_by_page: Bounding boxes des tableaux détectés par Docling, par page.
        max_height: Hauteur maximale (normalisée) de la zone note inférée.

    Returns:
        Dictionnaire page → liste de bounding boxes de zones notes potentielles.
    """
    footnote_bboxes_by_page: dict[int, list[list[float]]] = {}
    for page, boxes in table_bboxes_by_page.items():
        ordered = sorted(
            (list(box) for box in boxes if len(box) == 4), key=lambda bbox: (float(bbox[1]), float(bbox[0]))
        )
        page_regions: list[list[float]] = []
        for idx, bbox in enumerate(ordered):
            top = max(0.0, min(1.0, float(bbox[3])))
            next_top = float(ordered[idx + 1][1]) if idx + 1 < len(ordered) else 1.0
            bottom = min(1.0, next_top, top + max_height)
            if bottom - top < 0.01:
                continue
            page_regions.append([0.0, top, 1.0, bottom])
        if page_regions:
            footnote_bboxes_by_page[page] = page_regions
    return footnote_bboxes_by_page


def _section_window_for_page(
    section: ResolvedSection,
    page_number: int,
    next_section: ResolvedSection | None = None,
) -> tuple[float, float]:
    """Calcule la fenêtre verticale (top, bottom) d'une section sur une page donnée.

    Sur la première page de la section, la fenêtre commence sous l'ancre pour
    éviter d'inclure le titre de section lui-même. Sur la dernière page partagée
    avec la section suivante, la fenêtre se ferme au-dessus de l'ancre suivante.
    Les coordonnées sont normalisées dans [0, 1].

    Returns:
        Tuple ``(top, bottom)`` délimitant la zone de la section sur cette page.
    """
    top = 0.0
    bottom = 1.0
    if page_number == section.start_page and section.anchor_page == section.start_page and section.anchor_bbox_norm:
        top = max(top, float(section.anchor_bbox_norm[3]))
    if (
        next_section is not None
        and page_number == section.end_page == next_section.start_page
        and next_section.anchor_page == next_section.start_page
        and next_section.anchor_bbox_norm
    ):
        bottom = min(bottom, float(next_section.anchor_bbox_norm[1]))
    if bottom <= top:
        return top, 1.0
    return top, bottom


def _classify_block_type(
    block: PDFBlock,
    repeated_text_counts: dict[str, int],
    table_bboxes: list[list[float]] | None = None,
    footnote_bboxes: list[list[float]] | None = None,
) -> str:
    """Classifie un bloc PDF en l'une des catégories : narrative, table, footnote, header_footer, other.

    Priorités d'application :
    1. Type déjà assigné par Docling (table, footnote, header_footer) → conservé tel quel.
    2. Texte répété en haut/bas de page → header_footer.
    3. Bas de page avec marqueur de note → footnote.
    4. Chevauchement géométrique avec un tableau → table.
    5. Chevauchement avec zone de note + texte de légende → footnote.
    6. Heuristiques textuelles : narrative, table ou other.

    Args:
        block: Bloc PDF à classifier.
        repeated_text_counts: Nombre d'occurrences normalisées de chaque texte (toutes pages).
        table_bboxes: Bounding boxes des tableaux sur la même page.
        footnote_bboxes: Zones de notes inférées sur la même page.

    Returns:
        Chaîne parmi ``"narrative"``, ``"table"``, ``"footnote"``, ``"header_footer"``, ``"other"``.
    """
    if block.block_type in {"table", "footnote", "header_footer"}:
        return block.block_type
    norm = _normalized_block_text(block.text)
    if not norm:
        return "other"
    text = block.text.strip()
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    digits = re.findall(r"\d", text)
    numeric_tokens = re.findall(r"\b\S*\d\S*\b", text)
    rating_tokens = re.findall(r"\b(?:[A-Z]{1,4}[+-]?|[A-Z][a-z]\d|[A-Z]{1,3}\s*\(hyb\)|FPUNV)\b", text)
    short_word_ratio = sum(1 for word in words if len(word) <= 4) / max(1, len(words)) if words else 0.0
    digit_ratio = len(digits) / max(1, len(text))
    upper_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    repeated = repeated_text_counts.get(norm, 0)
    table_bboxes = table_bboxes or []
    footnote_bboxes = footnote_bboxes or []

    if repeated >= 2 and (block.y1 <= 0.12 or block.y0 >= 0.88):
        return "header_footer"
    if block.y0 >= 0.75 and re.match(r"^\s*(?:\(?\d+\)?|[*†‡]|note\b|source\b)", text, flags=re.IGNORECASE):
        return "footnote"
    if _block_overlaps_table(block, table_bboxes):
        return "table"
    if _block_overlaps_table(block, footnote_bboxes) and _looks_like_table_footnote_text(text):
        return "footnote"
    if _looks_like_footnote(text):
        return "footnote"
    if _looks_like_narrative_paragraph(text):
        return "narrative"
    if (
        _looks_like_table_or_financial_grid(text)
        or (digit_ratio >= 0.12 and len(words) <= 16)
        or (len(numeric_tokens) >= 10 and len(words) <= 20)
        or len(rating_tokens) >= 6
        or (len(numeric_tokens) >= 4 and short_word_ratio >= 0.45)
        or (short_word_ratio >= 0.62 and upper_ratio >= 0.18 and len(words) >= 18)
        or "\t" in text
        or "  " in text
        or ("|" in text)
        or (len(words) <= 8 and len(digits) >= 4)
        or (upper_ratio >= 0.7 and len(words) <= 12)
    ):
        return "table"
    if len(words) >= 8 and len(text) >= 45:
        return "narrative"
    return "other"


def _exclusion_reason_for_block(block_type: str, in_window: bool) -> str:
    """Retourne la raison d'exclusion d'un bloc non narratif.

    Un bloc hors fenêtre de section reçoit ``outside_target_section``.
    Un bloc dans la fenêtre mais non narratif reçoit une raison selon son type.
    Un bloc narratif inclus reçoit une chaîne vide (pas d'exclusion).
    """
    if not in_window:
        return "outside_target_section"
    return {
        "table": "table_like_block",
        "footnote": "footnote",
        "header_footer": "header_footer",
        "other": "non_narrative_block",
    }.get(block_type, "")


def _table_regions_for_pages(
    section_key: str,
    table_bboxes_by_page: dict[int, list[list[float]]],
    footnote_bboxes_by_page: dict[int, list[list[float]]],
    pages: list[int],
) -> list[dict[str, Any]]:
    """Construit la liste des régions de tableaux et de notes pour un ensemble de pages.

    Chaque région est un dictionnaire avec ``table_id``, ``page``, ``region_type``
    (``"table"`` ou ``"footnote"``) et ``bbox``. Stockée dans ``SectionAudit.table_regions``
    pour la traçabilité et l'audit de l'extraction.
    """
    regions: list[dict[str, Any]] = []
    for page in pages:
        for idx, bbox in enumerate(table_bboxes_by_page.get(page, []), start=1):
            regions.append(
                {
                    "table_id": f"{section_key}_p{page:03d}_tbl_{idx:02d}",
                    "page": page,
                    "region_type": "table",
                    "bbox": [round(v, 6) for v in bbox],
                }
            )
        for idx, bbox in enumerate(footnote_bboxes_by_page.get(page, []), start=1):
            regions.append(
                {
                    "table_id": f"{section_key}_p{page:03d}_ftn_{idx:02d}",
                    "page": page,
                    "region_type": "footnote",
                    "bbox": [round(v, 6) for v in bbox],
                }
            )
    return regions


def _repeated_text_counts(page_blocks: dict[int, list[PDFBlock]]) -> dict[str, int]:
    """Compte le nombre de pages distinctes où chaque texte normalisé apparaît.

    Un texte qui apparaît sur ≥ 2 pages est un candidat en-tête/pied de page.
    Utilisé par ``_classify_block_type`` pour détecter et exclure ces répétitions.
    """
    counts: dict[str, int] = {}
    for blocks in page_blocks.values():
        seen_on_page: set[str] = set()
        for block in blocks:
            norm = _normalized_block_text(block.text)
            if not norm or norm in seen_on_page:
                continue
            seen_on_page.add(norm)
            counts[norm] = counts.get(norm, 0) + 1
    return counts


def _build_section_audit(
    *,
    section: ResolvedSection,
    next_section: ResolvedSection | None,
    page_blocks: dict[int, list[PDFBlock]],
    repeated_text_counts: dict[str, int],
    table_bboxes_by_page: dict[int, list[list[float]]],
    footnote_bboxes_by_page: dict[int, list[list[float]]],
) -> SectionAudit:
    """Construit l'audit complet d'une section : blocs inclus, exclus et régions de tableaux.

    Pour chaque bloc de la section, applique la fenêtre verticale de la section,
    classifie le type de bloc et décide de son inclusion. Seuls les blocs
    ``narrative`` dans la fenêtre sont inclus ; les autres sont gardés dans
    ``excluded_blocks`` pour la traçabilité.

    Args:
        section: Section résolue avec ses pages et son ancre.
        next_section: Section suivante dans le PDF (pour délimiter la fenêtre basse).
        page_blocks: Blocs extraits par Docling, indexés par page.
        repeated_text_counts: Comptages de textes répétés pour détection en-têtes/pieds.
        table_bboxes_by_page: Bounding boxes des tableaux, par page.
        footnote_bboxes_by_page: Zones de notes inférées, par page.

    Returns:
        ``SectionAudit`` avec ``included_blocks``, ``excluded_blocks`` et ``table_regions``.
    """
    included_blocks: list[PDFBlock] = []
    excluded_blocks: list[PDFBlock] = []
    for page in section.pages:
        blocks = page_blocks.get(page, [])
        page_tables = table_bboxes_by_page.get(page, [])
        page_footnotes = footnote_bboxes_by_page.get(page, [])
        top_cutoff, bottom_cutoff = _section_window_for_page(section, page, next_section)
        for block in blocks:
            section_block = PDFBlock(
                block_id=block.block_id,
                page=block.page,
                bbox_norm=list(block.bbox_norm),
                text=block.text,
                line_number=block.line_number,
                block_type=block.block_type,
                source_label=block.source_label,
            )
            midpoint = (block.y0 + block.y1) / 2.0
            in_window = top_cutoff <= midpoint < bottom_cutoff
            block_type = _classify_block_type(section_block, repeated_text_counts, page_tables, page_footnotes)
            section_block.block_type = block_type
            section_block.included = in_window and block_type == "narrative"
            section_block.exclusion_reason = (
                "" if section_block.included else _exclusion_reason_for_block(block_type, in_window)
            )
            if section_block.included:
                included_blocks.append(section_block)
            else:
                excluded_blocks.append(section_block)
    return SectionAudit(
        section_key=section.section_key,
        section_title=section.title,
        start_page=section.start_page,
        end_page=section.end_page,
        anchor_page=section.anchor_page,
        anchor_text=section.anchor_text,
        anchor_bbox_norm=section.anchor_bbox_norm,
        included_blocks=included_blocks,
        excluded_blocks=excluded_blocks,
        table_regions=_table_regions_for_pages(
            section.section_key,
            table_bboxes_by_page,
            footnote_bboxes_by_page,
            section.pages,
        ),
    )
