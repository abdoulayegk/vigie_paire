"""Extraction de blocs de texte par section depuis un PDF via Docling.

Ce module extrait les paragraphes de texte brut des sections ciblées
(gestion_capital, gestion_risques, gestion_reglementation) en excluant
tous les tableaux et leurs zones de proximité.

Il réutilise le pattern d'initialisation Docling de DoclingProcessor
mais n'utilise PAS Vision GPT-4o — uniquement l'extraction structurelle
Docling pour localiser les blocs de texte.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECTION_SHORT: dict[str, str] = {
    "gestion_capital": "gc",
    "gestion_risques": "gr",
    "gestion_reglementation": "reg",
}

# Labels Docling à inclure (texte de corps + headings pour conserver le titre d'ancre)
_INCLUDE_LABELS: frozenset[str] = frozenset(
    {
        "text",
        "paragraph",
        "list_item",
        "body_text",
        "heading",
        "title",
        "header",
        "section_header",
    }
)

# Longueur minimale d'un bloc texte (caractères) — ignore les fragments courts
_MIN_TEXT_LENGTH: int = 40

# Marge de proximité autour des tableaux (coordonnées normalisées 0-1)
_TABLE_PROXIMITY_MARGIN: float = 0.02

# Tolérance pour considérer qu'un bloc est après le bas du titre de section
_ANCHOR_VERTICAL_EPSILON: float = 0.002

# Regex pour détecter les notes de bas de tableau : (1) ... ou 1) ...
_FOOTNOTE_RE = re.compile(r"^\s*\(?\d+\)[\s.]")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """Un bloc de texte extrait d'une section d'un rapport bancaire."""

    block_id: str
    section: str
    page: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Sérialise le bloc en dict pour JSON."""
        return {
            "block_id": self.block_id,
            "section": self.section,
            "page": self.page,
            "text": self.text,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_bbox_topleft(
    raw_bbox: Any,
    page_obj: Any,
) -> list[float] | None:
    """Convertit un BoundingBox Docling en [l, t, r, b] normalisé (top-left origin).

    Identique au pattern utilisé dans docling_processor.py:1641.
    Retourne None si la conversion échoue.
    """
    try:
        norm = raw_bbox.to_top_left_origin(page_height=page_obj.size.height)
        norm = norm.normalized(page_obj.size)
        return [norm.l, norm.t, norm.r, norm.b]
    except Exception:
        return None


def _normalize_text_for_match(value: Any) -> str:
    """Normaliser un texte pour les comparaisons d'ancre."""
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")
    return " ".join(ascii_text.lower().split())


def _collect_table_zones(
    doc: Any,
    page_objects: dict[int, Any],
) -> dict[int, list[list[float]]]:
    """Retourne {page_no: [[l, t, r, b], ...]} pour tous les TableItem du document.

    Utilisé pour filtrer les blocs de texte trop proches d'un tableau.
    """
    zones: dict[int, list[list[float]]] = {}
    try:
        from docling_core.types.doc.document import TableItem

        for item, _level in doc.iterate_items():
            if not isinstance(item, TableItem):
                continue
            if not item.prov:
                continue
            prov = item.prov[0]
            page_no = prov.page_no
            page_obj = page_objects.get(page_no)
            if page_obj is None:
                continue
            bbox = _normalize_bbox_topleft(prov.bbox, page_obj)
            if bbox:
                zones.setdefault(page_no, []).append(bbox)
    except Exception as exc:
        logger.debug("_collect_table_zones: %s", exc)
    return zones


def _is_near_table(
    block_bbox: list[float],
    table_zones: list[list[float]],
    margin: float = _TABLE_PROXIMITY_MARGIN,
) -> bool:
    """Retourne True si block_bbox chevauche verticalement une zone de tableau.

    Utilise les coordonnées normalisées top-left [l, t, r, b].
    Chevauchement : block_t < zone_b + margin ET block_b > zone_t - margin.
    """
    block_t = block_bbox[1]
    block_b = block_bbox[3]
    for zone in table_zones:
        zone_t = zone[1]
        zone_b = zone[3]
        if block_t < zone_b + margin and block_b > zone_t - margin:
            return True
    return False


def _page_in_ranges(page_no: int, section_ranges: list[dict[str, Any]]) -> str | None:
    """Retourne la section_key si la page appartient à une plage, sinon None.

    En cas de chevauchement (ex: gestion_reglementation inclus dans gestion_capital),
    retourne la section avec la plage la plus étroite (la plus spécifique).
    """
    best_section: str | None = None
    best_size: int = 10**9

    for rng in section_ranges:
        start = rng.get("start", 0)
        end = rng.get("end", start)
        if start <= page_no <= end:
            size = end - start
            if size < best_size:
                best_size = size
                best_section = str(rng.get("section", ""))

    return best_section


def _range_for_page(
    page_no: int,
    section_ranges: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Retourner la metadata de section la plus specifique pour une page."""
    best_range: dict[str, Any] | None = None
    best_size: int = 10**9

    for rng in section_ranges:
        start = int(rng.get("start", 0) or 0)
        end = int(rng.get("end", start) or start)
        if start <= page_no <= end:
            size = end - start
            if size < best_size:
                best_size = size
                best_range = rng

    return best_range


def _ranges_for_page(
    page_no: int,
    section_ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retourner toutes les sections couvrant une page."""
    matches: list[dict[str, Any]] = []
    for rng in section_ranges:
        start = int(rng.get("start", 0) or 0)
        end = int(rng.get("end", start) or start)
        if start <= page_no <= end:
            matches.append(rng)
    return matches


def _filter_valid_section_ranges(
    section_ranges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filtrer les sections invalides faute d'ancre explicite.

    Compatibilite legacy: si ``anchor_found`` n'est pas fourni, la section reste valide.
    """
    valid: list[dict[str, Any]] = []
    invalid_sections: list[str] = []
    for rng in section_ranges:
        if "anchor_found" in rng and not bool(rng.get("anchor_found")):
            invalid_sections.append(str(rng.get("section", "")))
            continue
        valid.append(rng)
    return valid, invalid_sections


def _is_anchor_text(text: str, section_range: dict[str, Any]) -> bool:
    """Verifier si un bloc texte correspond exactement au titre d'ancre."""
    anchor_text = str(section_range.get("anchor_text", "") or "").strip()
    if not anchor_text:
        return False
    return _normalize_text_for_match(text) == _normalize_text_for_match(anchor_text)


def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    """Calculer un ratio d'intersection sur la plus petite des deux bboxes."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    return inter / min_area


def _is_anchor_item(
    *,
    page_no: int,
    text: str,
    bbox: list[float] | None,
    section_range: dict[str, Any],
) -> bool:
    """Determiner si un bloc Docling est le bloc titre de section."""
    anchor_page = int(section_range.get("anchor_page", 0) or 0)
    if anchor_page and page_no != anchor_page:
        return False
    if _is_anchor_text(text, section_range):
        return True

    anchor_bbox = section_range.get("anchor_bbox_norm")
    if bbox and isinstance(anchor_bbox, list) and len(anchor_bbox) == 4:
        return _bbox_overlap_ratio(bbox, anchor_bbox) >= 0.7
    return False


def _anchor_bottom(section_range: dict[str, Any]) -> float:
    """Retourner le bord bas de l'ancre, ou -1 si absent."""
    bbox = section_range.get("anchor_bbox_norm")
    if isinstance(bbox, list) and len(bbox) == 4:
        return float(bbox[3])
    return -1.0


def _resolve_section_for_item(
    *,
    page_no: int,
    text: str,
    bbox: list[float] | None,
    section_ranges: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choisir la section d'un bloc texte, y compris en cas de chevauchement."""
    candidates = _ranges_for_page(page_no, section_ranges)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for rng in candidates:
        if _is_anchor_item(page_no=page_no, text=text, bbox=bbox, section_range=rng):
            return rng

    anchors_on_page = [
        rng for rng in candidates if bool(rng.get("anchor_found")) and int(rng.get("anchor_page", 0) or 0) == page_no
    ]
    anchors_on_page.sort(key=_anchor_bottom)

    if bbox and anchors_on_page:
        eligible = [rng for rng in anchors_on_page if _anchor_bottom(rng) <= float(bbox[1]) + _ANCHOR_VERTICAL_EPSILON]
        if eligible:
            return eligible[-1]

        continuation = [rng for rng in candidates if int(rng.get("start", 0) or 0) < page_no]
        if continuation:
            continuation.sort(
                key=lambda rng: (
                    int(rng.get("start", 0) or 0),
                    -(int(rng.get("end", 0) or 0) - int(rng.get("start", 0) or 0)),
                ),
                reverse=True,
            )
            return continuation[0]
        return None

    return _range_for_page(page_no, candidates)


def _extract_text_items(
    doc: Any,
    page_objects: dict[int, Any],
    table_zones: dict[int, list[list[float]]],
    section_ranges: list[dict[str, Any]],
) -> list[tuple[str, int, str]]:
    """Itère doc.iterate_items() et retourne les blocs texte filtrés.

    Retourne liste de (section_key, page_no, text).
    Applique :
    1. Filtre de label : label.value in _INCLUDE_LABELS
    2. Filtre de longueur minimale
    3. Filtre de page : dans une section cible
    4. Filtre de proximité tableau : pas dans zone de tableau ±margin
    """
    results: list[tuple[str, int, str]] = []
    dropped_before_anchor: dict[str, int] = {}

    try:
        from docling_core.types.doc.document import TextItem
    except ImportError:
        # Fallback : utiliser duck-typing sur le label
        TextItem = None  # type: ignore[assignment,misc]

    for item, _level in doc.iterate_items():
        label_val = str(getattr(getattr(item, "label", None), "value", "")).lower()
        if TextItem is not None and isinstance(item, TextItem):
            if label_val not in _INCLUDE_LABELS:
                continue
        elif label_val not in _INCLUDE_LABELS:
            continue

        text = str(getattr(item, "text", "") or "").strip()
        if not item.prov:
            continue

        prov = item.prov[0]
        page_no = prov.page_no

        # Filtre section
        page_obj = page_objects.get(page_no)
        bbox = _normalize_bbox_topleft(prov.bbox, page_obj) if page_obj is not None else None

        section_range = _resolve_section_for_item(
            page_no=page_no,
            text=text,
            bbox=bbox,
            section_ranges=section_ranges,
        )
        if not section_range:
            continue
        section_key = str(section_range.get("section", "") or "")
        if not section_key:
            continue
        is_anchor_item = _is_anchor_item(
            page_no=page_no,
            text=text,
            bbox=bbox,
            section_range=section_range,
        )

        if len(text) < _MIN_TEXT_LENGTH and not is_anchor_item:
            continue

        # Filtre notes de bas de tableau — ex: "(1) Le BSIF a prescrit..."
        if _FOOTNOTE_RE.match(text) and not is_anchor_item:
            continue

        anchor_page = int(section_range.get("anchor_page", 0) or 0)
        anchor_bbox = section_range.get("anchor_bbox_norm")
        if anchor_page and page_no == anchor_page and bool(section_range.get("anchor_found")) and not is_anchor_item:
            if not bbox or not isinstance(anchor_bbox, list) or len(anchor_bbox) != 4:
                dropped_before_anchor[section_key] = dropped_before_anchor.get(section_key, 0) + 1
                continue
            if float(bbox[1]) < float(anchor_bbox[3]) - _ANCHOR_VERTICAL_EPSILON:
                dropped_before_anchor[section_key] = dropped_before_anchor.get(section_key, 0) + 1
                continue

        # Filtre proximité tableau
        zones_on_page = table_zones.get(page_no, [])
        if zones_on_page and bbox and not is_anchor_item:
            if _is_near_table(bbox, zones_on_page):
                continue

        results.append((section_key, page_no, text))

    for section_key, count in sorted(dropped_before_anchor.items()):
        logger.info(
            "TextExtractor: section=%s dropped_blocks_before_anchor=%d",
            section_key,
            count,
        )

    return results


def _make_block_id(
    bank: str,
    year: int,
    quarter: str,
    section_key: str,
    seq: int,
) -> str:
    """Génère un block_id canonique.

    Format : {bank}_{year}_{quarter}_{section_short}_{seq:03d}
    Exemple : bns_2025_t2_gc_001
    """
    short = _SECTION_SHORT.get(section_key, section_key[:4])
    return f"{bank.lower()}_{year}_{quarter.lower()}_{short}_{seq:03d}"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class TextExtractor:
    """Extrait les blocs de texte d'un PDF via Docling.

    Réutilise le pattern d'initialisation de DoclingProcessor._initialize_docling().
    N'utilise PAS Vision GPT-4o.
    """

    def __init__(self) -> None:
        """Initialise l'extracteur avec un convertisseur Docling lazy."""
        self._converter: Any = None
        self._initialized: bool = False

    def _initialize_docling(self) -> None:
        """Initialisation lazy du convertisseur Docling.

        Pattern identique à DoclingProcessor._initialize_docling() (ligne 627).
        do_table_structure=True pour que les TableItem soient disponibles
        lors du filtrage de proximité.
        """
        if self._initialized:
            return
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.do_picture_description = False

            _raw_threads = os.environ.get("DOCLING_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
            num_threads = int(_raw_threads) if _raw_threads else (os.cpu_count() or 8)
            default_device = "mps" if sys.platform == "darwin" else "auto"
            device_str = os.environ.get("DOCLING_DEVICE", default_device)

            try:
                from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

                device_map = {
                    "auto": AcceleratorDevice.AUTO,
                    "cpu": AcceleratorDevice.CPU,
                    "cuda": AcceleratorDevice.CUDA,
                    "mps": AcceleratorDevice.MPS,
                }
                device = device_map.get(device_str.lower(), AcceleratorDevice.AUTO)
                pipeline_options.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
            except ImportError:
                pass

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            self._initialized = True
            logger.info("TextExtractor: Docling initialisé (threads=%s, device=%s)", num_threads, device_str)

        except ImportError as exc:
            logger.error("Docling non disponible pour TextExtractor: %s", exc)
            self._initialized = True
        except Exception as exc:
            logger.error("Erreur init Docling dans TextExtractor: %s", exc)
            self._initialized = True

    def extract_text_blocks(
        self,
        pdf_path: str | Path,
        bank_code: str,
        quarter: str,
        year: int,
        section_ranges: list[dict[str, Any]],
    ) -> list[TextBlock]:
        """Extrait les blocs de texte pour les sections cibles d'un PDF.

        Args:
            pdf_path: Chemin vers le PDF.
            bank_code: Code banque (ex. "bns").
            quarter: Trimestre normalisé (ex. "t2").
            year: Année du rapport.
            section_ranges: Liste de {section: str, start: int, end: int}
                            issues de section_locator. Vide → retourne [].

        Returns:
            Liste de TextBlock triée par (section, page).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        if not section_ranges:
            logger.info("TextExtractor: aucune section_range fournie pour %s, retour vide.", pdf_path.name)
            return []

        valid_section_ranges, invalid_sections = _filter_valid_section_ranges(section_ranges)
        for section_name in invalid_sections:
            logger.warning(
                "TextExtractor: section ignoree faute d'ancre resolue: %s",
                section_name,
            )
        if not valid_section_ranges:
            logger.warning(
                "TextExtractor: aucune section avec ancre valide pour %s.",
                pdf_path.name,
            )
            return []

        self._initialize_docling()

        if self._converter is None:
            logger.error("TextExtractor: convertisseur Docling indisponible.")
            return []

        try:
            # Déterminer la plage de pages à convertir
            all_pages = [p for rng in valid_section_ranges for p in range(rng.get("start", 1), rng.get("end", 1) + 1)]
            min_page = min(all_pages)
            max_page = max(all_pages)

            convert_kwargs: dict[str, Any] = {
                "page_range": (min_page, max_page),
                "raises_on_error": False,
            }

            result = self._converter.convert(str(pdf_path), **convert_kwargs)
            doc = result.document

            # Construire le dict page_no → page_object
            page_objects: dict[int, Any] = {}
            try:
                for page_no, page_obj in doc.pages.items():
                    page_objects[int(page_no)] = page_obj
            except Exception:
                pass

            # Collecter les zones de tableaux pour le filtrage
            table_zones = _collect_table_zones(doc, page_objects)

            # Extraire les blocs texte filtrés
            for rng in valid_section_ranges:
                logger.info(
                    "TextExtractor: section=%s anchor_found=%s anchor_text=%r anchor_bbox_norm=%s",
                    rng.get("section"),
                    rng.get("anchor_found", "legacy"),
                    rng.get("anchor_text"),
                    rng.get("anchor_bbox_norm"),
                )
            raw_items = _extract_text_items(
                doc,
                page_objects,
                table_zones,
                valid_section_ranges,
            )

            # Construire les TextBlock avec IDs canoniques
            # Compter par section pour les IDs séquentiels
            section_counters: dict[str, int] = {}
            blocks: list[TextBlock] = []

            for section_key, page_no, text in raw_items:
                section_counters[section_key] = section_counters.get(section_key, 0) + 1
                seq = section_counters[section_key]
                block_id = _make_block_id(bank_code, year, quarter, section_key, seq)
                blocks.append(
                    TextBlock(
                        block_id=block_id,
                        section=section_key,
                        page=page_no,
                        text=text,
                    )
                )

            logger.info(
                "TextExtractor: %d blocs extraits depuis %s (sections: %s)",
                len(blocks),
                pdf_path.name,
                list(section_counters.keys()),
            )
            return blocks

        except Exception as exc:
            logger.error("TextExtractor: échec extraction %s — %s", pdf_path.name, exc, exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------


def extract_text_blocks_by_sections(
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    section_ranges: list[dict[str, Any]] | None = None,
) -> list[TextBlock]:
    """Point d'entrée module-level pour l'extraction de blocs texte.

    Instancie TextExtractor et appelle extract_text_blocks().
    Miroir de extract_tables_docling_by_sections() dans docling_processor.py.

    Args:
        pdf_path: Chemin vers le PDF.
        bank_code: Code banque (ex. "bns").
        quarter: Trimestre normalisé (ex. "t2").
        year: Année du rapport.
        section_ranges: Liste de {section, start, end}. None → retourne [].

    Returns:
        Liste de TextBlock.
    """
    extractor = TextExtractor()
    return extractor.extract_text_blocks(
        pdf_path=pdf_path,
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        section_ranges=section_ranges or [],
    )
