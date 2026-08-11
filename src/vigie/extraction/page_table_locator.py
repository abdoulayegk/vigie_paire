"""Localisation geometrique des tableaux sur une page PDF complete.

Ce module constitue un garde-fou pour les recadrages Vision incomplets. Il
analyse une page complete uniquement pour localiser les tableaux, leurs titres
et leurs notes. Il n'extrait jamais les indicateurs ni les valeurs du tableau.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vigie.support.utils.openai_schema import build_strict_openai_response_format

from .vision_cache import cache_get, cache_put, get_vision_cache_dir

logger = logging.getLogger(__name__)

OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS = 120.0
DEFAULT_PAGE_LOCATOR_MIN_CONFIDENCE = 0.85
_PAGE_LOCATOR_CACHE_VERSION = "v2"
_MAX_TABLES_PER_PAGE = 16
_MIN_NEIGHBOR_GAP = 0.005
_GEOMETRY_RESCUE_REASONS = {
    "dominant_contamination",
    "generic_page_title",
    "generic_title_without_support",
    "low_density_vertical",
    "missing_result",
    "missing_expected_footnotes",
    "narrative_indicator_contamination",
    "no_viable_indicators",
    "qa_inspector_failed",
    "quality_critiques",
    "top_context_missing_title",
    "weak_indicator_only",
}

_PAGE_TABLE_LOCATOR_PROMPT = """
Tu es un moteur de LOCALISATION GEOMETRIQUE de tableaux financiers.

ENTREE
Une image de PAGE COMPLETE d'un rapport bancaire canadien. La page peut
contenir zero, un ou plusieurs tableaux, ainsi que du texte narratif, des
graphiques et des notes de bas de tableaux.

MISSION UNIQUE
Localiser chaque tableau visible et retourner sa geometrie :
- table_bbox : corps complet du tableau, incluant la COLONNE DE GAUCHE des
  libelles de lignes, les en-tetes et toutes les lignes;
- title_bbox : bloc du titre directement associe au tableau, ou null;
- footnotes_bbox : bloc des notes directement associees sous le tableau, ou null;
- title_text : texte du titre seulement, ou chaine vide;
- continuation : true si le tableau continue clairement une page precedente;
- confidence : confiance dans les limites et l'association des blocs.

INTERDICTIONS ABSOLUES
- N'EXTRAIS AUCUN indicateur, libelle de ligne, chiffre ou valeur de cellule.
- Ne traite pas un paragraphe, un graphique ou une liste comme un tableau.
- Ne fusionne jamais deux tableaux voisins dans le meme table_bbox.
- N'associe pas au tableau les notes ou titres d'un tableau voisin.
- Ne coupe jamais la colonne de gauche des libelles de lignes, meme si elle
  depasse a gauche de la zone grisee ou quadrillee.

COORDONNEES
Toutes les boites utilisent [x_min, y_min, x_max, y_max], normalisees entre
0.0 et 1.0, origine en haut a gauche. Respecte l'ordre visuel haut vers bas.

REGLES DE PRECISION
1. table_bbox englobe l'INTEGRALITE du tableau : ne coupe ni la premiere/derniere
   ligne, ni la premiere/derniere COLONNE. La colonne des libelles (a gauche)
   est TOUJOURS incluse.
2. Le bord gauche s'aligne sur le debut des LIBELLES, pas sur la grille des
   chiffres. Les libelles sont souvent a gauche de la zone grisee/coloree ou
   dans la marge blanche : etends la boite jusqu'a les inclure. Ne "serre"
   jamais la boite sur la seule grille des donnees.
3. title_bbox ne couvre que le titre situe immediatement au-dessus s'il exist sinon utilise celui qui existe en haut avant le text narratif.
4. footnotes_bbox ne couvre que les notes situees immediatement au-dessous.
5. Sur une page multi-tableaux, garde des boites separees et attribue chaque
   titre/note au tableau le plus proche.
6. En cas d'ambiguite, diminue confidence. N'invente aucune boite.

Retourne uniquement le JSON conforme au schema demande.
"""


def _coerce_bbox(value: Any, *, allow_none: bool) -> list[float] | None:
    """Normaliser une bbox fournie par le modele."""
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox coordinates must be numeric") from exc
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError("bbox coordinates must be finite")
    return bbox


class PageTableRegionSchema(BaseModel):
    """Region geometrique retournee pour un tableau de la page."""

    model_config = ConfigDict(extra="forbid")

    table_bbox: list[float] = Field(description="Corps complet du tableau.")
    title_bbox: list[float] | None = Field(description="Bloc titre associe, ou null.")
    footnotes_bbox: list[float] | None = Field(description="Bloc des notes associees, ou null.")
    title_text: str = Field(description="Titre visible, sans contenu du tableau.")
    continuation: bool = Field(description="Tableau continue depuis la page precedente.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiance geometrique.")

    @field_validator("table_bbox", mode="before")
    @classmethod
    def _validate_table_bbox(cls, value: Any) -> list[float]:
        bbox = _coerce_bbox(value, allow_none=False)
        assert bbox is not None
        return bbox

    @field_validator("title_bbox", "footnotes_bbox", mode="before")
    @classmethod
    def _validate_optional_bbox(cls, value: Any) -> list[float] | None:
        return _coerce_bbox(value, allow_none=True)

    @field_validator("title_text", mode="before")
    @classmethod
    def _coerce_title(cls, value: Any) -> str:
        return str(value or "").strip()


class PageTableLocatorResponseSchema(BaseModel):
    """Reponse stricte de localisation de tous les tableaux d'une page."""

    model_config = ConfigDict(extra="forbid")

    tables: list[PageTableRegionSchema] = Field(description="Tableaux visibles, ordonnes du haut vers le bas.")
    table_count: int = Field(ge=0, le=_MAX_TABLES_PER_PAGE)


@dataclass(frozen=True)
class PageTableRegion:
    """Region validee et utilisable par le pipeline de recadrage."""

    table_bbox: tuple[float, float, float, float]
    title_bbox: tuple[float, float, float, float] | None
    footnotes_bbox: tuple[float, float, float, float] | None
    title_text: str
    continuation: bool
    confidence: float


@dataclass(frozen=True)
class PageTableLayout:
    """Localisation partagee par tous les tableaux d'une meme page."""

    page_number: int
    tables: tuple[PageTableRegion, ...]


@dataclass(frozen=True)
class PageTableCropPlan:
    """Recadrage corrige, pret a etre rendu en haute resolution."""

    bbox_norm: tuple[float, float, float, float]
    top_extension: float
    bottom_extension: float
    confidence: float
    title_text: str
    continuation: bool
    table_count: int


def _is_bbox_sane(
    bbox: tuple[float, float, float, float] | list[float] | None,
    *,
    min_width: float,
    min_height: float,
) -> bool:
    """Verifier les bornes et une taille minimale sans dependance au profil banque."""
    if bbox is None or len(bbox) != 4:
        return False
    left, top, right, bottom = [float(value) for value in bbox]
    return bool(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in bbox)
        and right - left >= min_width
        and bottom - top >= min_height
    )


def _area(bbox: tuple[float, float, float, float] | list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _intersection_area(
    first: tuple[float, float, float, float] | list[float],
    second: tuple[float, float, float, float] | list[float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _horizontal_overlap_ratio(
    first: tuple[float, float, float, float] | list[float],
    second: tuple[float, float, float, float] | list[float],
) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    denominator = min(first[2] - first[0], second[2] - second[0])
    return overlap / denominator if denominator > 0 else 0.0


def _is_associated_title(
    title_bbox: tuple[float, float, float, float] | None,
    table_bbox: tuple[float, float, float, float],
) -> bool:
    if not _is_bbox_sane(title_bbox, min_width=0.02, min_height=0.002):
        return False
    assert title_bbox is not None
    gap = table_bbox[1] - title_bbox[3]
    return bool(-0.025 <= gap <= 0.15 and _horizontal_overlap_ratio(title_bbox, table_bbox) >= 0.15)


def _is_associated_footnotes(
    footnotes_bbox: tuple[float, float, float, float] | None,
    table_bbox: tuple[float, float, float, float],
) -> bool:
    if not _is_bbox_sane(footnotes_bbox, min_width=0.02, min_height=0.002):
        return False
    assert footnotes_bbox is not None
    gap = footnotes_bbox[1] - table_bbox[3]
    return bool(-0.035 <= gap <= 0.15 and _horizontal_overlap_ratio(footnotes_bbox, table_bbox) >= 0.15)


def _parse_page_layout(raw: str | dict[str, Any], page_number: int) -> PageTableLayout | None:
    """Valider le JSON du modele et eliminer les geometries dangereuses."""
    try:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("```"):
                first_newline = text.find("\n")
                if first_newline >= 0:
                    text = text[first_newline + 1 :]
                if text.endswith("```"):
                    text = text[:-3].rstrip()
            data = json.loads(text)
        else:
            data = raw
        response = PageTableLocatorResponseSchema.model_validate(data)
    except Exception as exc:
        logger.warning(
            "Page table locator: invalid response on page %s: %s",
            page_number,
            exc,
        )
        return None

    regions: list[PageTableRegion] = []
    for item in response.tables[:_MAX_TABLES_PER_PAGE]:
        table_bbox = tuple(item.table_bbox)
        if not _is_bbox_sane(table_bbox, min_width=0.04, min_height=0.025):
            logger.debug("Page table locator: rejected unsafe table bbox %s", table_bbox)
            continue
        title_bbox = tuple(item.title_bbox) if item.title_bbox is not None else None
        footnotes_bbox = tuple(item.footnotes_bbox) if item.footnotes_bbox is not None else None
        if not _is_associated_title(title_bbox, table_bbox):
            title_bbox = None
        if not _is_associated_footnotes(footnotes_bbox, table_bbox):
            footnotes_bbox = None
        regions.append(
            PageTableRegion(
                table_bbox=table_bbox,
                title_bbox=title_bbox,
                footnotes_bbox=footnotes_bbox,
                title_text=item.title_text,
                continuation=item.continuation,
                confidence=item.confidence,
            )
        )

    regions.sort(key=lambda region: (region.table_bbox[1], region.table_bbox[0]))
    if response.table_count != len(response.tables):
        logger.debug(
            "Page table locator: declared count=%s differs from returned count=%s on page %s",
            response.table_count,
            len(response.tables),
            page_number,
        )
    return PageTableLayout(page_number=page_number, tables=tuple(regions))


def _page_layout_cache_payload(layout: PageTableLayout) -> dict[str, Any]:
    """Serialiser un inventaire valide dans le format du schema du locator."""
    tables = [
        {
            "table_bbox": list(region.table_bbox),
            "title_bbox": (list(region.title_bbox) if region.title_bbox is not None else None),
            "footnotes_bbox": (list(region.footnotes_bbox) if region.footnotes_bbox is not None else None),
            "title_text": region.title_text,
            "continuation": region.continuation,
            "confidence": region.confidence,
        }
        for region in layout.tables
    ]
    return {"tables": tables, "table_count": len(tables)}


def build_page_table_crop_plan(
    layout: PageTableLayout,
    target_bbox: list[float],
    *,
    min_confidence: float = DEFAULT_PAGE_LOCATOR_MIN_CONFIDENCE,
) -> PageTableCropPlan | None:
    """Associer une bbox Docling a une region Vision et borner son contexte.

    Le plan est refuse si la region est peu fiable, trop eloignee ou ambigue.
    Les extensions de titre et de notes sont bornees par les tableaux voisins.
    """
    if not _is_bbox_sane(target_bbox, min_width=0.02, min_height=0.015):
        return None

    target_area = _area(target_bbox)
    candidates: list[tuple[float, PageTableRegion]] = []
    for region in layout.tables:
        if region.confidence < min_confidence:
            continue
        intersection = _intersection_area(target_bbox, region.table_bbox)
        region_area = _area(region.table_bbox)
        union = target_area + region_area - intersection
        target_overlap = intersection / target_area if target_area > 0 else 0.0
        iou = intersection / union if union > 0 else 0.0
        horizontal_overlap = _horizontal_overlap_ratio(target_bbox, region.table_bbox)
        target_center = (
            (target_bbox[0] + target_bbox[2]) / 2,
            (target_bbox[1] + target_bbox[3]) / 2,
        )
        region_center = (
            (region.table_bbox[0] + region.table_bbox[2]) / 2,
            (region.table_bbox[1] + region.table_bbox[3]) / 2,
        )
        center_distance = math.dist(target_center, region_center)
        if target_overlap < 0.20 and not (horizontal_overlap >= 0.35 and center_distance <= 0.12):
            continue
        score = (
            0.50 * target_overlap
            + 0.25 * iou
            + 0.15 * horizontal_overlap
            + 0.10 * region.confidence
            - 0.20 * center_distance
        )
        candidates.append((score, region))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    if len(candidates) > 1 and best_score - candidates[1][0] < 0.08:
        logger.info(
            "Page table locator: ambiguous target association on page %s (%.3f vs %.3f)",
            layout.page_number,
            best_score,
            candidates[1][0],
        )
        return None

    table_bbox = best.table_bbox
    desired_top = table_bbox[1]
    desired_bottom = table_bbox[3]
    if best.title_bbox is not None:
        desired_top = min(desired_top, best.title_bbox[1] - 0.005)
    if best.footnotes_bbox is not None:
        desired_bottom = max(desired_bottom, best.footnotes_bbox[3] + 0.005)

    previous_bottom: float | None = None
    next_top: float | None = None
    for region in layout.tables:
        if region is best or _horizontal_overlap_ratio(region.table_bbox, table_bbox) < 0.20:
            continue
        if region.table_bbox[3] <= table_bbox[1]:
            previous_bottom = max(
                previous_bottom if previous_bottom is not None else 0.0,
                region.table_bbox[3],
            )
        elif region.table_bbox[1] >= table_bbox[3]:
            next_top = min(
                next_top if next_top is not None else 1.0,
                region.table_bbox[1],
            )

    safe_top = max(0.0, desired_top)
    safe_bottom = min(1.0, desired_bottom)
    if previous_bottom is not None:
        safe_top = max(safe_top, previous_bottom + _MIN_NEIGHBOR_GAP)
    if next_top is not None:
        safe_bottom = min(safe_bottom, next_top - _MIN_NEIGHBOR_GAP)
    if safe_top > table_bbox[1] or safe_bottom < table_bbox[3]:
        logger.info(
            "Page table locator: context rejected because it crosses a neighbor on page %s",
            layout.page_number,
        )
        return None

    # Garde-fou horizontal : le bord gauche/droit ne peut jamais reculer
    # au-dela de l'ancre Docling (target_bbox). Sur les tableaux-matrices
    # larges, le localisateur Vision serre parfois la grille des chiffres et
    # coupe la colonne des libelles ; on restaure alors la largeur Docling.
    union_bbox = (
        min(table_bbox[0], float(target_bbox[0])),
        table_bbox[1],
        max(table_bbox[2], float(target_bbox[2])),
        table_bbox[3],
    )
    return PageTableCropPlan(
        bbox_norm=union_bbox,
        top_extension=max(0.0, table_bbox[1] - safe_top),
        bottom_extension=max(0.0, safe_bottom - table_bbox[3]),
        confidence=best.confidence,
        title_text=best.title_text,
        continuation=best.continuation,
        table_count=len(layout.tables),
    )


def build_near_full_page_crop_plan(
    layout: PageTableLayout,
    *,
    min_confidence: float = DEFAULT_PAGE_LOCATOR_MIN_CONFIDENCE,
) -> PageTableCropPlan | None:
    """Verifier une bbox presque pleine page sans choisir une region arbitraire.

    La bbox Docling initiale recouvre trop de contenu pour servir d'ancre.
    Le locator doit donc trouver exactement une region de tableau fiable.
    """
    reliable_regions = [region for region in layout.tables if region.confidence >= min_confidence]
    if len(reliable_regions) != 1:
        return None
    return build_page_table_crop_plan(
        layout,
        list(reliable_regions[0].table_bbox),
        min_confidence=min_confidence,
    )


def should_use_page_context_rescue(
    rescue_result_available: bool,
    rejection_reasons: list[str],
) -> bool:
    """Declencher la page complete lorsqu'un recadrage peut expliquer l'echec.

    L'absence totale de resultat est precisement un cas ou la page complete
    doit pouvoir verifier la geometrie. Le parametre ``rescue_result_available``
    est conserve pour compatibilite avec les appelants existants.
    """
    _ = rescue_result_available
    reasons = {str(reason or "").strip() for reason in rejection_reasons}
    return bool(reasons & _GEOMETRY_RESCUE_REASONS)


class PageTableLocator:
    """Localiser une page une seule fois et partager le resultat entre ses tableaux."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        min_confidence: float = DEFAULT_PAGE_LOCATOR_MIN_CONFIDENCE,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """Configure le modèle, le seuil et les caches partagés par page."""
        self._api_key = api_key
        self._model = model
        self._min_confidence = min_confidence
        self._use_cache = use_cache
        self._cache_dir = cache_dir or get_vision_cache_dir()
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._page_cache: dict[str, PageTableLayout | None] = {}
        self._inflight: dict[str, threading.Event] = {}

    @property
    def min_confidence(self) -> float:
        """Retourner le seuil utilise pour accepter une region."""
        return self._min_confidence

    def _ensure_client(self) -> Any:
        """Crée paresseusement un client OpenAI unique de manière thread-safe."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    timeout=OPENAI_PAGE_LOCATOR_TIMEOUT_SECONDS,
                )
        return self._client

    def _locate_uncached(
        self,
        page_image_bytes: bytes,
        page_number: int,
    ) -> PageTableLayout | None:
        """Localise les régions d'une page sans consulter les caches du service.

        L'appel Vision est non bloquant pour le pipeline au sens métier : toute
        erreur est journalisée et convertie en absence de résultat, afin que les
        stratégies d'extraction de repli puissent prendre le relais.
        """
        if not page_image_bytes:
            return None
        try:
            client = self._ensure_client()
            response_format = build_strict_openai_response_format(
                PageTableLocatorResponseSchema,
                name="page_table_locator",
            )
            image_b64 = base64.standard_b64encode(page_image_bytes).decode("ascii")
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PAGE_TABLE_LOCATOR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                response_format=response_format,
                temperature=0,
                max_completion_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            return _parse_page_layout(raw, page_number)
        except Exception as exc:
            logger.warning(
                "Page table locator failed on page %s (non-fatal): %s",
                page_number,
                exc,
            )
            return None

    def locate_page(
        self,
        page_image_bytes: bytes,
        page_number: int,
        *,
        pdf_sha: str = "",
    ) -> PageTableLayout | None:
        """Localiser la page avec caches memoire/persistant et appel unique."""
        cache_key = f"{pdf_sha or 'document'}:{page_number}"
        with self._cache_lock:
            if cache_key in self._page_cache:
                return self._page_cache[cache_key]
            event = self._inflight.get(cache_key)
            owner = event is None
            if owner:
                event = threading.Event()
                self._inflight[cache_key] = event

        assert event is not None
        if not owner:
            event.wait()
            with self._cache_lock:
                return self._page_cache.get(cache_key)

        result: PageTableLayout | None = None
        try:
            persistent_key = ""
            if self._use_cache and pdf_sha:
                raw_key = f"{_PAGE_LOCATOR_CACHE_VERSION}|{self._model}|{pdf_sha}|{page_number}"
                persistent_key = "page_locator_" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
                cached = cache_get(self._cache_dir, persistent_key)
                if cached is not None:
                    result = _parse_page_layout(cached, page_number)

            if result is None:
                result = self._locate_uncached(page_image_bytes, page_number)
                if result is not None and persistent_key:
                    cache_put(
                        self._cache_dir,
                        persistent_key,
                        _page_layout_cache_payload(result),
                    )
            return result
        finally:
            with self._cache_lock:
                self._page_cache[cache_key] = result
                self._inflight.pop(cache_key, None)
                event.set()
