"""
Moteur officiel de pairing de tableaux pour le pipeline de comparaison actif.

Ce module est le moteur de production utilisé par ``comparison_runner.py`` via
``run_strict_intra_section_compare``. Il remplace le moteur legacy de
``indicator_comparator.py`` pour la phase de pairing (association T1/T2).

Deux moteurs sont disponibles, sélectionnables via la configuration
``recall_first_engine_enabled`` :

Moteur conservateur (legacy)
-----------------------------
1. Construction des vues canoniques depuis les ``TableArtifact`` certifiés.
2. Génération d'une shortlist bornée (max 5 candidats) par tableau T2.
3. Routing de la shortlist via un routeur déterministe conservateur.
4. Émission des états ``matched`` / ``ambiguous`` / ``unmatched``.

Moteur recall-first (défaut si activé)
---------------------------------------
1. Construction des vues pour tous les tableaux éligibles (certifiés +
   nécessitant une revue).
2. Calcul de la matrice de scores complète T1 × T2 (sans troncature).
3. Assignation optimale 1:1 globale via l'algorithme hongrois.
4. Élimination progressive : les paires fortes sont figées, le reste est
   re-traité sur les tableaux résiduels.
5. Émission des états ``matched`` / ``review_candidate`` / ``added`` /
   ``deleted``.

Philosophie de conception
--------------------------
Le moteur est **conservateur et déterministe** : il préfère déclarer une paire
``ambiguous`` (nécessitant une revue humaine) plutôt que de risquer un faux
appariement. Cette approche est justifiée par le contexte métier : un faux
appariement peut masquer un vrai changement réglementaire important.

Formule de score (``_candidate_score``)
----------------------------------------
Le score total d'une paire T1/T2 est calculé comme suit :

.. code-block:: text

    score = 0.36 * distinctive_overlap      (indicateurs rares en commun)
          + 0.23 * indicator_containment    (le plus petit contenu dans le plus grand)
          + 0.12 * header_compatibility     (schéma de colonnes compatible)
          + 0.11 * title_similarity         (titres similaires)
          + 0.08 * section_compatibility    (même section réglementaire)
          + 0.06 * size_compatibility       (tailles proches)
          + 0.04 * page_proximity           (pages proches)
          + table_number_bonus              (+0.15 si même numéro, -0.05 si conflit)
          + page_local_order_bonus          (même position sur la page)
          + page_local_role_bonus           (même rôle sur la page)
          + bbox_y_similarity_bonus         (même position verticale)
          - instability_penalty             (instabilité brut/normalisé)
          - quality_penalty                 (extraction de mauvaise qualité)

Classes principales
-------------------
- ``TableView`` : vue pré-calculée d'un ``TableArtifact`` pour le pairing.
- ``CandidateScore`` : score multi-signaux détaillé d'une paire T1/T2 candidate.
- ``PairingDecision`` : décision finale du router (match/ambiguous/no_match).
- ``PairingRouter`` : protocole (interface) pour les routeurs de pairing.
- ``ConservativePairingRouter`` : routeur déterministe conservateur (défaut).
- ``BatchLLMPairingRouter`` : routeur optionnel basé sur GPT-4o.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:
    _scipy_linear_sum_assignment = None  # type: ignore[assignment]

from vigilance.config import get_matching_thresholds
from vigilance.models.table_models import (
    EXTRACTION_STATUS_BLOCKED,
    EXTRACTION_STATUS_CERTIFIED,
    EXTRACTION_STATUS_REVIEW_REQUIRED,
    TableArtifact,
    derive_extraction_blockers,
    get_comparison_indicators,
    get_extraction_confidence,
    get_extraction_quality_flags,
    get_extraction_quality_profile,
    get_extraction_status,
    get_vision_raw_indicators,
    is_auto_compare_eligible,
    is_matching_eligible,
)
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.matching_normalizer import (
    header_schema_similarity,
    is_generic_title,
    normalize_for_matching,
    strip_temporal_expressions,
)

logger = logging.getLogger(__name__)

UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}
DEFAULT_SHORTLIST_SIZE = 5
DEFAULT_ROUTER_MODEL = "gpt-4o-mini"
DEFAULT_ROUTER_TIMEOUT = 30.0

_TABLE_NUMBER_RE = re.compile(
    r"\b(?:tableau|table)\s*([a-z]?\d+[a-z]?|\d+[a-z]?)\b",
    re.IGNORECASE,
)
_TABLE_NUMBER_SHORT_RE = re.compile(r"\bT\s*([0-9]+[A-Za-z]?)\b")

# Page-local structure scoring (same-page / near-page multi-table matching)
PAGE_LOCAL_ORDER_MATCH_SAME_PAGE = 0.20
PAGE_LOCAL_ORDER_CONFLICT_SAME_PAGE = -0.15
PAGE_LOCAL_ORDER_MATCH_NEAR_PAGE = 0.12
PAGE_LOCAL_ORDER_CONFLICT_NEAR_PAGE = -0.08
PAGE_LOCAL_ROLE_MATCH_BONUS = 0.06
BBOX_Y_SIMILARITY_WEIGHT = 0.05

_GENERIC_INDICATOR_KEYS = frozenset(
    {
        "total",
        "autres",
        "other",
        "canada",
        "etats unis",
        "etatsunis",
        "united states",
        "europe",
        "royaume uni",
        "uk",
        "asie",
        "amerique latine",
        "amlat",
        "particuliers",
        "entreprises",
        "secteur public",
        "administrations publiques",
    }
)


def _is_generic_indicator_key(value: str) -> bool:
    """Détecter les indicateurs trop génériques pour servir d'ancrage de matching."""
    normalized = normalize_for_matching(str(value or ""), target="indicator")
    if not normalized:
        return True
    collapsed = normalized.replace(" ", "")
    if normalized in _GENERIC_INDICATOR_KEYS or collapsed in {
        item.replace(" ", "") for item in _GENERIC_INDICATOR_KEYS
    }:
        return True
    tokens = set(normalized.split())
    return tokens.issubset(
        {
            "total",
            "autres",
            "other",
            "canada",
            "etats",
            "unis",
            "united",
            "states",
            "europe",
            "royaume",
            "uni",
            "uk",
            "asie",
            "amerique",
            "latine",
            "particuliers",
            "entreprises",
            "secteur",
            "public",
            "administrations",
            "publiques",
        }
    )


def _safe_int(value: Any, default: int = 0) -> int:
    """Convertir en entier avec valeur de repli pour les métadonnées bruitées."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convertir en flottant avec valeur de repli pour les métriques incomplètes."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_uid(table: TableArtifact) -> str:
    """Construire l'identifiant stable local utilisé par le moteur de pairing."""
    return f"{table.section}|{table.table_id}|p{table.page_pdf}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Dédupliquer une liste textuelle sans perdre l'ordre d'origine."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _normalize_title_value(value: str | None) -> str:
    """Normaliser un titre en retirant le bruit temporel avant comparaison."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = strip_temporal_expressions(raw, target="title", aggressive=True)
    return normalize_for_matching(cleaned, target="title")


def _normalized_title(table: TableArtifact) -> str:
    """Retourner le meilleur titre normalisé disponible pour un tableau."""
    candidates = [
        getattr(table, "title_clean", None),
        getattr(table, "title_raw", None),
        getattr(table, "title", None),
    ]
    for candidate in candidates:
        normalized = _normalize_title_value(candidate)
        if normalized:
            return normalized
    return ""


def _normalized_headers(table: TableArtifact) -> list[str]:
    """Normaliser les en-têtes d'un tableau pour le scoring de compatibilité."""
    headers = [str(h or "").strip() for h in (getattr(table, "headers", None) or [])]
    return [normalize_for_matching(item, target="header") for item in headers if item]


def _normalized_table_number(table: TableArtifact) -> str:
    """Résoudre le numéro de tableau canonique depuis le champ dédié ou le titre."""
    explicit = normalize_for_matching(str(getattr(table, "table_number", "") or ""))
    if explicit:
        return explicit
    title_candidates = [
        getattr(table, "title_raw", None),
        getattr(table, "title_clean", None),
        getattr(table, "title", None),
    ]
    for title in title_candidates:
        text = str(title or "").strip()
        if not text:
            continue
        match = _TABLE_NUMBER_RE.search(text) or _TABLE_NUMBER_SHORT_RE.search(text)
        if match:
            return normalize_for_matching(match.group(1), target="generic")
    return ""


def _indicator_keys(table: TableArtifact) -> list[str]:
    """Return canonical indicator keys for pairing overlap (footnote markers normalized away)."""
    raw = get_comparison_indicators(table)
    keys: list[str] = []
    seen: set[str] = set()
    for s in raw:
        k = normalize_indicator_for_comparison(str(s or "").strip())
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _is_known_section(value: str | None) -> bool:
    """Dire si une section est exploitable pour un matching strict."""
    return bool(value and str(value).strip().lower() not in UNKNOWN_SECTIONS)


def _section_value(table: TableArtifact) -> str:
    """Lire la section d'un tableau dans une forme normalisée minimale."""
    return str(getattr(table, "section", "") or "").strip().lower()


def _same_or_unknown_section(left: TableArtifact, right: TableArtifact) -> bool:
    """Autoriser un pair si les sections sont identiques ou toutes deux inconnues."""
    left_section = _section_value(left)
    right_section = _section_value(right)
    if left_section == right_section:
        return True
    if not _is_known_section(left_section) and not _is_known_section(right_section):
        return True
    return False


def _row_count(table: TableArtifact) -> int:
    """Estimer la taille utile d'un tableau à partir des indicateurs ou des lignes."""
    indicators = _indicator_keys(table)
    if indicators:
        return len(indicators)
    return len(getattr(table, "rows", None) or [])


def _size_compatibility(left: TableArtifact, right: TableArtifact) -> float:
    """Mesurer la compatibilité de taille entre deux tableaux sur une échelle 0..1."""
    left_count = max(_row_count(left), 1)
    right_count = max(_row_count(right), 1)
    return min(left_count, right_count) / max(left_count, right_count)


def _jaccard(left: list[str], right: list[str]) -> float:
    """Calculer le Jaccard entre deux listes d'indicateurs normalisés."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def _containment(left: list[str], right: list[str]) -> float:
    """Mesurer à quel point le plus petit ensemble est contenu dans l'autre."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(min(len(left_set), len(right_set)), 1)


def _page_bonus(left: TableArtifact, right: TableArtifact) -> float:
    """Attribuer un bonus de proximité quand deux tableaux sont proches en pagination."""
    delta = abs(_safe_int(getattr(left, "page_pdf", 0)) - _safe_int(getattr(right, "page_pdf", 0)))
    if delta <= 1:
        return 1.0
    if delta <= 3:
        return 0.7
    if delta <= 6:
        return 0.4
    return 0.0


def _bbox_y_center(table: TableArtifact) -> float | None:
    """Normalized vertical center of table bbox in [0, 1]. None if no valid bbox."""
    bbox = getattr(table, "bbox", None)
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        top = float(bbox[1])
        bottom = float(bbox[3])
        return (top + bottom) / 2.0
    if isinstance(bbox, dict):
        if "t" in bbox and "b" in bbox:
            return (float(bbox["t"]) + float(bbox["b"])) / 2.0
        if "y0" in bbox and "y1" in bbox:
            return (float(bbox["y0"]) + float(bbox["y1"])) / 2.0
        if "y" in bbox and "height" in bbox:
            return float(bbox["y"]) + float(bbox["height"]) / 2.0
    return None


def _page_local_order_bonus(left: TableArtifact, right: TableArtifact) -> float:
    """Bonus/penalty for same/different table index on (same or near) page."""
    idx_left = getattr(left, "table_index_on_page", None)
    idx_right = getattr(right, "table_index_on_page", None)
    if idx_left is None or idx_right is None:
        return 0.0
    page_left = _safe_int(getattr(left, "page_pdf", 0))
    page_right = _safe_int(getattr(right, "page_pdf", 0))
    delta_page = abs(page_left - page_right)
    if delta_page == 0:
        if idx_left == idx_right:
            return PAGE_LOCAL_ORDER_MATCH_SAME_PAGE
        return PAGE_LOCAL_ORDER_CONFLICT_SAME_PAGE
    if delta_page == 1:
        if idx_left == idx_right:
            return PAGE_LOCAL_ORDER_MATCH_NEAR_PAGE
        return PAGE_LOCAL_ORDER_CONFLICT_NEAR_PAGE
    return 0.0


def _bbox_y_similarity(left: TableArtifact, right: TableArtifact) -> float:
    """Similarity of vertical position when both on same or near page. [0, 1]."""
    page_left = _safe_int(getattr(left, "page_pdf", 0))
    page_right = _safe_int(getattr(right, "page_pdf", 0))
    if abs(page_left - page_right) > 1:
        return 0.0
    y_left = _bbox_y_center(left)
    y_right = _bbox_y_center(right)
    if y_left is None or y_right is None:
        return 0.0
    return 1.0 - min(1.0, abs(y_left - y_right))


def _page_local_role_bonus(left: TableArtifact, right: TableArtifact) -> float:
    """Bonus when page_local_role matches (first/first, last/last, etc.)."""
    role_left = getattr(left, "page_local_role", None) or ""
    role_right = getattr(right, "page_local_role", None) or ""
    if not role_left or not role_right:
        return 0.0
    if role_left == role_right:
        return PAGE_LOCAL_ROLE_MATCH_BONUS
    return 0.0


def _title_similarity(left: TableArtifact, right: TableArtifact) -> float:
    """Mesurer la similarité entre titres normalisés en ignorant les titres génériques."""
    title_left = _normalized_title(left)
    title_right = _normalized_title(right)
    if not title_left or not title_right:
        return 0.0
    if is_generic_title(title_left) or is_generic_title(title_right):
        return 0.0
    return SequenceMatcher(None, title_left, title_right).ratio()


def _headers_similarity(left: TableArtifact, right: TableArtifact) -> float:
    """Comparer les schémas d'en-têtes pour renforcer ou affaiblir un candidat."""
    headers_left = _normalized_headers(left)
    headers_right = _normalized_headers(right)
    if not headers_left or not headers_right:
        return 0.0
    return header_schema_similarity(headers_left, headers_right)


def _build_section_indicator_frequency(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
) -> dict[str, dict[str, int]]:
    """Build indicator frequency from certified tables only (same population as views)."""
    counts: dict[str, dict[str, int]] = {}
    for table in [*tables_t1, *tables_t2]:
        if not is_auto_compare_eligible(table):
            continue
        section = _section_value(table)
        section_counts = counts.setdefault(section, {})
        for key in set(_indicator_keys(table)):
            section_counts[key] = section_counts.get(key, 0) + 1
    return counts


def _distinctive_indicator_keys(
    table: TableArtifact,
    *,
    section_frequencies: dict[str, dict[str, int]],
    section_table_count: int,
) -> list[str]:
    """Garder les indicateurs les plus distinctifs d'une section pour le shortlist."""
    keys = _indicator_keys(table)
    if not keys:
        return []
    section = _section_value(table)
    counts = section_frequencies.get(section, {})
    common_threshold = max(3, int(math.ceil(max(section_table_count, 1) * 0.45)))
    distinctive = [
        key
        for key in keys
        if not _is_generic_indicator_key(key)
        and counts.get(key, 0) < common_threshold
    ]
    if distinctive:
        return distinctive
    return [key for key in keys if not _is_generic_indicator_key(key)][:6]


def _quality_profile(table: TableArtifact) -> dict[str, Any]:
    """Normalized quality profile for pairing (avoid ad hoc debug_metrics)."""
    return get_extraction_quality_profile(table)


def _raw_indicator_stability(t1_view: TableView, t2_view: TableView) -> float:
    """
    Compare raw first-column indicators overlap vs normalized overlap.
    Returns a value in [0, 1]; low when normalization makes noisy extractions look similar.
    """
    raw_left = [
        normalize_for_matching(s, target="indicator")
        for s in get_vision_raw_indicators(t1_view.table)
        if normalize_for_matching(s, target="indicator")
    ]
    raw_right = [
        normalize_for_matching(s, target="indicator")
        for s in get_vision_raw_indicators(t2_view.table)
        if normalize_for_matching(s, target="indicator")
    ]
    raw_overlap = _jaccard(raw_left, raw_right)
    norm_overlap = _jaccard(t1_view.indicator_keys, t2_view.indicator_keys)
    if norm_overlap <= 0.2:
        return 1.0
    if norm_overlap <= 0:
        return 1.0
    return min(1.0, raw_overlap / max(norm_overlap, 0.01))


def _independent_anchor_count(score: "CandidateScore") -> int:
    """
    Return the precomputed anchor count from the candidate score.
    Used to require at least 2 anchors when extraction quality is low.
    """
    return score.anchor_count


@dataclass(slots=True)
class TableView:
    """Vue pré-calculée d'un ``TableArtifact`` pour le moteur de pairing.

    Cette dataclass encapsule toutes les représentations normalisées d'un tableau
    qui sont utilisées pour le calcul des scores de similarité. Elle est construite
    une seule fois par tableau (via ``from_table``) pour éviter les recalculs
    répétés lors de la comparaison de chaque paire T1/T2.

    Attributs
    ---------
    table:
        Référence vers le ``TableArtifact`` source.
    uid:
        Identifiant unique stable (format ``<section>|<table_id>|p<page>``).
    section:
        Nom de section normalisé en minuscules.
    normalized_title:
        Titre normalisé (sans dates, sans unités, sans bruit temporel).
    normalized_headers:
        Liste des en-têtes normalisés pour la comparaison de schéma.
    normalized_table_number:
        Numéro de tableau canonique extrait du champ dédié ou du titre.
    indicator_keys:
        Liste des clés canoniques de tous les indicateurs du tableau.
    indicator_distinctive_keys:
        Sous-ensemble des indicateurs les plus distinctifs (rares dans la section),
        utilisés comme ancres de matching fiables.
    table_size:
        Nombre d'indicateurs (ou de lignes si pas d'indicateurs).
    quality_profile:
        Profil de qualité de l'extraction (flags, confiance, méthode).
    """

    table: TableArtifact
    uid: str
    section: str
    normalized_title: str
    normalized_headers: list[str]
    normalized_table_number: str
    indicator_keys: list[str]
    indicator_distinctive_keys: list[str]
    table_size: int
    quality_profile: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_table(
        cls,
        table: TableArtifact,
        *,
        section_frequencies: dict[str, dict[str, int]],
        section_table_count: int,
    ) -> "TableView":
        return cls(
            table=table,
            uid=_table_uid(table),
            section=_section_value(table),
            normalized_title=_normalized_title(table),
            normalized_headers=_normalized_headers(table),
            normalized_table_number=_normalized_table_number(table),
            indicator_keys=_indicator_keys(table),
            indicator_distinctive_keys=_distinctive_indicator_keys(
                table,
                section_frequencies=section_frequencies,
                section_table_count=section_table_count,
            ),
            table_size=_row_count(table),
            quality_profile=_quality_profile(table),
        )


def _title_reliability_numeric(table: TableArtifact) -> float:
    """Map title_reliability string to a score in [0, 1] for pairing."""
    r = getattr(table, "title_reliability", None) or ""
    if str(r).strip().lower() == "reliable":
        return 1.0
    if str(r).strip().lower() == "weak":
        return 0.5
    return 0.3


@dataclass(slots=True)
class CandidateScore:
    """Score multi-signaux détaillé d'une paire T1/T2 candidate.

    Cette dataclass est produite par ``_candidate_score`` pour chaque paire
    (T1, T2) dans la shortlist. Elle contient tous les signaux individuels
    ainsi que le score total pondéré, ce qui permet au router de prendre une
    décision informée et de fournir des codes de raison explicites.

    Attributs principaux
    --------------------
    t1_view, t2_view:
        Vues des tableaux T1 et T2 de la paire.
    total_score:
        Score composite final dans [0, 1].
    indicator_distinctive_overlap:
        Jaccard entre les indicateurs distinctifs (rares) des deux tableaux.
        Signal le plus important (poids 36 %).
    indicator_containment:
        Mesure à quel point le plus petit ensemble d'indicateurs est contenu
        dans le plus grand (poids 23 %).
    header_compatibility:
        Similarité des schémas d'en-têtes (colonnes) (poids 12 %).
    title_similarity:
        Similarité textuelle des titres normalisés (poids 11 %).
    table_number_match / table_number_conflict:
        Indique si les numéros de tableau correspondent ou sont en conflit.
    anchor_count:
        Nombre de signaux forts indépendants confirmant l'appariement.
        Utilisé par le router pour décider si la qualité est suffisante.
    quality_suspect:
        ``True`` si l'extraction d'un des deux tableaux est de mauvaise qualité.
    """
    t1_view: TableView
    t2_view: TableView
    total_score: float
    section_compatibility: float
    indicator_distinctive_overlap: float
    indicator_global_overlap: float
    indicator_containment: float
    header_compatibility: float
    title_similarity: float
    table_number_bonus: float
    order_proximity_bonus: float
    size_compatibility: float
    table_number_match: bool
    table_number_conflict: bool
    explanation: list[str] = field(default_factory=list)
    quality_penalty: float = 0.0
    raw_indicator_stability: float = 1.0
    anchor_count: int = 0
    confidence_cap_reason: str | None = None
    title_reliability_score: float = 0.0
    quality_suspect: bool = False
    page_local_order_bonus: float = 0.0
    page_local_role_bonus: float = 0.0
    bbox_y_similarity: float = 0.0

    def as_feature_dict(self) -> dict[str, Any]:
        return {
            "t1_uid": self.t1_view.uid,
            "t2_uid": self.t2_view.uid,
            "score": round(self.total_score, 6),
            "section_compatibility": round(self.section_compatibility, 6),
            "indicator_distinctive_overlap": round(
                self.indicator_distinctive_overlap, 6
            ),
            "indicator_global_overlap": round(self.indicator_global_overlap, 6),
            "indicator_containment": round(self.indicator_containment, 6),
            "header_compatibility": round(self.header_compatibility, 6),
            "title_similarity": round(self.title_similarity, 6),
            "table_number_bonus": round(self.table_number_bonus, 6),
            "order_proximity_bonus": round(self.order_proximity_bonus, 6),
            "size_compatibility": round(self.size_compatibility, 6),
            "table_number_match": self.table_number_match,
            "table_number_conflict": self.table_number_conflict,
            "page_local_order_bonus": round(self.page_local_order_bonus, 6),
            "page_local_role_bonus": round(self.page_local_role_bonus, 6),
            "bbox_y_similarity": round(self.bbox_y_similarity, 6),
            "reasons": list(self.explanation),
        }


def _candidate_score(t1_view: TableView, t2_view: TableView) -> CandidateScore:
    """Calculer le score multi-signaux d'une paire de tableaux T1/T2 candidate.

    Calcule tous les signaux individuels (chevauchement d'indicateurs, similarité
    de titre, compatibilité des en-têtes, proximité de page, etc.) puis les
    combine en un score total pondéré selon la formule décrite dans la docstring
    du module.

    Applique également des corrections :
    - Réduction de la contribution du titre si sa fiabilité est faible.
    - Réduction de la contribution des en-têtes si la qualité d'extraction est
      suspecte.
    - Pénalité d'instabilité si le chevauchement brut/normalisé est incohérent.
    - Pénalité de qualité si l'extraction est de mauvaise qualité.

    Calcule aussi ``anchor_count`` : le nombre de signaux forts indépendants
    (numéro de tableau, chevauchement distinctif, containment, titre fiable,
    en-têtes compatibles) qui confirment l'appariement.
    """
    section_compatibility = 1.0 if _same_or_unknown_section(t1_view.table, t2_view.table) else 0.0
    indicator_distinctive_overlap = _jaccard(
        t1_view.indicator_distinctive_keys,
        t2_view.indicator_distinctive_keys,
    )
    indicator_global_overlap = _jaccard(t1_view.indicator_keys, t2_view.indicator_keys)
    indicator_containment = _containment(t1_view.indicator_keys, t2_view.indicator_keys)
    header_compatibility = _headers_similarity(t1_view.table, t2_view.table)
    title_similarity = _title_similarity(t1_view.table, t2_view.table)
    size_compatibility = _size_compatibility(t1_view.table, t2_view.table)
    order_proximity_bonus = _page_bonus(t1_view.table, t2_view.table)
    page_local_order_bonus = _page_local_order_bonus(t1_view.table, t2_view.table)
    page_local_role_bonus = _page_local_role_bonus(t1_view.table, t2_view.table)
    bbox_y_sim = _bbox_y_similarity(t1_view.table, t2_view.table)

    same_number = bool(
        t1_view.normalized_table_number
        and t1_view.normalized_table_number == t2_view.normalized_table_number
    )
    table_number_conflict = bool(
        t1_view.normalized_table_number
        and t2_view.normalized_table_number
        and t1_view.normalized_table_number != t2_view.normalized_table_number
    )
    table_number_bonus = 0.15 if same_number else -0.05 if table_number_conflict else 0.0

    explanation: list[str] = []
    if same_number:
        explanation.append("same_table_number")
    if table_number_conflict:
        explanation.append("table_number_conflict")
    if indicator_distinctive_overlap >= 0.45:
        explanation.append("distinctive_overlap_strong")
    if indicator_containment >= 0.65:
        explanation.append("indicator_containment_strong")
    if title_similarity >= 0.75:
        explanation.append("title_similarity_strong")
    if header_compatibility >= 0.70:
        explanation.append("headers_compatible")

    total = (
        0.36 * indicator_distinctive_overlap
        + 0.23 * indicator_containment
        + 0.12 * header_compatibility
        + 0.11 * title_similarity
        + 0.08 * section_compatibility
        + 0.06 * size_compatibility
        + 0.04 * order_proximity_bonus
        + table_number_bonus
    )
    raw_stability = _raw_indicator_stability(t1_view, t2_view)
    title_reliability_score = min(
        _title_reliability_numeric(t1_view.table),
        _title_reliability_numeric(t2_view.table),
    )
    flags1 = get_extraction_quality_flags(t1_view.table)
    flags2 = get_extraction_quality_flags(t2_view.table)
    low_quality_left = (
        flags1.get("crop_rejected")
        or flags1.get("recrop_failed_incomplete")
        or not flags1.get("vision_extraction_applied", True)
    )
    low_quality_right = (
        flags2.get("crop_rejected")
        or flags2.get("recrop_failed_incomplete")
        or not flags2.get("vision_extraction_applied", True)
    )
    conf_left = get_extraction_confidence(t1_view.table)
    conf_right = get_extraction_confidence(t2_view.table)
    low_conf = conf_left < 0.5 or conf_right < 0.5
    quality_suspect = low_quality_left or low_quality_right or low_conf
    if quality_suspect:
        if low_quality_left or low_quality_right:
            quality_suspect = True
    instability_penalty = 0.0
    if indicator_global_overlap >= 0.5 and raw_stability < 0.6:
        instability_penalty = 0.15 * (1.0 - raw_stability)
    title_contribution = 0.11 * title_similarity
    if title_reliability_score < 0.6:
        title_contribution *= 0.3
    header_contribution = 0.12 * header_compatibility
    if quality_suspect:
        header_contribution *= 0.7
    total = (
        0.36 * indicator_distinctive_overlap
        + 0.23 * indicator_containment
        + header_contribution
        + title_contribution
        + 0.08 * section_compatibility
        + 0.06 * size_compatibility
        + 0.04 * order_proximity_bonus
        + table_number_bonus
        + page_local_order_bonus
        + page_local_role_bonus
        + BBOX_Y_SIMILARITY_WEIGHT * bbox_y_sim
        - instability_penalty
    )
    quality_penalty = 0.0
    if quality_suspect:
        quality_penalty = 0.08
    total = total - quality_penalty
    total = max(0.0, min(1.0, total))

    anchor_count = 0
    if same_number:
        anchor_count += 1
    if indicator_distinctive_overlap >= 0.40:
        anchor_count += 1
    if indicator_containment >= 0.52:
        anchor_count += 1
    if title_reliability_score >= 0.7 and title_similarity >= 0.72:
        anchor_count += 1
    if header_compatibility >= 0.72:
        anchor_count += 1

    return CandidateScore(
        t1_view=t1_view,
        t2_view=t2_view,
        total_score=total,
        section_compatibility=section_compatibility,
        indicator_distinctive_overlap=indicator_distinctive_overlap,
        indicator_global_overlap=indicator_global_overlap,
        indicator_containment=indicator_containment,
        header_compatibility=header_compatibility,
        title_similarity=title_similarity,
        table_number_bonus=table_number_bonus,
        order_proximity_bonus=order_proximity_bonus,
        size_compatibility=size_compatibility,
        table_number_match=same_number,
        table_number_conflict=table_number_conflict,
        explanation=explanation,
        quality_penalty=quality_penalty,
        raw_indicator_stability=raw_stability,
        anchor_count=anchor_count,
        confidence_cap_reason=None,
        title_reliability_score=title_reliability_score,
        quality_suspect=quality_suspect,
        page_local_order_bonus=page_local_order_bonus,
        page_local_role_bonus=page_local_role_bonus,
        bbox_y_similarity=bbox_y_sim,
    )


def _eligible_table_views(
    tables: list[TableArtifact],
    *,
    section_frequencies: dict[str, dict[str, int]],
    section_counts: dict[str, int],
) -> tuple[list[TableView], list[dict[str, Any]]]:
    """Construire les vues uniquement pour les tableaux certifiés pour la comparaison.

    Les tableaux non certifiés (extraction échouée, blocages de comparaison) sont
    exclus et listés dans la liste ``ineligible`` avec les raisons explicites.

    Deux raisons d'inéligibilité possibles :
    - ``"extraction_not_certified"`` : le statut d'extraction n'est pas certifié
      (``is_auto_compare_eligible`` retourne ``False``).
    - ``"comparison_ineligible"`` : le tableau a des blocages de comparaison
      explicites (``comparison_eligible`` est ``False``).

    Retourne un tuple ``(views, ineligible)`` où ``views`` contient les
    ``TableView`` des tableaux éligibles et ``ineligible`` contient les
    métadonnées des tableaux exclus.
    """
    views: list[TableView] = []
    ineligible: list[dict[str, Any]] = []
    for table in tables:
        uid = _table_uid(table)
        if not is_auto_compare_eligible(table):
            extraction_blockers = derive_extraction_blockers(table)
            extraction_status = get_extraction_status(table)
            ineligible.append(
                {
                    "table_id": table.table_id,
                    "uid": uid,
                    "section": table.section,
                    "page": table.page_pdf,
                    "title": table.title or "",
                    "comparison_blockers": list(getattr(table, "comparison_blockers", []) or []),
                    "extraction_blockers": extraction_blockers,
                    "extraction_status": extraction_status,
                    "reason": "extraction_not_certified",
                }
            )
            continue
        if not bool(getattr(table, "comparison_eligible", False)):
            blockers = list(getattr(table, "comparison_blockers", []) or [])
            ineligible.append(
                {
                    "table_id": table.table_id,
                    "uid": uid,
                    "section": table.section,
                    "page": table.page_pdf,
                    "title": table.title or "",
                    "comparison_blockers": blockers,
                    "reason": "comparison_ineligible",
                }
            )
            continue
        section = _section_value(table)
        views.append(
            TableView.from_table(
                table,
                section_frequencies=section_frequencies,
                section_table_count=section_counts.get(section, 1),
            )
        )
    return views, ineligible


def _hard_filter_candidate(t1_view: TableView, t2_view: TableView) -> bool:
    """Écarter tôt les candidats incompatibles avant le scoring détaillé."""
    if not _same_or_unknown_section(t1_view.table, t2_view.table):
        return False
    if _size_compatibility(t1_view.table, t2_view.table) < 0.35:
        return False
    return True


def _shortlist_candidates(
    t2_view: TableView,
    t1_views: list[TableView],
    *,
    shortlist_size: int,
) -> list[CandidateScore]:
    """Construire une shortlist bornée et diversifiée de candidats T1 pour un tableau T2.

    Pour chaque tableau T2, cette fonction :
    1. Filtre les candidats T1 incompatibles (section différente, taille trop
       différente) via ``_hard_filter_candidate``.
    2. Calcule le score de chaque candidat restant.
    3. Sélectionne une shortlist diversifiée de taille ``shortlist_size`` (max 5)
       en incluant :
       - Le meilleur candidat par score global.
       - Le meilleur par ordre de page local.
       - Le meilleur par rôle de page (si applicable).
       - Le meilleur par numéro de tableau (si un match de numéro existe).
       - Le meilleur par chevauchement d'indicateurs distinctifs.
       - Le meilleur par chevauchement global d'indicateurs.
       - Le meilleur par similarité de titre (si > 0).
       - Les suivants par score global jusqu'à atteindre ``shortlist_size``.

    Cette diversification garantit que le router voit les candidats les plus
    pertinents selon différentes dimensions, même si le meilleur score global
    n'est pas le bon appariement.
    """
    candidates: list[CandidateScore] = []
    for t1_view in t1_views:
        if not _hard_filter_candidate(t1_view, t2_view):
            continue
        score = _candidate_score(t1_view, t2_view)
        if score.total_score < 0.12 and score.indicator_global_overlap < 0.15:
            continue
        candidates.append(score)
    if not candidates:
        return []
    candidates.sort(key=lambda item: item.total_score, reverse=True)

    selected: list[CandidateScore] = []
    seen: set[str] = set()

    def _add(candidate: CandidateScore) -> None:
        if candidate.t1_view.uid in seen:
            return
        seen.add(candidate.t1_view.uid)
        selected.append(candidate)

    _add(candidates[0])
    by_page_local_order = max(candidates, key=lambda item: item.page_local_order_bonus)
    by_role = max(candidates, key=lambda item: item.page_local_role_bonus)
    by_distinct = max(candidates, key=lambda item: item.indicator_distinctive_overlap)
    by_global = max(candidates, key=lambda item: item.indicator_global_overlap)
    by_title = max(candidates, key=lambda item: item.title_similarity)
    _add(by_page_local_order)
    if by_role.page_local_role_bonus > 0:
        _add(by_role)
    if any(candidate.table_number_match for candidate in candidates):
        by_number = max(
            candidates,
            key=lambda item: (1 if item.table_number_match else 0, item.total_score),
        )
        _add(by_number)
    _add(by_distinct)
    _add(by_global)
    if by_title.title_similarity > 0:
        _add(by_title)
    for candidate in candidates:
        _add(candidate)
        if len(selected) >= shortlist_size:
            break
    return selected[:shortlist_size]


@dataclass(slots=True)
class PairingDecision:
    """Décision finale du router pour un tableau courant et sa shortlist.

    Attributs
    ---------
    decision:
        Résultat de la décision : ``"match"``, ``"ambiguous"`` ou ``"no_match"``.
    matched_t1_uid:
        UID du tableau T1 apparié (``None`` si ``ambiguous`` ou ``no_match``).
    confidence:
        Score de confiance de la décision dans [0, 1].
    reason_codes:
        Liste de codes de raison explicites pour la décision
        (ex. ``["same_table_number", "distinctive_overlap_strong"]``).
    requires_review:
        ``True`` si la paire nécessite une validation humaine.
    pairing_quality_flags:
        Flags de qualité supplémentaires (ex. ``["low_quality_extraction"]``).
    pairing_confidence_cap:
        Raison d'un plafonnement de la confiance (ex. ``"quality_suspect"``).
    """

    decision: str
    matched_t1_uid: str | None
    confidence: float
    reason_codes: list[str]
    requires_review: bool
    pairing_quality_flags: list[str] = field(default_factory=list)
    pairing_confidence_cap: str | None = None


class PairingRouter(Protocol):
    """Contrat minimal (protocole) des routeurs chargés d'accepter ou rejeter une shortlist.

    Un routeur reçoit la vue du tableau T2 et sa shortlist de candidats T1,
    et retourne une ``PairingDecision``. Cette interface permet de substituer
    le routeur conservateur par défaut par un routeur LLM (GPT-4o) via la
    configuration, sans modifier le reste du pipeline.
    """
    def route(
        self,
        *,
        t2_view: TableView,
        candidates: list[CandidateScore],
    ) -> PairingDecision: ...


class ConservativePairingRouter:
    """Routeur déterministe conservateur utilisé par défaut.

    Ce routeur applique une cascade de règles de rejet avant d'accepter un
    appariement. Il préfère déclarer ``ambiguous`` plutôt que de risquer un
    faux positif. Les règles de rejet (dans l'ordre d'application) sont :

    1. **Similarité de famille sans ancre distinctive** : chevauchement global
       élevé (≥ 0.60) mais chevauchement distinctif faible (< 0.25) → ambiguous.
    2. **Instabilité brut/normalisé** : le chevauchement normalisé est bien
       supérieur au brut, suggérant une normalisation trop agressive → ambiguous.
    3. **Candidats proches** : le deuxième candidat a un score ≥ 0.55 et l'écart
       avec le premier est < 0.06 → ambiguous.
    4. **Numéro seul insuffisant** : le numéro de tableau correspond mais le
       chevauchement d'indicateurs est trop faible (< 0.35) → ambiguous.
    5. **Qualité faible + titre/en-têtes seulement** : extraction suspecte avec
       < 2 ancres indépendantes et seul le titre ou les en-têtes correspondent
       → ambiguous.
    6. **Score suffisant + ancres multiples** : si le score est ≥ 0.66, le
       containment ≥ 0.52, et au moins un signal fort supplémentaire est présent
       → match.
    7. Tous les autres cas → no_match.
    """

    def route(
        self,
        *,
        t2_view: TableView,
        candidates: list[CandidateScore],
    ) -> PairingDecision:
        if not candidates:
            return PairingDecision(
                decision="no_match",
                matched_t1_uid=None,
                confidence=0.0,
                reason_codes=["no_candidates"],
                requires_review=False,
            )

        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        reasons = list(top.explanation)

        low_quality = top.quality_suspect
        anchors = top.anchor_count

        if (
            top.indicator_global_overlap >= 0.60
            and top.indicator_distinctive_overlap < 0.25
        ):
            reasons = ["family_similarity_without_distinctive_anchor", *reasons]
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.75, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if top.raw_indicator_stability < 0.6 and top.indicator_global_overlap >= 0.5:
            reasons.append("raw_normalized_instability")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.75, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if second is not None and second.total_score >= 0.55:
            if abs(top.total_score - second.total_score) < 0.06:
                reasons.append("close_competing_candidates")
                if low_quality:
                    return PairingDecision(
                        decision="ambiguous",
                        matched_t1_uid=None,
                        confidence=min(0.75, top.total_score),
                        reason_codes=reasons,
                        requires_review=True,
                    )
                return PairingDecision(
                    decision="ambiguous",
                    matched_t1_uid=None,
                    confidence=min(0.80, top.total_score),
                    reason_codes=reasons,
                    requires_review=True,
                )

        if top.table_number_match and top.indicator_containment < 0.35:
            reasons.append("table_number_only_is_not_enough")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.70, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if low_quality and anchors < 2 and (
            top.title_similarity >= 0.72 or top.header_compatibility >= 0.72
        ):
            reasons.append("low_quality_title_or_header_only")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=min(0.70, top.total_score),
                reason_codes=reasons,
                requires_review=True,
            )

        if (
            top.total_score >= 0.66
            and top.indicator_containment >= 0.52
            and (
                top.indicator_distinctive_overlap >= 0.40
                or (top.title_reliability_score >= 0.7 and top.title_similarity >= 0.72)
                or top.header_compatibility >= 0.72
                or top.table_number_match
            )
        ):
            if low_quality and anchors < 2:
                reasons.append("low_quality_insufficient_anchors")
                return PairingDecision(
                    decision="ambiguous",
                    matched_t1_uid=None,
                    confidence=min(0.75, top.total_score),
                    reason_codes=reasons,
                    requires_review=True,
                )
            reasons.append("deterministic_router_match")
            confidence = top.total_score
            quality_flags: list[str] = []
            cap_reason: str | None = None
            if low_quality:
                confidence = min(0.82, confidence)
                quality_flags.append("low_quality_match")
                cap_reason = "extraction_quality_cap"
            return PairingDecision(
                decision="match",
                matched_t1_uid=top.t1_view.uid,
                confidence=confidence,
                reason_codes=reasons,
                requires_review=False,
                pairing_quality_flags=quality_flags,
                pairing_confidence_cap=cap_reason,
            )

        if top.total_score >= 0.56 and top.indicator_containment >= 0.45:
            reasons.append("conservative_router_ambiguous")
            return PairingDecision(
                decision="ambiguous",
                matched_t1_uid=None,
                confidence=top.total_score,
                reason_codes=reasons,
                requires_review=True,
            )

        reasons.append("insufficient_pairing_signal")
        return PairingDecision(
            decision="no_match",
            matched_t1_uid=None,
            confidence=top.total_score,
            reason_codes=reasons,
            requires_review=False,
        )


def _table_summary_for_router(view: TableView) -> dict[str, Any]:
    """Sérialiser une vue de tableau en résumé court pour routing ou debug."""
    return {
        "uid": view.uid,
        "section": view.section,
        "table_number_raw": getattr(view.table, "table_number", None),
        "title_raw": getattr(view.table, "title", None),
        "title_normalized": view.normalized_title,
        "headers_normalized": view.normalized_headers[:6],
        "top_distinctive_indicators": view.indicator_distinctive_keys[:8],
        "top_global_indicators": view.indicator_keys[:10],
        "table_size": view.table_size,
    }


class BatchLLMPairingRouter:
    """Optional final router using GPT-4o style batch routing.

    The shortlist is still deterministic; the LLM only chooses between provided
    candidates or returns ``no_match`` / ``ambiguous``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_ROUTER_MODEL,
        timeout: float = DEFAULT_ROUTER_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _build_prompt(self, *, t2_view: TableView, candidates: list[CandidateScore]) -> str:
        payload = {
            "instruction": (
                "Choisis le candidat qui represente le meme tableau conceptuel/reglementaire. "
                "N'utilise pas l'ordre des lignes comme preuve. Le numero et le titre "
                "sont secondaires. Une forte similarite de famille sans indicateurs "
                "distinctifs doit mener a 'ambiguous' ou 'no_match'."
            ),
            "allowed_answers": ["match:<t1_uid>", "no_match", "ambiguous"],
            "t2": _table_summary_for_router(t2_view),
            "candidates": [
                {
                    "candidate": _table_summary_for_router(score.t1_view),
                    "features": score.as_feature_dict(),
                }
                for score in candidates
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def route(
        self,
        *,
        t2_view: TableView,
        candidates: list[CandidateScore],
    ) -> PairingDecision:
        if not candidates:
            return PairingDecision(
                decision="no_match",
                matched_t1_uid=None,
                confidence=0.0,
                reason_codes=["no_candidates"],
                requires_review=False,
            )
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package missing; falling back to deterministic router")
            return ConservativePairingRouter().route(t2_view=t2_view, candidates=candidates)

        client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un routeur de tableaux bancaires. "
                        "Reponds uniquement en JSON avec decision, matched_t1_uid, "
                        "confidence et reason_codes."
                    ),
                },
                {"role": "user", "content": self._build_prompt(t2_view=t2_view, candidates=candidates)},
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            payload = json.loads(content or "{}")
        except json.JSONDecodeError:
            logger.warning("LLM router returned invalid JSON; using deterministic fallback")
            return ConservativePairingRouter().route(t2_view=t2_view, candidates=candidates)

        decision = str(payload.get("decision", "") or "").strip()
        matched_t1_uid = str(payload.get("matched_t1_uid", "") or "").strip() or None
        confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence", 0.0))))
        reason_codes = [
            str(item).strip()
            for item in (payload.get("reason_codes", []) or [])
            if str(item).strip()
        ]
        if decision not in {"match", "no_match", "ambiguous"}:
            return ConservativePairingRouter().route(t2_view=t2_view, candidates=candidates)
        if decision == "match" and not matched_t1_uid:
            return ConservativePairingRouter().route(t2_view=t2_view, candidates=candidates)
        return PairingDecision(
            decision=decision,
            matched_t1_uid=matched_t1_uid,
            confidence=confidence,
            reason_codes=reason_codes or ["batch_llm_router"],
            requires_review=decision != "match",
        )


def _build_router(
    *,
    api_key: str | None,
    bank_code: str | None,
) -> PairingRouter:
    """Choisir le router final selon la configuration et la présence d'une clé API."""
    cfg = get_matching_thresholds(bank_code=bank_code)
    enabled = bool(cfg.get("batch_llm_pairing_enabled", False))
    if enabled and api_key:
        model = str(cfg.get("batch_llm_pairing_model", DEFAULT_ROUTER_MODEL))
        timeout = _safe_float(cfg.get("batch_llm_pairing_timeout", DEFAULT_ROUTER_TIMEOUT), DEFAULT_ROUTER_TIMEOUT)
        return BatchLLMPairingRouter(api_key=api_key, model=model, timeout=timeout)
    return ConservativePairingRouter()


def _pair_dict(candidate: CandidateScore, decision: PairingDecision) -> dict[str, Any]:
    """Construire le payload canonique d'une paire retenue par le moteur."""
    t1 = candidate.t1_view.table
    t2 = candidate.t2_view.table
    payload = candidate.as_feature_dict()
    payload.update(
        {
            "t1_uid": candidate.t1_view.uid,
            "t2_uid": candidate.t2_view.uid,
            "t1_table_id": t1.table_id,
            "t2_table_id": t2.table_id,
            "page_t1": t1.page_pdf,
            "page_t2": t2.page_pdf,
            "title_t1": t1.title or "",
            "title_t2": t2.title or "",
            "section": t1.section or t2.section or "",
            "reason": decision.reason_codes[0] if decision.reason_codes else "matched_pair",
            "reason_codes": list(decision.reason_codes),
            "decision_level": "match",
            "router_decision": decision.decision,
            "pairing_confidence": round(decision.confidence, 6),
        }
    )
    if getattr(decision, "pairing_quality_flags", None):
        payload["pairing_quality_flags"] = list(decision.pairing_quality_flags)
    if getattr(decision, "pairing_confidence_cap", None):
        payload["pairing_confidence_cap"] = decision.pairing_confidence_cap
    return payload


def _unmatched_previous_entry(
    view: TableView,
    *,
    ambiguous: bool,
    reason: str,
) -> dict[str, Any]:
    """Sérialiser un tableau T1 resté sans match après le pairing."""
    table = view.table
    return {
        "t1_uid": view.uid,
        "t1_table_id": table.table_id,
        "section": table.section,
        "page_t1": table.page_pdf,
        "title_t1": table.title or "",
        "reason": reason,
        "unmatched_status": "ambiguous" if ambiguous else "confirmed",
        "suspicion_flags": [reason] if ambiguous else [],
    }


def _unmatched_current_entry(
    view: TableView,
    *,
    ambiguous: bool,
    reason: str,
) -> dict[str, Any]:
    """Sérialiser un tableau T2 resté sans match après le pairing."""
    table = view.table
    return {
        "t2_uid": view.uid,
        "t2_table_id": table.table_id,
        "section": table.section,
        "page_t2": table.page_pdf,
        "title_t2": table.title or "",
        "reason": reason,
        "unmatched_status": "ambiguous" if ambiguous else "confirmed",
        "suspicion_flags": [reason] if ambiguous else [],
    }


def _added_table_entry(view: TableView) -> dict[str, Any]:
    """Construire l'entrée canonique d'un tableau ajouté côté courant."""
    table = view.table
    return {
        "uid": view.uid,
        "table_id": table.table_id,
        "t2_uid": view.uid,
        "t2_table_id": table.table_id,
        "section": table.section,
        "page": table.page_pdf,
        "page_t2": table.page_pdf,
        "title": table.title or "",
        "title_t2": table.title or "",
        "reason": "added_table",
        "source_reason": "pairing_unmatched",
        "first_column_indicators": list(view.indicator_keys),
        "first_column_indicators_raw": list(getattr(table, "first_column_indicators_raw", None) or []),
    }


def _removed_table_entry(view: TableView) -> dict[str, Any]:
    """Construire l'entrée canonique d'un tableau supprimé côté précédent."""
    table = view.table
    return {
        "uid": view.uid,
        "table_id": table.table_id,
        "t1_uid": view.uid,
        "t1_table_id": table.table_id,
        "section": table.section,
        "page": table.page_pdf,
        "page_t1": table.page_pdf,
        "title": table.title or "",
        "title_t1": table.title or "",
        "reason": "removed_table",
        "source_reason": "pairing_unmatched",
        "first_column_indicators": list(view.indicator_keys),
        "first_column_indicators_raw": list(getattr(table, "first_column_indicators_raw", None) or []),
    }


def _candidate_debug_entry_t2(t2_uid: str, candidates: list[CandidateScore]) -> dict[str, Any]:
    """Exporter les meilleurs candidats T1 d'un tableau T2 non apparié."""
    return {
        "t2_uid": t2_uid,
        "candidates": [candidate.as_feature_dict() for candidate in candidates],
    }


def _candidate_debug_entry_t1(
    t1_uid: str,
    candidates: list[CandidateScore],
) -> dict[str, Any]:
    """Exporter les meilleurs candidats T2 d'un tableau T1 non apparié."""
    normalized = []
    for candidate in candidates:
        item = candidate.as_feature_dict()
        item["t1_uid"] = t1_uid
        normalized.append(item)
    return {"t1_uid": t1_uid, "candidates": normalized}


def _resolve_collisions(
    provisional_matches: list[tuple[TableView, CandidateScore, PairingDecision]],
) -> tuple[list[tuple[TableView, CandidateScore, PairingDecision]], list[dict[str, Any]]]:
    """Résoudre les collisions où plusieurs T2 pointent vers le même T1."""
    matches_by_t1: dict[str, list[tuple[TableView, CandidateScore, PairingDecision]]] = {}
    for entry in provisional_matches:
        _, candidate, _ = entry
        matches_by_t1.setdefault(candidate.t1_view.uid, []).append(entry)

    accepted: list[tuple[TableView, CandidateScore, PairingDecision]] = []
    ambiguous: list[dict[str, Any]] = []

    for t1_uid, entries in matches_by_t1.items():
        if len(entries) == 1:
            accepted.append(entries[0])
            continue
        entries_sorted = sorted(entries, key=lambda item: item[2].confidence, reverse=True)
        top = entries_sorted[0]
        runner_up = entries_sorted[1]
        if top[2].confidence - runner_up[2].confidence >= 0.05:
            accepted.append(top)
            for t2_view, candidate, decision in entries_sorted[1:]:
                ambiguous.append(
                    {
                        "decision": "ambiguous",
                        "matched_t1_uid": None,
                        "confidence": round(decision.confidence, 6),
                        "reason_codes": ["collision_after_assignment"],
                        "t2_uid": t2_view.uid,
                        "candidate_t1_uids": [t1_uid],
                    }
                )
            continue
        candidate_ids = [candidate.t1_view.uid for _, candidate, _ in entries_sorted]
        for t2_view, candidate, decision in entries_sorted:
            ambiguous.append(
                {
                    "decision": "ambiguous",
                    "matched_t1_uid": None,
                    "confidence": round(decision.confidence, 6),
                    "reason_codes": ["collision_after_assignment"],
                    "t2_uid": t2_view.uid,
                    "candidate_t1_uids": candidate_ids,
                }
            )
    return accepted, ambiguous


def run_strict_intra_section_compare(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    overlap_threshold: float | None = None,
    bank_code: str | None = None,
    embedding_service: Any | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Official pairing facade used by the active pipeline.

    Delegates to the recall-first engine when ``recall_first_engine_enabled``
    is set in matching config (default: True).  Falls back to the legacy
    conservative engine otherwise.
    """
    cfg = get_matching_thresholds(bank_code=bank_code)
    if bool(cfg.get("recall_first_engine_enabled", True)):
        try:
            return run_recall_first_compare(
                tables_t1,
                tables_t2,
                overlap_threshold=overlap_threshold,
                bank_code=bank_code,
                embedding_service=embedding_service,
                api_key=api_key,
            )
        except Exception:
            logger.exception(
                "recall_first_compare failed; falling back to legacy engine"
            )

    return _run_legacy_conservative_compare(
        tables_t1,
        tables_t2,
        bank_code=bank_code,
        api_key=api_key,
    )


def _run_legacy_conservative_compare(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    bank_code: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Legacy conservative pairing engine (pre-recall-first)."""
    overlap_threshold = None
    embedding_service = None
    del overlap_threshold, embedding_service

    section_frequencies = _build_section_indicator_frequency(tables_t1, tables_t2)
    section_counts: dict[str, int] = {}
    for table in [*tables_t1, *tables_t2]:
        if not is_auto_compare_eligible(table):
            continue
        section = _section_value(table)
        section_counts[section] = section_counts.get(section, 0) + 1

    t1_views, ineligible_t1_raw = _eligible_table_views(
        tables_t1,
        section_frequencies=section_frequencies,
        section_counts=section_counts,
    )
    t2_views, ineligible_t2_raw = _eligible_table_views(
        tables_t2,
        section_frequencies=section_frequencies,
        section_counts=section_counts,
    )

    router = _build_router(api_key=api_key, bank_code=bank_code)
    shortlist_size = _safe_int(
        get_matching_thresholds(bank_code=bank_code).get(
            "pairing_shortlist_size", DEFAULT_SHORTLIST_SIZE
        ),
        DEFAULT_SHORTLIST_SIZE,
    )
    shortlist_size = max(1, min(shortlist_size, 5))

    candidate_map_t2: dict[str, list[CandidateScore]] = {}
    candidate_map_t1: dict[str, list[CandidateScore]] = {}
    ambiguous_pairs: list[dict[str, Any]] = []
    provisional_matches: list[tuple[TableView, CandidateScore, PairingDecision]] = []
    explicit_no_match_t2: set[str] = set()

    for t2_view in t2_views:
        shortlist = _shortlist_candidates(
            t2_view,
            t1_views,
            shortlist_size=shortlist_size,
        )
        candidate_map_t2[t2_view.uid] = shortlist
        for candidate in shortlist:
            candidate_map_t1.setdefault(candidate.t1_view.uid, []).append(candidate)

        decision = router.route(t2_view=t2_view, candidates=shortlist)
        if decision.decision == "match" and decision.matched_t1_uid:
            chosen = next(
                (
                    candidate
                    for candidate in shortlist
                    if candidate.t1_view.uid == decision.matched_t1_uid
                ),
                None,
            )
            if chosen is not None:
                provisional_matches.append((t2_view, chosen, decision))
                continue
        if decision.decision == "ambiguous":
            ambiguous_pairs.append(
                {
                    "decision": "ambiguous",
                    "matched_t1_uid": None,
                    "confidence": round(decision.confidence, 6),
                    "reason_codes": list(decision.reason_codes),
                    "t2_uid": t2_view.uid,
                    "candidate_t1_uids": [candidate.t1_view.uid for candidate in shortlist],
                }
            )
        else:
            explicit_no_match_t2.add(t2_view.uid)

    accepted_matches, collision_ambiguous = _resolve_collisions(provisional_matches)
    ambiguous_pairs.extend(collision_ambiguous)

    matched_t1_uids = {candidate.t1_view.uid for _, candidate, _ in accepted_matches}
    matched_t2_uids = {t2_view.uid for t2_view, _, _ in accepted_matches}

    ambiguous_t2_uids = {
        str(item.get("t2_uid", "")).strip()
        for item in ambiguous_pairs
        if str(item.get("t2_uid", "")).strip()
    }
    ambiguous_t1_uids: set[str] = set()
    for item in ambiguous_pairs:
        for candidate_uid in item.get("candidate_t1_uids", []) or []:
            candidate_uid = str(candidate_uid).strip()
            if candidate_uid:
                ambiguous_t1_uids.add(candidate_uid)

    pairs = [_pair_dict(candidate, decision) for _, candidate, decision in accepted_matches]

    unmatched_t1: list[dict[str, Any]] = []
    unmatched_t2: list[dict[str, Any]] = []
    removed_tables: list[dict[str, Any]] = []
    added_tables: list[dict[str, Any]] = []

    for view in t1_views:
        if view.uid in matched_t1_uids:
            continue
        if view.uid in ambiguous_t1_uids:
            unmatched_t1.append(
                _unmatched_previous_entry(
                    view,
                    ambiguous=True,
                    reason="ambiguous_candidate",
                )
            )
            continue
        unmatched_t1.append(
            _unmatched_previous_entry(
                view,
                ambiguous=False,
                reason="removed_table",
            )
        )
        removed_tables.append(_removed_table_entry(view))

    for view in t2_views:
        if view.uid in matched_t2_uids:
            continue
        if view.uid in ambiguous_t2_uids:
            unmatched_t2.append(
                _unmatched_current_entry(
                    view,
                    ambiguous=True,
                    reason="ambiguous_candidate",
                )
            )
            continue
        reason = "no_match" if view.uid in explicit_no_match_t2 else "added_table"
        unmatched_t2.append(
            _unmatched_current_entry(
                view,
                ambiguous=False,
                reason=reason,
            )
        )
        added_tables.append(_added_table_entry(view))

    unmatched_confirmed_t1 = [
        item for item in unmatched_t1 if item.get("unmatched_status") == "confirmed"
    ]
    unmatched_ambiguous_t1 = [
        item for item in unmatched_t1 if item.get("unmatched_status") == "ambiguous"
    ]
    unmatched_confirmed_t2 = [
        item for item in unmatched_t2 if item.get("unmatched_status") == "confirmed"
    ]
    unmatched_ambiguous_t2 = [
        item for item in unmatched_t2 if item.get("unmatched_status") == "ambiguous"
    ]
    comparable_t1 = len(t1_views)
    comparable_t2 = len(t2_views)
    pairing_coverage = round(
        len(pairs) / max(min(comparable_t1, comparable_t2), 1),
        6,
    )

    def _ineligible_entry(item: dict[str, Any], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            f"{prefix}_table_id": item["table_id"],
            f"{prefix}_uid": item["uid"],
            "section": item["section"],
            f"page_{prefix}": item["page"],
            f"title_{prefix}": item["title"],
            "reason": item["reason"],
            "comparison_blockers": item.get("comparison_blockers", []),
        }
        if "extraction_blockers" in item:
            out["extraction_blockers"] = item["extraction_blockers"]
        if "extraction_status" in item:
            out["extraction_status"] = item["extraction_status"]
        return out

    ineligible_t1 = [_ineligible_entry(item, "t1") for item in ineligible_t1_raw]
    ineligible_t2 = [_ineligible_entry(item, "t2") for item in ineligible_t2_raw]

    return {
        "pairs": pairs,
        "matched_pairs": list(pairs),
        "probable_pairs": [],
        "suspicious_pairs": [],
        "ambiguous_pairs": ambiguous_pairs,
        "ambiguous_tables": [
            {
                "side": "previous",
                "uid": item.get("t1_uid"),
                "table_id": item.get("t1_table_id"),
                "title": item.get("title_t1"),
                "page": item.get("page_t1"),
                "section": item.get("section", ""),
                "reason": item.get("reason", ""),
            }
            for item in unmatched_ambiguous_t1
        ]
        + [
            {
                "side": "current",
                "uid": item.get("t2_uid"),
                "table_id": item.get("t2_table_id"),
                "title": item.get("title_t2"),
                "page": item.get("page_t2"),
                "section": item.get("section", ""),
                "reason": item.get("reason", ""),
            }
            for item in unmatched_ambiguous_t2
        ],
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "added_tables_confirmed": list(added_tables),
        "removed_tables_confirmed": list(removed_tables),
        "unmatched_t1": unmatched_t1,
        "unmatched_t2": unmatched_t2,
        "unmatched_confirmed_t1": unmatched_confirmed_t1,
        "unmatched_confirmed_t2": unmatched_confirmed_t2,
        "unmatched_ambiguous_t1": unmatched_ambiguous_t1,
        "unmatched_ambiguous_t2": unmatched_ambiguous_t2,
        "ambiguous_unmatched_previous": list(unmatched_ambiguous_t1),
        "ambiguous_unmatched_current": list(unmatched_ambiguous_t2),
        "ineligible_t1": ineligible_t1,
        "ineligible_t2": ineligible_t2,
        "debug_unmatched_candidates": [
            _candidate_debug_entry_t1(uid, candidates)
            for uid, candidates in sorted(candidate_map_t1.items())
            if candidates
        ],
        "debug_unmatched_candidates_t2": [
            _candidate_debug_entry_t2(uid, candidates)
            for uid, candidates in sorted(candidate_map_t2.items())
            if candidates
        ],
        "rescued_matches_count": 0,
        "split_merge_rescues_count": 0,
        "vision_rescued_pairs": [],
        "reasons": [pair.get("reason", "") for pair in pairs if pair.get("reason")],
        "diagnostics": {
            "router": router.__class__.__name__,
            "shortlist_size": shortlist_size,
        },
        "matching_diagnostics": {
            "pairs_count": len(pairs),
            "ambiguous_pairs_count": len(ambiguous_pairs),
            "unmatched_t1_count": len(unmatched_t1),
            "unmatched_t2_count": len(unmatched_t2),
            "ineligible_t1_count": len(ineligible_t1),
            "ineligible_t2_count": len(ineligible_t2),
            "tables_comparable_t1": comparable_t1,
            "tables_comparable_t2": comparable_t2,
            "pairing_coverage": pairing_coverage,
        },
        "tables_comparable_t1": comparable_t1,
        "tables_comparable_t2": comparable_t2,
        "pairing_coverage": pairing_coverage,
    }


# ---------------------------------------------------------------------------
# Recall-First Engine
# ---------------------------------------------------------------------------

_RF_MIN_MATCH_SCORE = 0.25
_RF_MIN_MATCH_MARGIN = 0.04
_RF_STRONG_PAIR_MIN_SCORE = 0.50
_RF_STRONG_PAIR_MIN_MARGIN = 0.08
_RF_REVIEW_CANDIDATE_MIN_SCORE = 0.15
_RF_MAX_ELIMINATION_ROUNDS = 5
_RF_MIN_INDICATOR_SIGNAL = 0.10


def _build_section_frequency_relaxed(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
) -> dict[str, dict[str, int]]:
    """Build indicator frequency from all eligible tables (certified + review_required)."""
    counts: dict[str, dict[str, int]] = {}
    for table in [*tables_t1, *tables_t2]:
        if not is_matching_eligible(table):
            continue
        if not bool(getattr(table, "comparison_eligible", False)):
            continue
        section = _section_value(table)
        section_counts = counts.setdefault(section, {})
        for key in set(_indicator_keys(table)):
            section_counts[key] = section_counts.get(key, 0) + 1
    return counts


def _eligible_table_views_relaxed(
    tables: list[TableArtifact],
    *,
    section_frequencies: dict[str, dict[str, int]],
    section_counts: dict[str, int],
) -> tuple[list[TableView], list[dict[str, Any]]]:
    """Build views for certified and review_required tables.  Blocked tables excluded.

    Uses ``is_matching_eligible`` to gate on extraction status.
    """
    views: list[TableView] = []
    ineligible: list[dict[str, Any]] = []
    for table in tables:
        uid = _table_uid(table)
        status = get_extraction_status(table)
        if not is_matching_eligible(table):
            ineligible.append(
                {
                    "table_id": table.table_id,
                    "uid": uid,
                    "section": table.section,
                    "page": table.page_pdf,
                    "title": table.title or "",
                    "comparison_blockers": list(
                        getattr(table, "comparison_blockers", []) or []
                    ),
                    "extraction_blockers": derive_extraction_blockers(table),
                    "extraction_status": status,
                    "reason": "extraction_not_certified",
                }
            )
            continue
        if not bool(getattr(table, "comparison_eligible", False)):
            blockers = list(getattr(table, "comparison_blockers", []) or [])
            ineligible.append(
                {
                    "table_id": table.table_id,
                    "uid": uid,
                    "section": table.section,
                    "page": table.page_pdf,
                    "title": table.title or "",
                    "comparison_blockers": blockers,
                    "reason": "comparison_ineligible",
                }
            )
            continue
        section = _section_value(table)
        views.append(
            TableView.from_table(
                table,
                section_frequencies=section_frequencies,
                section_table_count=section_counts.get(section, 1),
            )
        )
    return views, ineligible


def _precompute_all_scores(
    t1_views: list[TableView],
    t2_views: list[TableView],
) -> list[list[CandidateScore]]:
    """Precompute CandidateScore for every T2 x T1 pair (no filtering)."""
    return [
        [_candidate_score(t1_view, t2_view) for t1_view in t1_views]
        for t2_view in t2_views
    ]


def _hungarian_optimal_assign(
    float_scores: list[list[float]],
    n_t2: int,
    n_t1: int,
) -> list[tuple[int, int, float]]:
    """Global optimal 1:1 assignment via Hungarian algorithm.

    Uses an augmented (n1+n2) x (n1+n2) matrix so that each table can
    choose to stay unmatched (score 0) rather than accept a bad partner.
    """
    if _scipy_linear_sum_assignment is None:
        return _greedy_fallback_assign(float_scores, n_t2, n_t1)

    import numpy as np

    N = n_t1 + n_t2
    M = np.full((N, N), -1e9, dtype=np.float64)

    for i in range(n_t2):
        for j in range(n_t1):
            M[i, j] = float_scores[i][j]

    for i in range(n_t2):
        M[i, n_t1 + i] = 0.0
    for j in range(n_t1):
        M[n_t2 + j, j] = 0.0
    M[n_t2:, n_t1:] = 0.0

    row_ind, col_ind = _scipy_linear_sum_assignment(M, maximize=True)

    assignments: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        if r < n_t2 and c < n_t1:
            assignments.append((int(r), int(c), float_scores[r][c]))
    return assignments


def _greedy_fallback_assign(
    float_scores: list[list[float]],
    n_t2: int,
    n_t1: int,
) -> list[tuple[int, int, float]]:
    """Greedy best-first fallback when scipy is unavailable."""
    pool: list[tuple[float, int, int]] = []
    for i in range(n_t2):
        for j in range(n_t1):
            if float_scores[i][j] > 0:
                pool.append((float_scores[i][j], i, j))
    pool.sort(reverse=True)

    used_t2: set[int] = set()
    used_t1: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for score, i, j in pool:
        if i in used_t2 or j in used_t1:
            continue
        out.append((i, j, score))
        used_t2.add(i)
        used_t1.add(j)
    return out


def _has_content_signal(cs: CandidateScore, threshold: float) -> bool:
    """True when a candidate pair has enough indicator-level evidence to match."""
    return (
        cs.indicator_global_overlap >= threshold
        or cs.indicator_containment >= threshold
    )


def _has_plausible_signal(cs: CandidateScore, threshold: float) -> bool:
    """True when a candidate pair has enough evidence to warrant analyst review.

    More permissive than ``_has_content_signal``: accepts strong structural
    signals (table number, title) even without indicator overlap, since those
    cases often indicate a renamed or restructured table.
    """
    if _has_content_signal(cs, threshold):
        return True
    if cs.table_number_match:
        return True
    if cs.title_similarity >= 0.75:
        return True
    return False


def _pair_quality_sufficient(
    cs: CandidateScore,
    min_indicator_signal: float = _RF_MIN_INDICATOR_SIGNAL,
) -> bool:
    """Minimum quality gate for including a pair in the Hungarian assignment.

    Only requires basic content signal. Softer quality checks (family
    similarity, instability) are applied later as classification downgrades,
    not as hard gates on assignment.
    """
    return _has_content_signal(cs, min_indicator_signal)


def _pair_needs_review(cs: CandidateScore) -> bool:
    """True when a pair passes content gate but has suspicious quality signals.

    These pairs should be downgraded from ``match`` to ``review_candidate``
    rather than rejected outright.
    """
    if (
        cs.indicator_global_overlap >= 0.60
        and cs.indicator_distinctive_overlap < 0.25
    ):
        return True
    if cs.raw_indicator_stability < 0.6 and cs.indicator_global_overlap >= 0.5:
        return True
    return False


def _run_progressive_elimination(
    all_scores: list[list[CandidateScore]],
    t2_views: list[TableView],
    t1_views: list[TableView],
    *,
    min_match_score: float,
    min_match_margin: float,
    strong_score: float,
    strong_margin: float,
    max_iterations: int,
    min_indicator_signal: float = _RF_MIN_INDICATOR_SIGNAL,
) -> tuple[
    list[tuple[int, int, CandidateScore, str, float]],
    set[int],
    set[int],
    list[dict[str, Any]],
]:
    """Progressive elimination: freeze strong pairs, re-run on residual.

    Returns (frozen_pairs, remaining_t2_indices, remaining_t1_indices, log).
    Each frozen pair is (t2_idx, t1_idx, CandidateScore, match_stage, margin).
    """
    active_t2 = list(range(len(t2_views)))
    active_t1 = list(range(len(t1_views)))
    frozen: list[tuple[int, int, CandidateScore, str, float]] = []
    log: list[dict[str, Any]] = []

    for round_num in range(max_iterations):
        if not active_t2 or not active_t1:
            break

        n2, n1 = len(active_t2), len(active_t1)
        fmat = [
            [all_scores[t2i][t1i].total_score for t1i in active_t1]
            for t2i in active_t2
        ]

        assignments = _hungarian_optimal_assign(fmat, n2, n1)

        new_strong: list[tuple[int, int, CandidateScore, str, float]] = []
        new_weak: list[tuple[int, int, CandidateScore, str, float]] = []

        for local_r, local_c, score in assignments:
            if score < min_match_score:
                continue
            orig_t2 = active_t2[local_r]
            orig_t1 = active_t1[local_c]
            cs = all_scores[orig_t2][orig_t1]
            if not _pair_quality_sufficient(cs, min_indicator_signal):
                continue
            alts_row = [fmat[local_r][k] for k in range(n1) if k != local_c]
            alts_col = [fmat[k][local_c] for k in range(n2) if k != local_r]
            margin = min(
                score - (max(alts_row) if alts_row else 0.0),
                score - (max(alts_col) if alts_col else 0.0),
            )
            if margin < min_match_margin:
                continue
            if score >= strong_score and margin >= strong_margin:
                new_strong.append((orig_t2, orig_t1, cs, "strong_match", margin))
            else:
                new_weak.append((orig_t2, orig_t1, cs, "residual_match", margin))

        log.append(
            {
                "round": round_num,
                "active_t2": n2,
                "active_t1": n1,
                "strong_pairs": len(new_strong),
                "weak_pairs": len(new_weak),
            }
        )

        if not new_strong:
            frozen.extend(new_weak)
            break

        frozen.extend(new_strong)
        frozen_t2 = {t2 for t2, _, _, _, _ in new_strong}
        frozen_t1 = {t1 for _, t1, _, _, _ in new_strong}
        active_t2 = [i for i in active_t2 if i not in frozen_t2]
        active_t1 = [i for i in active_t1 if i not in frozen_t1]

    return frozen, set(active_t2), set(active_t1), log


# ---------------------------------------------------------------------------
# Rescue passes
# ---------------------------------------------------------------------------

_RF_CROSS_SECTION_RESCUE_MIN_SCORE = 0.20
_RF_SPLIT_MERGE_INDICATOR_MIN = 0.30


def _run_cross_section_rescue(
    all_scores: list[list[CandidateScore]],
    remaining_t2: set[int],
    remaining_t1: set[int],
    t2_views: list[TableView],
    t1_views: list[TableView],
    *,
    min_score: float = _RF_CROSS_SECTION_RESCUE_MIN_SCORE,
    min_indicator_signal: float = _RF_MIN_INDICATOR_SIGNAL,
) -> tuple[list[tuple[int, int, CandidateScore, str, float]], list[dict[str, Any]]]:
    """Rescue pass targeting cross-section pairs with strong content signals.

    Returns rescued pairs and a log of actions taken.
    """
    rescued: list[tuple[int, int, CandidateScore, str, float]] = []
    log: list[dict[str, Any]] = []

    if not remaining_t2 or not remaining_t1:
        return rescued, log

    candidates: list[tuple[float, int, int, CandidateScore]] = []
    for t2i in remaining_t2:
        for t1i in remaining_t1:
            cs = all_scores[t2i][t1i]
            if cs.section_compatibility > 0:
                continue
            if cs.total_score < min_score:
                continue
            if not _has_content_signal(cs, min_indicator_signal):
                continue
            if cs.anchor_count < 2:
                continue
            candidates.append((cs.total_score, t2i, t1i, cs))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_t2: set[int] = set()
    used_t1: set[int] = set()

    for score, t2i, t1i, cs in candidates:
        if t2i in used_t2 or t1i in used_t1:
            continue
        margin = score - max(
            (
                all_scores[t2i][t1j].total_score
                for t1j in remaining_t1
                if t1j != t1i and t1j not in used_t1
            ),
            default=0.0,
        )
        margin = min(
            margin,
            score
            - max(
                (
                    all_scores[t2j][t1i].total_score
                    for t2j in remaining_t2
                    if t2j != t2i and t2j not in used_t2
                ),
                default=0.0,
            ),
        )
        rescued.append((t2i, t1i, cs, "cross_section_rescue", margin))
        used_t2.add(t2i)
        used_t1.add(t1i)
        log.append(
            {
                "rescue_type": "cross_section",
                "t2_uid": t2_views[t2i].uid,
                "t1_uid": t1_views[t1i].uid,
                "score": round(score, 6),
                "margin": round(margin, 6),
                "anchor_count": cs.anchor_count,
            }
        )

    return rescued, log


def _run_split_merge_rescue(
    all_scores: list[list[CandidateScore]],
    remaining_t2: set[int],
    remaining_t1: set[int],
    t2_views: list[TableView],
    t1_views: list[TableView],
    *,
    min_combined_overlap: float = _RF_SPLIT_MERGE_INDICATOR_MIN,
    min_primary_score: float = _RF_CROSS_SECTION_RESCUE_MIN_SCORE,
    min_indicator_signal: float = _RF_MIN_INDICATOR_SIGNAL,
) -> tuple[list[dict[str, Any]], set[int], set[int], list[dict[str, Any]]]:
    """Rescue pass for split/merge cases.

    Detects when two adjacent unmatched tables on one side could together
    match a single table on the other side.

    Returns (rescued_pairs, used_t2_indices, used_t1_indices, log).
    """
    rescues: list[dict[str, Any]] = []
    used_t2: set[int] = set()
    used_t1: set[int] = set()
    log: list[dict[str, Any]] = []

    if len(remaining_t2) < 2 and len(remaining_t1) < 2:
        return rescues, used_t2, used_t1, log

    def _adjacent(views: list[TableView], idx_a: int, idx_b: int) -> bool:
        a, b = views[idx_a].table, views[idx_b].table
        if abs((a.page_pdf or 0) - (b.page_pdf or 0)) > 1:
            return False
        sec_a = (a.section or "").strip().lower()
        sec_b = (b.section or "").strip().lower()
        if sec_a and sec_b and sec_a != sec_b:
            return False
        return True

    def _combined_indicator_overlap(
        keys_a: list[str] | frozenset[str],
        keys_b: list[str] | frozenset[str],
        keys_target: list[str] | frozenset[str],
    ) -> float:
        sa = set(keys_a)
        sb = set(keys_b)
        st = set(keys_target)
        combined = sa | sb
        if not combined or not st:
            return 0.0
        return len(combined & st) / max(len(combined | st), 1)

    candidates: list[dict[str, Any]] = []

    sorted_remaining_t2 = sorted(
        remaining_t2,
        key=lambda i: (
            (t2_views[i].table.section or "").lower(),
            t2_views[i].table.page_pdf or 0,
            t2_views[i].table.table_id or "",
        ),
    )
    for idx, t2a in enumerate(sorted_remaining_t2):
        for t2b in sorted_remaining_t2[idx + 1 :]:
            if not _adjacent(t2_views, t2a, t2b):
                continue
            for t1i in remaining_t1:
                overlap = _combined_indicator_overlap(
                    t2_views[t2a].indicator_keys,
                    t2_views[t2b].indicator_keys,
                    t1_views[t1i].indicator_keys,
                )
                if overlap < min_combined_overlap:
                    continue
                cs_left = all_scores[t2a][t1i]
                cs_right = all_scores[t2b][t1i]
                primary_t2 = t2a if cs_left.total_score >= cs_right.total_score else t2b
                primary_cs = cs_left if primary_t2 == t2a else cs_right
                if primary_cs.total_score < min_primary_score:
                    continue
                if not _has_plausible_signal(primary_cs, min_indicator_signal):
                    continue
                candidates.append(
                    {
                        "rescue_type": "split_merge_rescue",
                        "direction": "t2_split",
                        "score": (0.65 * overlap) + (0.35 * primary_cs.total_score),
                        "combined_indicator_overlap": overlap,
                        "primary_score": primary_cs.total_score,
                        "primary_t2": primary_t2,
                        "t1i": t1i,
                        "t2_members": [t2a, t2b],
                    }
                )

    sorted_remaining_t1 = sorted(
        remaining_t1,
        key=lambda i: (
            (t1_views[i].table.section or "").lower(),
            t1_views[i].table.page_pdf or 0,
            t1_views[i].table.table_id or "",
        ),
    )
    for idx, t1a in enumerate(sorted_remaining_t1):
        for t1b in sorted_remaining_t1[idx + 1 :]:
            if not _adjacent(t1_views, t1a, t1b):
                continue
            for t2i in remaining_t2:
                overlap = _combined_indicator_overlap(
                    t1_views[t1a].indicator_keys,
                    t1_views[t1b].indicator_keys,
                    t2_views[t2i].indicator_keys,
                )
                if overlap < min_combined_overlap:
                    continue
                cs_left = all_scores[t2i][t1a]
                cs_right = all_scores[t2i][t1b]
                primary_t1 = t1a if cs_left.total_score >= cs_right.total_score else t1b
                primary_cs = cs_left if primary_t1 == t1a else cs_right
                if primary_cs.total_score < min_primary_score:
                    continue
                if not _has_plausible_signal(primary_cs, min_indicator_signal):
                    continue
                candidates.append(
                    {
                        "rescue_type": "split_merge_rescue",
                        "direction": "t1_split",
                        "score": (0.65 * overlap) + (0.35 * primary_cs.total_score),
                        "combined_indicator_overlap": overlap,
                        "primary_score": primary_cs.total_score,
                        "primary_t1": primary_t1,
                        "t2i": t2i,
                        "t1_members": [t1a, t1b],
                    }
                )

    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for candidate in candidates:
        direction = str(candidate.get("direction", ""))
        if direction == "t2_split":
            t1i = int(candidate["t1i"])
            member_t2 = [int(idx) for idx in candidate.get("t2_members", [])]
            if t1i in used_t1 or any(idx in used_t2 for idx in member_t2):
                continue
            primary_t2 = int(candidate["primary_t2"])
            primary_cs = all_scores[primary_t2][t1i]
            t1_tbl = t1_views[t1i].table
            t2_tbl = t2_views[primary_t2].table
            payload = primary_cs.as_feature_dict()
            payload.update(
                {
                    "t1_uid": t1_views[t1i].uid,
                    "t2_uid": t2_views[primary_t2].uid,
                    "t1_table_id": t1_tbl.table_id,
                    "t2_table_id": t2_tbl.table_id,
                    "page_t1": t1_tbl.page_pdf,
                    "page_t2": t2_tbl.page_pdf,
                    "title_t1": t1_tbl.title or "",
                    "title_t2": t2_tbl.title or "",
                    "section": t1_tbl.section or t2_tbl.section or "",
                    "reason": "split_merge_rescue",
                    "reason_codes": ["split_merge_rescue"],
                    "decision_level": "match",
                    "router_decision": "match",
                    "pairing_confidence": round(float(candidate["score"]), 6),
                    "match_stage": "split_merge_rescue",
                    "match_margin": round(
                        max(
                            float(candidate["combined_indicator_overlap"])
                            - float(candidate["primary_score"]),
                            0.0,
                        ),
                        6,
                    ),
                    "rescue_type": "split_merge_rescue",
                    "match_source": "split_merge_rescue",
                    "split_members_t2": [t2_views[idx].uid for idx in member_t2],
                    "split_probable": True,
                    "combined_indicator_overlap": round(
                        float(candidate["combined_indicator_overlap"]), 6
                    ),
                }
            )
            rescues.append(payload)
            used_t1.add(t1i)
            used_t2.update(member_t2)
            log.append(
                {
                    "rescue_type": "split_merge_rescue",
                    "direction": direction,
                    "t1_uid": t1_views[t1i].uid,
                    "t2_uids": [t2_views[idx].uid for idx in member_t2],
                    "combined_indicator_overlap": round(
                        float(candidate["combined_indicator_overlap"]), 6
                    ),
                    "score": round(float(candidate["score"]), 6),
                }
            )
            continue

        t2i = int(candidate["t2i"])
        member_t1 = [int(idx) for idx in candidate.get("t1_members", [])]
        if t2i in used_t2 or any(idx in used_t1 for idx in member_t1):
            continue
        primary_t1 = int(candidate["primary_t1"])
        primary_cs = all_scores[t2i][primary_t1]
        t1_tbl = t1_views[primary_t1].table
        t2_tbl = t2_views[t2i].table
        payload = primary_cs.as_feature_dict()
        payload.update(
            {
                "t1_uid": t1_views[primary_t1].uid,
                "t2_uid": t2_views[t2i].uid,
                "t1_table_id": t1_tbl.table_id,
                "t2_table_id": t2_tbl.table_id,
                "page_t1": t1_tbl.page_pdf,
                "page_t2": t2_tbl.page_pdf,
                "title_t1": t1_tbl.title or "",
                "title_t2": t2_tbl.title or "",
                "section": t1_tbl.section or t2_tbl.section or "",
                "reason": "split_merge_rescue",
                "reason_codes": ["split_merge_rescue"],
                "decision_level": "match",
                "router_decision": "match",
                "pairing_confidence": round(float(candidate["score"]), 6),
                "match_stage": "split_merge_rescue",
                "match_margin": round(
                    max(
                        float(candidate["combined_indicator_overlap"])
                        - float(candidate["primary_score"]),
                        0.0,
                    ),
                    6,
                ),
                "rescue_type": "split_merge_rescue",
                "match_source": "split_merge_rescue",
                "merge_members_t1": [t1_views[idx].uid for idx in member_t1],
                "merge_probable": True,
                "combined_indicator_overlap": round(
                    float(candidate["combined_indicator_overlap"]), 6
                ),
            }
        )
        rescues.append(payload)
        used_t2.add(t2i)
        used_t1.update(member_t1)
        log.append(
            {
                "rescue_type": "split_merge_rescue",
                "direction": direction,
                "t2_uid": t2_views[t2i].uid,
                "t1_uids": [t1_views[idx].uid for idx in member_t1],
                "combined_indicator_overlap": round(
                    float(candidate["combined_indicator_overlap"]), 6
                ),
                "score": round(float(candidate["score"]), 6),
            }
        )

    return rescues, used_t2, used_t1, log


def run_recall_first_compare(
    tables_t1: list[TableArtifact],
    tables_t2: list[TableArtifact],
    *,
    overlap_threshold: float | None = None,
    bank_code: str | None = None,
    embedding_service: Any | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Recall-first matching engine.

    1. Build views for all eligible tables (certified + review_required).
    2. Precompute full T1 x T2 score matrix (no shortlist, no hard section gate).
    3. Global 1:1 optimal assignment via Hungarian.
    4. Progressive elimination: freeze strong pairs, re-run on residual.
    5. Review candidates for uncertain unmatched tables with plausible partners.
    6. Confirmed added/deleted only when no plausible candidate exists.
    """
    del overlap_threshold, embedding_service

    cfg = get_matching_thresholds(bank_code=bank_code)
    min_match = float(cfg.get("rf_min_match_score", _RF_MIN_MATCH_SCORE))
    min_margin = float(cfg.get("rf_min_match_margin", _RF_MIN_MATCH_MARGIN))
    strong_sc = float(cfg.get("rf_strong_pair_min_score", _RF_STRONG_PAIR_MIN_SCORE))
    strong_mg = float(cfg.get("rf_strong_pair_min_margin", _RF_STRONG_PAIR_MIN_MARGIN))
    review_min = float(
        cfg.get("rf_review_candidate_min_score", _RF_REVIEW_CANDIDATE_MIN_SCORE)
    )
    max_rounds = int(cfg.get("rf_max_elimination_rounds", _RF_MAX_ELIMINATION_ROUNDS))
    min_indicator_signal = float(
        cfg.get("rf_min_indicator_signal", _RF_MIN_INDICATOR_SIGNAL)
    )

    # --- 1. Build views (relaxed eligibility) ---
    sec_freq = _build_section_frequency_relaxed(tables_t1, tables_t2)
    sec_counts: dict[str, int] = {}
    for table in [*tables_t1, *tables_t2]:
        if not is_matching_eligible(table):
            continue
        if not bool(getattr(table, "comparison_eligible", False)):
            continue
        sec = _section_value(table)
        sec_counts[sec] = sec_counts.get(sec, 0) + 1

    t1_views, ineligible_t1_raw = _eligible_table_views_relaxed(
        tables_t1, section_frequencies=sec_freq, section_counts=sec_counts,
    )
    t2_views, ineligible_t2_raw = _eligible_table_views_relaxed(
        tables_t2, section_frequencies=sec_freq, section_counts=sec_counts,
    )

    logger.info(
        "recall_first_compare: t1_views=%d t2_views=%d ineligible_t1=%d ineligible_t2=%d",
        len(t1_views), len(t2_views), len(ineligible_t1_raw), len(ineligible_t2_raw),
    )

    # --- 2. Precompute all scores ---
    all_scores = _precompute_all_scores(t1_views, t2_views)

    # --- 3+4. Progressive elimination ---
    frozen, remaining_t2, remaining_t1, elim_log = _run_progressive_elimination(
        all_scores,
        t2_views,
        t1_views,
        min_match_score=min_match,
        min_match_margin=min_margin,
        strong_score=strong_sc,
        strong_margin=strong_mg,
        max_iterations=max_rounds,
        min_indicator_signal=min_indicator_signal,
    )

    matched_t1 = {t1i for _, t1i, _, _, _ in frozen}
    matched_t2 = {t2i for t2i, _, _, _, _ in frozen}

    # --- 4b. Cross-section rescue ---
    cross_rescued, cross_rescue_log = _run_cross_section_rescue(
        all_scores,
        remaining_t2,
        remaining_t1,
        t2_views,
        t1_views,
        min_score=float(
            cfg.get("rf_cross_section_rescue_min_score", _RF_CROSS_SECTION_RESCUE_MIN_SCORE)
        ),
        min_indicator_signal=min_indicator_signal,
    )
    for t2i, t1i, cs, stage, margin in cross_rescued:
        frozen.append((t2i, t1i, cs, stage, margin))
        remaining_t2.discard(t2i)
        remaining_t1.discard(t1i)
        matched_t1.add(t1i)
        matched_t2.add(t2i)

    # --- 4c. Split-merge rescue ---
    split_merge_rescues, split_merge_used_t2, split_merge_used_t1, split_merge_log = (
        _run_split_merge_rescue(
            all_scores,
            remaining_t2,
            remaining_t1,
            t2_views,
            t1_views,
            min_indicator_signal=min_indicator_signal,
        )
    )
    remaining_t2.difference_update(split_merge_used_t2)
    remaining_t1.difference_update(split_merge_used_t1)
    matched_t2.update(split_merge_used_t2)
    matched_t1.update(split_merge_used_t1)

    # --- 5. Build matched pairs output ---
    pairs: list[dict[str, Any]] = []
    downgraded_t2: set[int] = set()
    downgraded_t1: set[int] = set()

    for t2i, t1i, cs, stage, margin in frozen:
        if _pair_needs_review(cs):
            downgraded_t2.add(t2i)
            downgraded_t1.add(t1i)
            remaining_t2.add(t2i)
            remaining_t1.add(t1i)
            matched_t1.discard(t1i)
            matched_t2.discard(t2i)
            continue

        t1_tbl = t1_views[t1i].table
        t2_tbl = t2_views[t2i].table
        payload = cs.as_feature_dict()
        confidence = cs.total_score
        quality_flags: list[str] = []
        cap_reason: str | None = None

        if cs.quality_suspect:
            confidence = min(0.82, confidence)
            quality_flags.append("low_quality_match")
            cap_reason = "extraction_quality_cap"

        st1 = get_extraction_status(t1_tbl)
        st2 = get_extraction_status(t2_tbl)
        if st1 != EXTRACTION_STATUS_CERTIFIED or st2 != EXTRACTION_STATUS_CERTIFIED:
            confidence = min(0.82, confidence)
            quality_flags.append("includes_review_required")
            cap_reason = cap_reason or "review_required_side"

        payload.update(
            {
                "t1_uid": t1_views[t1i].uid,
                "t2_uid": t2_views[t2i].uid,
                "t1_table_id": t1_tbl.table_id,
                "t2_table_id": t2_tbl.table_id,
                "page_t1": t1_tbl.page_pdf,
                "page_t2": t2_tbl.page_pdf,
                "title_t1": t1_tbl.title or "",
                "title_t2": t2_tbl.title or "",
                "section": t1_tbl.section or t2_tbl.section or "",
                "reason": stage,
                "reason_codes": [stage],
                "decision_level": "match",
                "router_decision": "match",
                "pairing_confidence": round(confidence, 6),
                "match_stage": stage,
                "match_margin": round(margin, 6),
            }
        )
        if quality_flags:
            payload["pairing_quality_flags"] = quality_flags
        if cap_reason:
            payload["pairing_confidence_cap"] = cap_reason
        pairs.append(payload)

    pairs.extend(split_merge_rescues)

    # --- 6. Classify unmatched ---
    ambiguous_pairs: list[dict[str, Any]] = []
    ambiguous_tables: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    unmatched_t2_list: list[dict[str, Any]] = []
    added_tables: list[dict[str, Any]] = []
    unmatched_t1_list: list[dict[str, Any]] = []
    removed_tables: list[dict[str, Any]] = []

    for t2i in sorted(remaining_t2):
        view = t2_views[t2i]
        best: list[CandidateScore] = sorted(
            [
                all_scores[t2i][t1i]
                for t1i in range(len(t1_views))
                if t1i not in matched_t1
                and all_scores[t2i][t1i].total_score >= review_min
                and _has_plausible_signal(all_scores[t2i][t1i], min_indicator_signal)
            ],
            key=lambda c: c.total_score,
            reverse=True,
        )[:3]

        if best:
            cand_uids = [c.t1_view.uid for c in best]
            ambiguous_pairs.append(
                {
                    "decision": "review_candidate",
                    "matched_t1_uid": best[0].t1_view.uid,
                    "confidence": round(best[0].total_score, 6),
                    "reason_codes": ["review_candidate_plausible_match"],
                    "t2_uid": view.uid,
                    "candidate_t1_uids": cand_uids,
                    "candidates": [c.as_feature_dict() for c in best],
                }
            )
            review_candidates.append(
                {
                    "side": "current",
                    "uid": view.uid,
                    "table_id": view.table.table_id,
                    "section": view.table.section or "",
                    "page": view.table.page_pdf,
                    "title": view.table.title or "",
                    "reason": "review_candidate",
                    "top_candidates": [
                        {
                            "uid": c.t1_view.uid,
                            "score": round(c.total_score, 6),
                            "reasons": list(c.explanation),
                        }
                        for c in best
                    ],
                }
            )
            ambiguous_tables.append(
                {
                    "side": "current",
                    "uid": view.uid,
                    "table_id": view.table.table_id,
                    "title": view.table.title or "",
                    "page": view.table.page_pdf,
                    "section": view.table.section or "",
                    "reason": "review_candidate",
                }
            )
            unmatched_t2_list.append(
                {
                    "t2_uid": view.uid,
                    "t2_table_id": view.table.table_id,
                    "section": view.table.section,
                    "page_t2": view.table.page_pdf,
                    "title_t2": view.table.title or "",
                    "reason": "review_candidate",
                    "unmatched_status": "ambiguous",
                    "suspicion_flags": ["review_candidate"],
                }
            )
        else:
            unmatched_t2_list.append(
                {
                    "t2_uid": view.uid,
                    "t2_table_id": view.table.table_id,
                    "section": view.table.section,
                    "page_t2": view.table.page_pdf,
                    "title_t2": view.table.title or "",
                    "reason": "added_table",
                    "unmatched_status": "confirmed",
                    "suspicion_flags": [],
                }
            )
            added_tables.append(_added_table_entry(view))

    for t1i in sorted(remaining_t1):
        view = t1_views[t1i]
        best: list[CandidateScore] = sorted(  # type: ignore[no-redef]
            [
                all_scores[t2i][t1i]
                for t2i in range(len(t2_views))
                if t2i not in matched_t2
                and all_scores[t2i][t1i].total_score >= review_min
                and _has_plausible_signal(all_scores[t2i][t1i], min_indicator_signal)
            ],
            key=lambda c: c.total_score,
            reverse=True,
        )[:3]

        if best:
            review_candidates.append(
                {
                    "side": "previous",
                    "uid": view.uid,
                    "table_id": view.table.table_id,
                    "section": view.table.section or "",
                    "page": view.table.page_pdf,
                    "title": view.table.title or "",
                    "reason": "review_candidate",
                    "top_candidates": [
                        {
                            "uid": c.t2_view.uid,
                            "score": round(c.total_score, 6),
                            "reasons": list(c.explanation),
                        }
                        for c in best
                    ],
                }
            )
            ambiguous_tables.append(
                {
                    "side": "previous",
                    "uid": view.uid,
                    "table_id": view.table.table_id,
                    "title": view.table.title or "",
                    "page": view.table.page_pdf,
                    "section": view.table.section or "",
                    "reason": "review_candidate",
                }
            )
            unmatched_t1_list.append(
                {
                    "t1_uid": view.uid,
                    "t1_table_id": view.table.table_id,
                    "section": view.table.section,
                    "page_t1": view.table.page_pdf,
                    "title_t1": view.table.title or "",
                    "reason": "review_candidate",
                    "unmatched_status": "ambiguous",
                    "suspicion_flags": ["review_candidate"],
                }
            )
        else:
            unmatched_t1_list.append(
                {
                    "t1_uid": view.uid,
                    "t1_table_id": view.table.table_id,
                    "section": view.table.section,
                    "page_t1": view.table.page_pdf,
                    "title_t1": view.table.title or "",
                    "reason": "removed_table",
                    "unmatched_status": "confirmed",
                    "suspicion_flags": [],
                }
            )
            removed_tables.append(_removed_table_entry(view))

    # --- 7. Build backward-compatible output ---
    comparable_t1 = len(t1_views)
    comparable_t2 = len(t2_views)
    pairing_coverage = round(
        len(pairs) / max(min(comparable_t1, comparable_t2), 1), 6
    )

    unmatched_confirmed_t1 = [
        i for i in unmatched_t1_list if i.get("unmatched_status") == "confirmed"
    ]
    unmatched_confirmed_t2 = [
        i for i in unmatched_t2_list if i.get("unmatched_status") == "confirmed"
    ]
    unmatched_ambiguous_t1 = [
        i for i in unmatched_t1_list if i.get("unmatched_status") == "ambiguous"
    ]
    unmatched_ambiguous_t2 = [
        i for i in unmatched_t2_list if i.get("unmatched_status") == "ambiguous"
    ]

    def _inelig_entry(item: dict[str, Any], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            f"{prefix}_table_id": item["table_id"],
            f"{prefix}_uid": item["uid"],
            "section": item["section"],
            f"page_{prefix}": item["page"],
            f"title_{prefix}": item["title"],
            "reason": item["reason"],
            "comparison_blockers": item.get("comparison_blockers", []),
        }
        if "extraction_blockers" in item:
            out["extraction_blockers"] = item["extraction_blockers"]
        if "extraction_status" in item:
            out["extraction_status"] = item["extraction_status"]
        return out

    ineligible_t1 = [_inelig_entry(i, "t1") for i in ineligible_t1_raw]
    ineligible_t2 = [_inelig_entry(i, "t2") for i in ineligible_t2_raw]
    candidate_pairs_scored = len(t1_views) * len(t2_views)
    downgraded_review_count = len(downgraded_t1)

    def _collision_lost_count() -> int:
        lost = 0
        for t2i in remaining_t2:
            if any(
                t1i not in remaining_t1
                and all_scores[t2i][t1i].total_score >= review_min
                and _has_plausible_signal(all_scores[t2i][t1i], min_indicator_signal)
                for t1i in range(len(t1_views))
            ):
                lost += 1
        for t1i in remaining_t1:
            if any(
                t2i not in remaining_t2
                and all_scores[t2i][t1i].total_score >= review_min
                and _has_plausible_signal(all_scores[t2i][t1i], min_indicator_signal)
                for t2i in range(len(t2_views))
            ):
                lost += 1
        return lost

    collision_lost_count = _collision_lost_count()

    logger.info(
        "recall_first_compare done: pairs=%d review=%d added=%d removed=%d "
        "cross_rescued=%d split_merge=%d",
        len(pairs), len(review_candidates), len(added_tables), len(removed_tables),
        len(cross_rescued), len(split_merge_rescues),
    )

    return {
        "pairs": pairs,
        "matched_pairs": list(pairs),
        "probable_pairs": [],
        "suspicious_pairs": [],
        "ambiguous_pairs": ambiguous_pairs,
        "ambiguous_tables": ambiguous_tables,
        "review_candidates": review_candidates,
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "added_tables_pending_review": list(added_tables),
        "removed_tables_pending_review": list(removed_tables),
        "added_tables_confirmed": [],
        "removed_tables_confirmed": [],
        "unmatched_t1": unmatched_t1_list,
        "unmatched_t2": unmatched_t2_list,
        "unmatched_confirmed_t1": unmatched_confirmed_t1,
        "unmatched_confirmed_t2": unmatched_confirmed_t2,
        "unmatched_ambiguous_t1": unmatched_ambiguous_t1,
        "unmatched_ambiguous_t2": unmatched_ambiguous_t2,
        "ambiguous_unmatched_previous": list(unmatched_ambiguous_t1),
        "ambiguous_unmatched_current": list(unmatched_ambiguous_t2),
        "ineligible_t1": ineligible_t1,
        "ineligible_t2": ineligible_t2,
        "debug_unmatched_candidates": [],
        "debug_unmatched_candidates_t2": [],
        "rescued_matches_count": len(cross_rescued),
        "split_merge_rescues_count": len(split_merge_rescues),
        "cross_section_rescued_pairs": [
            {
                "t1_uid": t1_views[t1i].uid,
                "t2_uid": t2_views[t2i].uid,
                "score": round(cs.total_score, 6),
                "source": "cross_section_rescue",
            }
            for t2i, t1i, cs, _, _ in cross_rescued
        ],
        "split_merge_rescue_candidates": split_merge_rescues,
        "vision_rescued_pairs": [],
        "reasons": [p.get("reason", "") for p in pairs if p.get("reason")],
        "diagnostics": {
            "engine": "recall_first",
            "elimination_rounds": len(elim_log),
            "cross_section_rescue_log": cross_rescue_log,
            "split_merge_rescue_log": split_merge_log,
        },
        "matching_diagnostics": {
            "engine": "recall_first",
            "candidate_search_strategy": "full_matrix_no_shortlist",
            "candidate_pairs_scored": candidate_pairs_scored,
            "shortlist_pruned_equivalent": 0,
            "pairs_count": len(pairs),
            "review_candidate_count": len(review_candidates),
            "ambiguous_pairs_count": len(ambiguous_pairs),
            "unmatched_t1_count": len(unmatched_t1_list),
            "unmatched_t2_count": len(unmatched_t2_list),
            "ineligible_t1_count": len(ineligible_t1),
            "ineligible_t2_count": len(ineligible_t2),
            "tables_comparable_t1": comparable_t1,
            "tables_comparable_t2": comparable_t2,
            "pairing_coverage": pairing_coverage,
            "elimination_log": elim_log,
            "rescue_summary": {
                "cross_section_rescued": len(cross_rescued),
                "split_merge_candidates": len(split_merge_rescues),
                "cross_section_rescue_log": cross_rescue_log,
                "split_merge_rescue_log": split_merge_log,
            },
            "recall_loss_tracking": {
                "tables_blocked_t1": sum(
                    1 for i in ineligible_t1_raw if i.get("reason") == "extraction_not_certified"
                ),
                "tables_blocked_t2": sum(
                    1 for i in ineligible_t2_raw if i.get("reason") == "extraction_not_certified"
                ),
                "tables_ineligible_t1": sum(
                    1 for i in ineligible_t1_raw if i.get("reason") == "comparison_ineligible"
                ),
                "tables_ineligible_t2": sum(
                    1 for i in ineligible_t2_raw if i.get("reason") == "comparison_ineligible"
                ),
                "review_required_included_t1": sum(
                    1
                    for v in t1_views
                    if get_extraction_status(v.table) == EXTRACTION_STATUS_REVIEW_REQUIRED
                ),
                "review_required_included_t2": sum(
                    1
                    for v in t2_views
                    if get_extraction_status(v.table) == EXTRACTION_STATUS_REVIEW_REQUIRED
                ),
                "strong_matches": sum(
                    1 for _, _, _, s, _ in frozen if s == "strong_match"
                ),
                "residual_matches": sum(
                    1 for _, _, _, s, _ in frozen if s == "residual_match"
                ),
                "cross_section_rescued": len(cross_rescued),
                "split_merge_candidates": len(split_merge_rescues),
                "section_mismatch": sum(
                    1
                    for t2i in remaining_t2
                    for t1i in remaining_t1
                    if all_scores[t2i][t1i].section_compatibility == 0
                    and all_scores[t2i][t1i].total_score >= review_min
                ),
                "no_plausible_candidate": len(added_tables) + len(removed_tables),
                "downgraded_to_review_candidate": downgraded_review_count,
                "collision_lost": collision_lost_count,
                "review_candidates_t2": sum(
                    1 for r in review_candidates if r.get("side") == "current"
                ),
                "review_candidates_t1": sum(
                    1 for r in review_candidates if r.get("side") == "previous"
                ),
                "confirmed_added": 0,
                "confirmed_removed": 0,
                "pending_added_candidates": len(added_tables),
                "pending_removed_candidates": len(removed_tables),
            },
        },
        "tables_comparable_t1": comparable_t1,
        "tables_comparable_t2": comparable_t2,
        "pairing_coverage": pairing_coverage,
    }
