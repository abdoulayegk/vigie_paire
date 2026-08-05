"""Resolution des bornes T4 a partir d'une TDM structurelle.

Mappe les entrees TDM vers gestion_capital / gestion_risques, calcule
start/end selon l'ordre TDM, et retombe sur les titres hardcodes en prior
explicite si la TDM est incomplete. Vision reste un arbitrage optionnel.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vigie.extraction.localisation_sections.models import LocatedSection, normalize_text
from vigie.extraction.localisation_sections.patterns import (
    SECTION_TITLE_ALIASES,
    T4_SECTION_TITLE_PROFILES,
)
from vigie.extraction.localisation_sections.toc_locator import TocStructure, TocStructureEntry

logger = logging.getLogger(__name__)

_TARGET_KEYS = ("gestion_capital", "gestion_risques")
_AMBIGUITY_PAGE_GAP = 3
_MIN_TOC_CONFIDENCE = 0.55


def _compact_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


@dataclass(slots=True)
class BoundaryResolveResult:
    """Resultat de resolution des bornes T4."""

    sections: list[LocatedSection] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    used_toc: bool = False
    confidence: float = 0.0


def _prototype_titles(bank_code: str, section_key: str) -> list[str]:
    """Assembler les prototypes de titres pour un concept (profil T4 + alias)."""
    titles: list[str] = []
    bank = str(bank_code or "").strip().lower()
    profile = T4_SECTION_TITLE_PROFILES.get(bank, {}).get(section_key, {})
    titles.extend(profile.get("start", []))
    titles.extend(SECTION_TITLE_ALIASES.get(section_key, []))
    # Completer avec les starts des autres banques pour robustesse inter-banque.
    for other_bank, sections in T4_SECTION_TITLE_PROFILES.items():
        if other_bank == bank:
            continue
        for title in sections.get(section_key, {}).get("start", []):
            # « Situation financière du Groupe » est un ancrage BNS, pas un
            # synonyme capital pour TD/RBC (leurre TDM en début de rapport).
            if normalize_text(title).strip() == "situation financiere du groupe" and bank != "bns":
                continue
            titles.append(title)
    deduped: list[str] = []
    seen: set[str] = set()
    for title in titles:
        key = normalize_text(title).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(title)
    return deduped


def map_toc_title_to_concept(title: str, bank_code: str = "") -> str | None:
    """Mapper un titre TDM vers gestion_capital / gestion_risques / None."""
    title_norm = normalize_text(title).strip()
    if not title_norm or len(title_norm) < 6:
        return None
    title_compact = re.sub(r"[^a-z0-9]", "", title_norm)
    bank = str(bank_code or "").strip().lower()
    # Eviter les sous-sections risques (risque de credit, etc.).
    if re.match(r"^risque\s+(de|du|des|lie|lié)\b", title_norm):
        return None
    if title_norm.startswith("divulgation"):
        return None
    # « Situation financière » / « du groupe » hors BNS n'est pas le chapitre capital.
    if "fonds" not in title_norm and "capital" not in title_norm:
        if title_norm in {"situation financiere", "situation financiere du groupe"}:
            if bank != "bns":
                return None

    best_key: str | None = None
    best_score = 0.0
    for section_key in _TARGET_KEYS:
        for proto in _prototype_titles(bank_code, section_key):
            proto_norm = normalize_text(proto).strip()
            if not proto_norm:
                continue
            proto_compact = re.sub(r"[^a-z0-9]", "", proto_norm)
            score = 0.0
            if title_norm == proto_norm or title_compact == proto_compact:
                score = 100.0
            elif title_norm.startswith(proto_norm):
                # Sous-titre trop specifique (ex. Cadre de gestion des fonds propres).
                score = 40.0 if len(title_norm) > len(proto_norm) + 8 else 85.0
            elif proto_norm.startswith(title_norm) and len(title_norm) >= 14:
                # Titre TDM tronque mais assez long pour etre discriminatif.
                # Ex: « FACTEURS DE RISQUE ET GESTION » vs titre complet TD.
                score = 82.0 if len(title_norm) >= 18 else 70.0
            elif title_norm in proto_norm and len(title_norm) >= 14:
                # Ex: « des fonds propres » suffixe de « gestion des fonds propres ».
                score = 88.0
            elif title_compact and proto_compact and title_compact == proto_compact:
                score = 90.0
            elif title_compact and proto_compact and len(title_compact) >= 14 and title_compact in proto_compact:
                score = 86.0
            elif (
                title_compact
                and proto_compact
                and len(proto_compact) >= 18
                and proto_compact in title_compact
            ):
                if title_compact.endswith(proto_compact):
                    prefix_len = len(title_compact) - len(proto_compact)
                    # Prefixe court (« cadre de... ») = sous-section; long = mash colonnes.
                    score = 30.0 if prefix_len <= 12 else 84.0
                else:
                    score = 86.0
            elif proto_norm in title_norm and len(proto_norm) >= 18:
                if title_norm.endswith(proto_norm):
                    prefix = title_norm[: -len(proto_norm)].strip()
                    score = 30.0 if len(prefix) <= 12 else 84.0
                else:
                    score = 35.0
            if score > best_score:
                best_score = score
                best_key = section_key
    # TD: le chapitre risques s'intitule « Facteurs de risque et gestion des risques ».
    # « Gestion des risques » seul est une sous-section plus bas dans la TDM.
    if (
        best_key == "gestion_risques"
        and bank == "td"
        and "facteur" not in title_norm
        and title_compact in {"gestiondesrisques", "gestiondurisque"}
    ):
        return None
    if best_score >= 80.0:
        return best_key
    return None


def _physical_page(entry: TocStructureEntry) -> int | None:
    if entry.physical_page is not None and entry.physical_page > 0:
        return int(entry.physical_page)
    if entry.printed_page > 0:
        return int(entry.printed_page)
    return None


def _ordered_concept_hits(
    toc: TocStructure,
    bank_code: str,
) -> list[tuple[str, TocStructureEntry, int]]:
    """Retourner les hits conceptuels tries par page physique croissante."""
    candidates: dict[str, list[tuple[TocStructureEntry, int]]] = {key: [] for key in _TARGET_KEYS}
    for entry in toc.entries:
        concept = map_toc_title_to_concept(entry.title, bank_code)
        if concept is None:
            continue
        page = _physical_page(entry)
        if page is None:
            continue
        candidates[concept].append((entry, page))

    hits: list[tuple[str, TocStructureEntry, int]] = []
    for concept, items in candidates.items():
        if not items:
            continue
        bank_starts = [
            normalize_text(t).strip()
            for t in T4_SECTION_TITLE_PROFILES.get(str(bank_code or "").strip().lower(), {})
            .get(concept, {})
            .get("start", [])
        ]
        # Preferer les hits « corps de rapport » si plusieurs ancres TDM existent.
        body = [item for item in items if item[1] >= 45]
        if body:
            pool = body
        else:
            # Hits precoces uniquement: ignorer les titres tronques (prefixe du start
            # banque) pour laisser le prior titres physiques prendre le relais.
            truncated_only = True
            for entry, _page in items:
                title_norm = normalize_text(entry.title).strip()
                if not bank_starts:
                    truncated_only = False
                    break
                if any(
                    title_norm == start
                    or _compact_norm(title_norm) == _compact_norm(start)
                    or (start in title_norm and len(title_norm) >= len(start))
                    for start in bank_starts
                ):
                    truncated_only = False
                    break
            if truncated_only:
                continue
            pool = items

        def _rank(item: tuple[TocStructureEntry, int]) -> tuple[int, int, int]:
            entry, page = item
            title_norm = normalize_text(entry.title)
            specificity = 0
            for start in bank_starts:
                if not start:
                    continue
                if title_norm == start or _compact_norm(title_norm) == _compact_norm(start):
                    specificity = 3
                    break
                if start.startswith(title_norm) and len(title_norm) >= 18:
                    specificity = max(specificity, 2)
                elif title_norm.endswith(start) or start in title_norm:
                    specificity = max(specificity, 2)
            cleanliness = 1 if len(title_norm) <= 70 else 0
            return (specificity, page, cleanliness)

        entry, page = max(pool, key=_rank)
        hits.append((concept, entry, page))
    hits.sort(key=lambda item: item[2])
    return hits


def _bounds_from_hits(
    hits: list[tuple[str, TocStructureEntry, int]],
    all_entries: list[TocStructureEntry],
    bank_code: str = "",
) -> dict[str, LocatedSection]:
    """Calculer start/end pour chaque concept hit selon le successeur TDM."""
    sections: dict[str, LocatedSection] = {}
    for index, (concept, entry, start_page) in enumerate(hits):
        end_page: int | None = None
        end_title = ""
        # Preferer le prochain concept cible (ignore les sous-sections TDM).
        if index + 1 < len(hits):
            end_page = hits[index + 1][2] - 1
            end_title = hits[index + 1][1].title
        else:
            for other in all_entries:
                page = _physical_page(other)
                if page is None or page <= start_page:
                    continue
                other_norm = normalize_text(other.title)
                if re.match(r"^risque\s+", other_norm):
                    continue
                other_concept = map_toc_title_to_concept(other.title, bank_code)
                if other_concept == concept:
                    continue
                if concept == "gestion_risques" and "gestion des risques" in other_norm:
                    continue
                end_page = page - 1
                end_title = other.title
                break
        sections[concept] = LocatedSection(
            section_type=concept,
            title_found=entry.title,
            start_page=start_page,
            end_page=end_page,
            confidence=0.9,
            detection_method="annual_t4_toc_structure",
            end_detection_method="annual_t4_toc_successor" if end_page is not None else "annual_t4_unresolved_no_successor",
            anchor_page=start_page,
            anchor_text=entry.title,
            end_anchor_page=(end_page + 1) if end_page is not None else None,
            end_anchor_text=end_title or None,
        )
    return sections


def resolve_t4_section_bounds(
    toc: TocStructure,
    *,
    bank_code: str = "",
    text_by_page: dict[int, str] | None = None,
    total_pages: int = 0,
    find_start: Callable[[str], tuple[int, str] | None] | None = None,
    find_end: Callable[[str, int], tuple[int | None, str]] | None = None,
    pdf_path: str | Path | None = None,
    vision_arbitrate: Callable[..., Any] | None = None,
) -> BoundaryResolveResult:
    """Resoudre les bornes capital/risques depuis la TDM, avec repli titres."""
    anomalies = list(toc.anomalies)
    result = BoundaryResolveResult(anomalies=anomalies, confidence=toc.confidence)

    hits = _ordered_concept_hits(toc, bank_code)
    toc_sections = _bounds_from_hits(hits, toc.entries, bank_code) if hits else {}

    used_toc = bool(toc_sections) and toc.confidence >= _MIN_TOC_CONFIDENCE and toc.rg_page is not None
    if used_toc:
        result.used_toc = True
        result.sections = list(toc_sections.values())
        result.confidence = toc.confidence
    else:
        anomalies.append("toc_insufficient_falling_back_to_titles")
        result.used_toc = False

    # Completer les concepts manquants via prior titres hardcodes.
    for section_key in _TARGET_KEYS:
        if any(section.section_type == section_key for section in result.sections):
            continue
        if find_start is None:
            anomalies.append(f"boundary_unresolved:{section_key}:no_start_fallback")
            continue
        root = find_start(section_key)
        if root is None:
            anomalies.append(f"boundary_unresolved:{section_key}:start_not_found")
            continue
        start_page, title_found = root
        end_page: int | None = None
        end_method = "annual_t4_unresolved_no_successor"
        if find_end is not None:
            end_page, end_method = find_end(section_key, start_page)
        result.sections.append(
            LocatedSection(
                section_type=section_key,
                title_found=title_found,
                start_page=start_page,
                end_page=end_page,
                confidence=0.7,
                detection_method="annual_t4_toc_fallback_title",
                end_detection_method=end_method,
                anchor_page=start_page,
                anchor_text=title_found,
            )
        )
        anomalies.append(f"fallback_title_used:{section_key}")

    # Arbitrage Vision si deux starts trop proches.
    starts = sorted(
        ((s.section_type, s.start_page, s) for s in result.sections),
        key=lambda item: item[1],
    )
    for idx in range(len(starts) - 1):
        left_key, left_page, left_sec = starts[idx]
        right_key, right_page, right_sec = starts[idx + 1]
        if abs(right_page - left_page) <= _AMBIGUITY_PAGE_GAP and vision_arbitrate is not None:
            try:
                vision_arbitrate(
                    pdf_path=pdf_path,
                    left=left_sec,
                    right=right_sec,
                    text_by_page=text_by_page,
                )
                anomalies.append(f"vision_arbitration_invoked:{left_key}/{right_key}")
            except Exception as exc:
                anomalies.append(f"vision_arbitration_failed:{exc}")

    # Borner / reconcilier end par la prochaine section cible connue.
    ordered = sorted(result.sections, key=lambda s: s.start_page)
    for i, section in enumerate(ordered):
        if i + 1 < len(ordered):
            next_start = ordered[i + 1].start_page
            if section.end_page is None or section.end_page >= next_start:
                section.end_page = next_start - 1
                section.end_detection_method = "annual_t4_next_target_section"
        elif section.end_page is None:
            if total_pages > 0:
                anomalies.append(f"boundary_unresolved:{section.section_type}:no_successor")
            section.end_detection_method = "annual_t4_unresolved_no_successor"
            # Pas de cap silencieux start+119.

        # Serrer via titres physiques si un successeur reel existe avant la
        # prochaine cible (ex. TD: capital 75-80 puis pages intercalaires).
        if find_end is not None and section.detection_method.startswith("annual_t4_toc"):
            try:
                alt_end, alt_method = find_end(section.section_type, section.start_page)
            except Exception:
                alt_end, alt_method = None, ""
            if alt_end is not None:
                next_limit = ordered[i + 1].start_page - 1 if i + 1 < len(ordered) else alt_end
                if alt_end <= next_limit and (section.end_page is None or alt_end < section.end_page):
                    section.end_page = alt_end
                    section.end_detection_method = alt_method

    result.anomalies = anomalies
    result.sections = sorted(result.sections, key=lambda s: s.start_page)
    logger.info(
        "BoundaryResolver: used_toc=%s sections=%s anomalies=%s",
        result.used_toc,
        [(s.section_type, s.start_page, s.end_page, s.detection_method) for s in result.sections],
        anomalies[:8],
    )
    return result
