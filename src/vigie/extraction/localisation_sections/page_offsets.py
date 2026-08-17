"""Résolution de la pagination imprimée vers les pages physiques d'un PDF.

La configuration bancaire reste un repli. Dès que plusieurs titres de la TDM
peuvent être retrouvés sur les pages physiques, leur consensus détermine
l'offset propre au document courant.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .models import TocEntry, normalize_text


@dataclass(slots=True)
class PageOffsetResolution:
    """Résultat auditable de la résolution d'offset."""

    offset: int
    configured_offset: int
    confidence: float
    status: str
    votes: dict[int, int]
    anchors: list[dict[str, object]]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Sérialiser le résultat pour les diagnostics de localisation."""
        data = asdict(self)
        data["votes"] = {str(key): value for key, value in sorted(self.votes.items())}
        return data


def _compact(text: str) -> str:
    """Normaliser fortement un titre pour comparer les extractions PDF."""
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def _top_lines(page_text: str, limit: int = 35) -> list[str]:
    """Retourner les lignes non vides de la zone de titre d'une page."""
    return [line.strip() for line in str(page_text or "").splitlines() if line.strip()][:limit]


def _title_match_score(title: str, line: str, line_index: int) -> float:
    """Noter une correspondance stricte de titre sur une page physique."""
    title_norm = normalize_text(title).strip()
    line_norm = normalize_text(line).strip()
    if not title_norm or not line_norm:
        return 0.0

    title_compact = _compact(title_norm)
    line_compact = _compact(line_norm)
    if len(title_compact) < 8:
        return 0.0

    score = 0.0
    if line_norm == title_norm or line_compact == title_compact:
        score = 1.0
    elif len(title_compact) >= 14 and line_compact.startswith(title_compact):
        extra = max(0, len(line_compact) - len(title_compact))
        if extra <= 24:
            score = 0.88
    elif len(line_compact) >= 14 and title_compact.startswith(line_compact):
        missing = max(0, len(title_compact) - len(line_compact))
        if missing <= 18:
            score = 0.82

    if score and line_index <= 5:
        score += 0.08
    return min(score, 1.0)


def _candidate_offsets(configured_offset: int, max_offset: int) -> list[int]:
    """Ordonner les offsets à tester, proches de la configuration en premier."""
    values = list(range(0, max_offset + 1))
    return sorted(values, key=lambda value: (abs(value - configured_offset), value))


def infer_page_offset(
    text_by_page: dict[int, str],
    toc_entries: Iterable[TocEntry],
    *,
    configured_offset: int = 0,
    max_offset: int = 20,
    min_consensus_anchors: int = 2,
) -> PageOffsetResolution:
    """Déduire l'offset imprimé→physique à partir de plusieurs titres de TDM.

    Un vote est produit au plus une fois par entrée. Les titres courts ou les
    lignes qui ne sont pas retrouvées près de la page annoncée sont ignorés.
    En cas de conflit non résolu, l'offset configuré est conservé et le statut
    devient ``ambiguous``.
    """
    configured = max(0, int(configured_offset or 0))
    total_pages = max(text_by_page.keys(), default=0)
    offsets = _candidate_offsets(configured, max_offset)
    votes: dict[int, int] = {}
    anchors: list[dict[str, object]] = []

    for entry in toc_entries:
        title = str(entry.title or "").strip()
        if entry.page < 3 or len(_compact(title)) < 8:
            continue

        matches: list[tuple[float, int, int, str]] = []
        for offset in offsets:
            physical_page = int(entry.page) + offset
            if physical_page < 1 or physical_page > total_pages:
                continue
            for line_index, line in enumerate(_top_lines(text_by_page.get(physical_page, "")), start=1):
                score = _title_match_score(title, line, line_index)
                if score:
                    matches.append((score, offset, line_index, line))
                    break

        if not matches:
            continue

        # Un en-tête répété peut matcher plusieurs pages. La proximité de la
        # configuration départage seulement après la force visuelle du titre.
        score, offset, line_index, observed = max(
            matches,
            key=lambda item: (item[0], -abs(item[1] - configured), -item[2], -item[1]),
        )
        votes[offset] = votes.get(offset, 0) + 1
        anchors.append(
            {
                "title": title,
                "printed_page": int(entry.page),
                "physical_page": int(entry.page) + offset,
                "offset": offset,
                "match_score": round(score, 3),
                "observed_title": observed,
            }
        )

    if not votes:
        return PageOffsetResolution(
            offset=configured,
            configured_offset=configured,
            confidence=0.0,
            status="configured_no_anchor",
            votes={},
            anchors=[],
            warnings=["page_offset_no_physical_anchor"],
        )

    ranked = sorted(votes.items(), key=lambda item: (-item[1], abs(item[0] - configured), item[0]))
    best_offset, best_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    total_votes = sum(votes.values())
    ratio = best_count / total_votes
    tied = second_count == best_count
    # Une ancre unique peut confirmer la configuration, jamais la remplacer.
    # Cela protège notamment les TDM clairsemées et les en-têtes répétés.
    enough = best_count >= min_consensus_anchors or (best_count == 1 and total_votes == 1 and best_offset == configured)

    if tied or ratio < 0.6 or not enough:
        return PageOffsetResolution(
            offset=configured,
            configured_offset=configured,
            confidence=round(ratio, 3),
            status="ambiguous",
            votes=votes,
            anchors=anchors,
            warnings=["page_offset_conflicting_anchors"],
        )

    return PageOffsetResolution(
        offset=best_offset,
        configured_offset=configured,
        confidence=round(ratio, 3),
        status="confirmed" if best_offset == configured else "inferred_override",
        votes=votes,
        anchors=anchors,
        warnings=[],
    )
