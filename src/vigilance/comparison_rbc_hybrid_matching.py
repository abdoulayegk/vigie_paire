"""Recuperation hybride, isolee et opt-in, des appariements de tableaux RBC.

Les embeddings servent uniquement a reduire le nombre de candidats presentes au
LLM. Ils ne constituent jamais une preuve de correspondance. La decision finale
combine une evaluation semantique, des signaux objectifs portant sur l'ensemble
du tableau, une affectation globale 1-a-1 et une seconde inspection LLM.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.matching_normalizer import normalize_for_matching, strip_temporal_expressions

logger = logging.getLogger(__name__)

_PREVIOUS_PREFIX = "PQ::"
_CURRENT_PREFIX = "CQ::"


class _CandidateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_table_id: str
    verdict: Literal["same_table", "different_table", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class _CandidateAssessmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[_CandidateAssessment]


class _FinalInspectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["confirmed", "rejected", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@dataclass(frozen=True)
class _Edge:
    previous_table_id: str
    current_table_id: str
    confidence: float
    score: float
    reason: str
    facts: dict[str, Any]


_CANDIDATE_SYSTEM_PROMPT = """
You assess candidate matches between one RBC Current Quarter table and several
Previous Quarter tables. Assess every candidate independently.

Embeddings selected the candidates but are NOT evidence that two tables match.
The title alone is insufficient. A first row such as Actif or Passif alone is
also insufficient. Use the complete indicator lists, business purpose, headers,
footnotes, row structure and continuation context. A table can move by several
pages. Do not force a match. Return `same_table` only when the two complete
tables represent the same business entity across reporting periods.

Identifiers are namespaced: CQ identifiers start with CQ:: and PQ identifiers
start with PQ::. Return only PQ identifiers supplied in the candidate list.
"""


_FINAL_INSPECTION_SYSTEM_PROMPT = """
You are the final RBC table-match inspector. Decide whether this single PQ/CQ
pair is the same complete business table across reporting periods.

The pair survived candidate retrieval and a first semantic review, but neither
fact proves it is correct. Re-evaluate the full title, summary, ALL indicators,
headers, footnotes, row structure and continuation context. A shared generic
title or only the first indicator is insufficient. Reject or mark ambiguous if
the evidence does not establish the same table. Return JSON only.
"""


def _alias_previous(table_id: str) -> str:
    return f"{_PREVIOUS_PREFIX}{table_id}"


def _alias_current(table_id: str) -> str:
    return f"{_CURRENT_PREFIX}{table_id}"


def _strip_alias(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    return text[len(prefix) :] if text.startswith(prefix) else ""


def _normalized_title(card: dict[str, Any]) -> str:
    value = strip_temporal_expressions(str(card.get("title", "") or ""), target="title", aggressive=True)
    value = re.sub(r"\btableau\s+\d+\b", " ", value, flags=re.IGNORECASE)
    return normalize_for_matching(value, target="title")


def _normalized_summary(card: dict[str, Any]) -> str:
    value = strip_temporal_expressions(str(card.get("table_summary", "") or ""), target="title", aggressive=True)
    return normalize_for_matching(value, target="title")


def _normalized_indicators(card: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for raw in list(card.get("indicators", []) or []):
        value = normalize_indicator_for_comparison(str(raw or ""))
        if value:
            out.append(value)
    return out


def _normalized_headers(card: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for raw in list(card.get("headers", []) or []):
        value = strip_temporal_expressions(str(raw or ""), target="header", aggressive=True)
        value = normalize_for_matching(value, target="header")
        if value:
            out.add(value)
    return out


def _footnote_texts(card: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in list(card.get("footnotes", []) or []):
        raw = item.get("text", "") if isinstance(item, dict) else item
        value = normalize_for_matching(str(raw or ""), target="generic")
        if value:
            out.append(value)
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _pair_facts(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_indicators = _normalized_indicators(previous)
    current_indicators = _normalized_indicators(current)
    previous_set = set(previous_indicators)
    current_set = set(current_indicators)
    common = previous_set & current_set
    smaller_count = min(len(previous_set), len(current_set))

    previous_rows = max(0, int(previous.get("row_count", len(previous_indicators)) or 0))
    current_rows = max(0, int(current.get("row_count", len(current_indicators)) or 0))
    max_rows = max(previous_rows, current_rows, 1)
    min_rows = min(previous_rows, current_rows)

    previous_title = _normalized_title(previous)
    current_title = _normalized_title(current)
    previous_summary = _normalized_summary(previous)
    current_summary = _normalized_summary(current)
    previous_notes = set(_footnote_texts(previous))
    current_notes = set(_footnote_texts(current))

    return {
        "title_exact": bool(previous_title and previous_title == current_title),
        "summary_exact": bool(previous_summary and previous_summary == current_summary),
        "indicator_common_count": len(common),
        "indicator_smaller_coverage": len(common) / smaller_count if smaller_count else 0.0,
        "indicator_jaccard": _jaccard(previous_set, current_set),
        "header_jaccard": _jaccard(_normalized_headers(previous), _normalized_headers(current)),
        "footnote_jaccard": _jaccard(previous_notes, current_notes),
        "row_ratio": min_rows / max_rows,
        "row_delta": abs(previous_rows - current_rows),
        "previous_indicator_count": len(previous_indicators),
        "current_indicator_count": len(current_indicators),
    }


def _objective_evidence_gate(facts: dict[str, Any]) -> bool:
    """Exige une preuve portant sur plus qu'un titre ou une premiere ligne."""
    common = int(facts["indicator_common_count"])
    coverage = float(facts["indicator_smaller_coverage"])
    row_ratio = float(facts["row_ratio"])
    title_exact = bool(facts["title_exact"])
    summary_exact = bool(facts["summary_exact"])
    footnote_overlap = float(facts["footnote_jaccard"])

    strong_indicator_signature = common >= 2 and coverage >= 0.50
    small_table_signature = common == 1 and coverage == 1.0 and row_ratio >= 0.75 and (title_exact or summary_exact)
    corroborated_footnotes = (
        footnote_overlap >= 0.50 and row_ratio >= 0.60 and (title_exact or summary_exact or common >= 1)
    )
    corroborated_extraction_expansion = (
        common >= 1 and coverage == 1.0 and title_exact and summary_exact and float(facts["header_jaccard"]) >= 0.80
    )
    return (
        strong_indicator_signature
        or small_table_signature
        or corroborated_footnotes
        or corroborated_extraction_expansion
    )


def _evidence_strength(facts: dict[str, Any]) -> float:
    return min(
        1.0,
        0.50 * float(facts["indicator_smaller_coverage"])
        + 0.15 * float(facts["indicator_jaccard"])
        + 0.10 * float(facts["header_jaccard"])
        + 0.10 * float(facts["footnote_jaccard"])
        + 0.075 * float(bool(facts["title_exact"]))
        + 0.075 * float(bool(facts["summary_exact"])),
    )


def _primary_pair_trust_gate(facts: dict[str, Any]) -> bool:
    """Seuil plus strict pour soustraire une paire au reexamen global RBC."""
    if not _objective_evidence_gate(facts):
        return False
    common = int(facts["indicator_common_count"])
    coverage = float(facts["indicator_smaller_coverage"])
    indicator_jaccard = float(facts["indicator_jaccard"])
    header_jaccard = float(facts["header_jaccard"])
    row_ratio = float(facts["row_ratio"])
    title_exact = bool(facts["title_exact"])
    summary_exact = bool(facts["summary_exact"])

    corroborated_identity = summary_exact and common >= 1
    title_structure_identity = title_exact and coverage >= 0.70 and header_jaccard >= 0.60
    near_complete_structure = indicator_jaccard >= 0.80 and row_ratio >= 0.75
    return corroborated_identity or title_structure_identity or near_complete_structure


def partition_trusted_rbc_primary_pairs(
    pairs: list[dict[str, Any]],
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verrouille seulement les paires primaires RBC ayant une preuve objective."""
    previous_lookup = {str(card.get("table_id", "")): card for card in previous_cards}
    current_lookup = {str(card.get("table_id", "")): card for card in current_cards}
    trusted: list[dict[str, Any]] = []
    released: list[dict[str, Any]] = []
    for pair in pairs:
        previous = previous_lookup.get(str(pair.get("previous_table_id", "")))
        current = current_lookup.get(str(pair.get("current_table_id", "")))
        if previous is not None and current is not None and _primary_pair_trust_gate(_pair_facts(previous, current)):
            trusted.append(pair)
        else:
            released.append(pair)
    return trusted, released


def _embedding_views(card: dict[str, Any]) -> dict[str, str]:
    indicators = [str(item or "").strip() for item in list(card.get("indicators", []) or []) if str(item).strip()]
    headers = [str(item or "").strip() for item in list(card.get("headers", []) or []) if str(item).strip()]
    notes = _footnote_texts(card)
    return {
        "concept": "\n".join(
            [
                f"Titre: {card.get('title', '')}",
                f"Objet: {card.get('table_summary', '')}",
                f"Section: {card.get('section', '')}",
            ]
        ),
        "indicators": "\n".join(f"{index + 1}. {value}" for index, value in enumerate(indicators)),
        "structure": f"Lignes: {card.get('row_count', len(indicators))}\nColonnes: {' | '.join(headers)}",
        "footnotes": "\n".join(notes) or "Aucune note de bas de tableau",
    }


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _embed_cards(
    cards: list[dict[str, Any]],
    *,
    model: str,
    call_openai_embeddings: Callable[..., list[list[float]]],
    usage_recorder: list[dict[str, Any]] | None,
) -> dict[str, dict[str, list[float]]]:
    view_names = ("concept", "indicators", "structure", "footnotes")
    texts: list[str] = []
    positions: list[tuple[str, str]] = []
    for card in cards:
        table_id = str(card.get("table_id", ""))
        views = _embedding_views(card)
        for view_name in view_names:
            positions.append((table_id, view_name))
            texts.append(views[view_name])
    vectors = call_openai_embeddings(
        model=model,
        inputs=texts,
        usage_recorder=usage_recorder,
        call_kind="rbc_hybrid_embeddings",
    )
    if len(vectors) != len(texts):
        raise ValueError(f"Embedding count mismatch: expected {len(texts)}, received {len(vectors)}")
    result: dict[str, dict[str, list[float]]] = {}
    for (table_id, view_name), vector in zip(positions, vectors):
        result.setdefault(table_id, {})[view_name] = [float(value) for value in vector]
    return result


def _candidate_shortlist(
    current: dict[str, Any],
    previous_cards: list[dict[str, Any]],
    embeddings: dict[str, dict[str, list[float]]],
    *,
    top_k: int,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    weights = {"concept": 0.35, "indicators": 0.40, "structure": 0.15, "footnotes": 0.10}
    current_id = str(current.get("table_id", ""))
    ranked: list[tuple[float, float, str, dict[str, Any], dict[str, Any]]] = []
    for previous in previous_cards:
        previous_id = str(previous.get("table_id", ""))
        similarity = sum(
            weight * _cosine(embeddings[current_id][view], embeddings[previous_id][view])
            for view, weight in weights.items()
        )
        facts = _pair_facts(previous, current)
        ranked.append((similarity, _evidence_strength(facts), previous_id, previous, facts))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = ranked[: max(1, min(top_k, len(ranked)))]

    # A strong objective anchor is always shown to the LLM, even if embedding
    # retrieval placed it just outside Top-K.
    selected_ids = {item[2] for item in selected}
    for item in sorted(ranked, key=lambda value: (-value[1], -value[0], value[2])):
        if item[2] not in selected_ids and _objective_evidence_gate(item[4]):
            selected.append(item)
            selected_ids.add(item[2])
    return [(item[3], item[4], item[0]) for item in selected]


def _judge_candidates(
    current: dict[str, Any],
    shortlist: list[tuple[dict[str, Any], dict[str, Any], float]],
    *,
    model: str,
    min_confidence: float,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None,
) -> list[_Edge]:
    current_id = str(current.get("table_id", ""))
    payload = {
        "current_table": {**current, "table_id": _alias_current(current_id)},
        "candidates": [
            {
                "previous_table": {**previous, "table_id": _alias_previous(str(previous.get("table_id", "")))},
                "objective_facts": facts,
            }
            for previous, facts, _similarity in shortlist
        ],
    }
    response = call_openai_json(
        model=model,
        messages=[
            {"role": "system", "content": _CANDIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        usage_recorder=usage_recorder,
        call_kind="rbc_hybrid_judge",
        response_model=_CandidateAssessmentResponse,
    )
    allowed = {str(previous.get("table_id", "")): (facts, similarity) for previous, facts, similarity in shortlist}
    edges: list[_Edge] = []
    for assessment in list(response.get("assessments", []) or []):
        previous_id = _strip_alias(assessment.get("previous_table_id"), _PREVIOUS_PREFIX)
        if previous_id not in allowed or assessment.get("verdict") != "same_table":
            continue
        confidence = float(assessment.get("confidence", 0.0) or 0.0)
        facts, _similarity = allowed[previous_id]
        if confidence < min_confidence or not _objective_evidence_gate(facts):
            continue
        evidence = _evidence_strength(facts)
        edges.append(
            _Edge(
                previous_table_id=previous_id,
                current_table_id=current_id,
                confidence=confidence,
                score=0.80 * confidence + 0.20 * evidence,
                reason=str(assessment.get("reason", "") or "").strip(),
                facts=facts,
            )
        )
    return edges


def _assign_edges(
    edges: list[_Edge],
    previous_ids: list[str],
    current_ids: list[str],
    excluded: set[tuple[str, str]],
) -> list[_Edge]:
    edge_lookup = {
        (edge.current_table_id, edge.previous_table_id): edge
        for edge in edges
        if (edge.current_table_id, edge.previous_table_id) not in excluded
    }
    if not edge_lookup:
        return []

    # One private dummy column per CQ row allows a table to remain unmatched.
    invalid_cost = 10_000.0
    costs: list[list[float]] = []
    for row_index, current_id in enumerate(current_ids):
        row = [
            -edge_lookup[(current_id, previous_id)].score if (current_id, previous_id) in edge_lookup else invalid_cost
            for previous_id in previous_ids
        ]
        row.extend(0.0 if dummy_index == row_index else invalid_cost for dummy_index in range(len(current_ids)))
        costs.append(row)
    row_indexes, column_indexes = linear_sum_assignment(costs)
    assigned: list[_Edge] = []
    for row_index, column_index in zip(row_indexes, column_indexes):
        if column_index >= len(previous_ids):
            continue
        key = (current_ids[row_index], previous_ids[column_index])
        edge = edge_lookup.get(key)
        if edge is not None:
            assigned.append(edge)
    return assigned


def _final_inspect(
    edge: _Edge,
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    payload = {
        "previous_table": {**previous, "table_id": _alias_previous(edge.previous_table_id)},
        "current_table": {**current, "table_id": _alias_current(edge.current_table_id)},
        "objective_facts": edge.facts,
        "first_review": {"confidence": edge.confidence, "reason": edge.reason},
    }
    response = call_openai_json(
        model=model,
        messages=[
            {"role": "system", "content": _FINAL_INSPECTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        usage_recorder=usage_recorder,
        call_kind="rbc_hybrid_final_inspector",
        response_model=_FinalInspectionResponse,
    )
    confirmed = response.get("verdict") == "confirmed" and float(response.get("confidence", 0.0) or 0.0) >= 0.75
    return confirmed, str(response.get("reason", "") or "").strip()


def run_rbc_hybrid_recovery(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    embedding_model: str,
    top_k: int,
    min_confidence: float,
    call_openai_json: Callable[..., dict[str, Any]],
    call_openai_embeddings: Callable[..., list[list[float]]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recupere les restes RBC avec embeddings, LLM et unicite globale."""
    metrics = {
        "executed": True,
        "hybrid_recovery_executed": 1,
        "hybrid_candidate_pairs_total": 0,
        "hybrid_judge_calls_total": 0,
        "hybrid_final_inspector_calls_total": 0,
        "hybrid_pairs_rejected_total": 0,
        "hybrid_embedding_calls_total": 0,
        "warnings": [],
    }
    if not current_cards:
        return {**metrics, "current_table_decisions": []}
    if not previous_cards:
        return {
            **metrics,
            "current_table_decisions": [
                {
                    "current_table_id": str(card.get("table_id", "")),
                    "decision": "added",
                    "reason": "No previous RBC table remained available for hybrid recovery.",
                }
                for card in current_cards
            ],
        }

    try:
        embeddings = _embed_cards(
            previous_cards + current_cards,
            model=embedding_model,
            call_openai_embeddings=call_openai_embeddings,
            usage_recorder=usage_recorder,
        )
        metrics["hybrid_embedding_calls_total"] = 1
    except Exception as exc:
        logger.warning("RBC hybrid embedding retrieval failed closed: %s", exc)
        metrics["warnings"] = [f"rbc_hybrid_embeddings_failed:{type(exc).__name__}"]
        return {
            **metrics,
            "current_table_decisions": [
                {
                    "current_table_id": str(card.get("table_id", "")),
                    "decision": "added",
                    "reason": "RBC hybrid candidate retrieval failed; left unmatched for review.",
                }
                for card in current_cards
            ],
        }

    edges: list[_Edge] = []
    for current in current_cards:
        shortlist = _candidate_shortlist(current, previous_cards, embeddings, top_k=top_k)
        metrics["hybrid_candidate_pairs_total"] += len(shortlist)
        try:
            edges.extend(
                _judge_candidates(
                    current,
                    shortlist,
                    model=model,
                    min_confidence=min_confidence,
                    call_openai_json=call_openai_json,
                    usage_recorder=usage_recorder,
                )
            )
            metrics["hybrid_judge_calls_total"] += 1
        except Exception as exc:
            logger.warning("RBC hybrid candidate judge failed closed for %s: %s", current.get("table_id"), exc)
            metrics["warnings"].append(
                f"rbc_hybrid_judge_failed:{str(current.get('table_id', ''))}:{type(exc).__name__}"
            )

    previous_ids = [str(card.get("table_id", "")) for card in previous_cards]
    current_ids = [str(card.get("table_id", "")) for card in current_cards]
    previous_lookup = {str(card.get("table_id", "")): card for card in previous_cards}
    current_lookup = {str(card.get("table_id", "")): card for card in current_cards}
    excluded: set[tuple[str, str]] = set()
    confirmed_cache: set[tuple[str, str]] = set()

    while True:
        assigned = _assign_edges(edges, previous_ids, current_ids, excluded)
        newly_rejected = False
        for edge in assigned:
            key = (edge.current_table_id, edge.previous_table_id)
            if key in confirmed_cache:
                continue
            try:
                metrics["hybrid_final_inspector_calls_total"] += 1
                confirmed, _reason = _final_inspect(
                    edge,
                    previous_lookup[edge.previous_table_id],
                    current_lookup[edge.current_table_id],
                    model=model,
                    call_openai_json=call_openai_json,
                    usage_recorder=usage_recorder,
                )
            except Exception as exc:
                logger.warning("RBC hybrid final inspection failed closed for %s: %s", key, exc)
                confirmed = False
                metrics["warnings"].append(f"rbc_hybrid_inspector_failed:{edge.current_table_id}:{type(exc).__name__}")
            if confirmed:
                confirmed_cache.add(key)
            else:
                excluded.add(key)
                metrics["hybrid_pairs_rejected_total"] += 1
                newly_rejected = True
        if not newly_rejected:
            break

    final_edges = _assign_edges(edges, previous_ids, current_ids, excluded)
    final_by_current = {edge.current_table_id: edge for edge in final_edges}
    decisions: list[dict[str, Any]] = []
    for current_id in current_ids:
        edge = final_by_current.get(current_id)
        if edge is None:
            decisions.append(
                {
                    "current_table_id": current_id,
                    "decision": "added",
                    "reason": "No RBC candidate passed semantic and final 1:1 validation.",
                }
            )
        else:
            decisions.append(
                {
                    "current_table_id": current_id,
                    "decision": "matched",
                    "previous_table_id": edge.previous_table_id,
                    "match_confidence": edge.confidence,
                    "reason": edge.reason,
                }
            )
    return {**metrics, "current_table_decisions": decisions}
