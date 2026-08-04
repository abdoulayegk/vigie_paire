"""Réconciliation sémantique globale des fragments ajoutés et retirés.

La comparaison locale travaille volontairement sous-section par sous-section.
Cette passe intervient ensuite : elle repère les cas où une divulgation a été
scindée, regroupée ou déplacée entre rubriques, puis soumet chaque composant
suspect à une décision GPT structurée avant de modifier les changements.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from vigie.analyse_texte.normalization import _json_dumps, _sanitize_explanation, _sanitize_semantic_text
from vigie.analyse_texte.openai_client import (
    _call_structured_completion_with_correction,
    _embed_texts,
)


logger = logging.getLogger(__name__)

_MIN_FRAGMENT_CHARS = 80
_MIN_TOKEN_OVERLAP = 0.24
_MIN_EMBEDDING_SCORE = 0.72
_MAX_COMPONENT_NODES = 48
_MAX_COHERENT_COMPONENT_NODES = 10
_COMPONENT_STRONG_TOKEN_OVERLAP = 0.50
_COMPONENT_STRONG_EMBEDDING_SCORE = 0.88
_COMPONENT_MUTUAL_MIN_SCORE = 0.78
_COMPONENT_NEAR_BEST_MARGIN = 0.08
_EMBEDDING_TRUNCATE_CHARS = 4000
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_SEMANTIC_DECISIONS = {
    "same_content_resegmented",
    "moved_unchanged",
    "same_disclosure_modified",
    "distinct_disclosures",
    "uncertain",
}
_STOPWORDS = {
    "ainsi",
    "avec",
    "cette",
    "comme",
    "dans",
    "des",
    "elle",
    "elles",
    "est",
    "les",
    "leur",
    "leurs",
    "mais",
    "par",
    "pour",
    "que",
    "qui",
    "sur",
    "une",
    "vers",
}


@dataclass(slots=True)
class _Node:
    node_id: str
    order: int
    change: dict[str, Any]
    side: Literal["t1", "t2"]
    text: str


class _ReconciliationMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t1_node_id: str
    t2_node_id: str
    text_t1: str
    text_t2: str

    @field_validator("t1_node_id", "t2_node_id", "text_t1", "text_t2", mode="before")
    @classmethod
    def _coerce_string(cls, value: Any) -> str:
        return str(value or "").strip()


class _ReconciliationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal[
        "same_content_resegmented",
        "moved_unchanged",
        "same_disclosure_modified",
        "distinct_disclosures",
        "uncertain",
    ]
    confidence: Literal["high", "medium", "low"]
    rationale: str
    matches: list[_ReconciliationMatch]

    @field_validator("rationale", mode="before")
    @classmethod
    def _coerce_rationale(cls, value: Any) -> str:
        return str(value or "").strip()


def _normalized_compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _meaningful_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", str(value or "").lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]{3,}", normalized)
    return {token for token in tokens if token not in _STOPWORDS}


def _candidate_score(text_t1: str, text_t2: str) -> float:
    """Lexical retrieval score only — GPT, not this score, decides the relationship."""
    compact_t1 = _normalized_compact(text_t1)
    compact_t2 = _normalized_compact(text_t2)
    if not compact_t1 or not compact_t2:
        return 0.0
    if compact_t1 in compact_t2 or compact_t2 in compact_t1:
        return 1.0
    tokens_t1 = _meaningful_tokens(text_t1)
    tokens_t2 = _meaningful_tokens(text_t2)
    if not tokens_t1 or not tokens_t2:
        return 0.0
    return len(tokens_t1 & tokens_t2) / len(tokens_t1 | tokens_t2)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _truncate_for_embedding(text: str) -> str:
    value = str(text or "").strip()
    if len(value) <= _EMBEDDING_TRUNCATE_CHARS:
        return value
    return value[:_EMBEDDING_TRUNCATE_CHARS]


def _pair_retrieval_scores(
    text_t1: str,
    text_t2: str,
    *,
    embedding_t1: list[float] | None = None,
    embedding_t2: list[float] | None = None,
) -> dict[str, float]:
    """Combine lexical overlap and embedding similarity for candidate retrieval."""
    token_overlap = _candidate_score(text_t1, text_t2)
    embedding_score = 0.0
    if embedding_t1 is not None and embedding_t2 is not None:
        embedding_score = _cosine_similarity(embedding_t1, embedding_t2)
    return {
        "token_overlap": token_overlap,
        "embedding_score": embedding_score,
        "hybrid_score": max(token_overlap, embedding_score),
    }


def _is_credible_pair(scores: dict[str, float]) -> bool:
    return (
        scores["token_overlap"] >= _MIN_TOKEN_OVERLAP
        or scores["embedding_score"] >= _MIN_EMBEDDING_SCORE
    )


def _one_sided_nodes(changes: list[dict[str, Any]]) -> list[_Node]:
    nodes: list[_Node] = []
    for order, change in enumerate(changes):
        diff_type = str(change.get("diff_type") or "").lower()
        if diff_type == "removed":
            text = str(change.get("source_text_t1") or "").strip()
            side: Literal["t1", "t2"] = "t1"
        elif diff_type == "added":
            text = str(change.get("source_text_t2") or "").strip()
            side = "t2"
        else:
            continue
        if len(text) < _MIN_FRAGMENT_CHARS:
            continue
        nodes.append(_Node(node_id=f"n{order:04d}", order=order, change=change, side=side, text=text))
    return nodes


def _build_node_embeddings(
    nodes: list[_Node],
    *,
    client: Any | None,
    embedding_model: str,
) -> dict[str, list[float]]:
    if client is None or not nodes:
        return {}
    try:
        vectors = _embed_texts(
            client,
            [_truncate_for_embedding(node.text) for node in nodes],
            model=embedding_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embeddings réconciliation globale indisponibles: %s", exc)
        return {}
    return {node.node_id: vector for node, vector in zip(nodes, vectors, strict=False)}


def _candidate_edges(
    nodes: list[_Node],
    *,
    embeddings_by_id: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Forme les arêtes candidates cross-subsection dans une même section.

    Les ``added``/``removed`` de toutes les sous-sections d'une section
    (ex. tout ``gestion_risques``) sont comparés ensemble. On ne mélange
    jamais ``gestion_risques`` avec ``gestion_capital``.
    """
    previous = [node for node in nodes if node.side == "t1"]
    current = [node for node in nodes if node.side == "t2"]
    edges: list[dict[str, Any]] = []
    for left in previous:
        left_section = str(left.change.get("section_key") or "")
        for right in current:
            right_section = str(right.change.get("section_key") or "")
            if left_section != right_section:
                continue
            scores = _pair_retrieval_scores(
                left.text,
                right.text,
                embedding_t1=embeddings_by_id.get(left.node_id),
                embedding_t2=embeddings_by_id.get(right.node_id),
            )
            if not _is_credible_pair(scores):
                continue
            edges.append(
                {
                    "t1_node_id": left.node_id,
                    "t2_node_id": right.node_id,
                    "t1_change_id": str(left.change.get("change_id") or ""),
                    "t2_change_id": str(right.change.get("change_id") or ""),
                    "section_key": left_section,
                    "token_overlap": round(scores["token_overlap"], 4),
                    "embedding_score": round(scores["embedding_score"], 4),
                    "hybrid_score": round(scores["hybrid_score"], 4),
                }
            )
    edges.sort(key=lambda edge: (-edge["hybrid_score"], edge["t1_node_id"], edge["t2_node_id"]))
    return edges


def _mark_component_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selects coherent graph edges without losing weaker retrieval evidence.

    All credible edges remain in the audit.  Only strong edges, or edges that
    are mutually near the best candidate on both sides, may construct a
    component.  This prevents a chain of weak similarities from merging
    unrelated themes while preserving strong 1→N and N→1 stars.
    """
    best_by_t1: dict[str, float] = {}
    best_by_t2: dict[str, float] = {}
    for edge in edges:
        score = float(edge.get("hybrid_score") or 0.0)
        t1_node_id = str(edge.get("t1_node_id") or "")
        t2_node_id = str(edge.get("t2_node_id") or "")
        best_by_t1[t1_node_id] = max(best_by_t1.get(t1_node_id, 0.0), score)
        best_by_t2[t2_node_id] = max(best_by_t2.get(t2_node_id, 0.0), score)

    marked: list[dict[str, Any]] = []
    for raw_edge in edges:
        edge = dict(raw_edge)
        score = float(edge.get("hybrid_score") or 0.0)
        token_overlap = float(edge.get("token_overlap") or 0.0)
        embedding_score = float(edge.get("embedding_score") or 0.0)
        t1_node_id = str(edge.get("t1_node_id") or "")
        t2_node_id = str(edge.get("t2_node_id") or "")
        strong = (
            token_overlap >= _COMPONENT_STRONG_TOKEN_OVERLAP
            or embedding_score >= _COMPONENT_STRONG_EMBEDDING_SCORE
        )
        mutually_near_best = (
            score >= _COMPONENT_MUTUAL_MIN_SCORE
            and score >= best_by_t1.get(t1_node_id, 0.0) - _COMPONENT_NEAR_BEST_MARGIN
            and score >= best_by_t2.get(t2_node_id, 0.0) - _COMPONENT_NEAR_BEST_MARGIN
        )
        edge["component_selected"] = strong or mutually_near_best
        edge["component_edge_strength"] = (
            "strong"
            if strong
            else "mutual_near_best"
            if mutually_near_best
            else "retrieval_only"
        )
        marked.append(edge)
    return marked


def _components(
    nodes: list[_Node],
    *,
    embeddings_by_id: dict[str, list[float]] | None = None,
) -> tuple[list[list[_Node]], list[dict[str, Any]]]:
    """Builds within-section candidate components from provisional changes.

    Subsections of the same section are mixed (e.g. all of ``gestion_risques``).
    Different top-level sections stay isolated (capital never reconciles with risks).
    """
    embeddings_by_id = embeddings_by_id or {}
    parents = {node.node_id: node.node_id for node in nodes}
    members = {node.node_id: {node.node_id} for node in nodes}

    def find(node_id: str) -> str:
        while parents[node_id] != node_id:
            parents[node_id] = parents[parents[node_id]]
            node_id = parents[node_id]
        return node_id

    def union(left: str, right: str, *, strong: bool) -> bool:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return True
        left_members = members[root_left]
        right_members = members[root_right]
        if len(left_members | right_members) > _MAX_COHERENT_COMPONENT_NODES:
            return False
        # A weaker edge may extend a 1→N/N→1 star, but it must not merge two
        # already formed families. Strong evidence can merge them, subject to
        # the hard component-size cap.
        if not strong and len(left_members) > 1 and len(right_members) > 1:
            return False
        parents[root_right] = root_left
        members[root_left] = left_members | right_members
        members.pop(root_right, None)
        return True

    edges = _mark_component_edges(
        _candidate_edges(nodes, embeddings_by_id=embeddings_by_id)
    )
    for edge in edges:
        if not edge["component_selected"]:
            continue
        merged = union(
            edge["t1_node_id"],
            edge["t2_node_id"],
            strong=edge["component_edge_strength"] == "strong",
        )
        if not merged:
            edge["component_selected"] = False
            edge["component_edge_strength"] = "rejected_component_bridge"

    grouped: dict[str, list[_Node]] = {}
    for node in nodes:
        grouped.setdefault(find(node.node_id), []).append(node)
    components = [
        sorted(component, key=lambda node: node.order)
        for component in grouped.values()
        if len(component) > 1
        and {node.side for node in component} == {"t1", "t2"}
        and len(component) <= _MAX_COMPONENT_NODES
        and len({str(node.change.get("section_key") or "") for node in component}) == 1
    ]
    return components, edges


def _edges_for_component(component: list[_Node], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_ids = {node.node_id for node in component}
    return [
        edge
        for edge in edges
        if edge["t1_node_id"] in node_ids and edge["t2_node_id"] in node_ids
    ]


def _component_prompt(component: list[_Node], section_key: str) -> str:
    records = [
        {
            "node_id": node.node_id,
            "period": "T1 précédent" if node.side == "t1" else "T2 courant",
            "change_id": node.change.get("change_id", ""),
            "section_key": node.change.get("section_key", ""),
            "heading": node.change.get("subsection_heading", ""),
            "pages": node.change.get("pages_t1" if node.side == "t1" else "pages_t2", []),
            "text": node.text,
        }
        for node in component
    ]
    return (
        "Tu arbitres la relation sémantique entre des fragments de divulgation bancaire. "
        "Ils proviennent de sous-sections différentes d'UNE MÊME section "
        f"({section_key}) et ont été provisoirement marqués ajoutés ou retirés. "
        "Ils peuvent représenter une même information redécoupée, déplacée ou réordonnée "
        "à l'intérieur de cette section uniquement.\n\n"
        "Retourne une seule décision pour tout le composant :\n"
        "- same_content_resegmented : le contenu est conservé à l'identique mais découpé/regroupé;\n"
        "- moved_unchanged : même contenu déplacé sous une autre rubrique;\n"
        "- same_disclosure_modified : même divulgation, avec résidus réellement modifiés;\n"
        "- distinct_disclosures : faits ou événements réellement différents;\n"
        "- uncertain : impossible de trancher avec confiance.\n\n"
        "Pour chaque portion que tu déclares commune, ajoute un match entre un node T1 et un node T2. "
        "`text_t1` et `text_t2` doivent être des extraits EXACTS, copiés verbatim des textes fournis. "
        "Un même node peut avoir plusieurs matches : c'est nécessaire lorsqu'un gros bloc est devenu plusieurs fragments. "
        "Ne fais aucun triage AMF et ne déduis pas une pertinence métier.\n\n"
        f"Section: {section_key}\nCandidats:\n{_json_dumps(records)}"
    )


def _valid_matches(response: _ReconciliationResponse, nodes_by_id: dict[str, _Node]) -> list[_ReconciliationMatch]:
    valid: list[_ReconciliationMatch] = []
    for match in response.matches:
        previous = nodes_by_id.get(match.t1_node_id)
        current = nodes_by_id.get(match.t2_node_id)
        if previous is None or current is None or previous.side != "t1" or current.side != "t2":
            continue
        if not match.text_t1 or not match.text_t2:
            continue
        if match.text_t1 not in previous.text or match.text_t2 not in current.text:
            continue
        valid.append(match)
    return valid


def _residual_text(text: str, fragments: list[str]) -> str:
    intervals: list[tuple[int, int]] = []
    for fragment in fragments:
        start = text.find(fragment)
        if start >= 0:
            intervals.append((start, start + len(fragment)))
    if not intervals:
        return text.strip()
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            pieces.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return " ".join(piece.strip() for piece in pieces if piece.strip()).strip()


def _update_one_sided_residual(change: dict[str, Any], side: Literal["t1", "t2"], residual: str) -> dict[str, Any] | None:
    if not residual:
        return None
    updated = dict(change)
    if side == "t1":
        updated.update(
            {
                "source_text_t1": residual,
                "semantic_text_t1": _sanitize_semantic_text(residual),
                "evidence_t1": {"pages": (change.get("pages_t1") or []), "snippet": residual[:400]},
            }
        )
    else:
        updated.update(
            {
                "source_text_t2": residual,
                "semantic_text_t2": _sanitize_semantic_text(residual),
                "evidence_t2": {"pages": (change.get("pages_t2") or []), "snippet": residual[:400]},
            }
        )
    updated["alignment_type"] = "global_reconciled_residual"
    updated["alignment_decision"] = "same_disclosure"
    return updated


def _reconcile_component(
    *,
    component: list[_Node],
    response: _ReconciliationResponse,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any]]:
    """Returns node replacements plus an auditable reconciliation outcome."""
    nodes_by_id = {node.node_id: node for node in component}
    matches = _valid_matches(response, nodes_by_id)
    matched_by_node: dict[str, list[str]] = {node.node_id: [] for node in component}
    for match in matches:
        matched_by_node[match.t1_node_id].append(match.text_t1)
        matched_by_node[match.t2_node_id].append(match.text_t2)
    residuals = {
        node.node_id: _residual_text(node.text, matched_by_node.get(node.node_id, []))
        for node in component
    }
    fully_covered = bool(matches) and all(not residuals[node.node_id] for node in component)
    audit = {
        "component_change_ids": [str(node.change.get("change_id") or "") for node in component],
        "decision": response.decision,
        "confidence": response.confidence,
        "rationale": _sanitize_explanation(response.rationale),
        "valid_match_count": len(matches),
        "fully_covered": fully_covered,
        "applied": False,
    }

    if response.decision in {"same_content_resegmented", "moved_unchanged"}:
        if not fully_covered:
            audit["decision"] = "uncertain"
            audit["rationale"] = "Réconciliation incomplète : les extraits validés ne couvrent pas tous les fragments."
            return {node.node_id: dict(node.change) for node in component}, audit
        audit["applied"] = True
        return {node.node_id: None for node in component}, audit

    if response.decision == "same_disclosure_modified":
        previous_with_residual = [
            node for node in component if node.side == "t1" and residuals[node.node_id]
        ]
        current_with_residual = [
            node for node in component if node.side == "t2" and residuals[node.node_id]
        ]
        if len(previous_with_residual) == 1 and len(current_with_residual) == 1:
            previous = previous_with_residual[0]
            current = current_with_residual[0]
            replacement = dict(previous.change)
            replacement.update(
                {
                    "diff_type": "modified",
                    "alignment_id": f"global_{previous.node_id}_{current.node_id}",
                    "alignment_type": "global_reconciled_modified",
                    "alignment_decision": "same_disclosure",
                    "alignment_confidence": response.confidence,
                    "alignment_rationale": _sanitize_explanation(response.rationale),
                    "source_text_t1": residuals[previous.node_id],
                    "source_text_t2": residuals[current.node_id],
                    "semantic_text_t1": _sanitize_semantic_text(residuals[previous.node_id]),
                    "semantic_text_t2": _sanitize_semantic_text(residuals[current.node_id]),
                    "evidence_t1": {
                        "pages": previous.change.get("pages_t1") or [],
                        "snippet": residuals[previous.node_id][:400],
                    },
                    "evidence_t2": {
                        "pages": current.change.get("pages_t2") or [],
                        "snippet": residuals[current.node_id][:400],
                    },
                    "change_summary": _sanitize_explanation(response.rationale),
                }
            )
            replacements = {node.node_id: None for node in component}
            replacements[previous.node_id] = replacement
            audit["applied"] = bool(matches)
            return replacements, audit
        replacements = {
            node.node_id: _update_one_sided_residual(node.change, node.side, residuals[node.node_id])
            for node in component
        }
        audit["applied"] = bool(matches)
        return replacements, audit

    if response.decision == "uncertain":
        replacements: dict[str, dict[str, Any] | None] = {}
        for node in component:
            updated = dict(node.change)
            updated.update(
                {
                    "alignment_decision": "uncertain",
                    "alignment_confidence": response.confidence,
                    "alignment_rationale": _sanitize_explanation(response.rationale),
                }
            )
            replacements[node.node_id] = updated
        return replacements, audit

    # Distinct disclosures remain as independently added/removed records, but
    # carry the semantic verdict into the AMF prompt and the audit artifact.
    replacements: dict[str, dict[str, Any] | None] = {}
    for node in component:
        updated = dict(node.change)
        updated.update(
            {
                "alignment_type": "global_semantic_distinct",
                "alignment_decision": "distinct_disclosures",
                "alignment_confidence": response.confidence,
                "alignment_rationale": _sanitize_explanation(response.rationale),
            }
        )
        replacements[node.node_id] = updated
    return replacements, audit


def reconcile_global_change_fragments(
    *,
    client: Any,
    model: str,
    changes: list[dict[str, Any]],
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconciles provisional added/removed fragments within each section.

    Subsections of the same section are mixed so moved/split text can be
    recovered.  Different sections (capital vs risks) are never reconciled
    together.

    Deterministic logic only forms candidate components and validates verbatim
    excerpts.  GPT makes the semantic decision; a failed or incomplete answer
    fails closed and leaves the original changes intact (or marks them
    ``uncertain`` when that was GPT's explicit conclusion).
    """
    nodes = _one_sided_nodes(changes)
    embeddings_by_id = _build_node_embeddings(
        nodes,
        client=client,
        embedding_model=embedding_model,
    )
    components, edges = _components(nodes, embeddings_by_id=embeddings_by_id)
    if not components:
        return changes, []

    active: dict[str, dict[str, Any] | None] = {node.node_id: dict(node.change) for node in nodes}
    audit_rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components, start=1):
        section_keys = {str(node.change.get("section_key") or "") for node in component}
        section_label = next(iter(section_keys)) if len(section_keys) == 1 else "unknown"
        if len(section_keys) != 1:
            logger.warning(
                "Composant multi-sections ignoré component=%s sections=%s",
                component_index,
                sorted(section_keys),
            )
            continue
        component_edges = _edges_for_component(component, edges)
        try:
            response = _call_structured_completion_with_correction(
                client,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un arbitre sémantique de divulgations bancaires. "
                            "Tu dois distinguer un contenu déplacé ou redécoupé de faits réellement distincts "
                            "à l'intérieur d'une même section."
                        ),
                    },
                    {"role": "user", "content": _component_prompt(component, section_label)},
                ],
                response_format=_ReconciliationResponse,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Réconciliation globale ignorée composant=%s erreur=%s", component_index, exc)
            audit_rows.append(
                {
                    "component_change_ids": [str(node.change.get("change_id") or "") for node in component],
                    "section_key": section_label,
                    "decision": "unavailable",
                    "confidence": "low",
                    "rationale": "Arbitrage GPT indisponible; changements locaux conservés.",
                    "valid_match_count": 0,
                    "fully_covered": False,
                    "applied": False,
                    "candidate_scores": component_edges,
                }
            )
            continue

        replacements, audit = _reconcile_component(component=component, response=response)
        audit["component_id"] = f"global_{component_index:03d}"
        audit["section_key"] = section_label
        audit["candidate_scores"] = component_edges
        audit_rows.append(audit)
        for node_id, replacement in replacements.items():
            active[node_id] = replacement

    node_by_order = {node.order: node for node in nodes}
    reconciled: list[dict[str, Any]] = []
    for order, original in enumerate(changes):
        node = node_by_order.get(order)
        if node is None:
            reconciled.append(original)
            continue
        replacement = active.get(node.node_id)
        if replacement is not None:
            reconciled.append(replacement)
    return reconciled, audit_rows
