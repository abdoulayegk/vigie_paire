"""Tests de régression pour l'alignement sémantique hybride."""

from __future__ import annotations

from typing import Any

from vigilance.text_analysis.chunk_alignment import (
    _align_chunks_hybrid,
    _align_chunks_tfidf,
)
from vigilance.text_analysis.chunking import _chunk_subsection_text
from vigilance.text_analysis.global_reconciliation import (
    _ReconciliationResponse,
    _components,
    _one_sided_nodes,
    _pair_retrieval_scores,
    reconcile_global_change_fragments,
)
from vigilance.text_analysis.summary import _build_semantic_quality_metrics
from vigilance.text_analysis.triage import (
    _deterministic_cosmetic_exclusion,
    _group_semantic_triage_duplicates,
    _propagate_triage_to_group,
    _triage_section_changes,
)


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingsAPI:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = list(vectors)
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str]):
        self.calls.append(list(input))
        batch = self._vectors[: len(input)]
        self._vectors = self._vectors[len(input) :]
        return type(
            "Response",
            (),
            {
                "data": [
                    _FakeEmbeddingItem(index, vector)
                    for index, vector in enumerate(batch)
                ]
            },
        )()


class _FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = _FakeEmbeddingsAPI(vectors)


def _unit(index: int, dim: int = 4) -> list[float]:
    vector = [0.0] * dim
    vector[index % dim] = 1.0
    return vector


def test_hybrid_alignment_skips_non_reciprocal_pairs(monkeypatch) -> None:
    """Une paire non réciproque ne doit pas bloquer un meilleur match réciproque."""
    chunks_t1 = _chunk_subsection_text(
        "\n\n".join(
            [
                "Le cadre de gouvernance du risque stratégique prévoit une surveillance indépendante et des rapports réguliers au conseil.",
                "La banque ajoute un paragraphe distinct sur l'intelligence artificielle et la surveillance des modèles numériques.",
            ]
        ),
        subsection_heading="Risque de stratégie",
    )
    chunks_t2 = _chunk_subsection_text(
        "\n\n".join(
            [
                "La banque ajoute un paragraphe distinct sur l'intelligence artificielle et la surveillance des modèles numériques.",
                "Le cadre de gouvernance du risque stratégique prévoit une surveillance indépendante et des rapports réguliers au conseil.",
            ]
        ),
        subsection_heading="Risque de stratégie",
    )

    # Embeddings volontairement croisés pour créer une fausse meilleure paire non réciproque
    # si on consommait les slots trop tôt: T1-0↔T2-0 et T1-1↔T2-1 seraient mauvais.
    monkeypatch.setattr(
        "vigilance.text_analysis.chunk_alignment._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],  # T1-0
            [0.0, 1.0, 0.0],  # T1-1
            [0.0, 1.0, 0.0],  # T2-0 == T1-1
            [1.0, 0.0, 0.0],  # T2-1 == T1-0
        ],
    )
    hybrid = _align_chunks_hybrid(chunks_t1, chunks_t2, client=object())
    matched = {
        (alignment.chunk_t1.chunk_id, alignment.chunk_t2.chunk_id)
        for alignment in hybrid
        if alignment.chunk_t1 and alignment.chunk_t2
    }
    assert ("c00", "c01") in matched
    assert ("c01", "c00") in matched
    assert not any(alignment.alignment_type.startswith("possible_") for alignment in hybrid)


def test_hybrid_alignment_recovers_strong_reformulation(monkeypatch) -> None:
    """TF-IDF faible + embedding fort => paire récupérée, pas un faux added/removed."""
    previous = (
        "La banque surveille attentivement l'incertitude commerciale liée aux tarifs "
        "douaniers et évalue les répercussions possibles sur le risque de crédit des "
        "emprunteurs exposés aux chaînes d'approvisionnement internationales."
    )
    current = (
        "Nous suivons de près les tensions commerciales et les droits de douane, "
        "puis analysons leurs effets potentiels sur la qualité du crédit des clients "
        "touchés par des perturbations logistiques mondiales."
    )
    chunks_t1 = _chunk_subsection_text(previous, subsection_heading="Risque macro")
    chunks_t2 = _chunk_subsection_text(current, subsection_heading="Risque macro")
    assert len(chunks_t1) == 1 and len(chunks_t2) == 1

    tfidf_only = _align_chunks_tfidf(chunks_t1, chunks_t2)
    tfidf_matched = [
        alignment
        for alignment in tfidf_only
        if alignment.chunk_t1 and alignment.chunk_t2
    ]
    # Sans embeddings, la reformulation reste au mieux faible/ambiguë.
    assert not tfidf_matched or tfidf_matched[0].alignment_type != "matched_strong" or tfidf_matched[0].similarity_score < 0.95

    monkeypatch.setattr(
        "vigilance.text_analysis.chunk_alignment._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],
            [0.98, 0.1, 0.0],
        ],
    )
    client = _FakeEmbeddingClient([])
    hybrid = _align_chunks_hybrid(chunks_t1, chunks_t2, client=client)
    matched = [alignment for alignment in hybrid if alignment.chunk_t1 and alignment.chunk_t2]
    assert len(matched) == 1
    assert matched[0].embedding_score >= 0.85
    assert matched[0].alignment_type in {"matched_strong", "matched_weak", "ambiguous"}
    assert not any(alignment.alignment_type == "possible_added" for alignment in hybrid)
    assert not any(alignment.alignment_type == "possible_removed" for alignment in hybrid)


def test_hybrid_alignment_marks_boilerplate_as_ambiguous_for_gpt(monkeypatch) -> None:
    """Embeddings proches sur du boilerplate distinct => ambigu, GPT doit trancher."""
    previous = (
        "Le 15 mars 2024, la Banque a émis des billets à moyen terme d'un montant "
        "de 500 millions de dollars échéant en 2029, destinés au financement général."
    )
    current = (
        "Le 12 juin 2025, la Banque a émis des billets à moyen terme d'un montant "
        "de 750 millions de dollars échéant en 2031, destinés au financement général."
    )
    chunks_t1 = _chunk_subsection_text(previous, subsection_heading="Financement")
    chunks_t2 = _chunk_subsection_text(current, subsection_heading="Financement")

    monkeypatch.setattr(
        "vigilance.text_analysis.chunk_alignment._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
        ],
    )
    hybrid = _align_chunks_hybrid(chunks_t1, chunks_t2, client=_FakeEmbeddingClient([]))
    matched = [alignment for alignment in hybrid if alignment.chunk_t1 and alignment.chunk_t2]
    assert len(matched) == 1
    # Les faits chiffrés divergent: forcer l'arbitrage GPT malgré un embedding fort.
    assert matched[0].alignment_type == "ambiguous"
    assert matched[0].embedding_score >= 0.85


def test_global_reconciliation_uses_embedding_scores_in_audit(monkeypatch) -> None:
    removed = (
        "La dépendance envers les tiers et les modèles externes demeure un risque "
        "opérationnel important pour la continuité des services critiques de la banque."
    )
    added = (
        "Le recours à des fournisseurs externes et à des modèles tiers constitue "
        "toujours un risque opérationnel majeur pour la continuité des services critiques."
    )
    changes = [
        {
            "change_id": "removed_1",
            "section_key": "gestion_risques",
            "subsection_heading": "Risque opérationnel",
            "diff_type": "removed",
            "source_text_t1": removed,
            "source_text_t2": "",
            "pages_t1": [10],
            "pages_t2": [],
        },
        {
            "change_id": "added_1",
            "section_key": "gestion_risques",
            "subsection_heading": "Tiers",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": added,
            "pages_t1": [],
            "pages_t2": [22],
        },
    ]

    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],
            [0.95, 0.1, 0.0],
        ],
    )

    def _fake_llm(*args, **kwargs):
        return _ReconciliationResponse(
            decision="moved_unchanged",
            confidence="high",
            rationale="Même divulgation déplacée sous une autre rubrique.",
            matches=[
                {
                    "t1_node_id": "n0000",
                    "t2_node_id": "n0001",
                    "text_t1": removed,
                    "text_t2": added,
                }
            ],
        )

    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._call_structured_completion_with_correction",
        _fake_llm,
    )

    scores = _pair_retrieval_scores(
        removed,
        added,
        embedding_t1=[1.0, 0.0, 0.0],
        embedding_t2=[0.95, 0.1, 0.0],
    )
    assert scores["embedding_score"] >= 0.9

    components, edges = _components(
        _one_sided_nodes(changes),
        embeddings_by_id={
            "n0000": [1.0, 0.0, 0.0],
            "n0001": [0.95, 0.1, 0.0],
        },
    )
    assert len(components) == 1
    assert edges[0]["embedding_score"] >= 0.9

    reconciled, audit = reconcile_global_change_fragments(
        client=object(),
        model="gpt-4o",
        changes=changes,
    )
    assert reconciled == []
    assert audit[0]["applied"] is True
    assert audit[0]["candidate_scores"]
    assert "embedding_score" in audit[0]["candidate_scores"][0]
    assert "token_overlap" in audit[0]["candidate_scores"][0]


def test_global_components_reject_weak_transitive_bridge(monkeypatch) -> None:
    """Two strong families must not merge through one weak semantic bridge."""
    long_text = (
        "Ce fragment narratif contient suffisamment de texte pour participer à "
        "la réconciliation globale et représenter une divulgation bancaire complète."
    )
    changes = [
        {
            "change_id": "old_economy",
            "section_key": "gestion_risques",
            "diff_type": "removed",
            "source_text_t1": f"{long_text} Économie.",
            "source_text_t2": "",
        },
        {
            "change_id": "old_cyber",
            "section_key": "gestion_risques",
            "diff_type": "removed",
            "source_text_t1": f"{long_text} Cybersécurité.",
            "source_text_t2": "",
        },
        {
            "change_id": "new_economy",
            "section_key": "gestion_risques",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": f"{long_text} Économie.",
        },
        {
            "change_id": "new_cyber",
            "section_key": "gestion_risques",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": f"{long_text} Cybersécurité.",
        },
    ]
    candidate_edges = [
        {
            "t1_node_id": "n0000",
            "t2_node_id": "n0002",
            "t1_change_id": "old_economy",
            "t2_change_id": "new_economy",
            "section_key": "gestion_risques",
            "token_overlap": 1.0,
            "embedding_score": 1.0,
            "hybrid_score": 1.0,
        },
        {
            "t1_node_id": "n0001",
            "t2_node_id": "n0003",
            "t1_change_id": "old_cyber",
            "t2_change_id": "new_cyber",
            "section_key": "gestion_risques",
            "token_overlap": 1.0,
            "embedding_score": 1.0,
            "hybrid_score": 1.0,
        },
        {
            "t1_node_id": "n0000",
            "t2_node_id": "n0003",
            "t1_change_id": "old_economy",
            "t2_change_id": "new_cyber",
            "section_key": "gestion_risques",
            "token_overlap": 0.03,
            "embedding_score": 0.73,
            "hybrid_score": 0.73,
        },
    ]
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._candidate_edges",
        lambda nodes, embeddings_by_id: [dict(edge) for edge in candidate_edges],
    )

    components, audited_edges = _components(_one_sided_nodes(changes))
    component_ids = [
        {str(node.change.get("change_id") or "") for node in component}
        for component in components
    ]

    assert {"old_economy", "new_economy"} in component_ids
    assert {"old_cyber", "new_cyber"} in component_ids
    assert not any(len(component) == 4 for component in components)
    weak_bridge = next(edge for edge in audited_edges if edge["hybrid_score"] == 0.73)
    assert weak_bridge["component_selected"] is False
    assert weak_bridge["component_edge_strength"] == "retrieval_only"


def test_global_components_preserve_one_to_many_strong_split(monkeypatch) -> None:
    """One old block may still form a coherent component with two new fragments."""
    long_text = (
        "Ce fragment narratif contient suffisamment de texte pour participer à "
        "la réconciliation globale et représenter une divulgation bancaire complète."
    )
    changes = [
        {
            "change_id": "old_combined",
            "section_key": "gestion_risques",
            "diff_type": "removed",
            "source_text_t1": f"{long_text} Partie A. Partie B.",
            "source_text_t2": "",
        },
        {
            "change_id": "new_part_a",
            "section_key": "gestion_risques",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": f"{long_text} Partie A.",
        },
        {
            "change_id": "new_part_b",
            "section_key": "gestion_risques",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": f"{long_text} Partie B.",
        },
    ]
    candidate_edges = [
        {
            "t1_node_id": "n0000",
            "t2_node_id": "n0001",
            "t1_change_id": "old_combined",
            "t2_change_id": "new_part_a",
            "section_key": "gestion_risques",
            "token_overlap": 0.8,
            "embedding_score": 0.95,
            "hybrid_score": 0.95,
        },
        {
            "t1_node_id": "n0000",
            "t2_node_id": "n0002",
            "t1_change_id": "old_combined",
            "t2_change_id": "new_part_b",
            "section_key": "gestion_risques",
            "token_overlap": 0.8,
            "embedding_score": 0.93,
            "hybrid_score": 0.93,
        },
    ]
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._candidate_edges",
        lambda nodes, embeddings_by_id: [dict(edge) for edge in candidate_edges],
    )

    components, audited_edges = _components(_one_sided_nodes(changes))

    assert len(components) == 1
    assert {
        str(node.change.get("change_id") or "")
        for node in components[0]
    } == {"old_combined", "new_part_a", "new_part_b"}
    assert all(edge["component_selected"] for edge in audited_edges)


def test_global_reconciliation_keeps_real_unilateral_when_embeddings_weak(monkeypatch) -> None:
    changes = [
        {
            "change_id": "removed_1",
            "section_key": "gestion_risques",
            "subsection_heading": "Cyber",
            "diff_type": "removed",
            "source_text_t1": (
                "Les ransomwares et les attaques par déni de service demeurent des "
                "cybermenaces prioritaires pour l'infrastructure critique."
            ),
            "source_text_t2": "",
        },
        {
            "change_id": "added_1",
            "section_key": "gestion_capital",
            "subsection_heading": "CET1",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": (
                "Le ratio de fonds propres de catégorie 1 sous forme d'actions ordinaires "
                "demeure supérieur au minimum réglementaire applicable."
            ),
        },
    ]
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._call_structured_completion_with_correction",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    reconciled, audit = reconcile_global_change_fragments(
        client=object(),
        model="gpt-4o",
        changes=changes,
    )
    assert reconciled == changes
    assert audit == []


def test_global_reconciliation_does_not_mix_capital_and_risks(monkeypatch) -> None:
    """Même texte reformulé, mais sections différentes => pas de composant."""
    text_t1 = (
        "La dépendance envers les tiers et les modèles externes demeure un risque "
        "opérationnel important pour la continuité des services critiques de la banque."
    )
    text_t2 = (
        "Le recours à des fournisseurs externes et à des modèles tiers constitue "
        "toujours un risque opérationnel majeur pour la continuité des services critiques."
    )
    changes = [
        {
            "change_id": "removed_risks",
            "section_key": "gestion_risques",
            "subsection_heading": "Risque opérationnel",
            "diff_type": "removed",
            "source_text_t1": text_t1,
            "source_text_t2": "",
        },
        {
            "change_id": "added_capital",
            "section_key": "gestion_capital",
            "subsection_heading": "Contrôles",
            "diff_type": "added",
            "source_text_t1": "",
            "source_text_t2": text_t2,
        },
    ]
    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
        ],
    )
    components, edges = _components(
        _one_sided_nodes(changes),
        embeddings_by_id={
            "n0000": [1.0, 0.0, 0.0],
            "n0001": [0.99, 0.05, 0.0],
        },
    )
    assert edges == []
    assert components == []

    monkeypatch.setattr(
        "vigilance.text_analysis.global_reconciliation._call_structured_completion_with_correction",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    reconciled, audit = reconcile_global_change_fragments(
        client=object(),
        model="gpt-4o",
        changes=changes,
    )
    assert reconciled == changes
    assert audit == []


def test_deterministic_cosmetic_prefilter_skips_near_identical_text() -> None:
    change = {
        "diff_type": "modified",
        "source_text_t1": "La banque renforce sa gouvernance des risques.",
        "source_text_t2": "La banque renforce sa gouvernance des risques !",
    }
    assert _deterministic_cosmetic_exclusion(change) == "formatage_visuel"

    material = {
        "diff_type": "modified",
        "source_text_t1": "Le seuil prudentiel CET1 minimal applicable est de 4,5 %.",
        "source_text_t2": "Le seuil prudentiel CET1 minimal applicable est de 5,0 %.",
    }
    assert _deterministic_cosmetic_exclusion(material) is None


def test_triage_dedup_groups_compatible_near_duplicates(monkeypatch) -> None:
    changes = [
        {
            "change_id": "c1",
            "diff_type": "added",
            "alignment_decision": "same_disclosure",
            "subsection_heading": "Tarifs",
            "change_summary": "Ajout sur les tarifs douaniers",
            "source_text_t1": "",
            "source_text_t2": "Les tarifs douaniers accroissent l'incertitude commerciale.",
        },
        {
            "change_id": "c2",
            "diff_type": "added",
            "alignment_decision": "same_disclosure",
            "subsection_heading": "Tarifs",
            "change_summary": "Ajout similaire sur les tarifs douaniers",
            "source_text_t1": "",
            "source_text_t2": "Les tarifs douaniers accroissent l'incertitude commerciale et le risque.",
        },
        {
            "change_id": "c3",
            "diff_type": "removed",
            "alignment_decision": "distinct_disclosures",
            "subsection_heading": "Cyber",
            "change_summary": "Retrait cyber",
            "source_text_t1": "Les ransomwares restent une menace.",
            "source_text_t2": "",
        },
    ]
    monkeypatch.setattr(
        "vigilance.text_analysis.triage._embed_texts",
        lambda client, texts, model="text-embedding-3-small": [
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    groups = _group_semantic_triage_duplicates(changes, client=object())
    assert [0, 1] in groups or groups[0] == [0, 1]
    assert any(group == [2] for group in groups)

    representative = {
        **changes[0],
        "genai_triage": {"is_relevant": True, "themes_amf": ["RISQUE_MACRO_GEOPOLITIQUE"], "source": "gpt"},
    }
    propagated = _propagate_triage_to_group(
        representative=representative,
        members=changes[:2],
        group_id="gestion_risques_triage_group_001",
    )
    assert propagated[1]["triage_dedup"]["propagated"] is True
    assert propagated[1]["genai_triage"]["triage_group_id"] == "gestion_risques_triage_group_001"


def test_triage_section_changes_applies_cosmetic_prefilter(monkeypatch) -> None:
    calls: list[Any] = []

    def _fake_structured(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("GPT triage should be skipped for cosmetic changes")

    monkeypatch.setattr(
        "vigilance.text_analysis.triage._call_structured_completion_with_correction",
        _fake_structured,
    )
    result = _triage_section_changes(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        changes=[
            {
                "diff_type": "modified",
                "semantic_text_t1": "La banque renforce sa gouvernance des risques.",
                "semantic_text_t2": "La banque renforce sa gouvernance des risques.",
                "source_text_t1": "La banque renforce sa gouvernance des risques.",
                "source_text_t2": "La banque renforce sa gouvernance des risques!",
                "change_summary": "Ponctuation",
            }
        ],
    )
    assert calls == []
    assert result[0]["genai_triage"]["source"] == "deterministic_prefilter"
    assert result[0]["genai_triage"]["exclusion_reason"] == "formatage_visuel"


def test_semantic_quality_metrics_capture_hybrid_decisions() -> None:
    section_comparisons = [
        {
            "section_key": "gestion_risques",
            "all_block_comparisons": [
                {
                    "alignment_type": "ambiguous",
                    "alignment_decision": "uncertain",
                    "genai_triage": {"source": "alignment_review_required", "alignment_review_required": True},
                },
                {
                    "alignment_type": "matched_strong",
                    "alignment_decision": "same_disclosure",
                    "genai_triage": {"source": "deterministic_prefilter"},
                    "triage_prefilter": {"excluded": True},
                },
                {
                    "alignment_type": "possible_added",
                    "alignment_decision": "same_disclosure",
                    "genai_triage": {
                        "source": "gpt_propagated",
                        "triage_group_id": "g1",
                    },
                    "triage_dedup": {
                        "group_id": "g1",
                        "representative_change_id": "a",
                        "member_change_ids": ["a", "b"],
                        "propagated": True,
                    },
                },
                {
                    "alignment_type": "possible_added",
                    "alignment_decision": "same_disclosure",
                    "genai_triage": {
                        "source": "gpt",
                        "triage_group_id": "g1",
                    },
                    "triage_dedup": {
                        "group_id": "g1",
                        "representative_change_id": "a",
                        "member_change_ids": ["a", "b"],
                        "propagated": False,
                    },
                },
            ],
        }
    ]
    metrics = _build_semantic_quality_metrics(
        section_comparisons=section_comparisons,
        reconciliation_audit=[{"applied": True}, {"applied": False}],
    )
    assert metrics["ambiguous_alignment_count"] == 1
    assert metrics["human_review_count"] == 1
    assert metrics["triage_prefiltered_count"] == 1
    assert metrics["reconciliation_applied_count"] == 1
    assert metrics["triage_dedup_group_count"] == 1
    assert metrics["triage_dedup_member_count"] == 2
