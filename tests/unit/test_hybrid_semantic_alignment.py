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

def _deterministic_bank_specific_exclusion(text: str, bank_code: str) -> bool:
    return False

def _deterministic_cosmetic_exclusion(text1: str, text2: str) -> bool:
    return False

def _group_semantic_triage_duplicates(changes: list, client: object = None) -> list:
    return changes

def _prefilter_triage_result(data: dict) -> dict:
    return data

def _propagate_triage_to_group(changes: list, triage: dict) -> list:
    return changes

def _triage_section_changes(
    changes: list[dict[str, Any]], client: object = None
) -> dict[str, Any]:
    return {}


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
    assert ("risque_de_stratégie_c00", "risque_de_stratégie_c01") in matched
    assert ("risque_de_stratégie_c01", "risque_de_stratégie_c00") in matched
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
    assert (
        _deterministic_bank_specific_exclusion(material)
        == "variation_numerique_propre_banque"
    )


def test_deterministic_bank_specific_excludes_numeric_and_operations() -> None:
    numeric = {
        "diff_type": "modified",
        "change_summary": "Le portefeuille hypothécaire passe de 287 G$ à 294 G$.",
        "source_text_t1": "Le portefeuille hypothécaire s'établit à 287 G$.",
        "source_text_t2": "Le portefeuille hypothécaire s'établit à 294 G$.",
    }
    assert (
        _deterministic_bank_specific_exclusion(numeric)
        == "variation_numerique_propre_banque"
    )

    calendar = {
        "diff_type": "modified",
        "change_summary": "Report du coefficient de plancher jusqu'à nouvel ordre.",
        "source_text_t1": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du coefficient de plancher jusqu'à l'exercice 2027."
        ),
        "source_text_t2": (
            "Le 12 février 2025, le BSIF a reporté toute augmentation "
            "supplémentaire du coefficient de plancher jusqu'à nouvel ordre."
        ),
    }
    assert _deterministic_bank_specific_exclusion(calendar) == "mise_a_jour_calendrier"

    acquisition = {
        "diff_type": "added",
        "change_summary": "Inclusion de CWB après l'acquisition.",
        "source_text_t1": "",
        "source_text_t2": (
            "L'inclusion de CWB à la suite de l'acquisition augmente "
            "l'actif pondéré en fonction des risques."
        ),
    }
    assert (
        _deterministic_bank_specific_exclusion(acquisition)
        == "operation_interne_banque"
    )

    cyber = {
        "diff_type": "added",
        "change_summary": "Ajout d'exercices annuels de simulation de cyberattaque.",
        "source_text_t1": "",
        "source_text_t2": (
            "La banque réalise désormais des simulations annuelles de "
            "cyberattaque avec ses unités d'affaires."
        ),
    }
    assert _deterministic_bank_specific_exclusion(cyber) is None


def _assert_natural_analyst_copy(text: str) -> None:
    lowered = text.lower()
    for forbidden in ("préfiltre", "prefiltre", "déterministe", "deterministe", "pipeline"):
        assert forbidden not in lowered, f"jargon interdit trouvé: {forbidden!r} dans {text!r}"


def test_deterministic_bank_specific_excludes_bnc_floor_reschedule() -> None:
    """Cas BNC réel : report du plancher BSIF → mise à jour de calendrier."""
    change = {
        "diff_type": "modified",
        "change_summary": (
            "Les deux fragments traitent de la même divulgation concernant "
            "le report de l'augmentation du coefficient de plancher."
        ),
        "source_text_t1": (
            "Le 5 juillet 2024, le BSIF a annoncé qu'il retardait d'un an "
            "l'augmentation du plancher de fonds propres. Par conséquent, "
            "le coefficient de plancher révisé atteindra 72,5 % à l'exercice 2027. "
            "Pour l'exercice 2024, et restera à ce niveau jusqu'à la fin de "
            "l'exercice 2025, pour ensuite augmenter jusqu'en 2027."
        ),
        "source_text_t2": (
            "Le 12 février 2025, le BSIF a reporté toute augmentation "
            "supplémentaire jusqu'à nouvel ordre. En conséquence, restera "
            "à ce niveau pour une période indéterminée."
        ),
    }
    assert (
        _deterministic_bank_specific_exclusion(change) == "mise_a_jour_calendrier"
    )
    enriched = _prefilter_triage_result(change, "mise_a_jour_calendrier")
    reason = enriched["genai_triage"]["relevance_reason"]
    justification = enriched["genai_triage"]["nouvelle_idee_justification"]
    _assert_natural_analyst_copy(reason)
    _assert_natural_analyst_copy(justification)
    assert "dates" in reason.lower() or "échéances" in reason.lower() or "passage" in reason.lower()


def test_deterministic_bank_specific_excludes_cwb_appetite_and_aprf() -> None:
    appetite = {
        "diff_type": "modified",
        "change_summary": (
            "Ajout d'une section sur la considération de la posture de risque "
            "et des impacts de l'acquisition récente de CWB"
        ),
        "source_text_t1": (
            "L'appétit pour le risque représente le niveau de risque qu'une "
            "entreprise est prête à assumer afin de réaliser sa stratégie "
            "d'affaires. L'appétit pour le risque est intégré aux processus "
            "de prise de décisions."
        ),
        "source_text_t2": (
            "L'appétit pour le risque représente le niveau de risque qu'une "
            "entreprise est prête à assumer afin de réaliser sa stratégie "
            "d'affaires. L'appétit pour le risque est intégré aux processus "
            "de prise de décisions. En établissant son appétit pour le risque, "
            "la Banque considère également sa posture de risque et tous les "
            "impacts pouvant découler d'un changement stratégique, tels que "
            "les impacts de l'acquisition récente de CWB."
        ),
    }
    assert (
        _deterministic_bank_specific_exclusion(appetite)
        == "operation_interne_banque"
    )
    enriched = _prefilter_triage_result(
        appetite,
        "operation_interne_banque",
        bank_code="bnc",
    )
    reason = enriched["genai_triage"]["relevance_reason"]
    _assert_natural_analyst_copy(reason)
    assert reason.startswith("BNC ")
    assert "acquisition" in reason.lower() or "opération" in reason.lower()

    emission = {
        "diff_type": "added",
        "change_summary": "Nouvelle divulgation concernant l'émission d'actions lors de l'acquisition de CWB",
        "source_text_t1": "",
        "source_text_t2": (
            "Le 3 février 2025, lors de la clôture de l'acquisition de CWB, "
            "la Banque a émis un total de 50 272 878 actions ordinaires, "
            "pour un produit brut de 6,3 G$."
        ),
    }
    assert (
        _deterministic_bank_specific_exclusion(emission)
        == "operation_interne_banque"
    )
    emission_copy = _prefilter_triage_result(
        emission,
        "operation_interne_banque",
        bank_code="bnc",
    )
    emission_reason = emission_copy["genai_triage"]["relevance_reason"]
    _assert_natural_analyst_copy(emission_reason)
    assert emission_reason.startswith("BNC ajoute ")
    assert "cwb" in emission_reason.lower() or "acquisition" in emission_reason.lower()

    aprf = {
        "diff_type": "modified",
        "change_summary": (
            "Les chiffres et les dates ont été mis à jour, et l'inclusion "
            "de CWB est mentionnée comme un nouveau facteur."
        ),
        "source_text_t1": (
            "L'actif pondéré en fonction des risques a augmenté de 15,4 G$ "
            "pour s'établir à 141,0 G$ au 31 octobre 2024. Cette augmentation "
            "découle de la croissance organique et des changements de méthode "
            "découlant principalement de la mise en œuvre des réformes de Bâle III."
        ),
        "source_text_t2": (
            "L'actif pondéré en fonction des risques a augmenté de 47,8 G$ "
            "pour s'établir à 188,8 G$ au 31 octobre 2025. Cette augmentation "
            "découle principalement de l'inclusion de CWB, ainsi que de la "
            "croissance organique de l'actif pondéré en fonction des risques."
        ),
    }
    exclusion = _deterministic_bank_specific_exclusion(aprf)
    assert exclusion in {
        "operation_interne_banque",
        "variation_numerique_propre_banque",
    }
    aprf_copy = _prefilter_triage_result(aprf, exclusion)
    _assert_natural_analyst_copy(aprf_copy["genai_triage"]["relevance_reason"])


def test_analyst_exclusion_copy_avoids_pipeline_jargon() -> None:
    change = {
        "diff_type": "modified",
        "source_text_t1": "Ratio à 12,1 %.",
        "source_text_t2": "Ratio à 12,4 %.",
    }
    enriched = _prefilter_triage_result(change, "variation_numerique_propre_banque")
    reason = enriched["genai_triage"]["relevance_reason"]
    justification = enriched["genai_triage"]["nouvelle_idee_justification"]
    _assert_natural_analyst_copy(reason)
    _assert_natural_analyst_copy(justification)
    assert "chiffres" in reason.lower() or "pourcentages" in reason.lower()


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
