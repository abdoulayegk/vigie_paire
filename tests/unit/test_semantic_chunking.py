from __future__ import annotations

import pytest

from vigie.analyse_texte.chunking import _chunk_subsection_text
from vigie.analyse_texte.semantic_chunking import (
    SemanticChunkingError,
    SemanticPartitionResponse,
    SemanticSentenceGroup,
    _partition_with_llm,
    _semantic_partition_paragraphs,
)


def _four_sentence_paragraph(prefix: str = "La Banque") -> str:
    return " ".join(
        [
            f"{prefix} applique une méthode de mesure des risques.",
            "Cette méthode couvre les expositions importantes du portefeuille.",
            "Les paramètres sont encadrés par la politique de gestion.",
            "La gouvernance surveille enfin la bonne application du cadre.",
        ]
    )


def test_simple_paragraph_does_not_request_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Les embeddings ne doivent pas être appelés")

    monkeypatch.setattr("vigie.analyse_texte.semantic_chunking._embed_texts", fail_if_called)
    text = "La Banque applique une politique de gestion du capital."

    chunks = _chunk_subsection_text(text, client=object())

    assert [chunk.text for chunk in chunks] == [text]


def test_complex_paragraph_has_no_fallback_when_embeddings_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_embeddings(*args, **kwargs):
        raise ConnectionError("service indisponible")

    monkeypatch.setattr("vigie.analyse_texte.semantic_chunking._embed_texts", fail_embeddings)

    with pytest.raises(SemanticChunkingError, match="Échec des embeddings sans fallback"):
        _chunk_subsection_text(_four_sentence_paragraph(), client=object())


def test_ambiguous_partition_has_no_fallback_when_llm_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._embed_texts",
        lambda client, texts, model: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.78] * (len(sentences) - 1),
    )

    def fail_llm(*args, **kwargs):
        raise TimeoutError("LLM indisponible")

    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._call_structured_completion_with_correction",
        fail_llm,
    )

    with pytest.raises(SemanticChunkingError, match="Échec du partitionnement LLM sans fallback"):
        _chunk_subsection_text(_four_sentence_paragraph(), client=object())


def test_invalid_llm_partition_is_rejected_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._embed_texts",
        lambda client, texts, model: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.78] * (len(sentences) - 1),
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._call_structured_completion_with_correction",
        lambda *args, **kwargs: SemanticPartitionResponse(groups=[SemanticSentenceGroup(start=1, end=2)]),
    )

    with pytest.raises(SemanticChunkingError, match="dernière phrase n'est pas couverte"):
        _chunk_subsection_text(_four_sentence_paragraph(), client=object())


def test_embeddings_are_batched_once_and_identical_sentences_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def record_embeddings(client, texts, model):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("vigie.analyse_texte.semantic_chunking._embed_texts", record_embeddings)
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.90] * (len(sentences) - 1),
    )
    paragraph = _four_sentence_paragraph()

    partitions = _semantic_partition_paragraphs([paragraph, paragraph], client=object())

    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert partitions == [[paragraph], [paragraph]]


def test_deterministic_single_sentence_partition_is_treated_as_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._embed_texts",
        lambda client, texts, model: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.50] * (len(sentences) - 1),
    )
    llm_calls: list[int] = []

    def coherent_partition(*, sentences, **kwargs):
        llm_calls.append(len(sentences))
        return [(0, len(sentences))]

    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._partition_with_llm",
        coherent_partition,
    )

    partitions = _semantic_partition_paragraphs(
        [_four_sentence_paragraph()],
        client=object(),
    )

    assert llm_calls == [4]
    assert partitions == [[_four_sentence_paragraph()]]


def test_llm_overfragmentation_is_corrected_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sentences = [f"Phrase {index} sur le même cadre réglementaire." for index in range(1, 7)]
    responses = iter(
        [
            SemanticPartitionResponse(groups=[SemanticSentenceGroup(start=index, end=index) for index in range(1, 7)]),
            SemanticPartitionResponse(
                groups=[
                    SemanticSentenceGroup(start=1, end=3),
                    SemanticSentenceGroup(start=4, end=6),
                ]
            ),
        ]
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._call_structured_completion_with_correction",
        lambda *args, **kwargs: next(responses),
    )

    ranges = _partition_with_llm(
        client=object(),
        model="gpt-4o",
        sentences=sentences,
        scores=[0.78] * 5,
    )

    assert ranges == [(0, 3), (3, 6)]


def test_llm_repeated_overfragmentation_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentences = [f"Phrase {index} sur le même cadre réglementaire." for index in range(1, 7)]
    fragmented = SemanticPartitionResponse(
        groups=[SemanticSentenceGroup(start=index, end=index) for index in range(1, 7)]
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._call_structured_completion_with_correction",
        lambda *args, **kwargs: fragmented,
    )

    with pytest.raises(SemanticChunkingError, match="toujours sur-fragmentée"):
        _partition_with_llm(
            client=object(),
            model="gpt-4o",
            sentences=sentences,
            scores=[0.78] * 5,
        )


def test_numbers_are_neutralized_for_similarity_but_preserved_in_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded_texts: list[str] = []

    def record_embeddings(client, texts, model):
        embedded_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("vigie.analyse_texte.semantic_chunking._embed_texts", record_embeddings)
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.90] * (len(sentences) - 1),
    )
    paragraph = (
        "En 2024, le ratio atteint 13,2 %. "
        "La Banque conserve 525 M$ de capital. "
        "En 2025, le cadre demeure applicable. "
        "La politique couvre toujours les mêmes risques."
    )

    chunks = _chunk_subsection_text(paragraph, client=object())

    assert [chunk.text for chunk in chunks] == [paragraph]
    assert "2024" in chunks[0].text
    assert "525 M$" in chunks[0].text
    assert all("2024" not in text and "525" not in text for text in embedded_texts)


def test_docling_private_bullets_become_atomic_list_items() -> None:
    text = " Première règle de gouvernance.\n Deuxième règle de surveillance."

    chunks = _chunk_subsection_text(text)

    assert len(chunks) == 2
    assert [chunk.kind for chunk in chunks] == ["list_item", "list_item"]
    assert [chunk.atomic_marker for chunk in chunks] == ["-", "-"]
    assert chunks[0].comparison_text == "Première règle de gouvernance."
    assert chunks[1].comparison_text == "Deuxième règle de surveillance."


def test_hard_word_limit_splits_at_sentence_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._embed_texts",
        lambda client, texts, model: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.90] * (len(sentences) - 1),
    )
    sentence = "Contrôle " + " ".join(["continu"] * 99) + "."
    paragraph = " ".join([sentence] * 4)

    chunks = _chunk_subsection_text(paragraph, client=object())

    assert len(chunks) >= 2
    assert all(len(chunk.text.split()) <= 240 for chunk in chunks)


def test_bnc_bale_partition_is_stable_when_reform_disclosure_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._embed_texts",
        lambda client, texts, model: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "vigie.analyse_texte.semantic_chunking._continuity_scores",
        lambda sentences, normalized, embeddings: [0.78] * (len(sentences) - 1),
    )

    def bnc_ranges(*, sentences, **kwargs):
        if any("révisions apportées par le BSIF" in sentence for sentence in sentences):
            return [(0, 1), (1, 2), (2, 6), (6, 10)]
        return [(0, 1), (1, 5), (5, 9)]

    monkeypatch.setattr("vigie.analyse_texte.semantic_chunking._partition_with_llm", bnc_ranges)
    common_a = "Comme l'exige l'Accord de Bâle, l'actif pondéré est calculé pour les risques de crédit, de marché et opérationnel."
    reform = "Certaines révisions apportées par le BSIF à ses règles de fonds propres ont pris effet en 2023."
    common_c = [
        "La Banque utilise les approches de notation interne pour le risque de crédit.",
        "L'approche NI fondation vise certains types d'expositions.",
        "L'approche NI avancée couvre les autres expositions.",
        "Ces approches encadrent l'évaluation du risque de crédit.",
    ]
    common_d = [
        "Selon l'approche NI avancée, la Banque estime ses paramètres de risque.",
        "Les paramètres sont assujettis à des limites plancher.",
        "Certains portefeuilles suivent l'approche standardisée révisée.",
        "Cette approche complète le cadre de calcul réglementaire.",
    ]
    previous = " ".join([common_a, reform, *common_c, *common_d])
    current = " ".join([common_a, *common_c, *common_d])

    chunks_2024 = _chunk_subsection_text(previous, client=object())
    chunks_2025 = _chunk_subsection_text(current, client=object())

    assert len(chunks_2024) == 4
    assert len(chunks_2025) == 3
    assert "révisions apportées par le BSIF" in chunks_2024[1].text
    assert all("révisions apportées par le BSIF" not in chunk.text for chunk in chunks_2025)
    assert chunks_2024[0].text == chunks_2025[0].text
    assert chunks_2024[2].text == chunks_2025[1].text
    assert chunks_2024[3].text == chunks_2025[2].text
