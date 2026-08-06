from __future__ import annotations

from typing import Any

from vigie.analyse_texte.chunk_alignment import ChunkAlignment
from vigie.analyse_texte.chunking import TextChunk
from vigie.analyse_texte.comparaison_sections import (
    ChunkComparisonLLMResponse,
    _compare_section_texts,
    _compare_texts_single_call,
)
from vigie.analyse_texte.comparaison_sections import comparaison_section
from vigie.analyse_texte.comparaison_sections import execution_llm


def test_single_call_names_bank_and_forbids_period_labels_as_subject(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_structured_completion(
        client,
        *,
        model,
        messages,
        response_format,
        max_retries,
        validation_retry_message,
    ):
        captured["messages"] = messages
        captured["validation_retry_message"] = validation_retry_message
        return ChunkComparisonLLMResponse(
            changes=[
                {
                    "alignment_id": "a00",
                    "diff_type": "modified",
                    "text_t1": "Ancienne description du risque.",
                    "text_t2": "Nouvelle description du risque.",
                    "change_summary": "CIBC modifie sa description du risque.",
                    "alignment_decision": "same_disclosure",
                    "alignment_confidence": "high",
                    "alignment_rationale": "Les passages décrivent la même divulgation.",
                }
            ]
        )

    monkeypatch.setattr(
        execution_llm,
        "_call_structured_completion_with_correction",
        fake_structured_completion,
    )

    _compare_texts_single_call(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        heading_label="Gestion des risques",
        heading_slug="gestion_des_risques",
        text_t1="[a00 | matched_weak]\nAncienne description du risque.",
        text_t2="[a00 | matched_weak]\nNouvelle description du risque.",
        idx_offset=0,
        bank_code="cibc",
    )

    prompt = "\n".join(message["content"] for message in captured["messages"])
    assert "Banque analysée : CIBC" in prompt
    assert ("Chaque change_summary doit commencer par « CIBC » suivi d'un verbe d'action direct") in prompt
    assert (
        "« rapport précédent » et « rapport courant » peuvent seulement servir de contexte de comparaison"
    ) in prompt
    assert "jamais être le sujet grammatical de change_summary" in prompt
    assert "N'inscris aucun trimestre" in prompt
    assert "T1, T2" in prompt

    retry_message = captured["validation_retry_message"]
    assert "commencer exactement par \"CIBC \" suivi d'un verbe d'action direct" in retry_message
    assert "n'utilise jamais rapport courant, rapport précédent, T1, T2" in retry_message


def test_compare_section_texts_propagates_bank_code_through_internal_chain(
    monkeypatch,
) -> None:
    received_bank_codes: list[str] = []

    def fake_chunk_subsection_bodies(**kwargs) -> list[TextChunk]:
        body = str(kwargs["body"])
        suffix = "previous" if "précédente" in body else "current"
        return [
            TextChunk(
                chunk_id=f"risk_{suffix}",
                kind="paragraph",
                text=body,
                subsection_heading=str(kwargs["heading"]),
                hierarchy_path=f"Gestion des risques > {kwargs['heading']}",
                order=0,
            )
        ]

    def fake_align_chunks(
        chunks_t1,
        chunks_t2,
        *,
        client,
        embedding_model,
    ) -> list[ChunkAlignment]:
        return [
            ChunkAlignment(
                alignment_id="a00",
                alignment_type="matched_weak",
                chunk_t1=chunks_t1[0],
                chunk_t2=chunks_t2[0],
                similarity_score=0.5,
                candidates_t1_for_t2=[],
                candidates_t2_for_t1=[],
                reason="test",
            )
        ]

    def fake_single_call(**kwargs) -> list[dict[str, Any]]:
        received_bank_codes.append(kwargs["bank_code"])
        return []

    monkeypatch.setattr(
        comparaison_section,
        "_chunk_subsection_bodies",
        fake_chunk_subsection_bodies,
    )
    monkeypatch.setattr(comparaison_section, "_align_chunks_hybrid", fake_align_chunks)
    monkeypatch.setattr(execution_llm, "_compare_texts_single_call", fake_single_call)

    result = _compare_section_texts(
        client=object(),
        model="gpt-4o",
        section_key="gestion_risques",
        text_t1="### Risque opérationnel\n\nVersion précédente du cadre de risque.",
        text_t2="### Risque opérationnel\n\nVersion courante du cadre de risque.",
        bank_code="cibc",
    )

    assert result == []
    assert received_bank_codes == ["cibc"]
