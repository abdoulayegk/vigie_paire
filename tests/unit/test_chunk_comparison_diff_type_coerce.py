"""Coercion of one-sided modified/unchanged LLM comparison payloads."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vigie.analyse_texte.comparaison_sections.modeles import ChunkComparisonLLMChange


def _base(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "alignment_id": "a00",
        "diff_type": "modified",
        "text_t1": "Texte précédent.",
        "text_t2": "Texte courant.",
        "change_summary": "BNC modifie la description du financement.",
        "alignment_decision": "same_disclosure",
        "alignment_confidence": "medium",
        "alignment_rationale": "Les deux passages décrivent la même divulgation.",
    }
    payload.update(overrides)
    return payload


def test_modified_with_empty_text_t1_becomes_added() -> None:
    result = ChunkComparisonLLMChange.model_validate(
        _base(
            text_t1="",
            text_t2=(
                "La priorité en matière de gestion du financement consiste à "
                "atteindre l'équilibre optimal entre les dépôts."
            ),
            change_summary=("BNC introduit une nouvelle priorité de gestion du financement."),
            alignment_decision="distinct_disclosures",
        )
    )
    assert result.diff_type == "added"
    assert result.text_t1 == ""
    assert result.text_t2.startswith("La priorité")


def test_modified_with_empty_text_t2_becomes_removed() -> None:
    result = ChunkComparisonLLMChange.model_validate(
        _base(
            text_t1="Ancienne priorité de financement retirée du rapport.",
            text_t2="",
            change_summary="BNC retire l'ancienne priorité de financement.",
            alignment_decision="distinct_disclosures",
        )
    )
    assert result.diff_type == "removed"
    assert result.text_t2 == ""
    assert result.text_t1.startswith("Ancienne priorité")


def test_unchanged_with_empty_text_t1_becomes_added() -> None:
    result = ChunkComparisonLLMChange.model_validate(
        _base(diff_type="unchanged", text_t1="", text_t2="Nouveau paragraphe.")
    )
    assert result.diff_type == "added"


def test_true_modified_with_both_texts_unchanged() -> None:
    result = ChunkComparisonLLMChange.model_validate(_base())
    assert result.diff_type == "modified"
    assert result.text_t1 == "Texte précédent."
    assert result.text_t2 == "Texte courant."


def test_both_sides_empty_still_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkComparisonLLMChange.model_validate(_base(text_t1="", text_t2=""))
