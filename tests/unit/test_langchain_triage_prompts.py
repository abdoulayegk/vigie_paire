"""Tests unitaires pour les objets ChatPromptTemplate LangChain."""

from __future__ import annotations

from vigilance.triage_prompts import get_triage_chat_prompt_template


def test_triage_chat_prompt_template() -> None:
    prompt_template = get_triage_chat_prompt_template()
    messages = prompt_template.format_messages(change_description="Modifications dans le tableau 40")

    assert len(messages) == 2
    assert messages[0].type == "system"
    assert "AMF" in messages[0].content
    assert messages[1].type == "human"
    assert "Modifications dans le tableau 40" in messages[1].content
