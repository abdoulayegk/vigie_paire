"""Tests unitaires pour la transcription sémantique des diagrammes par GPT-4o Vision."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vigilance.text_analysis.vision_diagram_transcriber import (
    transcribe_diagram_image_bytes_with_vision,
)


def test_transcribe_diagram_image_bytes_with_vision_empty_bytes():
    assert transcribe_diagram_image_bytes_with_vision(b"") == ""


@patch("vigilance.text_analysis.vision_diagram_transcriber.get_openai_api_key", return_value="dummy-key")
@patch("openai.OpenAI")
def test_transcribe_diagram_image_bytes_with_vision_success(mock_openai_cls, mock_api_key):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="### Diagramme : Architecture de gestion\n- Ligne 1 : Unités"))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    result = transcribe_diagram_image_bytes_with_vision(b"fake-png-bytes", diagram_title_context="Architecture")
    assert "Diagramme : Architecture" in result
    assert "Ligne 1" in result
    mock_client.chat.completions.create.assert_called_once()
