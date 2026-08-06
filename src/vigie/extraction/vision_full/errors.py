"""Classification des erreurs remontees par l'API OpenAI.

Extrait de ``vision_full_extractor.py`` sans modification.
"""

from __future__ import annotations


def _classify_openai_error(exc: Exception) -> str:
    """Classifie une erreur API OpenAI pour choisir un traitement deterministe."""
    msg = str(exc).lower()
    if (
        "could not parse the json body of your request" in msg
        or "what was sent was not valid json" in msg
        or ("json body" in msg and "not valid json" in msg)
    ):
        return "request_body_invalid"
    if "invalid schema for response_format" in msg or (
        "missing '" in msg and "response_format" in msg and "required" in msg
    ):
        return "schema_contract_invalid"
    if "json_schema" in msg and "response_format" in msg and ("not supported" in msg or "unsupported" in msg):
        return "structured_output_unsupported"
    if "rate" in msg and "limit" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "connect" in msg or "network" in msg:
        return "connection"
    if "max_tokens" in msg and ("too large" in msg or "at most" in msg):
        return "max_tokens_too_large"
    return "other"
