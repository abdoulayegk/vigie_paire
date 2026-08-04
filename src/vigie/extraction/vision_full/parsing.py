"""Lecture des reponses Vision : extraction du JSON, reparation et troncatures.

Extrait de ``vision_full_extractor.py`` sans modification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from .constants import _EXTRACTION_METHOD
from .result import VisionFullResult
from .schema import VisionFullResponseSchema

logger = logging.getLogger("vigie.extraction.vision_full_extractor")

_FULL_RESPONSE_KEYS = frozenset(
    {
        "table_title",
        "table_summary",
        "headers",
        "indicators",
        "footnotes_content",
        "no_table_detected",
    }
)

_FULL_REQUIRED_KEYS = frozenset({"table_summary", "indicators"})

def _strip_markdown_fences(text: str) -> str:
    """Retire les balises markdown de la reponse GPT et localise les limites JSON."""
    stripped = text.strip()

    # Étape 1 : Retirer les balises markdown si présentes
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    # Étape 2 : Chercher l'objet JSON (accolades)
    # L'API Vision / JSON mode retourne toujours un objet (dictionnaire) dans ce contexte
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]

    return stripped


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """Parse le JSON d'une reponse. Retourne None en cas d'echec."""
    try:
        cleaned = _strip_markdown_fences(raw)
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _preview_response_text(raw: str, limit: int = 500) -> str:
    """Retourne un apercu compact de la reponse adapte aux logs."""
    text = (raw or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head} ... {tail}"


def _extract_usage_metrics(response: Any) -> tuple[int | None, int | None, int | None]:
    """Extraction best-effort de l'utilisation de tokens a partir des reponses OpenAI."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    def _coerce_int(value: Any) -> int | None:
        """Convertit ``value`` en entier, retourne ``None`` si la conversion échoue."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return (
        _coerce_int(getattr(usage, "prompt_tokens", None)),
        _coerce_int(getattr(usage, "completion_tokens", None)),
        _coerce_int(getattr(usage, "total_tokens", None)),
    )


def _with_attempt_metadata(
    result: VisionFullResult,
    *,
    requested_max_completion_tokens: int,
    finish_reason: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    rescue_used: bool = False,
) -> VisionFullResult:
    """Retourne un resultat enrichi des metadonnees de la tentative."""
    return replace(
        result,
        requested_max_completion_tokens=requested_max_completion_tokens,
        finish_reason=finish_reason or None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        rescue_used=rescue_used,
    )


def _make_truncated_placeholder_result(
    *,
    requested_max_completion_tokens: int,
    finish_reason: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> VisionFullResult:
    """Retourne un resultat partiel minimal pour que la logique de sauvetage de niveau superieur puisse retenter avec plus de budget."""
    return VisionFullResult(
        table_title="",
        table_summary="",
        headers=[],
        indicators=[],
        footnotes_content=[],
        extraction_method=_EXTRACTION_METHOD,
        vision_status="partial",
        warnings=["vision_truncated"],
        retry_reasons=["output_budget_truncated"],
        requested_max_completion_tokens=requested_max_completion_tokens,
        finish_reason=finish_reason or None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


_FULL_RESPONSE_KEYS = frozenset(
    {
        "table_title",
        "table_summary",
        "headers",
        "indicators",
        "footnotes_content",
        "no_table_detected",
    }
)
_FULL_REQUIRED_KEYS = frozenset({"table_summary", "indicators"})


def _extract_embedded_schema_candidate(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Retourne le payload imbrique le plus probable correspondant au schema Vision complet attendu."""
    if not isinstance(raw, dict):
        return None
    response_keys = _FULL_RESPONSE_KEYS
    required_keys = _FULL_REQUIRED_KEYS
    raw_keys = set(raw.keys())
    if required_keys.issubset(raw_keys):
        return raw

    best_candidate: dict[str, Any] | None = None
    best_score = 0
    queue: list[dict[str, Any]] = [raw]
    seen_ids: set[int] = set()

    while queue:
        candidate = queue.pop(0)
        obj_id = id(candidate)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        keys = set(candidate.keys())
        score = len(keys & response_keys)
        if required_keys.issubset(keys):
            return candidate
        if score > best_score:
            best_candidate = candidate
            best_score = score

        for value in candidate.values():
            if isinstance(value, dict):
                queue.append(value)

    if best_candidate is not None and best_score >= 2:
        return best_candidate
    return None


def _parse_vision_result(
    raw: str | dict[str, Any],
) -> VisionFullResult | None:
    """Parse et valide le JSON en VisionFullResult via Pydantic. Retourne None en cas d'erreur de validation."""
    try:
        if isinstance(raw, dict):
            validated = VisionFullResponseSchema.model_validate(raw)
        else:
            validated = VisionFullResponseSchema.model_validate_json(raw)
    except Exception as e:
        if isinstance(raw, dict):
            candidate = _extract_embedded_schema_candidate(raw)
            if candidate is not None and candidate is not raw:
                try:
                    validated = VisionFullResponseSchema.model_validate(candidate)
                    logger.info(
                        "Vision response recovered from nested wrapper keys: %s",
                        list(raw.keys())[:3],
                    )
                except Exception:
                    logger.warning(
                        "Vision response validation failed (Pydantic schema error): %s",
                        e,
                    )
                    return None
            else:
                logger.warning("Vision response validation failed (Pydantic schema error): %s", e)
                return None
        else:
            logger.warning("Vision response validation failed (Pydantic schema error): %s", e)
            return None

    footnotes_ordered: list[dict[str, str]] = [
        {"id": str(item.id).strip(), "text": str(item.text).strip()}
        for item in validated.footnotes_content
        if str(item.id).strip() and str(item.text).strip()
    ]
    indicators_ordered = [str(item).rstrip() for item in validated.indicators if str(item).rstrip()]

    return VisionFullResult(
        table_title=validated.table_title or "",
        table_summary=validated.table_summary or "",
        headers=validated.headers or [],
        indicators=indicators_ordered,
        footnotes_content=footnotes_ordered,
        no_table_detected=bool(validated.no_table_detected),
        extraction_method=_EXTRACTION_METHOD,
        vision_status="ok",
        warnings=[],
    )


def _try_parse_truncated_result(raw_content: str) -> VisionFullResult | None:
    """Parsing best-effort de JSON tronque pour le contrat d'extraction minimale."""
    data = _parse_json_response(raw_content)
    if not data or not isinstance(data, dict):
        return None
    indicators_raw = data.get("indicators")
    if indicators_raw is None:
        return None
    if not isinstance(indicators_raw, list):
        return None
    indicators_ordered: list[str] = [str(item).strip() for item in indicators_raw if str(item).strip()]
    footnotes_content: list[dict[str, str]] = []
    try:
        fn_raw = data.get("footnotes_content")
        if isinstance(fn_raw, list):
            for item in fn_raw:
                if not isinstance(item, dict):
                    continue
                marker = str(item.get("id") or item.get("marker") or item.get("ref") or "").strip()
                text = str(item.get("text") or item.get("value") or "").strip()
                if marker and text:
                    footnotes_content.append({"id": marker, "text": text})
        elif isinstance(fn_raw, dict):
            for k, v in fn_raw.items():
                marker = str(k).strip()
                text = str(v).strip()
                if marker and text:
                    footnotes_content.append({"id": marker, "text": text})
    except Exception:
        footnotes_content = []

    return VisionFullResult(
        table_title=str(data.get("table_title") or "").strip(),
        table_summary=str(data.get("table_summary") or "").strip(),
        headers=[str(x).strip() for x in data.get("headers") or []],
        indicators=indicators_ordered,
        footnotes_content=footnotes_content,
        no_table_detected=bool(data.get("no_table_detected", False)),
        extraction_method=_EXTRACTION_METHOD,
        vision_status="partial",
        warnings=["vision_truncated"],
        retry_reasons=["output_budget_truncated"],
    )
