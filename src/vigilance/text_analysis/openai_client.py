"""Composants modulaires du pipeline texte."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from vigilance.text_analysis.constants import (
    _MODEL_MAX_OUTPUT_TOKENS,
    _OPENAI_TIMEOUT_SECONDS,
    _TRIAGE_LENGTH_RETRIES,
    _TRIAGE_TRANSPORT_RETRIES,
)
from vigilance.utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)


_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_BATCH_SIZE = 96


def _embed_texts(client: Any, texts: list[str], model: str = _DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Encode une liste de textes via l'API embeddings OpenAI."""
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend([list(item.embedding) for item in ordered])
    return embeddings


def _build_openai_client():
    """Instancie le client OpenAI avec la clé API du projet.

    Lève ``RuntimeError`` si la clé est absente — le pipeline texte ne peut pas
    fonctionner sans accès à l'API OpenAI.
    """
    from openai import OpenAI

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY absent: le pipeline texte GPT-first ne peut pas s'exécuter.")
    return OpenAI(api_key=api_key, timeout=_OPENAI_TIMEOUT_SECONDS, max_retries=1)


def _strip_markdown_fences(text: str) -> str:
    """Retire les clotures Markdown et isole l'objet JSON si present."""
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]
    return stripped


def _parse_json_object_response(raw: str) -> dict[str, Any]:
    """Parse un objet JSON reponse OpenAI et valide son type racine."""
    cleaned = _strip_markdown_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("OpenAI response is not a JSON object")
    return data


def _preview_response_text(raw: str, limit: int = 500) -> str:
    """Retourne un apercu compact de la reponse brute pour les logs."""
    text = (raw or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head} ... {tail}"


def _truncate_prompt_text(text: str, limit: int) -> str:
    """Borne un champ texte envoye a GPT tout en conservant le debut et la fin."""
    value = str(text or "").strip()
    if limit <= 0 or len(value) <= limit:
        return value
    marker = "\n[... texte tronque pour le triage ...]\n"
    available = max(limit - len(marker), 0)
    head_len = max(int(available * 0.7), 0)
    tail_len = max(available - head_len, 0)
    head = value[:head_len].rstrip() if head_len else ""
    tail = value[-tail_len:].lstrip() if tail_len else ""
    return f"{head}{marker}{tail}"


def _classify_openai_transport_error(exc: Exception) -> str:
    """Classe les erreurs transitoires OpenAI detectables sans dependance SDK."""
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    if "length limit" in message or "finish_reason=length" in message or "lengthfinishreason" in name:
        return "length_limit"
    if "timeout" in message or "timed out" in message or "timeout" in name:
        return "timeout"
    if "rate" in message and "limit" in message:
        return "rate_limit"
    if any(token in message for token in ("connection", "connect", "network")):
        return "connection"
    if any(token in name for token in ("connection", "connect", "network")):
        return "connection"
    return "other"


def _build_json_repair_messages(raw_response: str) -> list[dict[str, str]]:
    """Construit un mini-prompt de reparation pour un JSON mal forme."""
    return [
        {
            "role": "system",
            "content": (
                "Tu reçois une tentative de réponse JSON invalide. "
                "Réécris-la en un objet JSON valide uniquement, sans markdown, "
                "sans commentaire, sans texte avant ou après."
            ),
        },
        {
            "role": "user",
            "content": (
                "Répare cet objet JSON sans changer sa structure ni son sens. "
                "Si une chaîne est tronquée, ferme-la et ferme correctement les objets/listes.\n\n"
                f"{raw_response[:12000]}"
            ),
        },
    ]


def _max_output_tokens_for_model(model: str, fallback: int = 16_384) -> int:
    """Retourne le plafond de sortie connu du modele quand il est connu.

    Ce plafond n'est pas utilise comme limite par defaut dans le flux texte:
    les appels normaux laissent le modele s'arreter naturellement. Il sert
    seulement de borne de retry quand un appel explicitement cappe finit en
    ``finish_reason="length"``.
    """
    normalized = str(model or "").strip()
    for pattern, limit in _MODEL_MAX_OUTPUT_TOKENS:
        if pattern.match(normalized):
            return limit
    return fallback


def _call_json_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Execute un appel JSON OpenAI robuste pour le flux texte.

    Politique du pipeline texte:
    - pas de limite de completion explicite par defaut, afin de privilegier un
      arret naturel et de reduire les JSON tronques;
    - retry au plafond connu du modele seulement si un plafond explicite plus
      bas a ete fourni et que la reponse finit en ``length``;
    - tentative de reparation du JSON avant de lever une erreur finale.

    Returns:
        Objet JSON racine valide sous forme de dictionnaire.
    """
    model_max_tokens = _max_output_tokens_for_model(model)
    initial_max_tokens = None if max_tokens is None else min(int(max_tokens), model_max_tokens)

    def _request(request_messages: list[dict[str, Any]], token_budget: int | None) -> tuple[str, str | None]:
        """Exécute un appel brut à l'API OpenAI et retourne ``(contenu, finish_reason)``."""
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if token_budget is not None:
            request_kwargs["max_completion_tokens"] = token_budget

        response = client.chat.completions.create(
            **request_kwargs,
        )
        choice = response.choices[0]
        payload = choice.message.content or "{}"
        finish_reason = getattr(choice, "finish_reason", None)
        return payload, finish_reason

    raw_payload, finish_reason = _request(messages, initial_max_tokens)
    try:
        return _parse_json_object_response(raw_payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "OpenAI returned invalid JSON (finish_reason=%s): %s | preview=%s",
            finish_reason,
            exc,
            _preview_response_text(raw_payload),
        )

    if finish_reason == "length" and initial_max_tokens is not None and initial_max_tokens < model_max_tokens:
        retry_max_tokens = model_max_tokens
        raw_retry, retry_finish_reason = _request(messages, retry_max_tokens)
        try:
            logger.warning(
                "Retrying OpenAI JSON call after truncation with max_completion_tokens=%s (model_max=%s)",
                retry_max_tokens,
                model_max_tokens,
            )
            return _parse_json_object_response(raw_retry)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Retry after truncation still returned invalid JSON (finish_reason=%s): %s | preview=%s",
                retry_finish_reason,
                exc,
                _preview_response_text(raw_retry),
            )
            raw_payload = raw_retry
            finish_reason = retry_finish_reason
    elif finish_reason == "length":
        logger.warning(
            "OpenAI response hit the effective output ceiling (%s) and remained truncated",
            model_max_tokens,
        )

    repair_messages = _build_json_repair_messages(raw_payload)
    raw_repair, repair_finish_reason = _request(repair_messages, None)
    try:
        return _parse_json_object_response(raw_repair)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "OpenAI JSON parse failed after repair attempt "
            f"(finish_reason={repair_finish_reason or finish_reason or 'unknown'}): {exc}. "
            f"Preview: {_preview_response_text(raw_repair)}"
        ) from exc


_T_StructuredModel = TypeVar("_T_StructuredModel", bound=BaseModel)


def _append_concise_triage_retry_message(
    messages: list[dict[str, Any]],
    *,
    content: str | None = None,
) -> list[dict[str, Any]]:
    """Ajoute une consigne de concision apres une sortie structuree tronquee."""
    default_content = (
        "La réponse précédente a dépassé la limite de sortie du modèle. "
        "Renvoie le même batch complet, mais avec une rédaction beaucoup "
        "plus concise. Contraintes strictes de longueur : explanation en "
        "3 phrases courtes; nouvelle_idee_justification entre 220 et "
        "450 caractères; justification_posture entre 80 et 220 caractères "
        "si obligatoire; impact_it_justification entre 80 et 180 caractères "
        "si obligatoire. Ne répète pas la taxonomie, ne cite pas de longs "
        "passages et ne fournis aucun commentaire hors schéma."
    )
    return list(messages) + [
        {
            "role": "user",
            "content": content or default_content,
        }
    ]


def _call_structured_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[_T_StructuredModel],
    max_tokens: int | None = None,
    request_timeout: float | None = None,
) -> _T_StructuredModel:
    """Appel OpenAI à sortie structurée garantie par schéma Pydantic.

    Utilise ``client.beta.chat.completions.parse()`` qui fournit la sortie
    GPT-4o conforme au schéma JSON dérivé du modèle Pydantic passé en
    ``response_format``. La désérialisation et les ``model_validator`` de
    Pydantic s'exécutent côté SDK : si un invariant transversal du modèle
    est violé, ``parse()`` lève directement.

    Aucun fallback silencieux : tout refus, troncature ou réponse vide est
    converti en ``RuntimeError`` explicite avec contexte d'audit. Les
    erreurs de validation Pydantic remontent en ``pydantic.ValidationError``
    (à attraper par l'appelant qui souhaite faire un retry correctif).
    """
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": 0.0,
    }
    if max_tokens is not None:
        request_kwargs["max_completion_tokens"] = int(max_tokens)
    if request_timeout is not None:
        request_kwargs["timeout"] = float(request_timeout)

    response = client.beta.chat.completions.parse(**request_kwargs)
    choice = response.choices[0]
    message = choice.message

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise RuntimeError(f"OpenAI structured completion refused by model: {refusal}")

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise RuntimeError(
            f"OpenAI structured completion truncated (finish_reason=length, max_completion_tokens={max_tokens})"
        )

    parsed = getattr(message, "parsed", None)
    if parsed is None:
        raise RuntimeError(
            f"OpenAI structured completion returned no parsed payload (finish_reason={finish_reason or 'unknown'})"
        )

    return parsed


def _call_structured_completion_with_correction(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[_T_StructuredModel],
    max_tokens: int | None = None,
    max_retries: int = 1,
    max_transport_retries: int = _TRIAGE_TRANSPORT_RETRIES,
    max_length_retries: int = _TRIAGE_LENGTH_RETRIES,
    validation_retry_message: str | None = None,
    length_retry_message: str | None = None,
    request_timeout: float | None = None,
) -> _T_StructuredModel:
    """Appel structuré avec retry correctif borné sur ``ValidationError``.

    Sémantique :
    - Une ``ValidationError`` Pydantic signifie que GPT a respecté le schéma
      JSON mais violé un invariant transversal (ex : ``revue_prioritaire`` sans
      ``MAJEUR``). Ce type d'erreur est potentiellement corrigeable par
      re-prompt, donc on retry jusqu'à ``max_retries`` fois en injectant le
      détail de l'erreur dans la conversation pour permettre l'auto-correction.
      Si le modèle renvoie exactement le même payload invalide après correction,
      la relance suivante est supprimée puisqu'elle serait déterministe et
      redondante.
    - Les erreurs de sortie tronquée par limite de longueur sont retentées une
      fois avec une consigne de concision. Les autres ``RuntimeError`` (refus
      du modèle, payload vide) remontent immédiatement.

    Au-delà de ``max_retries``, la dernière ``ValidationError`` est propagée.
    """
    current_messages = list(messages)
    validation_attempt = 0
    transport_attempts = 0
    length_attempts = 0
    previous_validation_payload_fingerprint: str | None = None
    while validation_attempt <= max_retries:
        try:
            if request_timeout is None:
                return _call_structured_completion(
                    client,
                    model=model,
                    messages=current_messages,
                    response_format=response_format,
                    max_tokens=max_tokens,
                )
            return _call_structured_completion(
                client,
                model=model,
                messages=current_messages,
                response_format=response_format,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
        except RuntimeError as exc:
            err_kind = _classify_openai_transport_error(exc)
            if err_kind == "length_limit":
                length_attempts += 1
                if length_attempts > max_length_retries:
                    raise
                logger.warning(
                    "Triage structured completion reached output length limit; retrying with concise-output instruction. Error: %s",
                    exc,
                )
                current_messages = _append_concise_triage_retry_message(
                    current_messages,
                    content=length_retry_message,
                )
                continue
            if err_kind not in {"timeout", "rate_limit", "connection"}:
                raise
            transport_attempts += 1
            if transport_attempts > max_transport_retries:
                raise
            delay_seconds = min(2 ** (transport_attempts - 1), 4)
            logger.warning(
                "Triage structured completion %s on transport attempt %d/%d; retrying in %ss. Error: %s",
                err_kind,
                transport_attempts,
                max_transport_retries,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)
        except ValidationError as exc:
            validation_errors = exc.errors(include_input=True)
            validation_detail = json.dumps(
                validation_errors,
                ensure_ascii=False,
                default=str,
            )
            validation_payload_fingerprint = json.dumps(
                [error.get("input") for error in validation_errors],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if (
                previous_validation_payload_fingerprint is not None
                and validation_payload_fingerprint
                == previous_validation_payload_fingerprint
            ):
                logger.error(
                    "Triage validation returned an identical invalid payload; "
                    "skipping redundant corrective retry. Details: %s",
                    validation_detail,
                )
                raise
            if validation_attempt >= max_retries:
                raise
            previous_validation_payload_fingerprint = (
                validation_payload_fingerprint
            )
            validation_attempt += 1
            logger.warning(
                "Triage validation failed on attempt %d/%d, retrying with correction. Details: %s",
                validation_attempt,
                max_retries + 1,
                validation_detail,
            )
            default_validation_message = (
                "Ta réponse précédente a échoué la validation du schéma "
                "ou des invariants AMF. Détail de l'erreur :\n"
                f"{validation_detail}\n\n"
                "Corrige TOUS les invariants violés et renvoie le batch "
                "COMPLET (tous les change_index) en respectant strictement "
                "le schéma. Rappel des invariants stricts : "
                "nouvelle_idee_justification est TOUJOURS obligatoire, "
                "≥ 3 phrases complètes ET ≥ 200 caractères au total, "
                "rédigée comme une note d'analyste avec les rubriques "
                "exactes séparées par \\n\\n. Tu dois inclure ces cinq "
                "libellés EXACTS, avec deux-points, dans cet ordre : "
                "Nouvel élément à surveiller :, Sujet détecté :, "
                "Ce qui change :, Pertinence métier :, "
                "Point de surveillance :, "
                "commençant par 'OUI' ou 'NON' selon nouvelle_idee ; "
                "themes_amf est facultatif lorsque is_relevant=true ; "
                "is_relevant=false exige themes_amf=[] + "
                "exclusion_reason renseigné + nouvelle_idee=false + "
                "impact_level=MINEUR + action_requise='aucune' + "
                "explanation vide (mais justification OBLIGATOIRE expliquant "
                "pourquoi le changement n'est pas une nouvelle idée) ; "
                "action_requise='revue_prioritaire' exige impact_level='MAJEUR'."
            )
            current_messages = current_messages + [
                {
                    "role": "user",
                    "content": (
                        f"{validation_retry_message or default_validation_message}\n\n"
                        f"Détail de l'erreur :\n{validation_detail}"
                        if validation_retry_message
                        else default_validation_message
                    ),
                }
            ]
        except Exception as exc:
            err_kind = _classify_openai_transport_error(exc)
            if err_kind == "length_limit":
                length_attempts += 1
                if length_attempts > max_length_retries:
                    raise
                logger.warning(
                    "Triage structured completion reached output length limit; retrying with concise-output instruction. Error: %s",
                    exc,
                )
                current_messages = _append_concise_triage_retry_message(
                    current_messages,
                    content=length_retry_message,
                )
                continue
            if err_kind not in {"timeout", "rate_limit", "connection"}:
                raise
            transport_attempts += 1
            if transport_attempts > max_transport_retries:
                raise
            delay_seconds = min(2 ** (transport_attempts - 1), 4)
            logger.warning(
                "Triage structured completion %s on transport attempt %d/%d; retrying in %ss. Error: %s",
                err_kind,
                transport_attempts,
                max_transport_retries,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)
    raise RuntimeError("unreachable: retry loop exited without return or raise")
