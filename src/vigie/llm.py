"""Passerelle LLM unique pour Vigie (OpenAI public ou Azure OpenAI).

Centralise la creation de client, la resolution des modeles/deploiements,
le sampling (reasoning_effort pour gpt-5.*, temperature legacy sinon) et
les appels embeddings. Les prompts restent dans les modules metier.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from vigie.support.config.loader import _resolve_config_path, load_config

logger = logging.getLogger(__name__)


def _openai():
    """Import paresseux pour ne pas charger openai au simple import de vigie.llm."""
    import openai as openai_sdk  # noqa: PLC0415

    return openai_sdk


ProviderName = Literal["openai", "azure"]
ReasoningProfile = Literal["extraction", "default", "locator"]
ReasoningEffort = Literal["low", "medium", "high"]

_DEFAULT_CHAT_MODEL = "gpt-5.4"
_DEFAULT_EMBEDDING_SMALL = "text-embedding-3-small"
_DEFAULT_EMBEDDING_LARGE = "text-embedding-3-large"

_DEFAULT_MODELS: dict[str, str] = {
    "chat": _DEFAULT_CHAT_MODEL,
    "extraction_primary": _DEFAULT_CHAT_MODEL,
    "default_genai": _DEFAULT_CHAT_MODEL,
    "vision_qa": _DEFAULT_CHAT_MODEL,
    "vision_toc": _DEFAULT_CHAT_MODEL,
    "embedding_small": _DEFAULT_EMBEDDING_SMALL,
    "embedding_large": _DEFAULT_EMBEDDING_LARGE,
}

_ROLE_ALIASES: dict[str, str] = {
    "extraction_primary": "chat",
    "default_genai": "chat",
    "vision_qa": "chat",
    "vision_toc": "chat",
}

_DEFAULT_REASONING: dict[str, ReasoningEffort] = {
    "extraction": "high",
    "default": "medium",
    "locator": "low",
}

_MODEL_ENV_OVERRIDES: dict[str, str] = {
    "chat": "OPENAI_MODEL_CHAT",
    "extraction_primary": "OPENAI_MODEL_EXTRACTION_PRIMARY",
    "default_genai": "OPENAI_MODEL_DEFAULT_GENAI",
    "embedding_small": "OPENAI_MODEL_EMBEDDING_SMALL",
    "embedding_large": "OPENAI_MODEL_EMBEDDING_LARGE",
}

_AZURE_DEPLOYMENT_ENV: dict[str, str] = {
    "chat": "AZURE_OPENAI_DEPLOYMENT_CHAT",
    "extraction_primary": "AZURE_OPENAI_DEPLOYMENT_CHAT",
    "default_genai": "AZURE_OPENAI_DEPLOYMENT_CHAT",
    "vision_qa": "AZURE_OPENAI_DEPLOYMENT_CHAT",
    "vision_toc": "AZURE_OPENAI_DEPLOYMENT_CHAT",
    "embedding_small": "AZURE_OPENAI_DEPLOYMENT_EMBEDDING_SMALL",
    "embedding_large": "AZURE_OPENAI_DEPLOYMENT_EMBEDDING_LARGE",
}

_REASONING_ENV: dict[str, str] = {
    "extraction": "LLM_REASONING_EXTRACTION",
    "default": "LLM_REASONING_DEFAULT",
    "locator": "LLM_REASONING_LOCATOR",
}

_REASONING_MODEL_RE = re.compile(r"^gpt-5(?:\.|$|-)", flags=re.IGNORECASE)
_EMBEDDING_BATCH_SIZE = 96
_DEFAULT_OPENAI_TIMEOUT = 300.0
_DEFAULT_COMPARISON_TIMEOUT = 120.0

_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_PATH_CACHE: Path | None = None


def _get_openai_api_key() -> str | None:
    raw = os.getenv("OPENAI_API_KEY") or ""
    return raw.strip() if isinstance(raw, str) else None


def _extract_usage_metrics(response: Any) -> tuple[int, int, int]:
    from vigie.comparaison.io import _extract_usage_metrics as _io_extract_usage_metrics  # noqa: PLC0415

    return _io_extract_usage_metrics(response)


def _normalize_role(role: str) -> str:
    key = str(role or "").strip().lower()
    return _ROLE_ALIASES.get(key, key)


def _load_llm_config(config_path: str | Path | None = None) -> dict[str, Any]:
    global _CONFIG_CACHE, _CONFIG_PATH_CACHE
    path = _resolve_config_path(config_path or "configs/bank_profiles.yaml")
    if _CONFIG_CACHE is not None and _CONFIG_PATH_CACHE == path:
        return _CONFIG_CACHE
    cfg: dict[str, Any] = {}
    if path.exists():
        try:
            cfg = load_config(path)
        except Exception:
            cfg = {}
    _CONFIG_CACHE = cfg
    _CONFIG_PATH_CACHE = path
    return cfg


def get_provider(config_path: str | Path | None = None) -> ProviderName:
    """Retourne le provider LLM actif (openai par defaut)."""
    env_value = str(os.getenv("LLM_PROVIDER") or "").strip().lower()
    if env_value in {"openai", "azure"}:
        return env_value  # type: ignore[return-value]
    cfg = _load_llm_config(config_path)
    llm_block = cfg.get("llm")
    if isinstance(llm_block, dict):
        provider = str(llm_block.get("provider") or "").strip().lower()
        if provider in {"openai", "azure"}:
            return provider  # type: ignore[return-value]
    return "openai"


def get_azure_api_version(config_path: str | Path | None = None) -> str:
    env_value = str(os.getenv("AZURE_OPENAI_API_VERSION") or "").strip()
    if env_value:
        return env_value
    cfg = _load_llm_config(config_path)
    llm_block = cfg.get("llm")
    if isinstance(llm_block, dict):
        version = str(llm_block.get("api_version") or "").strip()
        if version:
            return version
    return "2024-10-21"


def _azure_credentials() -> tuple[str, str]:
    api_key = str(os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    return api_key, endpoint


def is_configured(config_path: str | Path | None = None) -> bool:
    """Indique si le provider LLM actif est correctement configure."""
    if get_provider(config_path) == "azure":
        api_key, endpoint = _azure_credentials()
        return bool(api_key and endpoint)
    return bool(_get_openai_api_key())


def require_configured(config_path: str | Path | None = None) -> None:
    """Leve si le provider LLM actif n'est pas configure."""
    if is_configured(config_path):
        return
    if get_provider(config_path) == "azure":
        raise RuntimeError("Azure OpenAI non configure: definir AZURE_OPENAI_API_KEY et AZURE_OPENAI_ENDPOINT.")
    raise RuntimeError("OPENAI_API_KEY absent: le pipeline LLM ne peut pas s'executer.")


def is_reasoning_model(model: str) -> bool:
    """Detecte les modeles de raisonnement (gpt-5.*) qui n'utilisent pas temperature."""
    return bool(_REASONING_MODEL_RE.match(str(model or "").strip()))


def resolve_model(role: str, config_path: str | Path | None = None) -> str:
    """Resout un role metier vers un identifiant OpenAI ou un deploiement Azure."""
    original_key = str(role or "").strip().lower()
    canonical = _normalize_role(role)
    if canonical not in _DEFAULT_MODELS:
        known = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(f"Unknown LLM role '{role}'. Known roles: {known}")

    provider = get_provider(config_path)

    if provider == "azure":
        for azure_key in (original_key, canonical):
            env_name = _AZURE_DEPLOYMENT_ENV.get(azure_key)
            if env_name:
                env_value = str(os.getenv(env_name) or "").strip()
                if env_value:
                    return env_value
        cfg = _load_llm_config(config_path)
        azure_block = cfg.get("llm_azure_deployments")
        if isinstance(azure_block, dict):
            value = azure_block.get(canonical) or azure_block.get("chat")
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise RuntimeError(
            f"Azure OpenAI deployment missing for role '{role}'. "
            f"Set {_AZURE_DEPLOYMENT_ENV.get(canonical, 'AZURE_OPENAI_DEPLOYMENT_CHAT')}."
        )

    for env_key in (original_key, canonical):
        env_name = _MODEL_ENV_OVERRIDES.get(env_key)
        if env_name:
            env_value = str(os.getenv(env_name) or "").strip()
            if env_value:
                return env_value

    cfg = _load_llm_config(config_path)
    models_block = cfg.get("llm_models")
    if isinstance(models_block, dict):
        for key in (canonical, "chat", role):
            value = models_block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        legacy = models_block.get("extraction_primary") if canonical == "chat" else None
        if isinstance(legacy, str) and legacy.strip():
            return legacy.strip()
        legacy_genai = models_block.get("default_genai") if canonical == "chat" else None
        if isinstance(legacy_genai, str) and legacy_genai.strip():
            return legacy_genai.strip()

    return _DEFAULT_MODELS[canonical]


def resolve_reasoning_effort(
    profile: ReasoningProfile | str,
    config_path: str | Path | None = None,
) -> ReasoningEffort:
    """Resout le niveau de raisonnement pour un profil d'appel."""
    key = str(profile or "default").strip().lower()
    if key not in _DEFAULT_REASONING:
        key = "default"

    env_name = _REASONING_ENV.get(key)
    if env_name:
        env_value = str(os.getenv(env_name) or "").strip().lower()
        if env_value in {"low", "medium", "high"}:
            return env_value  # type: ignore[return-value]

    cfg = _load_llm_config(config_path)
    reasoning_block = cfg.get("llm_reasoning")
    if isinstance(reasoning_block, dict):
        value = str(reasoning_block.get(key) or "").strip().lower()
        if value in {"low", "medium", "high"}:
            return value  # type: ignore[return-value]

    return _DEFAULT_REASONING[key]  # type: ignore[return-value]


def build_completion_kwargs(
    *,
    model: str,
    profile: ReasoningProfile | str = "default",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Construit les kwargs de sampling (reasoning_effort ou temperature legacy)."""
    if is_reasoning_model(model):
        return {"reasoning_effort": resolve_reasoning_effort(profile, config_path=config_path)}
    return {"temperature": 0.0}


def get_client(
    *,
    timeout: float | None = None,
    max_retries: int = 1,
    config_path: str | Path | None = None,
) -> Any:
    """Instancie un client OpenAI ou AzureOpenAI synchronise."""
    provider = get_provider(config_path)
    effective_timeout = float(timeout if timeout is not None else _DEFAULT_OPENAI_TIMEOUT)

    if provider == "azure":
        api_key, endpoint = _azure_credentials()
        if not api_key or not endpoint:
            raise RuntimeError("Azure OpenAI is not configured (AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT).")
        return _openai().AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=get_azure_api_version(config_path),
            timeout=effective_timeout,
            max_retries=max_retries,
        )

    api_key = _get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY absent: le pipeline LLM ne peut pas s'executer.")
    return _openai().OpenAI(api_key=api_key, timeout=effective_timeout, max_retries=max_retries)


def get_async_client(
    *,
    timeout: float | None = None,
    max_retries: int = 1,
    config_path: str | Path | None = None,
) -> Any:
    """Instancie un client OpenAI ou AzureOpenAI asynchrone."""
    provider = get_provider(config_path)
    effective_timeout = float(timeout if timeout is not None else _DEFAULT_OPENAI_TIMEOUT)

    if provider == "azure":
        api_key, endpoint = _azure_credentials()
        if not api_key or not endpoint:
            raise RuntimeError("Azure OpenAI is not configured (AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT).")
        return _openai().AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=get_azure_api_version(config_path),
            timeout=effective_timeout,
            max_retries=max_retries,
        )

    api_key = _get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY absent: le pipeline LLM ne peut pas s'executer.")
    return _openai().AsyncOpenAI(api_key=api_key, timeout=effective_timeout, max_retries=max_retries)


def embed(
    texts: list[str],
    *,
    role: str = "embedding_small",
    client: Any | None = None,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "embeddings",
    config_path: str | Path | None = None,
) -> list[list[float]]:
    """Encode une liste de textes via l'API embeddings."""
    if not texts:
        return []
    model = resolve_model(role, config_path=config_path)
    llm_client = client or get_client(timeout=_DEFAULT_COMPARISON_TIMEOUT, config_path=config_path)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
        response = llm_client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend([list(item.embedding) for item in ordered])
        if usage_recorder is not None:
            prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
            usage_recorder.append(
                {
                    "model": model,
                    "call_kind": call_kind,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )
    return vectors


_T_StructuredModel = TypeVar("_T_StructuredModel", bound=BaseModel)


def complete_openai_json(
    *,
    model: str,
    messages: list[dict[str, Any]],
    profile: ReasoningProfile | str = "default",
    max_completion_tokens: int | None = None,
    api_retry_max: int = 2,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison",
    response_model: type[_T_StructuredModel] | None = None,
    client: Any | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Appel JSON ou structured pour la comparaison tableaux."""
    llm_client = client or get_client(
        timeout=_DEFAULT_COMPARISON_TIMEOUT,
        max_retries=0,
        config_path=config_path,
    )
    sampling = build_completion_kwargs(model=model, profile=profile, config_path=config_path)
    last_error: Exception | None = None
    use_structured = response_model is not None

    for attempt in range(api_retry_max + 1):
        if attempt > 0:
            time.sleep(1.5 * (2 ** (attempt - 1)))
        try:
            if use_structured:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    **sampling,
                    "response_format": response_model,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = llm_client.beta.chat.completions.parse(**kwargs)
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Structured Output parsing returned None")
                data = parsed.model_dump()
            else:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    **sampling,
                    "response_format": {"type": "json_object"},
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = llm_client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("OpenAI response is not a JSON object")
            if usage_recorder is not None:
                prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
                usage_recorder.append(
                    {
                        "model": model,
                        "call_kind": call_kind,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
            return data
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = (
                "rate" in message
                and "limit" in message
                or "timeout" in message
                or "timed out" in message
                or "connection" in message
                or "connect" in message
            )
            if not retryable or attempt >= api_retry_max:
                break
    raise RuntimeError(f"OpenAI comparison call failed: {last_error}")


async def complete_json_async(
    client: Any,
    *,
    system: str,
    user: str,
    model: str | None = None,
    profile: ReasoningProfile | str = "default",
    max_tokens: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Appel asynchrone JSON pour le triage GenAI."""
    model_name = str(model or resolve_model("chat", config_path=config_path))
    sampling = build_completion_kwargs(model=model_name, profile=profile, config_path=config_path)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **sampling,
        "response_format": {"type": "json_object"},
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    try:
        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"GenAI async JSON call failed: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("GenAI async JSON call returned a non-object payload")
    return data


def chat_completions_create(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    profile: ReasoningProfile | str = "default",
    response_format: Any | None = None,
    max_completion_tokens: int | None = None,
    config_path: str | Path | None = None,
) -> Any:
    """Appel chat.completions.create bas niveau avec sampling centralise."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **build_completion_kwargs(model=model, profile=profile, config_path=config_path),
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    return client.chat.completions.create(**kwargs)


def structured_completions_parse(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[_T_StructuredModel],
    profile: ReasoningProfile | str = "default",
    max_tokens: int | None = None,
    config_path: str | Path | None = None,
) -> _T_StructuredModel:
    """Appel beta.chat.completions.parse bas niveau avec sampling centralise."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        **build_completion_kwargs(model=model, profile=profile, config_path=config_path),
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_tokens)
    response = client.beta.chat.completions.parse(**kwargs)
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


__all__ = [
    "ProviderName",
    "ReasoningProfile",
    "ReasoningEffort",
    "build_completion_kwargs",
    "chat_completions_create",
    "complete_json_async",
    "complete_openai_json",
    "embed",
    "get_async_client",
    "get_client",
    "get_provider",
    "is_configured",
    "is_reasoning_model",
    "require_configured",
    "resolve_model",
    "resolve_reasoning_effort",
    "structured_completions_parse",
]
