"""Extraction minimale par Vision : titre, resume, en-tetes, indicateurs et notes de bas de page.

L'implementation est repartie dans le sous-package ``vision_full`` : prompts,
schema, parsing, heuristiques de qualite et passes d'extraction y vivent dans
leur propre module. Ce module reste la facade publique et re-exporte tout ce qui
etait accessible avant le decoupage, y compris les symboles prives utilises par
les tests.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import replace
from typing import Any

from ..config import resolve_openai_model
from ..utils.genai import get_openai_api_key
from .vision_cache import cache_get, cache_put, get_vision_cache_dir, make_cache_key
from .vision_full.constants import (  # noqa: F401 - re-export de compatibilite
    _DEFAULT_MAX_COMPLETION_TOKENS,
    _EXTRACTION_METHOD,
    _MAX_COMPLETION_TOKENS_API_LIMIT,
    _MAX_COMPLETION_TOKENS_SAFE_FALLBACK,
    _MODEL_ROLE,
    _QUALITY_PASS_CACHE_VERSION,
    _RECROP_EXTENSION_INCREMENT,
    _RESCUE_MAX_COMPLETION_TOKENS,
    OPENAI_VISION_TIMEOUT_SECONDS,
)
from .vision_full.consensus import ConsensusMixin
from .vision_full.errors import _classify_openai_error
from .vision_full.parsing import (  # noqa: F401 - re-export de compatibilite
    _extract_embedded_schema_candidate,
    _extract_usage_metrics,
    _make_truncated_placeholder_result,
    _parse_json_response,
    _parse_vision_result,
    _preview_response_text,
    _strip_markdown_fences,
    _try_parse_truncated_result,
    _with_attempt_metadata,
)
from .vision_full.prompts import (  # noqa: F401 - re-export de compatibilite
    _CONSENSUS_PROMPT_VARIANTS,
    _DEFAULT_REFERENCE_TEXT_MAX_CHARS,
    _PROMPT_BASE,
    _PROMPT_BASE_PRECISION,
    _PROMPT_JSON_STRICT,
    _PROMPT_RESCUE_SUFFIX,
    _PROMPT_VARIANT_EXHAUSTIVE,
    _PROMPT_VARIANT_PRECISION,
    _build_content,
    _build_precision_prompt,
    _build_prompt,
    _build_repair_prompt,
)
from .vision_full.quality_grading import (  # noqa: F401 - re-export de compatibilite
    _build_result_debug_metadata,
    _candidate_quality_score,
    _collect_incompleteness_reasons,
    _finalize_selected_candidate,
    _grade_extraction_quality,
    _select_targeted_rescue_variant,
)
from .vision_full.quality_heuristics import (  # noqa: F401 - re-export de compatibilite
    _bbox_area,
    _contamination_score,
    _count_real_indicators,
    _extract_footnote_marker_ids,
    _extract_native_text_indicators,
    _has_dominant_contamination,
    _has_extracted_data,
    _has_generic_title_without_support,
    _has_multi_textual_headers,
    _has_strong_non_summary_signals,
    _is_generic_page_title,
    _is_period_like_indicator,
    _is_trivial_result,
    _is_viable_result,
    _is_weak_indicator,
    _looks_compact_textual_header,
    _looks_like_right_column_bleed_indicator,
    _looks_narrative_indicator,
    _narrative_indicator_count,
    _normalize_footnote_marker_id,
    _normalized_signal_text,
    _right_column_bleed_score,
    _structural_indicator_count,
    _token_count,
    _viable_indicator_count,
    _weak_indicator_count,
)
from .vision_full.quality_pass import QualityPassMixin
from .vision_full.result import (  # noqa: F401 - re-export de compatibilite
    VisionFullResult,
    _cache_payload_from_result,
    _result_from_cache_payload,
)
from .vision_full.schema import (  # noqa: F401 - re-export de compatibilite
    VisionFootnoteItem,
    VisionFullResponseSchema,
    VisionResponseCommonSchema,
    VisionSchemaContractError,
    _build_openai_json_schema,
    _validate_openai_strict_schema_contract,
)

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.vision_full_extractor")

__all__ = [
    "OPENAI_VISION_TIMEOUT_SECONDS",
    "VisionFootnoteItem",
    "VisionFullExtractor",
    "VisionFullResponseSchema",
    "VisionFullResult",
    "VisionResponseCommonSchema",
    "VisionSchemaContractError",
]


class VisionFullExtractor(ConsensusMixin, QualityPassMixin):
    """Extracteur Vision : une passe simple, une passe consensus, une passe qualite.

    L'implementation est repartie dans les modules du sous-package ``vision_full`` ;
    seuls l'initialisation, la validation de schema et la passe simple restent ici.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries_json: int = 2,
        use_cache: bool = False,
    ):
        """Prépare le client d'extraction Vision et ses paramètres d'exécution.

        Args:
            api_key: Clé OpenAI à utiliser. Si absente, elle est lue depuis la
                configuration locale.
            model: Nom du modèle à utiliser. Si absent, il est résolu depuis le
                rôle logique de l'extracteur.
            max_retries_json: Nombre maximal de reprises quand la réponse JSON
                n'est pas exploitable.
            use_cache: Active la lecture et l'écriture du cache d'extraction.
        """
        self._api_key = api_key or get_openai_api_key()
        self._model = str(model or "").strip() or resolve_openai_model(_MODEL_ROLE)
        self._max_retries_json = max_retries_json
        self._use_cache = use_cache
        self._client: Any = None
        self._disabled_reason: str | None = None
        self._schema_contract_checked: set[str] = set()
        self._schema_contract_error_logged = False

    @property
    @property
    def _ensure_schema_validated(self, schema: dict[str, Any] | None = None) -> None:
        """Valide le schema une seule fois et le marque comme verifie. Raises VisionSchemaContractError si invalide."""
        if "full" in self._schema_contract_checked:
            return
        schema = schema if schema is not None else _build_openai_json_schema()
        try:
            _validate_openai_strict_schema_contract(schema)
            self._schema_contract_checked.add("full")
        except VisionSchemaContractError as exc:
            self._schema_contract_checked.add("full")
            self._disabled_reason = str(exc)
            if not self._schema_contract_error_logged:
                logger.error(
                    "Vision schema contract invalid (local validation): %s",
                    exc,
                )
                self._schema_contract_error_logged = True
            raise

    def validate_schema(self) -> None:
        """Pre-valide le schema OpenAI Structured Outputs. Raises VisionSchemaContractError si invalide."""
        self._ensure_schema_validated()

    def _ensure_client(self) -> None:
        """Initialise le client OpenAI si pas encore cree."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI

            if not self._api_key:
                raise ValueError("OPENAI_API_KEY required for Vision extraction")
            self._client = OpenAI(
                api_key=self._api_key,
                timeout=OPENAI_VISION_TIMEOUT_SECONDS,
            )
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

    def extract(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        bottom_extension_used: float = 0.0,
        reference_text: str | None = None,
        max_completion_tokens_override: int | None = None,
        rescue_mode: bool = False,
        rescue_instruction: str = "",
        temperature: float = 0.0,
        prompt_override: str | None = None,
    ) -> VisionFullResult | None:
        """Extrait les indicateurs et notes de bas de page d'un recadrage de tableau.

        Args:
            crop_bytes: Octets PNG de l'image du tableau recadre.
            bank_code: Code banque pour les indices du prompt (bnc, rbc, td, etc.).
            pdf_sha: Hash du PDF pour la cle de cache (optionnel).
            page_number: Numero de page pour la cle de cache (optionnel).
            bbox_norm: Bbox normalisee pour la cle de cache (optionnel).
            vision_cfg: Surcharges de configuration (footnote_marker_type, expected_markers).
            bottom_extension_used: Extension basse deja appliquee (pour la variante de cle de cache).
            reference_text: Texte de reference pour guider l'extraction (optionnel).
            max_completion_tokens_override: Budget de sortie explicite pour cette passe d'extraction.
            rescue_mode: Activer le mode de sauvetage pour les extractions echouees.
            rescue_instruction: Instruction specifique pour le mode de sauvetage.
            temperature: Temperature de generation du modele.
            prompt_override: Prompt complet a utiliser a la place du prompt par defaut (optionnel).

        Returns:
            VisionFullResult ou None en cas d'echec.
        """
        if self._disabled_reason:
            raise VisionSchemaContractError(self._disabled_reason)

        vision_cfg = vision_cfg or {}
        cache_key = ""
        configured_max_completion_tokens = max(
            1,
            int(
                max_completion_tokens_override
                if max_completion_tokens_override is not None
                else vision_cfg.get(
                    "vision_max_completion_tokens",
                    vision_cfg.get(
                        "vision_max_completion_tokens_full",
                        _DEFAULT_MAX_COMPLETION_TOKENS,
                    ),
                )
            ),
        )
        configured_max_completion_tokens = min(configured_max_completion_tokens, _MAX_COMPLETION_TOKENS_API_LIMIT)

        if self._use_cache and not rescue_mode and pdf_sha and page_number and bbox_norm and len(bbox_norm) == 4:
            bbox_with_ext = list(bbox_norm)
            if len(bbox_with_ext) >= 4:
                bbox_with_ext[3] = min(1.0, bbox_with_ext[3] + bottom_extension_used)
            cache_key = make_cache_key(
                pdf_sha,
                page_number,
                bbox_with_ext,
                max_completion_tokens=configured_max_completion_tokens,
            )
            if cache_key:
                cache_dir = get_vision_cache_dir()
                cached = cache_get(cache_dir, cache_key)
                if cached:
                    cached_result = _result_from_cache_payload(cached)
                    if (
                        cached_result is not None
                        and _structural_indicator_count(cached_result) > 0
                    ):
                        logger.info(
                            "VisionFull cache hit: %d indicators",
                            len(cached_result.indicators),
                        )
                        return cached_result

        try:
            self._ensure_client()
        except (ImportError, ValueError) as e:
            logger.warning("VisionFullExtractor: client init failed: %s", e)
            return None

        client = self._client
        if client is None:
            return None

        try:
            from .vision_image_preprocessor import preprocess_for_vision

            preprocess_flag = vision_cfg.get("vision_preprocess")
            preprocess_enabled: bool | None = None
            if preprocess_flag is not None:
                preprocess_enabled = str(preprocess_flag).strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
            processed = preprocess_for_vision(crop_bytes, enabled=preprocess_enabled)
            image_b64 = base64.standard_b64encode(processed).decode("ascii")
        except Exception as e:
            logger.debug("Vision preprocessing failed, using raw: %s", e)
            image_b64 = base64.standard_b64encode(crop_bytes).decode("ascii")

        if prompt_override is not None:
            prompt = prompt_override
        else:
            prompt = _build_prompt(
                bank_code,
                vision_cfg,
                reference_text=reference_text,
                rescue_mode=rescue_mode,
                rescue_instruction=rescue_instruction,
            )
        max_completion_tokens = configured_max_completion_tokens
        openai_schema_full = _build_openai_json_schema()
        self._ensure_schema_validated(openai_schema_full)

        api_retry_max = int(vision_cfg.get("api_retry_max", 3))
        api_retry_backoff_ms = float(vision_cfg.get("api_retry_backoff_ms", 1000))

        def _issue_request(
            prompt_text: str,
            *,
            structured: bool,
            max_completion_tokens: int,
            label: str,
        ) -> tuple[str, str, bool, int, int | None, int | None, int | None] | None:
            """Lance un appel OpenAI avec retries et fallback structuré → texte libre."""
            local_use_structured = structured
            effective_max = max_completion_tokens
            transport_attempt = 0
            while transport_attempt <= api_retry_max:
                if transport_attempt > 0:
                    backoff_sec = (api_retry_backoff_ms / 1000.0) * (2 ** (transport_attempt - 1))
                    logger.info(
                        "Vision %s: transport retry %s/%s after %.1fs backoff",
                        label,
                        transport_attempt,
                        api_retry_max,
                        backoff_sec,
                    )
                    time.sleep(backoff_sec)
                try:
                    response_format: dict[str, Any] = (
                        openai_schema_full if local_use_structured else {"type": "json_object"}
                    )
                    response = client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {
                                "role": "user",
                                "content": _build_content(prompt_text, image_b64),
                            }
                        ],
                        response_format=response_format,
                        temperature=temperature,
                        max_completion_tokens=effective_max,
                    )
                    prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
                    return (
                        response.choices[0].message.content or "",
                        str(getattr(response.choices[0], "finish_reason", "") or ""),
                        local_use_structured,
                        effective_max,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    )
                except Exception as e:
                    err_kind = _classify_openai_error(e)
                    if err_kind == "schema_contract_invalid":
                        self._disabled_reason = f"Vision schema contract invalid: {e}"
                        if not self._schema_contract_error_logged:
                            logger.error("%s", self._disabled_reason)
                            self._schema_contract_error_logged = True
                        raise VisionSchemaContractError(self._disabled_reason) from e
                    if err_kind == "max_tokens_too_large" and effective_max > _MAX_COMPLETION_TOKENS_SAFE_FALLBACK:
                        logger.warning(
                            "Vision %s: model limits max_completion_tokens to %s; retrying with %s",
                            label,
                            effective_max,
                            _MAX_COMPLETION_TOKENS_SAFE_FALLBACK,
                        )
                        effective_max = _MAX_COMPLETION_TOKENS_SAFE_FALLBACK
                        continue
                    if local_use_structured and err_kind in (
                        "structured_output_unsupported",
                        "request_body_invalid",
                    ):
                        logger.debug(
                            "Structured Outputs unavailable for %s, falling back to json_object: %s",
                            label,
                            e,
                        )
                        local_use_structured = False
                        continue
                    if err_kind in ("rate_limit", "timeout", "connection"):
                        transport_attempt += 1
                        if transport_attempt <= api_retry_max:
                            continue
                        logger.warning(
                            "Vision %s API error (retries exhausted): %s",
                            label,
                            e,
                        )
                        return None
                    logger.warning("Vision %s API error: %s", label, e)
                    return None
            logger.warning("Vision %s: no successful API response", label)
            return None

        failure_causes: list[str] = []

        current_prompt = prompt
        max_self_healing_attempts = 2
        prev_real_indicator_count: int | None = None  # Quorum tracking

        for attempt in range(max_self_healing_attempts):
            issued = _issue_request(
                current_prompt,
                structured=True,
                max_completion_tokens=max_completion_tokens,
                label=f"full_pass_attempt_{attempt + 1}",
            )
            if issued is None:
                return None

            (
                raw_content,
                finish_reason,
                used_structured,
                effective_max_completion_tokens,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ) = issued

            if finish_reason == "length":
                failure_causes.append("vision_truncated")
                logger.warning(
                    "Vision full: response truncated (raw_len=%d)",
                    len(raw_content),
                )
                partial_result = _try_parse_truncated_result(raw_content)
                if partial_result is not None:
                    return _with_attempt_metadata(
                        partial_result,
                        requested_max_completion_tokens=effective_max_completion_tokens,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                return _make_truncated_placeholder_result(
                    requested_max_completion_tokens=effective_max_completion_tokens,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            data = _parse_json_response(raw_content)
            if data is None:
                failure_causes.append("vision_invalid_json")
                logger.info(
                    "Vision full: JSON parse failed (raw_len=%d, preview=%r)",
                    len(raw_content),
                    _preview_response_text(raw_content),
                )
                if self._max_retries_json >= 1:
                    repair_prompt = _build_repair_prompt(current_prompt, raw_content)
                    retry_issued = _issue_request(
                        repair_prompt,
                        structured=False,
                        max_completion_tokens=max_completion_tokens,
                        label="retry-json",
                    )
                    if retry_issued is not None:
                        (
                            retry_raw,
                            retry_reason,
                            _,
                            retry_effective_max_completion_tokens,
                            retry_prompt_tokens,
                            retry_completion_tokens,
                            retry_total_tokens,
                        ) = retry_issued
                        if retry_reason != "length":
                            retry_data = _parse_json_response(retry_raw)
                            if retry_data is not None:
                                result = _parse_vision_result(retry_data)
                                if result is not None:
                                    critiques = _grade_extraction_quality(result)
                                    if critiques and attempt < max_self_healing_attempts - 1:
                                        logger.warning(
                                            "Vision full: self-healing triggered after json-retry: %s",
                                            critiques,
                                        )
                                        current_prompt = (
                                            prompt
                                            + "\n\n### CRÉTIQUES DE LA TENTATIVE PRÉCÉDENTE, CORRIGEZ ###\n- "
                                            + "\n- ".join(critiques)
                                        )
                                        continue

                                    result = replace(
                                        result,
                                        vision_status="partial",
                                        warnings=list(
                                            dict.fromkeys(
                                                failure_causes
                                                + ["vision_structured_output_fallback"]
                                                + (["self_healing_failed"] if critiques else [])
                                            )
                                        ),
                                    )
                                    result = _with_attempt_metadata(
                                        result,
                                        requested_max_completion_tokens=(retry_effective_max_completion_tokens),
                                        finish_reason=retry_reason,
                                        prompt_tokens=retry_prompt_tokens,
                                        completion_tokens=retry_completion_tokens,
                                        total_tokens=retry_total_tokens,
                                    )
                                    if (
                                        self._use_cache
                                        and cache_key
                                        and _structural_indicator_count(result) > 0
                                    ):
                                        cache_dir = get_vision_cache_dir()
                                        cache_put(
                                            cache_dir,
                                            cache_key,
                                            _cache_payload_from_result(result),
                                        )
                                    return result
                logger.warning(
                    "Vision full extraction: invalid content after retry (%s)",
                    ", ".join(dict.fromkeys(failure_causes + ["vision_retry_exhausted"])),
                )
                return None

            result = _parse_vision_result(data)
            if result is None:
                failure_causes.append("vision_schema_validation_failed")
                logger.info(
                    "Vision full: schema validation failed (raw_len=%d, keys=%s)",
                    len(raw_content),
                    sorted(data.keys()),
                )
                if self._max_retries_json >= 1:
                    repair_prompt = _build_repair_prompt(current_prompt, raw_content)
                    retry_issued = _issue_request(
                        repair_prompt,
                        structured=False,
                        max_completion_tokens=max_completion_tokens,
                        label="retry-json",
                    )
                    if retry_issued is not None:
                        (
                            retry_raw,
                            retry_reason,
                            _,
                            retry_effective_max_completion_tokens,
                            retry_prompt_tokens,
                            retry_completion_tokens,
                            retry_total_tokens,
                        ) = retry_issued
                        if retry_reason != "length":
                            retry_data = _parse_json_response(retry_raw)
                            if retry_data is not None:
                                result = _parse_vision_result(retry_data)
                                if result is not None:
                                    critiques = _grade_extraction_quality(result)
                                    if critiques and attempt < max_self_healing_attempts - 1:
                                        logger.warning(
                                            "Vision full: self-healing triggered after json-retry: %s",
                                            critiques,
                                        )
                                        current_prompt = (
                                            prompt
                                            + "\n\n### CRÉTIQUES DE LA TENTATIVE PRÉCÉDENTE, CORRIGEZ ###\n- "
                                            + "\n- ".join(critiques)
                                        )
                                        continue

                                    result = replace(
                                        result,
                                        vision_status="partial",
                                        warnings=list(
                                            dict.fromkeys(
                                                failure_causes
                                                + ["vision_structured_output_fallback"]
                                                + (["self_healing_failed"] if critiques else [])
                                            )
                                        ),
                                    )
                                    result = _with_attempt_metadata(
                                        result,
                                        requested_max_completion_tokens=(retry_effective_max_completion_tokens),
                                        finish_reason=retry_reason,
                                        prompt_tokens=retry_prompt_tokens,
                                        completion_tokens=retry_completion_tokens,
                                        total_tokens=retry_total_tokens,
                                    )
                                    if (
                                        self._use_cache
                                        and cache_key
                                        and _structural_indicator_count(result) > 0
                                    ):
                                        cache_dir = get_vision_cache_dir()
                                        cache_put(
                                            cache_dir,
                                            cache_key,
                                            _cache_payload_from_result(result),
                                        )
                                    return result
                logger.warning(
                    "Vision full extraction: invalid content after retry (%s)",
                    ", ".join(dict.fromkeys(failure_causes + ["vision_retry_exhausted"])),
                )
                return None

            # --- Self-Healing Check ---
            critiques = _grade_extraction_quality(result)
            if critiques and attempt < max_self_healing_attempts - 1:
                # --- Quorum: if same indicator count as previous attempt, accept ---
                # Exception: footnote/header critiques don't change indicator count,
                # so the quorum check must not block self-healing for those.
                _footnote_or_header_critique = any(
                    kw in c for c in critiques for kw in ("footnote", "notes de bas de page", "headers", "en-têtes")
                )
                current_real_count = _count_real_indicators(result.indicators or [])
                if (
                    not _footnote_or_header_critique
                    and prev_real_indicator_count is not None
                    and abs(current_real_count - prev_real_indicator_count) <= 1
                ):
                    logger.info(
                        "Vision full: Quorum reached (attempt %d: %d real inds ≈ prev %d). "
                        "Accepting result despite critiques: %s",
                        attempt + 1,
                        current_real_count,
                        prev_real_indicator_count,
                        critiques,
                    )
                else:
                    prev_real_indicator_count = current_real_count
                    logger.warning("Vision full: self-healing triggered: %s", critiques)
                    current_prompt = (
                        prompt
                        + "\n\n### CRÉTIQUES DE LA TENTATIVE PRÉCÉDENTE, CORRIGEZ ###\n- "
                        + "\n- ".join(critiques)
                    )
                    continue

            if critiques:
                failure_causes.append("self_healing_failed")

            if used_structured is False:
                failure_causes.append("vision_structured_output_fallback")
            retry_reasons = list(result.retry_reasons or [])
            if "vision_truncated" in failure_causes and "output_budget_truncated" not in retry_reasons:
                retry_reasons.append("output_budget_truncated")
            if retry_reasons or failure_causes:
                result = replace(result, vision_status="partial")
            result = replace(
                result,
                warnings=list(dict.fromkeys(failure_causes)),
                retry_reasons=list(dict.fromkeys(retry_reasons)),
            )
            result = _with_attempt_metadata(
                result,
                requested_max_completion_tokens=effective_max_completion_tokens,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            if (
                self._use_cache
                and cache_key
                and _structural_indicator_count(result) > 0
            ):
                cache_dir = get_vision_cache_dir()
                cache_put(
                    cache_dir,
                    cache_key,
                    _cache_payload_from_result(result),
                )
            return result

        return None

    # ------------------------------------------------------------------
    # Multi-Shot Consensus extraction
    # ------------------------------------------------------------------

    _CONSENSUS_TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4)
