"""Passe qualite : relances ciblees et arbitrage final sur le resultat d'extraction.

Extrait de ``vision_full_extractor.py`` sans modification des corps
de methodes. Mixin consomme par ``VisionFullExtractor``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from ..page_table_locator import should_use_page_context_rescue
from ..vision_cache import cache_get, cache_put, get_vision_cache_dir, make_cache_key
from .constants import (
    _DEFAULT_MAX_COMPLETION_TOKENS,
    _MAX_COMPLETION_TOKENS_API_LIMIT,
    _QUALITY_PASS_CACHE_VERSION,
    _RECROP_EXTENSION_INCREMENT,
    _RESCUE_MAX_COMPLETION_TOKENS,
)
from .quality_grading import (
    _build_result_debug_metadata,
    _candidate_quality_score,
    _collect_incompleteness_reasons,
    _finalize_selected_candidate,
    _grade_extraction_quality,
    _select_targeted_rescue_variant,
)
from .quality_heuristics import (
    _extract_footnote_marker_ids,
    _has_extracted_data,
    _is_trivial_result,
    _is_viable_result,
    _normalize_footnote_marker_id,
    _structural_indicator_count,
)
from .result import (
    VisionFullResult,
    _cache_payload_from_result,
    _result_from_cache_payload,
)

logger = logging.getLogger("vigie.extraction.vision_full_extractor")


class QualityPassMixin:
    """Passe qualite : relances ciblees et arbitrage final sur le resultat d'extraction."""

    def extract_with_quality_pass(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        initial_bottom_extension: float = 0.0,
        initial_top_extension: float = 0.0,
        get_recrop_fn: Any = None,
        get_variant_crop_fn: Any = None,
        get_page_context_crop_fn: Any = None,
        reference_text: str | None = None,
    ) -> VisionFullResult | None:
        """Extraction avec passe de qualite deterministe.

        Flux :
        - extraction initiale avec consensus si active
        - choix d'un seul recadrage cible selon la cause d'incompletude
        - une variante de repli seulement si le sauvetage cible est inutilisable
        - inspection QA finale du recadrage effectivement selectionne
        - budget 128k de secours uniquement si une troncature est detectee

        get_recrop_fn(bottom_extension: float) -> bytes | None
        get_variant_crop_fn(bbox_override=..., bottom_extension=..., top_extension=...)
        get_page_context_crop_fn() -> dict | None
        """
        vision_cfg = vision_cfg or {}
        expected_markers = vision_cfg.get("expected_markers")
        if isinstance(expected_markers, list):
            allowed_marker_ids = {
                marker for value in expected_markers[:10] if (marker := _normalize_footnote_marker_id(value))
            }
        else:
            allowed_marker_ids = set()
        # Le profil de banque declare les marqueurs possibles. Seuls ceux
        # observes dans le tableau courant deviennent obligatoires.
        expected_set: set[str] = set()
        base_max_completion_tokens = max(
            1,
            min(
                int(
                    vision_cfg.get(
                        "vision_max_completion_tokens",
                        vision_cfg.get(
                            "vision_max_completion_tokens_full",
                            _DEFAULT_MAX_COMPLETION_TOKENS,
                        ),
                    )
                ),
                _MAX_COMPLETION_TOKENS_API_LIMIT,
            ),
        )
        rescue_enabled = bool(vision_cfg.get("vision_max_completion_tokens_rescue_enabled", False))
        rescue_max_completion_tokens = max(
            1,
            min(
                int(
                    vision_cfg.get(
                        "vision_max_completion_tokens_rescue",
                        _RESCUE_MAX_COMPLETION_TOKENS,
                    )
                ),
                _MAX_COMPLETION_TOKENS_API_LIMIT,
            ),
        )
        quality_cache_key = ""
        if (
            self._use_cache
            and pdf_sha
            and page_number
            and bbox_norm
            and len(bbox_norm) == 4
        ):
            bbox_with_context = list(bbox_norm)
            bbox_with_context[1] = max(
                0.0,
                bbox_with_context[1] - initial_top_extension,
            )
            bbox_with_context[3] = min(
                1.0,
                bbox_with_context[3] + initial_bottom_extension,
            )
            base_quality_key = make_cache_key(
                pdf_sha,
                page_number,
                bbox_with_context,
                max_completion_tokens=base_max_completion_tokens,
            )
            if base_quality_key:
                quality_cache_key = (
                    f"quality_pass_{_QUALITY_PASS_CACHE_VERSION}_"
                    f"{base_quality_key}"
                )
                cached_quality = cache_get(
                    get_vision_cache_dir(),
                    quality_cache_key,
                )
                if cached_quality:
                    cached_result = _result_from_cache_payload(cached_quality)
                    if cached_result is not None:
                        logger.info(
                            "VisionFull quality cache hit: %d indicators",
                            len(cached_result.indicators),
                        )
                        return cached_result

        def _cache_quality_result(result: VisionFullResult) -> VisionFullResult:
            """Persister la decision finale, y compris les passes de sauvetage."""
            accepted_status = str(result.extraction_status or "").strip()
            cacheable = (
                accepted_status in {"ok", "rescued"}
                and _structural_indicator_count(result) > 0
            )
            if self._use_cache and quality_cache_key and cacheable:
                cache_put(
                    get_vision_cache_dir(),
                    quality_cache_key,
                    _cache_payload_from_result(result),
                )
            return result

        def _footnote_ids(r: VisionFullResult) -> set[str]:
            """Retourne l'ensemble des identifiants de footnotes extraits du résultat."""
            return {
                _normalize_footnote_marker_id(item.get("id"))
                for item in list(r.footnotes_content or [])
                if isinstance(item, dict) and _normalize_footnote_marker_id(item.get("id"))
            }

        def _needs_recrop(result: VisionFullResult | None) -> bool:
            """Indique si le résultat porte des signaux d'incomplétude justifiant un re-crop."""
            return bool(
                _collect_incompleteness_reasons(
                    result,
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                )
            )

        def _has_truncation_signal(result: VisionFullResult | None) -> bool:
            """Indique si le résultat porte un signal de troncature (finish_reason / warnings)."""
            if result is None:
                return False
            if str(result.finish_reason or "").strip().lower() == "length":
                return True
            if "output_budget_truncated" in set(result.retry_reasons or []):
                return True
            return "vision_truncated" in {str(w).strip() for w in result.warnings or []}

        consensus_enabled = bool(vision_cfg.get("vision_consensus_enabled", False))

        def _run_pass(
            *,
            crop_bytes_for_pass: bytes,
            bottom_extension_used: float,
            bbox_for_pass: list[float] | None = None,
            rescue_mode: bool = False,
            rescue_instruction: str = "",
        ) -> VisionFullResult | None:
            """Exécute une passe complète d'extraction Vision avec rescue éventuel."""
            effective_bbox = bbox_for_pass or bbox_norm
            if consensus_enabled and not rescue_mode:
                primary = self.extract_with_consensus(
                    crop_bytes=crop_bytes_for_pass,
                    bank_code=bank_code,
                    pdf_sha=pdf_sha,
                    page_number=page_number,
                    bbox_norm=effective_bbox,
                    vision_cfg=vision_cfg,
                    bottom_extension_used=bottom_extension_used,
                    reference_text=reference_text,
                    max_completion_tokens_override=base_max_completion_tokens,
                    rescue_mode=rescue_mode,
                    rescue_instruction=rescue_instruction,
                )
            else:
                primary = self.extract(
                    crop_bytes=crop_bytes_for_pass,
                    bank_code=bank_code,
                    pdf_sha=pdf_sha,
                    page_number=page_number,
                    bbox_norm=effective_bbox,
                    vision_cfg=vision_cfg,
                    bottom_extension_used=bottom_extension_used,
                    reference_text=reference_text,
                    max_completion_tokens_override=base_max_completion_tokens,
                    rescue_mode=rescue_mode,
                    rescue_instruction=rescue_instruction,
                )
            if (
                rescue_enabled
                and rescue_max_completion_tokens > base_max_completion_tokens
                and _has_truncation_signal(primary)
            ):
                rescue = self.extract(
                    crop_bytes=crop_bytes_for_pass,
                    bank_code=bank_code,
                    pdf_sha=pdf_sha,
                    page_number=page_number,
                    bbox_norm=effective_bbox,
                    vision_cfg=vision_cfg,
                    bottom_extension_used=bottom_extension_used,
                    reference_text=reference_text,
                    max_completion_tokens_override=rescue_max_completion_tokens,
                    rescue_mode=rescue_mode,
                    rescue_instruction=rescue_instruction,
                )
                if rescue is not None:
                    return replace(rescue, rescue_used=True)
            return primary

        first = _run_pass(
            crop_bytes_for_pass=crop_bytes,
            bottom_extension_used=initial_bottom_extension,
        )

        if first is not None:
            marker_sources: list[Any] = [
                first.table_title,
                *list(first.headers or []),
                *list(first.indicators or []),
            ]
            found_markers = _extract_footnote_marker_ids(marker_sources)
            if allowed_marker_ids:
                found_markers &= allowed_marker_ids
            expected_set.update(found_markers)

        initial_is_suspect = _is_trivial_result(first, bbox_norm=bbox_norm)
        initial_rejection_reasons = _collect_incompleteness_reasons(
            first,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_set,
        )

        # Quality critiques (hiérarchie plate, footnotes orphelins, headers manquants)
        # doivent aussi bloquer l'early return pour forcer le rescue path.
        initial_quality_critiques = _grade_extraction_quality(first) if first is not None else []
        if initial_quality_critiques and not initial_rejection_reasons:
            initial_rejection_reasons.append("quality_critiques")
            logger.info(
                "Vision full: quality critiques present on initial result — forcing rescue path: %s",
                initial_quality_critiques,
            )

        # --- Dual LLM QA Inspector (Priority 1) ---
        qa_missing_str = ""
        passed_qa = False
        if first is not None and not initial_is_suspect:
            try:
                import dataclasses

                from vigie.extraction.vision_qa_inspector import (
                    VisionTableInspector,
                )

                first_dict = dataclasses.asdict(first)

                inspector = VisionTableInspector(model="gpt-4o")
                qa_result = inspector.inspect_extraction(crop_bytes, first_dict)

                if not qa_result.is_perfect:
                    initial_rejection_reasons.append("qa_inspector_failed")
                    qa_missing_str = ", ".join(qa_result.missing_elements)
                else:
                    passed_qa = True
            except Exception as e:
                logger.error("Failed to execute VisionTableInspector: %s", e)
        # ------------------------------------------

        if not initial_is_suspect and not initial_rejection_reasons:
            assert first is not None
            return _cache_quality_result(
                _build_result_debug_metadata(
                    replace(first, extraction_status="ok", qa_inspected=passed_qa),
                    acceptance_reason="initial_complete",
                    rejection_reasons=[],
                    selected_candidate_name="initial",
                    no_table_evidence_count=0,
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                )
            )

        candidates: list[tuple[str, VisionFullResult | None]] = [("initial", first)]
        candidate_crops: dict[str, bytes] = {"initial": crop_bytes}
        candidate_bottom_extensions: dict[str, float] = {"initial": initial_bottom_extension}
        candidate_geometry: dict[str, dict[str, Any]] = {
            "initial": {
                "bbox_norm": list(bbox_norm) if bbox_norm else None,
                "bbox_source": "docling",
                "bbox_confidence": None,
                "page_context_title": "",
                "page_context_continuation": None,
                "page_context_table_count": None,
            }
        }
        no_table_evidence = 0
        if first is not None and first.no_table_detected and _is_trivial_result(first, bbox_norm=bbox_norm):
            no_table_evidence += 1

        base_rescue_instruction = ""
        if "missing_body_row_labels" in initial_rejection_reasons:
            base_rescue_instruction = (
                "CRITICAL WARNING: The table has visible columns but the previous "
                "pass returned no first-column body row labels. Inspect the LEFTMOST "
                "cell of every horizontal data row. Dates or periods such as "
                "'Au 31 janvier 2025', 'Au 31 octobre 2024', 'T1 2025', or 'T2 2025' "
                "MUST be returned in indicators when the same row contains values "
                "in columns 2+. Exclude them only when they span the table as a "
                "delimiter or column sub-header."
            )
        elif "qa_inspector_failed" in initial_rejection_reasons:
            base_rescue_instruction = (
                f"CRITICAL WARNING: The rigid QA Inspector found you missed required first-column row labels or footnotes in the image: [{qa_missing_str}].\n"
                "You MUST execute the extraction again and GUARANTEE these missing FIRST-COLUMN row labels or footnotes are included. "
                "Do NOT add text from non-leftmost columns. Reread carefully line-by-line."
            )
        elif "low_density_vertical" in initial_rejection_reasons:
            base_rescue_instruction = (
                "WARNING: You failed to extract the full table height in the previous pass. "
                "Reread the entire image, line-by-line, and extract EVERY first-column row label including heavily indented sub-items. "
                "Do NOT convert text from columns 2+ into indicators."
            )
        elif initial_quality_critiques:
            base_rescue_instruction = (
                "ATTENTION: Des problèmes de qualité ont été détectés dans votre extraction précédente. "
                "Corrigez TOUS ces problèmes dans cette nouvelle tentative:\n- "
                + "\n- ".join(initial_quality_critiques)
            )

        def _append_candidate(
            name: str,
            crop_bytes_for_pass: bytes | None,
            *,
            bottom_extension_used: float,
            candidate_bbox: list[float] | None = None,
            custom_rescue_instr: str | None = None,
        ) -> VisionFullResult | None:
            """Lance une passe pour un crop alternatif et ajoute le résultat aux candidats."""
            nonlocal no_table_evidence
            if not crop_bytes_for_pass:
                return None
            candidate_crops[name] = crop_bytes_for_pass
            candidate_bottom_extensions[name] = bottom_extension_used
            candidate_geometry[name] = {
                "bbox_norm": list(candidate_bbox or bbox_norm or []),
                "bbox_source": "docling_variant" if name != "initial" else "docling",
                "bbox_confidence": None,
                "page_context_title": "",
                "page_context_continuation": None,
                "page_context_table_count": None,
            }
            result = _run_pass(
                crop_bytes_for_pass=crop_bytes_for_pass,
                bottom_extension_used=bottom_extension_used,
                bbox_for_pass=candidate_bbox,
                rescue_mode=True,
                rescue_instruction=custom_rescue_instr if custom_rescue_instr is not None else base_rescue_instruction,
            )
            candidates.append((name, result))
            if (
                result is not None
                and result.no_table_detected
                and _is_trivial_result(result, bbox_norm=candidate_bbox or bbox_norm)
            ):
                no_table_evidence += 1
            return result

        def _build_variant_crop(
            name: str,
        ) -> tuple[bytes | None, float, list[float] | None]:
            """Construire uniquement le crop demande par le diagnostic."""
            if name == "same_crop_rescue":
                return crop_bytes, initial_bottom_extension, bbox_norm

            if get_variant_crop_fn is None or not bbox_norm or len(bbox_norm) < 4:
                if name == "bottom_extended" and get_recrop_fn is not None:
                    extension = initial_bottom_extension + _RECROP_EXTENSION_INCREMENT
                    return get_recrop_fn(extension), extension, bbox_norm
                return None, initial_bottom_extension, bbox_norm

            left, top, right, bottom = [float(v) for v in bbox_norm[:4]]
            height = max(0.0, bottom - top)
            width = max(0.0, right - left)

            if name == "bottom_extended":
                extension = initial_bottom_extension + _RECROP_EXTENSION_INCREMENT
                return (
                    get_variant_crop_fn(bottom_extension=extension),
                    extension,
                    bbox_norm,
                )

            if name == "top_extended":
                return (
                    get_variant_crop_fn(
                        bottom_extension=initial_bottom_extension,
                        top_extension=(initial_top_extension + _RECROP_EXTENSION_INCREMENT),
                    ),
                    initial_bottom_extension,
                    bbox_norm,
                )

            if name == "top_trim":
                candidate_bbox = [
                    left,
                    min(bottom, top + min(height * 0.12, 0.06)),
                    right,
                    bottom,
                ]
                return (
                    get_variant_crop_fn(
                        bbox_override=candidate_bbox,
                        bottom_extension=initial_bottom_extension,
                    ),
                    initial_bottom_extension,
                    candidate_bbox,
                )

            if name == "tight_body":
                candidate_bbox = [
                    max(0.0, left + min(width * 0.015, 0.01)),
                    min(bottom, top + min(height * 0.08, 0.04)),
                    min(1.0, right - min(width * 0.015, 0.01)),
                    bottom,
                ]
                if candidate_bbox[2] <= candidate_bbox[0] or candidate_bbox[3] <= candidate_bbox[1]:
                    return None, initial_bottom_extension, bbox_norm
                extension = max(0.0, initial_bottom_extension * 0.5)
                return (
                    get_variant_crop_fn(
                        bbox_override=candidate_bbox,
                        bottom_extension=extension,
                    ),
                    extension,
                    candidate_bbox,
                )

            if name == "body_expanded":
                candidate_bbox = [
                    max(0.0, left - min(width * 0.015, 0.01)),
                    max(0.0, top - min(height * 0.08, 0.04)),
                    min(1.0, right + min(width * 0.015, 0.01)),
                    min(1.0, bottom + min(height * 0.10, 0.05)),
                ]
                return (
                    get_variant_crop_fn(
                        bbox_override=candidate_bbox,
                        bottom_extension=initial_bottom_extension,
                    ),
                    initial_bottom_extension,
                    candidate_bbox,
                )

            return None, initial_bottom_extension, bbox_norm

        target_variant = _select_targeted_rescue_variant(
            initial_rejection_reasons,
            initial_quality_critiques,
            qa_missing_str,
        )
        logger.info(
            "Vision full: targeted rescue variant=%s reasons=%s",
            target_variant,
            initial_rejection_reasons,
        )

        target_crop, target_bottom_extension, target_bbox = _build_variant_crop(target_variant)
        primary_rescue = _append_candidate(
            target_variant,
            target_crop,
            bottom_extension_used=target_bottom_extension,
            candidate_bbox=target_bbox,
        )
        same_crop_rescue = primary_rescue if target_variant == "same_crop_rescue" else None

        page_context_rescue: VisionFullResult | None = None
        primary_rescue_reasons = _collect_incompleteness_reasons(
            primary_rescue,
            bbox_norm=target_bbox or bbox_norm,
            expected_footnote_ids=expected_set,
        )
        if primary_rescue is not None and _grade_extraction_quality(primary_rescue):
            primary_rescue_reasons.append("quality_critiques")
        if "qa_inspector_failed" in initial_rejection_reasons:
            primary_rescue_reasons.append("qa_inspector_failed")

        if get_page_context_crop_fn is not None and should_use_page_context_rescue(
            primary_rescue is not None,
            primary_rescue_reasons,
        ):
            try:
                page_context = get_page_context_crop_fn()
            except Exception as exc:
                logger.warning("Vision page-context crop failed (non-fatal): %s", exc)
                page_context = None
            if isinstance(page_context, dict):
                page_crop = page_context.get("crop_bytes")
                page_bbox_raw = page_context.get("bbox_norm")
                page_bbox = (
                    [float(value) for value in page_bbox_raw]
                    if isinstance(page_bbox_raw, (list, tuple)) and len(page_bbox_raw) == 4
                    else bbox_norm
                )
                page_bottom_extension = max(
                    0.0,
                    float(page_context.get("bottom_extension", 0.0) or 0.0),
                )
                locator_confidence = float(page_context.get("confidence", 0.0) or 0.0)
                locator_instruction = (
                    "A page-level geometric locator independently corrected the "
                    "table boundaries with confidence "
                    f"{locator_confidence:.2f}. Read ONLY this corrected crop. "
                    "Extract every first-column row label and every visible footnote; "
                    "do not include neighboring narrative text or another table."
                )
                if "missing_body_row_labels" in (
                    set(initial_rejection_reasons)
                    | set(primary_rescue_reasons)
                ):
                    locator_instruction += (
                        " Dates or periods in the leftmost cell MUST be extracted "
                        "when they label body data rows with values across the same "
                        "horizontal row."
                    )
                page_context_rescue = _append_candidate(
                    "page_context_rescue",
                    page_crop if isinstance(page_crop, bytes) else None,
                    bottom_extension_used=page_bottom_extension,
                    candidate_bbox=page_bbox,
                    custom_rescue_instr=locator_instruction,
                )
                if "page_context_rescue" in candidate_geometry:
                    candidate_geometry["page_context_rescue"] = {
                        "bbox_norm": list(page_bbox or []),
                        "bbox_source": "page_context_locator",
                        "bbox_confidence": locator_confidence,
                        "page_context_title": str(
                            page_context.get("title_text") or ""
                        ).strip(),
                        "page_context_continuation": (
                            bool(page_context.get("continuation"))
                            if page_context.get("continuation") is not None
                            else None
                        ),
                        "page_context_table_count": (
                            int(page_context.get("table_count"))
                            if page_context.get("table_count") is not None
                            else None
                        ),
                    }

        if (primary_rescue is None or not _is_viable_result(primary_rescue)) and not _is_viable_result(
            page_context_rescue
        ):
            fallback_variant = "body_expanded" if target_variant == "same_crop_rescue" else "same_crop_rescue"
            fallback_crop, fallback_bottom_extension, fallback_bbox = _build_variant_crop(fallback_variant)
            fallback_result = _append_candidate(
                fallback_variant,
                fallback_crop,
                bottom_extension_used=fallback_bottom_extension,
                candidate_bbox=fallback_bbox,
            )
            if fallback_variant == "same_crop_rescue":
                same_crop_rescue = fallback_result

        usable_candidates = [item for item in candidates if item[1] is not None and _is_viable_result(item[1])]
        if usable_candidates:
            best_name, best_result = max(
                usable_candidates,
                key=lambda item: _candidate_quality_score(
                    item[1],
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                    baseline_result=first,
                ),
            )
            assert best_result is not None
            best_crop_bytes = candidate_crops.get(best_name, crop_bytes)
            best_bottom_extension = candidate_bottom_extensions.get(
                best_name,
                initial_bottom_extension,
            )
            best_geometry = dict(candidate_geometry.get(best_name, {}))
            selected_used_variant_crop = best_name not in {
                "initial",
                "same_crop_rescue",
            }

            # --- Post-rescue QA inspection ---
            # The QA Inspector only ran on `first` when it had zero rejection
            # reasons (in which case we already returned above).  Every path
            # that reaches this point means the selected candidate has NEVER
            # been QA-verified.  Run a targeted QA pass now and, if it finds
            # missing elements, do one additional rescue with a precise instruction.
            try:
                import dataclasses as _dataclasses

                from vigie.extraction.vision_qa_inspector import (
                    VisionTableInspector as _VisionTableInspector,
                )

                _best_dict = _dataclasses.asdict(best_result)
                _post_qa_inspector = _VisionTableInspector(model="gpt-4o")
                _post_qa_result = _post_qa_inspector.inspect_extraction(
                    best_crop_bytes,
                    _best_dict,
                )

                if not _post_qa_result.is_perfect and _post_qa_result.missing_elements:
                    _missing_str = ", ".join(_post_qa_result.missing_elements)
                    logger.info(
                        "Post-rescue QA: candidate '%s' is incomplete — missing: %s",
                        best_name,
                        _missing_str,
                    )
                    _targeted_instr = (
                        f"CRITICAL WARNING: The rigid QA Inspector found you missed "
                        f"required first-column row labels or footnotes: [{_missing_str}].\n"
                        "You MUST extract again and GUARANTEE every missing FIRST-COLUMN "
                        "row label and footnote is included. "
                        "Do NOT add text from non-leftmost columns. "
                        "Reread the image carefully, line-by-line, top to bottom."
                    )
                    _targeted = _run_pass(
                        crop_bytes_for_pass=best_crop_bytes,
                        bottom_extension_used=best_bottom_extension,
                        rescue_mode=True,
                        rescue_instruction=_targeted_instr,
                    )
                    if _targeted is not None and _is_viable_result(_targeted):
                        best_result = _targeted
                        best_name = "qa_targeted_rescue"
                        logger.info("Post-rescue QA: targeted rescue produced a viable result.")
                else:
                    logger.debug("Post-rescue QA: candidate '%s' passed inspection.", best_name)
            except Exception as _post_qa_exc:
                logger.error("Post-rescue QA inspection failed (non-fatal): %s", _post_qa_exc)
            # --- End post-rescue QA ---

            final_status = "ok" if best_name == "initial" and not initial_rejection_reasons else "rescued"
            selected = replace(
                best_result,
                rescue_used=best_name != "initial" or best_result.rescue_used,
                recrop_attempted=len(candidates) > 1,
                recrop_used=selected_used_variant_crop,
                recrop_failed_incomplete=False,
                extraction_status=final_status,
                selected_bbox_norm=(
                    list(best_geometry.get("bbox_norm") or [])
                    if len(list(best_geometry.get("bbox_norm") or [])) == 4
                    else None
                ),
                bbox_source=str(best_geometry.get("bbox_source") or "docling"),
                bbox_confidence=best_geometry.get("bbox_confidence"),
                page_context_title=str(
                    best_geometry.get("page_context_title") or ""
                ),
                page_context_continuation=best_geometry.get(
                    "page_context_continuation"
                ),
                page_context_table_count=best_geometry.get(
                    "page_context_table_count"
                ),
            )
            finalized = _finalize_selected_candidate(
                selected,
                best_name=best_name,
                initial_rejection_reasons=initial_rejection_reasons,
                no_table_evidence_count=no_table_evidence,
                bbox_norm=bbox_norm,
                expected_footnote_ids=expected_set,
            )
            if finalized is not None:
                return _cache_quality_result(finalized)

        if no_table_evidence >= 2:
            fallback = (
                same_crop_rescue
                or first
                or VisionFullResult(
                    table_title="",
                    table_summary="",
                    headers=[],
                    indicators=[],
                    footnotes_content=[],
                    no_table_detected=True,
                )
            )
            _override_status = "rescued" if _has_extracted_data(fallback) else "confirmed_no_table"
            _override_reason = (
                "data_richness_override_repeated_no_table"
                if _override_status == "rescued"
                else "confirmed_no_table_after_repeated_no_table_evidence"
            )
            return _cache_quality_result(
                _build_result_debug_metadata(
                    replace(
                        fallback,
                        no_table_detected=(_override_status == "confirmed_no_table"),
                        recrop_attempted=len(candidates) > 1,
                        recrop_used=False,
                        recrop_failed_incomplete=False,
                        extraction_status=_override_status,
                    ),
                    acceptance_reason=_override_reason,
                    rejection_reasons=initial_rejection_reasons
                    or ["no_table_evidence"],
                    selected_candidate_name=(
                        "same_crop_rescue" if same_crop_rescue else "initial"
                    ),
                    no_table_evidence_count=no_table_evidence,
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                )
            )

        fallback: VisionFullResult = same_crop_rescue or first  # type: ignore[assignment]
        if fallback is None:
            fallback = VisionFullResult(
                table_title="",
                table_summary="",
                headers=[],
                indicators=[],
                footnotes_content=[],
            )
        fallback_candidate_name = "same_crop_rescue" if same_crop_rescue else "initial"
        # --- Gate the fallback through the same acceptance validation as
        # viable candidates so that empty/trivial tables are never returned
        # with extraction_status="suspect_unresolved".
        fallback_for_gate = replace(
            fallback,
            recrop_attempted=len(candidates) > 1,
            recrop_used=False,
            recrop_failed_incomplete=True,
            extraction_status="suspect_unresolved",
        )
        finalized_fallback = _finalize_selected_candidate(
            fallback_for_gate,
            best_name=fallback_candidate_name,
            initial_rejection_reasons=initial_rejection_reasons,
            no_table_evidence_count=no_table_evidence,
            bbox_norm=bbox_norm,
            expected_footnote_ids=expected_set,
        )
        if finalized_fallback is not None:
            return _cache_quality_result(finalized_fallback)

        # Fallback rejected by acceptance gate → demote to confirmed_no_table.
        fallback_rejection_reasons = list(
            dict.fromkeys(
                initial_rejection_reasons
                + _collect_incompleteness_reasons(
                    fallback,
                    bbox_norm=bbox_norm,
                    expected_footnote_ids=expected_set,
                )
            )
        )
        _override_status_fb = "rescued" if _has_extracted_data(fallback) else "confirmed_no_table"
        _override_reason_fb = (
            "data_richness_override_rescue_exhaustion"
            if _override_status_fb == "rescued"
            else "confirmed_no_table_after_rescue_exhaustion"
        )
        return _cache_quality_result(
            _build_result_debug_metadata(
                replace(
                    fallback,
                    no_table_detected=(
                        _override_status_fb == "confirmed_no_table"
                    ),
                    recrop_attempted=len(candidates) > 1,
                    recrop_used=False,
                    recrop_failed_incomplete=True,
                    extraction_status=_override_status_fb,
                ),
                acceptance_reason=_override_reason_fb,
                rejection_reasons=fallback_rejection_reasons
                or ["rescue_exhausted"],
                selected_candidate_name=fallback_candidate_name,
                no_table_evidence_count=no_table_evidence,
                bbox_norm=bbox_norm,
                expected_footnote_ids=expected_set,
            )
        )
