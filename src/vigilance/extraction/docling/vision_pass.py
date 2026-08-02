"""Extraction Vision d'un tableau unique, avec relances et arbitrage.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

import logging
from typing import Any

from ...utils.indicator_cleaner import normalize_indicator_for_comparison
from ...utils.rbc_table_signals import classify_rbc_title_reliability
from .models import ExtractedTable

logger = logging.getLogger("vigilance.extraction.docling_processor")


class VisionPassMixin:
    """Extraction Vision d'un tableau unique, avec relances et arbitrage."""

    def _vision_extract_one_table(
        self,
        item: tuple[int, int, list[float] | None, str, str | None],
        shared: dict[str, Any],
    ) -> tuple[int, ExtractedTable, int]:
        """Extraire un tableau via Vision (recadrage + appel API).

        Args:
            item: Tuple (index, page, bbox, table_id, texte_reference).
            shared: Dictionnaire partage contenant le contexte d'extraction.

        Returns:
            Tuple (index, ExtractedTable, numero_page).
        """
        idx, page_num, table_bbox, table_id, reference_text = item
        pdf_path = shared["pdf_path"]
        bank_code = shared["bank_code"]
        quarter = shared["quarter"]
        year = shared["year"]
        pdf_sha = shared["pdf_sha"]
        vision_extraction_cfg = shared["vision_extraction_cfg"]
        bottom_extension_footnotes = shared["bottom_extension_footnotes"]
        top_extension_title = shared["top_extension_title"]
        horizontal_padding = shared["horizontal_padding"]
        vision_extractor = shared["vision_extractor"]
        page_table_locator = shared.get("page_table_locator")
        schema_failure_flag = shared["schema_failure_flag"]
        vision_schema_error_cls = shared["vision_schema_error_cls"]
        schema_failure_policy = shared["schema_failure_policy"]
        labels_only = shared["labels_only"]
        vision_crop_dpi: int = shared.get("vision_crop_dpi", 300)
        vision_preprocess: bool | None = shared.get("vision_preprocess")
        vision_model_name: str | None = shared.get("vision_model_name")

        vision_status_str = "failed"
        warnings_list: list[str] = []
        title = ""
        table_number: str | None = None
        title_clean: str | None = None
        table_summary: str | None = None
        title_raw: str | None = None
        out_headers: list[str] = []
        out_rows: list[list[str]] = []
        indicators_raw_text: list[str] = []
        indicators: list[str] = []
        indicators_spatial_raw: list[Any] = []
        footnotes: list[dict] = []
        vision_extraction_attempted = False
        vision_schema_contract_failed = False
        vision_extraction_disabled_reason: str | None = None
        crop_reject_reason: str | None = None
        bbox_sanity_profile: dict[str, Any] | None = None
        vision_result: Any = None
        extraction_status = "ok"
        page_context_observation: dict[str, Any] = dict(
            shared.get("page_context_seed", {}).get(idx, {})
        )
        seeded_original_bbox = page_context_observation.get("bbox_original")
        if "bbox_original" in page_context_observation:
            original_table_bbox = (
                list(seeded_original_bbox)
                if isinstance(seeded_original_bbox, (list, tuple))
                and len(seeded_original_bbox) == 4
                else None
            )
        else:
            original_table_bbox = list(table_bbox) if table_bbox else None
        final_table_bbox = list(table_bbox) if table_bbox else None

        if vision_extractor and table_bbox and len(table_bbox) == 4:
            vision_extraction_attempted = True
            try:
                from ...utils.pdf_crop import crop_table_region_to_bytes

                if schema_failure_flag[0]:
                    vision_extraction_attempted = False
                    vision_schema_contract_failed = True
                    warnings_list = ["Vision disabled after schema contract failure"]
                    vision_extraction_disabled_reason = shared.get("vision_extraction_disabled_reason")
                else:
                    from ...utils.pdf_crop import is_bbox_sane

                    sane, crop_reject_reason, bbox_sanity_profile = is_bbox_sane(table_bbox, vision_extraction_cfg)
                    if not sane:
                        vision_extraction_attempted = True
                        vision_status_str = "failed"
                        warnings_list = [f"bbox sanity gate: {crop_reject_reason or 'rejected'}; Vision skipped"]
                        if crop_reject_reason == "bbox_near_full_page":
                            extraction_status = "suspect_unresolved"
                            warnings_list.append(
                                str(
                                    page_context_observation.get(
                                        "bbox_verification_reason"
                                    )
                                    or "near_full_page_verification_unresolved"
                                )
                            )
                    else:
                        # Dynamic crop extensions based on page layout context
                        from ...utils.page_layout_context import compute_dynamic_extensions

                        page_table_map = shared.get("page_table_map", {})
                        dyn_top, dyn_bottom = compute_dynamic_extensions(
                            table_idx=idx,
                            page_num=page_num,
                            table_bbox=table_bbox,
                            page_table_map=page_table_map,
                            default_bottom=bottom_extension_footnotes,
                            default_top=top_extension_title,
                        )
                        initial_bottom_ext = dyn_bottom
                        top_extension_title = dyn_top

                        def _recrop(ext: float) -> bytes:
                            """Re-crope la région du tableau avec une extension verticale ajustée."""
                            return crop_table_region_to_bytes(
                                str(pdf_path),
                                page_num,
                                table_bbox,
                                bottom_extension=ext,
                                top_extension=top_extension_title,
                                horizontal_padding=horizontal_padding,
                                dpi=vision_crop_dpi,
                            )

                        def _render_variant_crop(
                            *,
                            bbox_override: list[float] | None = None,
                            bottom_extension: float | None = None,
                            top_extension: float | None = None,
                        ) -> tuple[bytes, list[float], float, float]:
                            """Rendre une variante et retourner sa geometrie effective."""
                            from ...utils.page_layout_context import (
                                clamp_variant_crop_to_neighbors,
                            )

                            safe_bbox, safe_bottom, safe_top = clamp_variant_crop_to_neighbors(
                                table_idx=idx,
                                page_num=page_num,
                                table_bbox=table_bbox,
                                page_table_map=page_table_map,
                                bbox_override=bbox_override,
                                bottom_extension=(
                                    initial_bottom_ext if bottom_extension is None else float(bottom_extension)
                                ),
                                top_extension=(top_extension_title if top_extension is None else float(top_extension)),
                            )
                            rendered = crop_table_region_to_bytes(
                                str(pdf_path),
                                page_num,
                                safe_bbox,
                                bottom_extension=safe_bottom,
                                top_extension=safe_top,
                                horizontal_padding=horizontal_padding,
                                dpi=vision_crop_dpi,
                            )
                            return rendered, safe_bbox, safe_bottom, safe_top

                        def _variant_crop(
                            *,
                            bbox_override: list[float] | None = None,
                            bottom_extension: float | None = None,
                            top_extension: float | None = None,
                        ) -> bytes:
                            """Variante de crop bornee par les tableaux voisins."""
                            rendered, _bbox, _bottom, _top = _render_variant_crop(
                                bbox_override=bbox_override,
                                bottom_extension=bottom_extension,
                                top_extension=top_extension,
                            )
                            return rendered

                        def _page_context_crop() -> dict[str, Any] | None:
                            """Localiser la page puis rendre uniquement le tableau corrige."""
                            if page_table_locator is None:
                                return None
                            if (
                                page_context_observation.get("bbox_source")
                                == "page_context_inventory_conflict_preserved_docling"
                            ):
                                return None
                            from .page_table_locator import build_page_table_crop_plan
                            from .pdf_preview import render_pdf_page

                            page_image = render_pdf_page(
                                str(pdf_path),
                                page_num,
                                scale=1.5,
                                format="png",
                            )
                            if not page_image:
                                return None
                            layout = page_table_locator.locate_page(
                                page_image,
                                page_num,
                                pdf_sha=pdf_sha,
                            )
                            if layout is None:
                                return None
                            plan = build_page_table_crop_plan(
                                layout,
                                table_bbox,
                                min_confidence=page_table_locator.min_confidence,
                            )
                            if plan is None:
                                logger.info(
                                    "Vision page-context: no safe match for page=%s table=%s",
                                    page_num,
                                    idx,
                                )
                                return None
                            page_context_observation.update(
                                {
                                    "bbox_norm": list(plan.bbox_norm),
                                    "bbox_source": (
                                        page_context_observation.get("bbox_source")
                                        or "page_context_locator"
                                    ),
                                    "confidence": plan.confidence,
                                    "title_text": plan.title_text,
                                    "continuation": plan.continuation,
                                    "table_count": plan.table_count,
                                }
                            )
                            rendered, safe_bbox, safe_bottom, safe_top = _render_variant_crop(
                                bbox_override=list(plan.bbox_norm),
                                bottom_extension=plan.bottom_extension,
                                top_extension=plan.top_extension,
                            )
                            if not rendered:
                                return None
                            logger.info(
                                "Vision page-context: corrected crop page=%s table=%s "
                                "confidence=%.2f tables_on_page=%s",
                                page_num,
                                idx,
                                plan.confidence,
                                plan.table_count,
                            )
                            return {
                                "crop_bytes": rendered,
                                "bbox_norm": safe_bbox,
                                "bottom_extension": safe_bottom,
                                "top_extension": safe_top,
                                "confidence": plan.confidence,
                                "title_text": plan.title_text,
                                "continuation": plan.continuation,
                                "table_count": plan.table_count,
                            }

                        crop_bytes = _recrop(initial_bottom_ext)
                        if not crop_bytes:
                            vision_extraction_attempted = True
                            vision_status_str = "failed"
                            warnings_list = [
                                "crop rejected or empty; Vision skipped (invalid bbox, page, or crop failure)"
                            ]
                        else:
                            vision_result = vision_extractor.extract_with_quality_pass(
                                crop_bytes=crop_bytes,
                                bank_code=bank_code,
                                pdf_sha=pdf_sha,
                                page_number=page_num,
                                bbox_norm=table_bbox,
                                vision_cfg=vision_extraction_cfg,
                                initial_bottom_extension=initial_bottom_ext,
                                initial_top_extension=top_extension_title,
                                get_recrop_fn=_recrop,
                                get_variant_crop_fn=_variant_crop,
                                get_page_context_crop_fn=_page_context_crop,
                                reference_text=reference_text,
                            )
                            if vision_result is not None:
                                title = vision_result.table_title or ""
                                selected_bbox = getattr(
                                    vision_result,
                                    "selected_bbox_norm",
                                    None,
                                )
                                if isinstance(selected_bbox, list) and len(selected_bbox) == 4:
                                    final_table_bbox = [float(value) for value in selected_bbox]
                                elif isinstance(
                                    page_context_observation.get("bbox_norm"),
                                    (list, tuple),
                                ) and len(page_context_observation["bbox_norm"]) == 4:
                                    final_table_bbox = [
                                        float(value)
                                        for value in page_context_observation["bbox_norm"]
                                    ]
                                locator_title = str(
                                    getattr(
                                        vision_result,
                                        "page_context_title",
                                        "",
                                    )
                                    or page_context_observation.get("title_text")
                                    or ""
                                ).strip()
                                if not title and locator_title:
                                    title = locator_title
                                table_number, title_clean = self._extract_table_number(title or None)
                                title_raw = title or None
                                table_summary = str(vision_result.table_summary or "").strip() or None
                                out_headers = [] if labels_only else (vision_result.headers or [])
                                out_rows = []
                                indicators_raw_text = [
                                    str(item).strip()
                                    for item in list(vision_result.indicators or [])
                                    if str(item).strip()
                                ]
                                indicators_spatial_raw = []
                                indicators = [normalize_indicator_for_comparison(text) for text in indicators_raw_text]
                                footnotes = [] if labels_only else vision_result.to_footnotes_list()
                                vision_status_str = vision_result.vision_status or "ok"
                                extraction_status = (
                                    str(
                                        getattr(
                                            vision_result,
                                            "extraction_status",
                                            "ok",
                                        )
                                        or "ok"
                                    ).strip()
                                    or "ok"
                                )
                                warnings_list = list(vision_result.warnings or [])
                                if (
                                    extraction_status == "confirmed_no_table"
                                    and page_context_observation.get("bbox_norm")
                                ):
                                    extraction_status = "suspect_unresolved"
                                    warnings_list.append(
                                        "page_context_locator_confirms_table_region"
                                    )
                            else:
                                vision_status_str = "failed"
                                warnings_list = ["VisionFullExtractor returned None"]
            except BaseException as e:
                if type(e) is vision_schema_error_cls:
                    reason = f"Vision schema contract invalid: {e}"
                    if schema_failure_policy == "degrade_to_docling":
                        schema_failure_flag[0] = True
                        shared["vision_extraction_disabled_reason"] = reason
                        vision_status_str = "failed"
                        warnings_list = [reason[:300]]
                        vision_schema_contract_failed = True
                        vision_extraction_disabled_reason = reason
                    else:
                        raise RuntimeError(reason) from e
                else:
                    vision_status_str = "failed"
                    warnings_list = [str(e)[:300]]
        else:
            if not vision_extractor:
                warnings_list = ["no vision extractor (API key missing or init failed)"]
            elif not table_bbox:
                warnings_list = ["no bbox from Docling"]
            else:
                warnings_list = ["bbox invalid"]

        if page_context_observation.get("bbox_norm"):
            observed_bbox = page_context_observation.get("bbox_norm")
            if isinstance(observed_bbox, (list, tuple)) and len(observed_bbox) == 4:
                final_table_bbox = [float(value) for value in observed_bbox]
            if not title:
                locator_title = str(
                    page_context_observation.get("title_text") or ""
                ).strip()
                if locator_title:
                    title = locator_title
                    table_number, title_clean = self._extract_table_number(
                        title
                    )
                    title_raw = title
            if vision_result is None and extraction_status == "ok":
                extraction_status = "suspect_unresolved"
                warnings_list.append(
                    "page_context_locator_confirms_table_region_without_extraction"
                )

        requested_max_completion_tokens = int(vision_extraction_cfg.get("vision_max_completion_tokens", 65536))
        debug_metrics: dict[str, Any] = {
            "vision_status": vision_status_str,
            "vision_extraction_attempted": vision_extraction_attempted,
            "vision_extraction_applied": vision_status_str in ("ok", "partial"),
            "vision_schema_contract_failed": vision_schema_contract_failed,
            "has_reference_text": bool(reference_text and len(reference_text.strip()) > 20),
            "warnings": warnings_list,
            "vision_max_completion_tokens_requested": requested_max_completion_tokens,
            "vision_max_completion_tokens_rescue_used": False,
        }
        if vision_model_name:
            debug_metrics["vision_model"] = vision_model_name
            debug_metrics["vision_role"] = "extraction_primary"
        if warnings_list:
            debug_metrics["vision_warning_codes"] = list(warnings_list)
            known_failure_codes = {
                "vision_truncated",
                "vision_invalid_json",
                "vision_schema_validation_failed",
                "vision_retry_exhausted",
                "vision_transport_error",
                "vision_structured_output_fallback",
                "vision_lean_mode",
                "vision_rows_missing_from_fallback",
            }
            failure_causes = [code for code in warnings_list if code in known_failure_codes]
            if failure_causes:
                debug_metrics["vision_failure_causes"] = failure_causes
        if vision_extraction_disabled_reason:
            debug_metrics["vision_extraction_disabled_reason"] = vision_extraction_disabled_reason
        if crop_reject_reason:
            debug_metrics["crop_reject_reason"] = crop_reject_reason
        if bbox_sanity_profile is not None:
            debug_metrics["bbox_sanity_profile"] = bbox_sanity_profile
        if page_context_observation.get("bbox_norm"):
            debug_metrics.update(
                {
                    "bbox_original": original_table_bbox,
                    "bbox_final": final_table_bbox,
                    "bbox_source": str(
                        page_context_observation.get("bbox_source")
                        or "page_context_locator"
                    ),
                    "bbox_confidence": float(
                        page_context_observation.get("confidence", 0.0) or 0.0
                    ),
                    "bbox_verified": True,
                    "page_context_title": str(
                        page_context_observation.get("title_text") or ""
                    ),
                    "page_context_continuation": page_context_observation.get(
                        "continuation"
                    ),
                    "page_context_table_count": page_context_observation.get(
                        "table_count"
                    ),
                }
            )
        else:
            unresolved_source = str(
                page_context_observation.get("bbox_source") or "docling"
            )
            debug_metrics.update(
                {
                    "bbox_original": original_table_bbox,
                    "bbox_final": final_table_bbox,
                    "bbox_source": unresolved_source,
                    "bbox_verified": False,
                    "page_context_table_count": page_context_observation.get(
                        "table_count"
                    ),
                }
            )
        if page_context_observation.get("bbox_verification_reason"):
            debug_metrics["bbox_verification_reason"] = str(
                page_context_observation["bbox_verification_reason"]
            )
        # Recrop and completeness (from vision result when available)
        if vision_result is not None:
            if hasattr(vision_result, "retry_reasons"):
                debug_metrics["retry_reasons"] = list(vision_result.retry_reasons or [])
            if hasattr(vision_result, "no_table_detected"):
                debug_metrics["no_table_detected"] = bool(vision_result.no_table_detected)
            if hasattr(vision_result, "recrop_attempted"):
                debug_metrics["recrop_attempted"] = vision_result.recrop_attempted
            if hasattr(vision_result, "recrop_used"):
                debug_metrics["recrop_used"] = vision_result.recrop_used
            if hasattr(vision_result, "recrop_failed_incomplete"):
                debug_metrics["recrop_failed_incomplete"] = vision_result.recrop_failed_incomplete
            if getattr(vision_result, "acceptance_reason", None):
                debug_metrics["acceptance_reason"] = vision_result.acceptance_reason
            if hasattr(vision_result, "rejection_reasons"):
                debug_metrics["rejection_reasons"] = list(vision_result.rejection_reasons or [])
            if getattr(vision_result, "selected_candidate_name", None):
                debug_metrics["selected_candidate_name"] = vision_result.selected_candidate_name
            if hasattr(vision_result, "no_table_evidence_count"):
                debug_metrics["no_table_evidence_count"] = int(vision_result.no_table_evidence_count or 0)
            if hasattr(vision_result, "summary_present"):
                debug_metrics["summary_present"] = bool(vision_result.summary_present)
            if hasattr(vision_result, "indicator_count"):
                debug_metrics["indicator_count"] = int(vision_result.indicator_count or 0)
            if hasattr(vision_result, "candidate_quality_rank"):
                debug_metrics["candidate_quality_rank"] = list(vision_result.candidate_quality_rank or [])
            debug_metrics["vision_consensus_confidence"] = float(getattr(vision_result, "confidence_score", 0.0))
            if getattr(vision_result, "requested_max_completion_tokens", None) is not None:
                debug_metrics["vision_max_completion_tokens_requested"] = vision_result.requested_max_completion_tokens
            debug_metrics["vision_max_completion_tokens_rescue_used"] = bool(
                getattr(vision_result, "rescue_used", False)
            )
            debug_metrics["extraction_status"] = extraction_status
            if getattr(vision_result, "finish_reason", None):
                debug_metrics["vision_finish_reason"] = vision_result.finish_reason
            if getattr(vision_result, "prompt_tokens", None) is not None:
                debug_metrics["vision_prompt_tokens"] = vision_result.prompt_tokens
            if getattr(vision_result, "completion_tokens", None) is not None:
                debug_metrics["vision_completion_tokens"] = vision_result.completion_tokens
            if getattr(vision_result, "total_tokens", None) is not None:
                debug_metrics["vision_total_tokens"] = vision_result.total_tokens

        extracted_table = ExtractedTable(
            table_id=table_id,
            page_number=page_num,
            title=title or None,
            headers=out_headers,
            rows=out_rows,
            first_column_indicators=indicators,
            first_column_indicators_raw=indicators_raw_text,
            first_column_indicators_spatial=indicators_spatial_raw if indicators_spatial_raw else None,
            footnotes=footnotes,
            bbox=final_table_bbox,
            tables_on_page=(
                int(page_context_observation["table_count"])
                if page_context_observation.get("table_count") is not None
                else None
            ),
            bbox_top=(
                float(final_table_bbox[1])
                if final_table_bbox and len(final_table_bbox) == 4
                else None
            ),
            table_number=table_number,
            title_clean=title_clean,
            table_summary=table_summary,
            title_raw=title_raw,
            title_reliability=classify_rbc_title_reliability(
                title_clean or title or title_raw,
                bank_code=bank_code,
            ),
            extraction_method=("vision_full_gpt4o" if vision_status_str in ("ok", "partial") else "vision_failed"),
            debug_metrics=debug_metrics,
            extraction_status=extraction_status,
        )
        return (idx, extracted_table, page_num)
