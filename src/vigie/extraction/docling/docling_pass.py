"""Extraction native docling : tableaux, bbox et metriques.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vigie.extraction.page_table_locator import (
    PageTableLocator,
    build_near_full_page_crop_plan,
    build_page_table_crop_plan,
)
from vigie.extraction.pdf_preview import render_pdf_page
from vigie.extraction.table_locator import (
    ENGINE_DOCLING,
    ENGINE_PYMUPDF_LAYOUT,
    anchors_to_vision_items,
    get_table_locator,
    resolve_table_locator_engine,
)
from vigie.extraction.vision_cache import compute_pdf_sha256
from vigie.extraction.vision_full import VisionFullExtractor, VisionSchemaContractError
from vigie.llm import require_configured
from vigie.support.config import get_vision_extraction_config, resolve_openai_model
from vigie.support.utils import page_layout_context, pdf_crop

from ..locator_merge_reconciliation import (
    _bbox_overlap_ratio,
    _is_locator_merge_conflict,
    _reconcile_on_demand_locator_merges,
)
from .config import _compute_vision_quality_summary
from .models import ExtractedDocument, ExtractedTable

logger = logging.getLogger("vigie.extraction.docling_processor")


class DoclingPassMixin:
    """Extraction native docling : tableaux, bbox et metriques."""

    def _extract_with_docling(
        self,
        pdf_path: Path,
        bank_code: str,
        quarter: str,
        year: int,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        labels_only: bool = False,
        use_vision_extraction: bool = False,
        force_extraction: bool = False,
    ) -> ExtractedDocument:
        """Extraire en utilisant la bibliotheque Docling avec pipeline de pretraitement.

        Args:
            pdf_path: Chemin vers le fichier PDF.
            bank_code: Code identifiant la banque.
            quarter: Identifiant du trimestre.
            year: Annee du rapport.
            page_ranges: Plages de pages optionnelles pour extraction ciblee.
            labels_only: Si True, ne stocker que la premiere colonne.
            use_vision_extraction: Si True, utiliser Vision GPT-4o pour le contenu.
            force_extraction: Si True, ignorer le cache Vision (re-extraction complete).

        Returns:
            ExtractedDocument contenant les tableaux et metadonnees extraits.
        """
        locator_engine = resolve_table_locator_engine()
        try:
            normalized_page_ranges = self._normalize_page_ranges(page_ranges)
            effective_page_ranges = normalized_page_ranges or None
            docling_page_range = (
                self._build_docling_page_range(effective_page_ranges) if locator_engine == ENGINE_DOCLING else None
            )

            def _get_vision_extraction_config(bank: str) -> dict:
                """Charger la configuration d'extraction Vision pour une banque."""
                try:
                    return get_vision_extraction_config(bank_code=bank) or {}
                except Exception:
                    return {}

            # Vision extraction: OpenAI Vision as content source (indicators + footnotes) for all tables
            vision_extraction_cfg: dict = {}
            bottom_extension_footnotes = 0.0
            top_extension_title = 0.03
            horizontal_padding = 0.02
            # fallback_to_docling removed: Vision is the sole content source (Rules 1+5)
            schema_failure_policy = "fail_fast"
            vision_extractor = None
            page_table_locator = None
            vision_model_name: str | None = None
            pdf_sha = ""
            vision_schema_error_cls: type[Exception] = Exception
            if use_vision_extraction:
                try:
                    vision_extraction_cfg = _get_vision_extraction_config(bank_code)
                    if force_extraction:
                        vision_extraction_cfg = {
                            **vision_extraction_cfg,
                            "vision_cache_enabled": False,
                        }
                        logger.info(
                            "force_extraction active: cache Vision desactive pour %s",
                            bank_code,
                        )
                    bottom_extension_footnotes = float(vision_extraction_cfg.get("bottom_extension_footnotes", 0.12))
                    top_extension_title = float(vision_extraction_cfg.get("top_extension_title", 0.03))
                    horizontal_padding = float(vision_extraction_cfg.get("horizontal_padding", 0.02))
                    # fallback_to_docling removed: Vision is the sole content source (Rules 1+5)
                    schema_failure_policy = (
                        str(vision_extraction_cfg.get("schema_failure_policy", "fail_fast")).strip().lower()
                    )
                    if schema_failure_policy not in {
                        "fail_fast",
                        "degrade_to_docling",
                    }:
                        schema_failure_policy = "fail_fast"
                    pdf_sha = compute_pdf_sha256(str(pdf_path))
                    vision_model_name = resolve_openai_model("extraction_primary")
                    vision_cache_enabled = bool(vision_extraction_cfg.get("vision_cache_enabled", True))
                    require_configured()
                    vision_extractor = VisionFullExtractor(
                        model=vision_model_name,
                        use_cache=vision_cache_enabled,
                    )
                    page_table_locator = PageTableLocator(
                        model=vision_model_name,
                        min_confidence=float(
                            vision_extraction_cfg.get(
                                "page_context_min_confidence",
                                0.85,
                            )
                        ),
                        use_cache=vision_cache_enabled,
                    )
                    vision_schema_error_cls = VisionSchemaContractError
                except Exception as e:
                    raise RuntimeError(f"Vision extraction init failed: {e}") from e

            # ---------------------------------------------------------------------------
            # Structure = engine selectionne (tables_layout | docling). Vision = contenu.
            # ---------------------------------------------------------------------------
            ref_max_chars = int(vision_extraction_cfg.get("vision_reference_text_max_chars", 6000))
            if locator_engine == ENGINE_DOCLING:
                locator = get_table_locator(ENGINE_DOCLING, converter=self._converter)
                location = locator.locate(
                    pdf_path,
                    effective_page_ranges,
                    reference_text_max_chars=ref_max_chars,
                    is_page_in_ranges=self._is_page_in_ranges,
                    docling_page_range=docling_page_range,
                )
            else:
                locator = get_table_locator(ENGINE_PYMUPDF_LAYOUT)
                location = locator.locate(
                    pdf_path,
                    effective_page_ranges,
                    reference_text_max_chars=ref_max_chars,
                )

            vision_items = anchors_to_vision_items(location.anchors)
            page_context_seed: dict[int, dict[str, Any]] = {}
            structure_inventory_pages = list(location.inventory_pages)
            structure_text_content = location.text_content or ""
            structure_total_pages = int(location.total_pages or 0)

            # Une bbox presque pleine page ne constitue pas une ancre fiable.
            # Faire verifier la page complete avant le garde-fou du worker :
            # une seule region Vision fiable est corrigee, sinon le candidat
            # est conserve et marque comme suspect.
            if page_table_locator is not None:
                near_full_positions_by_page: dict[int, list[int]] = {}
                for position, item in enumerate(vision_items):
                    _item_idx, item_page, item_bbox, _item_id, _item_reference = item
                    if not item_bbox:
                        continue
                    sane, reject_reason, _profile = pdf_crop.is_bbox_sane(
                        item_bbox,
                        vision_extraction_cfg,
                    )
                    if not sane and reject_reason == "bbox_near_full_page":
                        near_full_positions_by_page.setdefault(
                            item_page,
                            [],
                        ).append(position)

                for near_full_page, positions in near_full_positions_by_page.items():
                    layout = None
                    try:
                        page_image = render_pdf_page(
                            str(pdf_path),
                            near_full_page,
                            scale=1.5,
                            format="png",
                        )
                        if page_image:
                            layout = page_table_locator.locate_page(
                                page_image,
                                near_full_page,
                                pdf_sha=pdf_sha,
                            )
                    except Exception as exc:
                        logger.warning(
                            "Near-full-page bbox verification failed page=%s (non-fatal): %s",
                            near_full_page,
                            exc,
                        )

                    reliable_region_count = (
                        sum(region.confidence >= page_table_locator.min_confidence for region in layout.tables)
                        if layout is not None
                        else 0
                    )
                    plan = (
                        build_near_full_page_crop_plan(
                            layout,
                            min_confidence=page_table_locator.min_confidence,
                        )
                        if layout is not None and len(positions) == 1
                        else None
                    )
                    plan_reject_reason = None
                    if plan is not None:
                        plan_sane, plan_reject_reason, _plan_profile = pdf_crop.is_bbox_sane(
                            list(plan.bbox_norm),
                            vision_extraction_cfg,
                        )
                        if not plan_sane:
                            plan = None

                    for position in positions:
                        item_idx, item_page, item_bbox, item_id, item_reference = vision_items[position]
                        if plan is not None:
                            corrected_bbox = list(plan.bbox_norm)
                            page_context_seed[item_idx] = {
                                "bbox_original": list(item_bbox or []),
                                "bbox_norm": corrected_bbox,
                                "bbox_source": "page_context_near_full_page",
                                "confidence": plan.confidence,
                                "title_text": plan.title_text,
                                "continuation": plan.continuation,
                                "table_count": plan.table_count,
                                "bbox_verification_reason": ("near_full_page_single_region_confirmed"),
                            }
                            vision_items[position] = (
                                item_idx,
                                item_page,
                                corrected_bbox,
                                item_id,
                                item_reference,
                            )
                            continue

                        if len(positions) > 1:
                            verification_reason = "near_full_page_multiple_docling_candidates"
                        elif layout is None:
                            verification_reason = "near_full_page_locator_unavailable"
                        elif reliable_region_count == 0:
                            verification_reason = "near_full_page_no_reliable_region"
                        elif plan_reject_reason:
                            verification_reason = "near_full_page_locator_bbox_unsafe"
                        else:
                            verification_reason = "near_full_page_multiple_regions"
                        page_context_seed[item_idx] = {
                            "bbox_original": list(item_bbox or []),
                            "bbox_source": "near_full_page_unresolved",
                            "table_count": (len(layout.tables) if layout is not None else 0),
                            "bbox_verification_reason": verification_reason,
                        }

            # Inventaire pleine page optionnel : il corrige les boites Docling
            # existantes et cree les candidats que Docling n'a pas vus du tout.
            if page_table_locator is not None and bool(
                vision_extraction_cfg.get(
                    "page_context_inventory_enabled",
                    False,
                )
            ):
                inventory_page_padding = max(
                    0,
                    int(
                        vision_extraction_cfg.get(
                            "page_context_inventory_page_padding",
                            0,
                        )
                    ),
                )
                inventory_page_ranges = self._pad_page_ranges(
                    effective_page_ranges,
                    inventory_page_padding,
                )
                page_numbers: list[int] = []
                try:
                    page_numbers = sorted(
                        int(page_number)
                        for page_number in structure_inventory_pages
                        if self._is_page_in_ranges(
                            int(page_number),
                            inventory_page_ranges,
                        )
                    )
                except Exception:
                    page_numbers = []
                max_inventory_pages = max(
                    1,
                    int(
                        vision_extraction_cfg.get(
                            "page_context_inventory_max_pages",
                            80,
                        )
                    ),
                )
                page_numbers = page_numbers[:max_inventory_pages]
                new_table_min_confidence = max(
                    page_table_locator.min_confidence,
                    float(
                        vision_extraction_cfg.get(
                            "page_context_inventory_new_table_min_confidence",
                            0.93,
                        )
                    ),
                )
                excluded_new_table_titles = [
                    str(value or "").strip().casefold()
                    for value in list(
                        vision_extraction_cfg.get(
                            "page_context_inventory_excluded_title_patterns",
                            [],
                        )
                        or []
                    )
                    if str(value or "").strip()
                ]
                corrected_count = 0
                added_count = 0
                next_synthetic_idx = max((item[0] for item in vision_items), default=-1) + 1

                for inventory_page in page_numbers:
                    try:
                        page_image = render_pdf_page(
                            str(pdf_path),
                            inventory_page,
                            scale=1.5,
                            format="png",
                        )
                        layout = (
                            page_table_locator.locate_page(
                                page_image,
                                inventory_page,
                                pdf_sha=pdf_sha,
                            )
                            if page_image
                            else None
                        )
                    except Exception as exc:
                        logger.warning(
                            "Vision page inventory failed page=%s (non-fatal): %s",
                            inventory_page,
                            exc,
                        )
                        continue
                    if layout is None:
                        continue

                    page_item_positions = [
                        position for position, item in enumerate(vision_items) if item[1] == inventory_page
                    ]
                    planned_positions: list[tuple[int, list[float], Any]] = []
                    for position in page_item_positions:
                        _item_idx, _item_page, item_bbox, _item_id, _item_reference = vision_items[position]
                        if not item_bbox:
                            continue
                        if str(page_context_seed.get(_item_idx, {}).get("bbox_source") or "") in {
                            "page_context_near_full_page",
                            "near_full_page_unresolved",
                        }:
                            continue
                        plan = build_page_table_crop_plan(
                            layout,
                            item_bbox,
                            min_confidence=page_table_locator.min_confidence,
                        )
                        if plan is None:
                            continue
                        planned_positions.append((position, list(item_bbox), plan))

                    merge_conflict_positions: set[int] = set()
                    for plan_index, first_planned in enumerate(planned_positions):
                        first_position, first_original, first_plan = first_planned
                        for second_planned in planned_positions[plan_index + 1 :]:
                            (
                                second_position,
                                second_original,
                                second_plan,
                            ) = second_planned
                            if _is_locator_merge_conflict(
                                first_original,
                                second_original,
                                list(first_plan.bbox_norm),
                                list(second_plan.bbox_norm),
                            ):
                                merge_conflict_positions.update({first_position, second_position})

                    for position, item_bbox, plan in planned_positions:
                        item_idx, item_page, _bbox, item_id, item_reference = vision_items[position]
                        corrected_bbox = list(plan.bbox_norm)
                        bbox_source = "page_context_inventory"
                        if position in merge_conflict_positions:
                            corrected_bbox = list(item_bbox)
                            bbox_source = "page_context_inventory_conflict_preserved_docling"
                        page_context_seed[item_idx] = {
                            "bbox_original": list(item_bbox),
                            "bbox_norm": corrected_bbox,
                            "bbox_source": bbox_source,
                            "confidence": plan.confidence,
                            "title_text": plan.title_text,
                            "continuation": plan.continuation,
                            "table_count": plan.table_count,
                        }
                        vision_items[position] = (
                            item_idx,
                            item_page,
                            corrected_bbox,
                            item_id,
                            item_reference,
                        )
                        if position not in merge_conflict_positions and corrected_bbox != list(item_bbox):
                            corrected_count += 1

                    current_page_boxes = [
                        list(item[2]) for item in vision_items if item[1] == inventory_page and item[2]
                    ]
                    for region_index, region in enumerate(layout.tables, start=1):
                        region_bbox = list(region.table_bbox)
                        if region.confidence < new_table_min_confidence:
                            continue
                        normalized_region_title = str(region.title_text or "").casefold()
                        if any(pattern in normalized_region_title for pattern in excluded_new_table_titles):
                            logger.info(
                                "Vision page inventory ignored configured non-table region page=%s title=%s",
                                inventory_page,
                                region.title_text,
                            )
                            continue
                        if any(
                            _bbox_overlap_ratio(region_bbox, existing_bbox) >= 0.20
                            for existing_bbox in current_page_boxes
                        ):
                            continue
                        synthetic_idx = next_synthetic_idx
                        next_synthetic_idx += 1
                        synthetic_id = f"tableau_page_context_p{inventory_page}_{region_index}"
                        vision_items.append(
                            (
                                synthetic_idx,
                                inventory_page,
                                region_bbox,
                                synthetic_id,
                                None,
                            )
                        )
                        page_context_seed[synthetic_idx] = {
                            "bbox_original": None,
                            "bbox_norm": region_bbox,
                            "bbox_source": (
                                "page_context_inventory_new_candidate"
                                if self._is_page_in_ranges(
                                    inventory_page,
                                    effective_page_ranges,
                                )
                                else "page_context_inventory_boundary_candidate"
                            ),
                            "confidence": region.confidence,
                            "title_text": region.title_text,
                            "continuation": region.continuation,
                            "table_count": len(layout.tables),
                        }
                        current_page_boxes.append(region_bbox)
                        added_count += 1

                logger.info(
                    "Vision page inventory pages=%s padding=%s corrected=%s added=%s",
                    len(page_numbers),
                    inventory_page_padding,
                    corrected_count,
                    added_count,
                )

            def _detect_overlapping_bboxes(
                items: list[tuple[int, int, list[float] | None, str, str | None]],
            ) -> list[tuple[int, int, int, float]]:
                """Detecter les paires de boites englobantes qui se chevauchent par page."""
                by_page: dict[int, list[tuple[int, list[float]]]] = {}
                for idx, page_num, bbox, table_id, _ in items:
                    if bbox and len(bbox) >= 4:
                        by_page.setdefault(page_num, []).append((idx, bbox))
                overlaps: list[tuple[int, int, int, float]] = []
                for page_num, boxes in by_page.items():
                    if len(boxes) < 2:
                        continue
                    for i in range(len(boxes)):
                        for j in range(i + 1, len(boxes)):
                            idx_a, bbox_a = boxes[i]
                            idx_b, bbox_b = boxes[j]
                            ratio = _bbox_overlap_ratio(bbox_a, bbox_b)
                            if ratio > 0.01:
                                overlaps.append((page_num, idx_a, idx_b, ratio))
                return overlaps

            overlap_pairs = _detect_overlapping_bboxes(vision_items)
            if overlap_pairs:
                for page_num, idx_a, idx_b, ratio in overlap_pairs:
                    logger.warning(
                        "vision_extraction bbox_overlap page=%s idx=%s/%s ratio=%.3f",
                        page_num,
                        idx_a,
                        idx_b,
                        ratio,
                    )

            tables_per_page: dict[int, int] = {}
            for _idx, page_num, _bbox, _tid, _ref in vision_items:
                tables_per_page[page_num] = tables_per_page.get(page_num, 0) + 1
            if tables_per_page:
                logger.info(
                    "vision_extraction tables_detected_per_page %s",
                    dict(sorted(tables_per_page.items())),
                )

            # Build page-level layout context for dynamic crop extensions
            page_table_map = page_layout_context.build_page_table_map(vision_items)

            all_tables = []
            tables_by_page: dict[int, int] = {}
            if vision_items:
                schema_failure_flag: list[bool] = [False]
                shared: dict[str, Any] = {
                    "pdf_path": pdf_path,
                    "bank_code": bank_code,
                    "quarter": quarter,
                    "year": year,
                    "pdf_sha": pdf_sha,
                    "vision_extraction_cfg": vision_extraction_cfg,
                    "bottom_extension_footnotes": bottom_extension_footnotes,
                    "top_extension_title": top_extension_title,
                    "horizontal_padding": horizontal_padding,
                    "vision_extractor": vision_extractor,
                    "page_table_locator": page_table_locator,
                    "schema_failure_flag": schema_failure_flag,
                    "vision_schema_error_cls": vision_schema_error_cls,
                    "schema_failure_policy": schema_failure_policy,
                    "labels_only": labels_only,
                    "vision_crop_dpi": int(vision_extraction_cfg.get("vision_crop_dpi", 300)),
                    "vision_preprocess": vision_extraction_cfg.get("vision_preprocess", True),
                    "vision_model_name": vision_model_name,
                    "page_table_map": page_table_map,
                    "page_context_seed": page_context_seed,
                }
                if vision_extractor:
                    try:
                        vision_extractor.validate_schema()
                    except vision_schema_error_cls as e:
                        reason = str(e) or "Vision schema contract invalid"
                        if schema_failure_policy == "fail_fast":
                            raise
                        schema_failure_flag[0] = True
                        shared["vision_extraction_disabled_reason"] = reason
                vision_max_workers = int(vision_extraction_cfg.get("vision_extraction_max_workers", 4))
                max_workers = min(
                    max(1, vision_max_workers),
                    len(vision_items),
                )
                if max_workers <= 1:
                    for item in vision_items:
                        _idx, extracted_table, pnum = self._vision_extract_one_table(item, shared)
                        all_tables.append(extracted_table)
                        tables_by_page[pnum] = tables_by_page.get(pnum, 0) + 1
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                                self._vision_extract_one_table,
                                item,
                                shared,
                            )
                            for item in vision_items
                        ]
                        results: list[tuple[int, ExtractedTable, int]] = []
                        for fut in futures:
                            try:
                                results.append(fut.result())
                            except Exception as exc:
                                if type(exc) is vision_schema_error_cls:
                                    raise
                                raise
                        results.sort(key=lambda x: x[0])
                        for _idx, extracted_table, pnum in results:
                            all_tables.append(extracted_table)
                            tables_by_page[pnum] = tables_by_page.get(pnum, 0) + 1
                    if max_workers > 1:
                        logger.info(
                            "Vision extraction parallele: %d tableaux, %d workers",
                            len(vision_items),
                            max_workers,
                        )

            raw_table_count = len(all_tables)
            all_tables = _reconcile_on_demand_locator_merges(all_tables)
            collapsed_locator_tables = raw_table_count - len(all_tables)
            if collapsed_locator_tables:
                logger.info(
                    "Vision page-context: %d duplicate(s) semantique(s) du locator a la demande consolide(s)",
                    collapsed_locator_tables,
                )
                tables_by_page = {}
                for table in all_tables:
                    tables_by_page[table.page_number] = tables_by_page.get(table.page_number, 0) + 1

            if tables_by_page:
                counts_str = ", ".join(f"p{k}:{v}" for k, v in sorted(tables_by_page.items()))
                logger.info("Docling tableaux par page: %s", counts_str)

            rejected_bbox_sanity = sum(
                1
                for t in all_tables
                if getattr(t, "debug_metrics", None)
                and isinstance(t.debug_metrics, dict)
                and t.debug_metrics.get("crop_reject_reason")
            )
            if rejected_bbox_sanity:
                logger.info(
                    "vision_extraction tables_rejected_bbox_sanity count=%s",
                    rejected_bbox_sanity,
                )
            recrop_attempted = sum(
                1
                for t in all_tables
                if getattr(t, "debug_metrics", None)
                and isinstance(t.debug_metrics, dict)
                and t.debug_metrics.get("recrop_attempted")
            )
            recrop_used = sum(
                1
                for t in all_tables
                if getattr(t, "debug_metrics", None)
                and isinstance(t.debug_metrics, dict)
                and t.debug_metrics.get("recrop_used")
            )
            if recrop_attempted or recrop_used:
                logger.info(
                    "vision_extraction recrop attempted=%s used=%s",
                    recrop_attempted,
                    recrop_used,
                )

            # --- Vision extraction quality summary (one log line per run) ---
            if all_tables:
                _qsum = _compute_vision_quality_summary(all_tables)
                logger.info("vision_extraction_quality_summary %s", _qsum)

            # Extraire le contenu textuel pour les sections
            text_content = structure_text_content

            # Enrichir les titres manquants depuis le texte de la page (pdfplumber)
            # sans melanger contenu Docling/Vision : seul le champ titre est complete.
            all_tables = self._enrich_tables_with_titles(all_tables, pdf_path)

            # Associer les tableaux à leurs sections parentes
            all_tables = self._associate_tables_with_sections(all_tables, text_content)

            # Compter les sections détectées
            sections_found = set(t.section for t in all_tables if t.section)
            if sections_found:
                logger.info("Sections détectées: %s", ", ".join(sections_found))

            return ExtractedDocument(
                file_path=str(pdf_path),
                bank_code=bank_code,
                quarter=quarter,
                year=year,
                total_pages=structure_total_pages,
                all_tables=all_tables,
                metadata={
                    "extraction_method": "vision_full_gpt4o",
                    "table_locator_engine": locator_engine,
                    "sections_detected": list(sections_found),
                    "page_ranges": page_ranges,
                    "text_content": text_content[:50000],
                },
            )

        except Exception as e:
            if "Vision schema contract invalid" in str(e):
                raise
            if locator_engine == ENGINE_PYMUPDF_LAYOUT:
                logger.error(
                    "Echec de l'extraction tables_layout (%s): %s",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                raise
            logger.error(
                "Echec de l'extraction Docling (%s): %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return self._docling_unavailable_document(pdf_path, bank_code, quarter, year, page_ranges, error=str(e))
