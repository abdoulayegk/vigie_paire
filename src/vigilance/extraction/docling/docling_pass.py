"""Extraction native docling : tableaux, bbox et metriques.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..docling_bbox_helpers import _build_indicator_reference_text
from ..locator_merge_reconciliation import (
    _bbox_overlap_ratio,
    _is_locator_merge_conflict,
    _reconcile_on_demand_locator_merges,
)
from .config import _compute_vision_quality_summary
from .models import ExtractedDocument, ExtractedTable

logger = logging.getLogger("vigilance.extraction.docling_processor")


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

        Returns:
            ExtractedDocument contenant les tableaux et metadonnees extraits.
        """
        try:
            normalized_page_ranges = self._normalize_page_ranges(page_ranges)
            effective_page_ranges = normalized_page_ranges or None
            docling_page_range = self._build_docling_page_range(effective_page_ranges)
            convert_kwargs: dict = {}
            if docling_page_range is not None:
                convert_kwargs["page_range"] = docling_page_range

            result = self._converter.convert(str(pdf_path), **convert_kwargs)
            doc = result.document

            def _get_vision_extraction_config(bank: str) -> dict:
                """Charger la configuration d'extraction Vision pour une banque."""
                try:
                    from ...config import get_vision_extraction_config as _gvec

                    return _gvec(bank_code=bank) or {}
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
                    from ...config import resolve_openai_model
                    from ...utils.genai import get_openai_api_key
                    from .page_table_locator import PageTableLocator
                    from .vision_cache import compute_pdf_sha256
                    from .vision_full_extractor import (
                        VisionFullExtractor,
                        VisionSchemaContractError,
                    )

                    pdf_sha = compute_pdf_sha256(str(pdf_path))
                    api_key = self.openai_api_key or get_openai_api_key()
                    vision_model_name = resolve_openai_model("extraction_primary")
                    vision_cache_enabled = bool(vision_extraction_cfg.get("vision_cache_enabled", True))
                    if api_key:
                        vision_extractor = VisionFullExtractor(
                            api_key=api_key,
                            model=vision_model_name,
                            use_cache=vision_cache_enabled,
                        )
                        page_table_locator = PageTableLocator(
                            api_key=api_key,
                            model=vision_model_name,
                            min_confidence=float(
                                vision_extraction_cfg.get(
                                    "page_context_min_confidence",
                                    0.85,
                                )
                            ),
                            use_cache=vision_cache_enabled,
                        )
                    else:
                        logger.warning("Vision extraction: OPENAI_API_KEY absente, desactivation")
                        use_vision_extraction = False
                    vision_schema_error_cls = VisionSchemaContractError
                except Exception as e:
                    logger.warning("Vision extraction init failed: %s", e)
                    use_vision_extraction = False

            # ---------------------------------------------------------------------------
            # Steps 2+3: Docling = structure only. Vision = single content source.
            # ---------------------------------------------------------------------------
            # Construire la liste des tableaux a traiter (dans les plages de pages).
            vision_items: list[tuple[int, int, list[float] | None, str, str | None]] = []
            page_context_seed: dict[int, dict[str, Any]] = {}
            for idx, table in enumerate(doc.tables):
                page_num = table.prov[0].page_no if table.prov else 0
                table_bbox: list[float] | None = None
                try:
                    if table.prov and hasattr(table.prov[0], "bbox") and table.prov[0].bbox is not None:
                        raw_bbox = table.prov[0].bbox
                        page_obj = doc.pages.get(page_num) if hasattr(doc, "pages") else None
                        if page_obj and hasattr(page_obj, "size") and page_obj.size:
                            norm = raw_bbox.to_top_left_origin(page_height=page_obj.size.height)
                            norm = norm.normalized(page_obj.size)
                            table_bbox = [norm.l, norm.t, norm.r, norm.b]
                        elif hasattr(raw_bbox, "as_tuple"):
                            table_bbox = list(raw_bbox.as_tuple())
                except Exception:
                    table_bbox = None
                if not self._is_page_in_ranges(page_num, effective_page_ranges):
                    continue
                table_id = f"tableau_{idx}"
                reference_text: str | None = None
                try:
                    if hasattr(table, "text") and table.text:
                        _ref_raw = str(table.text).strip()
                        if len(_ref_raw) > 20:
                            ref_max_chars = int(vision_extraction_cfg.get("vision_reference_text_max_chars", 6000))
                            if ref_max_chars > 0:
                                reference_text = _build_indicator_reference_text(
                                    _ref_raw,
                                    max_chars=ref_max_chars,
                                )
                except Exception:
                    pass
                vision_items.append((idx, page_num, table_bbox, table_id, reference_text))

            # Une bbox presque pleine page ne constitue pas une ancre fiable.
            # Faire verifier la page complete avant le garde-fou du worker :
            # une seule region Vision fiable est corrigee, sinon le candidat
            # est conserve et marque comme suspect.
            if page_table_locator is not None:
                from ...utils.pdf_crop import is_bbox_sane
                from .page_table_locator import (
                    build_near_full_page_crop_plan,
                )
                from .pdf_preview import render_pdf_page

                near_full_positions_by_page: dict[int, list[int]] = {}
                for position, item in enumerate(vision_items):
                    _item_idx, item_page, item_bbox, _item_id, _item_reference = item
                    if not item_bbox:
                        continue
                    sane, reject_reason, _profile = is_bbox_sane(
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
                            "Near-full-page bbox verification failed page=%s "
                            "(non-fatal): %s",
                            near_full_page,
                            exc,
                        )

                    reliable_region_count = (
                        sum(
                            region.confidence
                            >= page_table_locator.min_confidence
                            for region in layout.tables
                        )
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
                        plan_sane, plan_reject_reason, _plan_profile = (
                            is_bbox_sane(
                                list(plan.bbox_norm),
                                vision_extraction_cfg,
                            )
                        )
                        if not plan_sane:
                            plan = None

                    for position in positions:
                        item_idx, item_page, item_bbox, item_id, item_reference = (
                            vision_items[position]
                        )
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
                                "bbox_verification_reason": (
                                    "near_full_page_single_region_confirmed"
                                ),
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
                            verification_reason = (
                                "near_full_page_multiple_docling_candidates"
                            )
                        elif layout is None:
                            verification_reason = (
                                "near_full_page_locator_unavailable"
                            )
                        elif reliable_region_count == 0:
                            verification_reason = (
                                "near_full_page_no_reliable_region"
                            )
                        elif plan_reject_reason:
                            verification_reason = (
                                "near_full_page_locator_bbox_unsafe"
                            )
                        else:
                            verification_reason = "near_full_page_multiple_regions"
                        page_context_seed[item_idx] = {
                            "bbox_original": list(item_bbox or []),
                            "bbox_source": "near_full_page_unresolved",
                            "table_count": (
                                len(layout.tables) if layout is not None else 0
                            ),
                            "bbox_verification_reason": verification_reason,
                        }

            # Inventaire pleine page optionnel : il corrige les boites Docling
            # existantes et cree les candidats que Docling n'a pas vus du tout.
            if (
                page_table_locator is not None
                and bool(
                    vision_extraction_cfg.get(
                        "page_context_inventory_enabled",
                        False,
                    )
                )
            ):
                from .page_table_locator import build_page_table_crop_plan
                from .pdf_preview import render_pdf_page

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
                if hasattr(doc, "pages"):
                    try:
                        page_numbers = sorted(
                            int(page_number)
                            for page_number in doc.pages.keys()
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
                next_synthetic_idx = (
                    max((item[0] for item in vision_items), default=-1) + 1
                )

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
                        position
                        for position, item in enumerate(vision_items)
                        if item[1] == inventory_page
                    ]
                    planned_positions: list[tuple[int, list[float], Any]] = []
                    for position in page_item_positions:
                        _item_idx, _item_page, item_bbox, _item_id, _item_reference = vision_items[position]
                        if not item_bbox:
                            continue
                        if str(
                            page_context_seed.get(_item_idx, {}).get(
                                "bbox_source"
                            )
                            or ""
                        ) in {
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
                        planned_positions.append(
                            (position, list(item_bbox), plan)
                        )

                    merge_conflict_positions: set[int] = set()
                    for plan_index, first_planned in enumerate(
                        planned_positions
                    ):
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
                                merge_conflict_positions.update(
                                    {first_position, second_position}
                                )

                    for position, item_bbox, plan in planned_positions:
                        item_idx, item_page, _bbox, item_id, item_reference = (
                            vision_items[position]
                        )
                        corrected_bbox = list(plan.bbox_norm)
                        bbox_source = "page_context_inventory"
                        if position in merge_conflict_positions:
                            corrected_bbox = list(item_bbox)
                            bbox_source = (
                                "page_context_inventory_conflict_preserved_docling"
                            )
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
                        if (
                            position not in merge_conflict_positions
                            and corrected_bbox != list(item_bbox)
                        ):
                            corrected_count += 1

                    current_page_boxes = [
                        list(item[2])
                        for item in vision_items
                        if item[1] == inventory_page and item[2]
                    ]
                    for region_index, region in enumerate(layout.tables, start=1):
                        region_bbox = list(region.table_bbox)
                        if region.confidence < new_table_min_confidence:
                            continue
                        normalized_region_title = str(
                            region.title_text or ""
                        ).casefold()
                        if any(
                            pattern in normalized_region_title
                            for pattern in excluded_new_table_titles
                        ):
                            logger.info(
                                "Vision page inventory ignored configured "
                                "non-table region page=%s title=%s",
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
                        synthetic_id = (
                            f"tableau_page_context_p{inventory_page}_{region_index}"
                        )
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
            from ...utils.page_layout_context import build_page_table_map

            page_table_map = build_page_table_map(vision_items)

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
                    "Vision page-context: %d duplicate(s) semantique(s) "
                    "du locator a la demande consolide(s)",
                    collapsed_locator_tables,
                )
                tables_by_page = {}
                for table in all_tables:
                    tables_by_page[table.page_number] = (
                        tables_by_page.get(table.page_number, 0) + 1
                    )

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
            text_content = doc.export_to_markdown()

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
                total_pages=len(doc.pages) if hasattr(doc, "pages") else 0,
                all_tables=all_tables,
                metadata={
                    "extraction_method": "vision_full_gpt4o",
                    "sections_detected": list(sections_found),
                    "page_ranges": page_ranges,
                    "text_content": text_content[:50000],
                },
            )

        except Exception as e:
            if "Vision schema contract invalid" in str(e):
                raise
            logger.error(
                "Echec de l'extraction Docling (%s): %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return self._docling_unavailable_document(pdf_path, bank_code, quarter, year, page_ranges, error=str(e))
