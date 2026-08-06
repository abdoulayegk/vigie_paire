"""Adaptateurs de payloads de comparaison vers les elements de revue analyste."""

from __future__ import annotations

from typing import Any

from vigie.support.i18n import t
from vigie.interface.review_models import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_FOOTNOTE,
    CHANGE_TYPE_MODIFIED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_STRUCTURE,
    CHANGE_TYPE_TABLE_ADDED,
    CHANGE_TYPE_TABLE_REMOVED,
    CHANGE_TYPE_UNCERTAIN,
    EVENT_TYPE_FOOTNOTE_ONLY,
    EVENT_TYPE_MATCHED_PAIR,
    EVENT_TYPE_TABLE_ADDED,
    EVENT_TYPE_TABLE_REMOVED,
    ReviewItem,
)


def _make_change_id(prefix: str, index: int) -> str:
    """Genere un identifiant de changement avec prefixe et index zero-padde."""
    return f"{prefix}_{index:04d}"


def build_review_items_from_indicator_result(
    indicator_result: dict[str, Any],
    *,
    bank_code: str,
    quarter_from: str,
    quarter_to: str,
    pdf_path_t1: str,
    pdf_path_t2: str,
) -> list[ReviewItem]:
    """Construit les elements de la file de revue groupes par tableau a partir du payload canonique.

    Chaque tableau avec des changements produit UN seul ReviewItem contenant
    tous ses indicateurs ajoutes/retires/renommes dans la liste ``indicators``.

    Args:
        indicator_result: Payload canonique de comparaison d'indicateurs.
        bank_code: Code de la banque (ex. ``"rbc"``).
        quarter_from: Trimestre source (ex. ``"2025_t1"``).
        quarter_to: Trimestre cible (ex. ``"2025_t2"``).
        pdf_path_t1: Chemin du PDF du trimestre precedent.
        pdf_path_t2: Chemin du PDF du trimestre courant.

    Returns:
        Liste de ReviewItem prets pour la file de revue analyste.
    """
    items: list[ReviewItem] = []
    table_comparisons = indicator_result.get("table_comparisons", [])
    seq = 1

    def _source_ref(comp_or_table: dict[str, Any], side: str) -> str:
        """Resout la reference PDF source pour le cote t1 ou t2."""
        embedded = str(comp_or_table.get("source_pdf_t1" if side == "t1" else "source_pdf_t2", "") or "").strip()
        if embedded:
            return embedded
        return pdf_path_t1 if side == "t1" else pdf_path_t2

    for comp in table_comparisons:
        if not isinstance(comp, dict):
            continue

        added = comp.get("added_indicators", []) or []
        removed = comp.get("removed_indicators", []) or []
        renamed = comp.get("renamed_indicators", []) or []
        added_raw = comp.get("added_indicators_raw", []) or []
        removed_raw = comp.get("removed_indicators_raw", []) or []
        renamed_raw = comp.get("renamed_indicators_raw", []) or []
        table_status_raw = str(comp.get("table_status", "") or "").strip().lower()

        # Skip completely stable tables with zero changes
        footnotes_counts = comp.get("footnotes_counts") or {}
        has_footnote_changes = any(footnotes_counts.values()) if footnotes_counts else False
        if (
            table_status_raw in ("inchange", "stable", "")
            and not added
            and not removed
            and not renamed
            and not has_footnote_changes
        ):
            continue

        if not added and not removed and not renamed:
            # Table avec changement sans détail indicateurs => ReviewItem "tableau"
            _status_without_indicators = frozenset(
                {
                    "modifie",
                    "incertain",
                    "needs_review",
                    "structure_change",
                }
            )
            if table_status_raw not in _status_without_indicators:
                continue
            # Skip redundant table-level "modified" ReviewItem when footnote pass
            # will handle the changes (avoids confusing noise for analysts)
            if table_status_raw == "modifie" and has_footnote_changes:
                continue
            # Créer un ReviewItem tableau type_changement modifié/incertain/fusion
            if table_status_raw == "structure_change":
                change_type = CHANGE_TYPE_STRUCTURE
            elif table_status_raw in ("incertain", "needs_review"):
                change_type = CHANGE_TYPE_UNCERTAIN
            else:
                change_type = CHANGE_TYPE_MODIFIED
            section = str(comp.get("section", ""))
            table_name = str(
                comp.get("title_t2") or comp.get("title_t1") or comp.get("table_id_t2") or comp.get("table_id_t1") or ""
            )
            page_t1 = comp.get("page_t1")
            page_t2 = comp.get("page_t2")
            table_id_t1 = str(comp.get("table_id_t1", ""))
            table_id_t2 = str(comp.get("table_id_t2", ""))
            confidence = float(comp.get("match_score", 0.0) or 0.0)
            match_method = str(comp.get("rescue_type") or comp.get("match_reason", ""))
            table_number = str(comp.get("table_number") or "")
            items.append(
                ReviewItem(
                    change_id=_make_change_id("tbl", seq),
                    change_type=change_type,
                    indicator="",
                    section=section,
                    table_name=table_name,
                    table_number=table_number,
                    table_id_t1=table_id_t1,
                    table_id_t2=table_id_t2,
                    page_t1=page_t1,
                    page_t2=page_t2,
                    source_ref_t1=_source_ref(comp, "t1"),
                    source_ref_t2=_source_ref(comp, "t2"),
                    confidence=confidence,
                    table_title_raw=table_name,
                    table_status=table_status_raw,
                    indicators=[],
                    match_method=match_method,
                    all_indicators_t1=comp.get("all_indicators_t1") or [],
                    all_indicators_t2=comp.get("all_indicators_t2") or [],
                    bbox_t1=comp.get("bbox_t1"),
                    bbox_t2=comp.get("bbox_t2"),
                    added_indicators=[],
                    removed_indicators=[],
                    genai_analysis=comp.get("genai_analysis") or {},
                    match_metadata=comp.get("match_metadata") or {},
                    event_type=EVENT_TYPE_MATCHED_PAIR,
                )
            )
            seq += 1
            continue

        section = str(comp.get("section", ""))
        table_name = str(
            comp.get("title_t2") or comp.get("title_t1") or comp.get("table_id_t2") or comp.get("table_id_t1") or ""
        )
        table_title_raw_value = str(comp.get("table_title_raw") or table_name)
        page_t1 = comp.get("page_t1")
        page_t2 = comp.get("page_t2")
        table_id_t1 = str(comp.get("table_id_t1", ""))
        table_id_t2 = str(comp.get("table_id_t2", ""))
        confidence = float(comp.get("match_score", 0.0) or 0.0)
        match_method = str(comp.get("rescue_type") or comp.get("match_reason", ""))

        indicators: list[dict[str, Any]] = []

        for idx, ind in enumerate(added):
            display_name = str(ind)
            assessment = {}
            if isinstance(added_raw, list) and idx < len(added_raw):
                raw_item = added_raw[idx]
                if isinstance(raw_item, dict):
                    display_name = str(raw_item.get("value", "")).strip() or display_name
                    assessment = dict(raw_item.get("analyst_assessment") or {})
                else:
                    candidate = str(raw_item).strip()
                    if candidate:
                        display_name = candidate
            indicators.append(
                {
                    "name": display_name,
                    "name_clean": str(ind),
                    "type": CHANGE_TYPE_ADDED,
                    "review_status": "pending",
                    "analyst_assessment": assessment,
                }
            )

        for idx, ind in enumerate(removed):
            display_name = str(ind)
            assessment = {}
            if isinstance(removed_raw, list) and idx < len(removed_raw):
                raw_item = removed_raw[idx]
                if isinstance(raw_item, dict):
                    display_name = str(raw_item.get("value", "")).strip() or display_name
                    assessment = dict(raw_item.get("analyst_assessment") or {})
                else:
                    candidate = str(raw_item).strip()
                    if candidate:
                        display_name = candidate
            indicators.append(
                {
                    "name": display_name,
                    "name_clean": str(ind),
                    "type": CHANGE_TYPE_REMOVED,
                    "review_status": "pending",
                    "analyst_assessment": assessment,
                }
            )

        for idx, ren in enumerate(renamed):
            ren_raw = renamed_raw[idx] if isinstance(renamed_raw, list) and idx < len(renamed_raw) else {}
            assessment = dict(ren_raw.get("analyst_assessment") or {}) if isinstance(ren_raw, dict) else {}
            if isinstance(ren, dict):
                old_clean = str(ren.get("from", ""))
                new_clean = str(ren.get("to", ""))
                old_val = old_clean
                new_val = new_clean
                if isinstance(ren_raw, dict):
                    old_raw = str(ren_raw.get("previous", "")).strip()
                    new_raw = str(ren_raw.get("current", "")).strip()
                    if not old_raw:
                        old_raw = str(ren_raw.get("from", "")).strip()
                    if not new_raw:
                        new_raw = str(ren_raw.get("to", "")).strip()
                    if old_raw:
                        old_val = old_raw
                    if new_raw:
                        new_val = new_raw
                label = f"{old_val} -> {new_val}" if old_val or new_val else ""
                indicators.append(
                    {
                        "name": label,
                        "type": CHANGE_TYPE_RENAMED,
                        "from": old_val,
                        "to": new_val,
                        "from_clean": old_clean,
                        "to_clean": new_clean,
                        "review_status": "pending",
                        "analyst_assessment": assessment,
                    }
                )
            else:
                display_name = str(ren)
                if isinstance(ren_raw, str) and ren_raw.strip():
                    display_name = ren_raw.strip()
                indicators.append(
                    {
                        "name": display_name,
                        "name_clean": str(ren),
                        "type": CHANGE_TYPE_RENAMED,
                        "review_status": "pending",
                        "analyst_assessment": assessment,
                    }
                )

        n_added = len(added)
        n_removed = len(removed)
        n_renamed = len(renamed)
        parts = []
        if n_added:
            parts.append(f"{n_added} {t('indicator_add').lower()}(s)")
        if n_removed:
            parts.append(f"{n_removed} {t('indicator_removal').lower()}(s)")
        if n_renamed:
            parts.append(f"{n_renamed} {t('indicator_rename').lower()}(s)")
        summary_indicator = ", ".join(parts)

        if n_removed >= n_added and n_removed >= n_renamed:
            primary_type = CHANGE_TYPE_REMOVED
        elif n_added >= n_renamed:
            primary_type = CHANGE_TYPE_ADDED
        else:
            primary_type = CHANGE_TYPE_RENAMED

        table_number = str(comp.get("table_number") or "")
        items.append(
            ReviewItem(
                change_id=_make_change_id("tbl", seq),
                change_type=primary_type,
                indicator=summary_indicator,
                section=section,
                table_name=table_name,
                table_number=table_number,
                table_id_t1=table_id_t1,
                table_id_t2=table_id_t2,
                page_t1=page_t1,
                page_t2=page_t2,
                source_ref_t1=_source_ref(comp, "t1"),
                source_ref_t2=_source_ref(comp, "t2"),
                confidence=confidence,
                table_title_raw=table_title_raw_value,
                table_status=str(comp.get("table_status", "")),
                indicators=indicators,
                match_method=match_method,
                all_indicators_t1=comp.get("all_indicators_t1") or [],
                all_indicators_t2=comp.get("all_indicators_t2") or [],
                bbox_t1=comp.get("bbox_t1"),
                bbox_t2=comp.get("bbox_t2"),
                added_indicators=added,
                removed_indicators=removed,
                genai_analysis=comp.get("genai_analysis") or {},
                match_metadata=comp.get("match_metadata") or {},
                event_type=EVENT_TYPE_MATCHED_PAIR,
            )
        )
        seq += 1

    # Footnote review items (one per table with footnote changes)
    for comp in table_comparisons:
        if not isinstance(comp, dict):
            continue
        fn_diff = comp.get("footnotes_diff")
        if not fn_diff:
            continue
        all_fn_changes = (fn_diff.get("added") or []) + (fn_diff.get("removed") or []) + (fn_diff.get("modified") or [])
        if not all_fn_changes:
            continue

        section = str(comp.get("section", ""))
        table_name = str(
            comp.get("title_t2") or comp.get("title_t1") or comp.get("table_id_t2") or comp.get("table_id_t1") or ""
        )
        fn_counts = fn_diff.get("counts", {})
        n_fn_a = fn_counts.get("added", 0)
        n_fn_r = fn_counts.get("removed", 0)
        n_fn_m = fn_counts.get("modified", 0)
        parts_fn = []
        if n_fn_a:
            parts_fn.append(f"+{n_fn_a} note(s)")
        if n_fn_r:
            parts_fn.append(f"-{n_fn_r} note(s)")
        if n_fn_m:
            parts_fn.append(f"~{n_fn_m} note(s)")
        summary_fn = ", ".join(parts_fn)

        fn_indicators: list[dict[str, str]] = []
        for fc in all_fn_changes:
            ref = fc.get("footnote_ref", "")
            ctype_raw = fc.get("change_type", "")
            old_text = fc.get("old_text") or ""
            new_text = fc.get("new_text") or ""
            if "new" in ctype_raw:
                label = f"[{ref}] {new_text[:120]}"
                fn_type = CHANGE_TYPE_ADDED
            elif "removed" in ctype_raw:
                label = f"[{ref}] {old_text[:120]}"
                fn_type = CHANGE_TYPE_REMOVED
            else:
                label = f"[{ref}] {old_text[:60]} -> {new_text[:60]}"
                fn_type = CHANGE_TYPE_MODIFIED
            fn_indicators.append(
                {
                    "name": label,
                    "type": fn_type,
                    "footnote_ref": ref,
                    "old_text": old_text,
                    "new_text": new_text,
                    "review_status": "pending",
                }
            )

        table_number = str(comp.get("table_number") or "")
        items.append(
            ReviewItem(
                change_id=_make_change_id("fn", seq),
                change_type=CHANGE_TYPE_FOOTNOTE,
                indicator=summary_fn,
                section=section,
                table_name=table_name,
                table_number=table_number,
                table_id_t1=str(comp.get("table_id_t1", "")),
                table_id_t2=str(comp.get("table_id_t2", "")),
                page_t1=comp.get("page_t1"),
                page_t2=comp.get("page_t2"),
                source_ref_t1=_source_ref(comp, "t1"),
                source_ref_t2=_source_ref(comp, "t2"),
                confidence=float(comp.get("match_score", 0.0) or 0.0),
                table_title_raw=table_name,
                table_status=str(comp.get("table_status", "")),
                indicators=fn_indicators,
                match_method=str(comp.get("rescue_type") or comp.get("match_reason", "")),
                bbox_t1=comp.get("bbox_t1"),
                bbox_t2=comp.get("bbox_t2"),
                item_type="footnote",
                event_type=EVENT_TYPE_FOOTNOTE_ONLY,
                footnote_changes=all_fn_changes,
                genai_analysis=comp.get("genai_analysis") or {},
                match_metadata=comp.get("match_metadata") or {},
            )
        )
        seq += 1

    tables_added = indicator_result.get("tables_added", []) or []
    for table in tables_added:
        if not isinstance(table, dict):
            continue

        table_name = str(table.get("title") or table.get("table_id", ""))
        section = str(table.get("section", ""))
        page_t2 = table.get("page")
        table_id_t2 = str(table.get("table_id", ""))
        table_number = str(table.get("table_number") or "")

        match_meta_added: dict[str, Any] = {}
        if table.get("semantic_judge") is not None:
            match_meta_added["semantic_judge"] = table["semantic_judge"]

        items.append(
            ReviewItem(
                change_id=_make_change_id("tbl_add", seq),
                change_type=CHANGE_TYPE_TABLE_ADDED,
                indicator=t("table_entire_added"),
                section=section,
                table_name=table_name,
                table_number=table_number,
                table_id_t2=table_id_t2,
                page_t2=page_t2,
                source_ref_t2=_source_ref(table, "t2"),
                confidence=1.0,
                table_title_raw=table_name,
                table_status="ajoute",
                indicators=[],
                all_indicators_t1=table.get("all_indicators_t1") or [],
                all_indicators_t2=table.get("all_indicators_t2") or [],
                bbox_t1=table.get("bbox_t1"),
                bbox_t2=table.get("bbox_t2"),
                added_indicators=[],
                removed_indicators=[],
                genai_analysis=table.get("genai_analysis") or {},
                match_metadata=match_meta_added,
                event_type=EVENT_TYPE_TABLE_ADDED,
            )
        )
        seq += 1

    tables_removed = indicator_result.get("tables_removed", []) or []
    for table in tables_removed:
        if not isinstance(table, dict):
            continue

        table_name = str(table.get("title") or table.get("table_id", ""))
        section = str(table.get("section", ""))
        page_t1 = table.get("page")
        table_id_t1 = str(table.get("table_id", ""))
        table_number = str(table.get("table_number") or "")

        match_meta_removed: dict[str, Any] = {}
        if table.get("semantic_judge") is not None:
            match_meta_removed["semantic_judge"] = table["semantic_judge"]

        items.append(
            ReviewItem(
                change_id=_make_change_id("tbl_rem", seq),
                change_type=CHANGE_TYPE_TABLE_REMOVED,
                indicator=t("table_entire_removed"),
                section=section,
                table_name=table_name,
                table_number=table_number,
                table_id_t1=table_id_t1,
                page_t1=page_t1,
                source_ref_t1=_source_ref(table, "t1"),
                confidence=1.0,
                table_title_raw=table_name,
                table_status="supprime",
                indicators=[],
                all_indicators_t1=table.get("all_indicators_t1") or [],
                all_indicators_t2=table.get("all_indicators_t2") or [],
                bbox_t1=table.get("bbox_t1"),
                bbox_t2=table.get("bbox_t2"),
                added_indicators=[],
                removed_indicators=[],
                genai_analysis=table.get("genai_analysis") or {},
                match_metadata=match_meta_removed,
                event_type=EVENT_TYPE_TABLE_REMOVED,
            )
        )
        seq += 1

    # Extraction suspects (tables_*_pending_review) remain visible in summary
    # metrics, but are intentionally excluded from the visible review queue.

    return items
