"""Review state persistence helpers for Dash callbacks.

Extracted from dash_app/app.py. app.py re-exports all names from this module
so that all existing monkeypatches (setattr on dash_app) continue to work.
"""

from __future__ import annotations

from vigilance.dash_app.services.comparison_store import build_file_comparison_store
from vigilance.dash_app.services.comparison_context import _comparison_path_from_meta
from vigilance.dash_app.services.export_helpers import _review_items_from_v2_queue


def _persist_review_state(
    *,
    indicator_meta: dict | None,
    indicator_result: dict | None = None,
    review_items: list[dict] | None = None,
    review_queue: list[dict] | None = None,
    review_selection: dict | None = None,
    review_current_idx: int | None = None,
    current_change_idx: int | None = None,
    current_indicator_idx: int | None = None,
    preferred_store: str = "review_queue",
    source: str = "dash",
) -> None:
    """Persist review state next to the comparison JSON when possible."""
    compare_path = _comparison_path_from_meta(indicator_meta, indicator_result)
    if not compare_path:
        return
    # Extract run_id from meta so the state can be invalidated on re-run.
    run_id = ""
    if indicator_meta:
        run_id = str(indicator_meta.get("run_id", ""))
    store = build_file_comparison_store()
    store.save_review_state(
        compare_path,
        review_items=review_items,
        review_queue=review_queue,
        review_selection=review_selection,
        review_current_idx=review_current_idx,
        current_change_idx=current_change_idx,
        current_indicator_idx=current_indicator_idx,
        preferred_store=preferred_store,
        source=source,
        comparison_run_id=run_id,
    )


def _load_review_state_for_comparison(
    *,
    indicator_meta: dict | None,
    indicator_result: dict | None = None,
) -> dict | None:
    """Load persisted review state for the active comparison when possible."""
    compare_path = _comparison_path_from_meta(indicator_meta, indicator_result)
    if not compare_path:
        return None
    store = build_file_comparison_store()
    return store.load_review_state(compare_path)


def _stored_review_items_from_state(state: dict | None) -> list[dict] | None:
    """Return persisted review items, rebuilding them from a saved queue if needed."""
    if not isinstance(state, dict):
        return None

    stored_items = state.get("review_items")
    if isinstance(stored_items, list) and stored_items:
        return stored_items

    stored_queue = state.get("review_queue")
    if isinstance(stored_queue, list) and stored_queue:
        return [it.to_dict() for it in _review_items_from_v2_queue(stored_queue)]
    return None
