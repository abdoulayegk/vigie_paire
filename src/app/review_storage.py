"""Persistence helpers for analyst review state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REVIEW_STATE_SCHEMA_VERSION = "review_state_v1"


def get_review_state_path(compare_path: str | Path | None) -> Path | None:
    """Return the sidecar review-state path for a comparison JSON."""
    if not compare_path:
        return None
    target = Path(compare_path)
    if target.suffix.lower() == ".json":
        filename = f"{target.stem}.review_state.json"
    else:
        filename = f"{target.name}.review_state.json"
    return target.with_name(filename)


def load_review_state(compare_path: str | Path | None) -> dict[str, Any] | None:
    """Load persisted review state if present and valid."""
    state_path = get_review_state_path(compare_path)
    if state_path is None or not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def save_review_state(
    compare_path: str | Path | None,
    *,
    review_items: list[dict[str, Any]] | None = None,
    review_queue: list[dict[str, Any]] | None = None,
    review_selection: dict[str, Any] | None = None,
    review_current_idx: int | None = None,
    current_change_idx: int | None = None,
    current_indicator_idx: int | None = None,
    preferred_store: str = "review_queue",
    source: str = "dash",
) -> Path | None:
    """Persist the current analyst review state next to the comparison JSON."""
    state_path = get_review_state_path(compare_path)
    if state_path is None:
        return None

    payload: dict[str, Any] = {
        "schema_version": REVIEW_STATE_SCHEMA_VERSION,
        "compare_path": str(compare_path),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source or "dash"),
        "preferred_store": str(preferred_store or "review_queue"),
        "review_selection": dict(review_selection or {}),
        "review_current_idx": int(review_current_idx or 0),
        "current_change_idx": int(current_change_idx or 0),
        "current_indicator_idx": int(current_indicator_idx or 0),
    }
    if review_items is not None:
        payload["review_items"] = list(review_items)
    if review_queue is not None:
        payload["review_queue"] = list(review_queue)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return state_path
