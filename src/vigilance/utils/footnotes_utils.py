"""Footnote conversion helpers."""

from __future__ import annotations

from typing import Any


def footnotes_list_to_dict(items: list[Any]) -> dict[str, str]:
    """Convert a list-based footnote payload to a stable dict."""
    output: dict[str, str] = {}
    for index, item in enumerate(items or [], start=1):
        if isinstance(item, dict):
            key = str(item.get("id") or item.get("ref") or index)
            value = str(item.get("text") or item.get("value") or "").strip()
        else:
            key = str(index)
            value = str(item or "").strip()
        if value:
            output[key] = value
    return output
