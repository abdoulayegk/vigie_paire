"""Segments verbatim deterministes pour le surlignage texte T1/T2."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_SUBSTANTIVE_RE = re.compile(r"\w", flags=re.UNICODE)


def _clean_fragment(value: str) -> str:
    """Retourne un fragment exact utilisable par ``str.find`` dans l'UI."""
    return value.strip()


def _has_substantive_text(value: str) -> bool:
    """Ignore les differences purement espace/ponctuation."""
    return bool(_SUBSTANTIVE_RE.search(value))


def _segment(kind: str, text_t1: str = "", text_t2: str = "") -> dict[str, str] | None:
    """Construit un segment valide selon les invariants added/removed/modified."""
    old = _clean_fragment(text_t1)
    new = _clean_fragment(text_t2)
    if kind == "added":
        if not _has_substantive_text(new):
            return None
        return {"kind": "added", "text_t1": "", "text_t2": new}
    if kind == "removed":
        if not _has_substantive_text(old):
            return None
        return {"kind": "removed", "text_t1": old, "text_t2": ""}
    if not (_has_substantive_text(old) or _has_substantive_text(new)):
        return None
    if not old:
        return _segment("added", text_t2=new)
    if not new:
        return _segment("removed", text_t1=old)
    return {"kind": "modified", "text_t1": old, "text_t2": new}


def build_change_segments_from_texts(
    text_t1: str,
    text_t2: str,
    *,
    diff_type: str = "",
) -> list[dict[str, str]]:
    """Calcule les segments exacts a surligner depuis les textes sources.

    Le LLM conserve le jugement metier; cette fonction ne fait que comparer les
    deux chaines affichees a l'analyste et produire des fragments verbatim.
    """
    old_text = str(text_t1 or "")
    new_text = str(text_t2 or "")
    normalized_diff_type = str(diff_type or "").lower()

    if not old_text and not new_text:
        return []
    if not old_text:
        segment = _segment("added", text_t2=new_text)
        return [segment] if segment else []
    if not new_text:
        segment = _segment("removed", text_t1=old_text)
        return [segment] if segment else []

    matcher = SequenceMatcher(None, old_text, new_text, autojunk=False)
    grouped_opcodes: list[tuple[int, int, int, int]] = []
    current: tuple[int, int, int, int] | None = None
    pending_punctuation: tuple[int, int, int, int] | None = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_part = old_text[i1:i2]
        new_part = new_text[j1:j2]
        if tag == "equal" and current is not None and not (
            _has_substantive_text(old_part) or _has_substantive_text(new_part)
        ):
            pending_punctuation = (i1, i2, j1, j2)
            continue
        if tag == "equal":
            if current is not None:
                grouped_opcodes.append(current)
                current = None
                pending_punctuation = None
            continue
        if current is None:
            current = (i1, i2, j1, j2)
        else:
            if pending_punctuation is not None:
                current = (
                    current[0],
                    pending_punctuation[1],
                    current[2],
                    pending_punctuation[3],
                )
                pending_punctuation = None
            current = (current[0], i2, current[2], j2)
    if current is not None:
        grouped_opcodes.append(current)

    segments: list[dict[str, str]] = []
    for i1, i2, j1, j2 in grouped_opcodes:
        old_part = old_text[i1:i2]
        new_part = new_text[j1:j2]
        if old_part and new_part:
            segment = _segment("modified", old_part, new_part)
        elif old_part:
            segment = _segment("removed", text_t1=old_part)
        else:
            segment = _segment("added", text_t2=new_part)
        if segment:
            segments.append(segment)

    if segments:
        return segments

    if normalized_diff_type == "added":
        segment = _segment("added", text_t2=new_text)
    elif normalized_diff_type == "removed":
        segment = _segment("removed", text_t1=old_text)
    else:
        segment = _segment("modified", old_text, new_text)
    return [segment] if segment else []


def build_change_segments(change: dict[str, Any]) -> list[dict[str, str]]:
    """Calcule les segments depuis un changement narratif du pipeline."""
    text_t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "")
    text_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    return build_change_segments_from_texts(
        text_t1,
        text_t2,
        diff_type=str(change.get("diff_type") or ""),
    )
