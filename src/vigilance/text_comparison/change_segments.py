"""Segments verbatim deterministes pour le surlignage texte T1/T2."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_SUBSTANTIVE_RE = re.compile(r"\w", flags=re.UNICODE)
_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)*|[^\W_]+(?:[’'][^\W_]+)*|[^\w\s]",
    flags=re.UNICODE,
)
_SMALL_EQUAL_GAP_TOKENS = 2
_CONTEXT_PREFIX_TOKENS = {
    "a",
    "au",
    "aux",
    "cours",
    "de",
    "des",
    "du",
    "en",
    "l'exercice",
    "l’exercice",
}
_GROUP_START_CONTEXT_TOKENS = {"de", "du", "au", "aux", "en"}
_RIGHT_ATTACHED_PUNCTUATION = {".", ",", ";", ":", "!", "?", ")", "]", "}"}

_Token = tuple[str, int, int]


def _clean_fragment(value: str) -> str:
    """Retourne un fragment exact utilisable par ``str.find`` dans l'UI."""
    return value.strip()


def _has_substantive_text(value: str) -> bool:
    """Ignore les differences purement espace/ponctuation."""
    return bool(_SUBSTANTIVE_RE.search(value))


def _tokens_with_positions(text: str) -> list[_Token]:
    """Découpe un texte en tokens avec bornes exactes dans la chaîne source."""
    return [(match.group(0), match.start(), match.end()) for match in _TOKEN_RE.finditer(text)]


def _normalize_token(token: str) -> str:
    """Normalise un token pour le matching sans perdre les bornes originales."""
    return str(token or "").casefold()


def _token_fragment(text: str, tokens: list[_Token], start: int, end: int) -> str:
    """Retourne le fragment source exact couvert par une tranche de tokens."""
    if start >= end or start < 0 or end > len(tokens):
        return ""
    return text[tokens[start][1] : tokens[end - 1][2]]


def _first_substantive_token(tokens: list[_Token], start: int, end: int) -> str:
    """Retourne le premier token porteur de contenu dans une tranche."""
    for idx in range(start, end):
        token = tokens[idx][0]
        if _has_substantive_text(token):
            return _normalize_token(token)
    return ""


def _expand_left_context(
    *,
    old_tokens: list[_Token],
    new_tokens: list[_Token],
    i1: int,
    i2: int,
    j1: int,
    j2: int,
) -> tuple[int, int]:
    """Ajoute un court contexte gauche quand il rend un changement lisible.

    Exemple: ``annoncé`` devient ``a annoncé``; ``de l'exercice 2025``
    devient ``au cours de l'exercice 2025``. On n'élargit que lorsque le
    token précédent est identique dans T1/T2 afin de garder des preuves exactes.
    """
    old_first = _first_substantive_token(old_tokens, i1, i2)
    new_first = _first_substantive_token(new_tokens, j1, j2)
    original_first = old_first or new_first
    max_steps = 2 if original_first in _GROUP_START_CONTEXT_TOKENS else 1

    steps = 0
    while i1 > 0 and j1 > 0 and steps < max_steps:
        prev_old = _normalize_token(old_tokens[i1 - 1][0])
        prev_new = _normalize_token(new_tokens[j1 - 1][0])
        if prev_old != prev_new:
            break
        should_expand = False
        if steps == 0 and prev_old == "a" and original_first:
            should_expand = True
        elif original_first in _GROUP_START_CONTEXT_TOKENS and prev_old in _CONTEXT_PREFIX_TOKENS:
            should_expand = True
        if not should_expand:
            break
        i1 -= 1
        j1 -= 1
        steps += 1
    return i1, j1


def _expand_right_attached_punctuation(tokens: list[_Token], start: int, end: int) -> int:
    """Inclut la ponctuation collée au dernier token pour éviter un span isolé."""
    while (
        start < end < len(tokens)
        and tokens[end][0] in _RIGHT_ATTACHED_PUNCTUATION
        and tokens[end][1] == tokens[end - 1][2]
    ):
        end += 1
    return end


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

    old_tokens = _tokens_with_positions(old_text)
    new_tokens = _tokens_with_positions(new_text)
    if not old_tokens or not new_tokens:
        segment = _segment("modified", old_text, new_text)
        return [segment] if segment else []

    matcher = SequenceMatcher(
        None,
        [_normalize_token(token) for token, _start, _end in old_tokens],
        [_normalize_token(token) for token, _start, _end in new_tokens],
        autojunk=False,
    )
    grouped_opcodes: list[tuple[int, int, int, int]] = []
    current: tuple[int, int, int, int] | None = None
    pending_equal: tuple[int, int, int, int] | None = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if current is not None and max(i2 - i1, j2 - j1) <= _SMALL_EQUAL_GAP_TOKENS:
                pending_equal = (i1, i2, j1, j2)
            elif current is not None:
                grouped_opcodes.append(current)
                current = None
                pending_equal = None
            continue
        if current is None:
            current = (i1, i2, j1, j2)
        else:
            if pending_equal is not None:
                current = (current[0], i2, current[2], j2)
            else:
                grouped_opcodes.append(current)
                current = (i1, i2, j1, j2)
        pending_equal = None
    if current is not None:
        grouped_opcodes.append(current)

    segments: list[dict[str, str]] = []
    for i1, i2, j1, j2 in grouped_opcodes:
        if i1 < i2 and j1 < j2:
            i1, j1 = _expand_left_context(
                old_tokens=old_tokens,
                new_tokens=new_tokens,
                i1=i1,
                i2=i2,
                j1=j1,
                j2=j2,
            )
        i2 = _expand_right_attached_punctuation(old_tokens, i1, i2)
        j2 = _expand_right_attached_punctuation(new_tokens, j1, j2)
        old_part = _token_fragment(old_text, old_tokens, i1, i2)
        new_part = _token_fragment(new_text, new_tokens, j1, j2)
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
