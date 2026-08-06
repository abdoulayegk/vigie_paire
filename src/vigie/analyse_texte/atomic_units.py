"""Décomposition déterministe des énumérations en unités comparables."""

from __future__ import annotations

import re
from dataclasses import dataclass

from vigie.analyse_texte.list_items import parse_list_item_line


_INLINE_MARKER_RE = re.compile(
    r"(?<![\w])"
    r"(?P<marker>"
    r"\((?:\d{1,3}|[A-Za-z]|[ivxlcdmIVXLCDM]{1,8})\)"
    r"|(?:\d{1,3}|[A-Za-z]|[ivxlcdmIVXLCDM]{1,8})[.)]"
    r")"
    r"(?=\s)"
)
_WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ'’-]+\b", flags=re.UNICODE)
_TRAILING_ENUMERATION_CONNECTOR_RE = re.compile(
    r"(?:[;,]\s*)?(?:et|ou|and|or)(?:\s+de)?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AtomicCandidate:
    """Bloc candidat enrichi de sa structure parent-enfant."""

    kind: str
    text: str
    comparison_text: str
    unit_role: str = "standalone"
    parent_key: str | None = None
    marker: str | None = None
    parent_context: str = ""


def _normalize_surface_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _marker_label(marker: str) -> str:
    return str(marker or "").strip().lower().strip("().")


def _roman_to_int(value: str) -> int | None:
    """Convertit un chiffre romain canonique, sinon retourne ``None``."""
    roman = str(value or "").upper()
    if not roman or not re.fullmatch(r"[IVXLCDM]+", roman):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(roman):
        current = values[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    if total <= 0 or _int_to_roman(total) != roman:
        return None
    return total


def _int_to_roman(value: int) -> str:
    remaining = int(value)
    parts: list[str] = []
    for amount, symbol in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while remaining >= amount:
            parts.append(symbol)
            remaining -= amount
    return "".join(parts)


def _consecutive_values(values: list[int]) -> bool:
    return len(values) >= 2 and all(right == left + 1 for left, right in zip(values, values[1:]))


def _valid_marker_sequence(markers: list[str]) -> bool:
    """Valide une séquence ordonnée sans se fier à un marqueur isolé."""
    labels = [_marker_label(marker) for marker in markers]
    if len(labels) < 2:
        return False

    if all(label.isdigit() for label in labels):
        return _consecutive_values([int(label) for label in labels])

    roman_values = [_roman_to_int(label) for label in labels]
    if all(value is not None for value in roman_values):
        return _consecutive_values([int(value) for value in roman_values])

    if all(len(label) == 1 and label.isalpha() for label in labels):
        return _consecutive_values([ord(label) for label in labels])

    return False


def _comparison_item_text(text: str) -> str:
    """Retire la ponctuation de liaison sans altérer l'extrait source."""
    value = _normalize_surface_text(text)
    value = _TRAILING_ENUMERATION_CONNECTOR_RE.sub("", value).strip()
    return value.rstrip(";,").strip()


def _standalone_context(prefix: str) -> bool:
    """Évite de créer un faux paragraphe depuis une amorce grammaticale."""
    value = str(prefix or "").strip()
    return len(_WORD_RE.findall(value)) >= 5 and bool(re.search(r"[:.!?]\s*$", value))


def _split_inline_enumeration(
    text: str,
    *,
    parent_key: str,
) -> list[AtomicCandidate] | None:
    """Découpe ``i)… ii)…`` seulement lorsque la séquence est explicite."""
    value = str(text or "").strip()
    matches = list(_INLINE_MARKER_RE.finditer(value))
    if len(matches) < 2:
        return None
    markers = [match.group("marker") for match in matches]
    if not _valid_marker_sequence(markers):
        return None

    context = value[: matches[0].start()].strip()
    units: list[AtomicCandidate] = []
    if context and _standalone_context(context):
        units.append(
            AtomicCandidate(
                kind="enumeration_context",
                text=context,
                comparison_text=_normalize_surface_text(context),
                unit_role="context",
                parent_key=parent_key,
                parent_context=context,
            )
        )

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        body_text = value[match.end() : end]
        comparison_text = _comparison_item_text(body_text)
        if not comparison_text:
            return None
        source_text = f"{match.group('marker')} {comparison_text}"
        units.append(
            AtomicCandidate(
                kind="enumeration_item",
                text=source_text,
                comparison_text=comparison_text,
                unit_role="item",
                parent_key=parent_key,
                marker=match.group("marker"),
                parent_context=context,
            )
        )
    return units


def _split_markdown_list(
    text: str,
    *,
    parent_key: str,
) -> list[AtomicCandidate] | None:
    """Crée une unité par ligne d'une vraie liste Markdown multi-item."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    parsed = [parse_list_item_line(line) for line in lines]
    if len(lines) < 2 or any(item is None for item in parsed):
        return None

    units: list[AtomicCandidate] = []
    for source_text, item in zip(lines, parsed, strict=True):
        assert item is not None
        units.append(
            AtomicCandidate(
                kind="list_item",
                text=source_text,
                comparison_text=_normalize_surface_text(item.text),
                unit_role="item",
                parent_key=parent_key,
                marker=item.marker,
            )
        )
    return units


def split_atomic_candidate(
    *,
    kind: str,
    text: str,
    parent_key: str,
) -> list[AtomicCandidate] | None:
    """Retourne des unités atomiques fiables ou laisse le chunker inchangé."""
    if kind == "list":
        return _split_markdown_list(text, parent_key=parent_key)
    if kind == "paragraph":
        return _split_inline_enumeration(text, parent_key=parent_key)
    return None
