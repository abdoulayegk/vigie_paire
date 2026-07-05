"""Parsing des sous-sections markdown et unites narratives."""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

from .constants import (
    _BULLET_LINE_RE,
    _NARRATIVE_UNIT_LONG_CHARS,
    _NARRATIVE_UNIT_LONG_WORDS,
    _NARRATIVE_UNIT_MIN_CHARS,
    _NARRATIVE_UNIT_TARGET_MAX_CHARS,
    _NARRATIVE_UNIT_TARGET_MIN_CHARS,
    _SECTION_LABELS,
    _SENTENCE_SPLIT_RE,
    _SUBSECTION_SPLIT_RE,
)
from .models import NarrativeUnit, _SubsectionRecord
from .text_normalization import _matching_tokens, _word_count
from .text_topics import _canonical_topic_for_text

def _parse_subsections(md_text: str) -> list[tuple[str, str]]:
    """Découpe un texte markdown en paires (heading, body).

    Le texte avant le premier ### devient (``__intro__``, body).
    Les headings ## de section ne sont pas inclus.
    """
    parts = _SUBSECTION_SPLIT_RE.split(md_text)
    result: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        result.append(("__intro__", intro))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if heading:
            result.append((heading, body))
    return result


def _hierarchy_path_for_subsection(section_key: str, heading: str) -> str:
    """Construit un chemin hiérarchique stable sans changer le schéma Dash."""
    section_label = _SECTION_LABELS.get(section_key, section_key)
    if not heading or heading == "__intro__":
        return section_label
    return f"{section_label} > {heading}"


def _split_markdown_paragraphs(body: str) -> list[str]:
    """Sépare le corps markdown en paragraphes/listes sans perdre les puces."""
    chunks: list[str] = []
    current: list[str] = []
    for raw_line in (body or "").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
            continue
        if _BULLET_LINE_RE.match(line):
            if current:
                chunks.append(" ".join(current).strip())
                current = []
            chunks.append(line)
            continue
        current.append(line)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Découpe un long paragraphe en unités de phrases, sans couper au milieu d'une phrase."""
    text = re.sub(r"\s+", " ", (paragraph or "").strip())
    if not text:
        return []
    if len(text) <= _NARRATIVE_UNIT_LONG_CHARS and _word_count(text) <= _NARRATIVE_UNIT_LONG_WORDS:
        return [text]

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return [text]

    units: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if (
            current
            and len(candidate) > _NARRATIVE_UNIT_TARGET_MAX_CHARS
            and len(current) >= _NARRATIVE_UNIT_TARGET_MIN_CHARS
        ):
            units.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        if units and len(current) < _NARRATIVE_UNIT_MIN_CHARS:
            units[-1] = f"{units[-1]} {current}".strip()
        else:
            units.append(current)

    merged: list[str] = []
    for unit in units:
        if merged and len(unit) < _NARRATIVE_UNIT_MIN_CHARS and len(merged[-1]) < _NARRATIVE_UNIT_TARGET_MAX_CHARS:
            merged[-1] = f"{merged[-1]} {unit}".strip()
        else:
            merged.append(unit)
    return merged or [text]


def _split_body_into_narrative_units(
    *,
    section_key: str,
    heading: str,
    body: str,
    start_index: int = 1,
) -> list[NarrativeUnit]:
    """Découpe une sous-section markdown en unités narratives comparables."""
    units: list[NarrativeUnit] = []
    unit_index = start_index
    hierarchy_path = _hierarchy_path_for_subsection(section_key, heading)
    for paragraph in _split_markdown_paragraphs(body):
        for text in _split_long_paragraph(paragraph):
            clean = text.strip()
            if not clean:
                continue
            units.append(
                NarrativeUnit(
                    section_key=section_key,
                    heading=heading,
                    canonical_topic=_canonical_topic_for_text(heading, clean),
                    unit_text=clean,
                    unit_index=unit_index,
                    source_heading=heading,
                    char_len=len(clean),
                    word_count=_word_count(clean),
                    hierarchy_path=hierarchy_path,
                )
            )
            unit_index += 1
    return units


def _build_subsection_records(
    section_key: str,
    subsections: list[tuple[str, str]],
) -> dict[str, _SubsectionRecord]:
    """Construit les sous-sections enrichies utilisées par le matching."""
    records: dict[str, _SubsectionRecord] = {}
    next_index = 1
    for heading, body in subsections:
        hierarchy_path = _hierarchy_path_for_subsection(section_key, heading)
        units = _split_body_into_narrative_units(
            section_key=section_key,
            heading=heading,
            body=body,
            start_index=next_index,
        )
        next_index += len(units)
        records[heading] = _SubsectionRecord(
            section_key=section_key,
            heading=heading,
            body=body,
            canonical_topic=_canonical_topic_for_text(heading, body),
            tokens=_matching_tokens(f"{heading} {body}"),
            units=units,
            hierarchy_path=hierarchy_path,
        )
    return records
