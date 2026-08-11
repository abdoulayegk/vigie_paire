"""Réparation conservatrice des frontières artificielles entre fragments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

_TERMINAL_PUNCTUATION_RE = re.compile(r"""[.!?…][»"')\]]*\s*$""")
_LEADING_LETTER_RE = re.compile(r"""^[\s«"'(\[]*([A-Za-zÀ-ÖØ-öø-ÿ])""")
_TRAILING_INCOMPLETE_WORD_RE = re.compile(
    r"\b(?:"
    r"à|au|aux|avec|car|ce|ces|cet|cette|comme|dans|de|des|du|en|et|leur|leurs|"
    r"mais|notamment|ou|par|parmi|pour|que|qui|sans|selon|sur|un|une|vers"
    r")\s*$",
    flags=re.IGNORECASE,
)
_SUBJECT_START_RE = re.compile(
    r"^(?:le|la|les|l['’]|un|une|ce|cet|cette|ces|il|elle|ils|elles|"
    r"nous|notre|nos|on|banque|groupe|soci[eé]t[eé])\b",
    flags=re.IGNORECASE,
)


class BoundaryDisposition(str, Enum):
    """Nature d'une frontière entre deux fragments."""

    HARD = "hard"
    KEEP = "keep"
    MERGE = "merge"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    """Décision déterministe portant sur une frontière."""

    disposition: BoundaryDisposition
    reason: str


@dataclass(frozen=True, slots=True)
class RepairableBlock:
    """Bloc minimal requis par le réparateur de frontières."""

    kind: str
    text: str
    hard_boundary_before: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryRepairResult:
    """Résultat sans perte : blocs réparés et frontières encore ambiguës."""

    blocks: list[RepairableBlock]
    merged_boundaries: list[dict[str, str]]
    ambiguous_boundaries: list[dict[str, str]]


def _starts_with_lowercase(text: str) -> bool:
    match = _LEADING_LETTER_RE.search(str(text or ""))
    return bool(match and match.group(1).islower())


def _ends_with_terminal_punctuation(text: str) -> bool:
    return bool(_TERMINAL_PUNCTUATION_RE.search(str(text or "")))


def _looks_like_short_block_label(text: str) -> bool:
    value = str(text or "").strip()
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ'’-]*", value)
    return bool(
        words
        and len(words) <= 6
        and words[0][0].isupper()
        and not _SUBJECT_START_RE.match(value)
        and not _TRAILING_INCOMPLETE_WORD_RE.search(value)
        and not re.search(r"[,;:]", value)
    )


def classify_boundary(
    previous: RepairableBlock,
    current: RepairableBlock,
) -> BoundaryDecision:
    """Classe une frontière sans tenter de réécrire les fragments."""
    if current.hard_boundary_before:
        return BoundaryDecision(BoundaryDisposition.HARD, "explicit_structural_boundary")
    if previous.kind != "paragraph" or current.kind != "paragraph":
        return BoundaryDecision(BoundaryDisposition.HARD, "block_type_transition")

    previous_text = previous.text.strip()
    current_text = current.text.strip()
    if not previous_text or not current_text:
        return BoundaryDecision(BoundaryDisposition.HARD, "empty_fragment")
    if _ends_with_terminal_punctuation(previous_text):
        return BoundaryDecision(BoundaryDisposition.KEEP, "previous_sentence_complete")
    if _looks_like_short_block_label(previous_text):
        return BoundaryDecision(BoundaryDisposition.KEEP, "short_block_label")
    if _starts_with_lowercase(current_text):
        reason = (
            "incomplete_connector_then_lowercase"
            if _TRAILING_INCOMPLETE_WORD_RE.search(previous_text)
            else "missing_terminal_then_lowercase"
        )
        return BoundaryDecision(BoundaryDisposition.MERGE, reason)
    return BoundaryDecision(BoundaryDisposition.AMBIGUOUS, "missing_terminal_then_uppercase")


def _join_without_rewriting(left: str, right: str) -> str:
    return f"{left.rstrip()} {right.lstrip()}".strip()


def repair_block_boundaries(blocks: Iterable[RepairableBlock]) -> BoundaryRepairResult:
    """Fusionne seulement les continuations certaines; conserve les ambiguïtés."""
    repaired: list[RepairableBlock] = []
    merges: list[dict[str, str]] = []
    ambiguous: list[dict[str, str]] = []

    for block in blocks:
        if not repaired:
            repaired.append(block)
            continue
        previous = repaired[-1]
        decision = classify_boundary(previous, block)
        if decision.disposition is BoundaryDisposition.MERGE:
            merged_text = _join_without_rewriting(previous.text, block.text)
            repaired[-1] = RepairableBlock(
                kind="paragraph",
                text=merged_text,
                hard_boundary_before=previous.hard_boundary_before,
            )
            merges.append(
                {
                    "reason": decision.reason,
                    "previous_text": previous.text,
                    "next_text": block.text,
                    "result_text": merged_text,
                }
            )
            continue
        if decision.disposition is BoundaryDisposition.AMBIGUOUS:
            ambiguous.append(
                {
                    "reason": decision.reason,
                    "previous_text": previous.text,
                    "next_text": block.text,
                }
            )
        repaired.append(block)

    return BoundaryRepairResult(repaired, merges, ambiguous)
