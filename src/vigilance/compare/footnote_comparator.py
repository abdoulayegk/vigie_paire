"""
Comparateur de notes de bas de page pour detecter les changements dans les annotations de tableaux.
Important pour les divulgations reglementaires ou les changements methodologiques sont dans les notes.
"""

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from vigilance.utils.text_normalize_base import normalize_text_base

logger = logging.getLogger(__name__)


@dataclass
class FootnoteChange:
    """Represents a change in footnotes."""

    change_type: str  # "new_footnote", "removed_footnote", "modified_footnote"
    footnote_ref: str
    table_id: Optional[str]
    description: str
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    significance: str = "MINOR"
    category: str = "OTHER"

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "footnote_ref": self.footnote_ref,
            "table_id": self.table_id,
            "description": self.description,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "significance": self.significance,
            "category": self.category,
        }


class FootnoteComparator:
    """
    Compares footnotes between quarterly reports.
    Footnotes often contain important methodology changes and regulatory updates.
    """

    METHODOLOGY_KEYWORDS = [
        "méthode",
        "méthodologie",
        "calcul",
        "formule",
        "changement",
        "modification",
        "révision",
        "ajustement",
    ]

    REGULATORY_KEYWORDS = [
        "bâle",
        "bsif",
        "réglementaire",
        "norme",
        "exigence",
        "conformité",
        "ligne directrice",
    ]

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self._short_footnote_similarity_threshold = 0.92

    def compare_footnotes(
        self, footnotes1: dict[str, str], footnotes2: dict[str, str], table_id: Optional[str] = None
    ) -> list[FootnoteChange]:
        changes = []

        norm1, raw1 = self._normalize_footnotes_with_raw(footnotes1)
        norm2, raw2 = self._normalize_footnotes_with_raw(footnotes2)

        for ref, text in norm2.items():
            if ref not in norm1:
                similar_ref = self._find_similar_footnote(text, norm1)
                if not similar_ref:
                    changes.append(
                        self._create_footnote_change(
                            "new_footnote", ref, raw2.get(ref, text), None, table_id
                        )
                    )

        for ref, text in norm1.items():
            if ref not in norm2:
                similar_ref = self._find_similar_footnote(text, norm2)
                if not similar_ref:
                    changes.append(
                        self._create_footnote_change(
                            "removed_footnote", ref, None, raw1.get(ref, text), table_id
                        )
                    )

        for ref in set(norm1.keys()) & set(norm2.keys()):
            text1 = norm1[ref]
            text2 = norm2[ref]
            similarity = SequenceMatcher(None, text1, text2).ratio()
            thresh = (
                self._short_footnote_similarity_threshold
                if max(len(text1), len(text2)) < 50
                else self.similarity_threshold
            )
            if similarity < thresh:
                changes.append(
                    self._create_footnote_change(
                        "modified_footnote",
                        ref,
                        raw2.get(ref, text2),
                        raw1.get(ref, text1),
                        table_id,
                    )
                )

        return changes

    def _normalize_footnotes_with_raw(
        self, footnotes: dict
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return (normalized_for_compare, raw_display) keyed by norm_ref."""
        normalized: dict[str, str] = {}
        raw_display: dict[str, str] = {}
        for ref, text in footnotes.items():
            norm_ref = str(ref).strip().lower()
            norm_ref = re.sub(r"[^\w]", "", norm_ref)
            raw = re.sub(r"\s+", " ", str(text).strip())
            norm_text = normalize_text_base(raw)
            if norm_text:
                normalized[norm_ref] = norm_text
                raw_display[norm_ref] = raw
        return normalized, raw_display

    def _normalize_footnotes(self, footnotes: dict) -> dict[str, str]:
        n, _ = self._normalize_footnotes_with_raw(footnotes)
        return n

    def _find_similar_footnote(self, target: str, footnotes: dict[str, str]) -> Optional[str]:
        target_lower = target.lower()
        for ref, text in footnotes.items():
            similarity = SequenceMatcher(None, target_lower, text.lower()).ratio()
            if similarity >= self.similarity_threshold:
                return ref
        return None

    def _create_footnote_change(
        self,
        change_type: str,
        ref: str,
        new_text: Optional[str],
        old_text: Optional[str],
        table_id: Optional[str],
    ) -> FootnoteChange:
        text_to_check = new_text or old_text or ""
        category = self._classify_footnote(text_to_check)
        significance = self._assess_significance(change_type, category, text_to_check)

        if change_type == "new_footnote":
            desc = f"Nouvelle note de bas de page ({ref})"
        elif change_type == "removed_footnote":
            desc = f"Note de bas de page supprimée ({ref})"
        else:
            desc = f"Note de bas de page modifiée ({ref})"

        return FootnoteChange(
            change_type=change_type,
            footnote_ref=ref,
            table_id=table_id,
            description=desc,
            old_text=old_text[:500] if old_text else None,
            new_text=new_text[:500] if new_text else None,
            significance=significance,
            category=category,
        )

    def _classify_footnote(self, text: str) -> str:
        text_lower = text.lower()
        for keyword in self.REGULATORY_KEYWORDS:
            if keyword in text_lower:
                return "REGULATORY"
        for keyword in self.METHODOLOGY_KEYWORDS:
            if keyword in text_lower:
                return "INDICATOR"
        return "OTHER"

    def _assess_significance(self, change_type: str, category: str, text: str) -> str:
        if category == "REGULATORY":
            return "MAJOR" if change_type != "modified_footnote" else "MODERATE"
        if any(kw in text.lower() for kw in self.METHODOLOGY_KEYWORDS):
            return "MODERATE"
        if change_type == "new_footnote":
            return "MODERATE" if len(text) > 200 else "MINOR"
        return "MINOR"


def compare_footnotes(
    footnotes1: dict, footnotes2: dict, table_id: str = None
) -> list[FootnoteChange]:
    comparator = FootnoteComparator()
    return comparator.compare_footnotes(footnotes1, footnotes2, table_id)
