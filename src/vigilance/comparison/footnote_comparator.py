"""
Comparateur de notes de bas de page pour detecter les changements dans les annotations de tableaux.
Important pour les divulgations reglementaires ou les changements methodologiques sont dans les notes.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
from difflib import SequenceMatcher

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

    # Keywords indicating important footnote changes
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
        """
        Initialize footnote comparator.

        Args:
            similarity_threshold: Threshold for considering footnotes as same
        """
        self.similarity_threshold = similarity_threshold

    def compare_footnotes(
        self, footnotes1: dict[str, str], footnotes2: dict[str, str], table_id: Optional[str] = None
    ) -> list[FootnoteChange]:
        """
        Compare footnotes from two tables.

        Args:
            footnotes1: Footnotes from older report {ref: text}
            footnotes2: Footnotes from newer report {ref: text}
            table_id: Optional table identifier

        Returns:
            List of FootnoteChange objects
        """
        changes = []

        # Normalize footnote references
        norm1 = self._normalize_footnotes(footnotes1)
        norm2 = self._normalize_footnotes(footnotes2)

        # Find new footnotes
        for ref, text in norm2.items():
            if ref not in norm1:
                # Check if similar content exists under different ref
                similar_ref = self._find_similar_footnote(text, norm1)
                if not similar_ref:
                    change = self._create_footnote_change("new_footnote", ref, text, None, table_id)
                    changes.append(change)

        # Find removed footnotes
        for ref, text in norm1.items():
            if ref not in norm2:
                similar_ref = self._find_similar_footnote(text, norm2)
                if not similar_ref:
                    change = self._create_footnote_change(
                        "removed_footnote", ref, None, text, table_id
                    )
                    changes.append(change)

        # Find modified footnotes
        for ref in set(norm1.keys()) & set(norm2.keys()):
            text1 = norm1[ref]
            text2 = norm2[ref]

            similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

            if similarity < self.similarity_threshold:
                change = self._create_footnote_change(
                    "modified_footnote", ref, text2, text1, table_id
                )
                changes.append(change)

        return changes

    def _normalize_footnotes(self, footnotes: dict) -> dict[str, str]:
        """Normalize footnote references and clean text."""
        normalized = {}

        for ref, text in footnotes.items():
            # Normalize reference
            norm_ref = str(ref).strip().lower()
            norm_ref = re.sub(r"[^\w]", "", norm_ref)

            # Clean text
            norm_text = str(text).strip()
            norm_text = re.sub(r"\s+", " ", norm_text)

            if norm_text:
                normalized[norm_ref] = norm_text

        return normalized

    def _find_similar_footnote(self, target: str, footnotes: dict[str, str]) -> Optional[str]:
        """Find a footnote with similar content."""
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
        """Create a FootnoteChange with proper classification."""
        text_to_check = new_text or old_text or ""

        # Classify
        category = self._classify_footnote(text_to_check)
        significance = self._assess_significance(change_type, category, text_to_check)

        # Generate description
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
        """Classify footnote content."""
        text_lower = text.lower()

        # Check for regulatory content
        for keyword in self.REGULATORY_KEYWORDS:
            if keyword in text_lower:
                return "REGULATORY"

        # Check for methodology changes
        for keyword in self.METHODOLOGY_KEYWORDS:
            if keyword in text_lower:
                return "INDICATOR"

        return "OTHER"

    def _assess_significance(self, change_type: str, category: str, text: str) -> str:
        """Assess significance of footnote change."""
        # Regulatory footnote changes are important
        if category == "REGULATORY":
            return "MAJOR" if change_type != "modified_footnote" else "MODERATE"

        # Methodology changes in footnotes can be important
        if any(kw in text.lower() for kw in self.METHODOLOGY_KEYWORDS):
            return "MODERATE"

        # New footnotes might indicate new disclosures
        if change_type == "new_footnote":
            return "MODERATE" if len(text) > 200 else "MINOR"

        return "MINOR"


def compare_footnotes(
    footnotes1: dict, footnotes2: dict, table_id: str = None
) -> list[FootnoteChange]:
    """
    Convenience function to compare footnotes.

    Args:
        footnotes1: Old footnotes
        footnotes2: New footnotes
        table_id: Optional table identifier

    Returns:
        List of footnote changes
    """
    comparator = FootnoteComparator()
    return comparator.compare_footnotes(footnotes1, footnotes2, table_id)
