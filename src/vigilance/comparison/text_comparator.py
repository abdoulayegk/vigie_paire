"""
Comparateur de texte pour detecter les changements significatifs dans le contenu narratif.
Se concentre sur les nouvelles idees, les mentions de risques et le texte reglementaire.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
from difflib import SequenceMatcher, unified_diff

logger = logging.getLogger(__name__)


@dataclass
class TextChange:
    """Represents a detected change in text content."""

    change_type: str  # "new_paragraph", "removed_paragraph", "modified_text", "new_mention"
    section: str
    description: str
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    page_number: int = 0
    significance: str = "MINOR"
    category: str = "OTHER"
    keywords_matched: list[str] = None

    def __post_init__(self):
        if self.keywords_matched is None:
            self.keywords_matched = []

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "section": self.section,
            "description": self.description,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "page_number": self.page_number,
            "significance": self.significance,
            "category": self.category,
            "keywords_matched": self.keywords_matched,
        }


class TextComparator:
    """
    Compares text content between quarterly reports to detect meaningful changes.
    Filters out noise and focuses on regulatory, risk, and strategic content.
    """

    # Keywords for classification
    REGULATORY_PATTERNS = [
        r"bâle\s*(III|3)",
        r"BSIF",
        r"AMF",
        r"NFP",
        r"réglementation",
        r"exigence\s+réglementaire",
        r"norme\s+de\s+fonds\s+propres",
        r"ligne\s+directrice",
        r"conformité",
    ]

    RISK_PATTERNS = [
        r"risque\s+(de\s+)?(crédit|marché|liquidité|opérationnel)",
        r"intelligence\s+artificielle",
        r"\bIA\b",
        r"cyber(sécurité)?",
        r"changement\s+climatique",
        r"risque\s+systémique",
        r"provision\s+pour\s+pertes",
        r"stress\s+test",
    ]

    ESG_PATTERNS = [
        r"ESG",
        r"développement\s+durable",
        r"émission\s+de\s+carbone",
        r"transition\s+(climatique|énergétique)",
        r"diversité",
        r"inclusion",
        r"gouvernance\s+d'entreprise",
    ]

    # Patterns to ignore (noise)
    NOISE_PATTERNS = [
        r"^\d+\s*$",  # Just numbers
        r"^page\s+\d+",  # Page numbers
        r"^www\.",  # URLs
        r"^\s*$",  # Empty lines
    ]

    def __init__(self, min_paragraph_length: int = 50):
        """
        Initialize text comparator.

        Args:
            min_paragraph_length: Minimum length for a paragraph to be considered
        """
        self.min_paragraph_length = min_paragraph_length
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.regulatory_regex = [re.compile(p, re.IGNORECASE) for p in self.REGULATORY_PATTERNS]
        self.risk_regex = [re.compile(p, re.IGNORECASE) for p in self.RISK_PATTERNS]
        self.esg_regex = [re.compile(p, re.IGNORECASE) for p in self.ESG_PATTERNS]
        self.noise_regex = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]

    def compare_text(
        self, text1: str, text2: str, section_name: str = "general"
    ) -> list[TextChange]:
        """
        Compare two text blocks and return meaningful changes.

        Args:
            text1: Text from older report
            text2: Text from newer report
            section_name: Name of the section being compared

        Returns:
            List of TextChange objects
        """
        changes = []

        # Split into paragraphs
        paragraphs1 = self._extract_paragraphs(text1)
        paragraphs2 = self._extract_paragraphs(text2)

        # Find new paragraphs
        for para in paragraphs2:
            if not self._find_similar_paragraph(para, paragraphs1):
                change = self._create_change_for_new_paragraph(para, section_name)
                if change:
                    changes.append(change)

        # Find removed paragraphs
        for para in paragraphs1:
            if not self._find_similar_paragraph(para, paragraphs2):
                changes.append(
                    TextChange(
                        change_type="removed_paragraph",
                        section=section_name,
                        description=f"Paragraphe supprimé dans la section {section_name}",
                        old_text=para[:500],
                        significance="MINOR",
                        category=self._classify_text(para),
                    )
                )

        # Look for new mentions of key topics
        topic_changes = self._detect_new_topic_mentions(text1, text2, section_name)
        changes.extend(topic_changes)

        return changes

    def _extract_paragraphs(self, text: str) -> list[str]:
        """Extract meaningful paragraphs from text."""
        if not text:
            return []

        # Split on double newlines or significant breaks
        raw_paragraphs = re.split(r"\n\s*\n", text)

        paragraphs = []
        for para in raw_paragraphs:
            # Clean up
            para = para.strip()
            para = re.sub(r"\s+", " ", para)

            # Skip noise
            if self._is_noise(para):
                continue

            # Skip short paragraphs
            if len(para) < self.min_paragraph_length:
                continue

            paragraphs.append(para)

        return paragraphs

    def _is_noise(self, text: str) -> bool:
        """Check if text is noise that should be ignored."""
        for pattern in self.noise_regex:
            if pattern.match(text):
                return True
        return False

    def _find_similar_paragraph(
        self, target: str, paragraphs: list[str], threshold: float = 0.7
    ) -> bool:
        """Check if a similar paragraph exists in the list."""
        target_lower = target.lower()

        for para in paragraphs:
            similarity = SequenceMatcher(None, target_lower, para.lower()).ratio()
            if similarity >= threshold:
                return True

        return False

    def _create_change_for_new_paragraph(
        self, paragraph: str, section: str
    ) -> Optional[TextChange]:
        """Create a TextChange for a new paragraph if it's meaningful."""
        category = self._classify_text(paragraph)
        keywords = self._extract_matched_keywords(paragraph)

        # Only report if it matches important patterns
        if category == "OTHER" and not keywords:
            # Check if it's still worth reporting
            if len(paragraph) < 200:
                return None

        significance = (
            "MAJOR"
            if category == "REGULATORY"
            else "MODERATE"
            if category in ["RISK_EMERGING", "ESG"]
            else "MINOR"
        )

        return TextChange(
            change_type="new_paragraph",
            section=section,
            description=f"Nouveau contenu dans {section}: {paragraph[:100]}...",
            new_text=paragraph[:1000],
            significance=significance,
            category=category,
            keywords_matched=keywords,
        )

    def _classify_text(self, text: str) -> str:
        """Classify text into category."""
        text_lower = text.lower()

        # Check regulatory patterns
        for pattern in self.regulatory_regex:
            if pattern.search(text_lower):
                return "REGULATORY"

        # Check ESG patterns
        for pattern in self.esg_regex:
            if pattern.search(text_lower):
                return "ESG"

        # Check risk patterns
        for pattern in self.risk_regex:
            if pattern.search(text_lower):
                return "RISK_EMERGING"

        return "OTHER"

    def _extract_matched_keywords(self, text: str) -> list[str]:
        """Extract keywords that matched in the text."""
        keywords = []
        text_lower = text.lower()

        all_patterns = [
            (self.regulatory_regex, "regulatory"),
            (self.risk_regex, "risk"),
            (self.esg_regex, "esg"),
        ]

        for patterns, _ in all_patterns:
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    keywords.append(match.group())

        return list(set(keywords))

    def _detect_new_topic_mentions(self, text1: str, text2: str, section: str) -> list[TextChange]:
        """Detect new mentions of important topics."""
        changes = []

        # Topics to track
        topics = {
            "IA": (r"\bintelligence\s+artificielle\b|\bIA\b", "RISK_EMERGING"),
            "Cybersécurité": (r"\bcyber(sécurité)?\b", "RISK_EMERGING"),
            "Climat": (r"\bchangement\s+climatique\b|\brisque\s+climatique\b", "ESG"),
            "IFRS 17": (r"\bIFRS\s*17\b", "REGULATORY"),
            "Bâle IV": (r"\bBâle\s*(IV|4)\b", "REGULATORY"),
        }

        text1_lower = text1.lower() if text1 else ""
        text2_lower = text2.lower() if text2 else ""

        for topic_name, (pattern, category) in topics.items():
            regex = re.compile(pattern, re.IGNORECASE)

            # Count mentions
            count1 = len(regex.findall(text1_lower))
            count2 = len(regex.findall(text2_lower))

            # New topic or significant increase
            if count2 > 0 and (count1 == 0 or count2 > count1 * 1.5):
                # Find context around the mention
                match = regex.search(text2)
                if match:
                    start = max(0, match.start() - 100)
                    end = min(len(text2), match.end() + 200)
                    context = text2[start:end].strip()

                    changes.append(
                        TextChange(
                            change_type="new_mention",
                            section=section,
                            description=f"Nouvelle mention ou augmentation de '{topic_name}'",
                            new_text=context,
                            significance="MODERATE" if count1 == 0 else "MINOR",
                            category=category,
                            keywords_matched=[topic_name],
                        )
                    )

        return changes


def compare_text_content(text1: str, text2: str, section: str = "general") -> list[TextChange]:
    """
    Convenience function to compare text content.

    Args:
        text1: Older text
        text2: Newer text
        section: Section name

    Returns:
        List of text changes
    """
    comparator = TextComparator()
    return comparator.compare_text(text1, text2, section)
