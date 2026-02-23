"""
Filtre de bruit pour eliminer les changements non pertinents des resultats de comparaison.
Resout le probleme Kofax de detection de trop de differences non pertinentes.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a change."""

    is_noise: bool
    reason: str | None = None
    confidence: float = 1.0


class NoiseFilter:
    """
    Filters out noise from detected changes.
    Eliminates cosmetic changes, simple numeric updates, and reformulations.
    """

    # Patterns that indicate noise
    NOISE_PATTERNS = {
        "date_change": [
            r"^\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)",
            r"^\d{4}[-/]\d{2}[-/]\d{2}",
            r"^(T|Q)[1-4]\s*\d{4}",
        ],
        "page_reference": [
            r"^page\s+\d+",
            r"^p\.\s*\d+",
            r"^\d+\s*$",
        ],
        "formatting": [
            r"^[\s\-–—]+$",
            r"^\*+$",
            r"^note\s*:?\s*$",
        ],
        "minor_punctuation": [
            r"^[,\.;:\s]+$",
        ],
        "url_email": [
            r"^https?://",
            r"^www\.",
            r"@\w+\.\w+",
        ],
    }

    # Numeric change patterns to ignore
    NUMERIC_ONLY_PATTERNS = [
        r"^[\d\s,\.\$%MG\(\)\-\+]+$",  # Only numbers and common formatting
    ]

    # Footnote patterns to strip from indicator text before comparison
    FOOTNOTE_PATTERNS = [
        r"\s*\(\d+\)\s*",  # (1), (2), etc.
        r"\s*\d+\)\s*",  # 2), 3), etc. (without opening paren)
        r"\s*\[\d+\]\s*",  # [1], [2], etc.
        r"\s*[a-z]\)\s*",  # a), b), c), etc.
        r"\s*[a-z],\s*[a-z]\s*",  # a,b  c,d patterns
        r"\s*[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*",  # Superscript numbers
        r"\s*\*+\s*",  # Asterisks
    ]

    # Keywords that make a change NOT noise (should be kept)
    KEEP_KEYWORDS = [
        # Regulatory
        "bâle",
        "bsif",
        "amf",
        "réglementation",
        "norme",
        "exigence",
        "cet1",
        "levier",
        "lcr",
        "nsfr",
        # Risks
        "risque",
        "provision",
        "créance",
        "défaut",
        "exposition",
        "intelligence artificielle",
        "ia",
        "cyber",
        "climat",
        # ESG
        "esg",
        "environnement",
        "carbone",
        "durable",
        # Methodology
        "méthode",
        "calcul",
        "changement",
        "nouvelle",
    ]

    def __init__(
        self,
        filter_numeric_only: bool = True,
        filter_date_changes: bool = True,
        min_text_length: int = 20,
    ):
        """
        Initialize noise filter.

        Args:
            filter_numeric_only: Filter out changes that are only numeric
            filter_date_changes: Filter out date-related changes
            min_text_length: Minimum text length to not be considered noise
        """
        self.filter_numeric_only = filter_numeric_only
        self.filter_date_changes = filter_date_changes
        self.min_text_length = min_text_length
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns."""
        self.compiled_noise = {}
        for category, patterns in self.NOISE_PATTERNS.items():
            self.compiled_noise[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

        self.compiled_numeric = [re.compile(p, re.IGNORECASE) for p in self.NUMERIC_ONLY_PATTERNS]

        self.compiled_footnotes = [re.compile(p, re.IGNORECASE) for p in self.FOOTNOTE_PATTERNS]

    def strip_footnotes(self, text: str) -> str:
        """
        Remove footnote markers from indicator text before comparison.

        Handles patterns like: (1), 2), [1], a), a,b, superscripts, asterisks

        Args:
            text: Indicator text potentially containing footnotes

        Returns:
            Cleaned text without footnote markers
        """
        if not text:
            return ""

        cleaned = text
        for pattern in self.compiled_footnotes:
            cleaned = pattern.sub(" ", cleaned)

        # Normalize whitespace after removal
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def is_noise(self, change: Any) -> FilterResult:
        """
        Determine if a change should be filtered as noise.

        Args:
            change: A change object with text content

        Returns:
            FilterResult indicating if change is noise
        """
        # Extract text from change
        text = self._extract_text_from_change(change)

        if not text:
            return FilterResult(is_noise=True, reason="empty_content", confidence=1.0)

        # First check if it contains important keywords (should be kept)
        if self._contains_important_keywords(text):
            return FilterResult(is_noise=False, reason="contains_keywords", confidence=0.9)

        # Check various noise patterns
        noise_result = self._check_noise_patterns(text)
        if noise_result.is_noise:
            return noise_result

        # Check numeric-only changes
        if self.filter_numeric_only and self._is_numeric_only(text):
            return FilterResult(is_noise=True, reason="numeric_only", confidence=0.95)

        # Check minimum length
        if len(text.strip()) < self.min_text_length:
            return FilterResult(is_noise=True, reason="too_short", confidence=0.8)

        return FilterResult(is_noise=False, confidence=0.9)

    def filter_changes(self, changes: list[Any]) -> tuple[list[Any], list[Any]]:
        """
        Filter a list of changes into relevant and noise.

        Args:
            changes: List of change objects

        Returns:
            Tuple of (relevant_changes, filtered_noise)
        """
        relevant = []
        noise = []

        for change in changes:
            result = self.is_noise(change)
            if result.is_noise:
                noise.append((change, result.reason))
            else:
                relevant.append(change)

        logger.info(f"Filtered {len(noise)} noise items, kept {len(relevant)} relevant changes")
        return relevant, noise

    def _extract_text_from_change(self, change: Any) -> str:
        """Extract text content from a change object."""
        text_parts = []

        # Try common attributes
        for attr in [
            "phrase",
            "description",
            "old_value",
            "new_value",
            "old_text",
            "new_text",
            "text",
        ]:
            if hasattr(change, attr):
                value = getattr(change, attr)
                if value:
                    text_parts.append(str(value))

        # Try dictionary access
        if isinstance(change, dict):
            for key in [
                "phrase",
                "description",
                "old_value",
                "new_value",
                "old_text",
                "new_text",
                "text",
            ]:
                if key in change and change[key]:
                    text_parts.append(str(change[key]))

        return " ".join(text_parts)

    def _contains_important_keywords(self, text: str) -> bool:
        """Check if text contains keywords that make it important."""
        text_lower = text.lower()

        for keyword in self.KEEP_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    def _check_noise_patterns(self, text: str) -> FilterResult:
        """Check if text matches noise patterns."""
        for category, patterns in self.compiled_noise.items():
            # Skip date patterns if not filtering
            if category == "date_change" and not self.filter_date_changes:
                continue

            for pattern in patterns:
                if pattern.match(text.strip()):
                    return FilterResult(is_noise=True, reason=category, confidence=0.95)

        return FilterResult(is_noise=False)

    def _is_numeric_only(self, text: str) -> bool:
        """Check if text is only numeric content."""
        # Remove common non-numeric tokens that might appear with numbers
        cleaned = text.strip()

        for pattern in self.compiled_numeric:
            if pattern.match(cleaned):
                return True

        return False


class ChangeQualifier:
    """
    Qualifies changes by assessing their relevance for Desjardins.
    """

    def __init__(self):
        """Initialize change qualifier."""
        self.noise_filter = NoiseFilter()

    def qualify_change(self, change: Any) -> dict:
        """
        Qualify a change with relevance assessment.

        Args:
            change: Change object to qualify

        Returns:
            Dictionary with qualification details
        """
        # First check if noise
        noise_result = self.noise_filter.is_noise(change)

        if noise_result.is_noise:
            return {
                "is_relevant": False,
                "filter_reason": noise_result.reason,
                "confidence": noise_result.confidence,
                "desjardins_relevance": "LOW",
            }

        # Assess Desjardins relevance
        relevance = self._assess_desjardins_relevance(change)

        return {
            "is_relevant": True,
            "filter_reason": None,
            "confidence": 0.9,
            "desjardins_relevance": relevance,
        }

    def _assess_desjardins_relevance(self, change: Any) -> str:
        """Assess relevance specifically for Desjardins."""
        text = ""

        # Extract text
        for attr in ["description", "new_text", "new_value"]:
            if hasattr(change, attr):
                value = getattr(change, attr)
                if value:
                    text += " " + str(value)

        text_lower = text.lower()

        # High relevance: regulatory changes
        high_keywords = ["réglementaire", "bâle", "bsif", "amf", "obligatoire", "exigence"]
        if any(kw in text_lower for kw in high_keywords):
            return "HIGH"

        # Medium relevance: industry trends
        medium_keywords = [
            "risque",
            "provision",
            "esg",
            "climat",
            "ia",
            "intelligence artificielle",
        ]
        if any(kw in text_lower for kw in medium_keywords):
            return "MEDIUM"

        return "LOW"


def filter_noise(changes: list) -> list:
    """
    Convenience function to filter noise from changes.

    Args:
        changes: List of changes

    Returns:
        Filtered list of relevant changes
    """
    noise_filter = NoiseFilter()
    relevant, _ = noise_filter.filter_changes(changes)
    return relevant
