"""
Hard Negative Checks: regles bloquantes pour eviter faux positifs.

- Incompatibilite table_type (requirements vs observed_results)
- Fingerprint headers (dates vs exigences)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Mots-cles reglementaires dans les headers (t2)
REGULATORY_HEADER_KEYWORDS = frozenset(
    {
        "minimum",
        "reserve",
        "reserves",
        "surcharge",
        "surcharges",
        "bsif",
        "ccbc",
    }
)

# Patterns indiquant des valeurs/dates dans les headers (t1)
VALUE_HEADER_PATTERNS = [
    re.compile(r"au\s+31", re.IGNORECASE),
    re.compile(r"as\s+at\s+\d", re.IGNORECASE),
    re.compile(r"20\d{2}", re.IGNORECASE),
    re.compile(r"en\s+millions", re.IGNORECASE),
    re.compile(r"in\s+millions", re.IGNORECASE),
]


@dataclass
class HardNegativeResult:
    """Resultat du check Hard Negative."""

    triggered: bool
    reason: str
    can_override: bool = False  # True si table_number + header_overlap suffisent


def check_hard_negative(
    table_type_t1: str,
    table_type_t2: str,
    table_number_match: bool = False,
    header_overlap: float = 0.0,
    headers_t1: Optional[list[str]] = None,
    headers_t2: Optional[list[str]] = None,
    robust_indicator_score: Optional[float] = None,
    override_indicator_min: float = 0.30,
) -> HardNegativeResult:
    """
    Applique les regles Hard Negative.

    Override (table_number + header + optional indicator support): when
    robust_indicator_score is provided, override also requires
    robust_indicator_score >= override_indicator_min.

    Args:
        table_type_t1: Type du tableau 1 (requirements | observed_results | unknown)
        table_type_t2: Type du tableau 2
        table_number_match: True si numeros de tableau identiques
        header_overlap: Jaccard/overlap des headers (0-1)
        headers_t1: En-tetes du tableau 1
        headers_t2: En-tetes du tableau 2
        robust_indicator_score: Optional robust indicator score for override check
        override_indicator_min: Min robust indicator score to allow override (when provided)

    Returns:
        HardNegativeResult avec triggered, reason, can_override
    """
    def _override_ok() -> bool:
        if not table_number_match or header_overlap < 0.85:
            return False
        if robust_indicator_score is not None:
            return robust_indicator_score >= override_indicator_min
        return True

    if table_type_t1 not in ("requirements", "observed_results", "unknown"):
        table_type_t1 = "unknown"
    if table_type_t2 not in ("requirements", "observed_results", "unknown"):
        table_type_t2 = "unknown"

    if table_type_t1 != "unknown" and table_type_t2 != "unknown" and table_type_t1 != table_type_t2:
        if _override_ok():
            return HardNegativeResult(
                triggered=False,
                reason="table_type mismatch overridden by table_number_match and header_overlap>=0.85",
                can_override=True,
            )
        return HardNegativeResult(
            triggered=True,
            reason=f"table_type mismatch: {table_type_t1} vs {table_type_t2}",
            can_override=table_number_match and header_overlap >= 0.85,
        )

    headers_1 = headers_t1 or []
    headers_2 = headers_t2 or []
    if not headers_1 or not headers_2:
        return HardNegativeResult(triggered=False, reason="", can_override=False)

    headers_1_joined = " ".join(str(h) for h in headers_1).lower()
    headers_2_joined = " ".join(str(h) for h in headers_2).lower()

    regulatory_count = sum(1 for kw in REGULATORY_HEADER_KEYWORDS if kw in headers_2_joined)
    value_match = any(p.search(headers_1_joined) for p in VALUE_HEADER_PATTERNS)

    if regulatory_count >= 2 and value_match:
        if _override_ok():
            return HardNegativeResult(
                triggered=False,
                reason="header fingerprint mismatch overridden by table_number and header_overlap",
                can_override=True,
            )
        return HardNegativeResult(
            triggered=True,
            reason="header fingerprint: t2 has regulatory cols, t1 has dates/values",
            can_override=table_number_match and header_overlap >= 0.85,
        )

    return HardNegativeResult(triggered=False, reason="", can_override=False)
