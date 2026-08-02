"""Package des vérifications de contrôle qualité modulaires (completeness, indicator, schema)."""

from __future__ import annotations

from vigilance.quality.checks.completeness_check import check_extraction_completeness
from vigilance.quality.checks.indicator_check import check_indicator_consistency
from vigilance.quality.checks.schema_check import check_schema_compliance

__all__ = [
    "check_extraction_completeness",
    "check_indicator_consistency",
    "check_schema_compliance",
]
