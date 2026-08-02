"""Tests unitaires pour les sous-composants de contrôle qualité (checks)."""

from __future__ import annotations

from vigilance.quality.checks import (
    check_extraction_completeness,
    check_indicator_consistency,
    check_schema_compliance,
)


def test_check_extraction_completeness() -> None:
    res = check_extraction_completeness({"tables": [{"title": "Ratio CET1"}]})
    assert res["is_complete"] is True
    assert res["table_count"] == 1

    empty_res = check_extraction_completeness({})
    assert empty_res["is_complete"] is False


def test_check_indicator_consistency() -> None:
    valid = check_indicator_consistency([{"label": "CET1", "value": "12.5%"}])
    assert valid["is_valid"] is True

    invalid = check_indicator_consistency([{"value": "12.5%"}])
    assert invalid["is_valid"] is False


def test_check_schema_compliance() -> None:
    res = check_schema_compliance({"bank_code": "rbc", "year": "2025", "quarter": "t4"})
    assert res["is_compliant"] is True

    missing_res = check_schema_compliance({"bank_code": "rbc"})
    assert missing_res["is_compliant"] is False
