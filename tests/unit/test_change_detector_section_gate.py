"""Section gating regression tests for comparison.change_detector."""

from __future__ import annotations

from vigilance.comparison.change_detector import ChangeDetector


def test_change_detector_does_not_match_cross_section_tables() -> None:
    doc_t1 = {
        "quarter": "t1",
        "year": 2025,
        "all_tables": [
            {
                "table_id": "t1_23",
                "title": "TABLEAU 23 - Prêts hypothécaires à l'habitation",
                "section": "capital_management",
                "headers": ["Région", "Valeur"],
                "rows": [["Atlantique", "1"], ["Québec", "2"]],
                "page_number": 10,
            }
        ],
    }
    doc_t2 = {
        "quarter": "t2",
        "year": 2025,
        "all_tables": [
            {
                "table_id": "t2_23",
                "title": "TABLEAU 23 - Prêts hypothécaires à l'habitation",
                "section": "risk_management",
                "headers": ["Région", "Valeur"],
                "rows": [["Atlantique", "100"], ["Québec", "200"]],
                "page_number": 12,
            }
        ],
    }

    detector = ChangeDetector(filter_noise=False)
    result = detector.compare_documents(doc_t1, doc_t2, bank_code="rbc")

    assert result.total_changes == 2
    descriptions = {change.description for change in result.changes}
    assert "Nouveau tableau: TABLEAU 23 - Prêts hypothécaires à l'habitation" in descriptions
    assert "Tableau supprimé: TABLEAU 23 - Prêts hypothécaires à l'habitation" in descriptions
    assert all(change.metadata.get("match_reason") in {"no_candidate_same_section", "unknown_section"} for change in result.changes)
