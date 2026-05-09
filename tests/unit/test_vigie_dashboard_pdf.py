from __future__ import annotations

from vigilance.dash_app.callbacks.vigie_dashboard_flow import _build_pdf_report


def test_build_pdf_report_generates_pdf_bytes() -> None:
    pdf_bytes = _build_pdf_report(
        bank="BNC",
        current_label="Q2-2025",
        previous_label="Q1-2025",
        updated_at="09 May 2026 a 02h20",
        text_metrics={
            "total": 25,
            "major": 4,
            "moderate": 5,
            "relevant": 10,
            "analyzed": 25,
            "top": [
                {
                    "section": "Gestion du capital",
                    "impact": "Majeur",
                    "summary": "Sous-section supprimee sur les exigences de communication publique.",
                }
            ],
        },
        indicator_metrics={
            "total_changes": 14,
            "priority": 5,
            "low_confidence": 0,
            "indicator_added": 5,
            "indicator_removed": 1,
            "indicator_renamed": 0,
            "footnote_added": 2,
            "footnote_removed": 3,
            "footnote_modified": 0,
            "footnote_renamed": 0,
            "tables_added": 1,
            "tables_removed": 2,
        },
        review_counts={"total": 9, "pending": 9},
        bars={"Ajouts": 13, "Suppressions": 7, "Modifications": 19, "Renommages": 0},
        priority_rows=[
            {
                "title": "Exigences - Ratios des fonds propres",
                "change": "Note(s) modifiee(s)",
                "impact": "Moyen",
                "confidence": "Elevee",
                "status_label": "En attente",
            }
        ],
        text_data={
            "section_changes": [
                {
                    "section_title": "Gestion du capital",
                    "changes": [
                        {
                            "diff_type": "modified",
                            "change_summary": "Modification narrative detectee.",
                        }
                    ],
                }
            ]
        },
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1_000
    assert pdf_bytes.count(b"/Type /Page") >= 8
