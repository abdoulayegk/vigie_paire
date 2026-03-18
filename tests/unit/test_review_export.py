from __future__ import annotations

import io

from openpyxl import load_workbook

from app.review_export import generate_validation_csv, generate_validation_excel
from app.review_models import ReviewItem


def test_generate_validation_csv_includes_comment_in_summary_without_schema_change() -> None:
    item = ReviewItem(
        change_id="chg-1",
        change_type="added",
        indicator="Nouvel indicateur",
        section="capital_management",
        page_t2=12,
        review_status="approved",
        comment="Verifier l'impact superviseur",
        edited_value="Nouvel indicateur ajuste",
        confidence=0.91,
    )

    csv_text = generate_validation_csv(
        [item],
        {
            "bank_code": "bnc",
            "quarter_from": "Q1-2025",
            "quarter_to": "Q2-2025",
            "year": 2025,
        },
    )

    assert "Commentaire analyste: Verifier l'impact superviseur." in csv_text
    assert "Valeur editee: Nouvel indicateur ajuste." in csv_text
    assert "validation_finale" in csv_text


def test_generate_validation_excel_adds_context_and_technical_sheets() -> None:
    item = ReviewItem(
        change_id="fn-1",
        change_type="footnote",
        indicator="+1 note(s)",
        section="risk_management",
        table_name="Table risque",
        page_t1=3,
        page_t2=4,
        review_status="pending",
        comment="A revoir",
        confidence=0.75,
        item_type="footnote",
        indicators=[
            {
                "name": "[1] Ancien texte -> Nouveau texte",
                "type": "modified",
                "review_status": "pending",
            }
        ],
    )

    excel_bytes = generate_validation_excel(
        [item],
        {
            "bank_code": "bnc",
            "quarter_from": "Q1-2025",
            "quarter_to": "Q2-2025",
            "year": 2025,
            "meta": {
                "compare_path": "/tmp/compare.json",
                "extraction_sources": {
                    "previous": {
                        "mode": "stored",
                        "tables_path": "/tmp/t1/tables.json",
                        "indicators_path": "/tmp/t1/indicators.json",
                        "footnotes_path": "/tmp/t1/footnotes.json",
                    },
                    "current": {
                        "mode": "fresh",
                        "tables_path": "/tmp/t2/tables.json",
                        "indicators_path": "/tmp/t2/indicators.json",
                        "footnotes_path": "/tmp/t2/footnotes.json",
                    },
                },
            },
        },
    )

    wb = load_workbook(io.BytesIO(excel_bytes))
    assert "Revue" in wb.sheetnames
    assert "Contexte" in wb.sheetnames
    assert "Technique" in wb.sheetnames

    ws_context = wb["Contexte"]
    context_pairs = {
        str(ws_context.cell(row=i, column=1).value): str(
            ws_context.cell(row=i, column=2).value
        )
        for i in range(2, ws_context.max_row + 1)
    }
    assert context_pairs["compare_path"] == "/tmp/compare.json"
    assert context_pairs["previous_source_mode"] == "stored"
    assert context_pairs["current_source_mode"] == "fresh"

    ws_tech = wb["Technique"]
    headers = [ws_tech.cell(row=1, column=i).value for i in range(1, ws_tech.max_column + 1)]
    assert "comment" in headers
