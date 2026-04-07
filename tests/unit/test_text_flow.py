from __future__ import annotations

from typing import Any

from vigilance.dash_app.callbacks.text_flow import download_text_excel


def test_download_text_excel_reload_latest_payload_before_export(monkeypatch) -> None:
    stale_payload = {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
        "section_comparisons": [
            {"block_comparisons": [{"change_id": "strict"}], "all_block_comparisons": []}
        ],
    }
    latest_payload = {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
        "section_comparisons": [
            {"block_comparisons": [{"change_id": "strict"}], "all_block_comparisons": [{"change_id": "all"}]}
        ],
    }
    captured: dict[str, Any] = {}

    def _fake_load_text_comparison_for_dash(**kwargs):
        captured["load_kwargs"] = kwargs
        return latest_payload

    def _fake_generate_text_comparison_excel(payload, output_path=None):
        captured["export_payload"] = payload
        assert output_path is None
        return b"excel-bytes"

    monkeypatch.setattr(
        "vigilance.dash_app.services.text_comparison_store.load_text_comparison_for_dash",
        _fake_load_text_comparison_for_dash,
    )
    monkeypatch.setattr(
        "vigilance.text_comparison.generate_text_comparison_excel",
        _fake_generate_text_comparison_excel,
    )

    response = download_text_excel(1, stale_payload)

    assert captured["load_kwargs"] == {
        "bank_code": "td",
        "quarter_current": "2026_t1",
        "quarter_previous": "2025_t3",
    }
    assert captured["export_payload"] is latest_payload
    assert response["filename"] == "veille_textuelle_TD_2026t1.xlsx"
