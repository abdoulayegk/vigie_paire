from __future__ import annotations

from pathlib import Path

from vigilance import ui_io


def test_get_available_indicator_comparison_options_recurses_and_skips_review_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    compare_root = tmp_path / "comparisons"
    nested = compare_root / "bnc" / "2026_t1_vs_2025_t3" / "20260323_143015"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "comparison.json").write_text("{}", encoding="utf-8")
    (nested / "comparison.review_state.json").write_text("{}", encoding="utf-8")
    (compare_root / "legacy.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ui_io, "INDICATOR_COMPARISON_DIR", compare_root)

    options = ui_io.get_available_indicator_comparison_options()

    values = {item["value"] for item in options}
    assert "bnc/2026_t1_vs_2025_t3/20260323_143015/comparison.json" in values
    assert "legacy.json" not in values
    assert (
        "bnc/2026_t1_vs_2025_t3/20260323_143015/comparison.review_state.json"
        not in values
    )
