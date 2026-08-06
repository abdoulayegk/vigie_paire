from __future__ import annotations

from vigie.analyse_texte.text_comparison.change_segments import build_change_segments_from_texts


def test_build_change_segments_detects_inline_removed_committee() -> None:
    text_t1 = (
        "Des rapports sur le profil de risques sont soumis périodiquement "
        "et en temps opportun au CGRO, au CRG et au CGR."
    )
    text_t2 = "Des rapports sur le profil de risques sont soumis périodiquement et en temps opportun au CGRO et au CGR."

    segments = build_change_segments_from_texts(text_t1, text_t2, diff_type="modified")

    assert segments == [{"kind": "removed", "text_t1": ", au CRG", "text_t2": ""}]


def test_build_change_segments_detects_modified_ratio() -> None:
    segments = build_change_segments_from_texts(
        "Le seuil prudentiel CET1 minimal applicable est de 4,5 %.",
        "Le seuil prudentiel CET1 minimal applicable est de 5,0 %.",
        diff_type="modified",
    )

    assert segments == [{"kind": "modified", "text_t1": "4,5", "text_t2": "5,0"}]
