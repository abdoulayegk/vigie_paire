from __future__ import annotations

from vigilance.text_comparison.change_segments import build_change_segments_from_texts


def test_build_change_segments_detects_inline_removed_committee() -> None:
    text_t1 = (
        "Des rapports sur le profil de risques sont soumis périodiquement "
        "et en temps opportun au CGRO, au CRG et au CGR."
    )
    text_t2 = (
        "Des rapports sur le profil de risques sont soumis périodiquement "
        "et en temps opportun au CGRO et au CGR."
    )

    segments = build_change_segments_from_texts(text_t1, text_t2, diff_type="modified")

    assert segments == [{"kind": "removed", "text_t1": ", au CRG", "text_t2": ""}]


def test_build_change_segments_detects_modified_ratio() -> None:
    segments = build_change_segments_from_texts(
        "Le seuil prudentiel CET1 minimal applicable est de 4,5 %.",
        "Le seuil prudentiel CET1 minimal applicable est de 5,0 %.",
        diff_type="modified",
    )

    assert segments == [{"kind": "modified", "text_t1": "4,5", "text_t2": "5,0"}]


def test_build_change_segments_uses_word_groups_not_character_fragments() -> None:
    text_t1 = (
        "Le BSIF a annoncé des changements proposés à sa ligne directrice Normes de "
        "liquidité (NL) qui devraient entrer en vigueur au cours de l'exercice 2025."
    )
    text_t2 = (
        "Le BSIF a mené une consultation sectorielle sur les changements proposés à sa "
        "ligne directrice Normes de liquidité (NL) qui devraient entrer en vigueur au "
        "cours du troisième trimestre de l'exercice 2026."
    )

    segments = build_change_segments_from_texts(text_t1, text_t2, diff_type="modified")

    assert segments == [
        {
            "kind": "modified",
            "text_t1": "a annoncé des",
            "text_t2": "a mené une consultation sectorielle sur les",
        },
        {
            "kind": "modified",
            "text_t1": "au cours de l'exercice 2025.",
            "text_t2": "au cours du troisième trimestre de l'exercice 2026.",
        },
    ]
    for segment in segments:
        assert all(len(value) > 2 for value in segment.values() if value)
