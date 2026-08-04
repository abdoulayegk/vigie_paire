from __future__ import annotations

from typing import Any

from vigie.interface.callbacks.dashboard_flow import render_results


def _component_texts(node: Any) -> list[str]:
    if node is None:
        return []
    if isinstance(node, str | int | float):
        return [str(node)]
    if isinstance(node, list | tuple):
        texts: list[str] = []
        for child in node:
            texts.extend(_component_texts(child))
        return texts
    return _component_texts(getattr(node, "children", None))


def _indicator_payload() -> dict[str, Any]:
    return {
        "bank_code": "bnc",
        "summary": {
            "tables_matched": 1,
            "total_added_indicators": 1,
            "total_removed_indicators": 2,
            "total_renamed_indicators": 3,
        },
        "table_comparisons": [
            {
                "table_name": "Capital",
                "added_indicators": [{"name": "CET1"}],
                "footnotes_counts": {"added": 2, "removed": 1, "modified": 3},
            }
        ],
        "tables_added": [],
        "tables_removed": [],
    }


def test_results_dashboard_shows_split_footnote_kpis() -> None:
    _header, _summary, kpis = render_results(
        comparison=None,
        indicator=_indicator_payload(),
        show_results=True,
    )

    text = " ".join(_component_texts(kpis))

    assert "Notes ajoutées" in text
    assert "Notes supprimées" in text
    assert "Notes modifiées" in text
    assert "Nouvelles notes de bas de tableau (total notes: 6)" in text
    assert "Notes présentes avant, absentes maintenant" in text
    assert "Notes dont le contenu ou la portée change" in text


def test_results_dashboard_shows_split_indicator_kpis() -> None:
    _header, _summary, kpis = render_results(
        comparison=None,
        indicator=_indicator_payload(),
        show_results=True,
    )

    text = " ".join(_component_texts(kpis))

    assert "Indicateurs ajoutés" in text
    assert "Indicateurs retirés" in text
    assert "Renommages" in text
    assert "Nouveaux indicateurs identifiés" in text
    assert "Indicateurs présents avant, absents maintenant" in text
    assert "Libellés rapprochés comme renommages" in text
    assert "Ajouts, suppressions et renommages d'indicateurs" not in text
