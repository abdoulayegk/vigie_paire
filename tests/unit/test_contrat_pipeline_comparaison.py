"""Contrat de compatibilite de la facade du pipeline de comparaison."""

from vigilance import compare_gpt as facade
from vigilance.pipeline_comparaison import ancrages_visuels
from vigilance.pipeline_comparaison import construction_resultat
from vigilance.pipeline_comparaison import orchestration


def test_facade_preserve_les_points_entree_et_helpers_historiques() -> None:
    """Les appelants et monkeypatchs historiques restent compatibles."""
    assert facade._compare_reports_gpt4o_impl is orchestration.compare_reports_gpt4o
    assert facade._archive_source_pdf is construction_resultat._archive_source_pdf
    assert facade._resolve_visual_table_anchor is ancrages_visuels._resolve_visual_table_anchor
    assert facade.COMPARISON_SCHEMA_VERSION == construction_resultat.COMPARISON_SCHEMA_VERSION
    assert callable(facade.compare_reports_gpt4o)
    assert callable(facade._call_openai_json)
