"""RBC-specific regression tests for the Adaptive Table Pairing Engine."""

from __future__ import annotations

import pytest

from vigilance.compare.table_pairing_engine import run_strict_intra_section_compare
from vigilance.models.table_models import TableArtifact


def _rbc_table(
    tid: str,
    *,
    section: str = "risk_management",
    title: str = "",
    indicators: list[str],
    page: int,
    headers: list[str] | None = None,
    title_reliability: str | None = None,
) -> TableArtifact:
    return TableArtifact(
        bank_code="rbc",
        section=section,
        page_pdf=page,
        table_id=tid,
        title=title,
        headers=headers or ["Indicateur", "Valeur"],
        rows=[[label, "1"] for label in indicators],
        first_column_indicators=list(indicators),
        first_column_indicators_raw=list(indicators),
        extraction_method="vision_full_gpt4o",
        quarter="t2-2025",
        table_number=None,
        footnotes=[],
        content_source="vision_gpt4o",
        title_reliability=title_reliability,
    )


def _build_rbc_fixture() -> tuple[list[TableArtifact], list[TableArtifact]]:
    t1_tables = [
        _rbc_table("t1_cap_cet1", section="capital_management", title="Fonds propres CET1",
                    indicators=["Actions", "Resultats", "AOCI", "Goodwill", "Ajustements", "Total CET1"],
                    page=5, title_reliability="reliable"),
        _rbc_table("t1_cap_leverage", section="capital_management", title="Ratio de levier",
                    indicators=["FP categorie 1", "Expositions bilan", "Derives", "Hors bilan", "Total expositions", "Ratio"],
                    page=6, title_reliability="reliable"),
        _rbc_table("t1_risk_credit", section="risk_management", title="Prets par region",
                    indicators=["Canada", "Etats-Unis", "Europe", "Asie", "Autres", "Total", "Provisions"],
                    page=10, title_reliability="reliable"),
        _rbc_table("t1_risk_market", section="risk_management", title="Evaluation du risque de marche",
                    indicators=["Taux interet", "Credit", "Actions", "Change", "Matieres premieres", "Diversification", "VAR total",
                                "SVaR", "IRC", "Risque specifique", "Mesure globale"],
                    page=15, title_reliability="reliable"),
        _rbc_table("t1_risk_var_summary", section="risk_management", title="VAR par facteur de risque",
                    indicators=["Taux interet", "Ecart de taux", "Actions", "Change", "Marchandises", "Diversification", "VAR total"],
                    page=20, headers=["Facteur", "Moyenne", "Maximum", "Minimum", "Fin periode"],
                    title_reliability="reliable"),
        _rbc_table("t1_risk_var_detail", section="risk_management", title="VAR par facteur de risque",
                    indicators=["Taux interet", "Ecart de taux", "Actions", "Change", "Marchandises", "Diversification", "VAR total"],
                    page=21, headers=["Facteur", "T2-2025", "T1-2025", "T4-2024", "T3-2024"],
                    title_reliability="reliable"),
        _rbc_table("t1_liq_lcr", section="liquidity_management", title="Ratio de liquidite a court terme",
                    indicators=["HQLA", "Sorties nettes", "LCR", "Excedent"],
                    page=30, title_reliability="reliable"),
        _rbc_table("t1_liq_funding", section="liquidity_management", title="Composition du financement de gros",
                    indicators=["Depot terme", "Papier commercial", "Obligations senior", "Obligations sécurisées",
                                "Titrisation", "Autres", "Total", "Echeance moyenne", "Cout moyen"],
                    page=31, title_reliability="reliable"),
        _rbc_table("t1_credit_quality", section="risk_management", title="Qualite du credit",
                    indicators=["Stade 1", "Stade 2", "Stade 3", "Total provisions"],
                    page=35, title_reliability="reliable"),
        _rbc_table("t1_tiny_a", section="risk_management", title="Notations",
                    indicators=["AAA", "AA", "A", "BBB"],
                    page=36, title_reliability="reliable"),
        _rbc_table("t1_risk_op", section="risk_management", title="Risque operationnel",
                    indicators=["Pertes internes", "Pertes externes", "Provisions", "Capital requis", "Ratio"],
                    page=37, title_reliability="reliable"),
        _rbc_table("t1_cap_dividendes", section="capital_management", title="Principales donnees concernant les actions",
                    indicators=["Dividende par action", "Ratio de distribution", "Rendement", "Cours de cloture",
                                "Actions en circulation", "Capitalisation boursiere", "Valeur comptable", "Ratio cours/valeur",
                                "BPA de base", "BPA dilue"],
                    page=40, title_reliability="reliable"),
        _rbc_table("t1_risk_concentration", section="risk_management", title="Concentration sectorielle",
                    indicators=["Services financiers", "Immobilier", "Energie", "Technologie", "Sante",
                                "Commerce detail", "Industrie", "Autres"],
                    page=38, title_reliability="reliable"),
        _rbc_table("t1_risk_geo", section="risk_management", title="Exposition geographique au risque de credit",
                    indicators=["Canada", "Etats-Unis", "Europe", "Asie-Pacifique", "Autres international", "Total"],
                    page=39, title_reliability="reliable"),
        _rbc_table("t1_liq_nsfr", section="liquidity_management", title="NSFR",
                    indicators=["Financement stable disponible", "Financement stable requis", "NSFR"],
                    page=32, title_reliability="reliable"),
        _rbc_table("t1_risk_interest", section="risk_management", title="Sensibilite au taux interet",
                    indicators=["Hausse 100pb", "Baisse 100pb", "Impact BII", "Impact capitaux propres"],
                    page=22, title_reliability="reliable"),
        _rbc_table("t1_cap_tlac", section="capital_management", title="TLAC",
                    indicators=["Fonds propres", "Dette subordonnee", "Dette senior", "Total TLAC", "APR", "Ratio TLAC"],
                    page=7, title_reliability="reliable"),
        _rbc_table("t1_risk_country", section="risk_management", title="Expositions pays",
                    indicators=["Royaume-Uni", "Allemagne", "France", "Japon", "Chine", "Bresil", "Autres"],
                    page=40, title_reliability="reliable"),
    ]

    t2_tables = [
        _rbc_table("t2_cap_cet1", section="capital_management", title="Fonds propres CET1",
                    indicators=["Actions", "Resultats", "AOCI", "Goodwill", "Ajustements", "Total CET1"],
                    page=7, title_reliability="reliable"),
        _rbc_table("t2_cap_leverage", section="capital_management", title="Ratio de levier",
                    indicators=["FP categorie 1", "Expositions bilan", "Derives", "Hors bilan", "Total expositions", "Ratio"],
                    page=8, title_reliability="reliable"),
        _rbc_table("t2_risk_credit", section="risk_management", title="Prets par region",
                    indicators=["Canada", "Etats-Unis", "Europe", "Asie", "Autres", "Total", "Provisions"],
                    page=12, title_reliability="reliable"),
        _rbc_table("t2_risk_market", section="risk_management", title="Evaluation du risque de marche",
                    indicators=["Taux interet", "Credit", "Actions", "Change", "Matieres premieres", "Diversification", "VAR total",
                                "SVaR", "IRC", "Risque specifique", "Mesure globale"],
                    page=18, title_reliability="reliable"),
        _rbc_table("t2_risk_var_summary", section="risk_management", title="VAR par facteur de risque",
                    indicators=["Taux interet", "Ecart de taux", "Actions", "Change", "Marchandises", "Diversification", "VAR total"],
                    page=25, headers=["Facteur", "Moyenne", "Maximum", "Minimum", "Fin periode"],
                    title_reliability="reliable"),
        _rbc_table("t2_risk_var_detail", section="risk_management", title="VAR par facteur de risque",
                    indicators=["Taux interet", "Ecart de taux", "Actions", "Change", "Marchandises", "Diversification", "VAR total"],
                    page=26, headers=["Facteur", "T3-2025", "T2-2025", "T1-2025", "T4-2024"],
                    title_reliability="reliable"),
        _rbc_table("t2_liq_lcr", section="liquidity_management", title="Ratio de liquidite a court terme",
                    indicators=["HQLA", "Sorties nettes", "LCR", "Excedent"],
                    page=34, title_reliability="reliable"),
        _rbc_table("t2_liq_funding", section="liquidity_management", title="Composition du financement de gros",
                    indicators=["Depot terme", "Papier commercial", "Obligations senior", "Obligations securisees",
                                "Titrisation", "Autres", "Total", "Echeance moyenne", "Cout moyen"],
                    page=35, title_reliability="reliable"),
        _rbc_table("t2_credit_quality", section="risk_management", title="Qualite du credit",
                    indicators=["Stade 1", "Stade 2", "Stade 3", "Total provisions"],
                    page=38, title_reliability="reliable"),
        _rbc_table("t2_tiny_a", section="risk_management", title="Notations",
                    indicators=["AAA", "AA", "A", "BBB"],
                    page=39, title_reliability="reliable"),
        _rbc_table("t2_risk_op", section="risk_management", title="Risque operationnel",
                    indicators=["Pertes internes", "Pertes externes", "Provisions", "Capital requis", "Ratio"],
                    page=40, title_reliability="reliable"),
        _rbc_table("t2_cap_dividendes", section="capital_management", title="Principales donnees concernant les actions",
                    indicators=["Dividende par action", "Ratio de distribution", "Rendement", "Cours de cloture",
                                "Actions en circulation", "Capitalisation boursiere", "Valeur comptable", "Ratio cours/valeur",
                                "BPA de base", "BPA dilue"],
                    page=44, title_reliability="reliable"),
        _rbc_table("t2_risk_concentration", section="risk_management", title="Concentration sectorielle",
                    indicators=["Services financiers", "Immobilier", "Energie", "Technologie", "Sante",
                                "Commerce detail", "Industrie", "Autres"],
                    page=41, title_reliability="reliable"),
        _rbc_table("t2_risk_geo", section="risk_management", title="Exposition geographique au risque de credit",
                    indicators=["Canada", "Etats-Unis", "Europe", "Asie-Pacifique", "Autres international", "Total"],
                    page=42, title_reliability="reliable"),
        _rbc_table("t2_liq_nsfr", section="liquidity_management", title="NSFR",
                    indicators=["Financement stable disponible", "Financement stable requis", "NSFR"],
                    page=36, title_reliability="reliable"),
        _rbc_table("t2_risk_interest", section="risk_management", title="Sensibilite au taux interet",
                    indicators=["Hausse 100pb", "Baisse 100pb", "Impact BII", "Impact capitaux propres"],
                    page=27, title_reliability="reliable"),
        _rbc_table("t2_cap_tlac", section="capital_management", title="TLAC",
                    indicators=["Fonds propres", "Dette subordonnee", "Dette senior", "Total TLAC", "APR", "Ratio TLAC"],
                    page=9, title_reliability="reliable"),
        _rbc_table("t2_risk_country", section="risk_management", title="Expositions pays",
                    indicators=["Royaume-Uni", "Allemagne", "France", "Japon", "Chine", "Bresil", "Autres"],
                    page=43, title_reliability="reliable"),
    ]
    return t1_tables, t2_tables


def test_rbc_regression_pairing_coverage() -> None:
    t1s, t2s = _build_rbc_fixture()
    assert len(t1s) >= 18
    assert len(t2s) >= 18
    result = run_strict_intra_section_compare(t1s, t2s, bank_code="rbc")
    coverage = result.get("pairing_coverage", 0.0)
    assert coverage >= 0.90, f"Coverage {coverage:.1%} below 90% threshold"


def test_rbc_var_family_no_cross_contamination() -> None:
    t1s, t2s = _build_rbc_fixture()
    result = run_strict_intra_section_compare(t1s, t2s, bank_code="rbc")
    pairs = result["pairs"]
    pair_map = {p["t1_table_id"]: p["t2_table_id"] for p in pairs}
    if "t1_risk_var_summary" in pair_map:
        assert pair_map["t1_risk_var_summary"] == "t2_risk_var_summary"
    if "t1_risk_var_detail" in pair_map:
        assert pair_map["t1_risk_var_detail"] == "t2_risk_var_detail"


def test_rbc_tiny_tables_matched() -> None:
    t1s, t2s = _build_rbc_fixture()
    result = run_strict_intra_section_compare(t1s, t2s, bank_code="rbc")
    pairs = result["pairs"]
    pair_map = {p["t1_table_id"]: p["t2_table_id"] for p in pairs}
    assert "t1_tiny_a" in pair_map, "Tiny table t1_tiny_a should be matched"


def test_rbc_diagnostics_contain_rescue_count() -> None:
    t1s, t2s = _build_rbc_fixture()
    result = run_strict_intra_section_compare(t1s, t2s, bank_code="rbc")
    assert "rescued_matches_count" in result
    assert "diagnostics" in result
