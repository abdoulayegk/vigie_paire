"""Modele Pydantic du resultat de comparaison indicateurs avant dump JSON."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReferenceResolution(BaseModel):
    """Regle de resolution du trimestre de reference."""

    model_config = ConfigDict(extra="allow")

    mode: str = "automatique"
    year_previous: int
    quarter_previous: str
    rule: str = ""


class MatchingBlock(BaseModel):
    """Bloc matching ecrit dans comparison.json."""

    model_config = ConfigDict(extra="allow")

    matched_pairs: list[Any] = Field(default_factory=list)
    tables_added: list[Any] = Field(default_factory=list)
    tables_removed: list[Any] = Field(default_factory=list)
    artifacts_confirmed_previous: list[Any] = Field(default_factory=list)
    artifacts_confirmed_current: list[Any] = Field(default_factory=list)
    extraction_suspects_previous: list[Any] = Field(default_factory=list)
    extraction_suspects_current: list[Any] = Field(default_factory=list)
    boundary_scope_exclusions_previous: list[Any] = Field(default_factory=list)
    boundary_scope_exclusions_current: list[Any] = Field(default_factory=list)


class ComparisonSummary(BaseModel):
    """Resume numerique du run de comparaison."""

    model_config = ConfigDict(extra="allow")

    matched_pairs_total: int = 0
    tables_added_total: int = 0
    tables_removed_total: int = 0
    artifacts_confirmed_previous_total: int = 0
    artifacts_confirmed_current_total: int = 0
    extraction_suspects_previous_total: int = 0
    extraction_suspects_current_total: int = 0
    boundary_scope_exclusions_previous_total: int = 0
    boundary_scope_exclusions_current_total: int = 0
    indicator_changes_total: int = 0
    footnote_changes_total: int = 0
    high_priority_items_total: int = 0


class ComparisonRunResult(BaseModel):
    """Resultat complet d'un run de comparaison (schema_version 3).

    Serialize via ``model_dump(mode="json")`` uniquement a la frontiere
    fichier / Dash — ne change pas la forme publique de ``comparison.json``.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = 3
    artifact_type: str = "report_comparison"
    run_id: str = ""
    bank_code: str = ""
    year_previous: int = 0
    quarter_previous: str = ""
    year_current: int = 0
    quarter_current: str = ""
    created_at: str = ""
    source_pdf_previous: str = ""
    source_pdf_current: str = ""
    archived_pdf_previous: str = ""
    archived_pdf_current: str = ""
    model_version: str = ""
    prompt_version_match: str = ""
    prompt_version_diff: str = ""
    reference_resolution: ReferenceResolution | dict[str, Any]
    matching: MatchingBlock
    pair_comparisons: list[Any] = Field(default_factory=list)
    run_metrics: dict[str, Any] = Field(default_factory=dict)
    summary: ComparisonSummary

    def to_json_dict(self) -> dict[str, Any]:
        """Dump stable pour ``comparison.json`` et consommation Dash."""
        return self.model_dump(mode="json")
