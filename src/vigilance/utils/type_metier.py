"""Business type mapping helpers."""

from __future__ import annotations

from vigilance.extraction.section_taxonomy import canonicalize_section


def compute_type_metier(section: str | None, change_type: str | None) -> str:
    """Map section/change type to a business family code."""
    section_norm = canonicalize_section(section or "")
    change_norm = (change_type or "").lower()

    if section_norm == "regulatory_updates":
        return "IFC"
    if section_norm == "risk_management":
        return "RG"
    if section_norm == "capital_management":
        return "PB"
    if "reg" in change_norm:
        return "IFC"
    if "risk" in change_norm:
        return "RG"
    return "PB"
