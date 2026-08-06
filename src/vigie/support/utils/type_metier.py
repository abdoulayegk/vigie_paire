"""Utilitaires de correspondance entre sections et types metier."""

from __future__ import annotations

from vigie.extraction.section_taxonomy import canonicalize_section


def compute_type_metier(section: str | None, change_type: str | None) -> str:
    """Associe une section ou un type de changement a un code de famille metier.

    Args:
        section: Nom de section brut (ex. ``"regulatory_updates"``).
        change_type: Type de changement brut (ex. ``"reg_update"``).

    Returns:
        Code famille metier : ``"IFC"``, ``"RG"`` ou ``"PB"``.
    """
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
