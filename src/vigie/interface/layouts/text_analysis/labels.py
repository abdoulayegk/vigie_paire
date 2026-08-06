"""Libelles, badges et tables de correspondance de l'onglet analyse textuelle.

Extrait de ``page_text_analysis.py`` sans modification.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc

_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_IMPACT_ORDER: dict[str, int] = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}

_IMPACT_BADGE: dict[str, tuple[str, str]] = {
    "MAJEUR": ("Majeur", "danger"),
    "MODERE": ("Modéré", "warning"),
    "MINEUR": ("Mineur", "secondary"),
}

_POSTURE_BADGE: dict[str, tuple[str, str]] = {
    "RENFORCEMENT": ("Posture renforcée", "success"),
    "ALLEGEMENT": ("Posture allégée", "warning"),
    "NOUVEAU_DISPOSITIF": ("Nouveau dispositif", "primary"),
    "RETRAIT_DISPOSITIF": ("Dispositif retiré", "danger"),
    "AUCUN": ("Posture inchangée", "secondary"),
}

_IMPLEMENTATION_DETAIL_LABEL: dict[str, str] = {
    "ANNONCE": "Annoncée",
    "PLANIFIE": "Planifiée",
    "EN_COURS": "En cours",
    "MIS_EN_OEUVRE": "Mise en œuvre",
    "INDETERMINE": "Indéterminée",
}

_POSTURE_CONFIDENCE_DETAIL_LABEL: dict[str, str] = {
    "ELEVEE": "Élevée",
    "MOYENNE": "Moyenne",
    "FAIBLE": "Faible",
    "INDETERMINE": "Indéterminée",
}

_DIFF_LABELS: dict[str, str] = {
    "added": "Ajouté",
    "removed": "Supprimé",
    "modified": "Modifié",
    "renamed": "Renommé",
}

_THEMES_AMF_SHORT: dict[str, str] = {
    "DIVULGATION_AJOUT": "Ajout divulgation",
    "DIVULGATION_RETRAIT": "Retrait divulgation",
    "MODIFICATION_TEXTE_RISQUE": "Modif. texte risque",
    "MODIFICATION_METHODOLOGIE": "Modif. méthodologie",
    "FACTEUR_RISQUE_CHANGEMENT": "Facteur risque",
    "CAPITAL_REGLEMENTAIRE": "Capital régl.",
    "LIQUIDITE": "Liquidité",
    "FONDS_PROPRES_REGLEMENTAIRES": "Fonds propres",
    "EXIGENCES_REGLEMENTAIRES": "Exigences régl.",
    "RATIOS_REGLEMENTAIRES": "Ratios régl.",
    "STRUCTURE_RAPPORT": "Structure rapport",
    "HYPOTHESES_EXPLICATIONS_RISQUES": "Hypothèses risques",
    "ESG_CLIMATIQUE": "ESG / Climat",
    "RISQUE_EMERGENT": "Risque émergent",
    "RISQUE_DONNEES": "Risque données",
    "RISQUE_TIERS_CLOUD": "Tiers / Cloud",
    "RISQUE_MACRO_GEOPOLITIQUE": "Commercial / géopolitique",
    "GOUVERNANCE_RISQUES": "Gouvernance",
    "CONTROLE_CONFORMITE": "Contrôle / Conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "Nouvelle mention régl.",
    "MONTANT_REGLEMENTAIRE": "Montant régl.",
}

_IMPACT_DOMAIN_BY_THEME: dict[str, str] = {
    "CAPITAL_REGLEMENTAIRE": "capital réglementaire",
    "LIQUIDITE": "liquidité",
    "FONDS_PROPRES_REGLEMENTAIRES": "fonds propres réglementaires",
    "EXIGENCES_REGLEMENTAIRES": "exigences réglementaires",
    "RATIOS_REGLEMENTAIRES": "ratios prudentiels",
    "MONTANT_REGLEMENTAIRE": "seuils réglementaires",
    "ESG_CLIMATIQUE": "ESG / climat",
    "RISQUE_EMERGENT": "risques émergents",
    "RISQUE_DONNEES": "données",
    "RISQUE_TIERS_CLOUD": "tiers / cloud",
    "RISQUE_MACRO_GEOPOLITIQUE": "risques macroéconomiques / géopolitiques",
    "GOUVERNANCE_RISQUES": "gouvernance des risques",
    "CONTROLE_CONFORMITE": "contrôle / conformité",
    "NOUVELLE_MENTION_REGLEMENTAIRE": "réglementation",
    "MODIFICATION_METHODOLOGIE": "méthodologie de risque",
    "MODIFICATION_TEXTE_RISQUE": "gestion des risques",
    "FACTEUR_RISQUE_CHANGEMENT": "facteurs de risque",
    "HYPOTHESES_EXPLICATIONS_RISQUES": "hypothèses de risque",
}

_IMPACT_DOMAIN_PRIORITY = (
    "RISQUE_DONNEES",
    "RISQUE_TIERS_CLOUD",
    "RISQUE_EMERGENT",
    "ESG_CLIMATIQUE",
    "RISQUE_MACRO_GEOPOLITIQUE",
    "CAPITAL_REGLEMENTAIRE",
    "LIQUIDITE",
    "FONDS_PROPRES_REGLEMENTAIRES",
    "RATIOS_REGLEMENTAIRES",
    "MONTANT_REGLEMENTAIRE",
    "EXIGENCES_REGLEMENTAIRES",
    "CONTROLE_CONFORMITE",
    "GOUVERNANCE_RISQUES",
    "MODIFICATION_METHODOLOGIE",
    "FACTEUR_RISQUE_CHANGEMENT",
    "MODIFICATION_TEXTE_RISQUE",
    "HYPOTHESES_EXPLICATIONS_RISQUES",
    "NOUVELLE_MENTION_REGLEMENTAIRE",
)

_TRIAGE_DETAIL_LABELS = (
    "Nouvel élément à surveiller",
    "Sujet détecté",
    "Ce qui change",
    "Pertinence métier",
    "Point de surveillance",
)

_ACTION_BADGE: dict[str, tuple[str, str]] = {
    "revue_prioritaire": ("Revue prioritaire", "danger"),
    "investigation": ("Analyse approfondie", "warning"),
    "confirmation": ("Confirmation", "success"),
    "information": ("Information", "info"),
    "aucune": ("Aucune", "secondary"),
}

_TEXT_REVIEW_STATUS_BADGES: dict[str, tuple[str, str]] = {
    "approved": ("Validé", "success"),
    "rejected": ("Rejeté", "danger"),
    "skipped": ("Passé", "secondary"),
}

_UNSET = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _badge(label: str, color: str, **kwargs) -> dbc.Badge:
    """Construit un ``dbc.Badge`` standardisé pour la page d'analyse texte."""
    return dbc.Badge(label, color=color, className="me-1", **kwargs)


def _plural_count(count: int, singular: str, plural: str) -> str:
    """Retourne un libellé compté avec accord simple."""
    return f"{count} {singular if count == 1 else plural}"
