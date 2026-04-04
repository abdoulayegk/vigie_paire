"""Fonctions d'evaluation des changements destinees a l'analyste pour les sorties de comparaison."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_HIGH_IMPACT_THEMES = frozenset({"capital", "liquidite", "financement"})
_RISK_THEMES = frozenset({"risque_credit", "risque_marche", "risque_operationnel"})

_SECTION_THEME_MAP = {
    "capital": "capital",
    "capital_management": "capital",
    "liquidite": "liquidite",
    "liquidity": "liquidite",
    "funding": "financement",
    "funding_management": "financement",
}

_TITLE_THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "risque_credit",
        (
            "risque de credit",
            "credit risk",
            "perte de credit",
            "pertes de credit",
            "expected credit",
            "expositions au risque de credit",
            "provisions pour pertes sur creances",
        ),
    ),
    (
        "risque_marche",
        (
            "risque de marche",
            "market risk",
            "trading",
            "valeur a risque",
            "value at risk",
            "var",
        ),
    ),
    (
        "risque_operationnel",
        (
            "risque operationnel",
            "operational risk",
            "cyber",
            "fraude",
        ),
    ),
    (
        "liquidite",
        (
            "liquidite",
            "liquidity",
            "lcr",
            "nsfr",
            "hqla",
            "ratio structurel de liquidite",
        ),
    ),
    (
        "capital",
        (
            "capital",
            "fonds propres",
            "cet1",
            "tier 1",
            "tlac",
            "levier",
            "leverage",
        ),
    ),
    (
        "financement",
        (
            "financement",
            "funding",
            "depot",
            "depots",
        ),
    ),
]

_INDICATOR_THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "capital",
        (
            "ratio cet1",
            "ratio tier 1",
            "ratio total",
            "ratio de levier",
            "tlac",
        ),
    ),
    (
        "liquidite",
        (
            "lcr",
            "nsfr",
            "hqla",
            "liquidite",
            "ratio structurel de liquidite",
        ),
    ),
    (
        "risque_credit",
        (
            "pertes de credit",
            "perte de credit",
            "expected credit",
            "provision",
            "provisions",
            "exposition",
            "expositions",
            "ecl",
        ),
    ),
    (
        "risque_marche",
        (
            "trading",
            "var",
            "value at risk",
            "valeur a risque",
            "sensibilite",
        ),
    ),
    (
        "risque_operationnel",
        (
            "operational",
            "operationnel",
            "cyber",
            "fraude",
        ),
    ),
    (
        "financement",
        (
            "depots",
            "depot",
            "funding",
            "financement",
        ),
    ),
]


def _ascii_fold(text: str) -> str:
    """Supprime les accents et passe en minuscules via decomposition NFKD."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text or ""))
        if not unicodedata.combining(c)
    ).lower()


def _theme_text(*parts: Any) -> str:
    """Concatene et normalise les parties de texte pour la detection de theme."""
    raw = " ".join(str(part or "") for part in parts if str(part or "").strip())
    folded = _ascii_fold(raw)
    folded = re.sub(r"[^a-z0-9\s]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _match_theme_from_text(
    text: str,
    rules: list[tuple[str, tuple[str, ...]]],
) -> str | None:
    """Retourne le premier theme dont un jeton apparait dans le texte, ou ``None``."""
    for theme, tokens in rules:
        if any(token in text for token in tokens):
            return theme
    return None


def _classify_theme(
    *,
    section: str,
    title: str,
    indicators: list[str],
    footnotes: list[dict[str, str]],
) -> str:
    """Classifie le theme reglementaire d'un tableau a partir de ses metadonnees.

    Applique une hierarchie de regles : titre, section, indicateurs, puis
    notes de bas de page. Retourne ``"autre"`` si aucun theme n'est detecte.

    Args:
        section: Nom de la section du rapport.
        title: Titre du tableau.
        indicators: Liste des libelles d'indicateurs.
        footnotes: Liste de dicts ``{"id": ..., "text": ...}`` des notes de bas de page.

    Returns:
        Identifiant du theme (ex. ``"capital"``, ``"liquidite"``, ``"autre"``).
    """
    title_theme = _match_theme_from_text(_theme_text(title), _TITLE_THEME_RULES)
    if title_theme:
        return title_theme

    section_key = str(section or "").strip().lower()
    if section_key in _SECTION_THEME_MAP:
        return _SECTION_THEME_MAP[section_key]

    section_theme = _match_theme_from_text(_theme_text(section), _TITLE_THEME_RULES)
    if section_theme:
        return section_theme

    indicator_theme = _match_theme_from_text(
        _theme_text(" ".join(indicators)),
        _INDICATOR_THEME_RULES,
    )
    if indicator_theme:
        return indicator_theme

    footnote_theme = _match_theme_from_text(
        _theme_text(" ".join(item.get("text", "") for item in footnotes)),
        _INDICATOR_THEME_RULES,
    )
    if footnote_theme:
        return footnote_theme

    return "autre"


def _technical_change_counts(technical_diff: dict[str, Any]) -> tuple[int, int, int]:
    """Calcule les compteurs de changements techniques depuis le diff.

    Returns:
        Tuple ``(ajout_suppression_indicateurs, renommages, ajout_suppression_footnotes)``.
    """
    indicator_add_remove = len(technical_diff.get("indicators_added", []) or []) + len(
        technical_diff.get("indicators_removed", []) or []
    )
    renamed = len(technical_diff.get("indicators_renamed", []) or []) + len(
        technical_diff.get("footnotes_renamed", []) or []
    )
    footnote_add_remove = len(technical_diff.get("footnotes_added", []) or []) + len(
        technical_diff.get("footnotes_removed", []) or []
    )
    return indicator_add_remove, renamed, footnote_add_remove


def _build_analyst_summary(
    *,
    theme: str,
    change_kind: str,
    technical_diff: dict[str, Any],
) -> str:
    """Genere un resume textuel en francais destine a l'analyste.

    Args:
        theme: Theme reglementaire du tableau.
        change_kind: Type de changement (``"ajoute"``, ``"supprime"`` ou autre).
        technical_diff: Dictionnaire du diff technique contenant les listes
            d'indicateurs et de notes modifies.

    Returns:
        Phrase resumant les changements detectes.
    """
    if change_kind == "ajoute":
        return f"Nouveau tableau sur le theme {theme} a revoir par l'analyste."
    if change_kind == "supprime":
        return f"Tableau supprime sur le theme {theme} a confirmer par l'analyste."
    total_indicators = (
        len(technical_diff.get("indicators_added", []) or [])
        + len(technical_diff.get("indicators_removed", []) or [])
        + len(technical_diff.get("indicators_renamed", []) or [])
    )
    total_footnotes = (
        len(technical_diff.get("footnotes_added", []) or [])
        + len(technical_diff.get("footnotes_removed", []) or [])
        + len(technical_diff.get("footnotes_renamed", []) or [])
    )
    if total_indicators == 0 and total_footnotes == 0:
        return f"Aucun changement semantique detecte sur le theme {theme}."
    return (
        f"Changements semantiques detectes sur le theme {theme}: "
        f"{total_indicators} changement(s) d'indicateur et {total_footnotes} changement(s) de footnote."
    )


def build_analyst_assessment(
    *,
    table_context: dict[str, Any],
    technical_diff: dict[str, Any],
    change_kind: str,
) -> dict[str, str]:
    """Produit l'evaluation analyste complete pour un tableau compare.

    Combine la classification de theme, le calcul de significance du
    changement et la priorite de revision.

    Args:
        table_context: Metadonnees du tableau (section, titre, indicateurs,
            notes de bas de page).
        technical_diff: Dictionnaire du diff technique.
        change_kind: Type de changement (``"ajoute"``, ``"supprime"`` ou autre).

    Returns:
        Dictionnaire avec ``theme``, ``change_significance``,
        ``review_priority`` et ``analyst_summary``.
    """
    theme = _classify_theme(
        section=str(table_context.get("section", "") or ""),
        title=str(table_context.get("title", "") or ""),
        indicators=[str(value) for value in list(table_context.get("indicators") or [])],
        footnotes=[
            {
                "id": str(item.get("id", "") or "").strip(),
                "text": str(item.get("text", "") or "").strip(),
            }
            for item in list(table_context.get("footnotes", []) or [])
            if isinstance(item, dict)
        ],
    )
    indicator_add_remove, renamed, footnote_add_remove = _technical_change_counts(
        technical_diff
    )
    total_changes = indicator_add_remove + renamed + footnote_add_remove

    if change_kind in {"ajoute", "supprime"}:
        if theme in _HIGH_IMPACT_THEMES:
            significance, priority = "eleve", "critique"
        elif theme in _RISK_THEMES:
            significance, priority = "moyen", "prioritaire"
        else:
            significance, priority = "moyen", "normale"
    elif total_changes == 0:
        significance, priority = "faible", "normale"
    elif theme in _HIGH_IMPACT_THEMES:
        significance = "eleve"
        priority = (
            "critique"
            if indicator_add_remove + footnote_add_remove >= 2
            else "prioritaire"
        )
    elif theme in _RISK_THEMES:
        significance = "moyen"
        priority = "prioritaire" if total_changes >= 2 else "normale"
    else:
        significance = "moyen" if total_changes >= 2 else "faible"
        priority = "prioritaire" if total_changes >= 3 else "normale"

    return {
        "theme": theme,
        "change_significance": significance,
        "review_priority": priority,
        "analyst_summary": _build_analyst_summary(
            theme=theme,
            change_kind=change_kind,
            technical_diff=technical_diff,
        ),
    }
