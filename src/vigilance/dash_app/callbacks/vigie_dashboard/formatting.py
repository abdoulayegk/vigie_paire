"""Conversions, comptages unitaires et libelles d'affichage du tableau de bord.

Extrait de ``vigie_dashboard_flow.py`` sans modification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from reportlab.lib import colors

from vigilance.dash_app.services.export_helpers import _is_high_priority_item

def _plot_layout(theme: str) -> dict[str, Any]:
    """Retourne le layout Plotly commun (couleurs adaptées au thème clair/sombre)."""
    is_light = theme == "light"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#243145" if is_light else "#d7dee9", "size": 11},
        "margin": {"l": 36, "r": 18, "t": 8, "b": 28},
        "legend": {"font": {"color": "#475569" if is_light else "#c8d2e0", "size": 11}},
    }

_PDF_NAVY = colors.HexColor("#0b1725")
_PDF_PANEL = colors.HexColor("#101f31")
_PDF_TEXT = colors.HexColor("#172033")
_PDF_MUTED = colors.HexColor("#5f6b7a")
_PDF_BLUE = colors.HexColor("#4b74f2")
_PDF_RED = colors.HexColor("#e45142")
_PDF_AMBER = colors.HexColor("#f3b23c")
_PDF_GREEN = colors.HexColor("#68b976")


def _safe_int(value: Any, default: int = 0) -> int:
    """Convertit ``value`` en entier, retourne ``default`` si la conversion échoue."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convertit ``value`` en float, retourne ``default`` si la conversion échoue."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _format_number(value: int | float | str) -> str:
    """Formate un nombre avec espaces comme séparateurs de milliers."""
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", " ")
    return str(value)


def _pdf_text(value: Any, max_len: int | None = None) -> str:
    """Échappe le HTML et tronque le texte pour insertion dans un PDF ReportLab."""
    text = str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = " ".join(text.split())
    if max_len and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _quarter_label(label: Any) -> str:
    """Normalise un libellé de trimestre (``T2`` → ``Q2``)."""
    text = str(label or "").strip().upper().replace("_", "-")
    if text.startswith("T"):
        return text.replace("T", "Q", 1)
    return text or "N/D"


def _comparisons(indicator: dict | None) -> list[dict]:
    """Retourne la liste des comparaisons de tableaux d'un payload d'indicateurs."""
    if not isinstance(indicator, dict):
        return []
    table_comparisons = indicator.get("table_comparisons")
    if isinstance(table_comparisons, list):
        return table_comparisons
    pair_comparisons = indicator.get("pair_comparisons")
    return pair_comparisons if isinstance(pair_comparisons, list) else []


def _technical_diff(comp: dict) -> dict:
    """Retourne le dictionnaire ``technical_diff`` d'une comparaison (ou vide)."""
    return comp.get("technical_diff", {}) or {}


def _table_title(comp: dict) -> str:
    """Retourne le titre d'un tableau apparié (T2 prioritaire, fallbacks variés)."""
    current = comp.get("current_table", {}) or {}
    previous = comp.get("previous_table", {}) or {}
    return str(
        comp.get("title_t2")
        or comp.get("title_t1")
        or comp.get("table_title")
        or current.get("title")
        or previous.get("title")
        or comp.get("current_table_id")
        or comp.get("previous_table_id")
        or "Sans titre"
    )


def _indicator_confidence(comp: dict) -> float | None:
    """Retourne le score de confiance de l'appariement d'indicateurs (ou ``None``)."""
    raw = comp.get("match_score", comp.get("match_confidence"))
    if raw is None:
        return None
    return _safe_float(raw)


def _count_list(comp: dict, canonical_key: str, diff_key: str) -> int:
    """Compte les éléments d'une liste canonique avec fallback sur ``technical_diff``."""
    diff = _technical_diff(comp)
    return len(comp.get(canonical_key, []) or diff.get(diff_key, []) or [])


def _footnote_counts(comp: dict) -> dict[str, int]:
    """Retourne les compteurs (added / removed / modified / renamed) de notes de bas de page."""
    existing = comp.get("footnotes_counts", {}) or {}
    diff = _technical_diff(comp)
    diff_renamed = len(diff.get("footnotes_renamed", []) or [])
    if existing:
        renamed = _safe_int(existing.get("renamed"), diff_renamed)
        modified = _safe_int(existing.get("modified"))
        if not existing.get("renamed") and diff_renamed and modified >= diff_renamed:
            modified -= diff_renamed
        return {
            "added": _safe_int(existing.get("added")),
            "removed": _safe_int(existing.get("removed")),
            "modified": modified,
            "renamed": renamed,
        }
    return {
        "added": len(diff.get("footnotes_added", []) or []),
        "removed": len(diff.get("footnotes_removed", []) or []),
        "modified": len(diff.get("footnotes_modified", []) or []),
        "renamed": diff_renamed,
    }


def _change_total(comp: dict) -> int:
    """Retourne le nombre total de changements (indicateurs + notes) pour une comparaison."""
    notes = _footnote_counts(comp)
    return (
        _count_list(comp, "added_indicators", "indicators_added")
        + _count_list(comp, "removed_indicators", "indicators_removed")
        + _count_list(comp, "renamed_indicators", "indicators_renamed")
        + notes["added"]
        + notes["removed"]
        + notes["modified"]
    )


def _low_confidence(comp: dict) -> bool:
    """Indique si la comparaison est de faible confiance (score < 85 % ou flags d'incertitude)."""
    score = _indicator_confidence(comp)
    if score is not None:
        normalized = score if score <= 1 else score / 100
        if normalized < 0.85:
            return True
    status = str(comp.get("table_status", "") or "").strip().lower()
    meta = comp.get("match_metadata", {}) or {}
    return status in {"incertain", "needs_review"} or bool(meta.get("drastic_row_drop", False))


def _change_label(comp: dict) -> str:
    """Retourne un libellé synthétique du type de changement (priorité aux indicateurs)."""
    notes = _footnote_counts(comp)
    if _count_list(comp, "added_indicators", "indicators_added"):
        return "Ajout d'indicateurs"
    if _count_list(comp, "removed_indicators", "indicators_removed"):
        return "Suppression d'indicateurs"
    if _count_list(comp, "renamed_indicators", "indicators_renamed"):
        return "Renommage d'indicateurs"
    if sum(notes.values()):
        return "Note(s) modifiée(s)"
    return str(_technical_diff(comp).get("table_level_change") or "Modification")


def _impact_label(comp: dict) -> str:
    """Retourne le label d'impact analyste (Élevé / Moyen / Faible) à partir du triage."""
    triage = comp.get("genai_triage", {}) or comp.get("genai_analysis", {}) or {}
    assessment = comp.get("analyst_assessment", {}) or {}
    raw = triage.get("impact_level") or assessment.get("change_significance") or assessment.get("review_priority") or ""
    normalized = str(raw).strip().upper()
    if normalized in {"ELEVE", "ELEVEE", "CRITIQUE", "PRIORITAIRE"}:
        return "Élevé"
    if normalized in {"MODERE", "MOYEN", "MOYENNE"}:
        return "Moyen"
    if normalized in {"FAIBLE", "MINEUR", "NORMAL", "NORMALE"}:
        return "Faible"
    if _low_confidence(comp) or _is_high_priority_item(comp):
        return "Élevé"
    return "Moyen" if _change_total(comp) else "Faible"


def _confidence_label(score: float | None) -> str:
    """Retourne un label de confiance (Élevée / Moyenne / Faible) à partir d'un score 0-1 ou 0-100."""
    if score is None:
        return "N/D"
    pct = score * 100 if score <= 1 else score
    if pct >= 80:
        return "Élevée"
    if pct >= 50:
        return "Moyenne"
    return "Faible"


def _badge_class(label: str) -> str:
    """Retourne la classe CSS de badge correspondant au label de criticité."""
    normalized = str(label or "").lower()
    if any(token in normalized for token in ("majeur", "élev", "eleve", "critique", "faible")):
        return "vigie-cockpit-badge is-danger"
    if any(token in normalized for token in ("mod", "moyen", "prioritaire", "attente")):
        return "vigie-cockpit-badge is-warning"
    if any(token in normalized for token in ("valid", "élevée", "elevee")):
        return "vigie-cockpit-badge is-success"
    return "vigie-cockpit-badge is-muted"


def _updated_at(indicator_meta: dict | None, indicator: dict | None, text_data: dict | None) -> str:
    """Retourne la date de mise à jour la plus récente disponible parmi les sources."""
    for source in (indicator_meta, indicator, text_data):
        if not isinstance(source, dict):
            continue
        for key in ("updated_at", "created_at", "generated_at"):
            raw = source.get(key)
            if raw:
                try:
                    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).strftime("%d %b %Y à %Hh%M")
                except ValueError:
                    return str(raw)
    return "Non disponible"
