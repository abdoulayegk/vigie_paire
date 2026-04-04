"""Filtres post-traitement pour retirer le bruit cosmetique des diffs techniques."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TABLE_LEVEL_CHANGE_KEYS = (
    "indicators_added",
    "indicators_removed",
    "indicators_renamed",
    "footnotes_added",
    "footnotes_removed",
    "footnotes_renamed",
)

_FOOTNOTE_MARKER_RE = re.compile(r"\s*\(\d{1,2}\)\s*")
_PAGE_REF_RE = re.compile(r"pages?\s+\d+\s*[àa]\s*\d+", re.IGNORECASE)
_DATE_REF_RE = re.compile(
    r"\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4}",
    re.IGNORECASE,
)


def _strip_footnote_markers(text: str) -> str:
    """Retire les marqueurs de notes de bas de page (ex. ``(1)``, ``(2)``) du texte."""
    return _FOOTNOTE_MARKER_RE.sub("", text).strip()


def _strip_page_and_date_refs(text: str) -> str:
    """Remplace les references de pages et de dates par des jetons generiques."""
    text = _PAGE_REF_RE.sub("__PAGE__", text)
    text = _DATE_REF_RE.sub("__DATE__", text)
    return text.strip()


def recompute_table_level_change(technical_diff: dict[str, Any]) -> str:
    """Retourne le statut canonique au niveau du tableau pour un diff filtre."""
    has_changes = any(technical_diff.get(key) for key in _TABLE_LEVEL_CHANGE_KEYS)
    return "modifie" if has_changes else "inchange"


def _filter_noise_from_diff(technical_diff: dict[str, Any]) -> dict[str, Any]:
    """Retire les renommages cosmetiques (marqueurs de notes, numeros de page) du diff.

    Args:
        technical_diff: Dictionnaire du diff technique contenant les listes de
            changements d'indicateurs et de notes de bas de page.

    Returns:
        Copie du diff avec les renommages cosmetiques retires.
    """
    # Filter indicator renames where only footnote marker differs
    clean_ind_renamed = []
    for item in technical_diff.get("indicators_renamed", []):
        prev = str(item.get("previous", item.get("from", "")) or "")
        cur = str(item.get("current", item.get("to", "")) or "")
        if _strip_footnote_markers(prev) == _strip_footnote_markers(cur):
            logger.debug(
                "Filtered noise: indicator rename '%s' -> '%s' (footnote marker only)",
                prev,
                cur,
            )
            continue
        clean_ind_renamed.append(item)

    # Filter footnote renames where only page/date references differ
    clean_fn_renamed = []
    for item in technical_diff.get("footnotes_renamed", []):
        prev_text = str(item.get("previous_text", "") or "")
        cur_text = str(item.get("current_text", "") or "")
        if _strip_page_and_date_refs(prev_text) == _strip_page_and_date_refs(cur_text):
            logger.debug("Filtered noise: footnote rename (page/date ref only)")
            continue
        clean_fn_renamed.append(item)

    return {
        **technical_diff,
        "indicators_renamed": clean_ind_renamed,
        "footnotes_renamed": clean_fn_renamed,
    }
