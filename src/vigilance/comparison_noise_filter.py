"""Filtres post-traitement pour retirer le bruit non substantif des diffs techniques."""

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


_YEAR_OR_NUM_RE = re.compile(r"\b\d{1,4}(?:\s*[\/\.-]\s*\d{1,4})?\b")


def is_pure_numeric_or_date_noise(prev: str, curr: str) -> bool:
    """Retourne True si la seule différence entre prev et curr concerne des nombres ou des dates."""
    p_norm = _YEAR_OR_NUM_RE.sub("", prev).strip()
    c_norm = _YEAR_OR_NUM_RE.sub("", curr).strip()
    return p_norm == c_norm


def _filter_noise_from_diff(technical_diff: dict[str, Any]) -> dict[str, Any]:
    """Retire les renommages non substantifs (marqueurs de notes, numeros de page, dates récurrentes, fausses ajouts) du diff."""
    clean_ind_renamed = []
    for item in technical_diff.get("indicators_renamed", []):
        prev = str(item.get("previous", item.get("from", "")) or "")
        cur = str(item.get("current", item.get("to", "")) or "")
        if _strip_footnote_markers(prev) == _strip_footnote_markers(cur) or is_pure_numeric_or_date_noise(prev, cur):
            logger.debug(
                "Filtered noise: indicator rename '%s' -> '%s' (footnote marker or date only)",
                prev,
                cur,
            )
            continue
        clean_ind_renamed.append(item)

    clean_fn_renamed = []
    for item in technical_diff.get("footnotes_renamed", []):
        prev_text = str(item.get("previous_text", "") or "")
        cur_text = str(item.get("current_text", "") or "")
        if _strip_page_and_date_refs(prev_text) == _strip_page_and_date_refs(cur_text) or is_pure_numeric_or_date_noise(prev_text, cur_text):
            logger.debug("Filtered noise: footnote rename (page/date/numeric ref only)")
            continue
        clean_fn_renamed.append(item)

    clean_fn_added = []
    fn_removed_texts = {
        _strip_page_and_date_refs(str(item.get("text", "") or ""))
        for item in technical_diff.get("footnotes_removed", [])
    }
    for item in technical_diff.get("footnotes_added", []):
        text = _strip_page_and_date_refs(str(item.get("text", "") or ""))
        if text in fn_removed_texts or any(is_pure_numeric_or_date_noise(text, r_text) for r_text in fn_removed_texts):
            logger.debug("Filtered noise: false positive footnote_added '%s' exists in removed list", text)
            continue
        clean_fn_added.append(item)

    return {
        **technical_diff,
        "indicators_renamed": clean_ind_renamed,
        "footnotes_renamed": clean_fn_renamed,
        "footnotes_added": clean_fn_added,
    }
