"""Propagation des decisions analystes vers comparison.json et comparison.xlsx.

Mandat metier : quand l'analyste approuve/rejette un changement dans Dash, les
fichiers sur disque (``comparison.json`` et ``comparison.xlsx``) doivent refleter
immediatement la decision, pour eviter le "double travail".

Source de verite durable : ``comparison.review_state.json`` (file de revue).
Ce module **re-propage** ces decisions dans ``comparison.json`` (champ
``_analyst_review`` par indicateur/note) puis regenere ``comparison.xlsx``.

Les erreurs sont **non-fatales** : on log et on continue. Le ``review_state.json``
reste ecrit meme si la propagation echoue.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.comparison_excel import generate_comparison_excel
from vigilance.comparison_io import _atomic_write_json

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _attach_review(target: dict[str, Any], decision: dict[str, Any]) -> None:
    target["_analyst_review"] = dict(decision)


def _strip_previous_decisions(comparison: dict[str, Any]) -> None:
    """Retire tous les champs ``_analyst_review`` prealablement poses.

    Ainsi chaque appel repart d'un etat propre et les decisions revertees
    (``pending`` apres approval) disparaissent du JSON/XLSX.
    """
    for pair in comparison.get("pair_comparisons") or []:
        if not isinstance(pair, dict):
            continue
        diff = pair.get("technical_diff") or {}
        if not isinstance(diff, dict):
            continue
        for key in (
            "indicators_added",
            "indicators_removed",
            "indicators_renamed",
            "footnotes_added",
            "footnotes_removed",
            "footnotes_renamed",
        ):
            for item in diff.get(key) or []:
                if isinstance(item, dict):
                    item.pop("_analyst_review", None)
    matching = comparison.get("matching") or {}
    for key in ("tables_added", "tables_removed"):
        for tbl in matching.get(key) or []:
            if isinstance(tbl, dict):
                tbl.pop("_analyst_review", None)


def _inject_into_pair(
    pair: dict[str, Any], change: dict[str, Any], decision: dict[str, Any]
) -> bool:
    """Localise l'item correspondant dans ``technical_diff`` et y attache la decision."""
    change_type = str(change.get("change_type", ""))
    payload = change.get("payload") or {}
    diff = pair.get("technical_diff") or {}
    if not isinstance(diff, dict):
        return False

    if change_type in ("indicator_added", "indicator_removed"):
        target_key = (
            "indicators_added"
            if change_type == "indicator_added"
            else "indicators_removed"
        )
        name_clean = _clean(
            payload.get("indicator_name_clean") or payload.get("indicator_name")
        )
        for ind in diff.get(target_key) or []:
            if isinstance(ind, dict) and _clean(ind.get("value")) == name_clean:
                _attach_review(ind, decision)
                return True
        return False

    if change_type == "indicator_renamed":
        from_clean = _clean(payload.get("from_clean") or payload.get("from"))
        to_clean = _clean(payload.get("to_clean") or payload.get("to"))
        for ind in diff.get("indicators_renamed") or []:
            if (
                isinstance(ind, dict)
                and _clean(ind.get("previous")) == from_clean
                and _clean(ind.get("current")) == to_clean
            ):
                _attach_review(ind, decision)
                return True
        return False

    if change_type.startswith("footnote_"):
        key_map = {
            "footnote_added": "footnotes_added",
            "footnote_removed": "footnotes_removed",
            "footnote_modified": "footnotes_renamed",
        }
        target_key = key_map.get(change_type, "")
        if not target_key:
            return False
        ref = str(payload.get("footnote_ref") or "").strip()
        for fn in diff.get(target_key) or []:
            if not isinstance(fn, dict):
                continue
            candidates = (
                str(fn.get("id", "")).strip(),
                str(fn.get("previous_id", "")).strip(),
                str(fn.get("current_id", "")).strip(),
            )
            if ref and ref in candidates:
                _attach_review(fn, decision)
                return True
        return False

    return False


def _inject_into_matching(
    comparison: dict[str, Any],
    table_item: dict[str, Any],
    change: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Gere ``table_added`` / ``table_removed`` via ``matching.tables_*``."""
    change_type = str(change.get("change_type", ""))
    key_map = {"table_added": "tables_added", "table_removed": "tables_removed"}
    target_key = key_map.get(change_type)
    if not target_key:
        return False
    if change_type == "table_added":
        target_id = str(table_item.get("table_id_t2") or "").strip()
    else:
        target_id = str(table_item.get("table_id_t1") or "").strip()
    if not target_id:
        return False
    matching = comparison.get("matching") or {}
    for tbl in matching.get(target_key) or []:
        if isinstance(tbl, dict) and str(tbl.get("table_id") or "").strip() == target_id:
            _attach_review(tbl, decision)
            return True
    return False


def _merge_decisions(
    comparison: dict[str, Any], review_queue: list[dict[str, Any]]
) -> dict[str, int]:
    stats = {"matched": 0, "unmatched": 0, "pending": 0}
    pairs = comparison.get("pair_comparisons") or []

    for table in review_queue:
        if not isinstance(table, dict):
            continue
        tid_prev = str(table.get("table_id_t1") or "").strip()
        tid_curr = str(table.get("table_id_t2") or "").strip()
        target_pair: dict[str, Any] | None = None
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            if (
                str(pair.get("previous_table_id") or "").strip() == tid_prev
                and str(pair.get("current_table_id") or "").strip() == tid_curr
            ):
                target_pair = pair
                break

        for change in table.get("changes", []) or []:
            status = str(change.get("validation_status", "") or "pending").lower()
            if status == "pending":
                stats["pending"] += 1
                continue
            decision = {
                "status": status,
                "notes": str(change.get("validation_notes") or "").strip(),
                "at": str(change.get("validated_at") or "").strip(),
                "by": str(change.get("validated_by") or "").strip(),
            }
            matched = False
            if target_pair is not None:
                matched = _inject_into_pair(target_pair, change, decision)
            if not matched:
                matched = _inject_into_matching(comparison, table, change, decision)
            if matched:
                stats["matched"] += 1
            else:
                stats["unmatched"] += 1

    return stats


def write_back_to_disk(
    compare_path: str | Path | None,
    review_queue: list[dict[str, Any]] | None,
) -> bool:
    """Injecte les decisions dans ``comparison.json`` puis regenere ``comparison.xlsx``.

    Returns:
        ``True`` si les deux ecritures reussissent, ``False`` sinon. Non-fatal.
    """
    if not compare_path or not isinstance(review_queue, list):
        return False
    cmp_path = Path(str(compare_path))
    if not cmp_path.exists():
        logger.warning("[review_writeback] comparison.json introuvable: %s", cmp_path)
        return False

    try:
        comparison = json.loads(cmp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("[review_writeback] lecture echec %s: %s", cmp_path, exc)
        return False

    _strip_previous_decisions(comparison)
    stats = _merge_decisions(comparison, review_queue)
    comparison["review_decisions_summary"] = {
        **stats,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    try:
        _atomic_write_json(cmp_path, comparison)
    except OSError as exc:
        logger.warning("[review_writeback] ecriture comparison.json echec: %s", exc)
        return False

    xlsx_path = cmp_path.parent / "comparison.xlsx"
    try:
        generate_comparison_excel(comparison, xlsx_path)
    except Exception as exc:  # noqa: BLE001 — non-fatal, log and continue
        logger.warning("[review_writeback] regeneration xlsx echec: %s", exc)
        return False

    logger.info(
        "[review_writeback] %s -> matched=%d unmatched=%d pending=%d",
        cmp_path.name,
        stats["matched"],
        stats["unmatched"],
        stats["pending"],
    )
    return True
