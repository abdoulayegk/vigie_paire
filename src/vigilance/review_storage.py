"""Utilitaires de persistance pour l'etat de revue analyste."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REVIEW_STATE_SCHEMA_VERSION = "review_state_v1"


def build_review_items_signature(review_items: list[dict[str, Any]] | None) -> str:
    """Retourne un digest stable pour un payload d'elements de revue persiste.

    Args:
        review_items: Liste de dictionnaires d'elements de revue.

    Returns:
        Hash SHA-256 hexadecimal, ou chaine vide si aucun element valide.
    """
    if not isinstance(review_items, list) or not review_items:
        return ""

    normalized_items: list[tuple[Any, ...]] = []
    for item in review_items:
        if not isinstance(item, dict):
            continue
        indicators = item.get("indicators", []) or []
        normalized_indicators = tuple(
            sorted(
                (
                    str(ind.get("type", "") or ""),
                    str(ind.get("name", "") or ""),
                    str(ind.get("from", "") or ""),
                    str(ind.get("to", "") or ""),
                    str(ind.get("footnote_ref", "") or ""),
                    str(ind.get("old_text", "") or ""),
                    str(ind.get("new_text", "") or ""),
                )
                for ind in indicators
                if isinstance(ind, dict)
            )
        )
        normalized_items.append(
            (
                str(item.get("section", "") or ""),
                str(item.get("table_name", "") or ""),
                str(item.get("table_id_t1", "") or ""),
                str(item.get("table_id_t2", "") or ""),
                str(item.get("change_type", "") or ""),
                str(item.get("item_type", "") or ""),
                normalized_indicators,
            )
        )

    if not normalized_items:
        return ""

    payload = json.dumps(sorted(normalized_items), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_review_state_compatible(
    state: dict[str, Any] | None,
    *,
    review_items: list[dict[str, Any]] | None = None,
    stored_review_items: list[dict[str, Any]] | None = None,
) -> bool:
    """Retourne True si l'etat persiste correspond encore aux elements derives de la comparaison courante."""
    if not state:
        return False
    if review_items is None:
        return True

    fresh_signature = build_review_items_signature(review_items)
    if not fresh_signature:
        return True

    stored_signature = str(state.get("review_items_signature", "") or "")
    if stored_signature:
        return stored_signature == fresh_signature

    candidate_stored_items = stored_review_items
    if candidate_stored_items is None:
        stored_items = state.get("review_items")
        if isinstance(stored_items, list) and stored_items:
            candidate_stored_items = stored_items

    if isinstance(candidate_stored_items, list) and candidate_stored_items:
        return build_review_items_signature(candidate_stored_items) == fresh_signature
    return True


def get_review_state_path(
    compare_path: str | Path | None,
    *,
    username: str | None = None,
) -> Path | None:
    """Retourne le chemin du fichier sidecar d'etat de revue pour un JSON de comparaison.

    Si ``username`` est fourni, le chemin est suffixe ``<stem>.review_state.<username>.json``
    pour la strategie multi-analystes (mode reader). Sans username, on conserve le
    nom historique ``<stem>.review_state.json`` (compatibilite pipeline complet).
    """
    if not compare_path:
        return None
    target = Path(compare_path)
    suffix = f".{username}" if username else ""
    if target.suffix.lower() == ".json":
        filename = f"{target.stem}.review_state{suffix}.json"
    else:
        filename = f"{target.name}.review_state{suffix}.json"
    return target.with_name(filename)


def load_review_state(
    compare_path: str | Path | None,
    *,
    username: str | None = None,
) -> dict[str, Any] | None:
    """Charge l'etat de revue persiste s'il est present et valide.

    Args:
        compare_path: Chemin du fichier JSON de comparaison.
        username: Identifiant analyste pour la strategie multi-utilisateurs.
            Si fourni, lit ``<stem>.review_state.<username>.json``. Si absent,
            tombe sur le sidecar partage historique.

    Returns:
        Dictionnaire d'etat ou ``None`` si absent/invalide.
    """
    state_path = get_review_state_path(compare_path, username=username)
    if state_path is None or not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def is_review_state_stale(
    state: dict[str, Any] | None,
    current_run_id: str,
) -> bool:
    """Retourne True si l'etat persiste ne correspond pas au run_id courant."""
    if not state or not current_run_id:
        return False
    stored_run_id = state.get("comparison_run_id", "")
    if not stored_run_id:
        # Legacy state without run_id — treat as stale to force refresh.
        return True
    return stored_run_id != current_run_id


def save_review_state(
    compare_path: str | Path | None,
    *,
    review_items: list[dict[str, Any]] | None = None,
    review_queue: list[dict[str, Any]] | None = None,
    review_selection: dict[str, Any] | None = None,
    review_current_idx: int | None = None,
    current_change_idx: int | None = None,
    current_indicator_idx: int | None = None,
    preferred_store: str = "review_queue",
    source: str = "dash",
    comparison_run_id: str = "",
    username: str | None = None,
) -> Path | None:
    """Persiste l'etat de revue analyste courant a cote du JSON de comparaison.

    Args:
        compare_path: Chemin du fichier JSON de comparaison.
        review_items: Elements de revue a persister.
        review_queue: File de revue normalisee.
        review_selection: Selection courante de l'analyste.
        review_current_idx: Index de l'element courant.
        current_change_idx: Index du changement courant.
        current_indicator_idx: Index de l'indicateur courant.
        preferred_store: Nom du store prefere (``review_queue`` par defaut).
        source: Source de la sauvegarde (``dash`` par defaut).
        comparison_run_id: Identifiant du run de comparaison.

    Returns:
        Chemin du fichier d'etat sauvegarde, ou ``None`` si impossible.
    """
    state_path = get_review_state_path(compare_path, username=username)
    if state_path is None:
        return None

    payload: dict[str, Any] = {
        "schema_version": REVIEW_STATE_SCHEMA_VERSION,
        "compare_path": str(compare_path),
        "comparison_run_id": str(comparison_run_id or ""),
        "review_items_signature": build_review_items_signature(review_items),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source or "dash"),
        "preferred_store": str(preferred_store or "review_queue"),
        "review_selection": dict(review_selection or {}),
        "review_current_idx": int(review_current_idx or 0),
        "current_change_idx": int(current_change_idx or 0),
        "current_indicator_idx": int(current_indicator_idx or 0),
    }
    if username:
        payload["username"] = str(username)
    if review_items is not None:
        payload["review_items"] = list(review_items)
    if review_queue is not None:
        payload["review_queue"] = list(review_queue)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return state_path
