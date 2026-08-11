"""Persistence de l'etat de revue pour les callbacks Dash."""

from __future__ import annotations

from vigie.interface import review_runtime
from vigie.interface.services.comparison_context import _comparison_path_from_meta
from vigie.interface.services.comparison_store import build_file_comparison_store
from vigie.interface.services.export_helpers import _review_items_from_v2_queue


def _persist_review_state(
    *,
    indicator_meta: dict | None,
    indicator_result: dict | None = None,
    review_items: list[dict] | None = None,
    review_queue: list[dict] | None = None,
    review_selection: dict | None = None,
    review_current_idx: int | None = None,
    current_change_idx: int | None = None,
    current_indicator_idx: int | None = None,
    preferred_store: str = "review_queue",
    source: str = "dash",
) -> None:
    """Persiste l'etat de revue a cote du fichier JSON de comparaison si possible.

    Args:
        indicator_meta: Metadonnees de la comparaison active.
        indicator_result: Resultat de comparaison (secours pour le chemin).
        review_items: Elements de revue serialises (format legacy).
        review_queue: File de revue V2.
        review_selection: Selection courante de l'analyste.
        review_current_idx: Index du tableau courant.
        current_change_idx: Index du changement courant dans le tableau.
        current_indicator_idx: Index de l'indicateur courant.
        preferred_store: Nom du store de preference (``review_queue`` par defaut).
        source: Origine de la sauvegarde (ex. ``dash``).
    """
    compare_path = _comparison_path_from_meta(indicator_meta, indicator_result)
    if not compare_path:
        return
    # Extract run_id from meta so the state can be invalidated on re-run.
    run_id = ""
    if indicator_meta:
        run_id = str(indicator_meta.get("run_id", ""))
    username = review_runtime.current_analyst()
    store = build_file_comparison_store()
    store.save_review_state(
        compare_path,
        review_items=review_items,
        review_queue=review_queue,
        review_selection=review_selection,
        review_current_idx=review_current_idx,
        current_change_idx=current_change_idx,
        current_indicator_idx=current_indicator_idx,
        preferred_store=preferred_store,
        source=source,
        comparison_run_id=run_id,
        username=username,
    )
    # Mandat : propager la decision analyste dans comparison.json + comparison.xlsx
    # sur disque. Non-fatal — le review_state.json reste la source durable.
    # La revue partage uniquement un sidecar par analyste. Le pipeline complet
    # conserve la propagation vers comparison.json et Excel.
    if not review_runtime.is_review_mode():
        from vigie.interface.services.review_writeback import (  # noqa: PLC0415 - exclu du mode revue minimal
            write_back_to_disk,
        )

        write_back_to_disk(compare_path, review_queue)


def _load_review_state_for_comparison(
    *,
    indicator_meta: dict | None,
    indicator_result: dict | None = None,
) -> dict | None:
    """Charge l'etat de revue persiste pour la comparaison active si disponible.

    Args:
        indicator_meta: Metadonnees de la comparaison active.
        indicator_result: Resultat de comparaison (secours pour le chemin).

    Returns:
        Dictionnaire d'etat de revue ou ``None`` si introuvable.
    """
    compare_path = _comparison_path_from_meta(indicator_meta, indicator_result)
    if not compare_path:
        return None
    store = build_file_comparison_store()
    username = review_runtime.current_analyst()
    if username:
        per_user_state = store.load_review_state(compare_path, username=username)
        if per_user_state is not None:
            return per_user_state
    return store.load_review_state(compare_path)


def _stored_review_items_from_state(state: dict | None) -> list[dict] | None:
    """Retourne les elements de revue persistes, reconstruits depuis la file si necessaire.

    Args:
        state: Dictionnaire d'etat de revue charge depuis le disque.

    Returns:
        Liste de dictionnaires d'elements de revue ou ``None`` si indisponible.
    """
    if not isinstance(state, dict):
        return None

    stored_items = state.get("review_items")
    if isinstance(stored_items, list) and stored_items:
        return stored_items

    stored_queue = state.get("review_queue")
    if isinstance(stored_queue, list) and stored_queue:
        return [it.to_dict() for it in _review_items_from_v2_queue(stored_queue)]
    return None
