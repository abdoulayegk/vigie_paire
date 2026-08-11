"""Callbacks de l'onglet Analyse Textuelle."""

from __future__ import annotations

from dash import ALL, Input, Output, State, callback, ctx, dcc
from dash.exceptions import PreventUpdate

from vigie.analyse_texte import text_comparison
from vigie.interface.layouts.text_analysis import (
    build_filtered_text_cards,
    build_text_analysis_tab,
)
from vigie.interface.services import text_comparison_store
from vigie.interface.services.text_review import (
    apply_text_review_decision,
    write_text_review_to_disk,
)
from vigie.support.quarter_utils import quarter_label_from_payload


def _filename_period(label: str) -> str:
    """Normalise un libelle de trimestre pour un nom de fichier."""
    return "_".join(str(label or "").strip().upper().split())


@callback(
    Output("text-analysis-tab-content", "children"),
    Input("store-text-comparison", "data"),
    Input("store-show-results-page", "data"),
    State("store-text-review-filters", "data"),
    prevent_initial_call=True,
)
def render_text_analysis(text_data, show_results, text_filters=None):
    """Reconstruit l'onglet textuel lorsque les données deviennent disponibles.

    Le résultat textuel, la visibilité et les filtres mémorisés produisent les
    cartes, compteurs et contrôles affichés à l'analyste.
    """
    if not show_results:
        raise PreventUpdate
    filters = text_filters if isinstance(text_filters, dict) else {}
    kwargs = {
        "filter_scope": filters.get("scope", "qualitative"),
        "filter_impact": filters.get("impact"),
        "filter_action": filters.get("action"),
        "filter_status": filters.get("status", "remaining"),
    }
    if "section" in filters:
        kwargs["filter_section"] = filters.get("section")
    return build_text_analysis_tab(text_data, **kwargs)


@callback(
    Output("text-cards-container", "children"),
    Output("text-filter-count", "children"),
    Input("store-text-comparison", "data"),
    Input("text-filter-section", "value"),
    Input("text-filter-impact", "value"),
    Input("text-filter-action", "value"),
    Input("text-filter-status", "value"),
    Input("text-filter-scope", "value"),
    prevent_initial_call=True,
)
def filter_text_cards(
    text_data,
    filter_section,
    filter_impact,
    filter_action,
    filter_status=None,
    filter_scope="qualitative",
):
    """Filtre et trie les changements textuels selon les choix analystes.

    Les listes de section, portée, impact, action et statut produisent les cartes
    visibles et les compteurs de progression associés.
    """
    if not text_data:
        raise PreventUpdate
    return build_filtered_text_cards(
        text_data,
        filter_section,
        filter_impact,
        filter_action,
        filter_status,
        filter_scope,
    )


@callback(
    Output("store-text-review-filters", "data"),
    Input("text-filter-section", "value"),
    Input("text-filter-scope", "value"),
    Input("text-filter-impact", "value"),
    Input("text-filter-action", "value"),
    Input("text-filter-status", "value"),
    prevent_initial_call=True,
)
def remember_text_review_filters(section, scope, impact, action, status):
    """Mémorise les filtres actifs de la revue textuelle.

    Les valeurs des menus sont regroupées dans un store afin que les décisions
    et les rafraîchissements conservent le même contexte de travail.
    """
    return {
        "section": section,
        "scope": scope or "qualitative",
        "impact": impact,
        "action": action,
        "status": status or "remaining",
    }


@callback(
    Output("text-filter-status", "value"),
    Input("text-progress-remaining", "n_clicks"),
    prevent_initial_call=True,
)
def show_remaining_text_changes(n_clicks):
    """Active le filtre des changements textuels restant à traiter.

    Un clic sur le compteur produit la valeur de statut attendue par le menu de
    filtrage; aucun changement métier n'est modifié.
    """
    if not n_clicks:
        raise PreventUpdate
    return "remaining"


@callback(
    Output("store-text-comparison", "data", allow_duplicate=True),
    Input({"type": "text-review-action", "change_id": ALL, "action": ALL}, "n_clicks"),
    State({"type": "text-review-action", "change_id": ALL, "action": ALL}, "id"),
    State({"type": "text-review-comment", "change_id": ALL}, "value"),
    State({"type": "text-review-comment", "change_id": ALL}, "id"),
    State("store-text-comparison", "data"),
    prevent_initial_call=True,
)
def review_text_change(action_clicks, action_ids, comments, comment_ids, text_data):
    """Applique puis persiste la décision analyste sur un changement textuel.

    Le bouton déclencheur, son identifiant et le commentaire mettent à jour le
    résultat textuel retourné au store ainsi que le message de confirmation.
    """
    if not text_data:
        raise PreventUpdate
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        raise PreventUpdate
    if not any(int(value or 0) > 0 for value in (action_clicks or [])):
        raise PreventUpdate

    change_id = str(triggered.get("change_id") or "").strip()
    action = str(triggered.get("action") or "").strip().lower()
    if not change_id or action not in {"approved", "rejected", "skipped"}:
        raise PreventUpdate

    comment = ""
    for value, id_value in zip(comments or [], comment_ids or [], strict=False):
        if isinstance(id_value, dict) and str(id_value.get("change_id") or "") == change_id:
            comment = str(value or "")
            break

    updated, found = apply_text_review_decision(
        text_data,
        change_id=change_id,
        status=action,
        comment=comment,
    )
    if not found:
        raise PreventUpdate
    write_text_review_to_disk(updated, regenerate_excel=action != "skipped")
    return updated


@callback(
    Output("download-text-excel", "data"),
    Input("btn-download-text-excel", "n_clicks"),
    State("store-text-comparison", "data"),
    prevent_initial_call=True,
)
def download_text_excel(n_clicks, text_data):
    """Génère l'export Excel de la revue textuelle sur demande.

    Le clic et le store textuel produisent la réponse de téléchargement Dash,
    avec les décisions et commentaires analystes courants.
    """
    if not n_clicks or not text_data:
        raise PreventUpdate

    bank_code = str(text_data.get("bank_code", "banque")).lower()
    quarter_current = str(text_data.get("quarter_current", "")).lower()
    quarter_previous = str(text_data.get("quarter_previous", "")).lower()
    latest_text_data = text_comparison_store.load_text_comparison_for_dash(
        bank_code=bank_code,
        quarter_current=quarter_current,
        quarter_previous=quarter_previous,
    )
    payload = latest_text_data or text_data

    excel_bytes = text_comparison.generate_text_comparison_excel(payload, output_path=None)
    bank = str(payload.get("bank_code", "banque")).upper()
    q_cur = _filename_period(quarter_label_from_payload(payload, "current"))
    filename = f"veille_textuelle_{bank}_{q_cur}.xlsx"
    return dcc.send_bytes(excel_bytes, filename)
