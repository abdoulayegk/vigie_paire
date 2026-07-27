"""Callbacks de l'onglet Analyse Textuelle."""

from __future__ import annotations

from dash import ALL, Input, Output, State, callback, ctx, dcc
from dash.exceptions import PreventUpdate

from vigilance.dash_app.services.text_review import (
    apply_text_review_decision,
    is_final_direct_triage,
    write_text_review_to_disk,
)
from vigilance.dash_app.layouts.page_text_analysis import (
    build_filtered_text_cards,
    build_text_analysis_tab,
)
from vigilance.quarter_utils import quarter_label_from_payload


def _pattern_value_for_change(values, ids, change_id: str, default=None):
    """Retourne la valeur d'un composant Dash indexe par ``change_id``."""
    for value, id_value in zip(values or [], ids or [], strict=False):
        if (
            isinstance(id_value, dict)
            and str(id_value.get("change_id") or "") == change_id
        ):
            return value
    return default


def _change_by_id(text_data: dict, change_id: str) -> dict | None:
    """Retrouve un changement sans supposer dans quel bucket il est affiche."""
    for section in text_data.get("section_comparisons") or []:
        for bucket in ("all_block_comparisons", "block_comparisons"):
            for change in section.get(bucket) or []:
                if (
                    isinstance(change, dict)
                    and str(change.get("change_id") or "") == change_id
                ):
                    return change
    return None


def _structured_text_correction(
    text_data: dict,
    *,
    change_id: str,
    comment: str,
    materiality: str,
    nature,
    equivalence: str,
    themes,
    is_relevant,
    nouvelle_idee,
    confidence: str,
    evidence_sufficiency: str,
    supporting_evidence: str,
    counterarguments: str,
) -> dict | None:
    """Construit une correction apprenable seulement si elle est explicite."""
    rationale = str(comment or "").strip()
    normalized_level = str(materiality or "").strip().upper()
    if not rationale or normalized_level not in {
        "MAJEUR",
        "MODERE",
        "MINEUR",
    }:
        return None
    normalized_natures = (
        [nature]
        if isinstance(nature, str)
        else list(nature or [])
    )
    normalized_natures = [
        str(value or "").strip().upper()
        for value in normalized_natures
        if str(value or "").strip()
    ]
    if not normalized_natures:
        return None
    normalized_themes = [
        str(value or "").strip().upper()
        for value in (
            [themes] if isinstance(themes, str) else list(themes or [])
        )
        if str(value or "").strip()
    ]
    if not isinstance(is_relevant, bool) or not isinstance(
        nouvelle_idee,
        bool,
    ):
        return None
    normalized_confidence = str(
        confidence or "INDETERMINE"
    ).strip().upper()
    normalized_evidence = str(
        evidence_sufficiency or "INDETERMINE"
    ).strip().upper()
    if (
        normalized_confidence == "INDETERMINE"
        or normalized_evidence == "INDETERMINE"
    ):
        return None

    if is_relevant and not normalized_themes:
        return None
    if not is_relevant:
        if normalized_level != "MINEUR" or nouvelle_idee:
            return None
        normalized_themes = []
    if (
        normalized_level in {"MAJEUR", "MODERE"}
        and str(equivalence or "").strip().upper() == "CONFIRMEE"
    ):
        return None
    normalized_supporting_evidence = str(
        supporting_evidence or ""
    ).strip()
    if not normalized_supporting_evidence:
        return None
    normalized_counterarguments = str(
        counterarguments or ""
    ).strip()

    uncertain_minor = (
        normalized_level == "MINEUR"
        and str(equivalence or "INDETERMINE").strip().upper()
        != "CONFIRMEE"
    )
    review_required = (
        normalized_confidence == "FAIBLE"
        or normalized_evidence != "SUFFISANTE"
        or uncertain_minor
    )
    return {
        "materiality_level": normalized_level,
        "change_nature": normalized_natures[:3],
        "business_equivalence": (
            str(equivalence or "INDETERMINE").strip().upper()
        ),
        "is_relevant": is_relevant,
        "nouvelle_idee": nouvelle_idee,
        "themes_amf": normalized_themes[:2],
        "materiality_confidence": normalized_confidence,
        "evidence_sufficiency": normalized_evidence,
        "decision_status": (
            "A_CONFIRMER" if review_required else "CONFIRME"
        ),
        "review_required": review_required,
        "materiality_rationale": rationale,
        "supporting_evidence": [normalized_supporting_evidence],
        "counterarguments": (
            [normalized_counterarguments]
            if normalized_counterarguments
            else []
        ),
    }


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
    """Reconstruit le layout du tab quand les données arrivent."""
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
    """Filtre et trie les cartes analytiques selon les dropdowns."""
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
    """Mémorise le contexte de travail pendant les décisions analystes."""
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
    """Active la file à traiter depuis le compteur de progression."""
    if not n_clicks:
        raise PreventUpdate
    return "remaining"


@callback(
    Output("store-text-comparison", "data", allow_duplicate=True),
    Input({"type": "text-review-action", "change_id": ALL, "action": ALL}, "n_clicks"),
    State({"type": "text-review-action", "change_id": ALL, "action": ALL}, "id"),
    State({"type": "text-review-comment", "change_id": ALL}, "value"),
    State({"type": "text-review-comment", "change_id": ALL}, "id"),
    State({"type": "text-review-materiality", "change_id": ALL}, "value"),
    State({"type": "text-review-materiality", "change_id": ALL}, "id"),
    State({"type": "text-review-nature", "change_id": ALL}, "value"),
    State({"type": "text-review-nature", "change_id": ALL}, "id"),
    State({"type": "text-review-equivalence", "change_id": ALL}, "value"),
    State({"type": "text-review-equivalence", "change_id": ALL}, "id"),
    State({"type": "text-review-themes", "change_id": ALL}, "value"),
    State({"type": "text-review-themes", "change_id": ALL}, "id"),
    State({"type": "text-review-relevance", "change_id": ALL}, "value"),
    State({"type": "text-review-relevance", "change_id": ALL}, "id"),
    State({"type": "text-review-new-idea", "change_id": ALL}, "value"),
    State({"type": "text-review-new-idea", "change_id": ALL}, "id"),
    State({"type": "text-review-confidence", "change_id": ALL}, "value"),
    State({"type": "text-review-confidence", "change_id": ALL}, "id"),
    State({"type": "text-review-evidence", "change_id": ALL}, "value"),
    State({"type": "text-review-evidence", "change_id": ALL}, "id"),
    State({"type": "text-review-supporting-evidence", "change_id": ALL}, "value"),
    State({"type": "text-review-supporting-evidence", "change_id": ALL}, "id"),
    State({"type": "text-review-counterarguments", "change_id": ALL}, "value"),
    State({"type": "text-review-counterarguments", "change_id": ALL}, "id"),
    State("store-text-comparison", "data"),
    prevent_initial_call=True,
)
def review_text_change(
    action_clicks,
    action_ids,
    comments,
    comment_ids,
    materialities,
    materiality_ids,
    natures,
    nature_ids,
    equivalences,
    equivalence_ids,
    themes,
    theme_ids,
    relevances,
    relevance_ids,
    new_ideas,
    new_idea_ids,
    confidences,
    confidence_ids,
    evidence_values,
    evidence_ids,
    supporting_evidence_values,
    supporting_evidence_ids,
    counterargument_values,
    counterargument_ids,
    text_data,
):
    """Applique et persiste une decision analyste sur un changement texte."""
    if not text_data:
        raise PreventUpdate
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        raise PreventUpdate
    if not any(int(value or 0) > 0 for value in (action_clicks or [])):
        raise PreventUpdate

    change_id = str(triggered.get("change_id") or "").strip()
    action = str(triggered.get("action") or "").strip().lower()
    if not change_id or action not in {
        "approved",
        "corrected",
        "rejected",
        "skipped",
    }:
        raise PreventUpdate

    comment = str(
        _pattern_value_for_change(comments, comment_ids, change_id, "") or ""
    ).strip()
    structured_correction = None
    current_change = _change_by_id(text_data, change_id) or {}
    current_triage = current_change.get("genai_triage") or {}
    current_review_status = str(
        (current_change.get("_analyst_review") or {}).get("status") or ""
    ).strip().lower()
    if action == "approved" and (
        current_review_status == "corrected"
        or not is_final_direct_triage(current_triage)
    ):
        raise PreventUpdate
    if action in {"corrected", "rejected"}:
        materiality = str(
            _pattern_value_for_change(
                materialities,
                materiality_ids,
                change_id,
                "",
            )
            or ""
        ).strip().upper()
        nature = _pattern_value_for_change(
            natures,
            nature_ids,
            change_id,
            [],
        )
        equivalence = str(
            _pattern_value_for_change(
                equivalences,
                equivalence_ids,
                change_id,
                "INDETERMINE",
            )
            or "INDETERMINE"
        ).strip().upper()
        selected_themes = _pattern_value_for_change(
            themes,
            theme_ids,
            change_id,
            [],
        )
        selected_relevance = _pattern_value_for_change(
            relevances,
            relevance_ids,
            change_id,
            None,
        )
        selected_new_idea = _pattern_value_for_change(
            new_ideas,
            new_idea_ids,
            change_id,
            None,
        )
        confidence = str(
            _pattern_value_for_change(
                confidences,
                confidence_ids,
                change_id,
                "INDETERMINE",
            )
            or "INDETERMINE"
        ).strip().upper()
        evidence_sufficiency = str(
            _pattern_value_for_change(
                evidence_values,
                evidence_ids,
                change_id,
                "INDETERMINE",
            )
            or "INDETERMINE"
        ).strip().upper()
        supporting_evidence = str(
            _pattern_value_for_change(
                supporting_evidence_values,
                supporting_evidence_ids,
                change_id,
                "",
            )
            or ""
        ).strip()
        counterarguments = str(
            _pattern_value_for_change(
                counterargument_values,
                counterargument_ids,
                change_id,
                "",
            )
            or ""
        ).strip()
        structured_correction = _structured_text_correction(
            text_data,
            change_id=change_id,
            comment=comment,
            materiality=materiality,
            nature=nature,
            equivalence=equivalence,
            themes=selected_themes,
            is_relevant=selected_relevance,
            nouvelle_idee=selected_new_idea,
            confidence=confidence,
            evidence_sufficiency=evidence_sufficiency,
            supporting_evidence=supporting_evidence,
            counterarguments=counterarguments,
        )
        if structured_correction is None:
            raise PreventUpdate

    updated, found = apply_text_review_decision(
        text_data,
        change_id=change_id,
        status=action,
        comment=comment,
        structured_correction=structured_correction,
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
    """Génère et envoie le fichier Excel analyste."""
    if not n_clicks or not text_data:
        raise PreventUpdate

    from vigilance.dash_app.services.text_comparison_store import load_text_comparison_for_dash
    from vigilance.text_comparison import generate_text_comparison_excel

    bank_code = str(text_data.get("bank_code", "banque")).lower()
    quarter_current = str(text_data.get("quarter_current", "")).lower()
    quarter_previous = str(text_data.get("quarter_previous", "")).lower()
    latest_text_data = load_text_comparison_for_dash(
        bank_code=bank_code,
        quarter_current=quarter_current,
        quarter_previous=quarter_previous,
    )
    payload = latest_text_data or text_data

    excel_bytes = generate_text_comparison_excel(payload, output_path=None)
    bank = str(payload.get("bank_code", "banque")).upper()
    q_cur = _filename_period(quarter_label_from_payload(payload, "current"))
    filename = f"veille_textuelle_{bank}_{q_cur}.xlsx"
    return dcc.send_bytes(excel_bytes, filename)
