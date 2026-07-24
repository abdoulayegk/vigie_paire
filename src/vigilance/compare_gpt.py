"""Pipeline de comparaison GPT-4o sur les artefacts canoniques tables.json."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from vigilance.comparison_analyst import build_analyst_assessment
from vigilance.comparison_devil_advocate import (
    _devil_advocate_review,
)
from vigilance.comparison_diff_gpt import diff_table_pair_gpt
from vigilance.comparison_io import (
    _atomic_write_json,
    _coerce_int,
    _coerce_pathlike,
    _extract_usage_metrics,
    _is_boundary_inventory_candidate,
    _load_tables_payload,
    _make_run_id,
    _merge_extraction_suspect_side,
    _partition_tables_by_status,
    _table_card,
    _table_detail,
    _table_snapshot,
    normalize_quarter,  # noqa: F401 — re-exported public API
    resolve_reference_period,  # noqa: F401 — re-exported public API
)
from vigilance.comparison_matching import (
    _MATCHING_VALIDATION_ATTEMPTS,
    _run_table_matching,
)
from vigilance.comparison_metrics import (
    _build_run_metrics,
    _count_high_priority_items,
    _count_pair_changes,
)
from vigilance.comparison_noise_filter import (
    _filter_noise_from_diff,
    recompute_table_level_change,
)
from vigilance.comparison_visual_sanity import (
    render_visual_sanity_proof,
    visual_sanity_check,
    visual_sanity_check_table_event,
)
from vigilance.config import get_matching_thresholds, resolve_openai_model
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.utils.genai import get_openai_api_key
from vigilance.utils.matching_normalizer import (
    is_generic_title,
    normalize_for_matching,
    strip_temporal_expressions,
)
from vigilance.utils.proof_rendering import normalize_proof_bbox

logger = logging.getLogger(__name__)

MATCH_PROMPT_VERSION = "table_match_v8"
DIFF_PROMPT_VERSION = "table_diff_v4"
COMPARISON_SCHEMA_VERSION = 3
OPENAI_COMPARISON_TIMEOUT_SECONDS = 120.0


REFERENCE_RESOLUTION_RULE = "t2->t1 meme annee; t3->t2 meme annee; t1->t3 annee precedente; t4->t4 annee precedente"


def _archive_source_pdf(source: str | Path | None, target: Path) -> str:
    """Copier un PDF source dans le repertoire du run pour la portabilite inter-OS.

    Retourne le chemin de la copie archivee en cas de succes ; sinon retourne
    le chemin source original (ou ``""`` si absent). Les echecs sont logges mais
    non fatals : la comparaison reste utilisable sur la machine d'origine via le
    chemin absolu, et Dash sait retomber sur le voisin archive lorsqu'il existe.
    """
    raw = str(source or "").strip()
    if not raw:
        return ""
    src_path = Path(raw)
    if not src_path.exists():
        logger.warning("PDF source introuvable pour archivage: %s", raw)
        return raw
    if target.exists():
        try:
            if src_path.samefile(target):
                return str(target)
        except OSError:
            pass
    try:
        shutil.copy2(src_path, target)
        return str(target)
    except OSError as exc:
        logger.warning(
            "Echec de l'archivage du PDF %s -> %s: %s", src_path, target, exc
        )
        return raw


def _visual_sanity_meta(
    *,
    applied: bool,
    rejected_count: int,
    render_status: str,
    render_mode: str = "full",
) -> dict[str, Any]:
    """Construire le bloc de metadonnees de la verification visuelle."""
    return {
        "visual_sanity_applied": applied,
        "visual_sanity_rejected_count": int(rejected_count),
        "visual_sanity_scope": ["indicators", "footnotes", "tables"],
        "visual_sanity_render_mode": render_mode,
        "visual_sanity_render_status": render_status,
    }


def _normalize_table_anchor_section(value: Any) -> str:
    """Normaliser le nom de section pour l'ancrage visuel d'une table."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        normalized = canonicalize_section(raw)
    except Exception:
        normalized = raw
    return str(normalized or "").strip()


def _normalize_table_anchor_title(value: Any) -> str:
    """Normaliser le titre d'une table pour l'ancrage visuel."""
    raw = strip_temporal_expressions(str(value or ""), target="title", aggressive=True)
    return normalize_for_matching(raw, target="title")


def _snapshot_has_visual_render_anchor(snapshot: dict[str, Any]) -> bool:
    """Verifier qu'un snapshot peut etre rendu sur sa page PDF."""
    try:
        page = int(snapshot.get("page") or 0)
    except (TypeError, ValueError):
        return False
    return page > 0 and normalize_proof_bbox(snapshot.get("bbox")) is not None


def _normalized_anchor_values(
    snapshot: dict[str, Any],
    field: str,
    *,
    target: str,
) -> list[str]:
    """Normaliser une valeur scalaire ou une liste pour le score d'ancrage."""
    raw_value = snapshot.get(field)
    raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized: list[str] = []
    for value in raw_values:
        if target == "title" and is_generic_title(str(value or "")):
            continue
        text = (
            _normalize_table_anchor_title(value)
            if target == "title"
            else normalize_for_matching(str(value or ""), target=target)
        )
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _best_text_similarity(left: list[str], right: list[str]) -> float:
    """Retourner la meilleure similarite textuelle entre deux petits ensembles."""
    if not left or not right:
        return 0.0
    return max(
        SequenceMatcher(None, first, second).ratio()
        for first in left
        for second in right
    )


def _jaccard_anchor_values(left: list[str], right: list[str]) -> float:
    """Calculer le Jaccard de deux listes normalisees."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _resolve_visual_table_anchor(
    event_snapshot: dict[str, Any],
    opposite_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resoudre une ancre opposee avec plusieurs signaux et une marge.

    Le titre exact reste le signal le plus fort, mais un titre reformule ou
    absent peut etre compense par le resume, les indicateurs et les en-tetes.
    Aucun candidat n'est retenu si le score est faible ou ambigu.
    """
    event_section = _normalize_table_anchor_section(event_snapshot.get("section"))
    event_titles = _normalized_anchor_values(
        {
            "values": [
                event_snapshot.get("title"),
                event_snapshot.get("page_context_title"),
            ]
        },
        "values",
        target="title",
    )
    event_summaries = _normalized_anchor_values(
        event_snapshot,
        "table_summary",
        target="generic",
    )
    event_indicators = _normalized_anchor_values(
        event_snapshot,
        "indicators",
        target="indicator",
    )
    event_headers = _normalized_anchor_values(
        event_snapshot,
        "headers",
        target="header",
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in opposite_snapshots.values():
        if not _snapshot_has_visual_render_anchor(candidate):
            continue
        candidate_section = _normalize_table_anchor_section(
            candidate.get("section")
        )
        if event_section and candidate_section != event_section:
            continue

        candidate_titles = _normalized_anchor_values(
            {
                "values": [
                    candidate.get("title"),
                    candidate.get("page_context_title"),
                ]
            },
            "values",
            target="title",
        )
        candidate_summaries = _normalized_anchor_values(
            candidate,
            "table_summary",
            target="generic",
        )
        candidate_indicators = _normalized_anchor_values(
            candidate,
            "indicators",
            target="indicator",
        )
        candidate_headers = _normalized_anchor_values(
            candidate,
            "headers",
            target="header",
        )

        score = 0.0
        if set(event_titles) & set(candidate_titles):
            score += 6.0
        else:
            title_similarity = _best_text_similarity(
                event_titles,
                candidate_titles,
            )
            if title_similarity >= 0.72:
                score += 4.0 * title_similarity

        if set(event_summaries) & set(candidate_summaries):
            score += 4.0
        else:
            summary_similarity = _best_text_similarity(
                event_summaries,
                candidate_summaries,
            )
            if summary_similarity >= 0.76:
                score += 2.5 * summary_similarity

        indicator_overlap = _jaccard_anchor_values(
            event_indicators,
            candidate_indicators,
        )
        if indicator_overlap >= 0.20:
            score += 4.0 * indicator_overlap
        if (
            event_indicators
            and candidate_indicators
            and event_indicators[0] == candidate_indicators[0]
        ):
            score += 1.0

        header_overlap = _jaccard_anchor_values(
            event_headers,
            candidate_headers,
        )
        if header_overlap >= 0.25:
            score += 1.5 * header_overlap

        try:
            event_rows = int(event_snapshot.get("row_count") or 0)
            candidate_rows = int(candidate.get("row_count") or 0)
        except (TypeError, ValueError):
            event_rows = candidate_rows = 0
        if event_rows > 0 and candidate_rows > 0:
            size_ratio = min(event_rows, candidate_rows) / max(
                event_rows,
                candidate_rows,
            )
            if size_ratio >= 0.60:
                score += 0.5 * size_ratio

        if score > 0:
            scored.append((score, candidate))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 4.0 or best_score - second_score < 0.75:
        return None
    return best_candidate


def _infer_opposite_page_from_matched_pairs(
    event_snapshot: dict[str, Any],
    matched_pairs: list[dict[str, Any]],
    event_snapshots: dict[str, dict[str, Any]],
    opposite_snapshots: dict[str, dict[str, Any]],
    *,
    event_side: str,
) -> int | None:
    """Estimer la page opposee a partir des paires voisines deja appariees."""
    try:
        event_page = int(event_snapshot.get("page") or 0)
    except (TypeError, ValueError):
        return None
    if event_page < 1 or event_side not in {"previous", "current"}:
        return None

    event_id_key = (
        "previous_table_id" if event_side == "previous" else "current_table_id"
    )
    opposite_id_key = (
        "current_table_id" if event_side == "previous" else "previous_table_id"
    )
    page_map: dict[int, list[int]] = {}
    for pair in matched_pairs:
        event_id = str(pair.get(event_id_key, "") or "").strip()
        opposite_id = str(pair.get(opposite_id_key, "") or "").strip()
        event_anchor = event_snapshots.get(event_id, {})
        opposite_anchor = opposite_snapshots.get(opposite_id, {})
        try:
            event_anchor_page = int(event_anchor.get("page") or 0)
            opposite_anchor_page = int(opposite_anchor.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if event_anchor_page < 1 or opposite_anchor_page < 1:
            continue
        page_map.setdefault(event_anchor_page, []).append(opposite_anchor_page)

    anchors = sorted(
        (
            source_page,
            round(sum(target_pages) / len(target_pages)),
        )
        for source_page, target_pages in page_map.items()
        if target_pages
    )
    if not anchors:
        return None

    previous_anchor = next(
        (
            anchor
            for anchor in reversed(anchors)
            if anchor[0] <= event_page
        ),
        None,
    )
    next_anchor = next(
        (anchor for anchor in anchors if anchor[0] >= event_page),
        None,
    )
    if previous_anchor and next_anchor:
        source_span = next_anchor[0] - previous_anchor[0]
        if source_span <= 0:
            return max(1, previous_anchor[1])
        progress = (event_page - previous_anchor[0]) / source_span
        inferred = previous_anchor[1] + progress * (
            next_anchor[1] - previous_anchor[1]
        )
        return max(1, round(inferred))
    if previous_anchor:
        return max(1, event_page + previous_anchor[1] - previous_anchor[0])
    if next_anchor:
        return max(1, event_page + next_anchor[1] - next_anchor[0])
    return None


def _call_openai_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int | None = None,
    temperature: float = 0.0,
    api_retry_max: int = 2,
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison",
    response_model: type | None = None,
) -> dict[str, Any]:
    """Appeler l'API OpenAI avec sortie JSON.

    Quand *response_model* est une sous-classe de ``pydantic.BaseModel``, l'appel
    utilise les **Structured Outputs** OpenAI pour garantir la conformite au schema.
    Le modele valide est reconverti en dict pour que les appelants gardent une
    interface identique.

    ``max_completion_tokens=None`` (defaut) laisse le modele s'arreter naturellement
    sans plafond artificiel — privilegier la qualite complete plutot que la vitesse.
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        timeout=OPENAI_COMPARISON_TIMEOUT_SECONDS,
        # Les retries sont geres par la boucle applicative ci-dessous afin
        # d'eviter de multiplier les tentatives avec celles du SDK.
        max_retries=0,
    )
    last_error: Exception | None = None
    use_structured = response_model is not None
    for attempt in range(api_retry_max + 1):
        if attempt > 0:
            time.sleep(1.5 * (2 ** (attempt - 1)))
        try:
            if use_structured:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": response_model,
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = client.beta.chat.completions.parse(**kwargs)
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Structured Output parsing returned None")
                data = parsed.model_dump()
            else:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                response = client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("OpenAI response is not a JSON object")
            if usage_recorder is not None:
                prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
                usage_recorder.append(
                    {
                        "model": model,
                        "call_kind": call_kind,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
            return data
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = (
                "rate" in message
                and "limit" in message
                or "timeout" in message
                or "timed out" in message
                or "connection" in message
                or "connect" in message
            )
            if not retryable or attempt >= api_retry_max:
                break
    raise RuntimeError(f"OpenAI comparison call failed: {last_error}")


def _call_openai_embeddings(
    *,
    model: str,
    inputs: list[str],
    usage_recorder: list[dict[str, Any]] | None = None,
    call_kind: str = "comparison_embeddings",
) -> list[list[float]]:
    """Encoder les vues de tableaux par lots pour la recuperation hybride RBC."""
    if not inputs:
        return []
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        timeout=OPENAI_COMPARISON_TIMEOUT_SECONDS,
    )
    vectors: list[list[float]] = []
    for start in range(0, len(inputs), 96):
        response = client.embeddings.create(model=model, input=inputs[start : start + 96])
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend([list(item.embedding) for item in ordered])
        if usage_recorder is not None:
            prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(response)
            usage_recorder.append(
                {
                    "model": model,
                    "call_kind": call_kind,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )
    return vectors


def compare_reports_gpt4o(
    previous_dir: Path | str,
    current_dir: Path | str,
    out_root: Path | str,
    *,
    model: str | None = None,
    config_path: str | None = None,
    reference_resolution: dict[str, Any] | None = None,
    source_pdf_previous: str | None = None,
    source_pdf_current: str | None = None,
    runtime_extraction_sec: float | None = None,
    extraction_run_metrics: dict[str, Any] | None = None,
) -> Path:
    """Executer le pipeline complet de comparaison rapport-a-rapport et ecrire l'artefact.

    Point d'entree public utilise par le CLI et l'application Dash. Charge les
    artefacts canoniques ``tables.json`` des deux trimestres, enrichit les tables
    pour le matching, execute le matcher multicouche, calcule les diffs semantiques
    par paire, agregue les resumes et metriques, puis ecrit ``comparison.json``
    dans un repertoire de sortie.

    Args:
        previous_dir: Repertoire d'extraction du trimestre de reference.
        current_dir: Repertoire d'extraction du trimestre courant.
        out_root: Repertoire racine ou le dossier de comparaison est cree.
        model: Surcharge optionnelle du modele OpenAI.
        config_path: Chemin optionnel de la configuration des modeles.
        reference_resolution: Metadonnees optionnelles decrivant la resolution
            du trimestre de reference.
        source_pdf_previous: Chemin PDF optionnel du rapport precedent.
        source_pdf_current: Chemin PDF optionnel du rapport courant.
        runtime_extraction_sec: Temps d'extraction optionnel propage dans les
            metriques finales.
        extraction_run_metrics: Metriques d'extraction optionnelles fusionnees
            dans les metriques finales.

    Returns:
        Chemin vers l'artefact ``comparison.json`` genere.
    """
    comparison_started_at = time.monotonic()
    previous_dir_path = _coerce_pathlike(previous_dir, "previous_dir")
    current_dir_path = _coerce_pathlike(current_dir, "current_dir")
    out_root_path = _coerce_pathlike(out_root, "out_root")

    previous_payload = _load_tables_payload(previous_dir_path)
    current_payload = _load_tables_payload(current_dir_path)

    previous_tables = [entry for entry in list(previous_payload.get("tables", []) or []) if isinstance(entry, dict)]
    current_tables = [entry for entry in list(current_payload.get("tables", []) or []) if isinstance(entry, dict)]
    (
        previous_business_tables,
        previous_artifact_refs,
        previous_suspect_refs,
    ) = _partition_tables_by_status(previous_tables)
    (
        current_business_tables,
        current_artifact_refs,
        current_suspect_refs,
    ) = _partition_tables_by_status(current_tables)

    def _build_views() -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        """Construit les vues (cards, details, snapshots) pour T1 et T2."""
        return (
            [_table_card(entry) for entry in previous_business_tables],
            [_table_card(entry) for entry in current_business_tables],
            {entry["table_id"]: _table_detail(entry) for entry in previous_business_tables},
            {entry["table_id"]: _table_detail(entry) for entry in current_business_tables},
            {entry["table_id"]: _table_snapshot(entry) for entry in previous_tables},
            {entry["table_id"]: _table_snapshot(entry) for entry in current_tables},
        )

    (
        previous_cards,
        current_cards,
        previous_lookup,
        current_lookup,
        previous_snapshots,
        current_snapshots,
    ) = _build_views()

    bank_code = str(current_payload.get("bank_code") or previous_payload.get("bank_code") or "")
    if not bank_code:
        raise ValueError("Missing bank_code in tables.json payloads")
    year_previous = int(previous_payload.get("year", 0) or 0)
    year_current = int(current_payload.get("year", 0) or 0)
    quarter_previous = str(previous_payload.get("quarter", "") or "")
    quarter_current = str(current_payload.get("quarter", "") or "")
    model_name = str(model or resolve_openai_model("default_genai", config_path=config_path))
    usage_records: list[dict[str, Any]] = []
    matching_settings = get_matching_thresholds(
        config_path or "configs/bank_profiles.yaml",
        bank_code=bank_code,
    )
    configured_hybrid_quarters = {
        str(value or "").strip().lower()
        for value in list(matching_settings.get("hybrid_embedding_quarters", []) or [])
        if str(value or "").strip()
    }
    hybrid_recovery_enabled = (
        bank_code.strip().lower() == "rbc"
        and bool(matching_settings.get("hybrid_embedding_recovery_enabled", False))
        and quarter_current.strip().lower() in configured_hybrid_quarters
    )

    match_result = _run_table_matching(
        previous_cards,
        current_cards,
        model=model_name,
        call_openai_json=_call_openai_json,
        usage_recorder=usage_records,
        hybrid_recovery_enabled=hybrid_recovery_enabled,
        hybrid_embedding_model=str(
            matching_settings.get("hybrid_embedding_model", "text-embedding-3-large")
        ),
        hybrid_top_k=max(1, int(matching_settings.get("hybrid_embedding_top_k", 5) or 5)),
        hybrid_min_confidence=float(matching_settings.get("hybrid_min_confidence", 0.75) or 0.75),
        call_openai_embeddings=_call_openai_embeddings,
    )

    def _exclude_unmatched_boundary_candidates(
        items: list[dict[str, Any]],
        snapshots: dict[str, dict[str, Any]],
        *,
        side: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Garder les candidats de bordure pour le matching, pas comme changements."""
        kept: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for item in items:
            table_id = str(item.get("table_id", "") or "").strip()
            snapshot = snapshots.get(table_id, {})
            if not _is_boundary_inventory_candidate(snapshot):
                kept.append(item)
                continue
            excluded.append(
                {
                    **item,
                    **snapshot,
                    "scope_side": side,
                    "exclusion_reason": (
                        "Tableau detecte uniquement sur une page limitrophe ajoutee "
                        "pour l'appariement; son absence de correspondance ne constitue "
                        "pas un ajout ou un retrait dans le perimetre compare."
                    ),
                }
            )
        return kept, excluded

    (
        match_result["tables_added"],
        boundary_scope_exclusions_current,
    ) = _exclude_unmatched_boundary_candidates(
        list(match_result.get("tables_added", []) or []),
        current_snapshots,
        side="current",
    )
    (
        match_result["tables_removed"],
        boundary_scope_exclusions_previous,
    ) = _exclude_unmatched_boundary_candidates(
        list(match_result.get("tables_removed", []) or []),
        previous_snapshots,
        side="previous",
    )

    tables_added: list[dict[str, Any]] = []
    for item in match_result["tables_added"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "ajoute",
        }
        tables_added.append(
            {
                **item,
                **current_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="ajoute",
                ),
            }
        )

    tables_removed: list[dict[str, Any]] = []
    for item in match_result["tables_removed"]:
        table_id = item["table_id"]
        technical_diff = {
            "indicators_added": [],
            "indicators_removed": [],
            "indicators_renamed": [],
            "footnotes_added": [],
            "footnotes_removed": [],
            "footnotes_renamed": [],
            "table_level_change": "supprime",
        }
        tables_removed.append(
            {
                **item,
                **previous_snapshots[table_id],
                "analyst_assessment": build_analyst_assessment(
                    table_context=previous_lookup[table_id],
                    technical_diff=technical_diff,
                    change_kind="supprime",
                ),
            }
        )

    # --- Devil's Advocate: second-opinion review on unmatched / low-confidence ---
    low_confidence_pairs = [
        p for p in match_result.get("matched_pairs", []) if float(p.get("match_confidence", 1.0)) < 0.90
    ]
    da_added_cards = [
        _table_card(entry)
        for entry in current_business_tables
        if any(a.get("table_id") == entry.get("table_id") for a in match_result.get("tables_added", []))
    ]
    da_removed_cards = [
        _table_card(entry)
        for entry in previous_business_tables
        if any(r.get("table_id") == entry.get("table_id") for r in match_result.get("tables_removed", []))
    ]
    # Le recuperateur RBC inclut deja une inspection finale fail-closed. Une
    # promotion ulterieure non inspectee recreerait precisement les cascades
    # que cette voie opt-in cherche a eliminer.
    if hybrid_recovery_enabled:
        da_result: dict[str, Any] = {
            "new_matches": [],
            "contested_pairs": [],
            "warnings": [],
        }
    else:
        da_result = _devil_advocate_review(
            da_added_cards,
            da_removed_cards,
            low_confidence_pairs,
            model=model_name,
            call_openai_json=_call_openai_json,
            usage_recorder=usage_records,
        )
    # Promote new matches found by Devil's Advocate
    for new_match in da_result.get("new_matches", []):
        prev_id = str(new_match.get("previous_table_id", "") or "").strip()
        cur_id = str(new_match.get("current_table_id", "") or "").strip()
        if not prev_id or not cur_id:
            continue
        if prev_id not in previous_snapshots or cur_id not in current_snapshots:
            logger.warning("Devil's Advocate: skipping invalid match %s <-> %s", prev_id, cur_id)
            continue
        # Add to matched_pairs
        match_result["matched_pairs"].append(
            {
                "previous_table_id": prev_id,
                "current_table_id": cur_id,
                "match_confidence": float(new_match.get("match_confidence", 0.75)),
                "reason": str(new_match.get("reason", "")),
                "source": "devil_advocate",
            }
        )
        # Remove from tables_added / tables_removed
        tables_added = [t for t in tables_added if t.get("table_id") != cur_id]
        tables_removed = [t for t in tables_removed if t.get("table_id") != prev_id]
        logger.info(
            "Devil's Advocate promoted match: %s <-> %s (conf=%.2f)",
            prev_id,
            cur_id,
            float(new_match.get("match_confidence", 0.75)),
        )
    # Mark contested pairs for review
    for contested in da_result.get("contested_pairs", []):
        prev_id = str(contested.get("previous_table_id", "") or "").strip()
        cur_id = str(contested.get("current_table_id", "") or "").strip()
        for pair in match_result["matched_pairs"]:
            if pair.get("previous_table_id") == prev_id and pair.get("current_table_id") == cur_id:
                pair["review_required"] = True
                pair["devil_advocate_reason"] = str(contested.get("reason", ""))
                logger.info("Devil's Advocate contested pair: %s <-> %s", prev_id, cur_id)

    artifacts_confirmed_previous: list[dict[str, Any]] = []
    for item in previous_artifact_refs:
        table_id = item["table_id"]
        artifacts_confirmed_previous.append({**item, **previous_snapshots[table_id]})

    artifacts_confirmed_current: list[dict[str, Any]] = []
    for item in current_artifact_refs:
        table_id = item["table_id"]
        artifacts_confirmed_current.append({**item, **current_snapshots[table_id]})

    extraction_suspects_previous = _merge_extraction_suspect_side(
        previous_tables,
        previous_suspect_refs,
        previous_snapshots,
    )
    extraction_suspects_current = _merge_extraction_suspect_side(
        current_tables,
        current_suspect_refs,
        current_snapshots,
    )

    pair_comparisons: list[dict[str, Any]] = []
    diff_calls_total = 0
    _sanity_check_enabled = bool(source_pdf_previous and source_pdf_current)

    def _worst_render_status(statuses: list[str]) -> str:
        """Retourne le pire statut de rendu visuel (priorité aux erreurs)."""
        for candidate in (
            "skipped_missing_pdf",
            "skipped_missing_anchor",
            "skipped_missing_bbox",
            "skipped_render_failed",
        ):
            if candidate in statuses:
                return candidate
        return "ok"

    def _resolve_opposite_table_anchor(
        event_snapshot: dict[str, Any],
        opposite_snapshots: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Trouve l'ancrage de tableau correspondant dans le trimestre opposé."""
        return _resolve_visual_table_anchor(
            event_snapshot,
            opposite_snapshots,
        )

    def _render_pair_proofs(
        previous_table_snapshot: dict[str, Any],
        current_table_snapshot: dict[str, Any],
    ) -> tuple[bytes | None, bytes | None, str]:
        """Rend les deux preuves visuelles (T1, T2) pour une paire appariée."""
        previous_render, previous_status = render_visual_sanity_proof(
            source_pdf_previous,
            page=previous_table_snapshot.get("page"),
            bbox=previous_table_snapshot.get("bbox"),
        )
        current_render, current_status = render_visual_sanity_proof(
            source_pdf_current,
            page=current_table_snapshot.get("page"),
            bbox=current_table_snapshot.get("bbox"),
        )
        return (
            previous_render,
            current_render,
            _worst_render_status([previous_status, current_status]),
        )

    def _render_table_event_proofs(
        *,
        event_type: str,
        event_snapshot: dict[str, Any],
    ) -> tuple[bytes | None, bytes | None, str, str]:
        """Rend les deux preuves visuelles pour un événement (ajout / retrait de tableau)."""
        normalized_event_type = str(event_type or "").strip().lower()
        render_mode = "full"
        if normalized_event_type == "table_added":
            opposite_anchor = _resolve_opposite_table_anchor(
                event_snapshot,
                previous_snapshots,
            )
            if opposite_anchor is None:
                opposite_page = _infer_opposite_page_from_matched_pairs(
                    event_snapshot,
                    match_result["matched_pairs"],
                    current_snapshots,
                    previous_snapshots,
                    event_side="current",
                )
                if opposite_page is None:
                    return None, None, "skipped_missing_anchor", render_mode
                render_mode = "full_page_context_fallback"
                previous_render, previous_status = render_visual_sanity_proof(
                    source_pdf_previous,
                    page=opposite_page,
                    bbox=None,
                    allow_full_page_fallback=True,
                )
            else:
                previous_render, previous_status = render_visual_sanity_proof(
                    source_pdf_previous,
                    page=opposite_anchor.get("page"),
                    bbox=opposite_anchor.get("bbox"),
                )
            current_render, current_status = render_visual_sanity_proof(
                source_pdf_current,
                page=event_snapshot.get("page"),
                bbox=event_snapshot.get("bbox"),
            )
        else:
            opposite_anchor = _resolve_opposite_table_anchor(
                event_snapshot,
                current_snapshots,
            )
            if opposite_anchor is None:
                opposite_page = _infer_opposite_page_from_matched_pairs(
                    event_snapshot,
                    match_result["matched_pairs"],
                    previous_snapshots,
                    current_snapshots,
                    event_side="previous",
                )
                if opposite_page is None:
                    return None, None, "skipped_missing_anchor", render_mode
                render_mode = "full_page_context_fallback"
            else:
                opposite_page = opposite_anchor.get("page")
            previous_render, previous_status = render_visual_sanity_proof(
                source_pdf_previous,
                page=event_snapshot.get("page"),
                bbox=event_snapshot.get("bbox"),
            )
            current_render, current_status = render_visual_sanity_proof(
                source_pdf_current,
                page=opposite_page,
                bbox=None if opposite_anchor is None else opposite_anchor.get("bbox"),
                allow_full_page_fallback=opposite_anchor is None,
            )
        return (
            previous_render,
            current_render,
            _worst_render_status([previous_status, current_status]),
            render_mode,
        )

    for pair in match_result["matched_pairs"]:
        previous_table_id = pair["previous_table_id"]
        current_table_id = pair["current_table_id"]
        diff = diff_table_pair_gpt(
            previous_lookup[previous_table_id],
            current_lookup[current_table_id],
            model=model_name,
            call_openai_json=_call_openai_json,
            usage_recorder=usage_records,
            max_validation_attempts=_MATCHING_VALIDATION_ATTEMPTS,
        )
        diff_calls_total += _coerce_int(diff.get("diff_calls_total"))

        # --- Visual Sanity Check (post-diff) ---
        diff.setdefault(
            "visual_sanity_scope",
            ["indicators", "footnotes", "tables"],
        )
        diff.setdefault("visual_sanity_render_mode", "full")
        diff.setdefault("visual_sanity_applied", False)
        diff.setdefault("visual_sanity_rejected_count", 0)
        diff.setdefault("visual_sanity_render_status", "ok")
        if _sanity_check_enabled and any(
            diff.get("technical_diff", {}).get(k)
            for k in (
                "indicators_added",
                "indicators_removed",
                "indicators_renamed",
                "footnotes_added",
                "footnotes_removed",
                "footnotes_renamed",
            )
        ):
            prev_render, curr_render, render_status = _render_pair_proofs(
                previous_snapshots[previous_table_id],
                current_snapshots[current_table_id],
            )
            if render_status == "ok":
                diff = visual_sanity_check(
                    prev_render,
                    curr_render,
                    diff,
                    model=model_name,
                    call_openai_json=_call_openai_json,
                    usage_recorder=usage_records,
                )
            else:
                diff.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                    )
                )

        filtered_diff = _filter_noise_from_diff(diff["technical_diff"])
        filtered_diff["table_level_change"] = recompute_table_level_change(filtered_diff)
        pair_comparisons.append(
            {
                "previous_table_id": previous_table_id,
                "current_table_id": current_table_id,
                "match_confidence": pair["match_confidence"],
                "match_reason": pair.get("reason", ""),
                "diff_mode": str(diff.get("diff_mode", "") or ""),
                "previous_table": previous_snapshots[previous_table_id],
                "current_table": current_snapshots[current_table_id],
                "technical_diff": filtered_diff,
                "analyst_assessment": build_analyst_assessment(
                    table_context=current_lookup[current_table_id],
                    technical_diff=filtered_diff,
                    change_kind="modifie",
                ),
                "reason": diff["reason"],
                "visual_sanity_applied": bool(diff.get("visual_sanity_applied", False)),
                "visual_sanity_rejected_count": _coerce_int(diff.get("visual_sanity_rejected_count")),
                "visual_sanity_scope": list(diff.get("visual_sanity_scope") or []),
                "visual_sanity_render_mode": str(diff.get("visual_sanity_render_mode", "") or ""),
                "visual_sanity_render_status": str(diff.get("visual_sanity_render_status", "") or ""),
            }
        )

    if _sanity_check_enabled:
        filtered_tables_added: list[dict[str, Any]] = []
        for item in tables_added:
            (
                previous_render,
                current_render,
                render_status,
                render_mode,
            ) = _render_table_event_proofs(
                event_type="table_added",
                event_snapshot=item,
            )
            if render_status != "ok":
                item.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                        render_mode=render_mode,
                    )
                )
                filtered_tables_added.append(item)
                continue
            verdict = visual_sanity_check_table_event(
                previous_render,
                current_render,
                event_type="table_added",
                table_id=str(item.get("table_id", "") or ""),
                table_title=str(item.get("title", "") or ""),
                model=model_name,
                call_openai_json=_call_openai_json,
                usage_recorder=usage_records,
            )
            item.update({key: value for key, value in verdict.items() if key != "confirmed"})
            item["visual_sanity_render_mode"] = render_mode
            if verdict.get("confirmed", True):
                filtered_tables_added.append(item)
        tables_added = filtered_tables_added

        filtered_tables_removed: list[dict[str, Any]] = []
        for item in tables_removed:
            (
                previous_render,
                current_render,
                render_status,
                render_mode,
            ) = _render_table_event_proofs(
                event_type="table_removed",
                event_snapshot=item,
            )
            if render_status != "ok":
                item.update(
                    _visual_sanity_meta(
                        applied=False,
                        rejected_count=0,
                        render_status=render_status,
                        render_mode=render_mode,
                    )
                )
                filtered_tables_removed.append(item)
                continue
            verdict = visual_sanity_check_table_event(
                previous_render,
                current_render,
                event_type="table_removed",
                table_id=str(item.get("table_id", "") or ""),
                table_title=str(item.get("title", "") or ""),
                model=model_name,
                call_openai_json=_call_openai_json,
                usage_recorder=usage_records,
            )
            item.update({key: value for key, value in verdict.items() if key != "confirmed"})
            item["visual_sanity_render_mode"] = render_mode
            if verdict.get("confirmed", True):
                filtered_tables_removed.append(item)
        tables_removed = filtered_tables_removed

    # --- T-1 Anchoring: flag likely extraction errors based on row count drift ---
    try:
        from vigilance.config.loader import load_config

        _anchor_cfg = load_config("configs/bank_profiles.yaml")
        _vision_cfg = _anchor_cfg.get("vision_extraction", {})
        _t1_anchor_enabled = bool(_vision_cfg.get("vision_t1_anchor_enabled", False))
        _t1_anchor_threshold = float(_vision_cfg.get("vision_t1_anchor_diff_threshold", 0.20))
    except Exception:
        _t1_anchor_enabled = False
        _t1_anchor_threshold = 0.20

    if _t1_anchor_enabled:
        try:
            from vigilance.extraction.vision_t1_anchor import anchor_against_previous as _anchor_check

            for pair_comp in pair_comparisons:
                prev_table = pair_comp.get("previous_table", {})
                curr_table = pair_comp.get("current_table", {})
                prev_indicators = [
                    str(i) if isinstance(i, str) else str(i.get("label", i.get("name", "")))
                    for i in (prev_table.get("indicators") or [])
                ]
                curr_indicators = [
                    str(i) if isinstance(i, str) else str(i.get("label", i.get("name", "")))
                    for i in (curr_table.get("indicators") or [])
                ]

                anchor_result = _anchor_check(
                    table_id=str(curr_table.get("table_id", "")),
                    table_title=str(curr_table.get("title", "")),
                    current_indicators=curr_indicators,
                    previous_indicators=prev_indicators,
                    diff_threshold=_t1_anchor_threshold,
                )

                if not anchor_result.skipped:
                    pair_comp["t1_anchor"] = {
                        "likely_extraction_error": anchor_result.likely_extraction_error,
                        "explanation": anchor_result.explanation,
                        "current_count": anchor_result.current_count,
                        "previous_count": anchor_result.previous_count,
                        "diff_ratio": anchor_result.diff_ratio,
                    }
                    if anchor_result.likely_extraction_error:
                        logger.warning(
                            "T-1 anchor: table %s flagged as likely extraction error (prev=%d, curr=%d, diff=%.0f%%)",
                            anchor_result.table_id,
                            anchor_result.previous_count,
                            anchor_result.current_count,
                            anchor_result.diff_ratio * 100,
                        )
        except Exception as _t1_exc:
            logger.warning("T-1 anchoring failed (non-fatal): %s", _t1_exc)
    # --- End T-1 Anchoring ---

    indicator_changes_total, footnote_changes_total = _count_pair_changes(pair_comparisons)
    high_priority_items_total = _count_high_priority_items(
        pair_comparisons,
        tables_added,
        tables_removed,
    )
    comparison_runtime_sec = round(max(0.0, time.monotonic() - comparison_started_at), 3)
    run_metrics = _build_run_metrics(
        usage_records=usage_records,
        match_result=match_result,
        diff_calls_total=diff_calls_total,
        comparison_runtime_sec=comparison_runtime_sec,
        model_name=model_name,
        extraction_run_metrics=extraction_run_metrics,
        runtime_extraction_sec=float(runtime_extraction_sec or 0.0),
    )

    out_dir = out_root_path / bank_code / f"{year_current}_{quarter_current}_vs_{year_previous}_{quarter_previous}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id()
    # Archive PDFs inside the run directory for cross-OS portability: Dash falls
    # back to previous_report.pdf / current_report.pdf when absolute paths stored
    # in the JSON become invalid (e.g. run produced on macOS then opened on Windows).
    archived_pdf_previous = _archive_source_pdf(
        source_pdf_previous, out_dir / "previous_report.pdf"
    )
    archived_pdf_current = _archive_source_pdf(
        source_pdf_current, out_dir / "current_report.pdf"
    )
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "artifact_type": "report_comparison",
        "run_id": run_id,
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": quarter_previous,
        "year_current": year_current,
        "quarter_current": quarter_current,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_pdf_previous": str(source_pdf_previous or "").strip(),
        "source_pdf_current": str(source_pdf_current or "").strip(),
        "archived_pdf_previous": archived_pdf_previous,
        "archived_pdf_current": archived_pdf_current,
        "model_version": model_name,
        "prompt_version_match": MATCH_PROMPT_VERSION,
        "prompt_version_diff": DIFF_PROMPT_VERSION,
        "reference_resolution": (
            dict(reference_resolution)
            if isinstance(reference_resolution, dict)
            else {
                "mode": "automatique",
                "year_previous": year_previous,
                "quarter_previous": quarter_previous,
                "rule": REFERENCE_RESOLUTION_RULE,
            }
        ),
        "matching": {
            "matched_pairs": match_result["matched_pairs"],
            "tables_added": tables_added,
            "tables_removed": tables_removed,
            "artifacts_confirmed_previous": artifacts_confirmed_previous,
            "artifacts_confirmed_current": artifacts_confirmed_current,
            "extraction_suspects_previous": extraction_suspects_previous,
            "extraction_suspects_current": extraction_suspects_current,
            "boundary_scope_exclusions_previous": boundary_scope_exclusions_previous,
            "boundary_scope_exclusions_current": boundary_scope_exclusions_current,
        },
        "pair_comparisons": pair_comparisons,
        "run_metrics": run_metrics,
        "summary": {
            "matched_pairs_total": len(match_result["matched_pairs"]),
            "tables_added_total": len(tables_added),
            "tables_removed_total": len(tables_removed),
            "artifacts_confirmed_previous_total": len(artifacts_confirmed_previous),
            "artifacts_confirmed_current_total": len(artifacts_confirmed_current),
            "extraction_suspects_previous_total": len(extraction_suspects_previous),
            "extraction_suspects_current_total": len(extraction_suspects_current),
            "boundary_scope_exclusions_previous_total": len(
                boundary_scope_exclusions_previous
            ),
            "boundary_scope_exclusions_current_total": len(
                boundary_scope_exclusions_current
            ),
            "indicator_changes_total": indicator_changes_total,
            "footnote_changes_total": footnote_changes_total,
            "high_priority_items_total": high_priority_items_total,
        },
    }
    return _atomic_write_json(out_dir / "comparison.json", payload)
