"""Assemblage des observations textuelles consolidées.

La décision de regrouper des chunks vient du LLM. Ce module valide seulement
les groupes reçus et construit les objets d'observation consommés par Dash et
l'export Excel. Si aucun groupe valide n'est fourni, les changements restent
atomiques.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_IMPACT_ORDER = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}
_ACTION_ORDER = {
    "revue_prioritaire": 0,
    "investigation": 1,
    "confirmation": 2,
    "information": 3,
    "aucune": 4,
}


def build_atomic_observations(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retourne une observation par changement, sans consolidation."""
    observations = [_atomic_observation(change) for change in changes if isinstance(change, dict)]
    observations.sort(key=_observation_sort_key)
    return observations


def candidate_batches_for_llm(
    changes: list[dict[str, Any]],
    *,
    max_changes_per_batch: int = 24,
    max_batch_chars: int = 20_000,
) -> list[list[dict[str, Any]]]:
    """Prépare des lots techniques pour la consolidation LLM.

    Ce n'est pas une décision de regroupement. Le lot borne seulement le volume
    transmis au modèle; il ne filtre pas par page, sous-section ou sujet.
    """
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for change in changes:
        if not isinstance(change, dict) or change.get("diff_type") == "unchanged":
            continue
        change_chars = _llm_payload_size(change)
        if current and (
            len(current) >= max_changes_per_batch
            or current_chars + change_chars > max_batch_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(change)
        current_chars += change_chars
    if current:
        batches.append(current)
    return batches


def _llm_payload_size(change: dict[str, Any]) -> int:
    total = 160
    for key in (
        "change_id",
        "diff_type",
        "subsection_heading",
        "current_subsection_heading",
        "previous_subsection_heading",
        "current_hierarchy_path",
        "previous_hierarchy_path",
        "change_summary",
        "chunk_topic",
        "canonical_topic",
        "source_text_t1",
        "source_text_t2",
        "semantic_text_t1",
        "semantic_text_t2",
        "genai_triage",
        "objective_matches",
    ):
        value = change.get(key)
        if value is not None:
            text = str(value)
            if key in {"source_text_t1", "source_text_t2", "semantic_text_t1", "semantic_text_t2"}:
                total += min(len(text), 900)
            elif key == "change_summary":
                total += min(len(text), 700)
            else:
                total += len(text)
    return total


def build_observations_from_group_specs(
    changes: list[dict[str, Any]],
    group_specs: list[dict[str, Any]],
    atomic_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Construit les observations à partir de groupes décidés par GPT."""
    by_id = {
        str(change.get("change_id") or "").strip(): change
        for change in changes
        if isinstance(change, dict) and str(change.get("change_id") or "").strip()
    }
    used: set[str] = set()
    observations: list[dict[str, Any]] = []

    for spec in group_specs:
        if not isinstance(spec, dict):
            continue
        raw_ids = [str(value).strip() for value in spec.get("source_change_ids") or []]
        if not raw_ids or len(raw_ids) != len(set(raw_ids)):
            continue
        if any(change_id not in by_id for change_id in raw_ids):
            continue
        if any(change_id in used for change_id in raw_ids):
            continue
        if len(raw_ids) == 1:
            reason = _atomic_reason_from_spec(spec)
            observations.append(_atomic_observation(by_id[raw_ids[0]], non_grouping_reason=reason))
            used.add(raw_ids[0])
            continue
        group_changes = [by_id[change_id] for change_id in raw_ids]
        observations.append(_consolidated_observation(group_changes, spec))
        used.update(raw_ids)

    for spec in atomic_specs or []:
        if not isinstance(spec, dict):
            continue
        change_id = _atomic_id_from_spec(spec)
        if not change_id or change_id not in by_id or change_id in used:
            continue
        observations.append(
            _atomic_observation(by_id[change_id], non_grouping_reason=_atomic_reason_from_spec(spec))
        )
        used.add(change_id)

    for change_id, change in by_id.items():
        if change_id not in used:
            observations.append(_atomic_observation(change))

    observations.sort(key=_observation_sort_key)
    return observations


def _atomic_id_from_spec(spec: dict[str, Any]) -> str:
    value = spec.get("change_id") or spec.get("source_change_id")
    if value:
        return str(value).strip()
    values = spec.get("source_change_ids")
    if isinstance(values, list) and len(values) == 1:
        return str(values[0]).strip()
    return ""


def _atomic_reason_from_spec(spec: dict[str, Any]) -> str | None:
    reason = str(
        spec.get("non_grouping_reason")
        or spec.get("rationale")
        or spec.get("reason")
        or ""
    ).strip()
    return reason or None


def _atomic_observation(
    change: dict[str, Any],
    *,
    non_grouping_reason: str | None = None,
) -> dict[str, Any]:
    observation = dict(change)
    observation["observation_type"] = "atomic_change"
    observation["source_change_ids"] = [str(observation.get("change_id") or "")]
    observation["consolidated_change_count"] = 1
    if non_grouping_reason:
        observation["non_grouping_reason"] = non_grouping_reason
    return observation


def _consolidated_observation(
    changes: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    primary = changes[0]
    source_ids = [str(change.get("change_id") or "") for change in changes if change.get("change_id")]
    title = str(spec.get("observation_title") or "").strip()
    summary = str(spec.get("analyst_summary") or "").strip()
    rationale = str(spec.get("rationale") or "").strip()
    if not title:
        title = f"Observation consolidée dans « {_display_subsection(primary)} »"
    if not summary:
        summary = _fallback_consolidated_summary(title, changes)

    observation = dict(primary)
    observation.update(
        {
            "change_id": _observation_id(primary, source_ids),
            "observation_type": "consolidated_intra_section",
            "consolidation_method": "llm",
            "consolidation_reason": rationale,
            "observation_title": title,
            "consolidated_change_count": len(changes),
            "source_change_ids": source_ids,
            "source_change_summaries": [
                {
                    "change_id": str(change.get("change_id") or ""),
                    "change_summary": str(change.get("change_summary") or "").strip(),
                    "pages_t1": list(change.get("pages_t1") or []),
                    "pages_t2": list(change.get("pages_t2") or []),
                    "impact_level": str((change.get("genai_triage") or {}).get("impact_level") or ""),
                    "nouvelle_idee": bool((change.get("genai_triage") or {}).get("nouvelle_idee", False)),
                }
                for change in changes
            ],
            "change_summary": summary,
            "source_text_t1": _join_texts(changes, "source_text_t1", "semantic_text_t1"),
            "source_text_t2": _join_texts(changes, "source_text_t2", "semantic_text_t2"),
            "semantic_text_t1": _join_texts(changes, "semantic_text_t1", "source_text_t1"),
            "semantic_text_t2": _join_texts(changes, "semantic_text_t2", "source_text_t2"),
            "pages_t1": _merged_pages(changes, "pages_t1"),
            "pages_t2": _merged_pages(changes, "pages_t2"),
            "diff_type": _combined_diff_type(changes),
            "genai_triage": _merged_triage(
                changes,
                title=title,
                summary=summary,
                rationale=rationale,
                spec=spec,
            ),
        }
    )
    observation["evidence_t1"] = _merged_evidence(observation, "t1")
    observation["evidence_t2"] = _merged_evidence(observation, "t2")
    return observation


def _observation_id(primary: dict[str, Any], source_ids: list[str]) -> str:
    section = _slug(str(primary.get("section_key") or "section"))
    subsection = _slug(_display_subsection(primary))[:60] or "section"
    digest = hashlib.sha256("|".join(source_ids).encode("utf-8")).hexdigest()[:10]
    return f"obs_{section}_{subsection}_{digest}"


def _display_subsection(change: dict[str, Any]) -> str:
    for key in (
        "current_hierarchy_path",
        "subsection_heading",
        "current_subsection_heading",
        "previous_subsection_heading",
    ):
        value = str(change.get(key) or "").strip()
        if value and value not in {"__intro__", "full"}:
            return value
    return str(change.get("section_key") or "").strip()


def _primary_page(change: dict[str, Any]) -> int | None:
    for key in ("pages_t2", "pages_t1"):
        pages = change.get(key)
        if isinstance(pages, list) and pages:
            try:
                return int(pages[0])
            except (TypeError, ValueError):
                continue
    for evidence_key in ("evidence_t2", "evidence_t1"):
        evidence = change.get(evidence_key)
        if isinstance(evidence, dict):
            pages = evidence.get("pages")
            if isinstance(pages, list) and pages:
                try:
                    return int(pages[0])
                except (TypeError, ValueError):
                    continue
    return None


def _observation_sort_key(change: dict[str, Any]) -> tuple[int, int, int, str]:
    triage = change.get("genai_triage") if isinstance(change.get("genai_triage"), dict) else {}
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    page = _primary_page(change)
    return (
        0 if bool(triage.get("is_relevant", False)) else 1,
        _IMPACT_ORDER.get(impact, 99),
        page if page is not None else 9999,
        str(change.get("change_id") or ""),
    )


def _fallback_consolidated_summary(title: str, changes: list[dict[str, Any]]) -> str:
    items: list[str] = []
    for change in changes:
        summary = str(change.get("change_summary") or "").strip()
        if summary and summary not in items:
            items.append(summary)
        if len(items) >= 4:
            break
    if not items:
        return f"{title}: plusieurs changements liés sont regroupés."
    return f"{title}: " + "; ".join(items)


def _join_texts(changes: list[dict[str, Any]], primary_key: str, fallback_key: str) -> str:
    parts: list[str] = []
    for index, change in enumerate(changes, start=1):
        text = str(change.get(primary_key) or change.get(fallback_key) or "").strip()
        if not text:
            continue
        summary = str(change.get("change_summary") or f"Changement {index}").strip()
        part = f"[{index}] {summary}\n{text}"
        if part not in parts:
            parts.append(part)
    return "\n\n".join(parts)


def _merged_pages(changes: list[dict[str, Any]], key: str) -> list[int]:
    pages: list[int] = []
    for change in changes:
        for page in change.get(key) or []:
            try:
                numeric = int(page)
            except (TypeError, ValueError):
                continue
            if numeric not in pages:
                pages.append(numeric)
    return sorted(pages)


def _merged_evidence(observation: dict[str, Any], side: str) -> dict[str, Any]:
    pages = observation.get(f"pages_{side}") or []
    text = str(observation.get(f"source_text_{side}") or "").strip()
    return {"pages": pages, "snippet": text[:800]}


def _combined_diff_type(changes: list[dict[str, Any]]) -> str:
    types = {str(change.get("diff_type") or "").lower() for change in changes}
    types.discard("")
    if len(types) == 1:
        return next(iter(types))
    return "modified"


def _merged_triage(
    changes: list[dict[str, Any]],
    *,
    title: str,
    summary: str,
    rationale: str,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = spec or {}
    triages = [
        change.get("genai_triage")
        for change in changes
        if isinstance(change.get("genai_triage"), dict)
    ]
    primary = dict(triages[0]) if triages else {}
    fallback_impact = _best_value(
        [str(triage.get("impact_level") or "MINEUR").upper() for triage in triages],
        _IMPACT_ORDER,
        default="MINEUR",
    )
    impact = _normalised_impact(spec.get("impact_level") or spec.get("risk_level"), fallback_impact)
    fallback_action = _best_value(
        [str(triage.get("action_requise") or "aucune").lower() for triage in triages],
        _ACTION_ORDER,
        default="aucune",
    )
    action = _normalised_action(spec.get("action_requise"), fallback_action)
    fallback_themes: list[str] = []
    for triage in triages:
        for theme in triage.get("themes_amf") or []:
            theme_str = str(theme)
            if theme_str and theme_str not in fallback_themes:
                fallback_themes.append(theme_str)
    themes = _unique_strings(spec.get("themes_amf") or spec.get("themes")) or fallback_themes

    fallback_nouvelle_idee = any(bool(triage.get("nouvelle_idee", False)) for triage in triages)
    nouvelle_idee = _optional_bool(spec.get("nouvelle_idee"), fallback=fallback_nouvelle_idee)
    is_relevant = (
        any(bool(triage.get("is_relevant", False)) for triage in triages)
        or nouvelle_idee
        or impact in {"MAJEUR", "MODERE"}
        or bool(themes)
    )
    justification = str(spec.get("nouvelle_idee_justification") or "").strip()
    if not justification:
        justification = _merged_justification(
            title,
            summary,
            rationale,
            nouvelle_idee=nouvelle_idee,
            impact=impact,
        )
    primary.update(
        {
            "is_relevant": is_relevant,
            "impact_level": impact,
            "risk_level": impact,
            "action_requise": action,
            "nouvelle_idee": nouvelle_idee,
            "themes_amf": themes,
            "explanation": summary,
            "nouvelle_idee_justification": justification,
            "change_segments": [],
        }
    )
    return primary


def _normalised_impact(value: Any, fallback: str) -> str:
    impact = str(value or "").strip().upper()
    return impact if impact in _IMPACT_ORDER else fallback


def _normalised_action(value: Any, fallback: str) -> str:
    action = str(value or "").strip().lower()
    return action if action in _ACTION_ORDER else fallback


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in items:
            items.append(text)
    return items


def _optional_bool(value: Any, *, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"oui", "true", "vrai", "yes", "1"}:
            return True
        if lowered in {"non", "false", "faux", "no", "0"}:
            return False
    return fallback


def _best_value(values: list[str], order: dict[str, int], *, default: str) -> str:
    best = default
    best_rank = order.get(default, 99)
    for value in values:
        rank = order.get(value, 99)
        if rank < best_rank:
            best = value
            best_rank = rank
    return best


def _merged_justification(
    title: str,
    summary: str,
    rationale: str,
    *,
    nouvelle_idee: bool,
    impact: str,
) -> str:
    prefix = "OUI" if nouvelle_idee else "NON"
    decision = "Oui" if nouvelle_idee else "Non"
    return (
        f"{prefix} — Nouvel élément à surveiller : {decision}.\n\n"
        f"Sujet détecté : {title}.\n\n"
        f"Ce qui change : {summary}\n\n"
        "Pertinence métier : Cette observation regroupe des changements que le LLM "
        "a jugés comme appartenant à la même observation métier. "
        f"{rationale}\n\n"
        f"Point de surveillance : Revoir l'observation consolidée au niveau {impact.lower()} "
        "et consulter les changements sources si un détail de paragraphe doit être validé séparément."
    )


def _slug(value: str) -> str:
    normalized = value.lower()
    replacements = {
        "à": "a",
        "â": "a",
        "ä": "a",
        "ç": "c",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "œ": "oe",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")
