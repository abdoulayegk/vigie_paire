"""Persistence des decisions analystes pour l'analyse textuelle."""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from vigilance.amf_taxonomy import (
    BusinessEquivalence,
    ChangeNature,
    DecisionStatus,
    EvidenceSufficiency,
    MaterialityConfidence,
    THEMES_AMF_PIPELINE_2,
)
from vigilance.comparison_io import _atomic_write_json
from vigilance.quarter_utils import get_payload_quarter_context
from vigilance.text_comparison.text_comparison_excel import generate_text_comparison_excel
from vigilance.ui_config import TEXT_COMPARISON_DIR

logger = logging.getLogger(__name__)

TEXT_REVIEW_STATUSES = {
    "approved",
    "corrected",
    "rejected",
    "skipped",
}
_TEXT_REVIEW_CORRECTION_FIELDS = {
    "materiality_level",
    "is_relevant",
    "nouvelle_idee",
    "themes_amf",
    "change_nature",
    "business_equivalence",
    "materiality_confidence",
    "evidence_sufficiency",
    "decision_status",
    "review_required",
    "materiality_rationale",
    "supporting_evidence",
    "counterarguments",
}
_CHANGE_NATURE_VALUES = set(get_args(ChangeNature))
_BUSINESS_EQUIVALENCE_VALUES = set(get_args(BusinessEquivalence))
_MATERIALITY_CONFIDENCE_VALUES = set(get_args(MaterialityConfidence))
_EVIDENCE_SUFFICIENCY_VALUES = set(get_args(EvidenceSufficiency))
_DECISION_STATUS_VALUES = set(get_args(DecisionStatus))
_THEME_VALUES = set(THEMES_AMF_PIPELINE_2)


def _normalize_structured_text_correction(
    correction: dict[str, Any] | None,
) -> dict[str, Any]:
    """Valide une correction explicite sans interpréter le commentaire libre."""
    if not correction:
        return {}
    unknown = set(correction) - _TEXT_REVIEW_CORRECTION_FIELDS
    if unknown:
        raise ValueError(
            "Champs de correction texte non supportés: "
            + ", ".join(sorted(unknown))
        )

    normalized = copy.deepcopy(correction)
    materiality = str(normalized.get("materiality_level") or "").strip().upper()
    if materiality not in {"MAJEUR", "MODERE", "MINEUR"}:
        raise ValueError(
            "Une correction structurée exige materiality_level="
            "MAJEUR, MODERE ou MINEUR."
        )
    normalized["materiality_level"] = materiality

    change_nature = normalized.get("change_nature")
    if isinstance(change_nature, str):
        change_nature = [change_nature]
    if not isinstance(change_nature, list) or not any(
        str(value or "").strip() for value in change_nature
    ):
        raise ValueError(
            "Une correction structurée exige au moins une nature de changement."
        )
    normalized["change_nature"] = [
        str(value).strip().upper()
        for value in change_nature
        if str(value or "").strip()
    ][:3]
    invalid_natures = set(normalized["change_nature"]) - _CHANGE_NATURE_VALUES
    if invalid_natures:
        raise ValueError(
            "Nature(s) de changement non supportée(s): "
            + ", ".join(sorted(invalid_natures))
        )

    rationale = str(normalized.get("materiality_rationale") or "").strip()
    if not rationale:
        raise ValueError(
            "Une correction structurée exige materiality_rationale."
        )
    normalized["materiality_rationale"] = rationale

    required_fields = {
        "is_relevant",
        "nouvelle_idee",
        "themes_amf",
        "business_equivalence",
        "materiality_confidence",
        "evidence_sufficiency",
        "decision_status",
        "review_required",
    }
    missing_fields = sorted(required_fields - set(normalized))
    if missing_fields:
        raise ValueError(
            "Correction structurée incomplète; champs requis: "
            + ", ".join(missing_fields)
        )

    enum_fields = {
        "business_equivalence": _BUSINESS_EQUIVALENCE_VALUES,
        "materiality_confidence": _MATERIALITY_CONFIDENCE_VALUES,
        "evidence_sufficiency": _EVIDENCE_SUFFICIENCY_VALUES,
        "decision_status": _DECISION_STATUS_VALUES,
    }
    for field_name, allowed_values in enum_fields.items():
        if field_name not in normalized:
            continue
        enum_value = str(normalized.get(field_name) or "").strip().upper()
        if enum_value not in allowed_values:
            raise ValueError(
                f"{field_name} contient une valeur non supportée: "
                f"{enum_value!r}."
            )
        normalized[field_name] = enum_value

    for field_name in (
        "is_relevant",
        "nouvelle_idee",
        "review_required",
    ):
        if field_name in normalized and not isinstance(
            normalized[field_name],
            bool,
        ):
            raise ValueError(f"{field_name} doit être un booléen.")

    if (
        materiality in {"MAJEUR", "MODERE"}
        and normalized.get("is_relevant") is False
    ):
        raise ValueError(
            "Une correction MAJEUR ou MODERE ne peut pas être non pertinente."
        )
    if (
        normalized.get("is_relevant") is False
        and normalized.get("nouvelle_idee") is True
    ):
        raise ValueError(
            "Une correction non pertinente exige nouvelle_idee=False."
        )

    for field_name in ("themes_amf", "supporting_evidence", "counterarguments"):
        value = normalized.get(field_name)
        if value is None:
            normalized[field_name] = []
        elif isinstance(value, str):
            normalized[field_name] = [value.strip()] if value.strip() else []
        elif isinstance(value, list):
            normalized[field_name] = [
                str(item).strip()
                for item in value
                if str(item or "").strip()
            ]
        else:
            raise ValueError(f"{field_name} doit être une liste de textes.")
    normalized["themes_amf"] = list(
        dict.fromkeys(
            str(theme).strip().upper()
            for theme in normalized["themes_amf"]
            if str(theme or "").strip()
        )
    )[:2]
    normalized["themes_amf"] = [
        theme
        for theme in normalized["themes_amf"]
        if theme in _THEME_VALUES
    ]
    if normalized.get("is_relevant") is False:
        normalized["themes_amf"] = []
    if (
        materiality in {"MAJEUR", "MODERE"}
        and normalized["business_equivalence"] == "CONFIRMEE"
    ):
        raise ValueError(
            "Une correction MODERE ou MAJEUR est incompatible avec une "
            "équivalence métier confirmée."
        )
    if (
        normalized["decision_status"] == "CONFIRME"
        and (
            normalized["evidence_sufficiency"] != "SUFFISANTE"
            or normalized["materiality_confidence"]
            not in {"ELEVEE", "MOYENNE"}
        )
    ):
        raise ValueError(
            "Une correction confirmée exige une preuve suffisante et une "
            "confiance élevée ou moyenne."
        )
    if (
        normalized["decision_status"] == "CONFIRME"
        and normalized["review_required"]
    ):
        raise ValueError(
            "Une correction confirmée ne peut pas exiger une revue."
        )
    if not normalized["supporting_evidence"]:
        raise ValueError(
            "Une correction structurée exige au moins une preuve analyste."
        )
    uncertain_minor = (
        materiality == "MINEUR"
        and normalized["business_equivalence"] != "CONFIRMEE"
    )
    requires_review = (
        normalized["decision_status"] in {"A_CONFIRMER", "PROVISOIRE"}
        or normalized["evidence_sufficiency"] != "SUFFISANTE"
        or normalized["materiality_confidence"] == "FAIBLE"
        or uncertain_minor
    )
    if requires_review and not normalized["review_required"]:
        raise ValueError(
            "Cette correction incertaine exige review_required=True."
        )
    return normalized


def _apply_correction_to_effective_triage(
    triage: dict[str, Any],
    correction: dict[str, Any],
) -> dict[str, Any]:
    """Applique la décision humaine au verdict affiché, filtré et exporté."""
    effective = copy.deepcopy(triage)
    level = correction["materiality_level"]
    is_relevant = bool(correction.get("is_relevant", True))
    effective.update(
        {
            key: copy.deepcopy(value)
            for key, value in correction.items()
            if key != "materiality_rationale"
        }
    )
    effective.update(
        {
            "impact_level": level,
            "materiality_level": level,
            "is_relevant": is_relevant,
            "nouvelle_idee": (
                bool(correction.get("nouvelle_idee"))
                if is_relevant
                else False
            ),
            "themes_amf": (
                list(correction.get("themes_amf") or [])
                if is_relevant
                else []
            ),
            "action_requise": (
                {
                    "MAJEUR": "revue_prioritaire",
                    "MODERE": "investigation",
                    "MINEUR": "information",
                }[level]
                if is_relevant
                else "aucune"
            ),
            "exclusion_reason": (
                None
                if is_relevant
                else (
                    effective.get("exclusion_reason")
                    or "non_pertinent_autre"
                )
            ),
            "materiality_rationale": correction[
                "materiality_rationale"
            ],
            "materiality_decision_basis": "analyst_correction",
            "source": "analyst_correction",
            "analyst_correction_applied": True,
        }
    )
    from vigilance.text_analysis.triage import _derive_legacy_fields

    effective.update(_derive_legacy_fields(effective))
    return effective


def _period_from_payload(payload: dict[str, Any], role: str) -> str:
    """Retourne le dossier periode ``YYYY_tN`` pour le role demande."""
    raw_key = "quarter_current" if role == "current" else "quarter_previous"
    raw = str(payload.get(raw_key) or "").lower().strip()
    if raw and raw[:4].isdigit() and "_t" in raw:
        return raw

    ctx = get_payload_quarter_context(payload)
    side = ctx.get(role) or {}
    year = side.get("year")
    code = str(side.get("code") or "").lower().strip()
    if year and code:
        return f"{int(year)}_{code}"
    return raw


def text_comparison_path_from_payload(
    payload: dict[str, Any],
    root_dir: Path | None = None,
) -> Path | None:
    """Resout le chemin ``text_comparison.json`` depuis le payload texte."""
    bank = str(payload.get("bank_code") or payload.get("bank") or "").lower().strip()
    current = _period_from_payload(payload, "current")
    previous = _period_from_payload(payload, "previous")
    if not bank or not current or not previous:
        return None
    root = Path(root_dir) if root_dir else TEXT_COMPARISON_DIR
    return root / bank / f"{current}_vs_{previous}" / "text_comparison.json"


_MATERIALITY_RANK = {"MINEUR": 0, "MODERE": 1, "MAJEUR": 2}


def is_final_direct_triage(triage: dict[str, Any]) -> bool:
    """Indique si une décision directe est finale, cohérente et apprenable."""
    level = str(triage.get("materiality_level") or "").upper()
    equivalence = str(
        triage.get("business_equivalence") or ""
    ).upper()
    confidence = str(
        triage.get("materiality_confidence") or ""
    ).upper()
    evidence = str(triage.get("evidence_sufficiency") or "").upper()
    is_relevant = triage.get("is_relevant")
    themes = triage.get("themes_amf")
    natures = triage.get("change_nature")
    if (
        level not in _MATERIALITY_RANK
        or str(triage.get("decision_status") or "").upper() != "CONFIRME"
        or bool(triage.get("review_required", False))
        or confidence not in {"ELEVEE", "MOYENNE"}
        or evidence != "SUFFISANTE"
        or not bool(triage.get("supporting_evidence"))
        or not isinstance(natures, list)
        or not natures
        or not set(natures) <= _CHANGE_NATURE_VALUES
        or not isinstance(is_relevant, bool)
        or not isinstance(themes, list)
    ):
        return False
    if is_relevant:
        if not 1 <= len(themes) <= 2 or not set(themes) <= _THEME_VALUES:
            return False
    elif (
        level != "MINEUR"
        or themes
        or bool(triage.get("nouvelle_idee", False))
    ):
        return False
    if level == "MINEUR":
        return equivalence == "CONFIRMEE"
    return equivalence in {"PROBABLE", "NON_DEMONTREE", "REFUTEE"}


def _target_change(
    text_data: dict[str, Any],
    change_id: str,
) -> dict[str, Any] | None:
    """Retrouve en priorité la copie canonique du changement."""
    for section in text_data.get("section_comparisons") or []:
        for bucket in ("all_block_comparisons", "block_comparisons"):
            for change in section.get(bucket) or []:
                if (
                    isinstance(change, dict)
                    and str(change.get("change_id") or "") == change_id
                ):
                    return change
    return None


def _refresh_consolidated_after_correction(
    section_comparisons: list[dict[str, Any]],
) -> None:
    """Rafraîchit le minimum consolidé et signale le jugement collectif périmé."""
    for section in section_comparisons:
        groups: dict[str, list[dict[str, Any]]] = {}
        for change in section.get("all_block_comparisons") or []:
            triage = change.get("genai_triage") or {}
            group_id = str(triage.get("triage_group_id") or "")
            if group_id:
                groups.setdefault(group_id, []).append(change)
        for members in groups.values():
            candidate_levels: list[str] = []
            corrected = False
            for member in members:
                triage = member.get("genai_triage") or {}
                level = str(
                    triage.get("materiality_level")
                    or triage.get("impact_level")
                    or ""
                ).upper()
                if level in _MATERIALITY_RANK:
                    candidate_levels.append(level)
                prior_consolidated = str(
                    triage.get("consolidated_materiality_level") or ""
                ).upper()
                if prior_consolidated in _MATERIALITY_RANK:
                    candidate_levels.append(prior_consolidated)
                assessment = triage.get(
                    "consolidated_materiality_assessment"
                )
                if isinstance(assessment, dict):
                    assessed = str(
                        assessment.get("materiality_level") or ""
                    ).upper()
                    if assessed in _MATERIALITY_RANK:
                        candidate_levels.append(assessed)
                corrected = corrected or bool(
                    triage.get("analyst_correction_applied")
                )
            if not candidate_levels:
                continue
            consolidated_level = max(
                candidate_levels,
                key=lambda value: _MATERIALITY_RANK[value],
            )
            consolidated_relevant = (
                any(
                    bool(
                        (member.get("genai_triage") or {}).get(
                            "is_relevant"
                        )
                    )
                    for member in members
                )
                or consolidated_level in {"MODERE", "MAJEUR"}
            )
            for member in members:
                triage = member.get("genai_triage") or {}
                triage["consolidated_materiality_level"] = (
                    consolidated_level
                )
                triage["consolidated_relevant"] = consolidated_relevant
                if corrected:
                    triage.update(
                        {
                            "consolidated_decision_status": "A_CONFIRMER",
                            "consolidated_review_required": True,
                            "consolidated_assessment_stale_after_analyst_correction": True,
                        }
                    )


def _rebuild_text_review_derivatives(text_data: dict[str, Any]) -> None:
    """Reconstruit les périmètres et résumés après un verdict humain."""
    from vigilance.text_analysis.summary import (
        _build_global_summary,
        _is_non_cosmetic_change,
        _retained_change_sort_key,
    )

    sections = text_data.get("section_comparisons") or []
    _refresh_consolidated_after_correction(sections)
    for section in sections:
        all_changes = [
            change
            for change in section.get("all_block_comparisons") or []
            if isinstance(change, dict)
        ]
        all_changes.sort(key=_retained_change_sort_key)
        section["all_block_comparisons"] = all_changes
        retained = [
            change
            for change in all_changes
            if _is_non_cosmetic_change(
                change.get("genai_triage") or {}
            )
        ]
        retained.sort(key=_retained_change_sort_key)
        section["block_comparisons"] = retained
        summary = dict(section.get("summary") or {})
        summary["retained_changes"] = len(retained)
        summary["all_changes"] = len(all_changes)
        section["summary"] = summary

    bank_code = str(text_data.get("bank_code") or "")
    text_data["global_summary"] = _build_global_summary(
        sections,
        bank_code=bank_code,
    )
    text_data["all_changes_summary"] = _build_global_summary(
        [
            {
                "section_key": section.get("section_key"),
                "block_comparisons": (
                    section.get("all_block_comparisons") or []
                ),
            }
            for section in sections
        ],
        bank_code=bank_code,
    )


def apply_text_review_decision(
    text_data: dict[str, Any],
    *,
    change_id: str,
    status: str,
    comment: str = "",
    reviewer: str = "analyste",
    structured_correction: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Retourne une copie du payload avec la decision appliquee au changement."""
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in TEXT_REVIEW_STATUSES:
        raise ValueError(f"Statut texte non supporte: {status!r}")

    updated = copy.deepcopy(text_data)
    target_id = str(change_id or "").strip()
    if not target_id:
        return updated, False

    normalized_correction = _normalize_structured_text_correction(
        structured_correction
    )
    if normalized_correction and normalized_status == "rejected":
        normalized_status = "corrected"
    if normalized_status == "corrected" and not normalized_correction:
        raise ValueError(
            "Le statut corrected exige une correction structurée."
        )
    target_change = _target_change(updated, target_id)
    target_review_status = str(
        ((target_change or {}).get("_analyst_review") or {}).get("status")
        or ""
    ).strip().lower()
    if (
        normalized_status == "approved"
        and target_review_status == "corrected"
    ):
        raise ValueError(
            "Une correction analyste doit rester enregistrée comme correction; "
            "utilisez une nouvelle correction structurée pour la modifier ou la "
            "confirmer."
        )
    if (
        normalized_status == "approved"
        and target_change is not None
        and not is_final_direct_triage(
            target_change.get("genai_triage") or {}
        )
    ):
        raise ValueError(
            "Une décision à confirmer doit être corrigée de façon "
            "structurée avant d'être validée."
        )
    decision = {
        "status": normalized_status,
        "comment": str(comment or "").strip(),
        "review_user": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "decision_scope": "materiality",
        "schema_version": "analyst_materiality_review_v1",
        "workflow_status": (
            "pending"
            if normalized_correction
            and normalized_correction.get("review_required")
            else (
                "deferred"
                if normalized_status == "skipped"
                else "completed"
            )
        ),
    }
    if normalized_status == "rejected" and not normalized_correction:
        decision["nouvelle_idee_override"] = False
    if normalized_correction:
        decision["structured_correction"] = normalized_correction
        decision.update(
            {
                f"corrected_{field_name}": value
                for field_name, value in normalized_correction.items()
            }
        )

    found = False
    for section in updated.get("section_comparisons") or []:
        for bucket in ("all_block_comparisons", "block_comparisons"):
            for change in section.get(bucket) or []:
                if isinstance(change, dict) and str(change.get("change_id") or "") == target_id:
                    change_decision = dict(decision)
                    previous_review = change.get("_analyst_review")
                    decision_history: list[dict[str, Any]] = []
                    if isinstance(previous_review, dict):
                        decision_history.extend(
                            copy.deepcopy(
                                previous_review.get("decision_history")
                                or []
                            )
                        )
                        previous_snapshot = {
                            key: copy.deepcopy(value)
                            for key, value in previous_review.items()
                            if key != "decision_history"
                        }
                        if previous_snapshot:
                            decision_history.append(previous_snapshot)
                    if decision_history:
                        change_decision["decision_history"] = (
                            decision_history
                        )
                    if normalized_correction:
                        previous_review_mapping = (
                            previous_review
                            if isinstance(previous_review, dict)
                            else {}
                        )
                        original_triage = copy.deepcopy(
                            previous_review_mapping.get(
                                "original_genai_triage"
                            )
                            or change.get("genai_triage")
                            or {}
                        )
                        change_decision["original_genai_triage"] = (
                            original_triage
                        )
                        change["genai_triage"] = (
                            _apply_correction_to_effective_triage(
                                original_triage,
                                normalized_correction,
                            )
                        )
                    change["_analyst_review"] = change_decision
                    found = True

    if found:
        _rebuild_text_review_derivatives(updated)
    return updated, found


def write_text_review_to_disk(
    text_data: dict[str, Any],
    *,
    regenerate_excel: bool = True,
) -> bool:
    """Ecrit ``text_comparison.json`` et, si demande, regenere l'Excel."""
    path = text_comparison_path_from_payload(text_data)
    if path is None:
        logger.warning("[text_review] chemin text_comparison introuvable")
        return False
    if not path.exists():
        logger.warning("[text_review] text_comparison.json introuvable: %s", path)
        return False

    try:
        _atomic_write_json(path, text_data)
        if regenerate_excel:
            generate_text_comparison_excel(text_data, path.with_suffix(".xlsx"))
    except Exception:
        logger.exception("[text_review] echec writeback texte: %s", path)
        return False
    return True
