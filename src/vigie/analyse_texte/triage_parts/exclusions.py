"""Regles deterministes d'exclusion d'un changement, avant tout appel au modele.

Extrait de ``triage.py`` sans modification.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from vigie.analyse_texte.constants import _NUMERIC_TOKEN_RE, _REGULATORY_REF_RE

from .constants import (
    _BANK_NOISE_SEQUENCE_THRESHOLD,
    _BANK_OPERATION_RE,
    _CALENDAR_SUBJECT_RE,
    _CALENDAR_UPDATE_RE,
    _COSMETIC_SEQUENCE_THRESHOLD,
    _GOVERNANCE_SIGNAL_RE,
    _ISOLATED_DATE_RE,
    _METHODOLOGY_SIGNAL_RE,
    _NEW_REGULATORY_SIGNAL_RE,
    _VOLATILE_TOKEN_RE,
    _WHITESPACE_RE,
)
from .themes import _normalize_for_cosmetic


def _is_semantic_text_move(change: dict[str, Any]) -> bool:
    return str(change.get("alignment_decision") or "").strip().lower() == "moved_text"


def _sequence_ratio(left: str, right: str) -> float:
    left_norm = _normalize_for_cosmetic(left)
    right_norm = _normalize_for_cosmetic(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def _numeric_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _NUMERIC_TOKEN_RE.finditer(str(text or ""))}


def _regulatory_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _REGULATORY_REF_RE.finditer(str(text or ""))}


def _is_isolated_date_change(text_t1: str, text_t2: str) -> bool:
    """True when the only material difference looks like an isolated date update."""
    without_dates_t1 = _ISOLATED_DATE_RE.sub(" ", text_t1)
    without_dates_t2 = _ISOLATED_DATE_RE.sub(" ", text_t2)
    if _sequence_ratio(without_dates_t1, without_dates_t2) < 0.98:
        return False
    dates_t1 = {match.group(0).lower() for match in _ISOLATED_DATE_RE.finditer(text_t1)}
    dates_t2 = {match.group(0).lower() for match in _ISOLATED_DATE_RE.finditer(text_t2)}
    return bool(dates_t1 or dates_t2) and dates_t1 != dates_t2


def _mask_volatile_tokens(text: str) -> str:
    """Retire dates, trimestres et montants pour comparer le fond textuel."""
    masked = _VOLATILE_TOKEN_RE.sub(" ", str(text or ""))
    return _WHITESPACE_RE.sub(" ", masked).strip()


def _combined_change_text(change: dict[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in (
            change.get("change_summary"),
            change.get("source_text_t1"),
            change.get("source_text_t2"),
            change.get("semantic_text_t1"),
            change.get("semantic_text_t2"),
        )
        if str(part or "").strip()
    )


def _has_methodology_signal(text_t1: str, text_t2: str) -> bool:
    markers_t1 = {m.group(0).lower() for m in _METHODOLOGY_SIGNAL_RE.finditer(text_t1)}
    markers_t2 = {m.group(0).lower() for m in _METHODOLOGY_SIGNAL_RE.finditer(text_t2)}
    return bool(markers_t1.symmetric_difference(markers_t2))


def _has_new_regulatory_substance(text_t1: str, text_t2: str) -> bool:
    """Conservé pertinent seulement si une mention réglementaire NOUVELLE apparaît en T2."""
    signals_t1 = {m.group(0).lower() for m in _NEW_REGULATORY_SIGNAL_RE.finditer(text_t1)}
    signals_t2 = {m.group(0).lower() for m in _NEW_REGULATORY_SIGNAL_RE.finditer(text_t2)}
    # Disappearance of a Bâle III / BSIF mention alone is reformulation, not substance.
    return bool(signals_t2 - signals_t1)


def _shares_calendar_subject(text_t1: str, text_t2: str) -> bool:
    subjects_t1 = {m.group(0).lower() for m in _CALENDAR_SUBJECT_RE.finditer(text_t1)}
    subjects_t2 = {m.group(0).lower() for m in _CALENDAR_SUBJECT_RE.finditer(text_t2)}
    return bool(subjects_t1 & subjects_t2)


def _has_calendar_reschedule_context(
    text_t1: str,
    text_t2: str,
    combined: str,
) -> bool:
    """True when the delta is a deferred-application update of a known requirement."""
    if not _CALENDAR_UPDATE_RE.search(combined):
        return False
    has_anchor = bool(
        _CALENDAR_SUBJECT_RE.search(combined)
        or re.search(
            r"\b(?:bsif|plancher\s+de\s+fonds|coefficient\s+de\s+plancher)\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    if not has_anchor:
        return False
    if _shares_calendar_subject(text_t1, text_t2):
        return True
    # Subject may appear only in T1 while T2 only updates the deferral wording.
    if _CALENDAR_SUBJECT_RE.search(text_t1) and _CALENDAR_UPDATE_RE.search(text_t2):
        return True
    return bool(
        re.search(
            r"\b(?:bsif|plancher|coefficient)\b",
            combined,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(?:report|retard|jusqu['’]à\s+nouvel\s+ordre)\b",
            combined,
            flags=re.IGNORECASE,
        )
    )


def _is_pure_new_regulatory_disclosure(change: dict[str, Any]) -> bool:
    """First mention of a shared regulatory requirement, without a bank deal."""
    diff_type = str(change.get("diff_type") or "").strip().lower()
    text_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    combined = _combined_change_text(change)
    if diff_type != "added" or not text_t2.strip():
        return False
    if _BANK_OPERATION_RE.search(combined):
        return False
    return bool(_NEW_REGULATORY_SIGNAL_RE.search(text_t2))


def _deterministic_bank_specific_exclusion(change: dict[str, Any]) -> str | None:
    """Exclut dates/montants/opérations internes sans fond réglementaire inter-pairs."""
    text_t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "")
    text_t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "")
    combined = _combined_change_text(change)
    diff_type = str(change.get("diff_type") or "").strip().lower()

    # Priority 1: bank-specific operations (acquisition, CWB, buyback, issuance).
    if _BANK_OPERATION_RE.search(combined):
        if _is_pure_new_regulatory_disclosure(change):
            pass
        else:
            return "operation_interne_banque"

    # Keep true methodology changes for analyst review — but only when no
    # bank operation already matched above.
    if text_t1.strip() and text_t2.strip():
        if _has_methodology_signal(text_t1, text_t2) and not _BANK_OPERATION_RE.search(combined):
            return None
        if _has_new_regulatory_substance(text_t1, text_t2):
            masked_ratio = _sequence_ratio(
                _mask_volatile_tokens(text_t1),
                _mask_volatile_tokens(text_t2),
            )
            if masked_ratio < _BANK_NOISE_SEQUENCE_THRESHOLD and not _BANK_OPERATION_RE.search(combined):
                return None

    if diff_type not in {"modified", "unchanged"}:
        return None
    if not text_t1.strip() or not text_t2.strip():
        return None

    masked_t1 = _mask_volatile_tokens(text_t1)
    masked_t2 = _mask_volatile_tokens(text_t2)
    if not masked_t1 or not masked_t2:
        return None

    masked_ratio = _sequence_ratio(masked_t1, masked_t2)
    numbers_differ = _numeric_tokens(text_t1) != _numeric_tokens(text_t2)
    dates_differ = _is_isolated_date_change(text_t1, text_t2) or (
        {m.group(0).lower() for m in _ISOLATED_DATE_RE.finditer(text_t1)}
        != {m.group(0).lower() for m in _ISOLATED_DATE_RE.finditer(text_t2)}
    )
    volatile_differ = numbers_differ or dates_differ

    # Calendar updates of a known requirement (e.g. BSIF floor deferral).
    if volatile_differ and _has_calendar_reschedule_context(text_t1, text_t2, combined):
        if not _has_methodology_signal(text_t1, text_t2):
            return "mise_a_jour_calendrier"

    if masked_ratio >= _BANK_NOISE_SEQUENCE_THRESHOLD and volatile_differ:
        if _CALENDAR_UPDATE_RE.search(combined) and dates_differ:
            return "mise_a_jour_calendrier"
        if dates_differ and not numbers_differ:
            return "mise_a_jour_calendrier"
        if numbers_differ:
            return "variation_numerique_propre_banque"
        return "variation_numerique_propre_banque"

    # Fallback: calendar reschedule when reformulation lowers similarity below 0.92.
    if (
        volatile_differ
        and dates_differ
        and _has_calendar_reschedule_context(text_t1, text_t2, combined)
        and not _has_methodology_signal(text_t1, text_t2)
    ):
        return "mise_a_jour_calendrier"
    return None


def _deterministic_cosmetic_exclusion(change: dict[str, Any]) -> str | None:
    """Return an exclusion reason when the change is manifestly cosmetic."""
    if _is_semantic_text_move(change):
        return "deplacement_texte"
    if str(change.get("alignment_type") or "").strip().lower() in {
        "global_reconciled_residual",
    } and str(change.get("alignment_decision") or "").strip().lower() in {
        "moved_text",
        "same_disclosure",
    }:
        # Residual after a confirmed move/resegmentation is already handled upstream.
        pass

    diff_type = str(change.get("diff_type") or "").strip().lower()
    if diff_type not in {"modified", "unchanged"}:
        return None

    text_t1 = str(change.get("source_text_t1") or "")
    text_t2 = str(change.get("source_text_t2") or "")
    if not text_t1.strip() or not text_t2.strip():
        return None

    if _numeric_tokens(text_t1) != _numeric_tokens(text_t2):
        return None
    if _regulatory_tokens(text_t1) != _regulatory_tokens(text_t2):
        return None

    compact_t1 = re.sub(r"[^\w]+", "", _normalize_for_cosmetic(text_t1), flags=re.UNICODE)
    compact_t2 = re.sub(r"[^\w]+", "", _normalize_for_cosmetic(text_t2), flags=re.UNICODE)
    if compact_t1 == compact_t2 and text_t1 != text_t2:
        return "formatage_visuel"

    # Une modification très courte peut changer le nom d'un comité, un mandat,
    # une responsabilité ou une ligne de défense. Ces cas doivent atteindre le
    # triage métier plutôt que d'être écartés selon leur seule similarité.
    if _GOVERNANCE_SIGNAL_RE.search(f"{text_t1} {text_t2}"):
        return None

    similarity = _sequence_ratio(text_t1, text_t2)
    if similarity >= _COSMETIC_SEQUENCE_THRESHOLD:
        return "reformulation_mineure"
    if _is_isolated_date_change(text_t1, text_t2):
        return "reformulation_mineure"
    return None
