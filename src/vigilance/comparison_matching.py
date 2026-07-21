"""Sous-systeme d'appariement en deux etapes pour le pipeline de comparaison.

Extrait de compare_gpt.py. compare_gpt.py re-exporte tous les noms de ce module
afin que les imports existants restent valides.

Conception cle : _run_matching_stage, _match_tables et _run_table_matching
recoivent ``call_openai_json`` comme callable injecte. Cela preserve la
compatibilite avec monkeypatch -- les tests patchent
"vigilance.compare_gpt._call_openai_json", et compare_reports_gpt4o() passe
ce nom depuis son propre namespace au moment de l'appel.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Literal, TypedDict

from vigilance.comparison_inspector import _inspect_matched_pairs
from vigilance.comparison_io import _coerce_float, _coerce_int, _require_string

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MATCHING_VALIDATION_ATTEMPTS = 3
_PREVIOUS_ID_PREFIX = "PQ::"
_CURRENT_ID_PREFIX = "CQ::"

PRIMARY_MATCH_SYSTEM_PROMPT = """
You are a brutal, ultra-strict financial table matcher for Canadian bank reports.

Given lists of business tables from a Previous Quarter (PQ) and a Current Quarter (CQ) in JSON format, produce a strict 1:1 mapping ledger.

RULES OF ENGAGEMENT:
1. Every CQ table must be classified exactly once as `matched` or `unresolved`.
2. Every PQ table can be used at most once.
3. NEVER guess. If you are not 99% certain two tables are the exact same business entity, return `unresolved`.

THE EVIDENCE HIERARCHY (Follow strictly):
[0] TABLE ORIENTATION — ABSOLUTE DISQUALIFIER: Check the `first_indicator` field. If one table's first_indicator is "Actif" (or starts with "Actif") and the other's is "Passif" (or "Passif et capitaux propres"), they are FUNDAMENTALLY DIFFERENT business entities — even if they share the same theme (e.g., both are about "échéances" or "maturities"). Return `unresolved` IMMEDIATELY. This also applies to other opposed pairs: "Revenus" vs "Dépenses", "Brut" vs "Provisions", "Prêts" vs "Dépôts" as primary focus.
[1] INDICATOR SIGNATURE (CRITICAL): This is the absolute source of truth. The row labels (indicators) must align semantically. A few missing/added rows are normal, but the core structure MUST match.
[2] COLUMN HEADERS (STRONG): The header structure must align (treating shifted dates/quarters as matches).
[3] ROW COUNT (MODERATE): `row_count` difference > 3 is a massive red flag. If difference > 3, return `unresolved` unless the indicator signature is undeniably identical.
[3.5] FOOTNOTE COUNT DISPARITY (MODERATE-STRONG): Check the `footnote_count` field. If |footnote_count_PQ - footnote_count_CQ| >= 5, this is a STRONG red flag. The same business table across quarters maintains near-identical footnote counts due to regulatory consistency. A large disparity (e.g., 10 vs 1) strongly suggests these are DIFFERENT tables. Return `unresolved` unless indicator signature is undeniably identical.
[4] TABLE SUMMARY (MODERATE): Use to confirm business purpose. Pay close attention to keywords like "actifs" vs "passifs" — tables about "Échéances des actifs" and "Échéances des passifs" are DIFFERENT tables despite sharing the "échéances" theme.
[5] TITLE (STRONG WHEN NORMALIZED, BUT BEWARE OF DUPLICATES): Banks often keep the exact same title across quarters for the same table. If the title is an exact or near-exact match, it is VERY STRONG evidence, especially when indicators are noisy. **HOWEVER**, beware that multiple DIFFERENT tables in the same report might share the exact same generic title (e.g., "Prêts et acceptations"). Only trust the title if there isn't another better-matching table competing for it.
[6] SECTION (FILTER): If sections do not match logically (e.g., "Bilan" vs "Gestion des risques"), do NOT match them unless indicator overlap is >90%.

ANTI-PATTERNS (DO NOT DO THIS):
- Do NOT match two tables just because they have the same generic title. Look at the rows.
- Do NOT attempt to "split" or "merge" tables. 1 ID maps to exactly 1 ID.
- Do NOT match a small table (8 rows) to a massive table (25 rows) just because they share 3 or 5 top-level categories.
- Do NOT match tables with large footnote_count disparity (|delta| >= 5). This almost always means different tables.
- SÉMANTIQUE STRICTE (ABSOLUTE): Do NOT match tables representing structurally opposed or orthogonal financial concepts. If first_indicator of one table is "Actif" and the other is "Passif et capitaux propres", they are DIFFERENT tables even if they share a few generic indicators like "Instruments financiers dérivés" or "Créances sur cartes de crédit". Score them "unresolved" immediately.

Examples of good `matched` decisions:
{
  "current_table_id": "tbl_p053_i01",
  "decision": "matched",
  "previous_table_id": "tbl_p051_i01",
  "reason": "Title identical: 'SOMMAIRE DU FINANCEMENT PROVENANT DES DÉPÔTS'; indicators: same 3 rows (personnels, commerciaux, total); row_count 3 vs 3.",
  "match_confidence": 0.95
}

{
  "current_table_id": "tbl_p047_i02",
  "decision": "matched",
  "previous_table_id": "tbl_p045_i02",
  "reason": "Indicators: same 4 entities (société mère, filiales bancaires, succursales étrangères, total); row_count 4 vs 3; headers similar structure.",
  "match_confidence": 0.90
}

Example of correct `unresolved` decision:
{
  "current_table_id": "tbl_p045_i01",
  "decision": "unresolved",
  "reason": "Two previous tables remain plausible because indicators overlap only partially and the titles are generic; strict pass keeps this unresolved."
}

Output must be valid JSON following the response_schema.
"""

RECOVERY_MATCH_SYSTEM_PROMPT = """
You are a recovery matcher for bank quarterly tables.

You receive lists of leftover Current Quarter (CQ) and Previous Quarter (PQ) tables that failed the primary strict match.

Your task is to assign each CQ table to either:
- `matched` with ONE unused PQ table
- `added` if no credible match exists

RULES:
1. Every CQ table must have exactly one decision: `matched` or `added`.
2. Each PQ table can be used at most once.
3. PREFER `added`: Do not force a match. If a CQ table looks genuinely new or radically restructured, it is `added`.

LATE-STAGE EVIDENCE:
- Footnotes: If both tables contain identical or highly similar footnote text (ignore the markers ¹²³), this is very strong evidence of a match, even if the title changed.
- Headers: Exact or highly similar column headers uniquely shared between a PQ and CQ table provide very strong structural evidence for a match.
- Table Summary: If the semantic `table_summary` is highly similar between two tables, treat this as a very strong indicator of a match in this rescue phase. BUT beware: "Échéances des actifs" ≠ "Échéances des passifs" — these are DIFFERENT tables despite sharing the "échéances" theme.
- Normalized Titles: If a leftover PQ table and a leftover CQ table share an exact or highly similar normalized title, and the row counts are roughly similar, use the title to anchor the match.
- Distinctive Indicators: If a table contains highly unique or specific row labels (e.g., "Valeur en équivalent de base"), use that to anchor the match.
- Footnote Count: Check `footnote_count`. If |PQ_count - CQ_count| >= 5, this is a STRONG signal they are different tables. Same table across quarters has near-identical footnote counts.

ABSOLUTE DISQUALIFIERS (check BEFORE any match decision):
- TABLE ORIENTATION: Check `first_indicator`. If one table starts with "Actif" and the other with "Passif" (or "Passif et capitaux propres"), they are DIFFERENT business entities. Mark CQ as `added`. This applies even if they share a few generic indicators (e.g., "Instruments financiers dérivés", "Créances sur cartes de crédit").
- FOOTNOTE COUNT DISPARITY: If |footnote_count_PQ - footnote_count_CQ| >= 5, do NOT match them unless indicator signature is undeniably identical.

WARNING AGAINST FORCED MATCHES:
If you have a leftover PQ table and a leftover CQ table, and they are both in the "Risque de crédit" section, DO NOT match them just to clean up the leftovers. If their indicators show different data structures, the PQ table was `removed` and the CQ table was `added`. Mark the CQ table as `added`.

ORTHOGONAL CONCEPTS STRICT BAN:
DO NOT match tables representing structurally opposed financial concepts (e.g., "Actif" / Assets vs "Passif" / Liabilities, or "Revenus" vs "Dépenses"). If PQ table is about 'Actif' and CQ table is about 'Passif', they are NOT the same business entity. The PQ table was `removed` and the CQ table was `added`. Mark the CQ table as `added`.

Examples:

Recovery match:
{
  "current_table_id": "tbl_p053_i02",
  "decision": "matched",
  "previous_table_id": "tbl_p051_i03",
  "reason": "Title identical: 'FINANCEMENT À LONG TERME¹'; indicators: same 5 currency rows (dollar canadien, américain, euro, livre sterling, total); row_count 5 vs 5.",
  "match_confidence": 0.85
}

If no reasonable remaining match exists:
{
  "current_table_id": "tbl_p039_i01",
  "decision": "added",
  "reason": "Indicators are unique; no remaining previous table has a sufficiently similar indicator structure, title, or business purpose."
}

Output must be valid JSON following the response_schema.
"""

MATCHING_REPAIR_SYSTEM_PROMPT = """
You are a structural repair agent for a financial table matching ledger.

The primary matcher already made the business decisions. Preserve every locked
decision and repair ONLY the current-table decisions explicitly listed in the
repair payload. Do not redo the full matching exercise.

IDENTIFIER CONTRACT:
- Every Previous Quarter identifier starts with `PQ::`.
- Every Current Quarter identifier starts with `CQ::`.
- Never place a `CQ::` identifier in `previous_table_id`.
- Never place a `PQ::` identifier in `current_table_id`.
- Use only identifiers permitted by the response schema.

Return one decision for every requested CQ identifier and use each PQ
identifier at most once. Prefer `unresolved` or `added` over a speculative
match. Return JSON only.
"""

MATCHING_ADJUDICATOR_SYSTEM_PROMPT = """
You are the final independent adjudicator for structurally invalid financial
table matching decisions.

Review only the remaining disputed Current Quarter tables. Respect the locked
ledger, the `PQ::`/`CQ::` identifier namespaces, and the exact identifier enums
in the response schema. Choose a match only when the table evidence is strong;
otherwise return `unresolved` or `added`. Return JSON only.
"""


# ---------------------------------------------------------------------------
# TypedDict interfaces (annotation-only)
# ---------------------------------------------------------------------------


class MatchedPair(TypedDict):
    """Paire de tableaux apparies entre le trimestre precedent et le trimestre courant."""

    previous_table_id: str
    current_table_id: str
    match_confidence: float
    reason: str


class MatchingResult(TypedDict):
    """Resultat complet du processus d'appariement des tableaux."""

    executed: bool
    matched_pairs: list[MatchedPair]
    tables_added: list[dict[str, str]]
    tables_removed: list[dict[str, str]]
    warnings: list[str]
    matching_passes_total: int


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class _MatchingValidationError(ValueError):
    """Erreur de validation structurelle de la reponse d'appariement du LLM."""

    def __init__(
        self,
        message: str,
        *,
        duplicate_count: int = 0,
        validation_failures: int = 1,
    ) -> None:
        """Initialise l'erreur avec les compteurs de duplicatas et d'échecs de validation."""
        super().__init__(message)
        self.duplicate_count = int(max(0, duplicate_count))
        self.validation_failures = int(max(1, validation_failures))


# ---------------------------------------------------------------------------
# Response normalization helpers
# ---------------------------------------------------------------------------


def _normalize_matching_warnings(items: Any) -> list[str]:
    """Filtre et normalise une liste brute d'avertissements d'appariement."""
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _previous_alias(table_id: str) -> str:
    """Retourne l'identifiant explicitement espace de noms du trimestre precedent."""
    return f"{_PREVIOUS_ID_PREFIX}{table_id}"


def _current_alias(table_id: str) -> str:
    """Retourne l'identifiant explicitement espace de noms du trimestre courant."""
    return f"{_CURRENT_ID_PREFIX}{table_id}"


def _decode_previous_alias(value: Any) -> str:
    """Decode un alias PQ tout en acceptant les IDs bruts des anciens appelants."""
    text = str(value or "").strip()
    return text[len(_PREVIOUS_ID_PREFIX) :] if text.startswith(_PREVIOUS_ID_PREFIX) else text


def _decode_current_alias(value: Any) -> str:
    """Decode un alias CQ tout en acceptant les IDs bruts des anciens appelants."""
    text = str(value or "").strip()
    return text[len(_CURRENT_ID_PREFIX) :] if text.startswith(_CURRENT_ID_PREFIX) else text


def _alias_table_card(card: dict[str, Any], *, previous: bool) -> dict[str, Any]:
    """Copie une fiche de tableau en remplacant son ID par un alias PQ/CQ."""
    aliased = dict(card)
    table_id = str(card.get("table_id", "") or "")
    aliased["table_id"] = _previous_alias(table_id) if previous else _current_alias(table_id)
    return aliased


def _canonical_matching_item(item: dict[str, Any]) -> dict[str, Any]:
    """Construit une decision canonique a partir d'un item structurellement valide."""
    decision = str(item.get("decision", "") or "").strip().lower()
    normalized: dict[str, Any] = {
        "current_table_id": _decode_current_alias(item.get("current_table_id")),
        "decision": decision,
        "reason": str(item.get("reason", "") or "").strip(),
    }
    if decision == "matched":
        normalized["previous_table_id"] = _decode_previous_alias(item.get("previous_table_id"))
        normalized["match_confidence"] = max(
            0.0,
            min(1.0, _coerce_float(item.get("match_confidence"))),
        )
    return normalized


def _analyze_matching_candidate(
    data: dict[str, Any],
    *,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Separe les decisions verrouillables des decisions a reparer.

    Cette analyse est strictement structurelle : elle ne choisit jamais une
    correspondance metier. Elle preserve seulement les decisions deja valides.
    """
    consumed_previous = set(consumed_previous_ids or ())
    raw_items = [item for item in list(data.get("current_table_decisions", []) or []) if isinstance(item, dict)]

    current_occurrences: dict[str, list[dict[str, Any]]] = {}
    previous_occurrences: dict[str, list[str]] = {}
    unknown_current_ids: set[str] = set()
    unknown_previous_ids: set[str] = set()

    for item in raw_items:
        current_id = _decode_current_alias(item.get("current_table_id"))
        if current_id not in current_ids:
            if current_id:
                unknown_current_ids.add(current_id)
            continue
        current_occurrences.setdefault(current_id, []).append(item)
        decision = str(item.get("decision", "") or "").strip().lower()
        if decision != "matched":
            continue
        previous_id = _decode_previous_alias(item.get("previous_table_id"))
        if previous_id in previous_ids:
            previous_occurrences.setdefault(previous_id, []).append(current_id)
        elif previous_id:
            unknown_previous_ids.add(previous_id)

    duplicate_current_ids = {current_id for current_id, items in current_occurrences.items() if len(items) != 1}
    duplicate_previous_assignments = {
        previous_id: sorted(set(assigned_current_ids))
        for previous_id, assigned_current_ids in previous_occurrences.items()
        if len(assigned_current_ids) > 1
    }
    missing_current_ids = current_ids - set(current_occurrences)
    repair_current_ids = set(missing_current_ids) | set(duplicate_current_ids)
    locked_decisions: list[dict[str, Any]] = []

    for current_id in sorted(current_ids):
        items = current_occurrences.get(current_id, [])
        if len(items) != 1 or current_id in repair_current_ids:
            repair_current_ids.add(current_id)
            continue
        item = items[0]
        decision = str(item.get("decision", "") or "").strip().lower()
        previous_id = _decode_previous_alias(item.get("previous_table_id"))
        confidence_raw = item.get("match_confidence")
        confidence_supplied = confidence_raw is not None and str(confidence_raw).strip() != ""

        is_valid = decision in allowed_decisions
        if decision == "matched":
            is_valid = bool(
                is_valid
                and previous_id in previous_ids
                and previous_id not in consumed_previous
                and previous_id not in duplicate_previous_assignments
            )
        else:
            is_valid = bool(is_valid and not previous_id and not confidence_supplied)

        if not is_valid:
            repair_current_ids.add(current_id)
            continue
        locked_decisions.append(_canonical_matching_item(item))

    used_locked_previous_ids = {
        str(item.get("previous_table_id", "") or "") for item in locked_decisions if item.get("decision") == "matched"
    }
    available_previous_ids = previous_ids - consumed_previous - used_locked_previous_ids

    invalid_decisions = [
        dict(item)
        for item in raw_items
        if _decode_current_alias(item.get("current_table_id")) in repair_current_ids
        or _decode_current_alias(item.get("current_table_id")) not in current_ids
    ]
    diagnostics = {
        "missing_current_table_ids": sorted(missing_current_ids),
        "duplicate_current_table_ids": sorted(duplicate_current_ids),
        "duplicate_previous_assignments": duplicate_previous_assignments,
        "unknown_current_table_ids": sorted(unknown_current_ids),
        "unknown_previous_table_ids": sorted(unknown_previous_ids),
        "wrong_namespace_previous_ids": sorted(unknown_previous_ids & current_ids),
    }
    return {
        "locked_decisions": locked_decisions,
        "repair_current_ids": repair_current_ids,
        "available_previous_ids": available_previous_ids,
        "invalid_decisions": invalid_decisions,
        "diagnostics": diagnostics,
        "warnings": _normalize_matching_warnings(data.get("warnings", [])),
    }


def _build_matching_repair_response_model(
    *,
    current_aliases: list[str],
    previous_aliases: list[str],
    allowed_decisions: set[str],
) -> type:
    """Construit un schema OpenAI dont les IDs sont des enums fermes PQ/CQ."""
    from pydantic import ConfigDict, Field, create_model

    current_id_type = Literal.__getitem__(tuple(current_aliases))
    previous_id_type = Literal.__getitem__(tuple(["", *previous_aliases]))
    decision_type = Literal.__getitem__(tuple(sorted(allowed_decisions)))
    decision_model = create_model(
        "MatchingRepairDecision",
        __config__=ConfigDict(extra="forbid"),
        current_table_id=(current_id_type, ...),
        decision=(decision_type, ...),
        reason=(str, ...),
        previous_table_id=(previous_id_type, ""),
        match_confidence=(float | None, Field(default=None, ge=0.0, le=1.0)),
    )
    return create_model(
        "MatchingRepairResponse",
        __config__=ConfigDict(extra="forbid"),
        current_table_decisions=(list[decision_model], ...),
        warnings=(list[str], Field(default_factory=list)),
    )


def _build_matching_repair_prompt(
    *,
    stage: str,
    repair_round: int,
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    current_ids: set[str],
    allowed_decisions: set[str],
    validation_feedback: str,
    repair_state: dict[str, Any],
) -> dict[str, Any]:
    """Construit le prompt cible du reparateur ou de l'arbitre."""
    repair_current_ids = set(repair_state["repair_current_ids"])
    available_previous_ids = set(repair_state["available_previous_ids"])
    repair_allowed_decisions = set(allowed_decisions)
    if not available_previous_ids:
        repair_allowed_decisions.discard("matched")

    current_aliases = [_current_alias(table_id) for table_id in sorted(repair_current_ids)]
    previous_aliases = [_previous_alias(table_id) for table_id in sorted(available_previous_ids)]
    return {
        "stage": stage,
        "agent_role": "matching_structure_repair" if repair_round == 1 else "matching_final_adjudicator",
        "task": "Repair only the structurally invalid decisions. Locked decisions are preserved by the application and must not be returned.",
        "rules": [
            "Return exactly one decision for every required_repair_current_table_id.",
            "Use CQ:: identifiers only in current_table_id.",
            "Use PQ:: identifiers only in previous_table_id.",
            "Use each PQ:: identifier at most once.",
            "For a non-matched decision, return previous_table_id as an empty string and match_confidence as null.",
            "Do not return locked decisions or any unrequested current table.",
            "Prefer unresolved or added when the evidence is ambiguous.",
        ],
        "validation_feedback": validation_feedback,
        "validation_diagnostics": repair_state["diagnostics"],
        "invalid_decisions": repair_state["invalid_decisions"],
        "locked_decisions": repair_state["locked_decisions"],
        # Kept for observability and backward-compatible diagnostics.
        "required_current_table_ids": sorted(current_ids),
        "required_repair_current_table_ids": current_aliases,
        "allowed_previous_table_ids": previous_aliases,
        "allowed_decisions": sorted(repair_allowed_decisions),
        "previous_tables": [
            _alias_table_card(card, previous=True)
            for card in previous_cards
            if str(card.get("table_id", "") or "") in available_previous_ids
        ],
        "current_tables": [
            _alias_table_card(card, previous=False)
            for card in current_cards
            if str(card.get("table_id", "") or "") in repair_current_ids
        ],
        "response_schema": {
            "current_table_decisions": [
                {
                    "current_table_id": f"one_of_{current_aliases}",
                    "decision": f"one_of_{sorted(repair_allowed_decisions)}",
                    "reason": "short evidence-grounded explanation",
                    "previous_table_id": f"one_of_{['', *previous_aliases]}",
                    "match_confidence": "number_0_to_1_or_null",
                }
            ],
            "warnings": ["string"],
        },
    }


def _merge_matching_repair_response(
    repair_data: dict[str, Any],
    *,
    repair_state: dict[str, Any],
) -> dict[str, Any]:
    """Fusionne les seules decisions reparees avec le registre verrouille."""
    repair_current_ids = set(repair_state["repair_current_ids"])
    repaired_decisions: list[dict[str, Any]] = []
    for item in list(repair_data.get("current_table_decisions", []) or []):
        if not isinstance(item, dict):
            continue
        decoded = dict(item)
        decoded["current_table_id"] = _decode_current_alias(item.get("current_table_id"))
        decoded["previous_table_id"] = _decode_previous_alias(item.get("previous_table_id"))
        if decoded["current_table_id"] in repair_current_ids:
            repaired_decisions.append(decoded)
    return {
        "current_table_decisions": [
            *list(repair_state["locked_decisions"]),
            *repaired_decisions,
        ],
        "warnings": _normalize_matching_warnings(
            [
                *list(repair_state.get("warnings", []) or []),
                *list(repair_data.get("warnings", []) or []),
            ]
        ),
    }


def _build_matching_fail_soft_response(
    data: dict[str, Any],
    *,
    stage: str,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Preserve les decisions valides et place les conflits en revue non bloquante."""
    repair_state = _analyze_matching_candidate(
        data,
        previous_ids=previous_ids,
        current_ids=current_ids,
        allowed_decisions=allowed_decisions,
        consumed_previous_ids=consumed_previous_ids,
    )
    fallback_decision = "unresolved" if "unresolved" in allowed_decisions else "added"
    repair_ids = sorted(set(repair_state["repair_current_ids"]))
    fallback_items = [
        {
            "current_table_id": current_id,
            "decision": fallback_decision,
            "reason": (
                f"Structural matching repair exhausted; this table requires review after the {stage} matching stage."
            ),
        }
        for current_id in repair_ids
    ]
    return {
        "current_table_decisions": [
            *list(repair_state["locked_decisions"]),
            *fallback_items,
        ],
        "warnings": _normalize_matching_warnings(
            [
                *list(repair_state.get("warnings", []) or []),
                f"matching_structure_repair_exhausted:{','.join(repair_ids)}",
            ]
        ),
    }


def _sort_matched_pairs(
    pairs: list[dict[str, Any]],
    previous_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Trie les paires appariees selon l'ordre d'apparition des tableaux precedents."""
    order = {str(card.get("table_id", "") or ""): index for index, card in enumerate(previous_cards)}
    return sorted(
        pairs,
        key=lambda item: (
            order.get(str(item.get("previous_table_id", "") or ""), 10**9),
            str(item.get("previous_table_id", "") or ""),
            str(item.get("current_table_id", "") or ""),
        ),
    )


def _normalize_matching_response(
    data: dict[str, Any],
    *,
    previous_ids: set[str],
    current_ids: set[str],
    allowed_decisions: set[str],
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Valide et normalise la reponse JSON brute du LLM pour l'appariement.

    Verifie les contraintes d'unicite, les identifiants valides et la
    couverture complete des tableaux courants. Leve ``_MatchingValidationError``
    en cas de violation.

    Args:
        data: Reponse JSON brute du LLM contenant ``current_table_decisions``.
        previous_ids: Ensemble des identifiants de tableaux du trimestre precedent.
        current_ids: Ensemble des identifiants de tableaux du trimestre courant.
        allowed_decisions: Decisions autorisees (ex. ``{"matched", "unresolved"}``).
        consumed_previous_ids: Identifiants precedents deja consommes par une
            etape anterieure.

    Returns:
        Dictionnaire normalise avec ``current_table_decisions``, ``warnings``
        et metriques de doublons/deduplication.

    Raises:
        _MatchingValidationError: Si la reponse viole les contraintes structurelles.
    """
    current_table_decisions: list[dict[str, Any]] = []
    used_previous: set[str] = set()
    used_current: set[str] = set()
    duplicate_total = 0
    raw_total = 0
    consumed_previous = set(consumed_previous_ids or ())

    for item in list(data.get("current_table_decisions", []) or []):
        if not isinstance(item, dict):
            raise _MatchingValidationError("current_table_decisions items must be objects")
        current_table_id = _require_string(item.get("current_table_id"), "current_table_id")
        decision = _require_string(item.get("decision"), "decision").lower()
        if decision not in allowed_decisions:
            raise _MatchingValidationError(
                f"Invalid matching decision returned by GPT: {decision!r}; allowed={sorted(allowed_decisions)}"
            )
        if current_table_id not in current_ids:
            raise _MatchingValidationError(f"Unknown current_table_id returned by GPT: {current_table_id}")
        if current_table_id in used_current:
            duplicate_total += 1
            raise _MatchingValidationError(
                f"Duplicate current_table_id returned by GPT: {current_table_id}",
                duplicate_count=duplicate_total,
            )
        used_current.add(current_table_id)

        normalized_item: dict[str, Any] = {
            "current_table_id": current_table_id,
            "decision": decision,
            "reason": str(item.get("reason", "") or "").strip(),
        }

        previous_table_id = str(item.get("previous_table_id", "") or "").strip()
        confidence_raw = item.get("match_confidence")
        confidence_supplied = confidence_raw is not None and str(confidence_raw).strip() != ""

        if decision == "matched":
            if not previous_table_id:
                raise _MatchingValidationError("Matched decisions must include previous_table_id.")
            if previous_table_id not in previous_ids:
                raise _MatchingValidationError(f"Unknown previous_table_id returned by GPT: {previous_table_id}")
            raw_total += 1
            if previous_table_id in consumed_previous or previous_table_id in used_previous:
                duplicate_total += 1
                raise _MatchingValidationError(
                    f"Duplicate or reused previous_table_id returned by GPT: {previous_table_id}",
                    duplicate_count=duplicate_total,
                )
            try:
                confidence = float(confidence_raw or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            normalized_item["previous_table_id"] = previous_table_id
            normalized_item["match_confidence"] = max(0.0, min(1.0, confidence))
            used_previous.add(previous_table_id)
        else:
            if previous_table_id:
                raise _MatchingValidationError(f"{decision!r} decisions must not include previous_table_id.")
            if confidence_supplied:
                raise _MatchingValidationError(f"{decision!r} decisions must not include match_confidence.")

        current_table_decisions.append(normalized_item)

    if used_current != current_ids:
        missing = sorted(current_ids - used_current)
        extra = sorted(used_current - current_ids)
        raise _MatchingValidationError(
            f"Matching output must cover exactly the business current tables. missing={missing} extra={extra}"
        )

    return {
        "current_table_decisions": current_table_decisions,
        "warnings": _normalize_matching_warnings(data.get("warnings", [])),
        "matching_pairs_llm_duplicates_total": duplicate_total,
        "matching_pairs_llm_deduped_total": max(0, raw_total - len(used_previous)),
    }


def _matching_decisions_to_pairs(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extrait les paires appariees depuis la liste des decisions d'appariement."""
    out: list[dict[str, Any]] = []
    for item in decisions:
        if item.get("decision") != "matched":
            continue
        out.append(
            {
                "previous_table_id": str(item.get("previous_table_id", "") or ""),
                "current_table_id": str(item.get("current_table_id", "") or ""),
                "match_confidence": _coerce_float(item.get("match_confidence")),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )
    return out


def _matching_decisions_to_table_refs(
    decisions: list[dict[str, Any]],
    *,
    decision: str,
) -> list[dict[str, str]]:
    """Extrait les references de tableaux pour un type de decision donne."""
    out: list[dict[str, str]] = []
    for item in decisions:
        if item.get("decision") != decision:
            continue
        out.append(
            {
                "table_id": str(item.get("current_table_id", "") or ""),
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )
    return out


def _empty_matching_result(
    *,
    tables_removed: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Construit un resultat d'appariement vide avec toutes les metriques a zero."""
    return {
        "executed": False,
        "matched_pairs": [],
        "tables_added": [],
        "tables_removed": list(tables_removed or []),
        "warnings": [],
        "matching_pairs_llm_duplicates_total": 0,
        "matching_pairs_llm_deduped_total": 0,
        "validation_retries_total": 0,
        "matching_validation_failures_total": 0,
        "stage1_validation_retries_total": 0,
        "stage2_validation_retries_total": 0,
        "unresolved_after_stage1_total": 0,
        "matched_in_stage2_total": 0,
        "matching_passes_total": 0,
        "inspector_passes_total": 0,
        "inspector_rejected_total": 0,
        "inspector_confirmed_total": 0,
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_matching_stage_prompt(
    *,
    stage: str,
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    current_ids: set[str],
    previous_ids: set[str],
    allowed_decisions: set[str],
    validation_feedback: str,
) -> dict[str, Any]:
    """Construit le prompt utilisateur JSON pour une etape d'appariement.

    Args:
        stage: Etape d'appariement (``"primary"`` ou ``"recovery"``).
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        current_ids: Identifiants des tableaux courants.
        previous_ids: Identifiants des tableaux precedents.
        allowed_decisions: Decisions autorisees pour cette etape.
        validation_feedback: Retour de validation a inclure si la tentative
            precedente a echoue.

    Returns:
        Dictionnaire JSON representant le prompt utilisateur complet.
    """
    decision_values = sorted(allowed_decisions)
    response_item: dict[str, Any] = {
        "current_table_id": "string",
        "decision": f"one_of_{decision_values}",
        "reason": "short explanation grounded in indicators, headers, row_count, title, table_summary, footnotes, section, and page only if needed",
    }
    if "matched" in allowed_decisions:
        response_item["previous_table_id"] = "string_required_when_decision_is_matched"
        response_item["match_confidence"] = "number_0_to_1_required_when_decision_is_matched"

    if stage == "primary":
        task = (
            "Perform a strict global reconciliation of business banking tables between the previous and current reports. "
            "For each current table, return either matched or unresolved."
        )
        rules = [
            "Return JSON only, strictly following response_schema.",
            "Every current_table_id must appear exactly once in current_table_decisions.",
            "Each previous_table_id can be used at most once across matched decisions.",
            "This is the strict precision-first pass: if any material doubt remains, return unresolved instead of forcing a match.",
            "Do not classify any table as added in this stage.",
        ]
    else:
        task = (
            "Resolve the remaining ambiguous current business tables against the remaining unused previous business tables. "
            "For each current table, return either matched or added."
        )
        rules = [
            "Return JSON only, strictly following response_schema.",
            "This is a recovery pass over unresolved current tables only.",
            "Every current_table_id must appear exactly once in current_table_decisions.",
            "Each previous_table_id can be used at most once across matched decisions.",
            "If the best remaining candidate is still ambiguous, return added instead of forcing a speculative match.",
            "Do not skip any current table.",
        ]

    prompt = {
        "stage": stage,
        "task": task,
        "rules": rules,
        "response_schema": {
            "current_table_decisions": [response_item],
            "warnings": ["string"],
        },
        "allowed_decisions": decision_values,
        "required_current_table_ids": sorted(current_ids),
        "allowed_previous_table_ids": sorted(previous_ids),
        "previous_tables": previous_cards,
        "current_tables": current_cards,
    }
    if validation_feedback:
        prompt["validation_feedback"] = validation_feedback
        prompt["rules"].append(
            "Your previous response was structurally invalid. Fix the listed validation problem and return a corrected JSON object."
        )
    return prompt


# ---------------------------------------------------------------------------
# Matching execution (call_openai_json is injected to preserve monkeypatch)
# ---------------------------------------------------------------------------


def _run_matching_stage(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    stage: str,
    allowed_decisions: set[str],
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
    consumed_previous_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Execute une etape d'appariement (primaire ou recuperation) avec validation.

    Appelle d'abord le matcher existant sans modifier son comportement. Si sa
    reponse est structurellement invalide, repare uniquement les decisions en
    conflit avec des identifiants PQ/CQ fermes, puis fait arbitrer les conflits
    persistants. Une derniere degradation non bloquante garde les decisions
    valides et classe le reliquat en ``unresolved`` ou ``added``.

    Args:
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        stage: Etape d'appariement (``"primary"`` ou ``"recovery"``).
        allowed_decisions: Decisions autorisees pour cette etape.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Callable injecte pour appeler l'API OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer les metriques d'utilisation.
        consumed_previous_ids: Identifiants precedents deja consommes.

    Returns:
        Dictionnaire normalise contenant les decisions, avertissements et
        metriques de validation.

    """
    previous_ids = {card["table_id"] for card in previous_cards}
    current_ids = {card["table_id"] for card in current_cards}
    system_prompt = PRIMARY_MATCH_SYSTEM_PROMPT if stage == "primary" else RECOVERY_MATCH_SYSTEM_PROMPT
    validation_feedback = ""
    validation_retries_total = 0
    matching_validation_failures_total = 0
    matching_pairs_llm_duplicates_total = 0
    candidate_data: dict[str, Any] = {}

    for attempt in range(_MATCHING_VALIDATION_ATTEMPTS):
        if attempt == 0:
            prompt = _build_matching_stage_prompt(
                stage=stage,
                previous_cards=previous_cards,
                current_cards=current_cards,
                current_ids=current_ids,
                previous_ids=previous_ids,
                allowed_decisions=allowed_decisions,
                validation_feedback="",
            )
            data = call_openai_json(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False),
                    },
                ],
                usage_recorder=usage_recorder,
                call_kind="matching",
            )
        else:
            repair_state = _analyze_matching_candidate(
                candidate_data,
                previous_ids=previous_ids,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                consumed_previous_ids=consumed_previous_ids,
            )
            repair_prompt = _build_matching_repair_prompt(
                stage=stage,
                repair_round=attempt,
                previous_cards=previous_cards,
                current_cards=current_cards,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                validation_feedback=validation_feedback,
                repair_state=repair_state,
            )
            response_model = _build_matching_repair_response_model(
                current_aliases=list(repair_prompt["required_repair_current_table_ids"]),
                previous_aliases=list(repair_prompt["allowed_previous_table_ids"]),
                allowed_decisions=set(repair_prompt["allowed_decisions"]),
            )
            repair_data = call_openai_json(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            MATCHING_REPAIR_SYSTEM_PROMPT if attempt == 1 else MATCHING_ADJUDICATOR_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(repair_prompt, ensure_ascii=False),
                    },
                ],
                usage_recorder=usage_recorder,
                call_kind="matching",
                response_model=response_model,
            )
            data = _merge_matching_repair_response(
                repair_data,
                repair_state=repair_state,
            )
        candidate_data = data
        try:
            normalized = _normalize_matching_response(
                data,
                previous_ids=previous_ids,
                current_ids=current_ids,
                allowed_decisions=allowed_decisions,
                consumed_previous_ids=consumed_previous_ids,
            )
            normalized["executed"] = True
            normalized["validation_retries_total"] = validation_retries_total
            normalized["matching_validation_failures_total"] = matching_validation_failures_total
            normalized["matching_pairs_llm_duplicates_total"] += matching_pairs_llm_duplicates_total
            return normalized
        except _MatchingValidationError as exc:
            validation_feedback = str(exc)
            validation_retries_total += 1
            matching_validation_failures_total += int(getattr(exc, "validation_failures", 1))
            matching_pairs_llm_duplicates_total += int(getattr(exc, "duplicate_count", 0))
            if attempt + 1 >= _MATCHING_VALIDATION_ATTEMPTS:
                fallback_data = _build_matching_fail_soft_response(
                    candidate_data,
                    stage=stage,
                    previous_ids=previous_ids,
                    current_ids=current_ids,
                    allowed_decisions=allowed_decisions,
                    consumed_previous_ids=consumed_previous_ids,
                )
                normalized = _normalize_matching_response(
                    fallback_data,
                    previous_ids=previous_ids,
                    current_ids=current_ids,
                    allowed_decisions=allowed_decisions,
                    consumed_previous_ids=consumed_previous_ids,
                )
                normalized["executed"] = True
                normalized["validation_retries_total"] = validation_retries_total
                normalized["matching_validation_failures_total"] = matching_validation_failures_total
                normalized["matching_pairs_llm_duplicates_total"] += matching_pairs_llm_duplicates_total
                return normalized

    raise RuntimeError("Unreachable matching validation loop")


def _match_tables(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Orchestre l'appariement complet en deux etapes avec inspection intermediaire.

    Etape 1 (primaire) : appariement strict precision-first.
    Etape 1.5 : inspection GenAI des paires appariees (rejet des faux positifs).
    Etape 2 (recuperation) : resolution des tableaux non resolus restants.

    Args:
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Callable injecte pour appeler l'API OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer les metriques d'utilisation.

    Returns:
        Dictionnaire contenant les paires appariees, tableaux ajoutes/supprimes,
        avertissements et metriques detaillees des deux etapes.
    """
    if not previous_cards and not current_cards:
        return _empty_matching_result()

    if not current_cards:
        tables_removed = [
            {
                "table_id": str(card.get("table_id", "") or ""),
                "reason": "No current business table was available for matching.",
            }
            for card in previous_cards
        ]
        return _empty_matching_result(tables_removed=tables_removed)

    stage1 = _run_matching_stage(
        previous_cards,
        current_cards,
        stage="primary",
        allowed_decisions={"matched", "unresolved"},
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    stage1_decisions = list(stage1.get("current_table_decisions", []) or [])
    stage1_pairs = _matching_decisions_to_pairs(stage1_decisions)

    # --- Stage 1.5: Match Inspector (pair-level GenAI verification) ----------
    inspector_result = _inspect_matched_pairs(
        stage1_pairs,
        previous_cards,
        current_cards,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    confirmed_stage1_pairs = inspector_result["confirmed_pairs"]
    rejected_stage1_pairs = inspector_result["rejected_pairs"]
    inspector_stats = inspector_result.get("inspection_stats", {})

    if rejected_stage1_pairs:
        logger.info(
            "Match Inspector rejected %d pairs — returning to unresolved pool",
            len(rejected_stage1_pairs),
        )

    # Use only confirmed pairs as consumed; rejected pairs go back to pools
    used_previous_stage1 = {
        item["previous_table_id"]
        for item in confirmed_stage1_pairs
        if str(item.get("previous_table_id", "") or "").strip()
    }
    # Unresolved = original unresolved + rejected CQ tables
    unresolved_ids = [item["current_table_id"] for item in stage1_decisions if item.get("decision") == "unresolved"] + [
        item["current_table_id"] for item in rejected_stage1_pairs
    ]
    unresolved_lookup = {card["table_id"]: card for card in current_cards}
    unresolved_current_cards = [
        unresolved_lookup[table_id] for table_id in unresolved_ids if table_id in unresolved_lookup
    ]
    remaining_previous_cards = [card for card in previous_cards if card["table_id"] not in used_previous_stage1]

    stage2_decisions: list[dict[str, Any]] = []
    stage2_metrics = {
        "executed": False,
        "validation_retries_total": 0,
        "matching_validation_failures_total": 0,
        "matching_pairs_llm_duplicates_total": 0,
        "matching_pairs_llm_deduped_total": 0,
        "warnings": [],
    }
    tables_added: list[dict[str, str]] = []

    if unresolved_current_cards and remaining_previous_cards:
        stage2 = _run_matching_stage(
            remaining_previous_cards,
            unresolved_current_cards,
            stage="recovery",
            allowed_decisions={"matched", "added"},
            model=model,
            call_openai_json=call_openai_json,
            usage_recorder=usage_recorder,
            consumed_previous_ids=used_previous_stage1,
        )
        stage2_metrics = stage2
        stage2_decisions = list(stage2.get("current_table_decisions", []) or [])
        tables_added = _matching_decisions_to_table_refs(
            stage2_decisions,
            decision="added",
        )
    elif unresolved_current_cards:
        tables_added = [
            {
                "table_id": str(item.get("current_table_id", "") or ""),
                "reason": (
                    str(item.get("reason", "") or "").strip()
                    or "No previous business table remained available for matching."
                ),
            }
            for item in stage1_decisions
            if item.get("decision") == "unresolved"
        ]

    matched_pairs = confirmed_stage1_pairs + _matching_decisions_to_pairs(stage2_decisions)
    used_previous_all = {
        str(item.get("previous_table_id", "") or "").strip()
        for item in matched_pairs
        if str(item.get("previous_table_id", "") or "").strip()
    }
    tables_removed = [
        {
            "table_id": str(card.get("table_id", "") or ""),
            "reason": "No current business table was matched to this previous table.",
        }
        for card in previous_cards
        if card["table_id"] not in used_previous_all
    ]
    warnings = _normalize_matching_warnings(
        list(stage1.get("warnings", []) or []) + list(stage2_metrics.get("warnings", []) or [])
    )

    return {
        "executed": bool(stage1.get("executed") or stage2_metrics.get("executed")),
        "matched_pairs": matched_pairs,
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "warnings": warnings,
        "matching_pairs_llm_duplicates_total": _coerce_int(stage1.get("matching_pairs_llm_duplicates_total"))
        + _coerce_int(stage2_metrics.get("matching_pairs_llm_duplicates_total")),
        "matching_pairs_llm_deduped_total": _coerce_int(stage1.get("matching_pairs_llm_deduped_total"))
        + _coerce_int(stage2_metrics.get("matching_pairs_llm_deduped_total")),
        "validation_retries_total": _coerce_int(stage1.get("validation_retries_total"))
        + _coerce_int(stage2_metrics.get("validation_retries_total")),
        "matching_validation_failures_total": _coerce_int(stage1.get("matching_validation_failures_total"))
        + _coerce_int(stage2_metrics.get("matching_validation_failures_total")),
        "stage1_validation_retries_total": _coerce_int(stage1.get("validation_retries_total")),
        "stage2_validation_retries_total": _coerce_int(stage2_metrics.get("validation_retries_total")),
        "unresolved_after_stage1_total": len(unresolved_current_cards),
        "matched_in_stage2_total": len([item for item in stage2_decisions if item.get("decision") == "matched"]),
        "matching_passes_total": int(bool(stage1.get("executed"))) + int(bool(stage2_metrics.get("executed"))),
        "inspector_passes_total": _coerce_int(inspector_stats.get("total_inspected")),
        "unmatched_after_primary_total": len(unresolved_current_cards) + len(remaining_previous_cards),
        "unmatched_after_rescue_total": len(tables_added) + len(tables_removed),
        "inspector_rejected_total": _coerce_int(inspector_stats.get("rejected")),
        "inspector_confirmed_total": _coerce_int(inspector_stats.get("confirmed")),
    }


def _run_table_matching(
    previous_cards: list[dict[str, Any]],
    current_cards: list[dict[str, Any]],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Point d'entree principal de l'appariement des tableaux.

    Appelle ``_match_tables`` puis trie les paires et structure le resultat
    final avec toutes les metriques de suivi.

    Args:
        previous_cards: Fiches des tableaux du trimestre precedent.
        current_cards: Fiches des tableaux du trimestre courant.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Callable injecte pour appeler l'API OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer les metriques d'utilisation.

    Returns:
        Dictionnaire structure contenant ``matched_pairs``, ``tables_added``,
        ``tables_removed``, ``warnings`` et toutes les metriques de suivi.
    """
    result = _match_tables(
        previous_cards,
        current_cards,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    result["matched_pairs"] = _sort_matched_pairs(result["matched_pairs"], previous_cards)
    tables_added = list(result.get("tables_added", []) or [])
    tables_removed = list(result.get("tables_removed", []) or [])
    return {
        "matched_pairs": result["matched_pairs"],
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "matching_passes_total": _coerce_int(result.get("matching_passes_total")),
        "inspector_passes_total": _coerce_int(result.get("inspector_passes_total")),
        "audit_passes_total": 0,
        "matching_output_retries_total": _coerce_int(result.get("validation_retries_total")),
        "matching_validation_failures_total": _coerce_int(result.get("matching_validation_failures_total")),
        "stage1_validation_retries_total": _coerce_int(result.get("stage1_validation_retries_total")),
        "stage2_validation_retries_total": _coerce_int(result.get("stage2_validation_retries_total")),
        "unresolved_after_stage1_total": _coerce_int(result.get("unresolved_after_stage1_total")),
        "matched_in_stage2_total": _coerce_int(result.get("matched_in_stage2_total")),
        "unmatched_previous_table_ids": [item["table_id"] for item in tables_removed],
        "unmatched_current_table_ids": [item["table_id"] for item in tables_added],
        "unmatched_after_primary_total": _coerce_int(result.get("unmatched_after_primary_total")),
        "unmatched_after_rescue_total": _coerce_int(result.get("unmatched_after_rescue_total")),
        "matching_pairs_llm_duplicates_total": _coerce_int(result.get("matching_pairs_llm_duplicates_total")),
        "matching_pairs_llm_deduped_total": _coerce_int(result.get("matching_pairs_llm_deduped_total")),
        "inspector_rejected_total": _coerce_int(result.get("inspector_rejected_total")),
        "inspector_confirmed_total": _coerce_int(result.get("inspector_confirmed_total")),
        "warnings": _normalize_matching_warnings(result.get("warnings", [])),
    }
