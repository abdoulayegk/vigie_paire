"""Facade historique du rapprochement de tableaux.

La logique est repartie dans ``vigilance.rapprochement_tableaux``. Les
reexports conservent les imports et les points de monkeypatch existants.
"""

from vigilance.rapprochement_tableaux.contrats import (  # noqa: F401
    MATCHING_ADJUDICATOR_SYSTEM_PROMPT,
    MATCHING_REPAIR_SYSTEM_PROMPT,
    PRIMARY_MATCH_SYSTEM_PROMPT,
    RECOVERY_MATCH_SYSTEM_PROMPT,
    MatchedPair,
    MatchingResult,
    _CURRENT_ID_PREFIX,
    _MATCHING_VALIDATION_ATTEMPTS,
    _MatchingValidationError,
    _PREVIOUS_ID_PREFIX,
)
from vigilance.rapprochement_tableaux.correction_reponses import (  # noqa: F401
    _analyze_matching_candidate,
    _build_matching_fail_soft_response,
    _build_matching_repair_prompt,
    _build_matching_repair_response_model,
    _merge_matching_repair_response,
)
from vigilance.rapprochement_tableaux.moteur_rapprochement import (  # noqa: F401
    _build_matching_stage_prompt,
    _match_tables,
    _run_matching_stage,
    _run_table_matching,
)
from vigilance.rapprochement_tableaux.normalisation_reponses import (  # noqa: F401
    _alias_table_card,
    _canonical_matching_item,
    _current_alias,
    _decode_current_alias,
    _decode_previous_alias,
    _empty_matching_result,
    _matching_decisions_to_pairs,
    _matching_decisions_to_table_refs,
    _normalize_matching_response,
    _normalize_matching_warnings,
    _previous_alias,
    _sort_matched_pairs,
)
