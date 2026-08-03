"""Facade historique des differences entre tableaux apparies.

La logique est repartie dans ``vigilance.differences_tableaux``. Les reexports
conservent les imports existants et les signatures injectant le client OpenAI.
"""

from vigilance.differences_tableaux.comparaison_deterministe import (  # noqa: F401
    _deterministic_footnote_diff,
    _deterministic_indicator_diff,
)
from vigilance.differences_tableaux.comparaison_llm import (  # noqa: F401
    FOOTNOTE_DIFF_SYSTEM_PROMPT,
    INDICATOR_DIFF_SYSTEM_PROMPT,
    _call_validated_diff_json,
    diff_footnotes_pair_gpt,
    diff_indicators_pair_gpt,
)
from vigilance.differences_tableaux.comparaison_paire import (  # noqa: F401
    _compose_pair_reason,
    diff_table_pair_gpt,
)
from vigilance.differences_tableaux.filtrage_artefacts import (  # noqa: F401
    INSPECTOR_SYSTEM_PROMPT,
    _inspect_diff_artifacts_gpt,
)
from vigilance.differences_tableaux.normalisation_elements import (  # noqa: F401
    _BLOC_SUFFIX_RE,
    _DASH_RE,
    _DATE_PREFIX_RE,
    _DATE_QUARTER_RE,
    _FOOTNOTE_MARKER_RE,
    _FOOTNOTE_PAREN_RE,
    _PAGE_REF_RE_DET,
    _STANDALONE_DATE_RE,
    _SUPERSCRIPT_DIGITS,
    _SUPERSCRIPT_STRIP,
    _enrich_indicators_with_normalized,
    _normalize_footnote_text,
    _normalize_footnotes,
    _normalize_for_diff,
    _normalize_indicator_text,
    _table_context,
    _token_overlap_ratio,
)
