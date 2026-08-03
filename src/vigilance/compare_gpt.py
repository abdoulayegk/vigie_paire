"""Facade historique du pipeline de comparaison GPT.

La logique vit dans ``vigilance.pipeline_comparaison``. Les wrappers conservent
les API publiques et les points de monkeypatch utilises par les tests.
"""

from vigilance.comparison_devil_advocate import _devil_advocate_review  # noqa: F401
from vigilance.comparison_diff_gpt import diff_table_pair_gpt  # noqa: F401
from vigilance.comparison_io import normalize_quarter, resolve_reference_period  # noqa: F401
from vigilance.comparison_matching import (  # noqa: F401
    _MATCHING_VALIDATION_ATTEMPTS,
    _run_table_matching,
)
from vigilance.comparison_visual_sanity import (  # noqa: F401
    render_visual_sanity_proof,
    visual_sanity_check,
    visual_sanity_check_table_event,
)
from vigilance.config import get_matching_thresholds, resolve_openai_model  # noqa: F401
from vigilance.pipeline_comparaison import client_openai as _client_mod
from vigilance.pipeline_comparaison import orchestration as _orchestration_mod
from vigilance.pipeline_comparaison.ancrages_visuels import (  # noqa: F401
    _best_text_similarity,
    _infer_opposite_page_from_matched_pairs,
    _jaccard_anchor_values,
    _normalize_table_anchor_section,
    _normalize_table_anchor_title,
    _normalized_anchor_values,
    _resolve_visual_table_anchor,
    _snapshot_has_visual_render_anchor,
    _visual_sanity_meta,
)
from vigilance.pipeline_comparaison.client_openai import (  # noqa: F401
    OPENAI_COMPARISON_TIMEOUT_SECONDS,
    _call_openai_embeddings as _call_openai_embeddings_impl,
    _call_openai_json as _call_openai_json_impl,
)
from vigilance.pipeline_comparaison.construction_resultat import (  # noqa: F401
    COMPARISON_SCHEMA_VERSION,
    DIFF_PROMPT_VERSION,
    MATCH_PROMPT_VERSION,
    REFERENCE_RESOLUTION_RULE,
    _archive_source_pdf,
)
from vigilance.pipeline_comparaison.orchestration import (
    compare_reports_gpt4o as _compare_reports_gpt4o_impl,
)
from vigilance.utils.genai import get_openai_api_key  # noqa: F401


def _call_openai_json(*args, **kwargs):
    """Delegue l appel JSON en propageant la cle monkeypatchee."""
    _client_mod.get_openai_api_key = globals()["get_openai_api_key"]
    return _call_openai_json_impl(*args, **kwargs)


def _call_openai_embeddings(*args, **kwargs):
    """Delegue les embeddings en propageant la cle monkeypatchee."""
    _client_mod.get_openai_api_key = globals()["get_openai_api_key"]
    return _call_openai_embeddings_impl(*args, **kwargs)


def compare_reports_gpt4o(*args, **kwargs):
    """Delegue le pipeline en propageant les points d injection historiques."""
    _orchestration_mod._call_openai_json = globals()["_call_openai_json"]
    _orchestration_mod._call_openai_embeddings = globals()["_call_openai_embeddings"]
    _orchestration_mod._run_table_matching = globals()["_run_table_matching"]
    _orchestration_mod._devil_advocate_review = globals()["_devil_advocate_review"]
    _orchestration_mod.diff_table_pair_gpt = globals()["diff_table_pair_gpt"]
    _orchestration_mod.render_visual_sanity_proof = globals()["render_visual_sanity_proof"]
    _orchestration_mod.visual_sanity_check = globals()["visual_sanity_check"]
    _orchestration_mod.visual_sanity_check_table_event = globals()["visual_sanity_check_table_event"]
    return _compare_reports_gpt4o_impl(*args, **kwargs)
