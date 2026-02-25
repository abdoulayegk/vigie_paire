"""Run T1/T2 comparison from uploaded PDFs and section ranges."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None  # type: ignore[assignment]

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]

# Fallback defaults when config keys absent (PASS 2 wires from get_matching_thresholds)
_INDICATOR_DEFAULTS = {
    "indicator_rename_min_score": 0.86,
    "indicator_gate_min_len_ratio": 0.55,
    "indicator_gate_min_token_overlap": 1,
}

from app.ui_config import INDICATOR_COMPARISON_DIR
from vigilance.compare import run_strict_intra_section_compare
from vigilance.compare.table_fragment_merger import merge_table_fragments
from vigilance.config import get_matching_thresholds
from vigilance.models.table_models import TableArtifact
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.indicator_normalizer import get_canonical_text, get_token_sorted_text
from vigilance.utils.matching_normalizer import _classify_excluded_line


def _section_uid(table: TableArtifact) -> str:
    return f"{table.section}|{table.table_id}|p{table.page_pdf}"


def _canonical_section_name(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return "unknown_section"
    try:
        from vigilance.extraction.section_taxonomy import canonicalize_section

        return canonicalize_section(value)
    except Exception:
        lowered = value.lower().replace("é", "e").replace("è", "e")
        if "capital" in lowered or "fonds propres" in lowered:
            return "gestion_capital"
        if "risque" in lowered:
            return "gestion_risques"
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        return lowered or "unknown_section"


def _normalize_ranges(sections: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in sections or []:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_page", 0) or 0)
        end = int(item.get("end_page", start) or start)
        if start <= 0:
            continue
        if end < start:
            end = start
        section_raw = str(item.get("type") or item.get("section") or item.get("label") or "")
        section = _canonical_section_name(section_raw)
        result.append({"section": section, "start": start, "end": end})
    result.sort(key=lambda row: (int(row["start"]), str(row["section"])))
    return result


def _infer_year(*paths: str) -> int:
    for value in paths:
        match = re.search(r"(19|20)\d{2}", str(value))
        if match:
            return int(match.group(0))
    return datetime.now().year


def _table_to_artifact(table: Any, *, bank_code: str, quarter: str, pdf_path: str) -> TableArtifact:
    rows = [list(row) for row in (getattr(table, "rows", []) or [])]
    headers = [str(h) for h in (getattr(table, "headers", []) or []) if h is not None]
    indicators = [
        str(item).strip()
        for item in (getattr(table, "first_column_indicators", []) or [])
        if str(item).strip()
    ]
    if not indicators:
        for row in rows:
            if row and str(row[0]).strip():
                indicators.append(str(row[0]).strip())

    raw = getattr(table, "first_column_indicators_raw", None)
    if raw is not None:
        raw = [str(x).strip() for x in raw if str(x).strip()]
    else:
        raw = None

    section = _canonical_section_name(str(getattr(table, "section", "")))
    return TableArtifact(
        bank_code=bank_code,
        section=section,
        page_pdf=int(getattr(table, "page_number", 0) or 0),
        table_id=str(getattr(table, "table_id", "")),
        title=getattr(table, "title", None),
        headers=headers,
        rows=rows,
        first_column_indicators=indicators,
        first_column_indicators_raw=raw,
        extraction_method=getattr(table, "extraction_method", None) or "docling",
        table_number=getattr(table, "table_number", None),
        bbox=getattr(table, "bbox", None),
        quarter=quarter,
        pdf_path=pdf_path,
    )


def _extract_tables(
    *,
    pdf_path: str,
    bank_code: str,
    quarter: str,
    year: int,
    section_ranges: list[dict[str, Any]],
    use_vision_fallback: bool,
    api_key: str | None,
) -> list[TableArtifact]:
    import os

    from vigilance.extraction.docling_processor import extract_tables_docling_by_sections

    if use_vision_fallback:
        os.environ["ENABLE_VISION_FALLBACK"] = "1"
    else:
        os.environ.pop("ENABLE_VISION_FALLBACK", None)
    del api_key

    raw_tables = extract_tables_docling_by_sections(
        pdf_path=pdf_path,
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        section_ranges=section_ranges,
    )

    return [
        _table_to_artifact(table, bank_code=bank_code, quarter=quarter, pdf_path=pdf_path)
        for table in raw_tables
    ]


def _canonical_indicator_key(text: str) -> str:
    """Canonical key for indicator comparison (shared with structural_comparator)."""
    return normalize_indicator_for_comparison(text)


def _normalize_bbox_ltrb_norm(bbox: Any) -> list[float] | None:
    """Normalize bbox to [l, t, r, b] in 0..1. Returns None if invalid or outside [0,1]."""
    if bbox is None:
        return None
    try:
        l_val, t_val, r_val, b_val = 0.0, 0.0, 1.0, 1.0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            l_val, t_val, r_val, b_val = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
        elif isinstance(bbox, dict):
            if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
                l_val = float(bbox["x0"])
                t_val = float(bbox["y0"])
                r_val = float(bbox["x1"])
                b_val = float(bbox["y1"])
            elif all(k in bbox for k in ("l", "t", "r", "b")):
                l_val = float(bbox["l"])
                t_val = float(bbox["t"])
                r_val = float(bbox["r"])
                b_val = float(bbox["b"])
            elif all(k in bbox for k in ("x", "y", "width", "height")):
                l_val = float(bbox["x"])
                t_val = float(bbox["y"])
                r_val = l_val + float(bbox["width"])
                b_val = t_val + float(bbox["height"])
            else:
                return None
        else:
            return None
        if r_val <= l_val or b_val <= t_val:
            return None
        if l_val < -0.05 or t_val < -0.05 or r_val > 1.05 or b_val > 1.05:
            return None
        return [
            max(0.0, min(1.0, l_val)),
            max(0.0, min(1.0, t_val)),
            max(0.0, min(1.0, r_val)),
            max(0.0, min(1.0, b_val)),
        ]
    except (TypeError, ValueError):
        return None


def _all_indicators_value_clean_ordered(table: TableArtifact) -> list[str]:
    """Return value_clean indicators in table order (same string space as added/removed)."""
    raw = getattr(table, "first_column_indicators_raw", None) or []
    if not raw:
        raw = getattr(table, "first_column_indicators", None) or []
    result: list[str] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        if _classify_excluded_line(s):
            continue
        cleaned = _strip_footnote_markers_from_indicator(s)
        key = _canonical_indicator_key(cleaned)
        if not key:
            continue
        result.append(cleaned)
    return result


# Trailing footnote/reference patterns (do not remove semantic numbers: Tier 1, CET1, Bâle III, Pillar 3, IFRS 9)
_INDICATOR_TRAILING_SUPER = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*$")
_INDICATOR_TRAILING_STARS = re.compile(r"\s*[*\u2020\u2021\u00A7]+\s*$")  # * ** † ‡ §
_INDICATOR_TRAILING_PAREN_NUM = re.compile(r"\s*[\(\[]\d+[\)\]]\s*$")
_INDICATOR_TRAILING_NOTE_NUM = re.compile(r"\s+Note\s+\d+\s*\.?\s*$", re.IGNORECASE)
_INDICATOR_TRAILING_COMMA_NUMS = re.compile(r"\s*,\s*\d+(?:\s*,\s*\d+)*\s*$")
_INDICATOR_TRAILING_SPACE_NUMS_COMMA = re.compile(r"\s+\d+(?:\s*,\s*\d+)+\s*$")


def _strip_footnote_markers_from_indicator(text: str) -> str:
    """Remove trailing footnote markers and refs from indicator label. Preserves semantic numbers (Tier 1, CET1, etc.)."""
    if not text:
        return ""
    value = (text or "").strip()
    while True:
        prev = value
        value = _INDICATOR_TRAILING_SUPER.sub("", value)
        value = _INDICATOR_TRAILING_STARS.sub("", value)
        value = _INDICATOR_TRAILING_PAREN_NUM.sub("", value)
        value = _INDICATOR_TRAILING_NOTE_NUM.sub("", value)
        value = _INDICATOR_TRAILING_COMMA_NUMS.sub("", value)
        value = _INDICATOR_TRAILING_SPACE_NUMS_COMMA.sub("", value)
        value = re.sub(r"\s+", " ", value).strip()
        if value == prev:
            break
    return value


_INDICATOR_STOPWORDS = frozenset(
    {"de", "du", "des", "la", "le", "les", "et", "ou", "and", "the", "of", "to", "en", "au", "aux", "a", "an"}
)
_INDICATOR_UNIT_TOKENS = frozenset({"%", "million", "millions", "milliard", "milliards", "dollars", "cad", "usd"})
# Acronyms for overlap gate: if both strings contain same one, gate passes
_INDICATOR_ACRONYM_RE = re.compile(
    r"\b(cet[-]?1|at[-]?1|tlac|rwa|ifrs[-]?9|tier[-]?\s*1|tier[-]?\s*2|bale[-]?\s*iii|pillar[-]?\s*3)\b",
    re.IGNORECASE,
)


def _indicator_strong_tokens(text: str) -> set[str]:
    """Tokenize for overlap gate: normalize hyphens/slashes, drop stopwords/units, drop pure numbers except 1-9."""
    if not text:
        return set()
    normalized = re.sub(r"[-/]", " ", (text or "").lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    tokens: set[str] = set()
    for t in normalized.split():
        if not t:
            continue
        if t in _INDICATOR_STOPWORDS or t in _INDICATOR_UNIT_TOKENS:
            continue
        if t.isdigit():
            if len(t) > 1:
                continue
            if t in ("1", "2", "3", "9"):
                tokens.add(t)
            continue
        tokens.add(t)
    return tokens


def _indicator_acronyms(text: str) -> set[str]:
    """Extract allowlisted acronyms (CET1, TLAC, RWA, etc.) for overlap gate."""
    if not text:
        return set()
    return {m.group(1).lower().replace(" ", "").replace("-", "") for m in _INDICATOR_ACRONYM_RE.finditer(text or "")}


_PREFILTER_MATRIX_CAP = 25_000
_PREFILTER_TOP_K_PER_REMOVED = 50

# Order-invariant matching and embedding gating (config overridable)
_INDICATOR_EMB_MIN = 0.45
_INDICATOR_MIN_TOKENS = 6
_INDICATOR_MIN_ALPHA_RATIO = 0.40


def _hungarian_pair_added_removed(
    removed_items: list[str],
    added_items: list[str],
    *,
    th: dict[str, Any] | None = None,
    embedding_service: Any = None,
) -> tuple[list[str], list[str], list[tuple[str, str]], dict[str, Any]]:
    """
    Global 1-to-1 pairing between removed and added indicators (renames).
    Uses Hungarian assignment when scipy is available; otherwise deterministic greedy.
    Returns (added_restant, removed_restant, list of (removed_text, added_text), debug_dict).
    Debug dict is for logging only; never written to JSON.
    """
    th = th or {}
    min_score = float(th.get("indicator_rename_min_score", _INDICATOR_DEFAULTS["indicator_rename_min_score"]))
    min_len_ratio = float(th.get("indicator_gate_min_len_ratio", _INDICATOR_DEFAULTS["indicator_gate_min_len_ratio"]))
    min_token_overlap = int(th.get("indicator_gate_min_token_overlap", _INDICATOR_DEFAULTS["indicator_gate_min_token_overlap"]))
    weights_raw = th.get("indicator_similarity_weights")
    weights: dict[str, float] | None = weights_raw if isinstance(weights_raw, dict) else None

    if not removed_items or not added_items or rapidfuzz_fuzz is None:
        return list(added_items), list(removed_items), [], {"gated_out_pairs": 0, "accepted_renames": 0}

    min_score_pct = int(min_score * 100)

    def _norm_for_sort(s: str) -> str:
        return _canonical_indicator_key(_strip_footnote_markers_from_indicator(s))

    removed = sorted(removed_items, key=_norm_for_sort)
    added = sorted(added_items, key=_norm_for_sort)

    def _length_ratio_ok(a: str, r: str) -> bool:
        la, lr = len(_norm_for_sort(a)), len(_norm_for_sort(r))
        if max(la, lr) <= 0:
            return True
        return (min(la, lr) / max(la, lr)) >= min_len_ratio

    def _token_overlap_ok(a: str, r: str) -> bool:
        na, nr = _norm_for_sort(a), _norm_for_sort(r)
        ta = _indicator_strong_tokens(na)
        tr = _indicator_strong_tokens(nr)
        if len(ta & tr) >= min_token_overlap:
            return True
        acro_a = _indicator_acronyms(na)
        acro_r = _indicator_acronyms(nr)
        return len(acro_a & acro_r) > 0

    def _similarity(a: str, r: str) -> float:
        ratio_score = rapidfuzz_fuzz.ratio(a, r)
        token_score = rapidfuzz_fuzz.token_set_ratio(a, r)
        if weights:
            return weights.get("ratio", 0.4) * ratio_score + weights.get("token_set", 0.6) * token_score
        return max(ratio_score, token_score)

    use_token_sorted = bool(th.get("use_indicator_token_sorted_matching", True))
    min_tokens = int(th.get("indicator_embed_min_tokens", _INDICATOR_MIN_TOKENS))
    emb_min = float(th.get("indicator_embed_min_sim", _INDICATOR_EMB_MIN))
    min_alpha_ratio = float(th.get("indicator_embed_min_alpha_ratio", _INDICATOR_MIN_ALPHA_RATIO))

    def _lex_similarity_both_forms(a: str, r: str) -> tuple[float, float, float]:
        """Lex similarity on canonical and token_sorted; returns (lex_canon, lex_ts, lex_final)."""
        lex_canon = _similarity(a, r)
        if not use_token_sorted:
            return lex_canon, 0.0, lex_canon
        canon_a, ts_a = get_canonical_text(a), get_token_sorted_text(a)
        canon_r, ts_r = get_canonical_text(r), get_token_sorted_text(r)
        lex_ts = 0.0
        if ts_a and ts_r:
            lex_ts = max(
                rapidfuzz_fuzz.ratio(ts_a, ts_r),
                rapidfuzz_fuzz.token_set_ratio(ts_a, ts_r),
            )
        return lex_canon, lex_ts, max(lex_canon, lex_ts)

    def _embed_gate_ok(a: str, r: str, emb_sim: float) -> bool:
        if emb_sim < emb_min:
            return False
        ts_a, ts_r = get_token_sorted_text(a), get_token_sorted_text(r)
        tokens_a = [t for t in ts_a.split() if t] if ts_a else []
        tokens_r = [t for t in ts_r.split() if t] if ts_r else []
        if len(tokens_a) < min_tokens or len(tokens_r) < min_tokens:
            return False
        for text in (a, r):
            if not text:
                continue
            alpha = sum(1 for c in text if c.isalpha())
            if (alpha / len(text)) < min_alpha_ratio:
                return False
        return True

    n_rem, n_add = len(removed), len(added)
    use_emb = (
        bool(th.get("use_embeddings", False))
        and embedding_service
        and getattr(embedding_service, "available", False)
    )
    embed_weight = float(th.get("embedding_weight_indicator", 0.35)) if use_emb else 0.0

    def _norm_for_embed(s: str) -> str:
        c = _canonical_indicator_key(_strip_footnote_markers_from_indicator(s))
        return c if c else (s or " ").strip()[:200]

    embed_matrix_canon = None
    embed_matrix_ts = None
    if use_emb and n_rem > 0 and n_add > 0:
        try:
            import numpy as np
            texts_rem_canon = [_norm_for_embed(removed[i]) for i in range(n_rem)]
            texts_add_canon = [_norm_for_embed(added[j]) for j in range(n_add)]
            embed_matrix_canon = embedding_service.get_pairwise_cosine(texts_rem_canon, texts_add_canon)
            if use_token_sorted:
                texts_rem_ts = [get_token_sorted_text(removed[i]) or " " for i in range(n_rem)]
                texts_add_ts = [get_token_sorted_text(added[j]) or " " for j in range(n_add)]
                embed_matrix_ts = embedding_service.get_pairwise_cosine(texts_rem_ts, texts_add_ts)
        except Exception as e:
            logger.debug("Indicator embedding batch failed: %s", e)
            embed_matrix_canon = None
            embed_matrix_ts = None
    matrix_size = n_rem * n_add
    prefilter_used = matrix_size > _PREFILTER_MATRIX_CAP
    candidate_set: set[tuple[int, int]] | None = None
    if prefilter_used:
        candidate_set = set()
        for i in range(n_rem):
            scored: list[tuple[float, int]] = []
            for j in range(n_add):
                if _length_ratio_ok(added[j], removed[i]) and _token_overlap_ok(added[j], removed[i]):
                    sc = rapidfuzz_fuzz.token_set_ratio(added[j], removed[i])
                    scored.append((sc, j))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, j in scored[:_PREFILTER_TOP_K_PER_REMOVED]:
                candidate_set.add((i, j))

    gated_out = 0
    accepted_scores: list[float] = []

    if linear_sum_assignment is not None:
        import numpy as np
        scores = np.full((n_rem, n_add), -1e9, dtype=np.float64)
        for i in range(n_rem):
            for j in range(n_add):
                if candidate_set is not None and (i, j) not in candidate_set:
                    gated_out += 1
                    continue
                if _length_ratio_ok(added[j], removed[i]) and _token_overlap_ok(added[j], removed[i]):
                    lex_canon, lex_ts, lex = _lex_similarity_both_forms(added[j], removed[i])
                    emb_sim_canon = float(embed_matrix_canon[i, j]) if embed_matrix_canon is not None else 0.0
                    emb_sim_ts = float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
                    emb_sim = max(emb_sim_canon, emb_sim_ts) if (embed_matrix_canon is not None or embed_matrix_ts is not None) else 0.0
                    embed_ok = embed_weight > 0 and _embed_gate_ok(added[j], removed[i], emb_sim)
                    w_eff = embed_weight if embed_ok else 0.0
                    calibrated_embed = (emb_sim * 100.0) if emb_sim >= emb_min else 0.0
                    scores[i, j] = (1.0 - w_eff) * lex + w_eff * calibrated_embed
                else:
                    gated_out += 1
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        renamed_pairs: list[tuple[str, str]] = []
        renamed_indices: list[tuple[int, int]] = []
        used_rem: set[int] = set()
        used_add: set[int] = set()
        for k in range(len(row_ind)):
            i, j = int(row_ind[k]), int(col_ind[k])
            if i >= n_rem or j >= n_add:
                continue
            sc = float(scores[i, j])
            if sc >= min_score_pct:
                renamed_pairs.append((removed[i], added[j]))
                renamed_indices.append((i, j))
                accepted_scores.append(sc)
                used_rem.add(i)
                used_add.add(j)
        added_restant = [added[j] for j in range(n_add) if j not in used_add]
        removed_restant = [removed[i] for i in range(n_rem) if i not in used_rem]

        def _debug_dict() -> dict[str, Any]:
            asc = sorted(accepted_scores) if accepted_scores else []
            unmatched_candidates: list[dict[str, Any]] = []
            for i in range(n_rem):
                if i in used_rem:
                    continue
                r = removed[i]
                cand: list[tuple[str, float]] = []
                for j in range(n_add):
                    if j in used_add:
                        continue
                    sc = float(scores[i, j])
                    if sc > -1e8:
                        cand.append((added[j], sc))
                cand.sort(key=lambda x: x[1], reverse=True)
                unmatched_candidates.append({"removed": r, "top3": cand[:3]})
            rename_pair_debug: list[dict[str, Any]] = []
            for (r, a), (i, j) in zip(renamed_pairs, renamed_indices):
                lc, lts, _ = _lex_similarity_both_forms(a, r)
                ec = float(embed_matrix_canon[i, j]) if embed_matrix_canon is not None else 0.0
                ets = float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
                reasons: list[str] = []
                if not _embed_gate_ok(a, r, max(ec, ets)):
                    reasons.append("embed_gated")
                rename_pair_debug.append({
                    "lex_canonical": round(lc, 2),
                    "lex_token_sorted": round(lts, 2),
                    "embed_canonical": round(ec, 3),
                    "embed_token_sorted": round(ets, 3),
                    "final_score": round(float(scores[i, j]), 2),
                    "reasons": reasons or ["ok"],
                })
            return {
                "gated_out_pairs": gated_out,
                "accepted_renames": len(renamed_pairs),
                "prefilter_used": prefilter_used,
                "rename_pair_debug": rename_pair_debug,
                "score_distribution": {
                    "min": min(asc) if asc else None,
                    "max": max(asc) if asc else None,
                    "mean": sum(asc) / len(asc) if asc else None,
                    "median": asc[len(asc) // 2] if asc else None,
                },
                "unmatched_removed_with_candidates": unmatched_candidates,
            }

        if logger.isEnabledFor(logging.DEBUG) and renamed_pairs:
            for (r, a), (i, j) in zip(renamed_pairs, renamed_indices):
                lc, lts, _ = _lex_similarity_both_forms(a, r)
                ec = float(embed_matrix_canon[i, j]) if embed_matrix_canon is not None else 0.0
                ets = float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
                chosen = float(scores[i, j])
                logger.debug(
                    "indicator_rename removed=%r added=%r lex_canon=%.1f lex_ts=%.1f chosen=%.1f embed_canon=%.3f embed_ts=%.3f",
                    r[:60] if r else "",
                    a[:60] if a else "",
                    lc, lts, chosen, ec, ets,
                )

        return added_restant, removed_restant, renamed_pairs, _debug_dict()
    else:
        # Greedy fallback: process in sorted order, pick best above threshold with same gating
        used_add_f: set[int] = set()
        used_rem_f: set[int] = set()
        renamed_pairs = []
        for i, r in enumerate(removed):
            best_j = -1
            best_score = -1.0
            for j, a in enumerate(added):
                if j in used_add_f:
                    continue
                if not _length_ratio_ok(a, r) or not _token_overlap_ok(a, r):
                    gated_out += 1
                    continue
                _, _, lex_final = _lex_similarity_both_forms(a, r)
                sc = lex_final
                if sc >= min_score_pct and sc > best_score:
                    best_score = sc
                    best_j = j
            if best_j >= 0:
                renamed_pairs.append((r, added[best_j]))
                _, _, lex_f = _lex_similarity_both_forms(added[best_j], r)
                accepted_scores.append(lex_f)
                used_rem_f.add(i)
                used_add_f.add(best_j)
        added_restant = [added[j] for j in range(n_add) if j not in used_add_f]
        removed_restant = [removed[i] for i in range(n_rem) if i not in used_rem_f]
        asc = sorted(accepted_scores) if accepted_scores else []
        debug = {
            "gated_out_pairs": gated_out,
            "accepted_renames": len(renamed_pairs),
            "prefilter_used": prefilter_used,
            "score_distribution": {
                "min": min(asc) if asc else None,
                "max": max(asc) if asc else None,
                "mean": sum(asc) / len(asc) if asc else None,
                "median": asc[len(asc) // 2] if asc else None,
            },
            "unmatched_removed_with_candidates": [
                {"removed": removed[i], "top3": []}
                for i in range(n_rem) if i not in used_rem_f
            ],
        }
        if logger.isEnabledFor(logging.DEBUG) and renamed_pairs and rapidfuzz_fuzz:
            for r, a in renamed_pairs:
                ratio_sc = rapidfuzz_fuzz.ratio(a, r)
                token_sc = rapidfuzz_fuzz.token_set_ratio(a, r)
                chosen = _similarity(a, r)
                logger.debug(
                    "indicator_rename removed=%r added=%r ratio=%.1f token=%.1f chosen_score=%.1f embed_sim=%.3f",
                    r[:60] if r else "",
                    a[:60] if a else "",
                    ratio_sc,
                    token_sc,
                    chosen,
                    0.0 if not use_emb else 0.0,
                )
        return added_restant, removed_restant, renamed_pairs, debug


def _detect_fusion_split(
    added: list[str], removed: list[str], ratio_threshold: float = 0.92
) -> tuple[list[str], list[str], bool]:
    """Merge fusion/split: 1 added = concat of 2 removed (or vice versa).
    Returns (added, removed, had_fusion_split).
    """
    from difflib import SequenceMatcher

    added = list(added)
    removed = list(removed)
    had_fusion_split = False

    def _merge_added_from_removed() -> None:
        nonlocal added, removed, had_fusion_split
        for i, a in enumerate(added[:]):
            a_norm = _canonical_indicator_key(a)
            for j, r1 in enumerate(removed):
                for k, r2 in enumerate(removed):
                    if j >= k:
                        continue
                    concat = _canonical_indicator_key(r1) + " " + _canonical_indicator_key(r2)
                    if not concat.strip():
                        continue
                    ratio = SequenceMatcher(None, a_norm, concat).ratio()
                    if ratio >= ratio_threshold or concat == a_norm:
                        added.remove(a)
                        removed.remove(r2)
                        removed.remove(r1)
                        had_fusion_split = True
                        return
            if not added:
                break

    def _merge_removed_from_added() -> None:
        nonlocal added, removed, had_fusion_split
        for i, r in enumerate(removed[:]):
            r_norm = _canonical_indicator_key(r)
            for j, a1 in enumerate(added):
                for k, a2 in enumerate(added):
                    if j >= k:
                        continue
                    concat = _canonical_indicator_key(a1) + " " + _canonical_indicator_key(a2)
                    if not concat.strip():
                        continue
                    ratio = SequenceMatcher(None, r_norm, concat).ratio()
                    if ratio >= ratio_threshold or concat == r_norm:
                        removed.remove(r)
                        added.remove(a2)
                        added.remove(a1)
                        had_fusion_split = True
                        return
            if not removed:
                break

    changed = True
    while changed:
        changed = False
        before_a, before_r = len(added), len(removed)
        _merge_added_from_removed()
        if len(added) != before_a or len(removed) != before_r:
            changed = True
            continue
        _merge_removed_from_added()
        if len(added) != before_a or len(removed) != before_r:
            changed = True

    return added, removed, had_fusion_split


def _indicator_diff(
    t1: TableArtifact, t2: TableArtifact
) -> tuple[list[str], list[str], bool, dict[str, int]]:
    # Matching/diff use first_column_indicators (clean) only; UI display prefers raw.
    left = [str(item).strip() for item in t1.first_column_indicators if str(item).strip()]
    right = [str(item).strip() for item in t2.first_column_indicators if str(item).strip()]

    def _norm(values: list[str]) -> tuple[dict[str, str], dict[str, int]]:
        mapped: dict[str, str] = {}
        excluded: dict[str, int] = {}
        for value in values:
            kind = _classify_excluded_line(value)
            if kind:
                excluded[kind] = excluded.get(kind, 0) + 1
                continue
            value_clean = _strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key and key not in mapped:
                mapped[key] = value_clean
        return mapped, excluded

    left_map, left_excluded = _norm(left)
    right_map, right_excluded = _norm(right)
    excluded_counts: dict[str, int] = {}
    for k in set(left_excluded) | set(right_excluded):
        excluded_counts[k] = left_excluded.get(k, 0) + right_excluded.get(k, 0)

    added = [right_map[key] for key in right_map.keys() - left_map.keys()]
    removed = [left_map[key] for key in left_map.keys() - right_map.keys()]
    added.sort()
    removed.sort()
    added, removed, had_fusion_split = _detect_fusion_split(added, removed)
    added.sort()
    removed.sort()
    return added, removed, had_fusion_split, excluded_counts


def _fuzzy_pair_added_removed(
    added: list[str],
    removed: list[str],
    bank_code: str | None,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """
    Appariement 1-1 flou entre indicateurs ajoutes et supprimes (reformulations/renommages).

    Retourne (added_restant, removed_restant, paires (removed_text, added_text)).
    Si rapidfuzz est indisponible, retourne (added, removed, []).
    """
    if not added or not removed or rapidfuzz_fuzz is None:
        return added, removed, []

    th = get_matching_thresholds(bank_code=bank_code)
    threshold = float(th.get("indicator_similarity_threshold", 0.88))
    token_threshold = float(th.get("indicator_fuzzy_token_threshold", 0.85))
    threshold_pct = int(threshold * 100)
    token_threshold_pct = int(token_threshold * 100)

    candidates: list[tuple[str, str, float]] = []
    for a in added:
        for r in removed:
            ratio_score = rapidfuzz_fuzz.ratio(a, r)
            token_score = rapidfuzz_fuzz.token_set_ratio(a, r)
            score = max(ratio_score, token_score)
            if score >= threshold_pct or token_score >= token_threshold_pct:
                candidates.append((a, r, float(score)))

    candidates.sort(key=lambda x: x[2], reverse=True)

    used_added: set[str] = set()
    used_removed: set[str] = set()
    renamed_pairs: list[tuple[str, str]] = []
    for a, r, _ in candidates:
        if a not in used_added and r not in used_removed:
            renamed_pairs.append((r, a))
            used_added.add(a)
            used_removed.add(r)

    added_restant = [x for x in added if x not in used_added]
    removed_restant = [x for x in removed if x not in used_removed]
    return added_restant, removed_restant, renamed_pairs


def _empty_result(bank_code: str, year: int, reason: str) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": "t1",
        "quarter_to": "t2",
        "year": year,
        "summary": {
            "tables_t1": 0,
            "tables_t2": 0,
            "tables_matched": 0,
            "tables_added": 0,
            "tables_removed": 0,
            "rescued_matches_count": 0,
            "split_merge_rescues_count": 0,
            "total_added_indicators": 0,
            "total_removed_indicators": 0,
            "total_renamed_indicators": 0,
            "status_counts": {
                "stable": 0,
                "modifie": 0,
                "renommage_probable": 0,
                "incertain": 0,
                "needs_review": 0,
                "structure_change": 0,
                "ajoute": 0,
                "supprime": 0,
            },
        },
        "table_comparisons": [],
        "tables_added": [],
        "tables_removed": [],
        "meta": {
            "generated_at": now,
            "provenance": "comparison_runner",
            "source_format": "empty",
            "error": reason,
            "executive_summary": {
                "content": "Aucun resultat produit. " + reason,
            },
            "embedding_debug": {
                "embedding_enabled": False,
                "embedding_table_used": False,
                "embedding_indicator_used": False,
                "embedding_api_calls": 0,
                "embedding_cache_hits": 0,
                "embedding_batch_sizes": [],
                "embedding_errors": 0,
                "config_use_embeddings": False,
                "fallback_vision_used": False,
                "hungarian_table": False,
                "hungarian_indicator": True,
                "table_pair_count": 0,
                "indicator_rename_count": 0,
                "table_pair_debug": [],
                "rename_pair_debug": [],
            },
        },
    }


def run_comparison_with_sections(
    *,
    pdf_path_t1: str,
    pdf_path_t2: str,
    bank_code: str,
    sections_t1: list[dict[str, Any]] | None,
    sections_t2: list[dict[str, Any]] | None,
    use_genai: bool = False,
    api_key: str | None = None,
    generate_visual_proofs: bool = False,
    use_vision_fallback: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end comparison used by the Dash Analyze callback."""
    del use_genai, generate_visual_proofs  # kept for backward-compatible signature

    year = _infer_year(pdf_path_t1, pdf_path_t2)
    ranges_t1 = _normalize_ranges(sections_t1)
    ranges_t2 = _normalize_ranges(sections_t2)

    if not ranges_t1 or not ranges_t2:
        return _empty_result(bank_code, year, "Aucune section valide fournie.")

    try:
        tables_t1 = _extract_tables(
            pdf_path=pdf_path_t1,
            bank_code=bank_code,
            quarter="t1",
            year=year,
            section_ranges=ranges_t1,
            use_vision_fallback=use_vision_fallback,
            api_key=api_key,
        )
        tables_t2 = _extract_tables(
            pdf_path=pdf_path_t2,
            bank_code=bank_code,
            quarter="t2",
            year=year,
            section_ranges=ranges_t2,
            use_vision_fallback=use_vision_fallback,
            api_key=api_key,
        )
    except Exception as exc:
        return _empty_result(bank_code, year, f"Extraction impossible: {exc}")

    if not tables_t1 and not tables_t2:
        return _empty_result(bank_code, year, "Aucun tableau extrait depuis les sections selectionnees.")

    try:
        from vigilance.config import get_matching_thresholds
        cfg = get_matching_thresholds(bank_code=bank_code) or {}
        algorithm_used = "hungarian" if cfg.get("use_hungarian_matching") else "greedy"
    except Exception:
        cfg = {}
        algorithm_used = "greedy"

    raw_tables_t1_count = len(tables_t1)
    raw_tables_t2_count = len(tables_t2)
    fragment_merges_t1: list[dict[str, Any]] = []
    fragment_merges_t2: list[dict[str, Any]] = []
    if bool(cfg.get("matching_v2_enabled", True)):
        try:
            merge_score_min = float(cfg.get("merge_fragment_score_min", 0.85) or 0.85)
        except (TypeError, ValueError):
            merge_score_min = 0.85
        tables_t1, fragment_merges_t1 = merge_table_fragments(
            tables_t1,
            merge_score_min=merge_score_min,
        )
        tables_t2, fragment_merges_t2 = merge_table_fragments(
            tables_t2,
            merge_score_min=merge_score_min,
        )

    embedding_service = None
    if cfg.get("use_embeddings") and api_key:
        try:
            from vigilance.embedding import EmbeddingService
            embedding_service = EmbeddingService(
                api_key=api_key,
                model=str(cfg.get("embedding_model", "text-embedding-3-small")),
            )
        except Exception as e:
            logger.warning("EmbeddingService init failed: %s", e)
            embedding_service = None

    strict = run_strict_intra_section_compare(
        tables_t1=tables_t1,
        tables_t2=tables_t2,
        bank_code=bank_code,
        embedding_service=embedding_service,
    )

    t1_by_uid = {_section_uid(table): table for table in tables_t1}
    t2_by_uid = {_section_uid(table): table for table in tables_t2}

    comparisons: list[dict[str, Any]] = []
    table_pair_embed_debug: list[dict[str, Any]] = []
    all_rename_pair_debug: list[dict[str, Any]] = []
    for pair in strict.get("pairs", []):
        t1_uid = str(pair.get("t1_uid", ""))
        t2_uid = str(pair.get("t2_uid", ""))
        table_t1 = t1_by_uid.get(t1_uid)
        table_t2 = t2_by_uid.get(t2_uid)
        if table_t1 is None or table_t2 is None:
            continue

        added, removed, had_fusion_split, excluded_counts = _indicator_diff(table_t1, table_t2)
        use_hungarian = cfg.get("indicator_hungarian_enabled", True)
        if use_hungarian:
            added, removed, renamed_pairs, indicator_debug = _hungarian_pair_added_removed(
                removed, added, th=cfg, embedding_service=embedding_service
            )
            if indicator_debug:
                rpd = indicator_debug.get("rename_pair_debug") or []
                for e in rpd:
                    e_with_ctx = dict(e)
                    e_with_ctx["table_id_t1"] = table_t1.table_id
                    e_with_ctx["table_id_t2"] = table_t2.table_id
                    all_rename_pair_debug.append(e_with_ctx)
            if indicator_debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "indicator_pairing %s:%s gated=%s renames=%s scores=%s",
                    table_t1.section, table_t1.table_id,
                    indicator_debug.get("gated_out_pairs"),
                    indicator_debug.get("accepted_renames"),
                    indicator_debug.get("score_distribution"),
                )
        else:
            added, removed, renamed_pairs = _fuzzy_pair_added_removed(added, removed, bank_code)
        renamed_indicators = [{"from": r, "to": a} for (r, a) in renamed_pairs]

        table_pair_embed_debug.append({
            "t1_uid": t1_uid,
            "t2_uid": t2_uid,
            "embed_sim_canonical": round(float(pair.get("embed_sim_canon", 0) or 0), 3),
            "embed_sim_token_sorted": round(float(pair.get("embed_sim_token_sorted", 0) or 0), 3),
            "gating_decision": str(pair.get("table_fp_gating") or ""),
            "fingerprint_token_count": int(pair.get("fingerprint_token_count", 0) or 0),
        })
        rescue_type = pair.get("rescue_type")
        match_decision_level = str(pair.get("decision_level") or "match")
        structure_change_detected = bool(had_fusion_split or rescue_type == "split_merge_rescue")
        if structure_change_detected:
            table_status = "structure_change"
        else:
            table_status = "modifie" if (added or removed or renamed_indicators) else "stable"

        comparisons.append(
            {
                "table_id_t1": table_t1.table_id,
                "table_id_t2": table_t2.table_id,
                "title_t1": table_t1.title or "",
                "title_t2": table_t2.title or "",
                "page_t1": table_t1.page_pdf,
                "page_t2": table_t2.page_pdf,
                "section": table_t1.section or table_t2.section,
                "match_score": float(pair.get("score", 0.0) or 0.0),
                "match_quality": "high" if float(pair.get("score", 0.0) or 0.0) >= 0.7 else "medium",
                "match_reason": pair.get("reason", ""),
                "match_decision_level": match_decision_level,
                "rescue_type": rescue_type,
                "added_indicators": added,
                "removed_indicators": removed,
                "renamed_indicators": renamed_indicators,
                "renamed_probable_indicators": [],
                "all_indicators_t1": _all_indicators_value_clean_ordered(table_t1),
                "all_indicators_t2": _all_indicators_value_clean_ordered(table_t2),
                "bbox_t1": _normalize_bbox_ltrb_norm(getattr(table_t1, "bbox", None)),
                "bbox_t2": _normalize_bbox_ltrb_norm(getattr(table_t2, "bbox", None)),
                "indicator_decisions": [],
                "review_reasons": [],
                "uncertain_diff": False,
                "structure_change_detected": structure_change_detected,
                "table_status": table_status,
                "counts": {
                    "added": len(added),
                    "removed": len(removed),
                    "renamed": len(renamed_indicators),
                    "renamed_probable": 0,
                    "excluded_totals": excluded_counts.get("total", 0),
                    "excluded_units": excluded_counts.get("unit", 0),
                    "excluded_dates": excluded_counts.get("date", 0),
                },
                "source_method_t1": table_t1.extraction_method,
                "source_method_t2": table_t2.extraction_method,
                "quality_flags_t1": [],
                "quality_flags_t2": [],
                "source_pdf_t1": table_t1.pdf_path or "",
                "source_pdf_t2": table_t2.pdf_path or "",
            }
        )

    def _source_method(uid: str, by_uid: dict) -> str:
        t = by_uid.get(uid)
        return (getattr(t, "extraction_method", None) or "docling") if t else "docling"

    tables_added = []
    for item in strict.get("added_tables", []):
        t2_t = t2_by_uid.get(str(item.get("t2_uid", "")))
        tables_added.append({
            "table_status": "ajoute",
            "table_id": str(item.get("t2_table_id", "")),
            "title": item.get("title_t2", ""),
            "page": item.get("page_t2"),
            "section": item.get("section", ""),
            "source_method": _source_method(str(item.get("t2_uid", "")), t2_by_uid),
            "quality_flags": [],
            "indicators": list(item.get("first_column_indicators", []) or []),
            "first_column_indicators_raw": list(
                item.get("first_column_indicators_raw")
                or (
                    getattr(t2_t, "first_column_indicators_raw", None)
                    or []
                )
            ),
            "all_indicators_t1": [],
            "all_indicators_t2": _all_indicators_value_clean_ordered(t2_t) if t2_t else [],
            "bbox_t1": None,
            "bbox_t2": _normalize_bbox_ltrb_norm(getattr(t2_t, "bbox", None)) if t2_t else None,
        })
    tables_removed = []
    for item in strict.get("removed_tables", []):
        t1_t = t1_by_uid.get(str(item.get("t1_uid", "")))
        tables_removed.append({
            "table_status": "supprime",
            "table_id": str(item.get("t1_table_id", "")),
            "title": item.get("title_t1", ""),
            "page": item.get("page_t1"),
            "section": item.get("section", ""),
            "source_method": _source_method(str(item.get("t1_uid", "")), t1_by_uid),
            "quality_flags": [],
            "indicators": list(item.get("first_column_indicators", []) or []),
            "first_column_indicators_raw": list(
                item.get("first_column_indicators_raw")
                or (
                    getattr(t1_t, "first_column_indicators_raw", None)
                    or []
                )
            ),
            "all_indicators_t1": _all_indicators_value_clean_ordered(t1_t) if t1_t else [],
            "all_indicators_t2": [],
            "bbox_t1": _normalize_bbox_ltrb_norm(getattr(t1_t, "bbox", None)) if t1_t else None,
            "bbox_t2": None,
        })

    total_added = sum(len(c.get("added_indicators", [])) for c in comparisons)
    total_removed = sum(len(c.get("removed_indicators", [])) for c in comparisons)
    total_renamed = sum(len(c.get("renamed_indicators", [])) for c in comparisons)

    status_counts = {
        "stable": sum(1 for c in comparisons if c.get("table_status") == "stable"),
        "modifie": sum(1 for c in comparisons if c.get("table_status") == "modifie"),
        "renommage_probable": 0,
        "incertain": 0,
        "needs_review": 0,
        "structure_change": sum(1 for c in comparisons if c.get("table_status") == "structure_change"),
        "ajoute": len(tables_added),
        "supprime": len(tables_removed),
    }

    now = datetime.now().isoformat(timespec="seconds")
    summary_text = (
        f"{len(comparisons)} tableaux apparies, {total_added} ajouts d'indicateurs, "
        f"{total_removed} suppressions, {len(tables_added)} tableaux ajoutes, {len(tables_removed)} supprimes."
    )

    result: dict[str, Any] = {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": "t1",
        "quarter_to": "t2",
        "year": year,
        "summary": {
            "tables_t1": len(tables_t1),
            "tables_t2": len(tables_t2),
            "tables_matched": len(comparisons),
            "tables_added": len(tables_added),
            "tables_removed": len(tables_removed),
            "rescued_matches_count": int(strict.get("rescued_matches_count", 0) or 0),
            "split_merge_rescues_count": int(strict.get("split_merge_rescues_count", 0) or 0),
            "total_added_indicators": total_added,
            "total_removed_indicators": total_removed,
            "total_renamed_indicators": total_renamed,
            "status_counts": status_counts,
        },
        "displaced_indicators": [],
        "table_comparisons": comparisons,
        "tables_added": tables_added,
        "tables_removed": tables_removed,
        "probable_pairs": list(strict.get("probable_pairs", [])),
        "debug_unmatched_candidates": list(strict.get("debug_unmatched_candidates", [])),
        "meta": {
            "generated_at": now,
            "provenance": "comparison_runner",
            "source_format": "strict_intra_section",
            "algorithm_used": algorithm_used,
            "raw_tables_t1": raw_tables_t1_count,
            "raw_tables_t2": raw_tables_t2_count,
            "fragment_merges_t1_count": len(fragment_merges_t1),
            "fragment_merges_t2_count": len(fragment_merges_t2),
            "fragment_merges_t1": fragment_merges_t1,
            "fragment_merges_t2": fragment_merges_t2,
            "executive_summary": {"content": summary_text},
            "embedding_debug": {
                "embedding_enabled": bool(cfg.get("use_embeddings", False)),
                "embedding_table_used": (
                    embedding_service is not None
                    and (embedding_service.stats.api_calls > 0 or embedding_service.stats.cache_hits > 0)
                    and cfg.get("use_embeddings")
                ),
                "embedding_indicator_used": (
                    embedding_service is not None
                    and cfg.get("use_embeddings")
                ),
                "embedding_api_calls": embedding_service.stats.api_calls if embedding_service else 0,
                "embedding_cache_hits": embedding_service.stats.cache_hits if embedding_service else 0,
                "embedding_batch_sizes": list(embedding_service.stats.batch_sizes) if embedding_service else [],
                "embedding_errors": embedding_service.stats.errors if embedding_service else 0,
                "config_use_embeddings": bool(cfg.get("use_embeddings", False)),
                "fallback_vision_used": use_vision_fallback,
                "hungarian_table": cfg.get("use_hungarian_matching", False),
                "hungarian_indicator": cfg.get("indicator_hungarian_enabled", True),
                "table_pair_count": len(comparisons),
                "indicator_rename_count": total_renamed,
                "table_pair_debug": table_pair_embed_debug,
                "rename_pair_debug": all_rename_pair_debug,
            },
        },
    }

    INDICATOR_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = INDICATOR_COMPARISON_DIR / f"{bank_code}_t1_vs_t2_{year}_{stamp}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["meta"]["compare_path"] = str(out_path)

    return result
