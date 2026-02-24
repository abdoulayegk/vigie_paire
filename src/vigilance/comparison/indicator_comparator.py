"""
Comparaison des indicateurs (premiere colonne) a partir des JSON par table.

Objectif:
- Matcher tableaux T1 vs T2
- Detecter ajouts/suppressions/renommages d'indicateurs
- Ne pas "sauter" de tableaux: tous les tableaux sont comptes
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except Exception:  # pragma: no cover - optional dependency
    rapidfuzz_fuzz = None

from vigilance.comparison.displacement_detector import (
    AddedItem,
    RemovedItem,
    detect_cross_table_displacements,
)
from vigilance.comparison.scoring_engine import compute_candidate_score
from vigilance.config import get_matching_thresholds
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)
UNCERTAINTY_FUSED_FLAG = "fused_first_column"
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}

try:
    from vigilance.extraction.section_taxonomy import canonicalize_section
except Exception:
    canonicalize_section = None


def _canonical_section(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if canonicalize_section is None:
        return raw.lower()
    try:
        return canonicalize_section(raw)
    except Exception:
        return raw.lower()


def _sections_strict_match(left: str | None, right: str | None) -> bool:
    left_norm = _canonical_section(left)
    right_norm = _canonical_section(right)
    if left_norm in UNKNOWN_SECTIONS or right_norm in UNKNOWN_SECTIONS:
        return False
    return left_norm == right_norm

_MONTH_NAMES = (
    "janvier",
    "fevrier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
    "décembre",
)

_META_LINE_PATTERNS = [
    re.compile(r"^en (millions?|milliards?)"),
    re.compile(r"^\(en (millions?|milliards?)"),
    re.compile(r"^pour le trimestre clos"),
    re.compile(r"^pour le trimestre clos le"),
    re.compile(r"^pour la periode"),
    re.compile(r"^pour la periode close"),
    re.compile(r"^trimestre termine"),
    re.compile(r"^trimestre termine le"),
    re.compile(r"^trimestre\s+clos\s+le"),
    re.compile(r"^au \d{1,2} [a-z]+ \d{4}$"),
    re.compile(r"^\d{1,2} [a-z]+ \d{4}$"),
]

_GENERIC_HEADER_CANONICAL = {
    "actif",
    "actifs",
    "total actif",
    "total actifs",
    "passifs",
    "passif",
    "total passif",
    "total passifs",
    "passif et capitaux propres",
    "passifs et capitaux propres",
    "capitaux propres",
    "engagements hors bilan",
    "total",
    "tableau principal",
    "canada",
    "etranger",
}

_SOFT_STOPWORDS = {"de", "des", "du", "la", "le", "les", "au", "aux", "d", "l"}
_GENERIC_HEADER_TOKENS = (
    {token for header in _GENERIC_HEADER_CANONICAL for token in header.split() if token}
    | _SOFT_STOPWORDS
    | {"et", "ou", "dune"}
)
_HEADER_SHORT_ALLOWLIST = {"depots", "actifs", "passifs", "passif", "capitaux", "total"}
_LOWERCASE_NUMERIC_KEEP_PREFIXES = {"augmentation", "diminution", "hausse", "baisse", "variation"}

# Labels generiques courts qui ne doivent pas etre apparies entre eux sauf similarite tres elevee.
_SHORT_GENERIC_LABELS = frozenset(
    {
        "total",
        "sous-total",
        "sous total",
        "autres",
        "net",
        "brut",
        "divers",
        "ecart",
        "solde",
        "variation",
        "ajustements",
    }
)
_SHORT_LABEL_MAX_LEN = 6
_MONTH_NAMES_ASCII = (
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
)
_ECHEANT_LE_TEXTUAL_DATE_PATTERN = re.compile(
    rf"\b(echeant le)\s+\d{{1,2}}\s+(?:{'|'.join(_MONTH_NAMES_ASCII)})(?:\s+\d{{4}})?\b"
)
_ECHEANT_LE_NUMERIC_DATE_PATTERN = re.compile(r"\b(echeant le)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_STRICT_STRUCTURAL_LINE_PATTERNS = (
    re.compile(r"^actif(?:s)?$"),
    re.compile(r"^passif(?:s)?$"),
    re.compile(r"^total actif(?:s)?$"),
    re.compile(r"^total passif(?:s)?$"),
    re.compile(r"^tableau principal$"),
)
_GROUP_PATTERNS = {
    "actifs": (
        re.compile(r"\bactif(?:s)?\b"),
        re.compile(r"\basset(?:s)?\b"),
    ),
    "passifs": (
        re.compile(r"\bpassif(?:s)?\b"),
        re.compile(r"\bliabilit(?:y|ies)\b"),
    ),
    "capitaux_propres": (
        re.compile(r"\bcapitaux?\s+propres?\b"),
        re.compile(r"\bfonds?\s+propres?\b"),
        re.compile(r"\bequit(?:y|ies)\b"),
    ),
    "cet1": (
        re.compile(r"\bcet\s*1\b"),
        re.compile(r"\bcommon equity tier 1\b"),
    ),
    "at1": (
        re.compile(r"\bat\s*1\b"),
        re.compile(r"\badditional tier 1\b"),
    ),
    "tier2": (re.compile(r"\btier\s*2\b"),),
}


@dataclass
class IndicatorItem:
    text: str
    text_norm: str
    indent_level: int = 0


@dataclass
class IndicatorTable:
    table_id: str
    title: str
    page: int
    section: str
    quarter: str
    year: int
    indicators: list[IndicatorItem] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    context_before: str = ""
    context_after: str = ""
    source_method: str = "vector"
    quality_flags: list[str] = field(default_factory=list)
    source_pdf: str = ""
    bbox: tuple[float, float, float, float] | None = None
    row_bboxes: list[tuple[str, float, float]] = field(default_factory=list)
    bbox_source: str = "none"

    @property
    def indicator_norm_set(self) -> set[str]:
        values: set[str] = set()
        for item in self.indicators:
            canonical = _canonical_indicator_text(item.text)
            if canonical and not _is_meta_indicator_line(item.text):
                values.add(canonical)
        return values

    def indicator_map(self) -> dict[str, IndicatorItem]:
        """Map normalise -> item (premiere occurrence)."""
        mapping = {}
        for item in self.indicators:
            if item.text_norm and item.text_norm not in mapping:
                mapping[item.text_norm] = item
        return mapping


@dataclass
class TableMatch:
    table_t1: IndicatorTable
    table_t2: IndicatorTable
    score: float
    match_reason: str


def load_indicator_tables(
    directory: Path,
    bank_code: str,
    quarter: str,
    year: int,
    section: str | None = None,
) -> list[IndicatorTable]:
    def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
        if not value:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
            except (TypeError, ValueError):
                return None
        if isinstance(value, dict):
            keys = ("x0", "y0", "x1", "y1")
            if all(k in value for k in keys):
                try:
                    return (
                        float(value["x0"]),
                        float(value["y0"]),
                        float(value["x1"]),
                        float(value["y1"]),
                    )
                except (TypeError, ValueError):
                    return None
        return None

    tables: list[IndicatorTable] = []
    if not directory.exists():
        return tables

    for file in directory.glob("*.json"):
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if data.get("bank_code") != bank_code:
            continue
        if data.get("quarter") != quarter:
            continue
        if data.get("year") != year:
            continue
        if section and data.get("section") != section:
            continue

        indicators = []
        for row in data.get("indicators", []):
            text = row.get("text") or ""
            text_norm = TextNormalizer.normalize_indicator(text)
            indicators.append(
                IndicatorItem(
                    text=text,
                    text_norm=text_norm,
                    indent_level=row.get("indent_level", 0),
                )
            )

        row_bboxes: list[tuple[str, float, float]] = []
        for item in data.get("row_bboxes") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    row_bboxes.append((str(item[0]), float(item[1]), float(item[2])))
                except (TypeError, ValueError):
                    pass

        headers = data.get("headers") or []
        if isinstance(headers, list):
            headers = [str(h) for h in headers if h is not None]
        else:
            headers = []

        tables.append(
            IndicatorTable(
                table_id=str(data.get("table_id", "")),
                title=str(data.get("table_title", "")),
                page=int(data.get("page", 0) or 0),
                section=str(data.get("section", "")),
                quarter=str(data.get("quarter", "")),
                year=int(data.get("year", 0) or 0),
                indicators=indicators,
                headers=headers,
                context_before=str(data.get("context_before", "") or ""),
                context_after=str(data.get("context_after", "") or ""),
                source_method=str(data.get("source_method", "vector")),
                quality_flags=list(data.get("quality_flags") or []),
                source_pdf=str(data.get("source_pdf", "")),
                bbox=_parse_bbox(data.get("bbox")),
                row_bboxes=row_bboxes,
                bbox_source=str(data.get("bbox_source", "none")),
            )
        )

    return tables


def match_tables(
    tables_t1: list[IndicatorTable],
    tables_t2: list[IndicatorTable],
    score_threshold: float | None = None,
) -> tuple[list[TableMatch], list[IndicatorTable], list[IndicatorTable]]:
    """Matcher les tableaux par id garde-fou, puis par score multi-signal."""
    thresholds = get_matching_thresholds()
    if score_threshold is None:
        score_threshold = float(
            thresholds.get("table_match_score", thresholds.get("minimum_match", 0.58))
        )
    low_score_uncertain = float(thresholds.get("indicator_low_score_uncertain", 0.60))
    reject_title_similarity = float(thresholds.get("indicator_match_reject_title_similarity", 0.70))
    reject_jaccard = float(thresholds.get("indicator_match_reject_jaccard", 0.35))
    matches: list[TableMatch] = []
    unmatched_t1 = tables_t1[:]
    unmatched_t2 = tables_t2[:]

    # 1) Matching direct par table_id (si unique) AVEC garde-fou de coherence
    t2_by_id: dict[str, IndicatorTable] = {}
    for t2 in unmatched_t2:
        if t2.table_id:
            t2_by_id.setdefault(t2.table_id, t2)

    matched_t1 = set()
    matched_t2 = set()

    for t1 in list(unmatched_t1):
        if _canonical_section(t1.section) in UNKNOWN_SECTIONS:
            continue
        if t1.table_id and t1.table_id in t2_by_id:
            t2 = t2_by_id[t1.table_id]
            if not _sections_strict_match(t1.section, t2.section):
                continue
            signals = _compute_table_match_signals(t1, t2)
            if _is_safe_id_match(t1, t2, signals):
                score = _compute_table_match_score(t1, t2, precomputed=signals)
                matches.append(TableMatch(table_t1=t1, table_t2=t2, score=score, match_reason="id"))
                matched_t1.add(id(t1))
                matched_t2.add(id(t2))
            else:
                logger.info(
                    "ID match rejete (coherence faible): "
                    f"id={t1.table_id} section={t1.section}/{t2.section} "
                    f"jaccard={signals['jaccard']:.3f} title={signals['title_similarity']:.3f}"
                )

    unmatched_t1 = [t for t in unmatched_t1 if id(t) not in matched_t1]
    unmatched_t2 = [t for t in unmatched_t2 if id(t) not in matched_t2]

    # 2) Matching par score (plus de skip section: penalite via section_match dans le score)
    scored_pairs: list[tuple[float, IndicatorTable, IndicatorTable]] = []
    for t1 in unmatched_t1:
        if _canonical_section(t1.section) in UNKNOWN_SECTIONS:
            continue
        for t2 in unmatched_t2:
            if not _sections_strict_match(t1.section, t2.section):
                continue
            score = _compute_table_match_score(t1, t2)
            scored_pairs.append((score, t1, t2))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    title_similarity_min = 0.75
    try:
        th = get_matching_thresholds()
        title_similarity_min = float(th.get("title_similarity_min", 0.75))
    except Exception:
        pass

    used_t1 = set()
    used_t2 = set()
    for score, t1, t2 in scored_pairs:
        if score < score_threshold:
            break
        if id(t1) in used_t1 or id(t2) in used_t2:
            continue
        signals = _compute_table_match_signals(t1, t2)
        title1 = (t1.title or "").strip()
        title2 = (t2.title or "").strip()
        if title1 and title2 and signals["title_similarity"] < title_similarity_min:
            continue
        if (
            score < low_score_uncertain
            and signals["title_similarity"] < reject_title_similarity
            and signals["jaccard"] < reject_jaccard
        ):
            continue
        matches.append(TableMatch(table_t1=t1, table_t2=t2, score=score, match_reason="score"))
        used_t1.add(id(t1))
        used_t2.add(id(t2))

    remaining_t1 = [t for t in unmatched_t1 if id(t) not in used_t1]
    remaining_t2 = [t for t in unmatched_t2 if id(t) not in used_t2]

    # 3) Rattrapage des faux non-matchs (split/merge de lignes ou fortes similarites locales)
    rescue_matches, remaining_t1, remaining_t2 = _rescue_unmatched_tables(
        remaining_t1, remaining_t2
    )
    matches.extend(rescue_matches)

    return matches, remaining_t1, remaining_t2


def compare_indicators(
    table_t1: IndicatorTable, table_t2: IndicatorTable, rename_threshold: float | None = None
) -> dict:
    """Comparer les indicateurs d'un tableau matché avec cascade exact/near/ambigu."""
    thresholds = get_matching_thresholds()
    if rename_threshold is None:
        rename_threshold = float(thresholds.get("indicator_rename_threshold", 0.86))
    near_exact_threshold = float(thresholds.get("indicator_exist_near_exact_threshold", 0.95))
    llm_band_min = float(thresholds.get("indicator_exist_llm_band_min", 0.85))
    llm_band_max = float(thresholds.get("indicator_exist_llm_band_max", near_exact_threshold))
    length_ratio_min = float(thresholds.get("indicator_length_ratio_min", 0.75))
    group_strict = bool(thresholds.get("indicator_group_strict", True))
    llm_same_context_only = bool(thresholds.get("llm_ambiguity_same_context_only", True))
    enable_llm_ambiguity = bool(thresholds.get("enable_llm_ambiguity_resolver", False))

    prepared_t1, ignored_t1 = _prepare_indicator_items(table_t1.indicators, table_t1.page)
    prepared_t2, ignored_t2 = _prepare_indicator_items(table_t2.indicators, table_t2.page)
    indicator_decisions: list[dict[str, Any]] = []

    exact_pairs, remaining_t1, remaining_t2 = _consume_exact_matches(
        prepared_t1, prepared_t2, group_strict=group_strict
    )
    for left, right in exact_pairs:
        indicator_decisions.append(
            _build_indicator_decision(
                left=left,
                right=right,
                decision="exists_exact",
                reason="match_exact_normalized",
                section=table_t1.section or table_t2.section,
                table_id_t1=table_t1.table_id,
                table_id_t2=table_t2.table_id,
                page_t1=table_t1.page,
                page_t2=table_t2.page,
            )
        )

    near_pairs, remaining_t1, remaining_t2 = _consume_near_exact_matches(
        remaining_t1=remaining_t1,
        remaining_t2=remaining_t2,
        threshold=near_exact_threshold,
        length_ratio_min=length_ratio_min,
        group_strict=group_strict,
    )
    for left, right, score_meta in near_pairs:
        indicator_decisions.append(
            _build_indicator_decision(
                left=left,
                right=right,
                decision="exists_near",
                reason="match_near_exact_strict",
                section=table_t1.section or table_t2.section,
                table_id_t1=table_t1.table_id,
                table_id_t2=table_t2.table_id,
                page_t1=table_t1.page,
                page_t2=table_t2.page,
                score_meta=score_meta,
            )
        )

    split_merge_t1, split_merge_t2 = _detect_split_merge_collisions(
        entries_t1=remaining_t1,
        entries_t2=remaining_t2,
        band_min=llm_band_min,
        band_max=llm_band_max,
        length_ratio_min=length_ratio_min,
        group_strict=group_strict,
    )
    if split_merge_t1:
        for idx in sorted(split_merge_t1):
            left = remaining_t1[idx]
            indicator_decisions.append(
                _build_indicator_decision(
                    left=left,
                    right=None,
                    decision="split_merge",
                    reason="multiple_ambiguous_candidates",
                    section=table_t1.section or table_t2.section,
                    table_id_t1=table_t1.table_id,
                    table_id_t2=table_t2.table_id,
                    page_t1=table_t1.page,
                    page_t2=table_t2.page,
                )
            )

    residual_t1 = [entry for idx, entry in enumerate(remaining_t1) if idx not in split_merge_t1]
    residual_t2 = [entry for idx, entry in enumerate(remaining_t2) if idx not in split_merge_t2]

    rename_probable_threshold = float(thresholds.get("indicator_rename_probable_threshold", 0.90))
    renamed, used_t1, used_t2 = _match_renames_with_context(
        residual_t1=residual_t1,
        residual_t2=residual_t2,
        threshold=rename_threshold,
        length_ratio_min=length_ratio_min,
        group_strict=group_strict,
        all_prepared_t1=prepared_t1,
        all_prepared_t2=prepared_t2,
    )
    renamed_probable_list: list[dict[str, Any]] = []
    for rename in renamed:
        text_score = rename["score_meta"]["score"]
        if text_score >= rename_probable_threshold:
            decision = "renamed"
            reason = "rename_threshold_reached"
        else:
            decision = "renamed_probable"
            reason = "rename_uncertain"
            renamed_probable_list.append(rename)
        indicator_decisions.append(
            _build_indicator_decision(
                left=rename["left"],
                right=rename["right"],
                decision=decision,
                reason=reason,
                section=table_t1.section or table_t2.section,
                table_id_t1=table_t1.table_id,
                table_id_t2=table_t2.table_id,
                page_t1=table_t1.page,
                page_t2=table_t2.page,
                score_meta=rename["score_meta"],
            )
        )

    ambiguous_t1: set[int] = set()
    ambiguous_used_t2: set[int] = set()
    for i, left in enumerate(residual_t1):
        if i in used_t1:
            continue
        best = _best_compatible_candidate(
            left=left,
            candidates=residual_t2,
            used_t2=used_t2 | ambiguous_used_t2,
            group_strict=group_strict,
            length_ratio_min=length_ratio_min,
        )
        if best is None:
            continue
        j, score_meta = best
        score = score_meta["score"]
        if llm_band_min <= score < llm_band_max:
            decision = "ambiguous"
            reason = "ambiguous_band"
            right = residual_t2[j]
            if enable_llm_ambiguity and (
                not llm_same_context_only or left["group"] == right["group"]
            ):
                llm_exists, llm_reason = _resolve_ambiguous_with_llm(left, right)
                reason = llm_reason
                if llm_exists:
                    decision = "exists_near"
                    ambiguous_used_t2.add(j)
                else:
                    ambiguous_t1.add(i)
            else:
                ambiguous_t1.add(i)
                if not enable_llm_ambiguity:
                    reason = "ambiguous_llm_disabled"
            indicator_decisions.append(
                _build_indicator_decision(
                    left=left,
                    right=right,
                    decision=decision,
                    reason=reason,
                    section=table_t1.section or table_t2.section,
                    table_id_t1=table_t1.table_id,
                    table_id_t2=table_t2.table_id,
                    page_t1=table_t1.page,
                    page_t2=table_t2.page,
                    score_meta=score_meta,
                )
            )

    removed_items = [
        entry
        for idx, entry in enumerate(residual_t1)
        if idx not in used_t1 and idx not in ambiguous_t1
    ]
    added_items = [
        entry
        for idx, entry in enumerate(residual_t2)
        if idx not in used_t2 and idx not in ambiguous_used_t2
    ]

    for entry in removed_items:
        indicator_decisions.append(
            _build_indicator_decision(
                left=entry,
                right=None,
                decision="removed",
                reason="no_compatible_match",
                section=table_t1.section or table_t2.section,
                table_id_t1=table_t1.table_id,
                table_id_t2=table_t2.table_id,
                page_t1=table_t1.page,
                page_t2=table_t2.page,
            )
        )
    for entry in added_items:
        indicator_decisions.append(
            _build_indicator_decision(
                left=None,
                right=entry,
                decision="added",
                reason="new_in_t2",
                section=table_t1.section or table_t2.section,
                table_id_t1=table_t1.table_id,
                table_id_t2=table_t2.table_id,
                page_t1=table_t1.page,
                page_t2=table_t2.page,
            )
        )

    review_reasons: list[str] = []
    if renamed_probable_list:
        review_reasons.append("rename_uncertain")
    if group_strict and any(
        (entry.get("group") or "unknown") == "unknown" for entry in prepared_t1 + prepared_t2
    ):
        review_reasons.append("group_unknown")
    if ambiguous_t1 and not enable_llm_ambiguity:
        review_reasons.append("ambiguous_llm_disabled")

    return {
        "added_indicators": [entry["text"] for entry in added_items],
        "removed_indicators": [entry["text"] for entry in removed_items],
        "ambiguous_indicators": [residual_t1[idx]["text"] for idx in sorted(ambiguous_t1)],
        "renamed_indicators": [
            {
                "from": item["left"]["text"],
                "to": item["right"]["text"],
                "similarity": round(item["score_meta"]["score"], 3),
                "composite_score": round(
                    item["score_meta"].get("composite_score", item["score_meta"]["score"]), 3
                ),
                "status": "confirmed" if item not in renamed_probable_list else "probable",
            }
            for item in renamed
        ],
        "renamed_probable_indicators": [
            {
                "from": item["left"]["text"],
                "to": item["right"]["text"],
                "similarity": round(item["score_meta"]["score"], 3),
                "composite_score": round(
                    item["score_meta"].get("composite_score", item["score_meta"]["score"]), 3
                ),
            }
            for item in renamed_probable_list
        ],
        "indicator_decisions": indicator_decisions,
        "review_reasons": review_reasons,
        "split_merge_ambiguous_count": len(split_merge_t1) + len(split_merge_t2),
        "structure_change_detected": bool(split_merge_t1 or split_merge_t2),
        "comparable_lines": max(1, max(len(prepared_t1), len(prepared_t2))),
        "counts": {
            "added": len(added_items),
            "removed": len(removed_items),
            "renamed": len(renamed) - len(renamed_probable_list),
            "renamed_probable": len(renamed_probable_list),
        },
        "ignored_non_indicator_lines_t1": ignored_t1,
        "ignored_non_indicator_lines_t2": ignored_t2,
        "suppressed_split_merge_artifacts": len(split_merge_t1) + len(split_merge_t2),
    }


def compare_indicator_exports(
    directory: Path,
    bank_code: str,
    quarter_from: str,
    quarter_to: str,
    year: int,
    section: str | None = None,
) -> dict:
    """Comparer deux trimestres d'indicateurs a partir des JSON."""
    thresholds = get_matching_thresholds()
    low_score_uncertain = float(thresholds.get("indicator_low_score_uncertain", 0.60))
    change_rate_review_threshold = float(thresholds.get("table_change_rate_review_threshold", 0.30))
    tables_t1 = load_indicator_tables(directory, bank_code, quarter_from, year, section=section)
    tables_t2 = load_indicator_tables(directory, bank_code, quarter_to, year, section=section)

    matches, removed_tables, added_tables = match_tables(tables_t1, tables_t2)
    removed_pool = _build_indicator_canonical_pool(removed_tables)
    added_pool = _build_indicator_canonical_pool(added_tables)

    comparisons = []
    for match in matches:
        diff = compare_indicators(match.table_t1, match.table_t2)
        suppressed_relocated = 0

        filtered_added: list[str] = []
        for value in diff.get("added_indicators", []):
            canonical = _canonical_indicator_text(value)
            if canonical and canonical in removed_pool:
                suppressed_relocated += 1
                continue
            filtered_added.append(value)

        filtered_removed: list[str] = []
        for value in diff.get("removed_indicators", []):
            canonical = _canonical_indicator_text(value)
            if canonical and canonical in added_pool:
                suppressed_relocated += 1
                continue
            filtered_removed.append(value)

        if suppressed_relocated:
            diff["added_indicators"] = filtered_added
            diff["removed_indicators"] = filtered_removed
            diff["counts"] = {
                "added": len(filtered_added),
                "removed": len(filtered_removed),
                "renamed": diff.get("counts", {}).get("renamed", 0),
                "renamed_probable": diff.get("counts", {}).get("renamed_probable", 0),
            }
            diff["suppressed_relocated_indicators"] = suppressed_relocated

        (
            filtered_added,
            filtered_removed,
            suppressed_header_artifacts,
        ) = _drop_header_containment_artifacts(
            diff.get("added_indicators", []),
            diff.get("removed_indicators", []),
        )
        if suppressed_header_artifacts:
            diff["added_indicators"] = filtered_added
            diff["removed_indicators"] = filtered_removed
            diff["counts"] = {
                "added": len(filtered_added),
                "removed": len(filtered_removed),
                "renamed": diff.get("counts", {}).get("renamed", 0),
                "renamed_probable": diff.get("counts", {}).get("renamed_probable", 0),
            }
            diff["suppressed_header_containment_artifacts"] = suppressed_header_artifacts

        (
            filtered_added,
            filtered_removed,
            suppressed_generic_artifacts,
        ) = _drop_generic_header_artifacts(
            diff.get("added_indicators", []),
            diff.get("removed_indicators", []),
        )
        if suppressed_generic_artifacts:
            diff["added_indicators"] = filtered_added
            diff["removed_indicators"] = filtered_removed
            diff["counts"] = {
                "added": len(filtered_added),
                "removed": len(filtered_removed),
                "renamed": diff.get("counts", {}).get("renamed", 0),
                "renamed_probable": diff.get("counts", {}).get("renamed_probable", 0),
            }
        diff["suppressed_generic_header_artifacts"] = suppressed_generic_artifacts

        fused_t1 = UNCERTAINTY_FUSED_FLAG in set(match.table_t1.quality_flags)
        fused_t2 = UNCERTAINTY_FUSED_FLAG in set(match.table_t2.quality_flags)
        title_missing = (
            not (match.table_t1.title or "").strip() and not (match.table_t2.title or "").strip()
        )
        low_score = match.score < low_score_uncertain
        weak_score_no_title = match.match_reason == "score" and title_missing and match.score < 0.65
        split_merge_noise = int(diff.get("suppressed_split_merge_artifacts", 0)) >= 2
        ambiguous_count = len(diff.get("ambiguous_indicators", []))
        split_merge_ambiguous = int(diff.get("split_merge_ambiguous_count", 0))
        review_reasons = list(diff.get("review_reasons", []))
        change_rate = (
            int(diff["counts"].get("added", 0))
            + int(diff["counts"].get("removed", 0))
            + split_merge_ambiguous
        ) / max(1, int(diff.get("comparable_lines", 1)))
        reliable_table_match = _is_reliable_table_match(match)
        needs_review = False
        if change_rate > change_rate_review_threshold and reliable_table_match:
            needs_review = True
            review_reasons.append("high_change_rate_suspect")
        structure_change = bool(diff.get("structure_change_detected", False))

        uncertain_diff = (
            low_score
            or fused_t1
            or fused_t2
            or weak_score_no_title
            or split_merge_noise
            or needs_review
            or structure_change
            or ambiguous_count > 0
        )
        uncertainty_reasons = []
        if low_score:
            uncertainty_reasons.append("score_faible")
        if fused_t1:
            uncertainty_reasons.append("fusion_detectee_t1")
        if fused_t2:
            uncertainty_reasons.append("fusion_detectee_t2")
        if weak_score_no_title:
            uncertainty_reasons.append("appariement_faible_sans_titre")
        if split_merge_noise:
            uncertainty_reasons.append("split_merge_detecte")
        for reason in review_reasons:
            if reason not in uncertainty_reasons:
                uncertainty_reasons.append(reason)
        if structure_change:
            table_status = "structure_change"
        elif needs_review:
            table_status = "needs_review"
        else:
            table_status = _determine_table_status(
                uncertain_diff=uncertain_diff,
                added=int(diff["counts"].get("added", 0)),
                removed=int(diff["counts"].get("removed", 0)),
                renamed=int(diff["counts"].get("renamed", 0)),
            )
        decision_reason = _aggregate_decision_reasons(diff.get("indicator_decisions", []))

        comparisons.append(
            {
                "table_id_t1": match.table_t1.table_id,
                "table_id_t2": match.table_t2.table_id,
                "title_t1": match.table_t1.title,
                "title_t2": match.table_t2.title,
                "page_t1": match.table_t1.page,
                "page_t2": match.table_t2.page,
                "section": match.table_t1.section or match.table_t2.section,
                "match_score": round(match.score, 3),
                "match_quality": _match_quality(match.score),
                "match_reason": match.match_reason,
                "source_method_t1": match.table_t1.source_method,
                "source_method_t2": match.table_t2.source_method,
                "quality_flags_t1": match.table_t1.quality_flags,
                "quality_flags_t2": match.table_t2.quality_flags,
                "source_pdf_t1": match.table_t1.source_pdf,
                "source_pdf_t2": match.table_t2.source_pdf,
                "bbox_t1": list(match.table_t1.bbox) if match.table_t1.bbox else None,
                "bbox_t2": list(match.table_t2.bbox) if match.table_t2.bbox else None,
                "row_bboxes_t1": [
                    [ind, y0, y1] for ind, y0, y1 in (match.table_t1.row_bboxes or [])
                ],
                "row_bboxes_t2": [
                    [ind, y0, y1] for ind, y0, y1 in (match.table_t2.row_bboxes or [])
                ],
                "bbox_source_t1": match.table_t1.bbox_source,
                "bbox_source_t2": match.table_t2.bbox_source,
                **diff,
                "uncertain_diff": uncertain_diff,
                "uncertainty_reasons": uncertainty_reasons,
                "table_status": table_status,
                "review_reasons": sorted(set(review_reasons)),
                "decision_reason": decision_reason,
                "indicators_t1": [item.text for item in match.table_t1.indicators],
                "indicators_t2": [item.text for item in match.table_t2.indicators],
            }
        )

    # Detection des deplacements cross-tableaux (meme indicateur supprime dans A, ajoute dans B)
    removed_items: list[RemovedItem] = []
    added_items: list[AddedItem] = []
    for comp in comparisons:
        key = (str(comp.get("table_id_t1", "")), str(comp.get("table_id_t2", "")))
        section = comp.get("section")
        for text in comp.get("removed_indicators", []):
            canonical = _canonical_indicator_text(text)
            if canonical:
                removed_items.append(
                    RemovedItem(
                        text=text,
                        canonical=canonical,
                        table_id_t1=str(comp.get("table_id_t1", "")),
                        page_t1=int(comp.get("page_t1") or 0),
                        table_id_t2=str(comp.get("table_id_t2", "")),
                        page_t2=int(comp.get("page_t2") or 0),
                        section=section,
                        comparison_key=key,
                    )
                )
        for text in comp.get("added_indicators", []):
            canonical = _canonical_indicator_text(text)
            if canonical:
                added_items.append(
                    AddedItem(
                        text=text,
                        canonical=canonical,
                        table_id_t1=str(comp.get("table_id_t1", "")),
                        page_t1=int(comp.get("page_t1") or 0),
                        table_id_t2=str(comp.get("table_id_t2", "")),
                        page_t2=int(comp.get("page_t2") or 0),
                        section=section,
                        comparison_key=key,
                    )
                )

    displacement_fuzzy_threshold = float(
        thresholds.get("indicator_displacement_fuzzy_threshold", 0.90)
    )
    displacement_section_strict = bool(
        thresholds.get("indicator_displacement_section_strict", True)
    )
    displaced_canonicals, displaced_list = detect_cross_table_displacements(
        removed_items,
        added_items,
        _canonical_indicator_text,
        fuzzy_threshold=displacement_fuzzy_threshold,
        section_strict=displacement_section_strict,
    )

    for d in displaced_list:
        for comp in comparisons:
            if str(comp.get("table_id_t1", "")) == d.from_table_id:
                comp["removed_indicators"] = [
                    t
                    for t in comp.get("removed_indicators", [])
                    if _canonical_indicator_text(t) != d.canonical
                ]
                comp["counts"]["removed"] = len(comp["removed_indicators"])
                break
        for comp in comparisons:
            if str(comp.get("table_id_t2", "")) == d.to_table_id:
                # to_canonical peut differer de canonical pour les deplacements flous
                target_canonical = getattr(d, "to_canonical", d.canonical)
                comp["added_indicators"] = [
                    t
                    for t in comp.get("added_indicators", [])
                    if _canonical_indicator_text(t) != target_canonical
                ]
                comp["counts"]["added"] = len(comp["added_indicators"])
                break

    for comp in comparisons:
        if comp.get("table_status") in {"structure_change", "needs_review"}:
            continue
        comp["table_status"] = _determine_table_status(
            uncertain_diff=comp.get("uncertain_diff", False),
            added=int(comp["counts"].get("added", 0)),
            removed=int(comp["counts"].get("removed", 0)),
            renamed=int(comp["counts"].get("renamed", 0)),
        )

    displaced_dicts = [d.to_dict() for d in displaced_list]

    certain_comparisons = [c for c in comparisons if not c["uncertain_diff"]]
    uncertain_comparisons = [c for c in comparisons if c["uncertain_diff"]]
    status_counts = {
        "stable": 0,
        "modifie": 0,
        "renommage_probable": 0,
        "incertain": 0,
        "needs_review": 0,
        "structure_change": 0,
        "ajoute": len(added_tables),
        "supprime": len(removed_tables),
    }
    for comp in comparisons:
        key = comp.get("table_status", "stable")
        status_counts[key] = status_counts.get(key, 0) + 1

    total_added_raw = sum(c["counts"]["added"] for c in comparisons)
    total_removed_raw = sum(c["counts"]["removed"] for c in comparisons)
    total_renamed_raw = sum(c["counts"]["renamed"] for c in comparisons)

    result = {
        "schema_version": "comparison_canonical_v1",
        "bank_code": bank_code,
        "quarter_from": quarter_from,
        "quarter_to": quarter_to,
        "year": year,
        "section": section,
        "summary": {
            "tables_t1": len(tables_t1),
            "tables_t2": len(tables_t2),
            "tables_matched": len(matches),
            "tables_added": len(added_tables),
            "tables_removed": len(removed_tables),
            "total_added_indicators": sum(c["counts"]["added"] for c in certain_comparisons),
            "total_removed_indicators": sum(c["counts"]["removed"] for c in certain_comparisons),
            "total_renamed_indicators": sum(c["counts"]["renamed"] for c in certain_comparisons),
            "total_displaced_indicators": len(displaced_list),
            "total_added_indicators_raw": total_added_raw,
            "total_removed_indicators_raw": total_removed_raw,
            "total_renamed_indicators_raw": total_renamed_raw,
            "uncertain_tables": len(uncertain_comparisons),
            "uncertain_added_indicators": sum(c["counts"]["added"] for c in uncertain_comparisons),
            "uncertain_removed_indicators": sum(
                c["counts"]["removed"] for c in uncertain_comparisons
            ),
            "uncertain_renamed_indicators": sum(
                c["counts"]["renamed"] for c in uncertain_comparisons
            ),
            "status_counts": status_counts,
        },
        "displaced_indicators": displaced_dicts,
        "table_comparisons": comparisons,
        "tables_added": [
            {
                "table_status": "ajoute",
                "table_id": t.table_id,
                "title": t.title,
                "page": t.page,
                "section": t.section,
                "source_method": t.source_method,
                "quality_flags": t.quality_flags,
                "indicators": [i.text for i in t.indicators],
            }
            for t in added_tables
        ],
        "tables_removed": [
            {
                "table_status": "supprime",
                "table_id": t.table_id,
                "title": t.title,
                "page": t.page,
                "section": t.section,
                "source_method": t.source_method,
                "quality_flags": t.quality_flags,
                "indicators": [i.text for i in t.indicators],
            }
            for t in removed_tables
        ],
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "provenance": "indicator_comparator",
            "source_format": "comparison_canonical_v1",
        },
    }

    return result


def _compute_table_match_score(
    t1: IndicatorTable, t2: IndicatorTable, precomputed: dict[str, float] | None = None
) -> float:
    s = precomputed or _compute_table_match_signals(t1, t2)
    title_missing = not (t1.title or "").strip() and not (t2.title or "").strip()

    # Quand les titres sont absents, on depend beaucoup plus des indicateurs
    if title_missing:
        w = {
            "jaccard": 0.65,
            "title_similarity": 0.0,
            "id_match": 0.20,
            "page_proximity": 0.10,
            "section_match": 0.05,
        }
    else:
        w = {
            "jaccard": 0.50,
            "title_similarity": 0.25,
            "id_match": 0.15,
            "page_proximity": 0.05,
            "section_match": 0.05,
        }

    return (
        w["jaccard"] * s["jaccard"]
        + w["title_similarity"] * s["title_similarity"]
        + w["id_match"] * s["id_match"]
        + w["page_proximity"] * s["page_proximity"]
        + w["section_match"] * s["section_match"]
    )


def _compute_table_match_signals(t1: IndicatorTable, t2: IndicatorTable) -> dict[str, float]:
    id_match = 1.0 if t1.table_id and t2.table_id and t1.table_id == t2.table_id else 0.0

    title1 = (t1.title or "").strip().lower()
    title2 = (t2.title or "").strip().lower()
    title_similarity = SequenceMatcher(None, title1, title2).ratio() if title1 and title2 else 0.0

    jaccard = _jaccard_similarity(t1.indicator_norm_set, t2.indicator_norm_set)

    section_match = 1.0 if _sections_strict_match(t1.section, t2.section) else 0.0

    if t1.page > 0 and t2.page > 0:
        delta = abs(t1.page - t2.page)
        page_proximity = max(0.0, 1.0 - min(delta, 15) / 15.0)
    else:
        page_proximity = 0.0

    return {
        "id_match": id_match,
        "title_similarity": title_similarity,
        "jaccard": jaccard,
        "section_match": section_match,
        "page_proximity": page_proximity,
    }


def _rescue_unmatched_tables(
    unmatched_t1: list[IndicatorTable],
    unmatched_t2: list[IndicatorTable],
) -> tuple[list[TableMatch], list[IndicatorTable], list[IndicatorTable]]:
    thresholds = get_matching_thresholds()
    rescue_split_merge_min = float(thresholds.get("indicator_rescue_split_merge_min", 0.60))
    rescue_jaccard_min = float(thresholds.get("indicator_rescue_jaccard_min", 0.55))
    title_similarity_min = float(thresholds.get("title_similarity_min", 0.75))
    rescue_candidates: list[tuple[float, IndicatorTable, IndicatorTable, str]] = []

    for t1 in unmatched_t1:
        if _canonical_section(t1.section) in UNKNOWN_SECTIONS:
            continue
        for t2 in unmatched_t2:
            if not _sections_strict_match(t1.section, t2.section):
                continue
            signals = _compute_table_match_signals(t1, t2)
            split_merge_score = _compute_split_merge_compatibility_score(t1, t2)

            split_merge_candidate = split_merge_score >= rescue_split_merge_min
            high_jaccard_candidate = (
                signals["jaccard"] >= rescue_jaccard_min and signals["page_proximity"] >= 0.85
            )
            if not split_merge_candidate and not high_jaccard_candidate:
                continue

            rescue_score = max(
                split_merge_score,
                signals["jaccard"],
                0.5 * split_merge_score + 0.5 * signals["page_proximity"],
            )
            reason = "rescue_split_merge" if split_merge_candidate else "rescue_high_jaccard"
            rescue_candidates.append((rescue_score, t1, t2, reason))

    rescue_candidates.sort(key=lambda x: x[0], reverse=True)

    used_t1: set[int] = set()
    used_t2: set[int] = set()
    matches: list[TableMatch] = []

    for score, t1, t2, reason in rescue_candidates:
        if id(t1) in used_t1 or id(t2) in used_t2:
            continue
        signals = _compute_table_match_signals(t1, t2)
        title1 = (t1.title or "").strip()
        title2 = (t2.title or "").strip()
        if title1 and title2 and signals["title_similarity"] < title_similarity_min:
            continue
        matches.append(TableMatch(table_t1=t1, table_t2=t2, score=score, match_reason=reason))
        used_t1.add(id(t1))
        used_t2.add(id(t2))

    remaining_t1 = [t for t in unmatched_t1 if id(t) not in used_t1]
    remaining_t2 = [t for t in unmatched_t2 if id(t) not in used_t2]
    return matches, remaining_t1, remaining_t2


def _compute_split_merge_compatibility_score(t1: IndicatorTable, t2: IndicatorTable) -> float:
    prepared_t1, _ = _prepare_indicator_items(t1.indicators, t1.page)
    prepared_t2, _ = _prepare_indicator_items(t2.indicators, t2.page)
    remaining_t1, remaining_t2 = _drop_exact_canonical_matches(prepared_t1, prepared_t2)
    if not remaining_t1 or not remaining_t2:
        return 0.0

    artifacts, _, _ = _consume_split_merge_artifacts(remaining_t1, remaining_t2)
    if artifacts <= 0:
        return 0.0

    residual_min = min(len(remaining_t1), len(remaining_t2))
    return artifacts / max(1, residual_min)


def _is_safe_id_match(t1: IndicatorTable, t2: IndicatorTable, signals: dict[str, float]) -> bool:
    """Valider un match par id pour eviter les faux appariements massifs."""
    if signals["id_match"] < 1.0:
        return False
    if signals["section_match"] <= 0.0:
        return False

    title_missing = not (t1.title or "").strip() and not (t2.title or "").strip()
    jaccard = signals["jaccard"]
    title_similarity = signals["title_similarity"]
    page_proximity = signals["page_proximity"]

    if title_similarity >= 0.75:
        return True
    if jaccard >= 0.25:
        return True
    if title_missing and jaccard < 0.15:
        return False
    if page_proximity >= 0.85 and jaccard >= 0.15:
        return True
    return False


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _canonical_indicator_text(text: str) -> str:
    """Unified canonical key for indicator labels; delegates to normalize_indicator_for_comparison."""
    return normalize_indicator_for_comparison(text or "")


def _normalize_total_order(canonical: str) -> str:
    tokens = [token for token in canonical.split() if token]
    if len(tokens) < 2:
        return canonical
    if tokens[0] == "total":
        return " ".join(tokens[1:] + ["total"])
    if tokens[-1] == "total":
        return " ".join(tokens[:-1] + ["total"])
    return canonical


def _strip_contextual_dates(canonical: str) -> str:
    """Strip dates only in contextual label patterns such as 'echeant le ...'."""
    value = _ECHEANT_LE_TEXTUAL_DATE_PATTERN.sub(r"\1", canonical)
    value = _ECHEANT_LE_NUMERIC_DATE_PATTERN.sub(r"\1", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;:-")
    return value


def _is_meta_indicator_line(text: str) -> bool:
    lowered = TextNormalizer.normalize(text, aggressive=False, remove_notes=False, lowercase=True)
    if not lowered:
        return True

    lowered = lowered.strip()
    lowered_ascii = lowered.replace("’", "'")
    canonical = _canonical_indicator_text(text)

    for pattern in _META_LINE_PATTERNS:
        if pattern.match(lowered_ascii):
            return True

    for pattern in _STRICT_STRUCTURAL_LINE_PATTERNS:
        if pattern.fullmatch(lowered_ascii):
            return True

    if canonical in _GENERIC_HEADER_CANONICAL:
        return True

    # Lignes de date (avec/sans "au")
    for month in _MONTH_NAMES:
        if lowered_ascii.startswith("au ") and month in lowered_ascii:
            return True
        if re.match(r"^\d{1,2}\s+", lowered_ascii) and month in lowered_ascii:
            return True

    # Lignes "trimestre termine le ..." ou "trimestre clos le ..." coupees
    if "trimestre" in lowered_ascii and ("termine" in lowered_ascii or "clos" in lowered_ascii):
        return True

    # Fragments de ligne OCR (continuation de la ligne precedente).
    raw = (text or "").strip()
    words = re.findall(r"\w+", raw, flags=re.UNICODE)
    if raw and raw[0].islower() and len(words) <= 8:
        # Garder seulement les lignes numeriques "actionnables" (ex: diminution/augmentation ...),
        # sinon considerer comme continuation OCR de la ligne precedente.
        has_numeric_marker = bool(re.search(r"\d", raw)) or bool(
            re.search(r"\b(?:p\.?\s*b\.?|bps|%|pour cent)\b", lowered_ascii)
        )
        first_token = words[0].lower() if words else ""
        if has_numeric_marker and first_token in _LOWERCASE_NUMERIC_KEEP_PREFIXES:
            return False
        return True
    return False


def _infer_group_from_text(text: str) -> str:
    canonical = _canonical_indicator_text(text)
    if not canonical:
        return "unknown"
    for group_name, patterns in _GROUP_PATTERNS.items():
        if any(pattern.search(canonical) for pattern in patterns):
            return group_name
    return "unknown"


def _is_group_compatible(group_left: str, group_right: str, strict: bool) -> bool:
    if not strict:
        return True
    if group_left == "unknown" or group_right == "unknown":
        return True
    return group_left == group_right


def _token_jaccard(left: str, right: str) -> float:
    lt = {token for token in left.split() if token}
    rt = {token for token in right.split() if token}
    if not lt or not rt:
        return 0.0
    return len(lt & rt) / len(lt | rt)


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def _score_indicator_similarity(left: str, right: str) -> dict[str, float]:
    seq_score = SequenceMatcher(None, left, right).ratio()
    token_sort_score = 0.0
    if rapidfuzz_fuzz is not None:
        token_sort_score = rapidfuzz_fuzz.token_sort_ratio(left, right) / 100.0
    score_jaccard = _token_jaccard(left, right)
    score = max(seq_score, token_sort_score, score_jaccard)
    return {
        "score": score,
        "score_sequence": seq_score,
        "score_token": token_sort_score,
        "score_jaccard": score_jaccard,
        "length_ratio": _length_ratio(left, right),
    }


def _build_indicator_decision(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    decision: str,
    reason: str,
    section: str,
    table_id_t1: str,
    table_id_t2: str,
    page_t1: int,
    page_t2: int,
    score_meta: dict[str, float] | None = None,
) -> dict[str, Any]:
    group = "unknown"
    if left and left.get("group") and left.get("group") != "unknown":
        group = str(left.get("group"))
    elif right and right.get("group"):
        group = str(right.get("group"))
    score_meta = score_meta or {}

    # Mapping des scores standarises
    text_score = score_meta.get("text_score", score_meta.get("score", 0.0))
    pos_score = score_meta.get("position_score", 0.0)
    neigh_score = score_meta.get("neighborhood_score", 0.0)
    comp_score = score_meta.get("composite_score", text_score)  # Fallback

    return {
        "t1_label_raw": left.get("text", "") if left else "",
        "t1_label_norm": left.get("canonical", "") if left else "",
        "best_match_t2_raw": right.get("text", "") if right else "",
        "best_match_t2_norm": right.get("canonical", "") if right else "",
        "decision": decision,
        "score_text": round(float(text_score), 4),
        "score_pos": round(float(pos_score), 4),
        "score_neigh": round(float(neigh_score), 4),
        "score_composite": round(float(comp_score), 4),
        "context": {
            "section": section,
            "table_id_t1": table_id_t1,
            "table_id_t2": table_id_t2,
            "group": group,
            "page_t1": page_t1,
            "page_t2": page_t2,
        },
        "decision_reason": reason,
    }


def _consume_exact_matches(
    entries_t1: list[dict[str, Any]],
    entries_t2: list[dict[str, Any]],
    group_strict: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    exact_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_t2: set[int] = set()
    remaining_t1: list[dict[str, Any]] = []

    for left in entries_t1:
        match_idx = None
        for j, right in enumerate(entries_t2):
            if j in used_t2:
                continue
            if left["canonical"] == right["canonical"] and _is_group_compatible(
                left["group"], right["group"], group_strict
            ):
                match_idx = j
                break
        if match_idx is None:
            remaining_t1.append(left)
            continue
        used_t2.add(match_idx)
        exact_pairs.append((left, entries_t2[match_idx]))

    remaining_t2 = [entry for j, entry in enumerate(entries_t2) if j not in used_t2]
    return exact_pairs, remaining_t1, remaining_t2


def _consume_near_exact_matches(
    remaining_t1: list[dict[str, Any]],
    remaining_t2: list[dict[str, Any]],
    threshold: float,
    length_ratio_min: float,
    group_strict: bool,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidates: list[tuple[float, int, int, dict[str, float]]] = []

    # Pour le scoring, on construit des contextes temporaires
    # Note: neighbor_prev/next sont deja dans remaining_t1/t2 grace a _prepare_indicator_items

    for i, left in enumerate(remaining_t1):
        for j, right in enumerate(remaining_t2):
            # Appel au moteur unifie
            result = compute_candidate_score(
                candidate_text=right["canonical"],
                reference_text=left["canonical"],
                candidate_context=right,  # Contient neighbor info
                reference_context=left,
                robust_mode=False,
                section_strict=False,  # Deja filtre en amont si besoin, ici on compare des listes pre-filtrees ?
                # En fait non, on compare tout venant, donc garder section strict ?
                # Mais compute_candidate_score verifie section/group.
                # Ici on veut juste filtrer par score.
            )

            # Verifier compatibilite groupe (si strict)
            if group_strict and not result["is_context_compatible"]:
                continue

            # Criteres d'acceptation (Near Exact)
            if result["text_score"] < threshold:
                continue
            if result["length_ratio"] < length_ratio_min:
                continue

            candidates.append((result["composite_score"], i, j, result))

    # Tri par score composite
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_t1: set[int] = set()
    used_t2: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]] = []

    for _, i, j, score_result in candidates:
        if i in used_t1 or j in used_t2:
            continue
        used_t1.add(i)
        used_t2.add(j)
        pairs.append((remaining_t1[i], remaining_t2[j], score_result))

    next_t1 = [entry for i, entry in enumerate(remaining_t1) if i not in used_t1]
    next_t2 = [entry for j, entry in enumerate(remaining_t2) if j not in used_t2]
    return pairs, next_t1, next_t2


def _detect_split_merge_collisions(
    entries_t1: list[dict[str, Any]],
    entries_t2: list[dict[str, Any]],
    band_min: float,
    band_max: float,
    length_ratio_min: float,
    group_strict: bool,
) -> tuple[set[int], set[int]]:
    collisions_t1: set[int] = set()
    collisions_t2: set[int] = set()
    t1_hits: dict[int, int] = Counter()
    t2_hits: dict[int, int] = Counter()

    for i, left in enumerate(entries_t1):
        for j, right in enumerate(entries_t2):
            if not _is_group_compatible(left["group"], right["group"], group_strict):
                continue
            score_meta = _score_indicator_similarity(left["canonical"], right["canonical"])
            if score_meta["length_ratio"] < length_ratio_min:
                continue
            if band_min <= score_meta["score"] < band_max:
                t1_hits[i] = t1_hits.get(i, 0) + 1
                t2_hits[j] = t2_hits.get(j, 0) + 1

    for idx, value in t1_hits.items():
        if value >= 2:
            collisions_t1.add(idx)
    for idx, value in t2_hits.items():
        if value >= 2:
            collisions_t2.add(idx)
    return collisions_t1, collisions_t2


def _best_compatible_candidate(
    left: dict[str, Any],
    candidates: list[dict[str, Any]],
    used_t2: set[int],
    group_strict: bool,
    length_ratio_min: float,
) -> tuple[int, dict[str, float]] | None:
    best: tuple[int, dict[str, float]] | None = None
    for j, right in enumerate(candidates):
        if j in used_t2:
            continue
        if not _is_group_compatible(left["group"], right["group"], group_strict):
            continue
        score_meta = _score_indicator_similarity(left["canonical"], right["canonical"])
        if score_meta["length_ratio"] < length_ratio_min:
            continue
        if best is None or score_meta["score"] > best[1]["score"]:
            best = (j, score_meta)
    return best


def _compute_position_score(
    row_idx_t1: int, total_t1: int, row_idx_t2: int, total_t2: int
) -> float:
    """Score de proximite positionnelle relative (0.0 a 1.0)."""
    if total_t1 <= 0 or total_t2 <= 0:
        return 0.5
    pos_t1 = row_idx_t1 / max(1, total_t1 - 1) if total_t1 > 1 else 0.5
    pos_t2 = row_idx_t2 / max(1, total_t2 - 1) if total_t2 > 1 else 0.5
    return max(0.0, 1.0 - abs(pos_t1 - pos_t2))


def _compute_neighborhood_score(
    idx_t1: int,
    idx_t2: int,
    all_t1: list[dict[str, Any]],
    all_t2: list[dict[str, Any]],
) -> float:
    """Bonus si les voisins (ligne au-dessus / en-dessous) sont identiques."""
    bonus = 0.0
    # Voisin au-dessus
    if idx_t1 > 0 and idx_t2 > 0:
        if all_t1[idx_t1 - 1]["canonical"] == all_t2[idx_t2 - 1]["canonical"]:
            bonus += 0.5
    # Voisin en-dessous
    if idx_t1 < len(all_t1) - 1 and idx_t2 < len(all_t2) - 1:
        if all_t1[idx_t1 + 1]["canonical"] == all_t2[idx_t2 + 1]["canonical"]:
            bonus += 0.5
    return bonus


def _is_short_generic_label(canonical: str) -> bool:
    """Verifie si un canonical est un label court et generique."""
    if canonical in _SHORT_GENERIC_LABELS:
        return True
    return len(canonical) <= _SHORT_LABEL_MAX_LEN and canonical not in _HEADER_SHORT_ALLOWLIST


def _match_renames_with_context(
    residual_t1: list[dict[str, Any]],
    residual_t2: list[dict[str, Any]],
    threshold: float,
    length_ratio_min: float,
    group_strict: bool,
    all_prepared_t1: list[dict[str, Any]] | None = None,
    all_prepared_t2: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    thresholds = get_matching_thresholds()
    short_label_threshold = float(thresholds.get("indicator_rename_short_label_threshold", 0.95))
    position_weight = float(thresholds.get("indicator_rename_position_weight", 0.10))
    neighborhood_weight = float(thresholds.get("indicator_rename_neighborhood_weight", 0.05))
    text_weight = 1.0 - position_weight - neighborhood_weight

    total_t1 = len(all_prepared_t1) if all_prepared_t1 else len(residual_t1)
    total_t2 = len(all_prepared_t2) if all_prepared_t2 else len(residual_t2)

    renamed: list[dict[str, Any]] = []
    used_t1: set[int] = set()
    used_t2: set[int] = set()

    while True:
        best_pair: tuple[float, int, int, dict[str, float]] | None = None
        for i, rem in enumerate(residual_t1):
            if i in used_t1:
                continue
            for j, add in enumerate(residual_t2):
                if j in used_t2:
                    continue
                if not _is_group_compatible(rem["group"], add["group"], group_strict):
                    continue
                rem_canonical = _normalize_total_order(rem["canonical"])
                add_canonical = _normalize_total_order(add["canonical"])

                # Protection labels courts/generiques
                both_short_generic = (
                    _is_short_generic_label(rem_canonical)
                    and _is_short_generic_label(add_canonical)
                    and rem_canonical != add_canonical
                )

                score_meta = _score_indicator_similarity(rem_canonical, add_canonical)
                if score_meta["length_ratio"] < length_ratio_min:
                    continue
                text_score = score_meta["score"]

                # Seuil plus eleve pour labels courts generiques
                effective_threshold = short_label_threshold if both_short_generic else threshold

                if text_score < effective_threshold:
                    continue

                # Score composite avec position et voisinage
                pos_score = _compute_position_score(
                    rem.get("row_idx", 0),
                    total_t1,
                    add.get("row_idx", 0),
                    total_t2,
                )
                neigh_score = 0.0
                if all_prepared_t1 and all_prepared_t2:
                    orig_idx_t1 = next(
                        (
                            k
                            for k, e in enumerate(all_prepared_t1)
                            if e["canonical"] == rem["canonical"]
                        ),
                        -1,
                    )
                    orig_idx_t2 = next(
                        (
                            k
                            for k, e in enumerate(all_prepared_t2)
                            if e["canonical"] == add["canonical"]
                        ),
                        -1,
                    )
                    if orig_idx_t1 >= 0 and orig_idx_t2 >= 0:
                        neigh_score = _compute_neighborhood_score(
                            orig_idx_t1, orig_idx_t2, all_prepared_t1, all_prepared_t2
                        )

                composite = (
                    text_weight * text_score
                    + position_weight * pos_score
                    + neighborhood_weight * neigh_score
                )
                score_meta["composite_score"] = composite
                score_meta["position_score"] = pos_score
                score_meta["neighborhood_score"] = neigh_score

                # Le seuil s'applique sur le score textuel; le composite sert au classement
                if best_pair is None or composite > best_pair[0]:
                    best_pair = (composite, i, j, score_meta)

        if best_pair is None:
            break

        _, i, j, score_meta = best_pair
        used_t1.add(i)
        used_t2.add(j)
        renamed.append(
            {
                "left": residual_t1[i],
                "right": residual_t2[j],
                "score_meta": score_meta,
            }
        )

    return renamed, used_t1, used_t2


def _resolve_ambiguous_with_llm(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str]:
    """Stub defensif: LLM desactive par defaut; retourne false si indisponible."""
    _ = (left, right)
    return False, "llm_resolver_unavailable"


def _prepare_indicator_items(
    items: list[IndicatorItem], page: int
) -> tuple[list[dict[str, Any]], int]:
    prepared: list[dict[str, Any]] = []
    ignored = 0
    seen_canonical: set[str] = set()
    current_group = "unknown"

    for row_idx, item in enumerate(items):
        text = (item.text or "").strip()
        if not text:
            continue

        explicit_group = _infer_group_from_text(text)
        if explicit_group != "unknown":
            current_group = explicit_group

        if _is_meta_indicator_line(text):
            ignored += 1
            continue

        canonical = _canonical_indicator_text(text)
        if not canonical:
            continue

        # Les doublons exacts dans un meme tableau sont tres souvent du bruit de segmentation.
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        if prepared and prepared[-1]["canonical"] == canonical:
            continue

        prepared.append(
            {
                "text": text,
                "canonical": canonical,
                "group": explicit_group if explicit_group != "unknown" else current_group,
                "row_idx": row_idx,
                "page": page,
                "neighbor_prev": None,
                "neighbor_next": None,
            }
        )

    # Calcul du voisinage
    for k in range(len(prepared)):
        if k > 0:
            prepared[k]["neighbor_prev"] = prepared[k - 1]["canonical"]
        if k < len(prepared) - 1:
            prepared[k]["neighbor_next"] = prepared[k + 1]["canonical"]

    return prepared, ignored


def _drop_exact_canonical_matches(
    entries_t1: list[dict[str, str]],
    entries_t2: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    c1 = Counter(entry["canonical"] for entry in entries_t1)
    c2 = Counter(entry["canonical"] for entry in entries_t2)
    shared = c1 & c2

    used_t1: Counter[str] = Counter()
    used_t2: Counter[str] = Counter()
    remaining_t1: list[dict[str, str]] = []
    remaining_t2: list[dict[str, str]] = []

    for entry in entries_t1:
        key = entry["canonical"]
        if used_t1[key] < shared[key]:
            used_t1[key] += 1
        else:
            remaining_t1.append(entry)

    for entry in entries_t2:
        key = entry["canonical"]
        if used_t2[key] < shared[key]:
            used_t2[key] += 1
        else:
            remaining_t2.append(entry)

    return remaining_t1, remaining_t2


def _consume_split_merge_artifacts(
    entries_t1: list[dict[str, str]],
    entries_t2: list[dict[str, str]],
) -> tuple[int, set[int], set[int]]:
    consumed_t1: set[int] = set()
    consumed_t2: set[int] = set()

    # 1 vs (2..3)
    for i, entry_t1 in enumerate(entries_t1):
        if i in consumed_t1:
            continue
        best_match: tuple[float, int, int] | None = None
        for j in range(len(entries_t2)):
            if j in consumed_t2:
                continue
            for width in range(2, 9):
                end = j + width
                if end > len(entries_t2):
                    continue
                if any(idx in consumed_t2 for idx in range(j, end)):
                    continue
                joined = " ".join(entries_t2[idx]["canonical"] for idx in range(j, end))
                score = SequenceMatcher(None, entry_t1["canonical"], joined).ratio()
                if _is_split_merge_equivalent(entry_t1["canonical"], joined, score):
                    if best_match is None or score > best_match[0]:
                        best_match = (score, j, end)
        if best_match is not None:
            _, start, end = best_match
            consumed_t1.add(i)
            consumed_t2.update(range(start, end))

    # (2..3) vs 1
    for j, entry_t2 in enumerate(entries_t2):
        if j in consumed_t2:
            continue
        best_match = None
        for i in range(len(entries_t1)):
            if i in consumed_t1:
                continue
            for width in range(2, 9):
                end = i + width
                if end > len(entries_t1):
                    continue
                if any(idx in consumed_t1 for idx in range(i, end)):
                    continue
                joined = " ".join(entries_t1[idx]["canonical"] for idx in range(i, end))
                score = SequenceMatcher(None, joined, entry_t2["canonical"]).ratio()
                if _is_split_merge_equivalent(joined, entry_t2["canonical"], score):
                    if best_match is None or score > best_match[0]:
                        best_match = (score, i, end)
        if best_match is not None:
            _, start, end = best_match
            consumed_t2.add(j)
            consumed_t1.update(range(start, end))

    artifacts = min(len(consumed_t1), len(consumed_t2))
    return artifacts, consumed_t1, consumed_t2


def _is_split_merge_equivalent(left: str, right: str, similarity: float) -> bool:
    if not left or not right:
        return False
    if min(len(left), len(right)) < 12:
        return False

    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))

    contains_relation = left in right or right in left
    if contains_relation and similarity >= 0.80 and overlap >= 0.70:
        return True
    if similarity >= 0.92 and overlap >= 0.60:
        return True
    return False


def _match_renames(
    residual_t1: list[dict[str, str]],
    residual_t2: list[dict[str, str]],
    threshold: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    renamed: list[dict[str, Any]] = []
    used_t1: set[int] = set()
    used_t2: set[int] = set()

    while True:
        best_pair: tuple[float, int, int] | None = None
        for i, rem in enumerate(residual_t1):
            if i in used_t1:
                continue
            for j, add in enumerate(residual_t2):
                if j in used_t2:
                    continue
                rem_canonical = _normalize_total_order(rem["canonical"])
                add_canonical = _normalize_total_order(add["canonical"])
                seq_score = SequenceMatcher(None, rem_canonical, add_canonical).ratio()
                token_sort_score = 0.0
                if rapidfuzz_fuzz is not None:
                    token_sort_score = (
                        rapidfuzz_fuzz.token_sort_ratio(rem_canonical, add_canonical) / 100.0
                    )
                score = max(seq_score, token_sort_score)
                if score >= threshold:
                    if best_pair is None or score > best_pair[0]:
                        best_pair = (score, i, j)

        if best_pair is None:
            break

        score, i, j = best_pair
        used_t1.add(i)
        used_t2.add(j)
        renamed.append(
            {
                "from": residual_t1[i]["text"],
                "to": residual_t2[j]["text"],
                "similarity": round(score, 3),
            }
        )

    return renamed, used_t1, used_t2


def _build_indicator_canonical_pool(tables: list[IndicatorTable]) -> set[str]:
    values: set[str] = set()
    for table in tables:
        for item in table.indicators:
            text = (item.text or "").strip()
            if not text or _is_meta_indicator_line(text):
                continue
            canonical = _canonical_indicator_text(text)
            if canonical:
                values.add(canonical)
    return values


def _determine_table_status(uncertain_diff: bool, added: int, removed: int, renamed: int) -> str:
    if uncertain_diff:
        return "incertain"
    if added > 0 or removed > 0:
        return "modifie"
    if renamed > 0:
        return "renommage_probable"
    return "stable"


def _match_quality(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _aggregate_decision_reasons(decisions: list[dict[str, Any]]) -> str:
    reasons = sorted(
        {
            str(item.get("decision_reason", "")).strip()
            for item in decisions
            if str(item.get("decision_reason", "")).strip()
        }
    )
    return "; ".join(reasons)


def _is_reliable_table_match(match: TableMatch) -> bool:
    if match.match_reason == "id":
        return True
    signals = _compute_table_match_signals(match.table_t1, match.table_t2)
    if signals["title_similarity"] >= 0.80:
        return True
    if signals["jaccard"] >= 0.45:
        return True
    return False


def _soft_canonical(canonical: str) -> str:
    tokens = [token for token in canonical.split() if token and token not in _SOFT_STOPWORDS]
    return " ".join(tokens).strip()


def _drop_soft_equivalent_matches(
    entries_t1: list[dict[str, str]],
    entries_t2: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    c1 = Counter(_soft_canonical(entry["canonical"]) for entry in entries_t1)
    c2 = Counter(_soft_canonical(entry["canonical"]) for entry in entries_t2)
    shared = c1 & c2

    used_t1: Counter[str] = Counter()
    used_t2: Counter[str] = Counter()
    remaining_t1: list[dict[str, str]] = []
    remaining_t2: list[dict[str, str]] = []

    for entry in entries_t1:
        key = _soft_canonical(entry["canonical"])
        if key and used_t1[key] < shared[key]:
            used_t1[key] += 1
        else:
            remaining_t1.append(entry)

    for entry in entries_t2:
        key = _soft_canonical(entry["canonical"])
        if key and used_t2[key] < shared[key]:
            used_t2[key] += 1
        else:
            remaining_t2.append(entry)

    return remaining_t1, remaining_t2


def _drop_header_containment_artifacts(
    added_values: list[str], removed_values: list[str]
) -> tuple[list[str], list[str], int]:
    if not added_values or not removed_values:
        return added_values, removed_values, 0

    consumed_added: set[int] = set()
    consumed_removed: set[int] = set()
    pairs: list[tuple[float, int, int]] = []

    for i, add_value in enumerate(added_values):
        add_canonical = _canonical_indicator_text(add_value)
        if not add_canonical:
            continue
        for j, rem_value in enumerate(removed_values):
            rem_canonical = _canonical_indicator_text(rem_value)
            if not rem_canonical:
                continue
            if not _is_header_containment_artifact(add_canonical, rem_canonical):
                continue
            score = SequenceMatcher(None, add_canonical, rem_canonical).ratio()
            pairs.append((score, i, j))

    pairs.sort(reverse=True)
    for _, i, j in pairs:
        if i in consumed_added or j in consumed_removed:
            continue
        consumed_added.add(i)
        consumed_removed.add(j)

    filtered_added = [value for idx, value in enumerate(added_values) if idx not in consumed_added]
    filtered_removed = [
        value for idx, value in enumerate(removed_values) if idx not in consumed_removed
    ]
    suppressed = min(len(consumed_added), len(consumed_removed))
    return filtered_added, filtered_removed, suppressed


def _drop_generic_header_artifacts(
    added_values: list[str], removed_values: list[str]
) -> tuple[list[str], list[str], int]:
    filtered_added: list[str] = []
    filtered_removed: list[str] = []
    suppressed = 0

    for value in added_values:
        canonical = _canonical_indicator_text(value)
        if canonical and any(p.fullmatch(canonical) for p in _STRICT_STRUCTURAL_LINE_PATTERNS):
            suppressed += 1
            continue
        filtered_added.append(value)

    for value in removed_values:
        canonical = _canonical_indicator_text(value)
        if canonical and any(p.fullmatch(canonical) for p in _STRICT_STRUCTURAL_LINE_PATTERNS):
            suppressed += 1
            continue
        filtered_removed.append(value)

    return filtered_added, filtered_removed, suppressed


def _is_header_containment_artifact(left_canonical: str, right_canonical: str) -> bool:
    if not left_canonical or not right_canonical:
        return False
    if left_canonical == right_canonical:
        return False

    shorter, longer = (left_canonical, right_canonical)
    if len(left_canonical) > len(right_canonical):
        shorter, longer = right_canonical, left_canonical

    if shorter not in longer:
        return False

    short_tokens = [token for token in shorter.split() if token]
    if not short_tokens:
        return False
    if len(short_tokens) == 1 and short_tokens[0] not in _HEADER_SHORT_ALLOWLIST:
        return False

    remainder = re.sub(re.escape(shorter), " ", longer, count=1).strip()
    remainder_tokens = [token for token in remainder.split() if token]
    if not remainder_tokens or len(remainder_tokens) > 6:
        return False

    if all(token in _GENERIC_HEADER_TOKENS for token in remainder_tokens):
        return True

    # Cas fallback: relation prefixe/suffixe tres proche + faible ajout lexical.
    boundary_relation = longer.startswith(f"{shorter} ") or longer.endswith(f" {shorter}")
    if boundary_relation:
        ratio = SequenceMatcher(None, shorter, longer).ratio()
        if ratio >= 0.78 and len(remainder_tokens) <= 3:
            return True

    return False


def run_strict_intra_section_compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper around the official strict comparator facade."""
    from vigilance.compare import run_strict_intra_section_compare as _run

    return _run(*args, **kwargs)
