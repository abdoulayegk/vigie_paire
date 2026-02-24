"""Run T1/T2 comparison from uploaded PDFs and section ranges."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None  # type: ignore[assignment]

from app.ui_config import INDICATOR_COMPARISON_DIR
from vigilance.compare import run_strict_intra_section_compare
from vigilance.compare.table_fragment_merger import merge_table_fragments
from vigilance.config import get_matching_thresholds
from vigilance.models.table_models import TableArtifact
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
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
        extraction_method="docling",
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
    from vigilance.extraction.docling_processor import extract_tables_docling_by_sections

    del use_vision_fallback, api_key

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
            key = _canonical_indicator_key(value)
            if key and key not in mapped:
                mapped[key] = value
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

    strict = run_strict_intra_section_compare(
        tables_t1=tables_t1,
        tables_t2=tables_t2,
        bank_code=bank_code,
    )

    t1_by_uid = {_section_uid(table): table for table in tables_t1}
    t2_by_uid = {_section_uid(table): table for table in tables_t2}

    comparisons: list[dict[str, Any]] = []
    for pair in strict.get("pairs", []):
        t1_uid = str(pair.get("t1_uid", ""))
        t2_uid = str(pair.get("t2_uid", ""))
        table_t1 = t1_by_uid.get(t1_uid)
        table_t2 = t2_by_uid.get(t2_uid)
        if table_t1 is None or table_t2 is None:
            continue

        added, removed, had_fusion_split, excluded_counts = _indicator_diff(table_t1, table_t2)
        added, removed, renamed_pairs = _fuzzy_pair_added_removed(added, removed, bank_code)
        renamed_indicators = [{"from": r, "to": a} for (r, a) in renamed_pairs]

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

    tables_added = [
        {
            "table_status": "ajoute",
            "table_id": str(item.get("t2_table_id", "")),
            "title": item.get("title_t2", ""),
            "page": item.get("page_t2"),
            "section": item.get("section", ""),
            "source_method": "docling",
            "quality_flags": [],
            "indicators": list(item.get("first_column_indicators", []) or []),
        }
        for item in strict.get("added_tables", [])
    ]
    tables_removed = [
        {
            "table_status": "supprime",
            "table_id": str(item.get("t1_table_id", "")),
            "title": item.get("title_t1", ""),
            "page": item.get("page_t1"),
            "section": item.get("section", ""),
            "source_method": "docling",
            "quality_flags": [],
            "indicators": list(item.get("first_column_indicators", []) or []),
        }
        for item in strict.get("removed_tables", [])
    ]

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
        },
    }

    INDICATOR_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = INDICATOR_COMPARISON_DIR / f"{bank_code}_t1_vs_t2_{year}_{stamp}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["meta"]["compare_path"] = str(out_path)

    return result
