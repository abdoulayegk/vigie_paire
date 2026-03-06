"""Vision-based semantic validation for table pair matching.

Uses GPT-4o as the single semantic arbiter to decide whether two extracted
tables represent the same regulatory concept across quarters (T1 vs T2).

Supports:
- Validation of proposed pairs (reject false positives).
- Rescue of orphan tables (recover false negatives) via parallel 1-vs-1 calls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tri-state decision contract
# ---------------------------------------------------------------------------

DECISION_MATCH = "match"
DECISION_NO_MATCH = "no_match"
DECISION_UNKNOWN = "unknown"


@dataclass
class VisionDecision:
    """Structured result from a Vision validation call."""

    decision: str = DECISION_UNKNOWN  # match | no_match | unknown
    confidence: float = 0.0
    reason_code: str = ""  # e.g. "api_error", "crop_failed", "low_confidence"
    analysis: dict[str, Any] = field(default_factory=dict)

    @property
    def same_concept(self) -> bool:
        """Legacy compat: True only when decision is 'match'."""
        return self.decision == DECISION_MATCH

    def as_legacy_tuple(self) -> tuple[bool, float]:
        """Return (same_concept, confidence) for backward-compatible callers."""
        # Preserve legacy fail-open behavior while callers migrate to tri-state.
        if self.decision == DECISION_UNKNOWN:
            return True, 0.0
        return self.same_concept, self.confidence


# ---------------------------------------------------------------------------
# Chain-of-Thought prompt — two separate images (T1 first, T2 second)
# ---------------------------------------------------------------------------

_VALIDATE_PROMPT = """\
Tu es un expert en rapports reglementaires bancaires canadiens (BSIF/OSFI).

IMAGES: Tu recois DEUX images separees.
- Image 1 = tableau extrait du rapport du trimestre precedent (T1).
- Image 2 = tableau extrait du rapport du trimestre courant (T2).

TACHE: Determine si ces deux tableaux correspondent au MEME concept \
reglementaire (ex: meme ratio, meme categorie d'exposition, meme divulgation).

RAISONNEMENT: Avant de conclure, analyse methodiquement:
1. Structure — comparer le nombre de colonnes/lignes, les en-tetes.
2. Theme — identifier le concept financier/reglementaire de chaque tableau.
3. Differences — noter renumerotation, reformulation, ajout/suppression de lignes.

Un tableau renumerote, legerement reformule ou avec quelques lignes ajoutees/\
supprimees reste le MEME concept.
Deux tableaux sur des themes differents (ex: risque de credit vs liquidite) \
sont des concepts DIFFERENTS.

{context_block}

Reponds UNIQUEMENT par un JSON valide:
{{
  "analyse_structurelle": "...",
  "analyse_themes": "...",
  "differences_notables": "...",
  "same_concept": true ou false,
  "confidence": 0.0-1.0
}}
"""


def _build_context_block(
    title_t1: str | None = None,
    title_t2: str | None = None,
    section_t1: str | None = None,
    section_t2: str | None = None,
) -> str:
    """Build optional textual context injected into the prompt."""
    lines: list[str] = []
    if title_t1:
        lines.append(f'Titre T1: "{title_t1}"')
    if title_t2:
        lines.append(f'Titre T2: "{title_t2}"')
    if section_t1:
        lines.append(f'Section T1: "{section_t1}"')
    if section_t2:
        lines.append(f'Section T2: "{section_t2}"')
    if lines:
        return "CONTEXTE TEXTUEL (extrait du PDF):\n" + "\n".join(lines)
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_bbox(bbox: Any) -> list[float] | None:
    """Normalize bbox to [l,t,r,b] 0-1."""
    if bbox is None:
        return None
    try:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        if isinstance(bbox, dict):
            if "l" in bbox and "t" in bbox:
                return [
                    float(bbox["l"]),
                    float(bbox["t"]),
                    float(bbox["r"]),
                    float(bbox["b"]),
                ]
            if "x0" in bbox and "y0" in bbox:
                return [
                    float(bbox["x0"]),
                    float(bbox["y0"]),
                    float(bbox["x1"]),
                    float(bbox["y1"]),
                ]
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _crop_table(
    pdf_path: str, page: int, bbox: list[float], bottom_extension: float
) -> bytes | None:
    """Crop a single table region; returns PNG bytes or None."""
    from ..utils.pdf_crop import crop_table_region_to_bytes

    try:
        data = crop_table_region_to_bytes(
            pdf_path,
            page,
            bbox,
            dpi=300,
            bottom_extension=bottom_extension,
        )
        return data if data else None
    except Exception as e:
        logger.debug("Crop failed: %s", e)
        return None


def _parse_vision_response(raw_json: str) -> VisionDecision:
    """Parse GPT-4o JSON response into a VisionDecision."""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return VisionDecision(
            decision=DECISION_UNKNOWN,
            confidence=0.0,
            reason_code="json_parse_error",
        )

    same = data.get("same_concept")
    conf = float(data.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))

    analysis = {
        k: data.get(k, "")
        for k in ("analyse_structurelle", "analyse_themes", "differences_notables")
    }

    if same is None:
        return VisionDecision(
            decision=DECISION_UNKNOWN,
            confidence=conf,
            reason_code="missing_same_concept_field",
            analysis=analysis,
        )

    decision = DECISION_MATCH if same else DECISION_NO_MATCH
    return VisionDecision(
        decision=decision,
        confidence=conf,
        reason_code="vision_ok",
        analysis=analysis,
    )


def _build_messages(
    crop1_b64: str,
    crop2_b64: str,
    title_t1: str | None = None,
    title_t2: str | None = None,
    section_t1: str | None = None,
    section_t2: str | None = None,
) -> list[Any]:
    """Build the OpenAI messages payload with two separate images."""
    context_block = _build_context_block(title_t1, title_t2, section_t1, section_t2)
    prompt_text = _VALIDATE_PROMPT.format(context_block=context_block)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{crop1_b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{crop2_b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Async core — single semantic arbiter
# ---------------------------------------------------------------------------


async def validate_pair_same_concept_async(
    pdf_path_t1: str,
    page_t1: int,
    bbox_t1: Any,
    pdf_path_t2: str,
    page_t2: int,
    bbox_t2: Any,
    api_key: str,
    bottom_extension: float = 0.12,
    title_t1: str | None = None,
    title_t2: str | None = None,
    section_t1: str | None = None,
    section_t2: str | None = None,
) -> VisionDecision:
    """Async GPT-4o validation: are two table crops the same regulatory concept?

    Returns a VisionDecision with decision = match | no_match | unknown.
    Never silently accepts on error — returns 'unknown' instead.
    """
    b1 = _normalize_bbox(bbox_t1)
    b2 = _normalize_bbox(bbox_t2)
    if not b1 or not b2:
        return VisionDecision(decision=DECISION_UNKNOWN, reason_code="invalid_bbox")

    loop = asyncio.get_running_loop()
    crop1, crop2 = await asyncio.gather(
        loop.run_in_executor(
            None, _crop_table, pdf_path_t1, page_t1, b1, bottom_extension
        ),
        loop.run_in_executor(
            None, _crop_table, pdf_path_t2, page_t2, b2, bottom_extension
        ),
    )

    if not crop1 or not crop2:
        return VisionDecision(decision=DECISION_UNKNOWN, reason_code="crop_failed")

    crop1_b64 = base64.standard_b64encode(crop1).decode("ascii")
    crop2_b64 = base64.standard_b64encode(crop2).decode("ascii")

    messages = _build_messages(
        crop1_b64,
        crop2_b64,
        title_t1=title_t1,
        title_t2=title_t2,
        section_t1=section_t1,
        section_t2=section_t2,
    )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=512,
        )
        raw = response.choices[0].message.content or "{}"
        return _parse_vision_response(raw)
    except Exception as e:
        logger.warning("Vision pair validation API error: %s", e)
        return VisionDecision(
            decision=DECISION_UNKNOWN,
            confidence=0.0,
            reason_code=f"api_error:{type(e).__name__}",
        )


# ---------------------------------------------------------------------------
# Sync wrapper — backward-compatible with existing comparison_runner calls
# ---------------------------------------------------------------------------


def validate_pair_same_concept(
    pdf_path_t1: str,
    page_t1: int,
    bbox_t1: Any,
    pdf_path_t2: str,
    page_t2: int,
    bbox_t2: Any,
    api_key: str,
    bottom_extension: float = 0.12,
    title_t1: str | None = None,
    title_t2: str | None = None,
    section_t1: str | None = None,
    section_t2: str | None = None,
) -> tuple[bool, float]:
    """Sync wrapper returning legacy (same_concept, confidence) tuple.

    'unknown' decisions map to (True, 0.0) to preserve current fail-open
    behavior at call sites that haven't migrated yet.
    """
    coro = validate_pair_same_concept_async(
        pdf_path_t1,
        page_t1,
        bbox_t1,
        pdf_path_t2,
        page_t2,
        bbox_t2,
        api_key,
        bottom_extension=bottom_extension,
        title_t1=title_t1,
        title_t2=title_t2,
        section_t1=section_t1,
        section_t2=section_t2,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(asyncio.run, coro).result()
    else:
        result = asyncio.run(coro)

    return result.as_legacy_tuple()


def validate_pair_full(
    pdf_path_t1: str,
    page_t1: int,
    bbox_t1: Any,
    pdf_path_t2: str,
    page_t2: int,
    bbox_t2: Any,
    api_key: str,
    bottom_extension: float = 0.12,
    title_t1: str | None = None,
    title_t2: str | None = None,
    section_t1: str | None = None,
    section_t2: str | None = None,
) -> VisionDecision:
    """Sync call returning the full VisionDecision (tri-state)."""
    coro = validate_pair_same_concept_async(
        pdf_path_t1,
        page_t1,
        bbox_t1,
        pdf_path_t2,
        page_t2,
        bbox_t2,
        api_key,
        bottom_extension=bottom_extension,
        title_t1=title_t1,
        title_t2=title_t2,
        section_t1=section_t1,
        section_t2=section_t2,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Rescue: parallel 1-vs-1 evaluation of candidates
# ---------------------------------------------------------------------------


@dataclass
class RescueCandidate:
    """Minimal info needed to identify a T2 candidate for rescue."""

    uid: str
    pdf_path: str
    page: int
    bbox: Any
    title: str | None = None
    section: str | None = None


@dataclass
class RescueResult:
    """Result of a rescue evaluation."""

    best_candidate_uid: str | None = None
    decision: VisionDecision | None = None
    all_results: list[tuple[str, VisionDecision]] = field(default_factory=list)


async def evaluate_rescue_candidates_async(
    pdf_path_t1: str,
    page_t1: int,
    bbox_t1: Any,
    title_t1: str | None,
    section_t1: str | None,
    candidates: list[RescueCandidate],
    api_key: str,
    bottom_extension: float = 0.12,
    confidence_min: float = 0.75,
    max_concurrent: int = 5,
) -> RescueResult:
    """Evaluate multiple T2 candidates against a single T1, in parallel 1-vs-1.

    Returns the best matching candidate (highest confidence above threshold),
    or None if no candidate matches.
    """
    if not candidates:
        return RescueResult()

    sem = asyncio.Semaphore(max_concurrent)

    async def _evaluate_one(cand: RescueCandidate) -> tuple[str, VisionDecision]:
        async with sem:
            vd = await validate_pair_same_concept_async(
                pdf_path_t1,
                page_t1,
                bbox_t1,
                cand.pdf_path,
                cand.page,
                cand.bbox,
                api_key,
                bottom_extension=bottom_extension,
                title_t1=title_t1,
                title_t2=cand.title,
                section_t1=section_t1,
                section_t2=cand.section,
            )
            return cand.uid, vd

    tasks = [_evaluate_one(c) for c in candidates]
    raw_results = await asyncio.gather(*tasks)

    all_results = list(raw_results)
    best_uid: str | None = None
    best_vd: VisionDecision | None = None
    best_conf = 0.0

    for uid, vd in all_results:
        if vd.decision == DECISION_MATCH and vd.confidence >= confidence_min:
            if vd.confidence > best_conf:
                best_uid = uid
                best_vd = vd
                best_conf = vd.confidence

    return RescueResult(
        best_candidate_uid=best_uid,
        decision=best_vd,
        all_results=all_results,
    )


# ---------------------------------------------------------------------------
# Phase 5: Global bijective assignment (anti-collision T1 <-> T2)
# ---------------------------------------------------------------------------


@dataclass
class AssignmentEntry:
    """A scored candidate pairing for the global assignment solver."""

    t1_uid: str
    t2_uid: str
    confidence: float
    decision: VisionDecision
    source: str = ""  # "heuristic", "rescue", "validation"


@dataclass
class BijectionResult:
    """Output of the global bijection solver."""

    assigned_pairs: list[AssignmentEntry] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    unmatched_t1: list[str] = field(default_factory=list)
    unmatched_t2: list[str] = field(default_factory=list)


def resolve_bijective_assignment(
    entries: list[AssignmentEntry],
) -> BijectionResult:
    """Resolve a list of scored (t1, t2) candidates into a bijective assignment.

    Each T1 is assigned to at most one T2 and vice-versa.
    When conflicts exist (multiple T1 claim the same T2, or vice-versa),
    the highest-confidence entry wins; others are logged as conflicts.

    Uses a greedy strategy sorted by descending confidence (deterministic
    tie-break on uid pair to ensure reproducibility).
    """
    if not entries:
        return BijectionResult()

    # Sort by confidence desc, then by (t1_uid, t2_uid) for deterministic tie-break
    sorted_entries = sorted(
        entries,
        key=lambda e: (-e.confidence, e.t1_uid, e.t2_uid),
    )

    used_t1: dict[str, AssignmentEntry] = {}
    used_t2: dict[str, AssignmentEntry] = {}
    assigned: list[AssignmentEntry] = []
    conflicts: list[dict[str, Any]] = []

    for entry in sorted_entries:
        t1_taken = entry.t1_uid in used_t1
        t2_taken = entry.t2_uid in used_t2

        if not t1_taken and not t2_taken:
            assigned.append(entry)
            used_t1[entry.t1_uid] = entry
            used_t2[entry.t2_uid] = entry
        else:
            conflict_info: dict[str, Any] = {
                "rejected_t1": entry.t1_uid,
                "rejected_t2": entry.t2_uid,
                "rejected_confidence": round(entry.confidence, 4),
                "rejected_source": entry.source,
                "reason": [],
            }
            if t1_taken:
                winner = used_t1[entry.t1_uid]
                conflict_info["reason"].append(
                    f"t1 already assigned to {winner.t2_uid} "
                    f"(conf={winner.confidence:.3f}, src={winner.source})"
                )
            if t2_taken:
                winner = used_t2[entry.t2_uid]
                conflict_info["reason"].append(
                    f"t2 already assigned to {winner.t1_uid} "
                    f"(conf={winner.confidence:.3f}, src={winner.source})"
                )
            conflicts.append(conflict_info)

    all_t1 = {e.t1_uid for e in entries}
    all_t2 = {e.t2_uid for e in entries}
    assigned_t1 = {e.t1_uid for e in assigned}
    assigned_t2 = {e.t2_uid for e in assigned}

    return BijectionResult(
        assigned_pairs=assigned,
        conflicts=conflicts,
        unmatched_t1=sorted(all_t1 - assigned_t1),
        unmatched_t2=sorted(all_t2 - assigned_t2),
    )


# ---------------------------------------------------------------------------
# Phase 7: Observability — metrics collector
# ---------------------------------------------------------------------------


@dataclass
class VisionMetrics:
    """Aggregated metrics for a Vision validation/rescue run."""

    total_calls: int = 0
    match_count: int = 0
    no_match_count: int = 0
    unknown_count: int = 0
    rescue_calls: int = 0
    rescue_matched: int = 0
    conflicts_resolved: int = 0
    api_errors: int = 0
    crop_failures: int = 0
    avg_confidence: float = 0.0
    _confidence_sum: float = 0.0

    def record_decision(self, vd: VisionDecision, *, is_rescue: bool = False) -> None:
        """Record a single Vision decision into the metrics."""
        self.total_calls += 1
        if vd.decision == DECISION_MATCH:
            self.match_count += 1
        elif vd.decision == DECISION_NO_MATCH:
            self.no_match_count += 1
        else:
            self.unknown_count += 1

        if is_rescue:
            self.rescue_calls += 1
            if vd.decision == DECISION_MATCH:
                self.rescue_matched += 1

        if vd.reason_code.startswith("api_error"):
            self.api_errors += 1
        elif vd.reason_code == "crop_failed":
            self.crop_failures += 1

        self._confidence_sum += vd.confidence
        self.avg_confidence = (
            self._confidence_sum / self.total_calls if self.total_calls > 0 else 0.0
        )

    def record_bijection(self, result: BijectionResult) -> None:
        """Record bijection conflict stats."""
        self.conflicts_resolved += len(result.conflicts)

    def as_dict(self) -> dict[str, Any]:
        """Export metrics as a plain dict for logging/JSON serialization."""
        return {
            "total_calls": self.total_calls,
            "match_count": self.match_count,
            "no_match_count": self.no_match_count,
            "unknown_count": self.unknown_count,
            "unknown_rate": round(self.unknown_count / self.total_calls, 4)
            if self.total_calls > 0
            else 0.0,
            "rescue_calls": self.rescue_calls,
            "rescue_matched": self.rescue_matched,
            "conflicts_resolved": self.conflicts_resolved,
            "api_errors": self.api_errors,
            "crop_failures": self.crop_failures,
            "avg_confidence": round(self.avg_confidence, 4),
        }
