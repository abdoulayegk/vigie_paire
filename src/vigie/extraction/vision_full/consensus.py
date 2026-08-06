"""Extraction en double passe et vote de consensus entre variantes de prompt.

Extrait de ``vision_full_extractor.py`` sans modification des corps
de methodes. Mixin consomme par ``VisionFullExtractor``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .prompts import (
    _CONSENSUS_PROMPT_VARIANTS,
    _PROMPT_VARIANT_PRECISION,
    _build_precision_prompt,
)
from .quality_grading import _candidate_quality_score
from .quality_heuristics import _count_real_indicators, _extract_native_text_indicators
from .result import VisionFullResult

logger = logging.getLogger("vigie.extraction.vision_full_extractor")


class ConsensusMixin:
    """Extraction en double passe et vote de consensus entre variantes de prompt."""

    def extract_with_consensus(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        bottom_extension_used: float = 0.0,
        reference_text: str | None = None,
        max_completion_tokens_override: int | None = None,
        rescue_mode: bool = False,
        rescue_instruction: str = "",
        temperatures: tuple[float, ...] | None = None,
    ) -> VisionFullResult | None:
        """Extraction multi-tir avec vote par consensus.

        Lance 2 extractions paralleles avec variantes de prompt
        (exhaustive et precision) a temperature 0.0, puis selectionne
        le resultat avec le meilleur consensus sur le nombre
        d'indicateurs et le recouvrement des libelles.

        Se rabat sur une extraction unique lorsque le consensus
        est unanime ou qu'un seul tir reussit.
        """
        temps = temperatures or self._CONSENSUS_TEMPERATURES
        if len(temps) <= 1:
            return self.extract(
                crop_bytes=crop_bytes,
                bank_code=bank_code,
                pdf_sha=pdf_sha,
                page_number=page_number,
                bbox_norm=bbox_norm,
                vision_cfg=vision_cfg,
                bottom_extension_used=bottom_extension_used,
                reference_text=reference_text,
                max_completion_tokens_override=max_completion_tokens_override,
                rescue_mode=rescue_mode,
                rescue_instruction=rescue_instruction,
                temperature=temps[0],
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _shot(variant: str) -> VisionFullResult | None:
            """Exécute un tir d'extraction Vision pour une variante de prompt donnée."""
            if variant == _PROMPT_VARIANT_PRECISION:
                prompt_override = _build_precision_prompt(
                    bank_code,
                    vision_cfg or {},
                    reference_text=reference_text,
                )
            else:
                prompt_override = None  # exhaustive uses default prompt
            return self.extract(
                crop_bytes=crop_bytes,
                bank_code=bank_code,
                pdf_sha=pdf_sha,
                page_number=page_number,
                bbox_norm=bbox_norm,
                vision_cfg=vision_cfg,
                bottom_extension_used=bottom_extension_used,
                reference_text=reference_text,
                max_completion_tokens_override=max_completion_tokens_override,
                rescue_mode=rescue_mode,
                rescue_instruction=rescue_instruction,
                temperature=0.0,
                prompt_override=prompt_override,
            )

        results: list[VisionFullResult] = []
        with ThreadPoolExecutor(max_workers=len(_CONSENSUS_PROMPT_VARIANTS)) as pool:
            futures = {pool.submit(_shot, v): v for v in _CONSENSUS_PROMPT_VARIANTS}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r is not None and r.indicators:
                        results.append(r)
                except Exception as exc:
                    logger.debug(
                        "Consensus shot variant=%s failed: %s",
                        futures[future],
                        exc,
                    )

        if not results:
            return None
        if len(results) == 1:
            return results[0]

        # --- Consensus voting ---
        text_vote = _extract_native_text_indicators(reference_text) if reference_text else None
        return self._select_consensus(results, bbox_norm=bbox_norm, text_vote_indicators=text_vote)

    @staticmethod
    def _select_consensus(
        results: list[VisionFullResult],
        *,
        bbox_norm: list[float] | None = None,
        text_vote_indicators: list[str] | None = None,
    ) -> VisionFullResult:
        """Selectionne le resultat avec le meilleur consensus sur les libelles d'indicateurs.

        Strategie :
        1. Calcule le nombre *median* d'indicateurs sur tous les tirs.
        2. Construit un sur-ensemble de libelles normalises et compte les votes
           par libelle.
        3. Note chaque resultat selon : (a) proximite au compte median, (b) somme
           des votes pour ses indicateurs (popularite des libelles).
        4. Retourne le resultat avec le score composite le plus eleve.
        """
        import statistics

        counts = [_count_real_indicators(r.indicators or []) for r in results]
        median_count = statistics.median(counts)

        # Normalise labels for comparison
        def _norm(label: str) -> str:
            """Normalise un libellé pour comparaison de consensus (lowercase + trim)."""
            return label.strip().lower()

        # Collect vote counts per normalised label
        label_votes: dict[str, int] = {}
        for r in results:
            for ind in r.indicators or []:
                key = _norm(str(ind))
                if key:
                    label_votes[key] = label_votes.get(key, 0) + 1

        # Text vote: reinforce labels confirmed by native PDF text
        text_negative_signal: list[str] = []
        if text_vote_indicators:
            norm_vision_labels = set(label_votes.keys())
            for candidate in text_vote_indicators:
                key = _norm(candidate)
                if key in norm_vision_labels:
                    label_votes[key] = label_votes.get(key, 0) + 1
            # Negative signal: text candidates not seen in any Vision shot
            for candidate in text_vote_indicators:
                key = _norm(candidate)
                if key and key not in norm_vision_labels:
                    text_negative_signal.append(candidate)
            if text_negative_signal:
                logger.info(
                    "Consensus text vote: %d native text candidates not found in any Vision shot",
                    len(text_negative_signal),
                )

        def _score(r: VisionFullResult) -> float:
            """Calcule le score composite (proximité médiane + popularité + qualité) d'un résultat."""
            real_count = _count_real_indicators(r.indicators or [])
            # Penalty for deviating from the median count
            count_penalty = abs(real_count - median_count)
            # Popularity: sum of votes for this result's labels
            popularity = sum(label_votes.get(_norm(str(ind)), 0) for ind in (r.indicators or []))
            # Quality bonus from existing scoring function (tuple of ints)
            quality_tuple = _candidate_quality_score(
                r,
                bbox_norm=bbox_norm,
                expected_footnote_ids=set(),
                baseline_result=None,
            )
            return popularity - count_penalty * 3 + sum(quality_tuple) * 0.5

        best = max(results, key=_score)

        # Compute confidence as mean Jaccard agreement between best and each shot
        best_labels = {_norm(str(ind)) for ind in (best.indicators or []) if _norm(str(ind))}
        if best_labels and len(results) > 1:
            jaccards = []
            for r in results:
                shot_labels = {_norm(str(ind)) for ind in (r.indicators or []) if _norm(str(ind))}
                union = best_labels | shot_labels
                intersection = best_labels & shot_labels
                jaccards.append(len(intersection) / len(union) if union else 1.0)
            confidence = sum(jaccards) / len(jaccards)
        else:
            confidence = 1.0
        best = replace(best, confidence_score=round(confidence, 3))

        logger.info(
            "Consensus: selected result with %d indicators (median=%s, from %d shots)",
            _count_real_indicators(best.indicators or []),
            median_count,
            len(results),
        )
        return best
