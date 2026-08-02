"""Normalisation et application des plages de pages.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vigilance.extraction.docling_processor")


class PageRangeMixin:
    """Normalisation et application des plages de pages."""

    def _is_page_in_ranges(self, page_num: int, page_ranges: list[tuple[int, int]] | None) -> bool:
        """Verifier si une page est dans les plages cibles."""
        if page_ranges is None:
            return True
        return any(start <= page_num <= end for start, end in page_ranges)

    def _pad_page_ranges(
        self,
        page_ranges: list[tuple[int, int]] | None,
        padding: int,
    ) -> list[tuple[int, int]] | None:
        """Etendre les plages ciblees pour verifier leurs pages limitrophes."""
        normalized = self._normalize_page_ranges(page_ranges)
        if not normalized:
            return None
        safe_padding = max(0, int(padding))
        return [
            (max(1, start - safe_padding), end + safe_padding)
            for start, end in normalized
        ]

    def _normalize_page_ranges(self, page_ranges: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
        """Normaliser les plages pour garantir start >= 1 et end >= start."""
        if not page_ranges:
            return []

        normalized: list[tuple[int, int]] = []
        for idx, page_range in enumerate(page_ranges):
            try:
                start_raw, end_raw = page_range
                start = int(start_raw)
                end = int(end_raw)
            except (TypeError, ValueError):
                logger.warning("Plage de pages invalide ignoree a l'index %s: %s", idx, page_range)
                continue

            start_norm = max(1, start)
            if start_norm != start:
                logger.warning(
                    "Plage de pages corrigee a l'index %s: start=%s -> %s",
                    idx,
                    start,
                    start_norm,
                )

            end_norm = max(start_norm, end)
            if end_norm != end:
                logger.warning(
                    "Plage de pages corrigee a l'index %s: end=%s -> %s",
                    idx,
                    end,
                    end_norm,
                )

            normalized.append((start_norm, end_norm))

        normalized.sort(key=lambda r: (r[0], r[1]))
        return normalized

    def _build_docling_page_range(self, page_ranges: list[tuple[int, int]] | None) -> tuple[int, int] | None:
        """Construire une plage Docling compatible (start, end), indexee a 1."""
        normalized = self._normalize_page_ranges(page_ranges)
        if not normalized:
            return None

        if len(normalized) == 1:
            return normalized[0]

        range_start = min(start for start, _ in normalized)
        range_end = max(end for _, end in normalized)
        logger.warning(
            "Plusieurs plages demandees (%s) compressees en plage englobante (%s-%s) "
            "pour compatibilite Docling; filtrage applicatif conserve.",
            normalized,
            range_start,
            range_end,
        )
        return (range_start, range_end)
