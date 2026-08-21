"""Protocole commun des localisateurs de tableaux."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import TableLocationResult


class TableLocator(Protocol):
    """Contrat minimal : detecter des ancres de tableaux dans un PDF."""

    def locate(
        self,
        pdf_path: Path,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        reference_text_max_chars: int = 6000,
    ) -> TableLocationResult:
        """Localiser les tableaux et retourner des ancres normalisees."""
