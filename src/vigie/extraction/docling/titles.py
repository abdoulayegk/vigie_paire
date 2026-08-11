"""Resolution et enrichissement des titres de tableaux.

Extrait de ``docling_processor.py`` sans modification des corps de
methodes. Mixin consomme par ``DoclingProcessor``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber

from vigie.support.utils.matching_normalizer import strip_temporal_expressions

from ..docling_normalization import _extract_table_context_split
from ..table_title_resolver import (
    extract_table_number_and_inline_title,
    is_table_number_line,
    is_unit_context_line,
    resolve_title_from_lines,
)
from .models import ExtractedTable

logger = logging.getLogger("vigie.extraction.docling_processor")


class TableTitleMixin:
    """Resolution et enrichissement des titres de tableaux."""

    @staticmethod
    def _normalize_text_lines(text: str) -> list[str]:
        """Nettoyer un bloc de texte en liste de lignes non vides."""
        if not text:
            return []
        lines = []
        for line in str(text).split("\n"):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _resolve_title_metadata_from_lines(
        self,
        lines: list[str],
        first_row_cells: list[str] | None = None,
    ) -> dict[str, str]:
        """Resoudre le titre semantique via le resolver central."""
        return resolve_title_from_lines(
            lines,
            bank_code=self.bank_code_for_patterns,
            first_row_cells=first_row_cells,
        )

    def _title_quality_score(self, title: str | None) -> int:
        """Evaluer la qualite d'un titre.

        Les lignes purement meta (TABLEAU N seul, unite, date) ont un score faible.
        """
        if not title or not str(title).strip():
            return 0

        value = str(title).strip()
        score = 1

        if not is_unit_context_line(value):
            score += 2
        if not is_table_number_line(value):
            score += 2

        number, inline = extract_table_number_and_inline_title(value)
        if number and inline:
            inline_temporal = bool(strip_temporal_expressions(inline, target="title", aggressive=True).strip())
            if inline_temporal:
                score += 1

        temporal_free = strip_temporal_expressions(value, target="title", aggressive=True)
        if temporal_free.strip():
            score += 1

        if len(value) >= 12:
            score += 1

        return score

    def _resolve_page_title_candidates(self, page_text: str) -> list[dict[str, str]]:
        """Construire les candidats titre sur une page (1 candidat par ligne TABLEAU quand possible)."""
        lines = self._normalize_text_lines(page_text)
        if not lines:
            return []

        number_indices = [idx for idx, line in enumerate(lines) if is_table_number_line(line)]
        candidates: list[dict[str, str]] = []

        if not number_indices:
            candidate = self._resolve_title_metadata_from_lines(lines)
            if candidate.get("title") or candidate.get("table_number") or candidate.get("title_raw"):
                candidates.append(candidate)
            return candidates

        for idx in number_indices:
            start = max(0, idx - 4)
            end = min(len(lines), idx + 4)
            window_lines = lines[start:end]
            candidate = self._resolve_title_metadata_from_lines(window_lines)

            line_number, _ = extract_table_number_and_inline_title(lines[idx])
            if line_number and not candidate.get("table_number"):
                candidate["table_number"] = line_number
            if not candidate.get("title_raw"):
                candidate["title_raw"] = lines[idx]
            if not candidate.get("resolution_method"):
                candidate["resolution_method"] = "layout_anchor"
            candidates.append(candidate)

        # Dedup simple en preservant l'ordre
        deduped: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate.get("table_number", ""), candidate.get("title", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(candidate)

        return deduped

    def _find_table_title(self, table) -> str | None:
        """Extraire le titre du tableau a partir de l'objet tableau Docling."""
        try:
            if hasattr(table, "caption") and table.caption:
                return str(table.caption)
        except Exception:
            pass
        return None

    def _find_table_titles_in_text(self, text_content: str) -> dict[int, list[str]]:
        """Cherche les titres de tableaux (TABLEAU N, TABLE N, TN) dans le texte par page.

        Args:
            text_content: Contenu textuel complet avec marqueurs de page

        Returns:
            Dict {page_num: [liste des titres trouvés sur cette page]}
        """
        titles_by_page = {}

        # Patterns pour tous les formats de tableaux
        patterns = [
            # TD/BMO: TABLEAU 28 : Titre
            re.compile(r"TABLEAU\s+(\d+)\s*[:\-–—]?\s*([^\n]+)", re.IGNORECASE),
            # Anglais: TABLE 28 : Title
            re.compile(r"TABLE\s+(\d+)\s*[:\-–—]?\s*([^\n]+)", re.IGNORECASE),
            # BNS: T5 Titre
            re.compile(r"^T(\d+[A-Za-z]?)\s+([^\n]+)", re.MULTILINE),
        ]

        # Parser le texte page par page
        current_page = 1
        for line in text_content.split("\n"):
            # Détecter les marqueurs de page
            if line.startswith("--- Page ") or line.startswith("## Page "):
                try:
                    current_page = int(re.search(r"Page\s+(\d+)", line).group(1))
                except:
                    pass
                continue

            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    full_title = line.strip()
                    if current_page not in titles_by_page:
                        titles_by_page[current_page] = []
                    titles_by_page[current_page].append(full_title)
                    break

        return titles_by_page

    def _enrich_tables_with_titles(self, tables: list[ExtractedTable], pdf_path: Path) -> list[ExtractedTable]:
        """Enrichit les tableaux sans titre en cherchant dans le texte PDF.

        Args:
            tables: Liste des tableaux extraits
            pdf_path: Chemin vers le PDF source

        Returns:
            Liste des tableaux avec titres enrichis
        """
        tables_by_page: dict[int, list[ExtractedTable]] = {}
        for table in tables:
            tables_by_page.setdefault(table.page_number, []).append(table)

        bank_code = (self.bank_code_for_patterns or "").lower()

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num not in tables_by_page:
                    continue

                page_text = page.extract_text() or ""
                candidates = self._resolve_page_title_candidates(page_text)
                page_tables = tables_by_page[page_num]

                # CIBC: un candidat par tableau (lignes page + first_column de chaque tableau)
                if bank_code == "cibc" and len(page_tables) >= 1 and len(candidates) <= 1:
                    lines = self._normalize_text_lines(page_text)
                    per_table_candidates: list[dict[str, str]] = []
                    for table in page_tables:
                        first_row_cells = list(table.first_column_indicators or [])
                        if not first_row_cells and getattr(table, "rows", None):
                            first_row_cells = [str(row[0]).strip() for row in table.rows if row and len(row) > 0]
                        cand = resolve_title_from_lines(
                            lines,
                            bank_code="cibc",
                            first_row_cells=first_row_cells or None,
                        )
                        per_table_candidates.append(cand)
                    candidates = per_table_candidates

                if not candidates:
                    continue

                by_number: dict[str, list[int]] = {}
                for idx, candidate in enumerate(candidates):
                    number = (candidate.get("table_number") or "").strip()
                    if number:
                        by_number.setdefault(number, []).append(idx)

                available = set(range(len(candidates)))

                for table in page_tables:
                    # Ne pas ecraser le contenu fourni par Vision (seule source de verite).
                    if getattr(table, "extraction_method", None) == "vision_full_gpt4o":
                        continue
                    selected_idx: int | None = None
                    current_number = str(table.table_number or "").strip()

                    # 1) Priorite numero si disponible
                    if current_number and current_number in by_number:
                        for idx in by_number[current_number]:
                            if idx in available:
                                selected_idx = idx
                                break

                    # 2) Fallback positionnel (ordre des tableaux sur la page)
                    if selected_idx is None and available:
                        selected_idx = min(available)

                    if selected_idx is None:
                        continue

                    available.remove(selected_idx)
                    candidate = candidates[selected_idx]

                    candidate_title = (candidate.get("title") or "").strip()
                    candidate_title_raw = (candidate.get("title_raw") or "").strip()
                    candidate_number = (candidate.get("table_number") or "").strip()
                    candidate_unit = (candidate.get("unit_context") or "").strip()
                    candidate_method = (candidate.get("resolution_method") or "").strip()

                    # On remplace si le candidat est clairement meilleur semantiquement.
                    current_title = (table.title or "").strip()
                    if self._title_quality_score(candidate_title) > self._title_quality_score(current_title):
                        table.title = candidate_title or current_title or None
                        table.title_clean = candidate_title or table.title_clean
                    if candidate_method and not table.title_resolution_method:
                        table.title_resolution_method = candidate_method

                    if candidate_title_raw:
                        table.title_raw = candidate_title_raw
                    elif not table.title_raw:
                        table.title_raw = current_title or None

                    if candidate_number and not table.table_number:
                        table.table_number = candidate_number

                    if candidate_unit:
                        table.unit_context = candidate_unit

                    # Si le titre reste vide, fallback explicite sur title_raw.
                    if not table.title and table.title_raw:
                        table.title = table.title_raw
                    if not table.title_clean and table.title:
                        table.title_clean = table.title

        return tables

    def _enrich_tables_with_context(self, tables: list[ExtractedTable], pdf_path: Path) -> list[ExtractedTable]:
        """Enrichir les tableaux avec le contexte textuel avant/apres (pour table_type_classifier).

        Args:
            tables: Liste des tableaux extraits a enrichir.
            pdf_path: Chemin vers le PDF source.

        Returns:
            Liste des tableaux avec ``context_before`` et ``context_after`` renseignes.
        """
        if not pdf_path or not str(pdf_path) or not Path(pdf_path).exists():
            return tables

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for table in tables:
                    page_num = table.page_number
                    if page_num < 1 or page_num > len(pdf.pages):
                        continue
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text() or ""
                    cb, ca = _extract_table_context_split(text, table.title or table.title_clean)
                    table.context_before = cb
                    table.context_after = ca
        except (FileNotFoundError, OSError) as e:
            logger.debug("Skip context enrichment (file unavailable): %s", e)

        return tables

    def _extract_table_number(self, title: str | None) -> tuple[str | None, str | None]:
        """Extrait le numéro du tableau depuis le titre.

        Formats supportés:
        - TD/BMO: TABLEAU 28 : Titre..., TABLE 31 - Title..., TABLEAU 1
        - BNS: T5 Titre..., T11A Titre..., T14A Titre...

        Args:
            title: Titre complet du tableau

        Returns:
            Tuple (numéro, titre_nettoyé) ou (None, titre_original)
        """
        if not title:
            return None, None

        table_number, inline_title = extract_table_number_and_inline_title(title)
        if inline_title:
            return table_number, inline_title
        return table_number, (None if table_number else title)
