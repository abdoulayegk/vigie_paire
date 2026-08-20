"""Validation indépendante des bornes de sections dans les rapports annuels.

Cette couche complète ``SectionLocator`` sans dupliquer son moteur historique :

1. OpenAI Vision confirme la vraie table des matières.
2. Docling en relit la structure et les numéros de page.
3. Les numéros imprimés sont rapprochés des pages physiques.
4. Vision confirme les transitions sur les pages complètes.

Une borne existante n'est corrigée que lorsque les preuves convergent.
"""

from __future__ import annotations

import logging
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from vigie.support.config import resolve_openai_model

from .genai_toc_detector import (
    AnnualTOCAnalysis,
    GenAITOCDetector,
    PageTransitionValidation,
    TOCBoundaryRole,
)
from .section_taxonomy import canonicalize_section

logger = logging.getLogger(__name__)

_TARGET_TYPES = {"capital_management", "risk_management"}
_MIN_TOC_CONFIDENCE = 0.80
_MIN_TRANSITION_CONFIDENCE = 0.80


@dataclass(frozen=True)
class StructuredTOCEntry:
    """Entrée de TDM indépendante du format de sortie Docling/Vision."""

    title: str
    page: int
    level: int = 0
    source: str = ""


@dataclass
class AnnualBoundaryValidationOutcome:
    """Résultat complet remis au localisateur historique."""

    sections: list
    toc_entries: list[StructuredTOCEntry]
    diagnostics: dict


def _normalize(value: str) -> str:
    """Normaliser un titre pour les rapprochements entre extracteurs."""
    normalized = unicodedata.normalize("NFD", str(value or ""))
    normalized = normalized.encode("ascii", "ignore").decode("utf-8").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _unstutter(value: str) -> str:
    """Réduire les glyphes doublés produits par certains PDF BNC."""
    text = str(value or "")
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 8:
        return text
    pairs = [compact[index : index + 2] for index in range(0, len(compact) - 1, 2)]
    if not pairs:
        return text
    duplicated = sum(1 for pair in pairs if len(pair) == 2 and pair[0] == pair[1])
    if duplicated / len(pairs) < 0.70:
        return text
    return compact[::2]


def _title_similarity(left: str, right: str) -> float:
    """Comparer deux titres en combinant séquence et couverture de mots."""
    left_norm = _normalize(_unstutter(left))
    right_norm = _normalize(_unstutter(right))
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    containment = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return 0.60 * sequence + 0.40 * containment


def parse_docling_toc_markdown(markdown: str, max_pages: int) -> list[StructuredTOCEntry]:
    """Extraire les couples titre/page des tableaux Markdown produits par Docling."""
    entries: list[StructuredTOCEntry] = []
    pending_titles: dict[int, str] = {}
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(re.fullmatch(r"[-: ]+", cell or "-") for cell in cells):
            continue

        for index in range(0, len(cells) - 1, 2):
            title = cells[index].strip()
            page_text = cells[index + 1].strip()
            if title and not page_text:
                # Docling conserve souvent un grand chapitre dans une cellule
                # sans page; sa première sous-entrée numérotée donne alors le
                # vrai début imprimé (cas TD).
                pending_titles[index] = title
                continue
            if not title or not re.fullmatch(r"\d{1,3}", page_text):
                continue
            page = int(page_text)
            if page < 3 or page > max_pages:
                continue
            pending_title = pending_titles.pop(index, "")
            if pending_title:
                entries.append(
                    StructuredTOCEntry(
                        title=re.sub(r"\s+", " ", pending_title),
                        page=page,
                        level=0,
                        source="docling_inferred_group_start",
                    )
                )
            entries.append(
                StructuredTOCEntry(
                    title=re.sub(r"\s+", " ", title),
                    page=page,
                    level=0,
                    source="docling",
                )
            )

    deduplicated: list[StructuredTOCEntry] = []
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        key = (_normalize(entry.title), entry.page)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(entry)
    return deduplicated


def _vision_entries(analysis: AnnualTOCAnalysis, max_pages: int) -> list[StructuredTOCEntry]:
    """Normaliser les entrées libres retournées par Vision."""
    entries: list[StructuredTOCEntry] = []
    for raw in analysis.entries:
        try:
            title = str(raw.get("title") or "").strip()
            page = int(raw.get("page"))
            level = int(raw.get("level") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if title and 3 <= page <= max_pages:
            entries.append(StructuredTOCEntry(title, page, level, "vision"))
    return entries


def _best_entry_match(
    title: str,
    entries: list[StructuredTOCEntry],
    *,
    minimum: float = 0.58,
) -> StructuredTOCEntry | None:
    """Trouver l'entrée structurée la plus proche d'un titre Vision."""
    scored = [(_title_similarity(title, entry.title), entry) for entry in entries]
    if not scored:
        return None
    score, entry = max(scored, key=lambda item: item[0])
    return entry if score >= minimum else None


def reconcile_boundary_roles(
    roles: list[TOCBoundaryRole],
    docling_entries: list[StructuredTOCEntry],
    vision_entries: list[StructuredTOCEntry],
) -> tuple[list[TOCBoundaryRole], list[str]]:
    """Corriger les lectures Vision des pages à partir de la structure Docling."""
    resolved: list[TOCBoundaryRole] = []
    warnings: list[str] = []
    for role in roles:
        start_match = _best_entry_match(role.title_found, docling_entries)
        if start_match is None:
            start_match = _best_entry_match(role.title_found, vision_entries)
        if start_match is None:
            start_match = StructuredTOCEntry(
                role.title_found,
                role.start_page,
                source="vision_boundary_unreconciled",
            )
            warnings.append(f"{role.section_type}:start_uses_vision_boundary")
        successor_match = _best_entry_match(role.successor_title, docling_entries)
        if successor_match is None:
            successor_match = _best_entry_match(role.successor_title, vision_entries)
        if successor_match is None:
            successor_match = StructuredTOCEntry(
                role.successor_title,
                role.successor_page,
                source="vision_boundary_unreconciled",
            )
            warnings.append(f"{role.section_type}:successor_uses_vision_boundary")
        if role.section_type == "capital_management":
            # Le modèle peut sauter des chapitres pairs intercalés entre le
            # capital et les risques (titrisation, instruments financiers,
            # contrôles...). La première entrée Docling postérieure au capital
            # prévaut alors comme candidat de transition; Vision vérifiera
            # ensuite sa page physique complète.
            earlier_successors = [
                entry
                for entry in docling_entries
                if start_match.page < entry.page < successor_match.page
                and _title_similarity(entry.title, start_match.title) < 0.72
            ]
            if earlier_successors:
                first_successor = min(earlier_successors, key=lambda entry: entry.page)
                warnings.append(
                    "capital_management:skipped_toc_successor_"
                    f"{successor_match.page}_reconciled_to_{first_successor.page}"
                )
                successor_match = first_successor
        if start_match is None or successor_match is None:
            warnings.append(f"{role.section_type}:toc_entry_unresolved")
            continue

        if start_match.page != role.start_page:
            warnings.append(f"{role.section_type}:vision_start_page_{role.start_page}_reconciled_to_{start_match.page}")
        if successor_match.page != role.successor_page:
            warnings.append(
                f"{role.section_type}:vision_successor_page_{role.successor_page}_reconciled_to_{successor_match.page}"
            )
        resolved.append(
            TOCBoundaryRole(
                section_type=role.section_type,
                title_found=start_match.title,
                start_page=start_match.page,
                successor_title=successor_match.title,
                successor_page=successor_match.page,
                confidence=role.confidence,
            )
        )
    return resolved, warnings


def _heading_similarity(page_text: str, expected_title: str) -> float:
    """Mesurer si un titre attendu apparaît comme en-tête de page."""
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()][:80]
    candidates = list(lines)
    candidates.extend(f"{lines[index]} {lines[index + 1]}" for index in range(len(lines) - 1))
    return max((_title_similarity(candidate, expected_title) for candidate in candidates), default=0.0)


class AnnualSectionBoundaryValidator:
    """Orchestrateur Docling + Vision pour les bornes annuelles T4."""

    def __init__(
        self,
        bank_code: str,
        year: int,
        *,
        detector: GenAITOCDetector | None = None,
        docling_reader: Callable[[Path, int, int], list[StructuredTOCEntry]] | None = None,
    ):
        """Configure le contexte bancaire et les stratégies de lecture des bornes."""
        self.bank_code = str(bank_code or "").strip().lower()
        self.year = int(year)
        if detector is None:
            try:
                model = resolve_openai_model("default_genai")
            except Exception:
                model = "gpt-5.4"
            detector = GenAITOCDetector(model=model)
        self.detector = detector
        self.docling_reader = docling_reader or self._read_toc_with_docling
        self._transition_cache: dict[tuple[int, str], PageTransitionValidation] = {}

    @staticmethod
    def _read_toc_with_docling(
        pdf_path: Path,
        toc_page: int,
        total_pages: int,
    ) -> list[StructuredTOCEntry]:
        """Convertir uniquement la page TDM avec Docling et lire son tableau."""
        try:
            options = PdfPipelineOptions()
            options.do_ocr = False
            options.do_table_structure = True
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                }
            )
            result = converter.convert(pdf_path, page_range=(toc_page, toc_page))
            markdown = result.document.export_to_markdown()
            return parse_docling_toc_markdown(markdown, max_pages=total_pages)
        except Exception as exc:
            logger.warning("Lecture Docling de la TDM impossible p.%s: %s", toc_page, exc)
            return []

    @property
    def _vision_available(self) -> bool:
        """Indiquer si les validations de pages complètes peuvent être appelées."""
        return bool(getattr(self.detector, "api_key", None))

    def _select_master_toc(
        self,
        pdf_path: Path,
        candidate_pages: list[int],
    ) -> AnnualTOCAnalysis | None:
        """Choisir une TDM principale contenant les deux sections cibles."""
        best_partial: AnnualTOCAnalysis | None = None
        for page in candidate_pages:
            analysis = self.detector.analyze_annual_toc_page(pdf_path, page)
            if not analysis.is_master_toc or analysis.confidence < _MIN_TOC_CONFIDENCE:
                continue
            roles = {role.section_type for role in analysis.boundaries}
            if _TARGET_TYPES.issubset(roles):
                return analysis
            if best_partial is None or analysis.confidence > best_partial.confidence:
                best_partial = analysis
        return best_partial

    @staticmethod
    def _infer_offset(
        sections: list,
        roles: list[TOCBoundaryRole],
    ) -> tuple[int | None, float, list[int]]:
        """Inférer page physique - page imprimée depuis plusieurs débuts connus."""
        sections_by_type = {
            canonicalize_section(section.section_type): section
            for section in sections
            if canonicalize_section(section.section_type) in _TARGET_TYPES
        }
        votes: list[int] = []
        for role in roles:
            section = sections_by_type.get(role.section_type)
            if section is None or not getattr(section, "start_page", None):
                continue
            vote = int(section.start_page) - int(role.start_page)
            if -10 <= vote <= 60:
                votes.append(vote)
        if not votes:
            return None, 0.0, []
        offset = int(round(statistics.median(votes)))
        spread = max(votes) - min(votes)
        confidence = 0.95 if len(votes) >= 2 and spread <= 1 else 0.65 if spread <= 2 else 0.35
        return offset, confidence, votes

    @staticmethod
    def _physical_candidates(
        title: str,
        document_page: int,
        offset: int,
        text_by_page: dict[int, str],
        *,
        existing_page: int | None = None,
    ) -> list[tuple[int, float]]:
        """Classer les pages physiques plausibles à vérifier."""
        predicted = document_page + offset
        page_numbers: set[int] = {page for page in range(predicted - 3, predicted + 4) if page in text_by_page}
        if existing_page and existing_page in text_by_page:
            page_numbers.add(existing_page)

        global_scores = [(page, _heading_similarity(text, title)) for page, text in text_by_page.items() if page > 5]
        page_numbers.update(page for page, score in global_scores if score >= 0.82)
        scored = [(page, _heading_similarity(text_by_page.get(page, ""), title)) for page in page_numbers]
        # La page prédite par l'offset multi-ancre est vérifiée en premier.
        # Les pages suivantes portent souvent un en-tête courant identique au
        # titre du chapitre et ne doivent pas supplanter sa vraie page d'ouverture.
        return sorted(scored, key=lambda item: (abs(item[0] - predicted), -item[1], item[0]))

    def _verify_transition(
        self,
        pdf_path: Path,
        candidates: list[tuple[int, float]],
        *,
        section_type: str,
        expected_title: str,
        offset_confidence: float,
    ) -> tuple[int | None, PageTransitionValidation | None, float]:
        """Vérifier progressivement jusqu'à trois pages candidates."""
        for page, text_score in candidates[:3]:
            if self._vision_available:
                key = (page, _normalize(expected_title))
                validation = self._transition_cache.get(key)
                if validation is None:
                    validation = self.detector.validate_section_transition(
                        pdf_path,
                        max(1, page - 1),
                        page,
                        section_type=section_type,
                        expected_title=expected_title,
                    )
                    self._transition_cache[key] = validation
                if (
                    validation.confirmed
                    and validation.previous_page_belongs_to_prior_section
                    and validation.candidate_page_starts_expected_section
                    and validation.confidence >= _MIN_TRANSITION_CONFIDENCE
                ):
                    return page, validation, text_score
            elif text_score >= 0.92 and offset_confidence >= 0.80:
                return (
                    page,
                    PageTransitionValidation(
                        confirmed=True,
                        confidence=min(0.90, text_score),
                        observed_title=expected_title,
                        previous_page_belongs_to_prior_section=True,
                        candidate_page_starts_expected_section=True,
                        reason="exact_physical_heading_without_vision",
                    ),
                    text_score,
                )
        return None, None, 0.0

    def validate(
        self,
        pdf_path: str | Path,
        sections: list,
        text_by_page: dict[int, str],
        candidate_pages: list[int],
    ) -> AnnualBoundaryValidationOutcome:
        """Valider et, si les preuves convergent, corriger les bornes T4."""
        path = Path(pdf_path)
        diagnostics: dict = {
            "enabled": True,
            "status": "not_validated",
            "toc_page": None,
            "toc_confidence": 0.0,
            "vision_available": self._vision_available,
            "docling_entry_count": 0,
            "vision_entry_count": 0,
            "page_offset": None,
            "offset_confidence": 0.0,
            "offset_votes": [],
            "sections": {},
            "warnings": [],
        }
        if not self._vision_available:
            diagnostics["warnings"].append("openai_vision_unavailable")
            return AnnualBoundaryValidationOutcome(sections, [], diagnostics)

        analysis = self._select_master_toc(path, candidate_pages)
        if analysis is None:
            diagnostics["warnings"].append("master_toc_not_confirmed")
            return AnnualBoundaryValidationOutcome(sections, [], diagnostics)

        diagnostics["toc_page"] = analysis.page_number
        diagnostics["toc_confidence"] = analysis.confidence
        diagnostics["warnings"].extend(analysis.warnings)

        total_pages = max(text_by_page, default=0)
        docling_entries = self.docling_reader(path, analysis.page_number, total_pages)
        vision_entries = _vision_entries(analysis, total_pages)
        diagnostics["docling_entry_count"] = len(docling_entries)
        diagnostics["vision_entry_count"] = len(vision_entries)

        roles, reconciliation_warnings = reconcile_boundary_roles(
            analysis.boundaries,
            docling_entries,
            vision_entries,
        )
        diagnostics["warnings"].extend(reconciliation_warnings)
        if {role.section_type for role in roles} != _TARGET_TYPES:
            diagnostics["warnings"].append("target_boundaries_not_reconciled")
            return AnnualBoundaryValidationOutcome(
                sections,
                docling_entries or vision_entries,
                diagnostics,
            )

        offset, offset_confidence, votes = self._infer_offset(sections, roles)
        diagnostics["page_offset"] = offset
        diagnostics["offset_confidence"] = offset_confidence
        diagnostics["offset_votes"] = votes
        if offset is None or offset_confidence < 0.80:
            diagnostics["warnings"].append("physical_page_offset_ambiguous")
            return AnnualBoundaryValidationOutcome(
                sections,
                docling_entries or vision_entries,
                diagnostics,
            )

        updated = list(sections)
        index_by_type = {
            canonicalize_section(section.section_type): index
            for index, section in enumerate(updated)
            if canonicalize_section(section.section_type) in _TARGET_TYPES
        }
        verified_count = 0

        for role in roles:
            index = index_by_type.get(role.section_type)
            if index is None:
                diagnostics["warnings"].append(f"{role.section_type}:section_missing")
                continue
            section = updated[index]

            start_candidates = self._physical_candidates(
                role.title_found,
                role.start_page,
                offset,
                text_by_page,
                existing_page=getattr(section, "start_page", None),
            )
            start_page, start_validation, start_text_score = self._verify_transition(
                path,
                start_candidates,
                section_type=role.section_type,
                expected_title=role.title_found,
                offset_confidence=offset_confidence,
            )

            successor_candidates = self._physical_candidates(
                role.successor_title,
                role.successor_page,
                offset,
                text_by_page,
            )
            successor_page, successor_validation, successor_text_score = self._verify_transition(
                path,
                successor_candidates,
                section_type=f"successor_of_{role.section_type}",
                expected_title=role.successor_title,
                offset_confidence=offset_confidence,
            )

            section_diagnostics = {
                "toc_start_page": role.start_page,
                "toc_start_title": role.title_found,
                "physical_start_page": start_page,
                "start_verified": bool(start_page),
                "start_text_score": round(start_text_score, 4),
                "toc_successor_page": role.successor_page,
                "toc_successor_title": role.successor_title,
                "physical_successor_page": successor_page,
                "successor_verified": bool(successor_page),
                "successor_text_score": round(successor_text_score, 4),
                "start_validation": asdict(start_validation) if start_validation else None,
                "successor_validation": asdict(successor_validation) if successor_validation else None,
            }
            diagnostics["sections"][role.section_type] = section_diagnostics

            if str(getattr(section, "detection_method", "")).startswith("manual_override"):
                diagnostics["warnings"].append(f"{role.section_type}:manual_override_preserved")
                continue

            changes = {}
            if start_page:
                changes.update(
                    {
                        "start_page": start_page,
                        "title_found": start_validation.observed_title or role.title_found,
                        "confidence": max(float(getattr(section, "confidence", 0.0)), start_validation.confidence),
                        "detection_method": "annual_t4_vision_verified_title",
                    }
                )
                verified_count += 1
            effective_start = int(changes.get("start_page", getattr(section, "start_page", 0)))
            if successor_page and successor_page > effective_start:
                changes.update(
                    {
                        "end_page": successor_page - 1,
                        "end_detection_method": "annual_t4_vision_verified_successor",
                        "detected_span": successor_page - effective_start,
                        "final_span": successor_page - effective_start,
                        "constraint_applied": False,
                        "constraint_reason": "",
                    }
                )
                verified_count += 1
            if changes:
                updated[index] = replace(section, **changes)

        diagnostics["status"] = "verified" if verified_count == 4 else "partial"
        if verified_count < 4:
            diagnostics["warnings"].append(f"verified_boundaries_{verified_count}_of_4")
        return AnnualBoundaryValidationOutcome(
            updated,
            docling_entries or vision_entries,
            diagnostics,
        )
