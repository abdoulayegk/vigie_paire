"""Localisateur de sections pour identifier les pages des sections cibles dans les rapports bancaires.

Ce module detecte automatiquement les pages des sections:
- Gestion des risques
- Gestion du capital / fonds propres

Strategies de detection (par ordre de priorite):
1. Override manuel (configuration bank_profiles.json)
2. Detection via la Table des matieres (TDM) complete
3. Detection des sections suivantes (pour determiner la fin)
4. Scan des titres de sections dans le PDF

L'implementation est repartie dans le sous-package ``locator`` : chaque
responsabilite y vit dans son module sous forme de mixin, assemble ici.
Ce module reste la facade publique et re-exporte tout ce qui etait accessible
avant le decoupage.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .locator.annual_t4 import AnnualT4Mixin
from .locator.bank_config import (  # noqa: F401 - re-export de compatibilite
    BANK_SECTION_NAMES,
    BankConfigMixin,
    _get_bank_section_names,
    _load_bank_config,
)
from .locator.banks_cibc import CibcRefinementMixin
from .locator.bounds import BoundsMixin
from .locator.genai_fallback import GenAIFallbackMixin
from .locator.models import (
    SHARED_PAGE_TOP_THRESHOLD,
    LocatedSection,
    SectionMapping,
    TocEntry,
    VisualTextElement,
    normalize_text,
)
from .locator.patterns import (
    FOLLOWING_SECTION_PATTERNS,
    RISK_SUBSECTIONS,
    SECTION_PATTERNS,
    SECTION_TITLE_ALIASES,
    T4_SECTION_TITLE_PROFILES,
    TOC_PATTERNS,
)
from .locator.title_scan import TitleScanMixin
from .locator.toc_parser import TocParserMixin
from .locator.validation import ValidationMixin
from .locator.visual_layout import VisualLayoutMixin
from .section_taxonomy import canonicalize_section

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")

__all__ = [
    "BANK_SECTION_NAMES",
    "FOLLOWING_SECTION_PATTERNS",
    "RISK_SUBSECTIONS",
    "SECTION_PATTERNS",
    "SECTION_TITLE_ALIASES",
    "SHARED_PAGE_TOP_THRESHOLD",
    "T4_SECTION_TITLE_PROFILES",
    "TOC_PATTERNS",
    "LocatedSection",
    "SectionLocator",
    "SectionMapping",
    "TocEntry",
    "VisualTextElement",
    "locate_sections_in_pdf",
    "normalize_text",
]


class SectionLocator(
    BankConfigMixin,
    TocParserMixin,
    VisualLayoutMixin,
    TitleScanMixin,
    AnnualT4Mixin,
    BoundsMixin,
    ValidationMixin,
    CibcRefinementMixin,
    GenAIFallbackMixin,
):
    """Localisateur de sections dans les rapports bancaires.

    Utilise une approche hybride a 3 niveaux:
    1. Override manuel (configuration bank_profiles.json)
    2. Detection via la Table des matieres (TDM) complete
    3. Detection des sections suivantes + scan des titres

    L'implementation est repartie dans les mixins du sous-package ``locator`` ;
    seuls l'initialisation et l'orchestration restent ici.
    """

    def __init__(self, bank_code: str | None = None, quarter: str | None = None, year: int = 2025):
        """Initialiser le localisateur.

        Args:
            bank_code: Code de la banque pour utiliser les patterns specifiques
            quarter: Trimestre (t1, t2, t3) pour les overrides manuels
            year: Annee pour les overrides manuels
        """
        self.bank_code = bank_code
        self.quarter = quarter
        self.year = year
        self.bank_config = _load_bank_config()
        self._compile_patterns()
        self._load_following_patterns()

    def locate_sections(self, pdf_path: str | Path) -> SectionMapping:
        """Localiser les sections cibles dans un PDF.

        Strategie hybride a 3 niveaux:
        1. Verifier les overrides manuels (configuration)
        2. Parser la TDM complete pour les limites exactes
        3. Scanner le PDF et detecter les sections suivantes

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            SectionMapping avec les sections localisees
        """
        raw_pdf_path = str(pdf_path or "").strip()
        if not raw_pdf_path:
            raise ValueError("Chemin PDF requis pour la localisation des sections.")
        pdf_path = Path(raw_pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trouve: {pdf_path}")

        logger.info(f"Localisation des sections dans: {pdf_path}")

        # Extraire le texte du PDF
        text_by_page = self._extract_text_by_page(pdf_path)
        total_pages = len(text_by_page)

        # ETAPE 1: Parser la TDM complete
        toc_entries = self._parse_full_toc(text_by_page)
        logger.info(f"TDM: {len(toc_entries)} entrees trouvees")

        toc_sections = []
        toc_score = 0.0
        toc_reliable = False
        toc_used = False
        override_applied = False
        if toc_entries:
            toc_sections = self._detect_sections_from_full_toc(toc_entries)
            toc_score = self._assess_toc_quality(toc_entries, toc_sections, total_pages)
            toc_reliable = toc_score >= 0.6
            logger.info(f"TDM: score fiabilite {toc_score:.2f} -> {'fiable' if toc_reliable else 'faible'}")

        # ETAPE 2: Chercher les sections cibles
        sections = []
        found_types = set()
        visual_elements: dict[int, list[VisualTextElement]] | None = None

        # ETAPE 2: TDM en priorite si fiable
        if toc_reliable:
            for toc_section in toc_sections:
                if toc_section.section_type not in found_types:
                    if toc_section.start_page > 5:
                        sections.append(toc_section)
                        found_types.add(toc_section.section_type)
                        toc_used = True
                        logger.info(f"Section {toc_section.section_type}: TDM page {toc_section.start_page}")

        # ETAPE 2.1: Overrides manuels en garde-fou
        for section_type in [
            "gestion_capital",
            "gestion_risques",
            "gestion_reglementation",
        ]:
            override = self._get_manual_override(section_type)
            if not (override and override[0] and override[1]):
                continue

            apply_override = False
            if not toc_reliable:
                apply_override = True
            else:
                existing = next((s for s in sections if s.section_type == section_type), None)
                if existing is None:
                    apply_override = True
                elif self._is_section_bounds_suspicious(existing, total_pages):
                    apply_override = True

            if apply_override:
                sections = [s for s in sections if s.section_type != section_type]
                section = LocatedSection(
                    section_type=section_type,
                    title_found=f"[Override manuel {self.quarter}_{self.year}]",
                    start_page=override[0],
                    end_page=override[1],
                    confidence=1.0,
                    detection_method="manual_override_guardrail" if toc_reliable else "manual_override",
                    end_detection_method="manual_override_guardrail" if toc_reliable else "manual_override",
                )
                sections.append(section)
                found_types.add(section_type)
                override_applied = True
                logger.info(f"Section {section_type}: override manuel pages {override[0]}-{override[1]}")

        # ETAPE 2.2: Si TDM faible mais disponible, l'utiliser apres override
        if toc_entries and not toc_reliable:
            for toc_section in toc_sections:
                if toc_section.section_type not in found_types:
                    if toc_section.start_page > 5:
                        sections.append(toc_section)
                        found_types.add(toc_section.section_type)
                        toc_used = True
                        logger.info(f"Section {toc_section.section_type}: TDM page {toc_section.start_page}")

        # Ensuite scanner le PDF pour les sections non trouvees dans la TDM
        scanned_sections = self._scan_section_titles(text_by_page)
        for scanned in scanned_sections:
            if scanned.section_type not in found_types:
                # Valider que la page est raisonnable (pas dans les 5 premieres pages)
                if scanned.start_page > 5:
                    sections.append(scanned)
                    found_types.add(scanned.section_type)

        # NOUVEAU: Detection visuelle (pdfplumber) pour les sections non encore trouvees
        # Utilise taille de police, gras, position pour identifier les titres
        if len(found_types) < 2:
            logger.info("Detection visuelle activee pour sections manquantes...")
            visual_elements = self._extract_visual_elements(pdf_path)
            if visual_elements:
                visual_sections = self._detect_section_headers_visual(visual_elements, text_by_page)
                for visual_section in visual_sections:
                    if (
                        visual_section.section_type == "gestion_reglementation"
                        and not self._bank_has_regulatory_section()
                    ):
                        continue
                    if visual_section.section_type not in found_types:
                        if visual_section.start_page > 5:
                            sections.append(visual_section)
                            found_types.add(visual_section.section_type)
                            logger.info(
                                f"Section {visual_section.section_type}: detection visuelle "
                                f"page {visual_section.start_page} (conf={visual_section.confidence:.2f})"
                            )

        # ETAPE 2.5 (NOUVEAU): Fallback GenAI si confiance faible ou sections manquantes
        if self._needs_genai_fallback(sections):
            logger.info("Activation du fallback GenAI pour sections manquantes...")
            genai_sections = self._detect_with_genai(pdf_path)
            for genai_section in genai_sections:
                if genai_section.section_type == "gestion_reglementation" and not self._bank_has_regulatory_section():
                    continue
                if genai_section.section_type not in found_types:
                    if genai_section.start_page > 5:
                        sections.append(genai_section)
                        found_types.add(genai_section.section_type)
                        logger.info(
                            f"Section {genai_section.section_type}: GenAI fallback "
                            f"page {genai_section.start_page} (conf={genai_section.confidence:.2f})"
                        )

        # T4 annuel: la TDM sert à trouver le chapitre, mais les bornes sont
        # confirmées sur les pages physiques du PDF. Cette étape ne s'applique
        # pas aux rapports T1-T3.
        sections = self._rebase_annual_t4_section_starts(sections, text_by_page, total_pages)

        # ETAPE 3: Determiner les pages de fin avec la logique hybride
        sections = self._determine_end_pages(sections, text_by_page, toc_entries, total_pages)

        # ETAPE 4: Validation croisee multi-methodes (Amélioration 1)
        # Cette etape utilise aussi la validation contextuelle (Amélioration 2)
        # Note: L'affinage via sous-sections (Amélioration 3) est deja fait dans _determine_end_pages()
        sections = self._validate_with_cross_reference(sections, toc_entries, scanned_sections, text_by_page)

        # ETAPE 4.6: Re-appliquer les contraintes de longueur apres validation croisee
        # (la correction multi-methodes peut deplacer les bornes).
        sections = [self._apply_section_length_constraints(s, total_pages, source="post_validation") for s in sections]

        # ETAPE 4.5: Offset numerotation document -> physique (CIBC et autres banques avec offset)
        # ============================================================================
        # L'offset s'applique UNIQUEMENT aux sections dont les numeros sont en numerotation
        # document (toc, manual_override). Les methodes scan/genai/visual donnent deja
        # des numeros physiques -> pas d'offset pour eviter double application.
        #   page_document 20 + offset 3 = page_physique 23
        # ============================================================================
        offset = self._get_page_number_offset()
        if offset > 0:
            bank_name = (self.bank_code or "unknown").upper()
            logger.info(
                f"[{bank_name}] Offset de numerotation: +{offset} pages "
                f"(methodes document uniquement: toc, manual_override)"
            )
            adjusted = []
            for s in sections:
                if self._uses_document_page_numbers(s.detection_method):
                    new_start = s.start_page + offset
                    new_end = (s.end_page + offset) if s.end_page is not None else None
                    adjusted.append(
                        replace(
                            s,
                            start_page=new_start,
                            end_page=new_end,
                        )
                    )
                    logger.info(
                        f"  -> {s.section_type} ({s.detection_method}): "
                        f"p.{s.start_page}-{s.end_page or '?'} -> physique p.{new_start}-{new_end or '?'}"
                    )
                else:
                    adjusted.append(s)
                    logger.debug(
                        f"  -> {s.section_type} ({s.detection_method}): "
                        f"p.{s.start_page}-{s.end_page or '?'} (deja physique, pas d'offset)"
                    )
            sections = adjusted

        # ETAPE 4.7: Recalage specifique CIBC des 2 sections cibles sur titres reels
        sections = self._refine_cibc_target_sections(sections, text_by_page, total_pages)

        # ETAPE 4.75: Pour les rapports annuels, valider la TDM et les
        # transitions physiques avec la couche Docling + Vision indépendante.
        boundary_validation: dict = {}
        if self._is_t4_quarter():
            try:
                from .annual_section_boundary_validator import AnnualSectionBoundaryValidator

                front_pages = list(range(1, min(31, total_pages + 1)))
                candidate_pages = sorted(
                    front_pages,
                    key=lambda page: (
                        self._score_toc_candidate_page(
                            page,
                            text_by_page.get(page, ""),
                        ),
                        1 if 15 <= page <= 20 else 0,
                        1 if 10 <= page <= 25 else 0,
                        -abs(page - 17),
                    ),
                    reverse=True,
                )[:12]
                outcome = AnnualSectionBoundaryValidator(
                    bank_code=self.bank_code or "",
                    year=self.year,
                ).validate(
                    pdf_path,
                    sections,
                    text_by_page,
                    candidate_pages,
                )
                sections = outcome.sections
                boundary_validation = outcome.diagnostics
                if outcome.toc_entries:
                    toc_entries = [
                        TocEntry(
                            title=entry.title,
                            page=entry.page,
                            level=entry.level,
                            raw_line=f"[{entry.source}] {entry.title} {entry.page}",
                        )
                        for entry in outcome.toc_entries
                    ]
                    toc_sections = self._detect_sections_from_full_toc(toc_entries)
                    toc_score = self._assess_toc_quality(toc_entries, toc_sections, total_pages)
                    toc_reliable = toc_score >= 0.6
                    toc_used = toc_used or boundary_validation.get("status") in {
                        "verified",
                        "partial",
                    }
            except Exception as exc:
                logger.warning("Validation annuelle Docling + Vision indisponible: %s", exc)
                boundary_validation = {
                    "enabled": True,
                    "status": "error",
                    "warnings": [str(exc)],
                }

        # ETAPE 4.8: Normaliser la taxonomie des sections en sortie.
        for section in sections:
            section.section_type = canonicalize_section(section.section_type)

        # ETAPE 4.85: Étendre les sections sur les pages partagées avec ancre de fin.
        if sections:
            if visual_elements is None:
                visual_elements = self._extract_visual_elements(pdf_path)
            if visual_elements:
                sections = self._refine_shared_page_boundaries(sections, toc_entries, visual_elements)

        # ETAPE 4.9: Resoudre une ancre intra-page sur le vrai bloc titre.
        if sections:
            if visual_elements is None:
                visual_elements = self._extract_visual_elements(pdf_path)
            if visual_elements:
                sections = self._resolve_section_anchors(sections, visual_elements)
            else:
                sections = [replace(section, anchor_found=False) for section in sections]

        # Creer le mapping
        mapping = SectionMapping(
            bank_code=self.bank_code or "",
            bank_name="",  # Sera rempli par l'appelant
            quarter=self.quarter or "",
            year=self.year,
            file_path=str(pdf_path),
            sections=sections,
            total_pages=total_pages,
            toc_entries=toc_entries,
            toc_score=toc_score,
            toc_reliable=toc_reliable,
            toc_used=toc_used,
            override_applied=override_applied,
            boundary_validation=boundary_validation,
        )

        logger.info(f"Sections localisees: {len(sections)}")
        for section in sections:
            logger.info(
                f"  - {section.section_type}: pages {section.start_page}-{section.end_page} "
                f"(debut: {section.detection_method}, fin: {section.end_detection_method}, "
                f"confiance {section.confidence:.2f}, ancre={'ok' if section.anchor_found else 'missing'})"
            )

        return mapping

    def _extract_text_by_page(self, pdf_path: Path) -> dict[int, str]:
        """Extraire le texte de chaque page du PDF.

        Args:
            pdf_path: Chemin vers le PDF

        Returns:
            Dict {page_number: texte}
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber non installe")
            return {}

        text_by_page = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text_by_page[page_num] = text

        return text_by_page


def locate_sections_in_pdf(
    pdf_path: str | Path,
    bank_code: str | None = None,
    quarter: str | None = None,
    year: int = 2025,
) -> SectionMapping:
    """Fonction utilitaire pour localiser les sections dans un PDF.

    Args:
        pdf_path: Chemin vers le PDF
        bank_code: Code de la banque (optionnel)
        quarter: Trimestre (optionnel)
        year: Annee (defaut 2025)

    Returns:
        SectionMapping avec les sections localisees
    """
    locator = SectionLocator(bank_code=bank_code, quarter=quarter, year=year)
    return locator.locate_sections(pdf_path)
