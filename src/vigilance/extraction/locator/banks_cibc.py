"""Ajustements de detection propres a CIBC.

Extrait de ``section_locator.py`` sans modification des corps de methodes.
Mixin consomme par ``SectionLocator``.
"""

from __future__ import annotations

import logging

from .models import LocatedSection

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")


class CibcRefinementMixin:
    """Ajustements de detection propres a CIBC."""

    def _refine_cibc_target_sections(
        self,
        sections: list[LocatedSection],
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> list[LocatedSection]:
        """Recaler les bornes des sections CIBC pour les sections capital et risques.

        Le recalage est fait sur les pages physiques:
        1) debut reel par recherche du titre autour de la page estimee
        2) fin capital = debut risque - 1
        3) fin risques = page avant le prochain grand titre

        Args:
            sections: Sections detectees a recaler
            text_by_page: Texte du PDF indexe par numero de page
            total_pages: Nombre total de pages du document

        Returns:
            Liste de sections avec bornes recalees pour CIBC.
        """
        if self.bank_code != "cibc" or not sections:
            return sections

        target_types = {"gestion_capital", "gestion_risques"}
        adjusted: list[LocatedSection] = []

        for section in sections:
            if section.section_type not in target_types:
                adjusted.append(section)
                continue

            found_start = None
            if not section.detection_method.startswith(("manual_override", "scan_exact")):
                section_names = self._get_config_section_names(section.section_type)
                found_start = self._find_section_start_in_window(
                    estimated_page=section.start_page,
                    text_by_page=text_by_page,
                    section_names=section_names,
                    total_pages=total_pages,
                )

            new_start = found_start if found_start else section.start_page
            detection_method = section.detection_method
            if found_start and found_start != section.start_page:
                detection_method = f"{section.detection_method}_cibc_recalibrated"
                logger.info(f"[CIBC] Recalage {section.section_type}: p.{section.start_page} -> p.{found_start}")

            adjusted.append(
                LocatedSection(
                    section_type=section.section_type,
                    title_found=section.title_found,
                    start_page=new_start,
                    end_page=section.end_page,
                    confidence=section.confidence,
                    detection_method=detection_method,
                    end_detection_method=section.end_detection_method,
                    detected_span=section.detected_span,
                    final_span=section.final_span,
                    constraint_applied=section.constraint_applied,
                    constraint_reason=section.constraint_reason,
                )
            )

        # Enchainement explicite des 2 sections cibles (si presentes)
        by_type = {s.section_type: s for s in adjusted}
        capital = by_type.get("gestion_capital")
        risk = by_type.get("gestion_risques")

        if (
            capital
            and risk
            and risk.start_page > capital.start_page
            and not capital.detection_method.startswith("manual_override")
            and not self._is_annual_t4_target_section(capital)
            and capital.end_detection_method != "following_section_scan"
        ):
            capital.end_page = risk.start_page - 1
            capital.end_detection_method = "cibc_next_section_start"
            self._apply_section_length_constraints(capital, total_pages, source="cibc_recalibration")

        if (
            risk
            and not self._is_annual_t4_target_section(risk)
            and not risk.detection_method.startswith("manual_override")
        ):
            next_header = self._find_next_header_page(
                section_type="gestion_risques",
                start_page=risk.start_page,
                text_by_page=text_by_page,
                total_pages=total_pages,
            )
            if next_header and next_header > risk.start_page:
                risk.end_page = next_header - 1
                risk.end_detection_method = "cibc_next_section_header"
                self._apply_section_length_constraints(risk, total_pages, source="cibc_recalibration")

        # Conserver un ordre stable par page de debut
        adjusted.sort(key=lambda s: s.start_page)
        return adjusted

    def _bank_has_regulatory_section(self) -> bool:
        """Indiquer si la banque courante a une section reglementation (gestion_reglementation).

        Seules les banques listees dans banks_with_regulatory (RBC, Scotia, BMO)
        peuvent avoir des sections de type gestion_reglementation.
        BNC, CIBC, TD n'ont que capital et risque.

        Returns:
            True si la banque est dans banks_with_regulatory, False sinon.
        """
        if not self.bank_code or not self.bank_config:
            return False
        if str(self.quarter or "").strip().lower() == "t4":
            return False
        return self.bank_code in self.bank_config.get("banks_with_regulatory", [])
