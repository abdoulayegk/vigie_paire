"""Validation croisée des sections détectées et calcul de consensus entre stratégies.

Mixin consommé par ``SectionLocator``.
"""

from __future__ import annotations

import logging

from .models import LocatedSection, TocEntry

# Nom de logger conservé à l'identique après le découpage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigie.extraction.section_locator")


class ValidationMixin:
    """Validation croisée des sections détectées et calcul de consensus entre stratégies."""

    def _calculate_consensus(
        self,
        section: LocatedSection,
        toc_detections: list[TocEntry],
        scan_detections: list[LocatedSection],
    ) -> float:
        """Calculer le score de consensus entre les differentes methodes de detection.

        Compare les pages de debut/fin entre override, TDM et scan.
        Score base sur la proximite des pages detectees.

        Args:
            section: Section detectee (peut etre depuis override, TDM ou scan)
            toc_detections: Entrees TDM correspondant a cette section
            scan_detections: Sections detectees par scan correspondant a cette section

        Returns:
            Score de consensus entre 0.0 et 1.0
        """
        # Collecter toutes les pages de debut detectees
        start_pages = []
        end_pages = []

        # Page de debut de la section actuelle
        if section.start_page:
            start_pages.append((section.start_page, 1.0))  # Override a poids 1.0

        # Pages de debut depuis TDM
        for toc_entry in toc_detections:
            if toc_entry.page:
                start_pages.append((toc_entry.page, 0.8))  # TDM a poids 0.8

        # Pages de debut depuis scan
        for scan_section in scan_detections:
            if scan_section.start_page:
                start_pages.append((scan_section.start_page, 0.6))  # Scan a poids 0.6

        # Pages de fin
        if section.end_page:
            end_pages.append((section.end_page, 1.0))

        for scan_section in scan_detections:
            if scan_section.end_page:
                end_pages.append((scan_section.end_page, 0.6))

        # Calculer le consensus pour les pages de debut
        consensus_start = 0.0
        if start_pages:
            # Calculer la mediane ponderee
            sorted_starts = sorted(start_pages, key=lambda x: x[0])
            total_weight = sum(w for _, w in sorted_starts)

            if total_weight > 0:
                # Calculer la variance ponderee (plus la variance est faible, plus le consensus est eleve)
                weighted_mean = sum(page * weight for page, weight in sorted_starts) / total_weight
                variance = sum(weight * (page - weighted_mean) ** 2 for page, weight in sorted_starts) / total_weight

                # Score de consensus: 1.0 si toutes les pages sont identiques, diminue avec la variance
                # Normaliser: variance de 0 = consensus 1.0, variance de 10+ = consensus ~0.5
                consensus_start = max(0.0, 1.0 - min(variance / 10.0, 0.5))

        # Calculer le consensus pour les pages de fin
        consensus_end = 0.0
        if end_pages:
            sorted_ends = sorted(end_pages, key=lambda x: x[0])
            total_weight = sum(w for _, w in sorted_ends)

            if total_weight > 0:
                weighted_mean = sum(page * weight for page, weight in sorted_ends) / total_weight
                variance = sum(weight * (page - weighted_mean) ** 2 for page, weight in sorted_ends) / total_weight
                consensus_end = max(0.0, 1.0 - min(variance / 10.0, 0.5))

        # Score de consensus global (moyenne ponderee)
        if start_pages and end_pages:
            consensus = consensus_start * 0.6 + consensus_end * 0.4
        elif start_pages:
            consensus = consensus_start
        elif end_pages:
            consensus = consensus_end
        else:
            consensus = 0.0

        return consensus

    def _validate_with_cross_reference(
        self,
        sections: list[LocatedSection],
        toc_entries: list[TocEntry],
        scanned_sections: list[LocatedSection],
        text_by_page: dict[int, str],
    ) -> list[LocatedSection]:
        """Valider les sections en croisant les resultats de toutes les methodes.

        Pour chaque section:
        1. Verifier si TDM et scan donnent des resultats coherents
        2. Valider le contenu contextuel
        3. Ajuster la confiance selon le consensus
        4. Corriger les sections incoherentes

        Args:
            sections: Sections detectees (peut inclure override, TDM, scan)
            toc_entries: Toutes les entrees TDM
            scanned_sections: Toutes les sections detectees par scan
            text_by_page: Texte par page

        Returns:
            Sections validees et corrigees
        """
        validated = []

        for section in sections:
            # Collecter toutes les detections pour cette section
            toc_detections = [e for e in toc_entries if self._matches_section(e.title, section.section_type)]

            scan_detections = [s for s in scanned_sections if s.section_type == section.section_type]

            # Calculer le score de consensus
            consensus_score = self._calculate_consensus(section, toc_detections, scan_detections)

            # Valider le contenu contextuel (Amélioration 2)
            is_valid_content, content_score = self._validate_section_content(section, text_by_page)

            # Ajuster la confiance selon le consensus et la validation du contenu
            original_confidence = section.confidence

            if consensus_score > 0.7:
                # Consensus eleve: augmenter la confiance
                section.confidence = min(1.0, section.confidence + 0.2)
            elif consensus_score < 0.5:
                # Consensus faible: reduire la confiance
                section.confidence = max(0.0, section.confidence - 0.3)

                # Pour les risques T4, la borne provient déjà d'un titre
                # physique suivant. La médiane TDM/scan pourrait la ramener à
                # un faux sous-thème; elle sert donc uniquement au score, pas à
                # modifier les pages.
                if not self._is_annual_t4_target_section(section):
                    corrected = self._correct_section_bounds(section, toc_detections, scan_detections)
                    if corrected:
                        section = corrected
                        # Restaurer partiellement la confiance apres correction
                        section.confidence = min(original_confidence, 0.7)

            # Ajuster selon la validation du contenu
            if is_valid_content:
                # Contenu valide: legere augmentation
                section.confidence = min(1.0, section.confidence + 0.1 * content_score)
            else:
                # Contenu invalide: reduction significative
                section.confidence = max(0.0, section.confidence - 0.4)
                logger.warning(
                    f"Section {section.section_type} a un contenu invalide "
                    f"(score: {content_score:.2f}), confiance reduite a {section.confidence:.2f}"
                )

            validated.append(section)

            logger.debug(
                f"Validation croisee {section.section_type}: "
                f"consensus={consensus_score:.2f}, contenu={content_score:.2f}, "
                f"confiance={original_confidence:.2f}->{section.confidence:.2f}"
            )

        return validated

    def _validate_section_content(self, section: LocatedSection, text_by_page: dict[int, str]) -> tuple[bool, float]:
        """Valider que le contenu d'une section correspond au type de section attendu.

        Verifie:
        - Presence de mots-cles specifiques a la section
        - Absence de mots-cles d'autres sections
        - Coherence du contenu

        Args:
            section: Section a valider
            text_by_page: Texte par page

        Returns:
            Tuple (is_valid, validation_score) ou is_valid est True si validation_score > 0.4
        """
        if not section.start_page:
            return False, 0.0

        # Extraire le texte de la section
        section_text = self._extract_section_text(section, text_by_page)

        if not section_text or len(section_text.strip()) < 50:
            # Section trop courte ou vide
            return False, 0.0

        section_text_lower = section_text.lower()

        # Mots-cles attendus pour cette section
        expected_keywords = self.compiled_patterns.get(section.section_type, {}).get("keywords", [])

        if not expected_keywords:
            # Pas de mots-cles configures, validation basee uniquement sur l'absence de conflits
            keyword_ratio = 0.5
        else:
            # Compter les mots-cles trouves (insensible a la casse)
            found_keywords = sum(1 for kw in expected_keywords if kw.lower() in section_text_lower)

            # Ratio de mots-cles trouves
            # Les vocabulaires specialises enrichissent la couverture sans rendre
            # les sections historiques plus difficiles a valider.
            keyword_target = min(len(expected_keywords) * 0.3, 7.2)
            keyword_ratio = min(1.0, found_keywords / keyword_target)

        # Verifier l'absence de mots-cles d'autres sections
        other_section_type = "gestion_risques" if section.section_type == "gestion_capital" else "gestion_capital"
        other_keywords = self.compiled_patterns.get(other_section_type, {}).get("keywords", [])

        conflicting_keywords = 0
        if other_keywords:
            # Analyser seulement le debut de la section pour eviter les faux positifs
            section_start = section_text_lower[:2000]  # Premiers 2000 caracteres
            conflicting_keywords = sum(1 for kw in other_keywords if kw.lower() in section_start)

        # Score de validation
        # 70% basé sur la présence de mots-clés attendus
        # 30% pénalité pour les mots-clés conflictuels
        validation_score = keyword_ratio * 0.7 - min(conflicting_keywords / 10, 0.3) * 0.3

        # Normaliser entre 0 et 1
        validation_score = max(0.0, min(1.0, validation_score))

        # Seuil de validation abaisse (sections deja validees par TDM/scan)
        is_valid = validation_score > 0.25

        logger.debug(
            f"Validation contenu {section.section_type}: score={validation_score:.2f}, "
            f"keywords={found_keywords}/{len(expected_keywords) if expected_keywords else 0}, "
            f"conflits={conflicting_keywords}, valide={is_valid}"
        )

        return is_valid, validation_score

    def _extract_section_text(self, section: LocatedSection, text_by_page: dict[int, str]) -> str:
        """Extraire le texte complet d'une section.

        Args:
            section: Section dont on veut extraire le texte
            text_by_page: Texte par page

        Returns:
            Texte complet de la section
        """
        if not section.start_page:
            return ""

        section_text_parts = []

        # Determiner la page de fin (ou utiliser une limite par defaut)
        end_page = section.end_page
        if not end_page:
            # Si pas de page de fin, prendre les 20 pages suivantes
            end_page = min(
                section.start_page + 20,
                max(text_by_page.keys()) if text_by_page else section.start_page + 20,
            )

        # Extraire le texte de chaque page
        for page_num in range(section.start_page, end_page + 1):
            page_text = text_by_page.get(page_num, "")
            if page_text:
                section_text_parts.append(page_text)

        return "\n".join(section_text_parts)
