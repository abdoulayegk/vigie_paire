"""Règles spécifiques aux rapports annuels T4, dont les bornes de section diffèrent des trimestriels.

Mixin consommé par ``SectionLocator``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from ..section_taxonomy import canonicalize_section
from .models import LocatedSection, normalize_text
from .patterns import SECTION_TITLE_ALIASES, T4_SECTION_TITLE_PROFILES

# Nom de logger conservé à l'identique après le découpage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigie.extraction.section_locator")


class AnnualT4Mixin:
    """Règles spécifiques aux rapports annuels T4, dont les bornes de section diffèrent des trimestriels."""

    def _is_t4_quarter(self) -> bool:
        """Indiquer si le rapport courant est un T4."""
        return str(self.quarter or "").strip().lower() == "t4"

    def _is_annual_t4_target_section(self, section: LocatedSection) -> bool:
        """Indiquer si une section relève de la stratégie annuelle T4."""
        return self._is_t4_quarter() and canonicalize_section(section.section_type) in {
            "capital_management",
            "risk_management",
        }

    def _is_annual_t4_risk_section(self, section: LocatedSection) -> bool:
        """Indiquer si une section est la section risques annuelle T4."""
        return (
            self._is_annual_t4_target_section(section)
            and canonicalize_section(section.section_type) == "risk_management"
        )

    def _annual_t4_profile(self, section_type: str) -> dict[str, list[str]]:
        """Retourner les repères sémantiques T4 de la banque courante."""
        section_type = {
            "capital_management": "gestion_capital",
            "risk_management": "gestion_risques",
        }.get(section_type, section_type)
        bank_profiles = T4_SECTION_TITLE_PROFILES.get(str(self.bank_code or "").lower(), {})
        return bank_profiles.get(section_type, {})

    def _annual_t4_start_titles(self, section_type: str) -> list[str]:
        """Assembler les titres racines possibles, sans numéros de pages."""
        candidates = list(self._annual_t4_profile(section_type).get("start", []))
        candidates.extend(self._get_config_section_names(section_type))
        candidates.extend(SECTION_TITLE_ALIASES.get(section_type, []))
        return list(dict.fromkeys(title for title in candidates if str(title or "").strip()))

    def _annual_t4_end_titles(self, section_type: str) -> list[str]:
        """Assembler les titres possibles du chapitre suivant T4."""
        candidates = list(self._annual_t4_profile(section_type).get("end", []))
        # Un profil T4 validé décrit la frontière annuelle exacte. Les anciens
        # ``followed_by`` restent utiles aux T1-T3 et aux banques inconnues,
        # mais ne doivent pas ramener une borne T4 avant ce chapitre.
        if candidates:
            return list(dict.fromkeys(title for title in candidates if str(title or "").strip()))
        section_map = {"gestion_capital": "capital_management", "gestion_risques": "risk_management"}
        config_name = section_map.get(section_type)
        if config_name and self.bank_code:
            bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
            candidates.extend(bank_data.get("sections", {}).get(config_name, {}).get("followed_by", []))
        return list(dict.fromkeys(title for title in candidates if str(title or "").strip()))

    def _annual_t4_title_match(self, line: str, titles: list[str]) -> str | None:
        """Comparer un titre PDF à des libellés, avec accents et doublons tolérés."""
        line_variants = self._title_match_variants(line)
        for title in titles:
            if line_variants & self._title_match_variants(title):
                return title
            title_normalized = normalize_text(title).strip()
            title_compact = re.sub(r"[^a-z0-9]+", "", title_normalized)
            for value in (line, self._unstutter_pdf_text(line)):
                line_normalized = normalize_text(value).strip()
                line_compact = re.sub(r"[^a-z0-9]+", "", line_normalized)
                if title_normalized and (
                    line_normalized.startswith(title_normalized)
                    or (title_compact and line_compact.startswith(title_compact))
                ):
                    return title
        return None

    def _annual_t4_successor_match(self, line: str, titles: list[str]) -> str | None:
        """Comparer un titre suivant, y compris lorsqu'il est coupé sur deux lignes."""
        for value in (line, self._unstutter_pdf_text(line)):
            line_normalized = normalize_text(value).strip()
            if len(line_normalized) < 10:
                continue
            line_compact = re.sub(r"[^a-z0-9]+", "", line_normalized)
            for title in titles:
                title_normalized = normalize_text(title).strip()
                title_compact = re.sub(r"[^a-z0-9]+", "", title_normalized)
                if title_normalized and (
                    line_normalized.startswith(title_normalized)
                    or title_normalized.startswith(line_normalized)
                    or (title_compact and line_compact.startswith(title_compact))
                ):
                    return title
        return None

    @staticmethod
    def _top_page_lines(page_text: str, max_lines: int = 35) -> list[str]:
        """Retourner les lignes utiles du haut d'une page physique.

        Les titres de chapitre sont normalement placés en tête de page. Limiter
        l'analyse à cette zone évite de confondre une mention dans une note ou un
        tableau avec une nouvelle section.
        """
        return [line.strip() for line in str(page_text or "").splitlines() if line.strip()][:max_lines]

    def _find_annual_t4_section_start(
        self,
        section_type: str,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> tuple[int, str] | None:
        """Trouver le meilleur titre racine T4 dans les pages physiques.

        L'ordre de lecture de ``pdfplumber`` peut placer un titre visuel après
        les 35 premières lignes. De plus, un alias générique peut apparaître
        avant le vrai chapitre (par exemple dans un sommaire narratif RBC).
        On classe donc tous les candidats en donnant la priorité au repère T4
        propre à la banque et en pénalisant les pages qui ressemblent à une TDM.
        """
        titles = self._annual_t4_start_titles(section_type)
        profile_titles = {
            normalize_text(title).strip()
            for title in self._annual_t4_profile(section_type).get("start", [])
            if str(title or "").strip()
        }
        candidates: list[tuple[float, int, str]] = []
        # Une ancre exacte du profil bancaire est plus fiable qu'un alias
        # générique. On la conserve séparément pour appliquer une règle
        # d'ordre physique à ces racines (notamment lorsque l'extraction d'un
        # diagramme répète « Gestion du risque » plus loin dans le rapport).
        profile_exact_candidates: list[tuple[int, int, str]] = []

        for page_num in range(6, total_pages + 1):
            page_text = text_by_page.get(page_num, "")
            if not page_text:
                continue
            lines = [line.strip() for line in str(page_text).splitlines() if line.strip()]
            # Une vraie page racine peut elle-même contenir un mini-sommaire
            # détaillé (notamment CIBC). Le signal TDM doit donc être borné.
            toc_penalty = min(self._score_toc_candidate_page(page_num, page_text) * 2.0, 50.0)
            noise_penalty = 50.0 if self._is_section_scan_noise_page(page_text) else 0.0
            for line_index, line in enumerate(lines, start=1):
                matched_title = self._annual_t4_title_match(line, titles)
                if not matched_title:
                    continue
                if not self._is_likely_section_title(line, page_text, matches_configured_pattern=True):
                    continue

                matched_normalized = normalize_text(matched_title).strip()
                exact_match = bool(self._title_match_variants(line) & self._title_match_variants(matched_title))
                if matched_normalized in profile_titles and exact_match:
                    profile_exact_candidates.append((page_num, line_index, matched_title))
                score = 0.0
                if matched_normalized in profile_titles:
                    score += 1000.0
                if exact_match:
                    score += 100.0
                if line_index <= 5:
                    score += 15.0
                elif line_index <= 35:
                    score += 5.0
                else:
                    # Une mention de titre profondément enfouie dans une page
                    # narrative/tableau ne doit pas prévaloir sur la même
                    # ancre présente en tête d'un vrai chapitre. On pénalise
                    # sans exclure : certains PDF placent un titre visuel
                    # valide plus bas dans leur ordre d'extraction.
                    score -= min(30.0, (line_index - 35) * 0.6)
                score -= toc_penalty
                score -= noise_penalty
                # Les sommaires annuels se trouvent normalement avant la page
                # physique 30. Entre deux vrais titres de même qualité, le
                # premier chapitre physique est le bon.
                if page_num < 30:
                    score -= 200.0
                score -= page_num / 2.0
                candidates.append((score, page_num, matched_title))

        if profile_exact_candidates:
            # Les titres de chapitre se trouvent normalement en tête de page.
            # S'il y en a un, il prévaut sur une mention narrative antérieure.
            top_level_roots = [candidate for candidate in profile_exact_candidates if candidate[1] <= 5]
            if top_level_roots:
                page_num, _, matched_title = min(top_level_roots, key=lambda item: item[0])
                return page_num, matched_title

            # Certains PDF placent néanmoins le vrai titre après un tableau
            # volumineux. Sans ancre de tête, le premier titre exact du profil
            # est la racine du chapitre; les répétitions ultérieures sont des
            # sous-thèmes ou du texte de diagramme.
            page_num, _, matched_title = min(profile_exact_candidates, key=lambda item: item[0])
            return page_num, matched_title

        if not candidates:
            return None
        _, page_num, matched_title = max(candidates, key=lambda item: (item[0], -item[1]))
        return page_num, matched_title

    def _rebase_annual_t4_section_starts(
        self,
        sections: list[LocatedSection],
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> list[LocatedSection]:
        """Recaler ou créer les sections capital et risques T4 sur titres physiques."""
        if not self._is_t4_quarter():
            return sections

        rebased = list(sections)
        for section_type in ("gestion_capital", "gestion_risques"):
            root = self._find_annual_t4_section_start(section_type, text_by_page, total_pages)
            existing_index = next(
                (
                    index
                    for index, section in enumerate(rebased)
                    if canonicalize_section(section.section_type) == canonicalize_section(section_type)
                ),
                None,
            )
            if root is None:
                logger.warning("T4 annuel: ancre physique introuvable pour %s", section_type)
                continue

            start_page, title_found = root
            if existing_index is None:
                rebased.append(
                    LocatedSection(
                        section_type=section_type,
                        title_found=title_found,
                        start_page=start_page,
                        confidence=0.9,
                        detection_method="annual_t4_physical_title",
                    )
                )
                logger.info("T4 annuel: %s créé depuis le titre physique p.%s", section_type, start_page)
                continue

            existing = rebased[existing_index]
            if existing.detection_method.startswith("manual_override"):
                continue
            rebased[existing_index] = replace(
                existing,
                title_found=title_found,
                start_page=start_page,
                end_page=None,
                detection_method="annual_t4_physical_title",
                end_detection_method="",
                detected_span=None,
                final_span=None,
                constraint_applied=False,
                constraint_reason="",
            )
        return sorted(rebased, key=lambda section: section.start_page)

    def _detect_annual_t4_section_end(
        self,
        section_type: str,
        start_page: int,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> tuple[int | None, str]:
        """Trouver la borne T4 avec le premier vrai titre du chapitre suivant."""
        section_type = {
            "capital_management": "gestion_capital",
            "risk_management": "gestion_risques",
        }.get(section_type, section_type)
        successor_titles = self._annual_t4_end_titles(section_type)
        following_patterns = self.following_patterns.get(section_type, [])
        has_profile_successor = bool(self._annual_t4_profile(section_type).get("end"))
        for page_num in range(start_page + 1, total_pages + 1):
            page_text = text_by_page.get(page_num, "")
            if not page_text:
                continue
            lines = [line.strip() for line in str(page_text).splitlines() if line.strip()]
            for line_index, line in enumerate(lines, start=1):
                variants = (line, self._unstutter_pdf_text(line))
                successor = next(
                    (self._annual_t4_successor_match(value, successor_titles) for value in variants),
                    None,
                )
                successor_is_exact = bool(
                    successor
                    and any(
                        self._title_match_variants(value) & self._title_match_variants(successor) for value in variants
                    )
                )
                if (
                    not any(
                        self._is_likely_section_title(value, page_text, matches_configured_pattern=True)
                        for value in variants
                    )
                    and not successor_is_exact
                ):
                    continue
                if section_type == "gestion_risques" and any(self._is_risk_subsection(value) for value in variants):
                    continue
                # Un titre exact peut être rejeté très bas dans l'ordre textuel
                # de pdfplumber. Les correspondances seulement préfixées et les
                # patterns génériques restent limitées au haut de page.
                if (
                    successor_is_exact
                    or (successor and line_index <= 35)
                    or (
                        not has_profile_successor
                        and line_index <= 35
                        and any(pattern.match(value) for pattern in following_patterns for value in variants)
                    )
                ):
                    logger.info(
                        "T4 annuel: fin %s validée par titre physique p.%s: %s",
                        section_type,
                        page_num,
                        line[:80],
                    )
                    return page_num - 1, "annual_t4_physical_successor"

        logger.warning(
            "T4 annuel: aucune section suivante validée après %s p.%s; application de la borne de sécurité documentée.",
            section_type,
            start_page,
        )
        return min(total_pages, start_page + 119), "annual_t4_safety_cap_no_successor"

    def _find_annual_t4_risk_start(
        self,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> tuple[int, str] | None:
        """Compatibilité: trouver la racine T4 de la section risques."""
        return self._find_annual_t4_section_start("gestion_risques", text_by_page, total_pages)

    def _rebase_annual_t4_risk_start(
        self,
        sections: list[LocatedSection],
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> list[LocatedSection]:
        """Compatibilité avec l'ancien point d'entrée spécialisé risques."""
        return self._rebase_annual_t4_section_starts(sections, text_by_page, total_pages)

    def _detect_annual_t4_risk_end(
        self,
        start_page: int,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> tuple[int | None, str]:
        """Compatibilité: trouver la borne T4 de la section risques."""
        return self._detect_annual_t4_section_end("gestion_risques", start_page, text_by_page, total_pages)
