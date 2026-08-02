"""Analyse de la table des matieres : parsing, validation et derivation de sections.

Extrait de ``section_locator.py`` sans modification des corps de methodes.
Mixin consomme par ``SectionLocator``.
"""

from __future__ import annotations

import logging
import re

from .models import LocatedSection, TocEntry, normalize_text

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")


class TocParserMixin:
    """Analyse de la table des matieres : parsing, validation et derivation de sections."""

    def _parse_full_toc(self, text_by_page: dict[int, str]) -> list[TocEntry]:
        """Parser la Table des matieres complete pour extraire TOUTES les sections.

        Cette methode extrait toutes les entrees de la TDM, pas seulement
        les sections cibles, ce qui permet de determiner les limites exactes.

        Args:
            text_by_page: Texte par page

        Returns:
            Liste de TocEntry triee par page
        """
        entries = []

        # Chercher la TDM. Les T4/rapports annuels peuvent placer la vraie TDM
        # plus loin qu'un sommaire preliminaire; les T1-T3 gardent la fenetre
        # historique des premieres pages.
        toc_page = None
        toc_text = ""

        if self._is_t4_quarter():
            candidate_scores: list[tuple[float, int, str]] = []
            for page_num in range(1, min(26, len(text_by_page) + 1)):
                page_text = text_by_page.get(page_num, "")
                score = self._score_toc_candidate_page(page_num, page_text)
                if score > 0:
                    candidate_scores.append((score, page_num, page_text))
            if candidate_scores:
                _, toc_page, toc_text = max(candidate_scores, key=lambda item: item[0])
        else:
            for page_num in range(1, min(7, len(text_by_page) + 1)):
                page_text = text_by_page.get(page_num, "")

                for pattern in self.toc_patterns:
                    if pattern.search(page_text):
                        toc_page = page_num
                        toc_text = page_text
                        break

                if toc_page:
                    break

        if toc_page:
            for next_page in range(toc_page + 1, min(toc_page + 4, len(text_by_page) + 1)):
                toc_text += "\n" + text_by_page.get(next_page, "")

        if not toc_page:
            logger.debug("Table des matieres non trouvee")
            return entries

        logger.info(f"Table des matieres trouvee page {toc_page}")
        logger.debug(f"TDM: Extraction du texte depuis pages {toc_page}-{min(toc_page + 3, len(text_by_page))}")

        # Determiner le nombre max de pages pour validation
        max_pages = max(text_by_page.keys()) if text_by_page else 200
        logger.debug(f"TDM: Nombre max de pages pour validation: {max_pages}")

        # Parser chaque ligne de la TDM
        lines = toc_text.split("\n")

        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 5:
                continue

            # Ignorer les lignes qui sont clairement pas des entrees TDM
            if line_clean.lower().startswith(("note:", "voir", "page", "www.")):
                continue

            # FILTRE: Ignorer les lignes trop longues (probablement du texte, pas une entree TDM)
            # RBC utilise des titres longs (~93 chars): "Examen de la conjoncture economique..."
            if len(line_clean) > 150:
                continue

            # FILTRE: Ignorer les lignes avec trop de chiffres (ratios, donnees financieres)
            digit_count = sum(1 for c in line_clean if c.isdigit())
            if digit_count > 10:
                continue

            # FILTRE: Ignorer les lignes qui ressemblent a des phrases (trop de mots)
            # RBC utilise des titres longs (~15 mots) et des lignes multi-colonnes (~20 mots)
            # Le format multi-colonnes combine plusieurs entrees sur une seule ligne
            word_count = len(line_clean.split())
            if word_count > 22:
                continue

            parsed = self._parse_toc_line(line_clean, max_pages=max_pages)
            if parsed:
                # _parse_toc_line peut retourner une liste (format multi-colonnes)
                if isinstance(parsed, list):
                    # Filtrer les entrees avec titres trop longs ou suspects
                    filtered = [e for e in parsed if self._is_valid_toc_entry(e)]
                    entries.extend(filtered)
                else:
                    if self._is_valid_toc_entry(parsed):
                        entries.append(parsed)

        # Trier par page et deduplicer
        entries = self._deduplicate_toc_entries(entries)
        entries.sort(key=lambda e: e.page)

        logger.debug(f"TDM parsee: {len(entries)} entrees")
        # Log des entrees principales (level 0) pour debug
        level0_entries = [e for e in entries if e.level == 0]
        if level0_entries:
            logger.debug(f"TDM: {len(level0_entries)} sections principales (level 0) trouvees:")
            for e in level0_entries[:10]:  # Limiter a 10 pour eviter trop de logs
                logger.debug(f"  - Page {e.page}: '{e.title}' (level={e.level})")

        return entries

    def _parse_toc_line(self, line: str, max_pages: int = 200) -> TocEntry | list[TocEntry] | None:
        """Parser une ligne de la Table des matieres.

        Formats supportes:
        - "Titre ... 25"
        - "Titre 25"
        - "25 Titre"
        - "Titre 25-30"
        - Format multi-colonnes BNC: "Acquisition 4 Gestion du capital 25"

        Args:
            line: Ligne de la TDM
            max_pages: Nombre max de pages (pour validation)

        Returns:
            TocEntry, liste de TocEntry (multi-colonnes), ou None
        """
        # D'abord, essayer le format multi-colonnes (BNC)
        # Pattern: "Titre nombre Titre nombre" repete
        # Ex: "Acquisition 4 Gestion du capital 25"
        multi_entries = self._try_parse_multi_column_toc(line, max_pages)
        if multi_entries and len(multi_entries) >= 2:
            return multi_entries

        # Pattern 1: Numero a la fin "Titre ... 25" ou "Titre 25-30"
        page_match = re.search(r"(\d{1,3})(?:\s*[-–]\s*\d{1,3})?\s*$", line)

        if page_match:
            page_num = int(page_match.group(1))
            title_part = line[: page_match.start()].strip()
        else:
            # Pattern 2: Numero au debut "25 Titre"
            page_match_start = re.match(r"^(\d{1,3})\s+", line)
            if page_match_start:
                page_num = int(page_match_start.group(1))
                title_part = line[page_match_start.end() :].strip()
            else:
                return None

        # Nettoyer le titre (enlever les points de suite, tirets, etc.)
        title_part = re.sub(r"\.{2,}", " ", title_part)
        title_part = re.sub(r"[-–]{2,}", " ", title_part)
        title_part = re.sub(r"\s{2,}", " ", title_part).strip()

        if not title_part or len(title_part) < 3:
            return None

        # Ignorer les pages < 3 (probablement TDM elle-meme)
        if page_num < 3:
            return None

        # VALIDATION: Ignorer les pages > max_pages (clairement erreur de parsing)
        if page_num > max_pages:
            return None

        # Determiner le niveau (0 = section principale, 1+ = sous-section)
        level = 0
        if line.startswith("  ") or line.startswith("\t"):
            level = 1
        # Les titres en minuscules sont souvent des sous-sections
        if not title_part[0].isupper():
            level = max(level, 1)

        # AMELIORATION: Verifier si le titre correspond a une section cible ou suivante
        # Si oui, forcer level = 0 (section principale)
        title_normalized = normalize_text(title_part)

        # Verifier si c'est une section cible (capital_management ou risk_management)
        if self.bank_code and self.bank_config:
            bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
            sections = bank_data.get("sections", {})

            # Verifier les sections cibles
            for config_name in ["capital_management", "risk_management"]:
                section_config = sections.get(config_name, {})
                section_names = section_config.get("names", [])

                for section_name in section_names:
                    section_name_normalized = normalize_text(section_name)
                    # Match partiel ou exact
                    if (
                        section_name_normalized in title_normalized
                        or title_normalized in section_name_normalized
                        or self._text_similarity(title_normalized, section_name_normalized) > 0.7
                    ):
                        level = 0  # Forcer comme section principale
                        logger.debug(
                            f"TDM: '{title_part}' identifie comme section principale (correspond a {config_name})"
                        )
                        break

                if level == 0:
                    break

            # Si pas encore identifie comme principale, verifier les sections suivantes
            if level != 0:
                for config_name in ["capital_management", "risk_management"]:
                    section_config = sections.get(config_name, {})
                    followed_by = section_config.get("followed_by", [])

                    for followed_name in followed_by:
                        followed_normalized = normalize_text(followed_name)
                        # Match partiel ou exact
                        if (
                            followed_normalized in title_normalized
                            or title_normalized in followed_normalized
                            or self._text_similarity(title_normalized, followed_normalized) > 0.7
                        ):
                            level = 0  # Forcer comme section principale
                            logger.debug(
                                f"TDM: '{title_part}' identifie comme section principale (section suivante: {followed_name})"
                            )
                            break

                    if level == 0:
                        break

        return TocEntry(title=title_part, page=page_num, level=level, raw_line=line)

    def _try_parse_multi_column_toc(self, line: str, max_pages: int) -> list[TocEntry]:
        """Essayer de parser une ligne de TDM en format multi-colonnes.

        Formats supportes:
        - BNC: "Acquisition 4 Gestion du capital 25" (Titre numero Titre numero)
        - BMO: "16 Benefice net 43 Gestion des risques" (numero Titre numero Titre)

        Args:
            line: Ligne de la TDM
            max_pages: Nombre max de pages pour validation

        Returns:
            Liste de TocEntry ou liste vide
        """
        entries = []

        # FORMAT 1 (BNC): "Titre numero Titre numero"
        # Pattern pour capturer: "Texte nombre" repete
        pattern_bnc = re.compile(
            r"([A-ZÀ-ÜÉÈÊËÎÏÔÛÙÇŒÆa-zà-üéèêëîïôûùçœæ][^0-9]{2,}?)\s+(\d{1,3})(?=\s+[A-ZÀ-ÜÉÈÊËÎÏÔÛÙÇŒÆa-z]|\s*$)",
            re.UNICODE,
        )

        # FORMAT 2 (BMO/RBC): "numero Titre numero Titre"
        # Pattern pour capturer: "nombre Texte" repete (peut etre n'importe ou dans la ligne)
        pattern_bmo = re.compile(
            r"(\d{1,3})\s+([A-ZÀ-ÜÉÈÊËÎÏÔÛÙÇŒÆa-zà-ü][^0-9]+?)(?=\s+\d{1,3}\s+[A-ZÀ-Ü]|\s*$)",
            re.UNICODE,
        )

        # Essayer le format BMO (numero Titre) - applicable meme si ligne ne commence pas par chiffre
        # Car le format multi-colonnes peut avoir du texte avant le numero
        matches_bmo = pattern_bmo.findall(line)
        for page_str, title in matches_bmo:
            title = title.strip().rstrip(".")
            title = re.sub(r"\s{2,}", " ", title)

            try:
                page_num = int(page_str)
            except ValueError:
                continue

            if page_num < 3 or page_num > max_pages:
                continue
            if len(title) < 3:
                continue

            entries.append(TocEntry(title=title, page=page_num, level=0, raw_line=line))

        # Si format BMO n'a rien trouve, essayer format BNC (Titre numero)
        if not entries:
            matches = pattern_bnc.findall(line)
            for title, page_str in matches:
                title = title.strip().rstrip(".")
                title = re.sub(r"\s{2,}", " ", title)

                try:
                    page_num = int(page_str)
                except ValueError:
                    continue

                if page_num < 3 or page_num > max_pages:
                    continue
                if len(title) < 3:
                    continue

                entries.append(TocEntry(title=title, page=page_num, level=0, raw_line=line))

        return entries

        return entries

    def _is_valid_toc_entry(self, entry: TocEntry) -> bool:
        """Valider qu'une entree TDM est probablement un vrai titre de section.

        Filtre le bruit: ratios, phrases completes, donnees financieres.

        Args:
            entry: Entree TDM a valider

        Returns:
            True si l'entree semble valide
        """
        title = entry.title

        # Titre trop court ou trop long
        # RBC utilise des titres longs comme "Examen de la conjoncture economique..." (~93 chars)
        if len(title) < 5 or len(title) > 120:
            return False

        # Trop de chiffres dans le titre (probablement des donnees financieres)
        digit_count = sum(1 for c in title if c.isdigit())
        if digit_count > 2:
            return False

        # Contient des symboles financiers (%, $, M$)
        if any(x in title for x in ["%", "$", "M$", "G$"]):
            return False

        # Commence par un chiffre ou minuscule (probablement un ratio ou une sous-phrase)
        if title[0].isdigit() or title[0].islower():
            return False

        # Trop de mots (probablement une phrase, pas un titre)
        # RBC utilise des titres longs (~15 mots): "Examen de la conjoncture economique, des marches et du contexte reglementaire et perspectives"
        word_count = len(title.split())
        if word_count > 15:
            return False

        # Contient des patterns de donnees financieres ou de bruit
        noise_patterns = [
            r"\\d+[,.]\\d+",  # Nombres decimaux
            r"\\(\\d+\\)",  # Nombres entre parentheses
            r"trimestre",  # Mentions de trimestre dans le titre
            r"terminé le",
            r"en million",
            r"en pourcentage",
            r"autres techniques",  # Sous-sections, pas sections principales
            r"essais",
            r": ",  # Titres avec deux-points sont souvent des sous-sections ou bruit
        ]
        title_lower = title.lower()
        for pattern in noise_patterns:
            if re.search(pattern, title_lower):
                return False

        return True

    def _deduplicate_toc_entries(self, entries: list[TocEntry]) -> list[TocEntry]:
        """Deduplicer les entrees TDM par titre similaire et page proche.

        Args:
            entries: Liste d'entrees TDM potentiellement dupliquees

        Returns:
            Liste d'entrees TDM sans doublons.
        """
        if not entries:
            return entries

        unique = []
        seen_titles = {}

        for entry in entries:
            # Normaliser le titre pour la comparaison
            title_key = entry.title.lower()[:30]

            if title_key in seen_titles:
                # Garder celui avec la page la plus basse
                if entry.page < seen_titles[title_key].page:
                    unique.remove(seen_titles[title_key])
                    unique.append(entry)
                    seen_titles[title_key] = entry
            else:
                unique.append(entry)
                seen_titles[title_key] = entry

        return unique

    def _detect_sections_from_full_toc(self, toc_entries: list[TocEntry]) -> list[LocatedSection]:
        """Detecter les sections cibles depuis la TDM complete.

        Args:
            toc_entries: Entrees TDM

        Returns:
            Liste des sections localisees avec pages de fin
        """
        sections = []
        entries_by_page = sorted(toc_entries, key=lambda e: e.page)

        for i, entry in enumerate(entries_by_page):
            for section_type, config_patterns in self.compiled_patterns.items():
                if section_type == "gestion_reglementation" and not self._bank_has_regulatory_section():
                    continue
                # Verifier les patterns d'exclusion
                should_exclude = False
                for excl in config_patterns.get("exclude_patterns", []):
                    if re.search(excl, entry.title, re.IGNORECASE):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

                # Verifier si le titre correspond a un pattern
                for pattern in config_patterns["regex"]:
                    if pattern.search(entry.title):
                        # Verifier que ce n'est pas une sous-section de risques
                        if section_type == "gestion_capital" and self._is_risk_subsection(entry.title):
                            continue

                        # Trouver la page de fin depuis la TDM
                        end_page = None
                        end_method = "toc_next_section"

                        # Etape 1: Chercher la prochaine section principale (level 0)
                        # qui est au moins min_length pages apres
                        min_length = self._get_section_length_constraints(section_type)["min_length"]
                        min_end_page = entry.page + min_length
                        logger.debug(
                            f"TDM: Recherche fin section '{entry.title}' (page {entry.page}, "
                            f"type: {section_type}, min_end_page: {min_end_page})"
                        )

                        for next_entry in entries_by_page[i + 1 :]:
                            if next_entry.level == 0 and next_entry.page >= min_end_page:
                                if self._matches_section(next_entry.title, section_type):
                                    logger.debug(
                                        f"TDM: Section suivante ignoree (meme famille {section_type}): "
                                        f"'{next_entry.title}' page {next_entry.page}"
                                    )
                                    continue
                                if section_type == "gestion_risques" and self._is_risk_subsection(next_entry.title):
                                    logger.debug(
                                        f"TDM: Section suivante ignoree (sous-section risques): "
                                        f"'{next_entry.title}' page {next_entry.page}"
                                    )
                                    continue
                                end_page = next_entry.page - 1
                                end_method = "toc_next_section"
                                logger.debug(
                                    f"TDM: Fin trouvee par section principale (level 0): "
                                    f"'{next_entry.title}' page {next_entry.page} -> end_page={end_page}"
                                )
                                break

                        # Etape 2: Si pas trouve, chercher par pattern "followed_by"
                        if end_page is None:
                            logger.debug(
                                f"TDM: Aucune section principale trouvee, "
                                f"recherche par pattern 'followed_by' pour {section_type}"
                            )
                            next_section = self._find_next_section_by_pattern(section_type, entry.page, toc_entries)
                            if next_section:
                                end_page = next_section[0] - 1
                                end_method = "toc_followed_by_pattern"
                                logger.debug(
                                    f"TDM: Fin trouvee par pattern 'followed_by': "
                                    f"'{next_section[1]}' page {next_section[0]} -> end_page={end_page}"
                                )
                            else:
                                logger.debug(
                                    "TDM: Aucune section suivante trouvee par pattern, "
                                    "end_page sera determine par _determine_end_pages"
                                )

                        # Si toujours pas trouve, laisser end_page = None (sera gere par _determine_end_pages)

                        section = LocatedSection(
                            section_type=section_type,
                            title_found=entry.title,
                            start_page=entry.page,
                            end_page=end_page,
                            confidence=0.95,
                            detection_method="toc",
                            end_detection_method=end_method if end_page else "",
                        )

                        # Eviter les doublons
                        if not any(s.section_type == section_type for s in sections):
                            sections.append(section)
                            logger.info(
                                f"Section TDM detectee: {section_type} '{entry.title}' -> "
                                f"pages {entry.page}-{end_page if end_page else '?'} "
                                f"(methode fin: {end_method if end_page else 'a determiner'})"
                            )
                        break

        return sections

    def _detect_from_toc(self, text_by_page: dict[int, str]) -> list[LocatedSection]:
        """Detecter les sections depuis la Table des matieres (methode legacy).

        Args:
            text_by_page: Texte par page

        Returns:
            Liste des sections localisees
        """
        toc_entries = self._parse_full_toc(text_by_page)
        return self._detect_sections_from_full_toc(toc_entries)

    def _score_toc_candidate_page(self, page_num: int, page_text: str) -> float:
        """Scorer une page candidate TDM pour les rapports T4."""
        if not page_text:
            return 0.0

        normalized = normalize_text(page_text)
        score = 0.0
        strong_markers = [r"table\s+des\s+matieres", r"table\s+of\s+contents", r"\bcontents\b"]
        soft_markers = [r"\bsommaire\b", r"rapport\s+de\s+gestion", r"guide\s+du\s+lecteur"]

        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in strong_markers):
            score += 50.0
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in soft_markers):
            score += 20.0
        if 10 <= page_num <= 25:
            score += 10.0
        if 15 <= page_num <= 20:
            score += 20.0

        toc_like_lines = 0
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if len(line) < 5 or len(line) > 160:
                continue
            if re.search(r"\d{1,3}\s*$", line) or re.match(r"^\d{1,3}\s+", line):
                toc_like_lines += 1
        score += min(toc_like_lines, 12) * 3.0

        for section_type in ("gestion_capital", "gestion_risques"):
            for name in self._get_config_section_names(section_type):
                name_norm = normalize_text(name)
                if name_norm and name_norm in normalized:
                    score += 20.0
                    break

        return score

    def _assess_toc_quality(
        self,
        toc_entries: list[TocEntry],
        toc_sections: list[LocatedSection],
        total_pages: int,
    ) -> float:
        """Evaluer la fiabilite de la Table des matieres.

        Args:
            toc_entries: Entrees TDM extraites
            toc_sections: Sections detectees depuis la TDM
            total_pages: Nombre total de pages du document

        Returns:
            Score de fiabilite entre 0.0 et 1.0.
        """
        if not toc_entries:
            return 0.0

        score = 0.0

        entry_score = min(len(toc_entries) / 25.0, 1.0)
        score += 0.4 * entry_score

        section_types = {s.section_type for s in toc_sections}
        expected = {"gestion_risques", "gestion_capital"}
        coverage = len(section_types.intersection(expected)) / len(expected)
        score += 0.4 * coverage

        if toc_sections:
            pages = [s.start_page for s in toc_sections if s.start_page]
            if pages:
                min_page = min(pages)
                max_page = max(pages)
                range_ratio = (max_page - min_page + 1) / max(total_pages, 1)
                score += 0.2 * min(range_ratio * 2.0, 1.0)

        return min(max(score, 0.0), 1.0)

    def _find_end_from_toc(
        self, section_type: str, start_page: int, toc_entries: list[TocEntry]
    ) -> tuple[int | None, str]:
        """Trouver la fin d'une section depuis la TDM.

        Args:
            section_type: Type de section
            start_page: Page de debut
            toc_entries: Entrees TDM

        Returns:
            Tuple (end_page, method) ou (None, "")
        """
        # Obtenir la longueur minimale de la section courante
        min_length = self._get_section_length_constraints(section_type)["min_length"]

        # Trouver les entrees TDM apres notre section
        # On cherche des entrees qui sont au moins min_length pages apres le debut
        min_end_page = start_page + min_length
        entries_after = [e for e in toc_entries if e.page >= min_end_page]

        logger.debug(
            f"_find_end_from_toc: Recherche fin pour {section_type} (start_page={start_page}, "
            f"min_end_page={min_end_page}, {len(entries_after)} entrees candidates)"
        )

        if not entries_after:
            logger.debug(f"_find_end_from_toc: Aucune entree TDM apres page {min_end_page}")
            return None, ""

        # Etape 1: Chercher une section principale (level 0) qui suit
        level0_candidates = [e for e in entries_after if e.level == 0]
        logger.debug(f"_find_end_from_toc: {len(level0_candidates)} sections principales (level 0) candidates")

        for entry in sorted(entries_after, key=lambda e: e.page):
            if entry.level == 0:
                if self._matches_section(entry.title, section_type):
                    logger.debug(
                        f"_find_end_from_toc: Entree ignoree "
                        f"(meme famille {section_type}): "
                        f"'{entry.title}' page {entry.page}"
                    )
                    continue
                end_page = entry.page - 1
                logger.debug(
                    f"_find_end_from_toc: Fin trouvee par section principale: "
                    f"'{entry.title}' page {entry.page} -> end_page={end_page}"
                )
                return end_page, "toc_next_section"

        # Etape 2: Si pas trouve, chercher par pattern "followed_by"
        logger.debug("_find_end_from_toc: Aucune section principale trouvee, recherche par pattern 'followed_by'")
        next_section = self._find_next_section_by_pattern(section_type, start_page, toc_entries)
        if next_section:
            end_page = next_section[0] - 1
            logger.debug(
                f"_find_end_from_toc: Fin trouvee par pattern 'followed_by': "
                f"'{next_section[1]}' page {next_section[0]} -> end_page={end_page}"
            )
            return end_page, "toc_followed_by_pattern"

        logger.debug(f"_find_end_from_toc: Aucune fin trouvee pour {section_type}")
        return None, ""

    def _next_toc_boundary_title_candidates(
        self,
        section: LocatedSection,
        toc_entries: list[TocEntry],
    ) -> list[str]:
        """Retourne les titres TDM de la section suivante sur la page frontière."""
        if section.end_page is None or not toc_entries:
            return []
        boundary_page = int(section.end_page) + 1
        offset = self._get_page_number_offset() if self._uses_document_page_numbers(section.detection_method) else 0
        candidates: list[str] = []
        for entry in sorted(toc_entries, key=lambda e: e.page):
            physical_page = int(entry.page) + offset
            if physical_page != boundary_page or entry.level != 0:
                continue
            if self._matches_section(entry.title, section.section_type):
                continue
            if section.section_type == "gestion_risques" and self._is_risk_subsection(entry.title):
                continue
            title = str(entry.title or "").strip()
            if title:
                candidates.append(title)
        return candidates
