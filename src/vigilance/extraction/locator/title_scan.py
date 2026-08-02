"""Balayage textuel des titres de sections et scores de confiance associes.

Extrait de ``section_locator.py`` sans modification des corps de methodes.
Mixin consomme par ``SectionLocator``.
"""

from __future__ import annotations

import logging
import re

from .bank_config import _get_bank_section_names
from .models import LocatedSection, normalize_text
from .patterns import RISK_SUBSECTIONS, SECTION_TITLE_ALIASES

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")


class TitleScanMixin:
    """Balayage textuel des titres de sections et scores de confiance associes."""

    def _scan_section_titles(self, text_by_page: dict[int, str]) -> list[LocatedSection]:
        """Scanner le PDF pour trouver les titres de sections.

        Args:
            text_by_page: Texte par page

        Returns:
            Liste des sections localisees
        """
        sections = []
        found_types = set()

        # Premier passage: chercher les sections principales
        # On commence apres les premieres pages (TDM, intro) - typiquement page 5+
        start_page = 5

        # Passe stricte: chercher d'abord les vrais titres configures/connus,
        # puis retenir le meilleur candidat par section. Cette passe evite les
        # faux positifs dans les phrases de gouvernance ou les tableaux qui
        # contiennent "gestion du risque" / "fonds propres".
        strict_candidates: dict[str, list[tuple[LocatedSection, float]]] = {}
        for page_num in sorted(text_by_page.keys()):
            if page_num < start_page:
                continue

            page_text = text_by_page[page_num]
            lines = [line.strip() for line in page_text.split("\n") if line.strip()]
            page_is_noise = self._is_section_scan_noise_page(page_text)

            for line_index, line_stripped in enumerate(lines, start=1):
                if self._is_risk_subsection(line_stripped):
                    continue

                for section_type in self.compiled_patterns:
                    if section_type == "gestion_reglementation" and not self._bank_has_regulatory_section():
                        continue
                    if section_type in found_types:
                        continue
                    matched_title = self._strict_section_title_match(line_stripped, section_type)
                    if not matched_title:
                        continue
                    if page_is_noise or self._is_weak_section_scan_line(line_stripped, section_type):
                        continue
                    section = LocatedSection(
                        section_type=section_type,
                        title_found=matched_title,
                        start_page=page_num,
                        end_page=min(page_num + 10, max(text_by_page.keys())),
                        confidence=1.0,
                        detection_method="scan_exact",
                    )
                    configured_names = {normalize_text(name) for name in self._get_config_section_names(section_type)}
                    score = 100.0
                    if normalize_text(matched_title) in configured_names:
                        score += 25.0
                    if line_index <= 5:
                        score += 10.0
                    elif line_index <= 20:
                        score += 5.0
                    score -= page_num / 100.0
                    section.end_page = None
                    strict_candidates.setdefault(section_type, []).append((section, score))
                    logger.debug(
                        "Candidat titre exact: %s -> page %s score=%.2f",
                        matched_title,
                        page_num,
                        score,
                    )

        for section_type, candidates in strict_candidates.items():
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    item[1],
                    -item[0].start_page,
                ),
                reverse=True,
            )
            section, score = candidates[0]
            sections.append(section)
            found_types.add(section_type)
            logger.debug(
                "Section retenue par titre exact: %s -> page %s score=%.2f",
                section.title_found,
                section.start_page,
                score,
            )

        for page_num in sorted(text_by_page.keys()):
            if page_num < start_page:
                continue

            page_text = text_by_page[page_num]
            if self._is_section_scan_noise_page(page_text):
                continue

            lines = page_text.split("\n")

            for line in lines:
                line_stripped = line.strip()

                # Ignorer les sous-sections de risques (elles font partie de gestion_risques)
                if self._is_risk_subsection(line_stripped):
                    continue

                # Verifier d'abord si la ligne correspond a un pattern configure
                # Cela permet de bypasser le filtre de longueur strict pour les sections longues
                matches_pattern = False
                matching_section_type = None
                matching_config = None

                for section_type, config in self.compiled_patterns.items():
                    if section_type == "gestion_reglementation" and not self._bank_has_regulatory_section():
                        continue
                    # Eviter les doublons
                    if section_type in found_types:
                        continue

                    # Verifier les patterns d'exclusion
                    exclude_patterns = config.get("exclude_patterns", [])
                    should_exclude = False
                    for excl in exclude_patterns:
                        if re.search(excl, line_stripped, re.IGNORECASE):
                            should_exclude = True
                            break

                    if should_exclude:
                        continue

                    # Verifier si un pattern correspond
                    for pattern in config["regex"]:
                        if pattern.search(line_stripped):
                            matches_pattern = True
                            matching_section_type = section_type
                            matching_config = config
                            break

                    if matches_pattern:
                        break

                # Si un pattern correspond, on peut bypasser le filtre de longueur strict
                if matches_pattern:
                    if self._is_weak_section_scan_line(line_stripped, matching_section_type):
                        continue
                    # Un pattern correspond: verifier que c'est quand meme un titre valide
                    # mais avec limite de longueur etendue
                    if not self._is_likely_section_title(line_stripped, page_text, matches_configured_pattern=True):
                        continue

                    # Calculer la confiance
                    confidence = self._calculate_title_confidence(line_stripped, page_text, matching_config["keywords"])

                    if confidence > 0.5:
                        section = LocatedSection(
                            section_type=matching_section_type,
                            title_found=line_stripped,
                            start_page=page_num,
                            confidence=confidence,
                            detection_method="scan",
                        )
                        sections.append(section)
                        found_types.add(matching_section_type)
                        logger.debug(f"Section trouvee par scan: {line_stripped} -> page {page_num}")
                else:
                    # Aucun pattern ne correspond: appliquer le filtre normal
                    # Verifier si c'est un titre potentiel (avec filtre de longueur normal)
                    if not self._is_likely_section_title(line_stripped, page_text, matches_configured_pattern=False):
                        continue

                    # Meme si aucun pattern ne correspond initialement, verifier les patterns
                    # pour les sections standards (peut-etre que le titre est une variante)
                    for section_type, config in self.compiled_patterns.items():
                        # Eviter les doublons
                        if section_type in found_types:
                            continue

                        # Verifier les patterns d'exclusion
                        exclude_patterns = config.get("exclude_patterns", [])
                        should_exclude = False
                        for excl in exclude_patterns:
                            if re.search(excl, line_stripped, re.IGNORECASE):
                                should_exclude = True
                                break

                        if should_exclude:
                            continue
                        if self._is_weak_section_scan_line(line_stripped, section_type):
                            continue

                        # Verifier si un pattern correspond
                        for pattern in config["regex"]:
                            if pattern.search(line_stripped):
                                # Calculer la confiance
                                confidence = self._calculate_title_confidence(
                                    line_stripped, page_text, config["keywords"]
                                )

                                if confidence > 0.5:
                                    section = LocatedSection(
                                        section_type=section_type,
                                        title_found=line_stripped,
                                        start_page=page_num,
                                        confidence=confidence,
                                        detection_method="scan",
                                    )
                                    sections.append(section)
                                    found_types.add(section_type)
                                    logger.debug(f"Section trouvee par scan: {line_stripped} -> page {page_num}")
                                break

        # Si on n'a pas trouve "gestion_risques" mais qu'on trouve "Risque de credit",
        # utiliser cette page comme debut de la section risques
        if "gestion_risques" not in found_types:
            risk_subsection = self._find_first_risk_subsection(text_by_page)
            if risk_subsection:
                sections.append(risk_subsection)
                found_types.add("gestion_risques")
                logger.info(f"Section risques inferee depuis sous-section: {risk_subsection.title_found}")

        return sections

    def _is_likely_section_title(self, line: str, page_text: str, matches_configured_pattern: bool = False) -> bool:
        """Verifier si une ligne ressemble a un titre de section.

        Args:
            line: Ligne a verifier
            page_text: Texte complet de la page
            matches_configured_pattern: Si True, le titre correspond a un pattern configure
                                        et on permet une longueur jusqu'a 150 caracteres

        Returns:
            True si c'est probablement un titre
        """
        line_stripped = line.strip()

        # Limite de longueur: 80 caracteres par defaut, 150 si c'est un pattern configure
        max_length = 150 if matches_configured_pattern else 80

        # Trop court ou trop long
        if len(line_stripped) < 10 or len(line_stripped) > max_length:
            return False

        # Contient trop de chiffres (probablement une ligne de donnees)
        digit_ratio = sum(c.isdigit() for c in line_stripped) / len(line_stripped)
        if digit_ratio > 0.3:
            return False

        # Contient des caracteres de tableau
        if any(c in line_stripped for c in ["|", "$", "%", "€"]):
            return False

        # Format titre (majuscules ou Title Case)
        if line_stripped.isupper() or line_stripped.istitle():
            return True

        # Premiere lettre majuscule est souvent un titre
        if line_stripped[0].isupper():
            # Verifier que c'est pas une phrase normale (pas de point final)
            if not line_stripped.endswith("."):
                return True

        # Commence par un mot-cle de section
        keywords = [
            "gestion",
            "risque",
            "capital",
            "fonds",
            "situation",
            "facteurs",
            "examen",
        ]
        if any(line_stripped.lower().startswith(kw) for kw in keywords):
            return True

        return False

    def _strict_section_title_match(self, line: str, section_type: str) -> str | None:
        """Matcher uniquement un vrai titre de section configure, pas une phrase."""
        line_variants = self._title_match_variants(line)
        if not line_variants:
            return None

        aliases: list[str] = []
        aliases.extend(SECTION_TITLE_ALIASES.get(section_type, []))
        aliases.extend(self._get_config_section_names(section_type))
        for alias in aliases:
            alias = str(alias or "").strip()
            if not alias:
                continue
            if line_variants & self._title_match_variants(alias):
                return alias
        return None

    def _is_section_scan_noise_page(self, page_text: str) -> bool:
        """Identifier les pages qui ne doivent pas servir d'ancre de section."""
        page_lower = normalize_text(page_text)
        page_top = normalize_text("\n".join(str(page_text or "").splitlines()[:25]))

        toc_markers = [
            r"table\s+des\s+matieres",
            r"table\s+of\s+contents",
            r"guide\s+du\s+lecteur",
        ]
        if any(re.search(pattern, page_lower, re.IGNORECASE) for pattern in toc_markers):
            return True

        noise_markers = [
            "rapport de l auditeur independant",
            "etats financiers consolides",
            "notes afferentes aux etats financiers",
            "notes aux etats financiers",
            "bilans consolides",
            "etats consolides du resultat",
        ]
        return any(marker in page_top for marker in noise_markers)

    def _is_weak_section_scan_line(self, line: str, section_type: str) -> bool:
        """Ecarter les phrases qui contiennent les mots cibles sans etre la section."""
        line_lower = normalize_text(line)
        weak_patterns = {
            "gestion_capital": [
                r"actif\s+pond[eé]r[eé]\s+en\s+fonction\s+des?\s+risques?",
                r"rendement\s+des?\s+capitaux\s+propres",
                r"capitaux\s+propres\s+attribuables",
                r"variation\s+des?\s+capitaux\s+propres",
                r"[eé]tat\s+.*capitaux\s+propres",
            ],
            "gestion_risques": [
                r"chef\s+des?\s+risques",
                r"chef\s+de\s+la\s+gestion\s+des?\s+risques?",
                r"comit[ée]\s+de\s+gestion\s+des?\s+risques?",
                r"structure\s+de\s+gestion\s+des?\s+risques?",
                r"gestion\s+du\s+risque\s+d['e]\s*entreprise",
                r"gestion\s+du\s+risque\s+li[eé]",
            ],
        }
        return any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in weak_patterns.get(section_type, []))

    def _unstutter_pdf_text(self, text: str) -> str:
        """Corriger les mots dont chaque caractere est double par l'extraction PDF."""

        def _unstutter_token(token: str) -> str:
            if len(token) < 4 or len(token) % 2 != 0:
                return token
            if all(token[i] == token[i + 1] for i in range(0, len(token), 2)):
                return "".join(token[i] for i in range(0, len(token), 2))
            return token

        return " ".join(_unstutter_token(token) for token in str(text or "").split())

    def _title_match_variants(self, text: str) -> set[str]:
        """Retourner des variantes normalisees pour matcher un titre exact."""
        variants: set[str] = set()
        for value in {str(text or ""), self._unstutter_pdf_text(text)}:
            normalized = normalize_text(value).strip()
            if not normalized:
                continue
            variants.add(normalized)
            compact = re.sub(r"[^a-z0-9]+", "", normalized)
            if compact:
                variants.add(compact)
        return variants

    def _line_matches_section_title(self, line: str, section_names: list[str]) -> bool:
        """Verifier si une ligne correspond a un des titres de section attendus.

        Args:
            line: Ligne candidate
            section_names: Titres attendus (config)

        Returns:
            True si la ligne correspond a un titre de section.
        """
        if not line or not section_names:
            return False

        normalized_line = normalize_text(line.strip())
        if len(normalized_line) < 8:
            return False

        for section_name in section_names:
            normalized_name = normalize_text(section_name)
            if not normalized_name:
                continue
            if (
                normalized_name in normalized_line
                or normalized_line in normalized_name
                or self._text_similarity(normalized_line, normalized_name) >= 0.85
            ):
                return True
        return False

    def _find_section_start_in_window(
        self,
        estimated_page: int,
        text_by_page: dict[int, str],
        section_names: list[str],
        total_pages: int,
    ) -> int | None:
        """Recaler le debut reel d'une section autour d'une page estimee.

        Strategie:
        - Fenetre etroite d'abord (rapide, limite les faux positifs)
        - Fenetre plus large en fallback
        - Ignorer les toutes premieres pages pour eviter les matchs TDM

        Args:
            estimated_page: Page estimee du debut de la section
            text_by_page: Texte du PDF indexe par numero de page
            section_names: Noms de section attendus (depuis la config)
            total_pages: Nombre total de pages du document

        Returns:
            Numero de page du debut reel, ou None si non trouve.
        """
        if estimated_page <= 0 or not section_names or not text_by_page:
            return None

        # Fenetres de recherche progressives autour de l'estimation
        windows = [(-2, 4), (-6, 8)]
        min_allowed_page = 6

        for window_start, window_end in windows:
            start_page = max(min_allowed_page, estimated_page + window_start)
            end_page = min(total_pages, estimated_page + window_end)
            if start_page > end_page:
                continue

            for page_num in range(start_page, end_page + 1):
                page_text = text_by_page.get(page_num, "")
                if not page_text:
                    continue
                lines = page_text.split("\n")
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if not self._line_matches_section_title(line_stripped, section_names):
                        continue
                    if not self._is_likely_section_title(line_stripped, page_text, matches_configured_pattern=True):
                        continue
                    return page_num

        return None

    def _find_next_header_page(
        self,
        section_type: str,
        start_page: int,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> int | None:
        """Trouver la prochaine section/titre principal apres une section.

        Args:
            section_type: Type de section courante
            start_page: Debut de la section courante (physique)
            text_by_page: Texte du PDF par page
            total_pages: Nombre total de pages

        Returns:
            Numero de page du prochain grand titre, ou None.
        """
        following_patterns = self.following_patterns.get(section_type, [])
        if not following_patterns:
            return None

        search_start = max(start_page + 1, 6)
        search_end = min(total_pages, start_page + 60)

        for page_num in range(search_start, search_end + 1):
            page_text = text_by_page.get(page_num, "")
            if not page_text:
                continue

            for line in page_text.split("\n"):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if not self._is_likely_section_title(line_stripped, page_text):
                    continue
                if self._is_risk_subsection(line_stripped):
                    continue
                for pattern in following_patterns:
                    if pattern.search(line_stripped):
                        return page_num
        return None

    def _calculate_title_confidence(self, title: str, page_text: str, keywords: list[str]) -> float:
        """Calculer le score de confiance pour un titre de section.

        Args:
            title: Titre trouve
            page_text: Texte de la page
            keywords: Mots-cles attendus

        Returns:
            Score entre 0.0 et 1.0
        """
        score = 0.4  # Score de base plus bas

        # Utiliser la normalisation pour ignorer les accents
        title_normalized = normalize_text(title)

        # Bonus significatif si le titre correspond exactement aux noms de la banque
        if self.bank_code:
            bank_section_names = _get_bank_section_names(self.bank_code)
            for section_type, names in bank_section_names.items():
                for name in names:
                    # Comparer avec normalisation (ignore les accents)
                    if normalize_text(name) in title_normalized:
                        score += 0.3
                        break

        # Bonus si le titre est court (format titre typique)
        if len(title) < 40:
            score += 0.15
        elif len(title) < 60:
            score += 0.05

        # Bonus si majuscules ou title case
        if title.isupper():
            score += 0.1
        elif title.istitle():
            score += 0.05

        # Bonus pour les mots-cles dans la page
        page_lower = page_text.lower()
        keyword_count = sum(1 for kw in keywords if kw.lower() in page_lower)
        score += min(keyword_count * 0.03, 0.15)

        # Bonus si le titre est seul sur sa ligne (probable titre de section)
        for line in page_text.split("\n")[:30]:
            line_stripped = line.strip()
            if normalize_text(line_stripped) == title_normalized and len(line_stripped) < 60:
                score += 0.15
                break

        # Penalite si le titre contient des elements de TDM
        if re.search(r"\d{2,}.*\d{2,}", title):  # Plusieurs numeros = probablement ligne TDM
            score -= 0.2

        return max(0.0, min(score, 1.0))

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculer une similarite simple entre deux textes normalises.

        Utilise le ratio de caracteres communs et la longueur des mots communs.

        Args:
            text1: Premier texte (deja normalise)
            text2: Deuxieme texte (deja normalise)

        Returns:
            Score de similarite entre 0.0 et 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Si identique, similarite parfaite
        if text1 == text2:
            return 1.0

        # Si l'un contient l'autre, similarite elevee
        if text1 in text2 or text2 in text1:
            min_len = min(len(text1), len(text2))
            max_len = max(len(text1), len(text2))
            return min_len / max_len if max_len > 0 else 0.0

        # Calculer les mots communs
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        common_words = words1.intersection(words2)
        total_words = words1.union(words2)

        # Ratio de mots communs
        word_ratio = len(common_words) / len(total_words) if total_words else 0.0

        # Bonus si les mots importants (longs) sont communs
        important_words1 = {w for w in words1 if len(w) > 4}
        important_words2 = {w for w in words2 if len(w) > 4}
        common_important = important_words1.intersection(important_words2)

        if important_words1 or important_words2:
            important_ratio = len(common_important) / max(len(important_words1), len(important_words2))
            # Combiner les ratios (poids plus eleve pour les mots importants)
            return word_ratio * 0.4 + important_ratio * 0.6

        return word_ratio

    def _matches_section(self, title: str, section_type: str) -> bool:
        """Verifier si un titre correspond a un type de section.

        Args:
            title: Titre a verifier
            section_type: Type de section (gestion_capital ou gestion_risques)

        Returns:
            True si le titre correspond au type de section
        """
        title_normalized = normalize_text(title)
        patterns = []
        for section_key in self._section_alias_keys(section_type):
            patterns.extend(self.compiled_patterns.get(section_key, {}).get("regex", []))

        for pattern in patterns:
            if pattern.search(title_normalized):
                return True

        return False

    def _is_risk_subsection(self, title: str) -> bool:
        """Verifier si un titre est une sous-section de Gestion des risques.

        Les sous-sections comme "Risque de credit" font partie de "Gestion des risques"
        et ne doivent pas etre traitees comme des sections principales.

        Args:
            title: Titre a verifier

        Returns:
            True si c'est une sous-section de risques
        """
        # Utiliser la normalisation pour ignorer les accents
        title_normalized = normalize_text(title)

        # Si c'est "Gestion des risques" ou "Gestion du risque", ce n'est PAS une sous-section
        if re.search(r"gestion\s+(des\s+risques|du\s+risque)\b", title_normalized):
            return False

        # Verifier contre les sous-sections connues (avec normalisation)
        for subsection in RISK_SUBSECTIONS:
            if normalize_text(subsection) in title_normalized:
                return True

        # Patterns specifiques de sous-sections
        subsection_patterns = [
            r"^risque\s+de\s+cr[eé]dit",
            r"^risque\s+de\s+march[eé]",
            r"^risque\s+de\s+liquidit[eé]",
            r"^risque\s+op[eé]rationnel",
            r"^credit\s+risk",
            r"^market\s+risk",
        ]

        for pattern in subsection_patterns:
            if re.search(pattern, title_normalized):
                return True

        return False

    def _find_first_risk_subsection(self, text_by_page: dict[int, str]) -> LocatedSection | None:
        """Trouver la premiere sous-section de risques comme proxy pour la section principale.

        Args:
            text_by_page: Texte par page

        Returns:
            LocatedSection ou None
        """
        for page_num in sorted(text_by_page.keys()):
            if page_num < 10:  # Commencer apres l'intro
                continue

            page_text = text_by_page[page_num]
            if self._is_section_scan_noise_page(page_text):
                continue

            lines = page_text.split("\n")

            for line in lines:
                line_stripped = line.strip()
                line_normalized = normalize_text(line_stripped)

                if len(line_stripped) < 10 or len(line_stripped) > 80:
                    continue

                # Chercher les sous-sections de risques
                for subsection in RISK_SUBSECTIONS:
                    subsection_normalized = normalize_text(subsection)
                    if subsection_normalized in line_normalized:
                        # Verifier que c'est bien un titre (pas dans une phrase)
                        if len(line_stripped) < 50 and (
                            line_stripped.istitle()
                            or line_stripped.isupper()
                            or line_normalized.startswith(subsection_normalized)
                        ):
                            return LocatedSection(
                                section_type="gestion_risques",
                                title_found=f"[Infere depuis: {line_stripped}]",
                                start_page=page_num,
                                confidence=0.7,
                                detection_method="scan_subsection",
                            )

        return None
