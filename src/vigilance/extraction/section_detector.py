"""
Detecteur de sections pour identifier les sections cles dans les rapports bancaires.

Focus sur les sections principales:
- Gestion du capital / Fonds propres
- Gestion des risques
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectedSection:
    """Represente une section detectee dans le document."""

    name: str
    phase: int
    start_page: int
    end_page: int
    title_found: str
    confidence: float
    content_preview: str = ""


class SectionDetector:
    """
    Detecte et classifie les sections dans les rapports bancaires.

    Focus sur les sections cibles:
    - Gestion du capital / Fonds propres
    - Gestion des risques
    """

    # Patterns de sections pour les rapports bancaires francais
    # Note: "Gestion du capital" peut etre nomme "Gestion des fonds propres" ou "Situation des fonds propres"
    # Note: "Risque de crédit" est une sous-section de "Gestion des risques"
    SECTION_PATTERNS = {
        # Gestion du capital et des fonds propres
        # Variantes: "Gestion du capital", "Gestion des fonds propres", "Situation des fonds propres"
        "gestion_capital": {
            "phase": 1,
            "patterns": [
                r"gestion\s+du\s+capital",
                r"gestion\s+des\s+fonds\s+propres",
                r"situation\s+des\s+fonds\s+propres",
                r"capital\s+r[eé]glementaire",
                r"fonds\s+propres\s+r[eé]glementaires",
                r"capital\s+management",
                r"capitaux\s+propres",
            ],
            "keywords": [
                "CET1",
                "Tier 1",
                "levier",
                "LCR",
                "NSFR",
                "Bale",
                "BSIF",
                "NFP",
                "leverage ratio",
                "TLAC",
                "ratio",
            ],
        },
        # Gestion des risques
        # Note: Les sous-sections (Risque de credit, Risque de marche, etc.) font partie de cette section
        "gestion_risques": {
            "phase": 1,
            "patterns": [
                # Titres de section principale
                r"gestion\s+des\s+risques",
                r"gestion\s+du\s+risque",
                r"facteurs?\s+de\s+risque\s+et\s+gestion",
                r"facteurs?\s+de\s+risque",
                r"risk\s+management",
                r"exposition\s+aux?\s+risques?",
                # Sous-sections (font partie de Gestion des risques)
                r"risque\s+de\s+cr[eé]dit",
                r"risque\s+de\s+march[eé]",
                r"risque\s+de\s+liquidit[eé]",
                r"risque\s+op[eé]rationnel",
                r"credit\s+risk",
                r"market\s+risk",
            ],
            "keywords": [
                "provision",
                "creances",
                "defaut",
                "exposition",
                "VaR",
                "stress",
                "PCL",
                "allowance",
                "ECL",
                "credit",
            ],
        },
    }

    # Patterns additionnels pour les risques emergents (IA, ESG, Cyber)
    EMERGING_RISK_PATTERNS = {
        "risque_ia": [
            r"intelligence\s+artificielle",
            r"\bIA\b",
            r"apprentissage\s+(automatique|machine)",
            r"modeles?\s+algorithmiques?",
        ],
        "risque_esg": [
            r"ESG",
            r"environnement(al)?",
            r"climat(ique)?",
            r"developpement\s+durable",
            r"carbone",
            r"emissions?",
        ],
        "risque_cyber": [
            r"cyber(securite)?",
            r"securite\s+informatique",
            r"menaces?\s+informatiques?",
            r"piratage",
            r"donnees\s+personnelles",
        ],
    }

    def __init__(self, bank_config: dict | None = None):
        """
        Initialiser le detecteur de sections.

        Args:
            bank_config: Configuration specifique a la banque (optionnel)
        """
        self.bank_config = bank_config or {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compiler les patterns regex pour l'efficacite."""
        self.compiled_patterns = {}

        for section_name, config in self.SECTION_PATTERNS.items():
            self.compiled_patterns[section_name] = {
                "phase": config["phase"],
                "regex": [re.compile(p, re.IGNORECASE) for p in config["patterns"]],
                "keywords": config["keywords"],
            }

        self.compiled_emerging = {}
        for risk_type, patterns in self.EMERGING_RISK_PATTERNS.items():
            self.compiled_emerging[risk_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def detect_sections(
        self, text_content: str, page_breaks: list[int] | None = None
    ) -> list[DetectedSection]:
        """
        Detecter les sections cibles dans le texte du document.

        Args:
            text_content: Contenu textuel complet du document
            page_breaks: Liste optionnelle des positions de caracteres pour les sauts de page

        Returns:
            Liste des sections detectees (gestion_capital et gestion_risques uniquement)
        """
        detected = []
        lines = text_content.split("\n")

        current_page = 1
        char_count = 0
        page_breaks = page_breaks or []

        for line_num, line in enumerate(lines):
            # Mettre a jour le suivi des pages
            char_count += len(line) + 1
            while page_breaks and char_count > page_breaks[0]:
                page_breaks.pop(0)
                current_page += 1

            # Verifier chaque pattern de section
            for section_name, config in self.compiled_patterns.items():
                for pattern in config["regex"]:
                    match = pattern.search(line)
                    if match:
                        # Calculer la confiance basee sur la presence de mots-cles
                        confidence = self._calculate_confidence(
                            line,
                            lines[max(0, line_num - 5) : min(len(lines), line_num + 10)],
                            config["keywords"],
                        )

                        if confidence > 0.3:  # Seuil de confiance minimum
                            section = DetectedSection(
                                name=section_name,
                                phase=config["phase"],
                                start_page=current_page,
                                end_page=current_page,  # Sera mis a jour
                                title_found=line.strip()[:200],
                                confidence=confidence,
                                content_preview="\n".join(lines[line_num : line_num + 5])[:500],
                            )
                            detected.append(section)
                            break

        # Fusionner et dedupliquer les sections
        detected = self._merge_sections(detected)

        return detected

    def _calculate_confidence(
        self, title_line: str, context_lines: list[str], keywords: list[str]
    ) -> float:
        """Calculer le score de confiance pour la detection de section."""
        score = 0.5  # Score de base pour la correspondance de pattern

        context_text = " ".join(context_lines).lower()

        # Bonus pour la presence de mots-cles
        keyword_matches = sum(1 for kw in keywords if kw.lower() in context_text)
        score += min(keyword_matches * 0.1, 0.3)

        # Bonus pour le formatage de type titre (lignes plus courtes, majuscules)
        if len(title_line) < 100:
            score += 0.1
        if title_line.isupper() or title_line.istitle():
            score += 0.1

        return min(score, 1.0)

    def _merge_sections(self, sections: list[DetectedSection]) -> list[DetectedSection]:
        """Fusionner les sections adjacentes du meme type."""
        if not sections:
            return []

        # Trier par page et nom
        sections.sort(key=lambda s: (s.name, s.start_page))

        merged = []
        current = None

        for section in sections:
            if current is None:
                current = section
            elif current.name == section.name and section.start_page - current.end_page <= 2:
                # Fusionner les sections adjacentes
                current.end_page = section.start_page
                current.confidence = max(current.confidence, section.confidence)
            else:
                merged.append(current)
                current = section

        if current:
            merged.append(current)

        return merged

    def detect_emerging_risks(self, text_content: str) -> dict[str, list[str]]:
        """
        Detecter les mentions de risques emergents (IA, ESG, Cyber).

        Args:
            text_content: Texte du document

        Returns:
            Dictionnaire associant le type de risque a la liste des extraits correspondants
        """
        findings = {}

        for risk_type, patterns in self.compiled_emerging.items():
            matches = []
            for pattern in patterns:
                for match in pattern.finditer(text_content):
                    # Extraire le contexte autour de la correspondance
                    start = max(0, match.start() - 100)
                    end = min(len(text_content), match.end() + 200)
                    context = text_content[start:end].strip()

                    # Nettoyer le contexte
                    context = re.sub(r"\s+", " ", context)
                    if context not in matches:
                        matches.append(context)

            if matches:
                findings[risk_type] = matches[:10]  # Limiter a 10 par type

        return findings

    def classify_section(self, section_title: str, section_content: str) -> str:
        """
        Classifier une section.

        Args:
            section_title: Titre de la section
            section_content: Contenu de la section

        Returns:
            Type de section: "gestion_capital", "gestion_risques", ou "autre"
        """
        combined_text = f"{section_title} {section_content[:2000]}"

        section_scores = {"gestion_capital": 0, "gestion_risques": 0}

        for section_name, config in self.compiled_patterns.items():
            for pattern in config["regex"]:
                if pattern.search(combined_text):
                    section_scores[section_name] += 1

            for keyword in config["keywords"]:
                if keyword.lower() in combined_text.lower():
                    section_scores[section_name] += 0.5

        max_score = max(section_scores.values())
        if max_score > 0:
            return max(section_scores, key=section_scores.get)
        return "autre"


def detect_document_sections(text_content: str) -> list[DetectedSection]:
    """
    Fonction utilitaire pour detecter les sections dans le texte du document.

    Args:
        text_content: Texte complet du document

    Returns:
        Liste des sections detectees
    """
    detector = SectionDetector()
    return detector.detect_sections(text_content)
