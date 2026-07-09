"""Localisateur de sections pour identifier les pages des sections cibles dans les rapports bancaires.

Ce module detecte automatiquement les pages des sections:
- Gestion des risques
- Gestion du capital / fonds propres

Strategies de detection (par ordre de priorite):
1. Override manuel (configuration bank_profiles.json)
2. Detection via la Table des matieres (TDM) complete
3. Detection des sections suivantes (pour determiner la fin)
4. Scan des titres de sections dans le PDF
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

from .section_taxonomy import canonicalize_section

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normaliser le texte en supprimant les accents et en mettant en minuscules.

    Permet de matcher "réglementation" avec "reglementation", etc.

    Args:
        text: Texte a normaliser

    Returns:
        Texte sans accents, en minuscules
    """
    if not text:
        return ""
    # NFD decompose les caracteres accentues (e + accent)
    # encode/decode supprime les caracteres non-ASCII (les accents)
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")
    return ascii_text.lower()


@dataclass
class VisualTextElement:
    """Element de texte avec ses caracteristiques visuelles."""

    text: str
    page: int
    x0: float  # Position horizontale gauche
    y0: float  # Position verticale haute
    x1: float  # Position horizontale droite
    y1: float  # Position verticale basse
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    is_uppercase: bool = False
    line_number: int = 0  # Position relative sur la page
    page_width: float = 0.0
    page_height: float = 0.0

    @property
    def height(self) -> float:
        """Hauteur de l'element en points."""
        return abs(self.y1 - self.y0)

    @property
    def width(self) -> float:
        """Largeur de l'element en points."""
        return abs(self.x1 - self.x0)

    @property
    def is_likely_header(self) -> bool:
        """Determiner si l'element a les caracteristiques d'un titre."""
        # Criteres: grande taille, gras, ou majuscules
        return self.font_size >= 12.0 or self.is_bold or (self.is_uppercase and len(self.text) > 10)

    @property
    def bbox_norm(self) -> list[float] | None:
        """Retourner la bbox normalisee [x0, y0, x1, y1] si la taille de page est connue."""
        if self.page_width <= 0 or self.page_height <= 0:
            return None
        return [
            max(0.0, min(1.0, self.x0 / self.page_width)),
            max(0.0, min(1.0, self.y0 / self.page_height)),
            max(0.0, min(1.0, self.x1 / self.page_width)),
            max(0.0, min(1.0, self.y1 / self.page_height)),
        ]


@dataclass
class TocEntry:
    """Entree de la Table des matieres."""

    title: str
    page: int
    level: int = 0  # 0 = section principale, 1+ = sous-section
    raw_line: str = ""

    def __repr__(self):
        """Representation textuelle courte de l'entree TDM."""
        return f"TocEntry('{self.title[:30]}...', page={self.page}, level={self.level})"


@dataclass
class LocatedSection:
    """Represente une section localisee dans le document."""

    section_type: str  # "gestion_capital" ou "gestion_risques"
    title_found: str
    start_page: int
    end_page: int | None = None
    confidence: float = 0.0
    detection_method: str = ""  # "toc", "scan", "manual_override", "following_section"
    end_detection_method: str = ""  # Comment la fin a ete determinee
    detected_span: int | None = None
    final_span: int | None = None
    constraint_applied: bool = False
    constraint_reason: str = ""
    anchor_page: int | None = None
    anchor_text: str | None = None
    anchor_bbox_norm: list[float] | None = None
    anchor_found: bool = False
    end_anchor_page: int | None = None
    end_anchor_text: str | None = None
    end_anchor_bbox_norm: list[float] | None = None


SHARED_PAGE_TOP_THRESHOLD = 0.12


@dataclass
class SectionMapping:
    """Mapping complet des sections pour un document."""

    bank_code: str
    bank_name: str
    quarter: str
    year: int
    file_path: str
    sections: list[LocatedSection] = field(default_factory=list)
    total_pages: int = 0
    toc_entries: list[TocEntry] = field(default_factory=list)  # TDM complete
    toc_score: float = 0.0
    toc_reliable: bool = False
    toc_used: bool = False
    override_applied: bool = False

    def to_dict(self) -> dict:
        """Convertir le mapping de sections en dictionnaire serialisable.

        Returns:
            Dictionnaire contenant toutes les informations du mapping.
        """
        sections_dict = {}
        for section in self.sections:
            sections_dict[section.section_type] = {
                "pages": f"{section.start_page}-{section.end_page}" if section.end_page else str(section.start_page),
                "start_page": section.start_page,
                "end_page": section.end_page,
                "title_found": section.title_found,
                "confidence": section.confidence,
                "detection_method": section.detection_method,
                "end_detection_method": section.end_detection_method,
                "detected_span": section.detected_span,
                "final_span": section.final_span,
                "constraint_applied": section.constraint_applied,
                "constraint_reason": section.constraint_reason,
                "anchor_page": section.anchor_page,
                "anchor_text": section.anchor_text,
                "anchor_bbox_norm": section.anchor_bbox_norm,
                "anchor_found": section.anchor_found,
                "end_anchor_page": section.end_anchor_page,
                "end_anchor_text": section.end_anchor_text,
                "end_anchor_bbox_norm": section.end_anchor_bbox_norm,
            }

        return {
            "bank_code": self.bank_code,
            "bank_name": self.bank_name,
            "quarter": self.quarter,
            "year": self.year,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "sections": sections_dict,
            "toc_entry_count": len(self.toc_entries),
            "toc_score": self.toc_score,
            "toc_reliable": self.toc_reliable,
            "toc_used": self.toc_used,
            "override_applied": self.override_applied,
        }


# Patterns de detection par type de section
# Note: L'ordre des patterns est important - les plus specifiques en premier
SECTION_PATTERNS = {
    "gestion_capital": {
        "patterns": [
            # Variantes exactes (prioritaires)
            r"gestion\s+du\s+capital",
            r"gestion\s+des\s+fonds\s+propres",
            r"situation\s+des\s+fonds\s+propres",
            # RBC: Examen de la conjoncture economique (avec/sans accents)
            r"examen\s+de\s+la\s+conjoncture\s+[eé]conomique",
            # Variantes avec contexte reglementaire
            r"fonds\s+propres\s+r[eé]glementaires",
            r"capital\s+r[eé]glementaire",
            # Variantes anglaises
            r"capital\s+management",
            r"regulatory\s+capital",
            # Variantes partielles (moins prioritaires)
            r"capitaux\s+propres",
        ],
        # Mots-cles pour valider le contenu (variantes avec/sans accents)
        "keywords": [
            "cet1",
            "tier 1",
            "tier 2",
            "fonds propres",
            "capital",
            "capitaux",
            "ratio",
            "levier",
            "leverage",
            "bâle",
            "bale",
            "bsif",
            "tlac",
            "lcr",
            "nsfr",
            "liquidit",
            "dividende",
            "rachat",
            "actions",
        ],
        # Termes qui indiquent que ce n'est PAS la bonne section
        "exclude_patterns": [
            r"risque\s+de",  # Eviter confusion avec sections risques
            r"rendement\s+des?\s+capitaux\s+propres",
        ],
    },
    "gestion_risques": {
        "patterns": [
            # Variantes principales (titre de section)
            r"gestion\s+des\s+risques",
            r"gestion\s+du\s+risque(?!\s+de\s+cr[eé]dit)",  # Pas suivi de "de credit"
            r"risk\s+management",
            r"facteurs?\s+de\s+risque\s+et\s+gestion",
            # Variantes avec contexte
            r"facteurs?\s+de\s+risque",
            r"exposition\s+aux?\s+risques?",
            # Sections autonomes pouvant remplacer le titre global des risques
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?donn[eé]es(?:,\s*technologie\s+et\s+cybers[eé]curit[eé])?",
            r"risques?\s+technologiques?",
            r"technolog(?:ie|ique),?\s+cybers[eé]curit[eé]\s+et\s+donn[eé]es",
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?tiers",
            r"gestion\s+des?\s+fournisseurs",
            r"services?\s+infonuagiques?",
            r"r[eé]silience\s+op[eé]rationnelle",
            r"protection\s+des?\s+donn[eé]es\s+et\s+vie\s+priv[eé]e",
        ],
        # Mots-cles pour valider le contenu (variantes avec/sans accents)
        "keywords": [
            "risque",
            "risk",
            "crédit",
            "credit",
            "marché",
            "marche",
            "market",
            "liquidité",
            "liquidite",
            "liquidity",
            "opérationnel",
            "operationnel",
            "operational",
            "var",
            "exposition",
            "exposure",
            "provision",
            "perte",
            "loss",
            "portefeuille",
            "portfolio",
            "stress",
            "scénario",
            "scenario",
            "données",
            "donnees",
            "data",
            "technologie",
            "technology",
            "cybersécurité",
            "cybersecurite",
            "cloud",
            "infonuagique",
            "tiers",
            "fournisseur",
            "impartition",
            "résilience",
            "resilience",
            "vie privée",
            "vie privee",
            "qualité des données",
            "qualite des donnees",
            "intégrité des données",
            "integrite des donnees",
            "confidentialité",
            "confidentialite",
            "protection des données",
            "protection des donnees",
            "localisation des données",
            "localisation des donnees",
            "souveraineté",
            "souverainete",
            "conservation des données",
            "conservation des donnees",
            "traçabilité",
            "tracabilite",
            "lignage",
            "cycle de vie des données",
            "cycle de vie des donnees",
            "fuite de données",
            "fuite de donnees",
            "tiers critique",
            "fournisseur critique",
            "sous-traitant",
            "concentration des fournisseurs",
            "verrouillage fournisseur",
            "stratégie de sortie",
            "strategie de sortie",
            "continuité des services",
            "continuite des services",
            "exigence contractuelle",
        ],
        # Sous-sections qui font partie de "Gestion des risques"
        "subsections": [
            r"risque\s+de\s+cr[eé]dit",
            r"risque\s+de\s+march[eé]",
            r"risque\s+de\s+liquidit[eé]",
            r"risque\s+op[eé]rationnel",
            r"credit\s+risk",
            r"market\s+risk",
            r"liquidity\s+risk",
            r"operational\s+risk",
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?donn[eé]es",
            r"risques?\s+technologiques?",
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?tiers",
            r"risques?\s+li[eé]s?\s+[àa]\s+l['’]impartition",
            r"services?\s+infonuagiques?",
            r"r[eé]silience\s+op[eé]rationnelle",
            r"protection\s+des?\s+donn[eé]es",
            r"vie\s+priv[eé]e",
            r"data\s+risk",
            r"technology\s+risk",
            r"third[-\s]party\s+risk",
            r"cloud\s+risk",
            r"operational\s+resilience",
        ],
        "exclude_patterns": [
            r"chef\s+des?\s+risques",
            r"chef\s+de\s+la\s+gestion\s+des?\s+risques?",
            r"comit[ée]\s+de\s+gestion\s+des?\s+risques?",
            r"structure\s+de\s+gestion\s+des?\s+risques?",
            r"gestion\s+du\s+risque\s+d['e]\s*entreprise",
            r"gestion\s+du\s+risque\s+li[eé]",
        ],
    },
    "gestion_reglementation": {
        "patterns": [
            # RBC: Examen de la conjoncture economique
            r"examen\s+de\s+la\s+conjoncture\s+[eé]conomique",
            r"contexte\s+r[eé]glementaire\s+et\s+perspectives",
            # BNS/BMO: Faits nouveaux en matiere de reglementation
            r"faits?\s+nouveaux?\s+en\s+mati[eè]re\s+de\s+r[eé]glementation",
            r"autres?\s+faits?\s+nouveaux?\s+en\s+mati[eè]re\s+de\s+r[eé]glementation",
            # Variantes generiques
            r"mise\s+[àa]\s+jour\s+r[eé]glementaire",
            r"[eé]volution\s+r[eé]glementaire",
        ],
        "keywords": [
            "reglementation",
            "réglementation",
            "bsif",
            "bale",
            "bâle",
            "normes",
            "conjoncture",
            "perspectives",
            "contexte",
            "economique",
            "économique",
        ],
        "exclude_patterns": [],
    },
}

# Patterns des sections qui suivent typiquement nos sections cibles
# Utilises pour determiner la FIN d'une section
FOLLOWING_SECTION_PATTERNS = {
    "gestion_capital": [
        r"gestion\s+des?\s+risques?",
        r"gestion\s+du\s+risque",
        r"risque\s+de\s+cr[eé]dit",
        r"facteurs?\s+de\s+risque",
        r"r[eé]sultats?\s+consolid[eé]s?",
        r"analyse\s+des?\s+r[eé]sultats?",
    ],
    "gestion_risques": [
        r"normes\s+et\s+m[eé]thodes\s+comptables",
        r"m[eé]thodes\s+et\s+estimations\s+comptables",
        r"m[eé]thodes\s+comptables\s+significatives",
        r"[eé]tats?\s+financiers?",
        r"informations?\s+compl[eé]mentaires?",
        r"renseignements?\s+compl[eé]mentaires?",
        r"donn[eé]es?\s+compl[eé]mentaires?",
        r"annexes?",
        r"notes?\s+aux?\s+[eé]tats",
        r"glossaire",
        r"d[eé]finitions?",
    ],
    "gestion_reglementation": [
        r"gestion\s+des?\s+fonds?\s+propres?",
        r"gestion\s+du\s+capital",
        r"gestion\s+des?\s+risques?",
        r"gestion\s+du\s+risque",
        r"[eé]tats?\s+financiers?",
    ],
}

SECTION_TITLE_ALIASES: dict[str, list[str]] = {
    "gestion_capital": [
        "Gestion du capital",
        "Gestion des fonds propres",
        "Situation des fonds propres",
    ],
    "gestion_risques": [
        "Gestion des risques",
        "Gestion du risque",
        "Risk management",
    ],
    "gestion_reglementation": [
        "Réglementation",
        "Reglementation",
    ],
}

# NOTE: Les noms de sections par banque sont maintenant charges dynamiquement
# depuis bank_profiles.json via la fonction _get_bank_section_names()
# Cela garantit une source de verite unique pour la configuration.


def _get_bank_section_names(bank_code: str) -> dict:
    """Charger les noms de sections depuis bank_profiles.json (source unique de verite).

    Args:
        bank_code: Code de la banque (bnc, rbc, td, bmo, bns, cibc)

    Returns:
        Dict avec gestion_capital, gestion_risques et gestion_reglementation contenant les listes de noms
    """
    config = _load_bank_config()
    # Support both shapes:
    # - {"banks": {...}} (legacy raw file)
    # - {...} where keys are bank codes (load_bank_profiles helper)
    banks_cfg = config.get("banks", {}) if isinstance(config.get("banks"), dict) else config

    if bank_code in banks_cfg and isinstance(banks_cfg.get(bank_code), dict):
        sections = banks_cfg[bank_code].get("sections", {})
        return {
            "gestion_capital": sections.get("capital_management", {}).get("names", []),
            "gestion_risques": sections.get("risk_management", {}).get("names", []),
            "gestion_reglementation": sections.get("regulatory_updates", {}).get("names", []),
        }

    # Fallback par defaut si la banque n'est pas configuree
    return {
        "gestion_capital": ["Gestion du capital", "Gestion des fonds propres"],
        "gestion_risques": ["Gestion des risques", "Gestion du risque"],
        "gestion_reglementation": [],
    }


# Sous-sections de "Gestion des risques" qui ne doivent pas etre confondues
# avec la section principale. Ces sous-sections font PARTIE de la section risques.
RISK_SUBSECTIONS = [
    "Risque de credit",
    "Risque de marche",
    "Risque de liquidite",
    "Risque operationnel",
    "Risque de taux d'interet",
    "Risque de change",
    "Divulgation d'information sur les risques",
    "Divulgation d'informations sur les risques",
    "Divulgation d’information sur les risques",
    "Divulgation d’informations sur les risques",
    "Divulgation dinformation sur les risques",
    "Divulgation dinformations sur les risques",
    "Cotes de credit",
    "Cotes de crédit",
    "Credit Risk",
    "Market Risk",
    "Liquidity Risk",
    "Operational Risk",
    "Risque lié aux données",
    "Risque lié aux donnees",
    "Risque technologique",
    "Risque lié aux tiers",
    "Risque lie aux tiers",
    "Risque lié à l'impartition",
    "Services infonuagiques",
    "Résilience opérationnelle",
    "Resilience operationnelle",
    "Protection des données",
    "Protection des donnees",
    "Vie privée",
    "Vie privee",
    "Data Risk",
    "Technology Risk",
    "Third-Party Risk",
    "Cloud Risk",
    "Operational Resilience",
]

# Patterns pour detecter la Table des matieres
TOC_PATTERNS = [
    r"table\s+des\s+mati[eè]res",
    r"sommaire",
    r"table\s+of\s+contents",
    r"contents",
    # BNC utilise "Rapport de gestion" comme en-tete de la page TDM
    r"rapport\s+de\s+gestion",
    # Patterns additionnels pour detecter les pages avec TDM
    r"aper[çc]u\s+du\s+rapport",
    r"guide\s+du\s+lecteur",
]


def _load_bank_config() -> dict:
    """Charger la configuration des banques (YAML ou JSON)."""
    try:
        from vigilance.config.loader import load_config

        return load_config("configs/bank_profiles.yaml")
    except Exception as e:
        if "beyond top-level package" in str(e):
            logger.debug("Configuration parent package indisponible, fallback local actif")
        else:
            logger.warning(f"Impossible de charger la configuration bancaire: {e}")

    project_root = Path(__file__).resolve().parents[3]
    yaml_path = project_root / "configs" / "bank_profiles.yaml"
    json_path = project_root / "bank_profiles.json"

    if yaml_path.exists():
        try:
            import yaml

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Impossible de charger {yaml_path}: {e}")

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Impossible de charger {json_path}: {e}")

    return {}


# Build BANK_SECTION_NAMES for backward compatibility
# This is used by tests and other modules that expect a dict
BANK_SECTION_NAMES = {bank: _get_bank_section_names(bank) for bank in ["bnc", "rbc", "td", "bmo", "bns", "cibc"]}


class SectionLocator:
    """Localisateur de sections dans les rapports bancaires.

    Utilise une approche hybride a 3 niveaux:
    1. Override manuel (configuration bank_profiles.json)
    2. Detection via la Table des matieres (TDM) complete
    3. Detection des sections suivantes + scan des titres
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

    def _compile_patterns(self):
        """Compiler les patterns regex de detection de sections."""
        self.compiled_patterns = {}

        for section_type, config in SECTION_PATTERNS.items():
            # Ajouter les patterns specifiques a la banque si disponibles
            patterns = list(config["patterns"])

            # Charger dynamiquement les noms depuis bank_profiles.json
            if self.bank_code:
                bank_section_names = _get_bank_section_names(self.bank_code)
                bank_names = bank_section_names.get(section_type, [])
                for name in bank_names:
                    # Creer un pattern normalise (sans accents) pour matcher le texte PDF
                    # Le texte PDF peut avoir des accents differents ou manquants
                    normalized_name = normalize_text(name)
                    # Pattern flexible: autoriser des espaces/tirets entre les mots
                    escaped = re.escape(normalized_name).replace(r"\ ", r"\s+")
                    patterns.insert(0, escaped)  # Priorite aux noms specifiques

            self.compiled_patterns[section_type] = {
                "regex": [re.compile(p, re.IGNORECASE) for p in patterns],
                "keywords": config["keywords"],
                "exclude_patterns": config.get("exclude_patterns", []),
                "subsections": config.get("subsections", []),
            }

        # Compiler les patterns TDM
        self.toc_patterns = [re.compile(p, re.IGNORECASE) for p in TOC_PATTERNS]

    def _load_following_patterns(self):
        """Charger les patterns des sections suivantes depuis la configuration et les valeurs par defaut."""
        self.following_patterns = {}

        # D'abord les patterns par defaut
        for section_type, patterns in FOLLOWING_SECTION_PATTERNS.items():
            self.following_patterns[section_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Ajouter les patterns specifiques de la banque depuis la config
        if self.bank_code and self.bank_config:
            bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
            sections = bank_data.get("sections", {})

            # Mapper les noms de section config -> noms internes
            section_mapping = {
                "capital_management": "gestion_capital",
                "risk_management": "gestion_risques",
            }

            for config_name, internal_name in section_mapping.items():
                section_config = sections.get(config_name, {})
                followed_by = section_config.get("followed_by", [])

                if followed_by:
                    # Ajouter ces patterns en priorite
                    for name in followed_by:
                        escaped = re.escape(name)
                        pattern = re.compile(
                            rf"^\s*{escaped}(?:$|\b|\s|[:;,.–—-])",
                            re.IGNORECASE,
                        )
                        if internal_name in self.following_patterns:
                            self.following_patterns[internal_name].insert(0, pattern)
                        else:
                            self.following_patterns[internal_name] = [pattern]

    def _get_section_length_constraints(self, section_type: str) -> dict[str, int]:
        """Recuperer les contraintes de longueur pour un type de section.

        Priorite:
        1. Default code (gestion_reglementation = section courte 1-3 pages)
        2. section_boundary_detection.section_length_overrides
        3. banks.<bank>.sections.<section>.length_constraints
        """
        boundary_config = self.bank_config.get("section_boundary_detection", {})
        constraints = {
            "min_length": int(boundary_config.get("min_section_length", 3)),
            "max_length": int(boundary_config.get("max_section_length", 50)),
            "default_length": int(boundary_config.get("default_section_length", 20)),
        }

        # Default metier pour la section reglementaire (regulatory_updates)
        if section_type == "gestion_reglementation":
            constraints.update({"min_length": 1, "max_length": 3, "default_length": 3})

        # Overrides globaux optionnels
        overrides = boundary_config.get("section_length_overrides", {})
        override = overrides.get(section_type, {}) if isinstance(overrides, dict) else {}
        if override:
            constraints["min_length"] = int(
                override.get("min_length", override.get("min_pages", constraints["min_length"]))
            )
            constraints["max_length"] = int(
                override.get("max_length", override.get("max_pages", constraints["max_length"]))
            )
            constraints["default_length"] = int(
                override.get(
                    "default_length",
                    override.get("default_span", constraints["default_length"]),
                )
            )

        # Overrides par banque optionnels
        section_name_map = {
            "capital_management": "capital_management",
            "risk_management": "risk_management",
            "regulatory_updates": "regulatory_updates",
            "gestion_capital": "capital_management",
            "gestion_risques": "risk_management",
            "gestion_reglementation": "regulatory_updates",
        }
        if self.bank_code:
            bank_sections = self.bank_config.get("banks", {}).get(self.bank_code, {}).get("sections", {})
            section_name = section_name_map.get(section_type)
            section_cfg = bank_sections.get(section_name, {}) if section_name else {}
            bank_override = section_cfg.get("length_constraints", {})
            if isinstance(bank_override, dict) and bank_override:
                constraints["min_length"] = int(
                    bank_override.get(
                        "min_length",
                        bank_override.get("min_pages", constraints["min_length"]),
                    )
                )
                constraints["max_length"] = int(
                    bank_override.get(
                        "max_length",
                        bank_override.get("max_pages", constraints["max_length"]),
                    )
                )
                constraints["default_length"] = int(
                    bank_override.get(
                        "default_length",
                        bank_override.get("default_span", constraints["default_length"]),
                    )
                )

        # Normalisation defensive
        constraints["min_length"] = max(1, constraints["min_length"])
        constraints["max_length"] = max(constraints["min_length"], constraints["max_length"])
        constraints["default_length"] = min(
            max(constraints["default_length"], constraints["min_length"]),
            constraints["max_length"],
        )
        return constraints

    def _apply_section_length_constraints(
        self, section: LocatedSection, total_pages: int, source: str = ""
    ) -> LocatedSection:
        """Appliquer les contraintes min/max/default de longueur a une section.

        Args:
            section: Section a contraindre
            total_pages: Nombre total de pages du document
            source: Etiquette indiquant l'origine de l'appel (pour le log)

        Returns:
            La section modifiee en place avec les contraintes appliquees.
        """
        constraints = self._get_section_length_constraints(section.section_type)
        min_length = constraints["min_length"]
        max_length = constraints["max_length"]
        default_length = constraints["default_length"]

        if not section.start_page:
            return section

        reason_parts: list[str] = []
        applied = section.constraint_applied

        # Fin absente -> fallback deterministic
        if section.end_page is None:
            section.end_page = min(total_pages, section.start_page + default_length - 1)
            reason_parts.append(f"end_missing->default_{default_length}")
            applied = True

        if section.end_page is not None:
            detected_span = max(1, section.end_page - section.start_page + 1)
            section.detected_span = detected_span

            # Respecter le minimum
            if detected_span < min_length:
                section.end_page = min(total_pages, section.start_page + min_length - 1)
                reason_parts.append(f"min_enforced_{detected_span}->{min_length}")
                applied = True

            # Respecter le maximum
            current_span = max(1, section.end_page - section.start_page + 1)
            if current_span > max_length:
                section.end_page = min(total_pages, section.start_page + max_length - 1)
                reason_parts.append(f"max_enforced_{current_span}->{max_length}")
                applied = True

            section.final_span = max(1, section.end_page - section.start_page + 1)

        if applied and reason_parts:
            suffix = f" [{source}]" if source else ""
            section.constraint_reason = "; ".join(reason_parts) + suffix
            section.constraint_applied = True
            logger.info(
                f"Contrainte section appliquee ({section.section_type}): {section.constraint_reason} "
                f"(pages {section.start_page}-{section.end_page})"
            )
        elif section.end_page is not None:
            # Renseigner aussi dans le cas sans ajustement
            section.detected_span = section.detected_span or (section.end_page - section.start_page + 1)
            section.final_span = section.final_span or section.detected_span

        return section

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

    def _is_section_bounds_suspicious(self, section: LocatedSection, total_pages: int) -> bool:
        """Verifier si les bornes d'une section semblent anormales.

        Args:
            section: Section a verifier
            total_pages: Nombre total de pages du document

        Returns:
            True si les bornes sont suspectes (trop courtes, trop longues, etc.).
        """
        if not section.start_page or not section.end_page:
            return True

        constraints = self._get_section_length_constraints(section.section_type)
        min_length = constraints["min_length"]
        max_length = constraints["max_length"]
        length = section.end_page - section.start_page + 1
        if length < min_length:
            return True

        if length > max_length:
            return True

        if total_pages and length > total_pages * 0.8:
            return True

        return False

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

    def _get_manual_override(self, section_type: str) -> tuple[int, int] | None:
        """Obtenir l'override manuel de pages depuis la configuration.

        Args:
            section_type: Type de section (gestion_capital ou gestion_risques)

        Returns:
            Tuple (start_page, end_page) ou None
        """
        if not self.bank_code or not self.quarter:
            return None

        bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
        sections = bank_data.get("sections", {})

        # Mapper les noms internes -> configuration
        section_map = {
            "gestion_capital": "capital_management",
            "gestion_risques": "risk_management",
            "gestion_reglementation": "regulatory_updates",
        }
        config_name = section_map.get(section_type)
        if not config_name:
            return None

        section_config = sections.get(config_name, {})
        page_ranges = section_config.get("page_ranges", {})

        # Chercher la cle exacte (ex: "t1_2025")
        range_key = f"{self.quarter}_{self.year}"
        if range_key in page_ranges:
            range_data = page_ranges[range_key]
            return (range_data.get("start"), range_data.get("end"))

        return None

    def _get_page_number_offset(self) -> int:
        """Obtenir l'offset de numerotation document -> physique pour la banque courante.

        CONVENTION DE NUMEROTATION:
        ===========================
        Certaines banques (ex: CIBC) ont un decalage entre:
        - La numerotation DOCUMENT: ce qui est affiche dans le pied de page et la TOC
        - La numerotation PHYSIQUE: la position reelle de la page dans le PDF (Adobe, pdfplumber)

        Exemple CIBC (offset = 3):
        - Les 3 premieres pages (couverture, TOC) n'ont pas de numero dans le document
        - Page document 1 = Page physique 4
        - Page document 20 = Page physique 23

        Usage:
        - page_ranges dans bank_profiles.json utilisent la numerotation DOCUMENT
        - L'offset est ajoute automatiquement: page_physique = page_document + offset
        - page_number_offsets peut definir un offset plus precis par periode
          (ex. t4_2025) sans modifier l'offset par defaut des autres trimestres

        Returns:
            Offset (ex. 3 pour CIBC) ou 0 si pas de decalage.
        """
        if not self.bank_code or not self.bank_config:
            return 0
        bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
        period_offsets = bank_data.get("page_number_offsets", {})
        if isinstance(period_offsets, dict):
            quarter_key = str(self.quarter or "").strip().lower()
            period_keys = []
            if quarter_key and self.year:
                period_keys.append(f"{quarter_key}_{self.year}")
            if quarter_key:
                period_keys.append(quarter_key)
            for key in period_keys:
                if key in period_offsets:
                    offset = period_offsets.get(key, 0)
                    return int(offset) if offset else 0
        offset = bank_data.get("page_number_offset", 0)
        return int(offset) if offset else 0

    def _uses_document_page_numbers(self, detection_method: str) -> bool:
        """Indiquer si la methode de detection fournit des numeros en numerotation document.

        Seules les sections issues de la TOC ou des overrides manuels utilisent la
        numerotation document (pied de page / config). Les methodes scan, genai_fallback,
        visual parcourent le PDF par page physique et retournent deja des numeros physiques.

        Returns:
            True si l'offset page_number_offset doit etre applique (toc, manual_override*).
        """
        if not detection_method:
            return False
        return detection_method.startswith("toc") or detection_method.startswith("manual_override")

    def _get_config_section_names(self, section_type: str) -> list[str]:
        """Recuperer les noms configures pour un type de section (banque courante).

        Args:
            section_type: Type interne (gestion_capital, gestion_risques, ...)

        Returns:
            Liste des noms de section configures.
        """
        if not self.bank_code or not self.bank_config:
            return []

        section_name_map = {
            "gestion_capital": "capital_management",
            "gestion_risques": "risk_management",
            "gestion_reglementation": "regulatory_updates",
        }
        config_name = section_name_map.get(section_type)
        if not config_name:
            return []

        bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
        section_cfg = bank_data.get("sections", {}).get(config_name, {})
        names = section_cfg.get("names", [])
        return [n for n in names if isinstance(n, str) and n.strip()]

    def _section_alias_keys(self, section_type: str) -> list[str]:
        """Retourner les cles d'alias compatibles avec la taxonomie courante."""
        canonical = canonicalize_section(section_type)
        legacy_key = {
            "capital_management": "gestion_capital",
            "risk_management": "gestion_risques",
            "regulatory_updates": "gestion_reglementation",
        }.get(canonical, "")

        keys: list[str] = []
        for key in (section_type, canonical, legacy_key):
            key = str(key or "").strip()
            if key and key not in keys:
                keys.append(key)
        return keys

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
            and capital.end_detection_method != "following_section_scan"
        ):
            capital.end_page = risk.start_page - 1
            capital.end_detection_method = "cibc_next_section_start"
            self._apply_section_length_constraints(capital, total_pages, source="cibc_recalibration")

        if risk and not risk.detection_method.startswith("manual_override"):
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

    def _is_t4_quarter(self) -> bool:
        """Indiquer si le rapport courant est un T4."""
        return str(self.quarter or "").strip().lower() == "t4"

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

    def _needs_genai_fallback(self, sections: list[LocatedSection]) -> bool:
        """Determiner si le fallback GenAI est necessaire.

        Criteres:
        - Moins de 2 sections trouvees
        - Confiance moyenne inferieure a 0.7

        Args:
            sections: Sections deja detectees

        Returns:
            True si GenAI fallback necessaire
        """
        # Cas 1: Moins de 2 sections trouvees
        if len(sections) < 2:
            logger.info("GenAI fallback: moins de 2 sections trouvees")
            return True

        # Cas 2: Confiance moyenne trop faible
        avg_confidence = sum(s.confidence for s in sections) / len(sections)
        if avg_confidence < 0.7:
            logger.info(f"GenAI fallback: confiance moyenne faible ({avg_confidence:.2f})")
            return True

        return False

    def _detect_with_genai(self, pdf_path: Path) -> list[LocatedSection]:
        """Utiliser GenAI pour detecter les sections.

        Args:
            pdf_path: Chemin vers le PDF

        Returns:
            Liste de LocatedSection detectees par GenAI
        """
        try:
            from .genai_toc_detector import GenAITOCDetector
        except ImportError:
            logger.warning("genai_toc_detector non disponible")
            return []

        try:
            detector = GenAITOCDetector()
            genai_results = detector.find_and_extract_sections(pdf_path)

            # Convertir en LocatedSection
            sections = []
            for result in genai_results:
                section = LocatedSection(
                    section_type=result.section_type,
                    title_found=result.title_found,
                    start_page=result.start_page,
                    end_page=None,  # Sera determine plus tard
                    confidence=result.confidence,
                    detection_method="genai_fallback",
                    end_detection_method="",
                )
                sections.append(section)

            return sections

        except Exception as e:
            logger.error(f"Erreur GenAI fallback: {e}")
            return []

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

    def _extract_visual_elements(self, pdf_path: Path) -> dict[int, list[VisualTextElement]]:
        """Extraire les elements de texte avec leurs caracteristiques visuelles.

        Utilise pdfplumber pour obtenir:
        - Taille de police
        - Nom de la police (pour detecter le gras)
        - Position sur la page

        Args:
            pdf_path: Chemin vers le PDF

        Returns:
            Dict {page_number: liste de VisualTextElement}
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber non installe pour detection visuelle")
            return {}

        visual_elements: dict[int, list[VisualTextElement]] = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_elements = []

                    # Extraire les caracteres individuels avec leurs proprietes
                    chars = page.chars or []

                    if not chars:
                        continue

                    # Regrouper les caracteres par ligne (meme position Y approximative)
                    lines: dict[int, list] = {}
                    tolerance = 3  # Tolerance pour regrouper sur la meme ligne

                    for char in chars:
                        y_pos = round(char.get("top", 0) / tolerance)
                        if y_pos not in lines:
                            lines[y_pos] = []
                        lines[y_pos].append(char)

                    # Traiter chaque ligne - construire UNE entree par ligne
                    for line_idx, (y_key, line_chars) in enumerate(sorted(lines.items())):
                        # Trier par position X
                        line_chars.sort(key=lambda c: c.get("x0", 0))

                        # Collecter toutes les informations de la ligne
                        line_text = "".join(c.get("text", "") for c in line_chars)

                        if len(line_text.strip()) < 5:
                            continue

                        # Taille de police: prendre le MAX (pas la moyenne)
                        sizes = [c.get("size", 0) for c in line_chars if c.get("size", 0) > 0]
                        max_font_size = max(sizes) if sizes else 0

                        # Police: verifier si au moins un caractere est en gras
                        fonts = set(c.get("fontname", "") for c in line_chars)
                        is_bold = any(self._is_bold_font(f) for f in fonts)

                        # Position
                        x0 = min(c.get("x0", 0) for c in line_chars)
                        y0 = min(c.get("top", 0) for c in line_chars)
                        x1 = max(c.get("x1", 0) for c in line_chars)
                        y1 = max(c.get("bottom", 0) for c in line_chars)

                        # Creer l'element
                        elem = VisualTextElement(
                            text=line_text.strip(),
                            page=page_num,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            font_size=max_font_size,
                            font_name=next(iter(fonts), ""),
                            is_bold=is_bold,
                            is_uppercase=line_text.strip().isupper(),
                            line_number=line_idx,
                            page_width=float(getattr(page, "width", 0) or 0),
                            page_height=float(getattr(page, "height", 0) or 0),
                        )
                        page_elements.append(elem)

                    visual_elements[page_num] = page_elements

        except Exception as e:
            logger.warning(f"Erreur extraction visuelle: {e}")
            return {}

        return visual_elements

    def _is_bold_font(self, font_name: str) -> bool:
        """Determiner si le nom de police indique une graisse en gras."""
        if not font_name:
            return False
        font_lower = font_name.lower()
        return any(marker in font_lower for marker in ["bold", "heavy", "black", "demi", "semi", "medium"])

    def _merge_adjacent_elements(self, elements: list[VisualTextElement]) -> list[VisualTextElement]:
        """Fusionner les elements adjacents sur la meme ligne.

        Args:
            elements: Liste d'elements a fusionner

        Returns:
            Liste d'elements fusionnes par ligne
        """
        if not elements:
            return []

        # Trier par ligne puis par position X
        elements.sort(key=lambda e: (e.line_number, e.x0))

        merged = []
        current = None

        for elem in elements:
            if current is None:
                current = elem
                continue

            # Meme ligne et proche (ecart < 50 pixels)?
            if elem.line_number == current.line_number and elem.x0 - current.x1 < 50:
                # Fusionner
                current = VisualTextElement(
                    text=current.text + " " + elem.text,
                    page=current.page,
                    x0=current.x0,
                    y0=min(current.y0, elem.y0),
                    x1=elem.x1,
                    y1=max(current.y1, elem.y1),
                    font_size=(current.font_size + elem.font_size) / 2,
                    font_name=current.font_name,
                    is_bold=current.is_bold or elem.is_bold,
                    is_uppercase=current.is_uppercase and elem.is_uppercase,
                    line_number=current.line_number,
                    page_width=current.page_width or elem.page_width,
                    page_height=current.page_height or elem.page_height,
                )
            else:
                merged.append(current)
                current = elem

        if current:
            merged.append(current)

        return merged

    def _detect_section_headers_visual(
        self,
        visual_elements: dict[int, list[VisualTextElement]],
        text_by_page: dict[int, str],
    ) -> list[LocatedSection]:
        """Detecter les titres de sections en utilisant les caracteristiques visuelles.

        Cette methode cherche les elements qui:
        - Ont une grande taille de police (> moyenne de la page)
        - Sont en gras
        - Sont en majuscules
        - Sont positionnes en haut de page
        - Correspondent aux patterns de sections cibles

        Args:
            visual_elements: Elements visuels par page
            text_by_page: Texte par page (pour validation contextuelle)

        Returns:
            Liste des sections detectees visuellement
        """
        # Collecter TOUS les candidats d'abord, puis choisir le meilleur par type
        candidates: dict[str, list[tuple[LocatedSection, float]]] = {
            "gestion_capital": [],
            "gestion_risques": [],
        }

        # Calculer la taille de police moyenne du document
        all_sizes = []
        for page_elements in visual_elements.values():
            for elem in page_elements:
                if elem.font_size > 0:
                    all_sizes.append(elem.font_size)

        if not all_sizes:
            return []

        avg_font_size = sum(all_sizes) / len(all_sizes)
        header_threshold = avg_font_size * 1.2  # 20% plus grand que la moyenne

        logger.debug(f"Detection visuelle: taille moyenne={avg_font_size:.1f}, seuil titres={header_threshold:.1f}")

        # Scanner les pages (ignorer les premieres pages = TDM, intro)
        for page_num in sorted(visual_elements.keys()):
            if page_num < 5:
                continue

            page_elements = visual_elements[page_num]
            page_text = text_by_page.get(page_num, "")

            for elem in page_elements:
                # Verifier si c'est potentiellement un titre
                is_header_candidate = (
                    elem.font_size >= header_threshold or elem.is_bold or (elem.is_uppercase and len(elem.text) > 15)
                )

                if not is_header_candidate:
                    continue

                # Position: titre en haut de page (premier tiers)
                # Ou en debut de ligne (x0 proche de la marge gauche)
                is_top_of_page = elem.line_number < 10
                is_left_aligned = elem.x0 < 150  # Marge gauche typique

                if not (is_top_of_page or is_left_aligned):
                    continue

                # Verifier si le texte correspond a un pattern de section
                text_normalized = normalize_text(elem.text)

                for section_type, config in self.compiled_patterns.items():
                    # Verifier les patterns
                    for pattern in config["regex"]:
                        if pattern.search(elem.text):
                            # Calculer un score de confiance base sur les caracteristiques visuelles
                            visual_score = self._calculate_visual_confidence(
                                elem, avg_font_size, is_top_of_page, is_left_aligned
                            )

                            # Creer une section temporaire pour validation contextuelle
                            temp_section = LocatedSection(
                                section_type=section_type,
                                title_found=elem.text,
                                start_page=page_num,
                                end_page=min(page_num + 10, max(text_by_page.keys())),
                                detection_method="visual_temp",
                            )

                            # Validation contextuelle
                            is_valid, content_score = self._validate_section_content(temp_section, text_by_page)

                            final_confidence = visual_score * 0.6 + content_score * 0.4

                            if final_confidence > 0.3:  # Seuil plus bas pour collecter
                                section = LocatedSection(
                                    section_type=section_type,
                                    title_found=elem.text,
                                    start_page=page_num,
                                    confidence=final_confidence,
                                    detection_method="visual",
                                )
                                # Ajouter aux candidats avec le score visuel brut
                                # (taille de police comme critere de departage)
                                candidates[section_type].append((section, elem.font_size))
                                logger.debug(
                                    f"Candidat visuel: {section_type} page {page_num} "
                                    f"(taille={elem.font_size:.1f}, gras={elem.is_bold}, "
                                    f"conf={final_confidence:.2f})"
                                )
                            break

        # Selectionner le meilleur candidat pour chaque type de section
        # Critere: priorite a la taille de police (titres plus grands = plus fiables)
        sections = []
        for section_type, section_candidates in candidates.items():
            if not section_candidates:
                continue

            # Trier par taille de police (desc) puis par confiance (desc)
            section_candidates.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
            best_section, best_size = section_candidates[0]

            # Verifier que le meilleur a une confiance acceptable
            if best_section.confidence > 0.4:
                sections.append(best_section)
                logger.info(
                    f"Section detectee visuellement: {section_type} page {best_section.start_page} "
                    f"(taille={best_size:.1f}, conf={best_section.confidence:.2f}) "
                    f"[{len(section_candidates)} candidats]"
                )

        return sections

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

    def _matches_boundary_title(
        self,
        text: str,
        patterns: list[re.Pattern],
        title_candidates: list[str],
    ) -> bool:
        """Indique si un bloc titre correspond à une frontière de section suivante."""
        stripped = str(text or "").strip()
        if not stripped:
            return False
        unstuttered = self._unstutter_pdf_text(stripped)
        value = normalize_text(stripped)
        for title in title_candidates:
            title_norm = normalize_text(title)
            if not title_norm:
                continue
            if title_norm in value or value in title_norm or self._text_similarity(value, title_norm) > 0.75:
                return True
        for pattern in patterns:
            if pattern.search(stripped) or pattern.search(unstuttered):
                return True
        return False

    def _find_boundary_header_on_page(
        self,
        page: int,
        patterns: list[re.Pattern],
        title_candidates: list[str],
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> VisualTextElement | None:
        """Localise le titre de la section suivante sur une page partagée."""
        matches: list[VisualTextElement] = []
        for elem in visual_elements.get(page, []):
            if not elem.is_likely_header:
                continue
            if not self._matches_boundary_title(elem.text, patterns, title_candidates):
                continue
            bbox = elem.bbox_norm
            if not bbox or float(bbox[1]) <= SHARED_PAGE_TOP_THRESHOLD:
                continue
            matches.append(elem)
        if not matches:
            return None
        return max(matches, key=lambda elem: float(elem.y0))

    def _refine_shared_page_boundaries(
        self,
        sections: list[LocatedSection],
        toc_entries: list[TocEntry],
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> list[LocatedSection]:
        """Étend end_page à la page partagée quand la section suivante ne commence pas en haut."""
        if not sections or not visual_elements:
            return sections

        refined: list[LocatedSection] = []
        for section in sections:
            if section.end_page is None:
                refined.append(section)
                continue

            boundary_page = int(section.end_page) + 1
            patterns = self.following_patterns.get(section.section_type, [])
            title_candidates = self._next_toc_boundary_title_candidates(section, toc_entries)
            boundary = self._find_boundary_header_on_page(
                boundary_page,
                patterns,
                title_candidates,
                visual_elements,
            )
            if boundary is None:
                refined.append(section)
                continue

            bbox_norm = boundary.bbox_norm
            if not bbox_norm:
                refined.append(section)
                continue

            logger.info(
                "Page partagée détectée pour %s: extension p.%s -> p.%s, frontière '%s' y=%.3f",
                section.section_type,
                section.end_page,
                boundary_page,
                boundary.text[:60],
                float(bbox_norm[1]),
            )
            refined.append(
                replace(
                    section,
                    end_page=boundary_page,
                    end_anchor_page=boundary_page,
                    end_anchor_text=boundary.text,
                    end_anchor_bbox_norm=list(bbox_norm),
                    end_detection_method=f"{section.end_detection_method}+shared_page"
                    if section.end_detection_method
                    else "shared_page",
                )
            )
        return refined

    def _get_section_anchor_candidates(self, section: LocatedSection) -> list[str]:
        """Retourner les libelles exacts a tester pour l'ancre de debut de section."""
        candidates: list[str] = []
        title_found = str(section.title_found or "").strip()
        if title_found:
            candidates.append(title_found)

        for section_key in self._section_alias_keys(section.section_type):
            for alias in SECTION_TITLE_ALIASES.get(section_key, []):
                alias = str(alias or "").strip()
                if alias and normalize_text(alias) not in {normalize_text(existing) for existing in candidates}:
                    candidates.append(alias)

            for alias in self._get_config_section_names(section_key):
                alias = str(alias or "").strip()
                if alias and normalize_text(alias) not in {normalize_text(existing) for existing in candidates}:
                    candidates.append(alias)

        return candidates

    def _resolve_section_anchor(
        self,
        section: LocatedSection,
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> LocatedSection:
        """Resoudre une ancre intra-page a partir du bloc titre reel de la section."""
        if not section.start_page:
            return replace(section, anchor_found=False)

        page_elements = visual_elements.get(section.start_page, [])
        if not page_elements:
            return replace(section, anchor_found=False)

        candidates = self._get_section_anchor_candidates(section)
        if not candidates:
            return replace(section, anchor_found=False)

        def _candidate_sort_key(elem: VisualTextElement) -> tuple[int, float, float, int]:
            """Clé de tri privilégiant les en-têtes, position verticale, grande police."""
            return (
                0 if elem.is_likely_header else 1,
                float(elem.y0),
                -float(elem.font_size),
                int(elem.line_number),
            )

        for candidate_text in candidates:
            candidate_variants = self._title_match_variants(candidate_text)
            matches = [elem for elem in page_elements if self._title_match_variants(elem.text) & candidate_variants]
            if not matches:
                continue

            best = sorted(matches, key=_candidate_sort_key)[0]
            bbox_norm = best.bbox_norm
            if not bbox_norm:
                continue

            return replace(
                section,
                anchor_page=section.start_page,
                anchor_text=best.text,
                anchor_bbox_norm=bbox_norm,
                anchor_found=True,
            )

        return replace(section, anchor_found=False)

    def _resolve_section_anchors(
        self,
        sections: list[LocatedSection],
        visual_elements: dict[int, list[VisualTextElement]],
    ) -> list[LocatedSection]:
        """Resoudre les ancres de toutes les sections localisees."""
        resolved: list[LocatedSection] = []
        for section in sections:
            anchored = self._resolve_section_anchor(section, visual_elements)
            if anchored.anchor_found:
                logger.info(
                    "Ancre section resolue: %s page %s -> '%s'",
                    anchored.section_type,
                    anchored.anchor_page,
                    anchored.anchor_text,
                )
            elif section.detection_method.startswith("manual_override"):
                logger.debug(
                    "Ancre section non resolue pour override manuel: %s page %s",
                    section.section_type,
                    section.start_page,
                )
            else:
                logger.warning(
                    "Ancre section introuvable: %s page %s title_found='%s'",
                    section.section_type,
                    section.start_page,
                    section.title_found,
                )
            resolved.append(anchored)
        return resolved

    def _calculate_visual_confidence(
        self,
        elem: VisualTextElement,
        avg_font_size: float,
        is_top_of_page: bool,
        is_left_aligned: bool,
    ) -> float:
        """Calculer un score de confiance base sur les caracteristiques visuelles.

        Args:
            elem: Element visuel
            avg_font_size: Taille moyenne de police du document
            is_top_of_page: Element en haut de page
            is_left_aligned: Element aligne a gauche

        Returns:
            Score entre 0 et 1
        """
        score = 0.0

        # Taille de police (max 0.35)
        if elem.font_size > avg_font_size * 1.5:
            score += 0.35  # Beaucoup plus grand
        elif elem.font_size > avg_font_size * 1.2:
            score += 0.25  # Plus grand
        elif elem.font_size > avg_font_size:
            score += 0.15

        # Gras (max 0.25)
        if elem.is_bold:
            score += 0.25

        # Majuscules (max 0.15)
        if elem.is_uppercase:
            score += 0.15

        # Position (max 0.25)
        if is_top_of_page:
            score += 0.15
        if is_left_aligned:
            score += 0.10

        return min(1.0, score)

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

    def _determine_end_pages(
        self,
        sections: list[LocatedSection],
        text_by_page: dict[int, str],
        toc_entries: list[TocEntry],
        total_pages: int,
    ) -> list[LocatedSection]:
        """Determiner les pages de fin avec la logique hybride a 3 niveaux.

        Priorite:
        1. Override manuel (deja applique)
        2. TDM - page debut section suivante
        3. Scan - detection pattern section suivante
        4. Fallback - estimation contextuelle

        Args:
            sections: Sections avec start_page
            text_by_page: Texte par page
            toc_entries: Entrees TDM
            total_pages: Nombre total de pages

        Returns:
            Sections avec end_page determine
        """
        if not sections:
            return sections

        # Trier par page de debut
        sections = sorted(sections, key=lambda s: s.start_page)

        for i, section in enumerate(sections):
            # Si end_page deja defini (override, TDM, etc.), appliquer quand meme
            # les contraintes de longueur avant de continuer.
            if section.end_page is not None:
                self._apply_section_length_constraints(section, total_pages, source="predefined")
                continue

            constraints = self._get_section_length_constraints(section.section_type)
            default_length = constraints["default_length"]

            end_page = None
            end_method = ""

            # Niveau 2: Scanner pour la section suivante explicite. Les titres
            # "followed_by" bornent mieux les sections vigie que la prochaine
            # section cible quand des blocs intermediaires existent.
            if not end_page:
                end_page, end_method = self._detect_section_end(
                    section.section_type, section.start_page, text_by_page, total_pages
                )

            # Niveau 3: utiliser la prochaine section cible detectee quand elle
            # existe et qu'aucun titre de fin plus precis n'a ete trouve.
            if not end_page and i + 1 < len(sections):
                end_page = sections[i + 1].start_page - 1
                end_method = "next_target_section"

            # Niveau 4: Utiliser la TDM si aucune borne locale n'a ete trouvee.
            if toc_entries and not end_page:
                end_page, end_method = self._find_end_from_toc(section.section_type, section.start_page, toc_entries)

            # Niveau 5: Fallback - estimation contextuelle
            if not end_page:
                # Estimation contextuelle bornee par contraintes de la section
                end_page = min(section.start_page + default_length - 1, total_pages)
                end_method = "estimation"

            section.end_page = end_page
            section.end_detection_method = end_method

            # Affiner les limites avec les sous-sections (Amélioration 3)
            if section.section_type in {"gestion_capital", "gestion_risques"}:
                section = self._refine_bounds_with_subsections(section, text_by_page)

            self._apply_section_length_constraints(section, total_pages, source="determine_end")

        return sections

    def _get_subsection_patterns(self, section_type: str) -> list[re.Pattern]:
        """Obtenir les patterns regex pour les sous-sections d'un type de section.

        Args:
            section_type: Type de section (gestion_capital ou gestion_risques)

        Returns:
            Liste de patterns regex compiles pour les sous-sections
        """
        # Patterns de sous-sections selon le type
        subsection_patterns_dict = {
            "gestion_risques": [
                r"risque\s+de\s+cr[eé]dit",
                r"risque\s+de\s+march[eé]",
                r"risque\s+de\s+liquidit[eé]",
                r"risque\s+op[eé]rationnel",
                r"risque\s+de\s+taux\s+d['']inter[eé]t",
                r"risque\s+de\s+change",
                r"credit\s+risk",
                r"market\s+risk",
                r"liquidity\s+risk",
                r"operational\s+risk",
            ],
            "gestion_capital": [
                r"ratio\s+CET1",
                r"ratio\s+de\s+levier",
                r"ratio\s+de\s+liquidit[eé]",
                r"fonds\s+propres\s+r[eé]glementaires",
                r"capital\s+r[eé]glementaire",
                r"Tier\s+1",
                r"Tier\s+2",
                r"TLAC",
                r"LCR",
                r"NSFR",
            ],
        }

        patterns = subsection_patterns_dict.get(section_type, [])

        # Utiliser aussi les patterns depuis SECTION_PATTERNS si disponibles
        if section_type in SECTION_PATTERNS:
            config_subsections = SECTION_PATTERNS[section_type].get("subsections", [])
            patterns.extend(config_subsections)

        # Compiler les patterns
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        return compiled

    def _refine_bounds_with_subsections(self, section: LocatedSection, text_by_page: dict[int, str]) -> LocatedSection:
        """Affiner les limites d'une section en detectant les sous-sections.

        Pour "Gestion des risques":
        - Detecte "Risque de credit", "Risque de marche", etc.
        - La section commence au premier sous-titre
        - La section se termine avant la prochaine section principale

        Args:
            section: Section a affiner
            text_by_page: Texte par page

        Returns:
            Section avec limites affinees si des sous-sections sont trouvees
        """
        # NOTE: L'affinement des limites de section a ete desactive car il causait
        # des sections trop courtes (1-2 pages au lieu de 20+).
        #
        # L'ancienne logique:
        # 1. Cherchait des sous-sections dans les 5 premieres pages et deplacait le debut
        # 2. Cherchait des sous-sections dans les 10 dernieres pages et coupait la fin
        #
        # Problemes:
        # - La page de debut detectee par scan/TOC est generalement correcte
        # - Deplacer le debut faisait perdre le titre de section et l'introduction
        # - Couper la fin a la premiere sous-section trouvee eliminait le reste de la section
        #
        # Solution: Les limites de section sont maintenant determinees uniquement par:
        # - Le scan de titres (detection_method: scan)
        # - La table des matieres (detection_method: toc)
        # - Les overrides manuels (detection_method: manual_override)
        # - La detection de la section suivante (end_detection_method: following_section_*)

        return section

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

    def _correct_section_bounds(
        self,
        section: LocatedSection,
        toc_detections: list[TocEntry],
        scan_detections: list[LocatedSection],
    ) -> LocatedSection | None:
        """Corriger les limites d'une section selon le consensus des methodes.

        Utilise la mediane ponderee des pages detectees par toutes les methodes.

        Args:
            section: Section a corriger
            toc_detections: Entrees TDM correspondant a cette section
            scan_detections: Sections detectees par scan

        Returns:
            Section corrigee ou None si aucune correction n'est possible
        """
        # Collecter toutes les pages de debut
        start_pages = []
        if section.start_page:
            start_pages.append((section.start_page, 1.0))

        for toc_entry in toc_detections:
            if toc_entry.page:
                start_pages.append((toc_entry.page, 0.8))

        for scan_section in scan_detections:
            if scan_section.start_page:
                start_pages.append((scan_section.start_page, 0.6))

        # Collecter toutes les pages de fin
        end_pages = []
        if section.end_page:
            end_pages.append((section.end_page, 1.0))

        for scan_section in scan_detections:
            if scan_section.end_page:
                end_pages.append((scan_section.end_page, 0.6))

        # Calculer la mediane ponderee pour le debut
        corrected_start = section.start_page
        if start_pages:
            sorted_starts = sorted(start_pages, key=lambda x: x[0])
            total_weight = sum(w for _, w in sorted_starts)

            if total_weight > 0:
                # Calculer la mediane ponderee
                cumulative_weight = 0
                median_weight = total_weight / 2

                for page, weight in sorted_starts:
                    cumulative_weight += weight
                    if cumulative_weight >= median_weight:
                        corrected_start = page
                        break

        # Calculer la mediane ponderee pour la fin
        corrected_end = section.end_page
        if end_pages:
            sorted_ends = sorted(end_pages, key=lambda x: x[0])
            total_weight = sum(w for _, w in sorted_ends)

            if total_weight > 0:
                cumulative_weight = 0
                median_weight = total_weight / 2

                for page, weight in sorted_ends:
                    cumulative_weight += weight
                    if cumulative_weight >= median_weight:
                        corrected_end = page
                        break

        # Verifier si une correction est necessaire
        start_changed = corrected_start != section.start_page
        end_changed = corrected_end != section.end_page

        if start_changed or end_changed:
            # Creer une copie corrigee
            corrected = LocatedSection(
                section_type=section.section_type,
                title_found=section.title_found,
                start_page=corrected_start,
                end_page=corrected_end,
                confidence=section.confidence,
                detection_method=f"{section.detection_method}_corrected",
                end_detection_method=f"{section.end_detection_method}_corrected",
            )

            logger.info(
                f"Limites corrigees pour {section.section_type}: "
                f"{section.start_page}-{section.end_page} -> {corrected_start}-{corrected_end}"
            )

            return corrected

        return None

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

                # Essayer de corriger
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

    def _find_next_section_by_pattern(
        self, section_type: str, start_page: int, toc_entries: list[TocEntry]
    ) -> tuple[int, str] | None:
        """Trouver la prochaine section via les patterns 'followed_by' et les sections cibles.

        Cette methode cherche dans les entrees TDM :
        1. Les sections qui correspondent aux patterns 'followed_by'
        2. Les autres sections cibles (capital_management si on cherche risk_management, etc.)

        Args:
            section_type: Type de section actuelle (gestion_capital ou gestion_risques)
            start_page: Page de debut de la section actuelle
            toc_entries: Entrees TDM

        Returns:
            Tuple (page, title) ou None si aucune section suivante trouvee
        """
        if not self.bank_code or not self.bank_config:
            return None

        # Obtenir les sections suivantes configurees
        config_name = "capital_management" if section_type == "gestion_capital" else "risk_management"
        bank_data = self.bank_config.get("banks", {}).get(self.bank_code, {})
        sections = bank_data.get("sections", {})
        section_config = sections.get(config_name, {})
        followed_by = section_config.get("followed_by", [])

        # Obtenir aussi les noms de l'autre section cible
        other_config_name = "risk_management" if section_type == "gestion_capital" else "capital_management"
        other_section_config = sections.get(other_config_name, {})
        other_section_names = other_section_config.get("names", [])

        # Combiner les patterns a chercher
        patterns_to_search = list(followed_by)  # D'abord les patterns 'followed_by'
        patterns_to_search.extend(other_section_names)  # Puis les autres sections cibles

        if not patterns_to_search:
            return None

        # Obtenir la longueur minimale de la section courante
        min_length = self._get_section_length_constraints(section_type)["min_length"]

        # Chercher dans les entrees TDM apres start_page
        min_end_page = start_page + min_length
        entries_after = [e for e in toc_entries if e.page >= min_end_page]

        if not entries_after:
            return None

        # Trier par page croissante
        entries_after.sort(key=lambda e: e.page)

        # Chercher la premiere entree qui correspond a un pattern
        for entry in entries_after:
            entry_title_normalized = normalize_text(entry.title)

            for pattern_name in patterns_to_search:
                pattern_normalized = normalize_text(pattern_name)

                # Match partiel ou exact
                if (
                    pattern_normalized in entry_title_normalized
                    or entry_title_normalized in pattern_normalized
                    or self._text_similarity(entry_title_normalized, pattern_normalized) > 0.7
                ):
                    # Determiner la source du match pour le log
                    if pattern_name in followed_by:
                        source = "followed_by"
                    else:
                        source = f"other_target_section ({other_config_name})"

                    logger.debug(
                        f"Section suivante trouvee par pattern: '{entry.title}' page {entry.page} "
                        f"(pattern: '{pattern_name}', section: {section_type}, source: {source})"
                    )
                    return (entry.page, entry.title)

        return None

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

    def _detect_section_end(
        self,
        section_type: str,
        start_page: int,
        text_by_page: dict[int, str],
        total_pages: int,
    ) -> tuple[int | None, str]:
        """Detecter la fin d'une section en scannant le PDF.

        Cherche les patterns des sections qui suivent typiquement cette section.

        Args:
            section_type: Type de section
            start_page: Page de debut
            text_by_page: Texte par page
            total_pages: Nombre total de pages

        Returns:
            Tuple (end_page, method) ou (None, "")
        """
        following_patterns = self.following_patterns.get(section_type, [])

        if not following_patterns:
            return None, ""

        # Scanner les pages apres le debut de la section
        constraints = self._get_section_length_constraints(section_type)
        min_length = constraints["min_length"]
        max_length = constraints["max_length"]

        search_start = start_page + min_length
        search_end = min(start_page + max_length, total_pages)

        for page_num in range(search_start, search_end + 1):
            page_text = text_by_page.get(page_num, "")
            lines = page_text.split("\n")

            for line in lines:
                line_stripped = line.strip()
                line_unstuttered = self._unstutter_pdf_text(line_stripped)

                # Verifier si c'est un titre potentiel
                if not (
                    self._is_likely_section_title(line_stripped, page_text)
                    or self._is_likely_section_title(line_unstuttered, page_text)
                ):
                    continue
                if self._is_weak_section_scan_line(line_stripped, section_type) or self._is_weak_section_scan_line(
                    line_unstuttered, section_type
                ):
                    continue

                # Verifier contre les patterns des sections suivantes
                for pattern in following_patterns:
                    if pattern.search(line_stripped) or pattern.search(line_unstuttered):
                        # Verifier que ce n'est pas une sous-section
                        if self._is_risk_subsection(line_stripped):
                            continue

                        logger.debug(f"Fin de {section_type} detectee page {page_num}: {line_stripped[:40]}...")
                        return page_num - 1, "following_section_scan"

        return None, ""

    def _estimate_end_pages(self, sections: list[LocatedSection], total_pages: int) -> list[LocatedSection]:
        """Estimer les pages de fin pour les sections (methode legacy).

        Args:
            sections: Sections avec start_page
            total_pages: Nombre total de pages

        Returns:
            Sections avec end_page estime
        """
        if not sections:
            return sections

        # Trier par page de debut
        sections = sorted(sections, key=lambda s: s.start_page)

        for i, section in enumerate(sections):
            if section.end_page is None:
                if i + 1 < len(sections):
                    # La section se termine avant la section suivante
                    section.end_page = sections[i + 1].start_page - 1
                    section.end_detection_method = "next_section"
                else:
                    # Derniere section: estimer la fin
                    constraints = self._get_section_length_constraints(section.section_type)
                    estimated_length = constraints["default_length"]
                    section.end_page = min(section.start_page + estimated_length - 1, total_pages)
                    section.end_detection_method = "estimation"

                self._apply_section_length_constraints(section, total_pages, source="legacy_estimate")

        return sections


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
