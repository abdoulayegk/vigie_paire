"""Acces a la configuration par banque : patterns compiles, overrides et pagination.

Extrait de ``section_locator.py`` sans modification des corps de methodes.
Mixin consomme par ``SectionLocator``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..section_taxonomy import canonicalize_section
from .models import normalize_text
from .patterns import FOLLOWING_SECTION_PATTERNS, SECTION_PATTERNS, TOC_PATTERNS

# Nom de logger conserve a l'identique apres le decoupage, pour ne pas invalider
# une configuration de logging qui filtrerait sur ce nom.
logger = logging.getLogger("vigilance.extraction.section_locator")


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

    project_root = Path(__file__).resolve().parents[4]  # locator/ est un niveau plus bas que section_locator.py
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


# Conserve pour compatibilite : plusieurs tests et modules attendent un dict.
BANK_SECTION_NAMES = {bank: _get_bank_section_names(bank) for bank in ["bnc", "rbc", "td", "bmo", "bns", "cibc"]}


class BankConfigMixin:
    """Acces a la configuration par banque : patterns compiles, overrides et pagination."""

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
