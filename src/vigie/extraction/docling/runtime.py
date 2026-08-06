"""Dependances optionnelles, constantes de reglage et patterns de sections.

Extrait de ``docling_processor.py`` sans modification. Regroupe les imports
gardes par try/except (utilitaires memoire, cache PDF, constantes de config)
pour que les mixins y accedent sans dupliquer la logique de repli.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("vigie.extraction.docling_processor")

# Import de la gestion memoire
try:
    from vigie.support.utils.memory import (  # noqa: F401 - re-export
        ChunkedProcessor,
        check_memory_threshold,
        cleanup_memory,
        get_memory_usage_mb,
        with_memory_check,
    )

    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False
    # Replis explicites : dans le module d'origine ces noms restaient simplement
    # indefinis, car ils ne sont lus que sous la garde MEMORY_UTILS_AVAILABLE.
    # Les definir a None permet aux mixins de les importer sans changer le
    # comportement (la garde est False, ils ne sont jamais appeles).
    ChunkedProcessor = None
    check_memory_threshold = None
    cleanup_memory = None
    get_memory_usage_mb = None
    with_memory_check = None
    logger.debug("Utilitaires memoire non disponibles")

# Import des constantes
try:
    from vigie.support.config.constants import EXTRACTION, MEMORY  # noqa: F401 - re-export

    CHUNK_SIZE_PAGES = EXTRACTION.CHUNK_SIZE_PAGES
    DPI = EXTRACTION.DPI
    DPI_FAST = EXTRACTION.DPI_FAST
except ImportError:
    CHUNK_SIZE_PAGES = 15
    DPI = 300
    DPI_FAST = 150

# Nombre de workers pour parallelliser les appels Vision (1 = sequentiel)
VISION_EXTRACTION_MAX_WORKERS = 4

# Import du gestionnaire de cache
try:
    from vigie.support.utils.pdf_cache import PDFCacheManager  # noqa: F401 - re-export

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    PDFCacheManager = None
    logger.debug("PDFCacheManager non disponible")

# Patterns pour détecter les titres de sections principales
# Focus: Gestion du capital et Gestion des risques
SECTION_TITLE_PATTERNS = [
    # Sections de capital
    (r"^gestion\s+du\s+capital$", "Gestion du capital", 1),
    (r"^capital\s+management$", "Capital Management", 1),
    (r"^situation\s+des\s+fonds\s+propres$", "Situation des fonds propres", 1),
    (r"^gestion\s+des\s+fonds\s+propres$", "Gestion des fonds propres", 1),
    (r"^fonds\s+propres\s+r[eé]glementaires$", "Fonds propres réglementaires", 1),
    # Sections de risque
    (r"^gestion\s+des?\s+risques?$", "Gestion des risques", 1),
    (r"^gestion\s+du\s+risque$", "Gestion du risque", 1),
    (r"^risk\s+management$", "Risk Management", 1),
    (r"^facteurs?\s+de\s+risque", "Facteurs de risque", 1),
    (r"^risque\s+de\s+cr[eé]dit$", "Risque de crédit", 1),
    (r"^risque\s+de\s+march[eé]$", "Risque de marché", 1),
    (r"^risque\s+de\s+liquidit[eé]$", "Risque de liquidité", 1),
    (r"^risque\s+op[eé]rationnel$", "Risque opérationnel", 1),
    (r"^credit\s+risk$", "Credit Risk", 1),
    (r"^market\s+risk$", "Market Risk", 1),
    (r"^liquidity\s+risk$", "Liquidity Risk", 1),
]

# Compiler les patterns
COMPILED_SECTION_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), name, phase) for pattern, name, phase in SECTION_TITLE_PATTERNS
]
