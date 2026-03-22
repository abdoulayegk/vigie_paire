"""
Chargeur optionnel de patterns d'extraction configurables par banque.

Si aucun fichier de config ou module externe n'est fourni, get_patterns()
retourne None; les appelants font alors sans patterns (comportement par defaut).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_patterns(bank_code: str | None = None) -> dict[str, Any] | None:
    """
    Retourne les patterns d'extraction pour la banque donnee, ou None si non configures.

    Args:
        bank_code: Code banque optionnel pour des patterns specifiques.

    Returns:
        Dictionnaire de patterns ou None si aucun pattern n'est configure.
    """
    # Stub: pas de fichier de config des patterns pour l'instant.
    # Les appelants gerent None.
    return None
