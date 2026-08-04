"""Configuration UI/runtime partagee par les utilitaires Dash."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Racine du depot (dossier contenant ``pyproject.toml``).

    Evite un ``parents[N]`` fragile si le module bouge dans l'arborescence.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: src/vigie/interface/ui_config.py -> repo
    return current.parents[3]


ROOT_DIR = _repo_root()
OUTPUT_DIR = ROOT_DIR / "outputs"
INDICATOR_EXPORT_DIR = OUTPUT_DIR / "indicator_tables"
LOGS_DIR = ROOT_DIR / "logs"

# Racine unique pour comparaisons indicateurs et texte (Dash + pipelines).
# ``VIGIE_RESULTATS_DIR`` permet de pointer vers un dossier de resultats externe.
_RESULTATS_OVERRIDE = os.environ.get("VIGIE_RESULTATS_DIR", "").strip()
RESULTATS_DIR = Path(_RESULTATS_OVERRIDE) if _RESULTATS_OVERRIDE else OUTPUT_DIR / "resultats"
INDICATOR_COMPARISON_DIR = RESULTATS_DIR
TEXT_COMPARISON_DIR = RESULTATS_DIR

TEXT_EXTRACTION_DIR = OUTPUT_DIR / "text_extractions"

for _path in (
    OUTPUT_DIR,
    INDICATOR_EXPORT_DIR,
    RESULTATS_DIR,
    LOGS_DIR,
    TEXT_EXTRACTION_DIR,
):
    try:
        _path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Un chemin externe peut etre en lecture seule. On ignore silencieusement
        # les erreurs pour ne pas bloquer le demarrage.
        pass

# Cle technique (dossiers, CLI, YAML) -> nom long affiche.
AVAILABLE_BANKS = {
    "bnc": "Banque Nationale du Canada",
    "bns": "Banque Scotia",
    "rbc": "Banque Royale du Canada",
    "td": "Toronto-Dominion",
    "bmo": "Banque de Montreal",
    "cibc": "CIBC",
}

# Libelle court UI (peut differer du code, ex. bns -> Scotia).
BANK_SHORT_NAMES = {
    "bnc": "BNC",
    "bns": "Scotia",
    "rbc": "RBC",
    "td": "TD",
    "bmo": "BMO",
    "cibc": "CIBC",
}


def bank_short_name(bank_code: str) -> str:
    """Retourne le libelle court UI pour un code banque technique."""
    code = str(bank_code or "").strip().lower()
    if not code:
        return ""
    return BANK_SHORT_NAMES.get(code, code.upper())


def bank_option_label(bank_code: str) -> str:
    """Libelle du menu Banque : ``Scotia - Banque Scotia``, etc."""
    code = str(bank_code or "").strip().lower()
    short = bank_short_name(code)
    full = AVAILABLE_BANKS.get(code, "")
    if full and full != short:
        return f"{short} - {full}"
    return short or code.upper()
