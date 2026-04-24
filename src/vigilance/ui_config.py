"""Configuration UI/runtime partagee par les utilitaires Dash."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
INDICATOR_EXPORT_DIR = OUTPUT_DIR / "indicator_tables"
LOGS_DIR = ROOT_DIR / "logs"

# Racine unique pour comparaisons indicateurs et texte (Dash + pipelines).
# Le mode reader (.exe VigieRegDesjardins) peut surcharger via la variable
# d'environnement ``VIGIE_RESULTATS_DIR`` -> chemin vers le dossier ``resultats``
# synchronise depuis SharePoint (OneDrive).
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
        # Mode reader (.exe) : OUTPUT_DIR peut pointer dans un bundle en
        # lecture seule. RESULTATS_DIR pointe alors vers SharePoint et existe
        # deja. On ignore silencieusement les erreurs pour ne pas bloquer
        # le demarrage.
        pass

AVAILABLE_BANKS = {
    "bnc": "Banque Nationale du Canada",
    "bns": "Banque Scotia",
    "rbc": "Banque Royale du Canada",
    "td": "Toronto-Dominion",
    "bmo": "Banque de Montreal",
    "cibc": "CIBC",
}
