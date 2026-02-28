"""UI/runtime configuration shared by Dash helpers."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
INDICATOR_EXPORT_DIR = OUTPUT_DIR / "indicator_tables"
INDICATOR_COMPARISON_DIR = OUTPUT_DIR / "comparisons"
LOGS_DIR = ROOT_DIR / "logs"

for _path in (OUTPUT_DIR, INDICATOR_EXPORT_DIR, INDICATOR_COMPARISON_DIR, LOGS_DIR):
    _path.mkdir(parents=True, exist_ok=True)

AVAILABLE_BANKS = {
    "bnc": "Banque Nationale du Canada",
    "bns": "Banque Scotia",
    "rbc": "Banque Royale du Canada",
    "td": "Toronto-Dominion",
    "bmo": "Banque de Montreal",
    "cibc": "CIBC",
}
