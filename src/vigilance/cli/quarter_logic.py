"""Quarter deduction and PDF discovery utilities for the batch pipeline.

Reuses the canonical ``resolve_reference_period`` from compare_gpt but adds
directory-crawling helpers that locate PDF pairs inside the standardised
``Inputs/{BANK}/{YEAR}/`` tree **or** the legacy ``data/{bank}/`` tree.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Quarter deduction (mirrors compare_gpt but kept standalone so the CLI
# module does not need to import the heavy comparison stack).
# ---------------------------------------------------------------------------

_QUARTER_RE = re.compile(r"[qtQT]\s*([1-4])")


def normalize_quarter(value: str) -> str:
    """Return canonical form ``t1``…``t4``."""
    m = _QUARTER_RE.search(str(value or "").strip())
    if m:
        return f"t{m.group(1)}"
    raise ValueError(f"Trimestre invalide: {value!r}. Attendu: t1, t2, t3 ou t4.")


def resolve_previous_quarter(year: int, quarter: str) -> Tuple[int, str]:
    """Deduce the previous quarter from the current one.

    Rules
    -----
    T2-2025 → T1-2025
    T3-2025 → T2-2025
    T1-2025 → T3-2024  (year N-1)
    T4-2025 → T3-2025
    """
    q = normalize_quarter(quarter)
    mapping = {
        "t2": (year, "t1"),
        "t3": (year, "t2"),
        "t4": (year, "t3"),
        "t1": (year - 1, "t3"),
    }
    result = mapping.get(q)
    if result is None:
        raise ValueError(f"Trimestre invalide: {quarter!r}")
    return result


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

def _find_pdf_in_dir(directory: Path, bank: str, year: int, quarter: str) -> Path | None:
    """Search *directory* for a PDF matching the bank/year/quarter.

    Priority order:
    1. Strict canonical name:  ``{BANK}_{YEAR}_{Q}.pdf``  (e.g. BNC_2025_T2.pdf)
    2. Regex fallback for legacy names.
    """
    if not directory.is_dir():
        return None
    q = normalize_quarter(quarter)
    q_num = q[1]  # "1", "2", …

    # --- Priority 1: strict canonical name  {BANK}_{YEAR}_{T#}.pdf ---
    canonical = directory / f"{bank.upper()}_{year}_T{q_num}.pdf"
    if canonical.exists():
        return canonical

    # --- Priority 2: regex patterns (legacy support) ---
    patterns = [
        # BNC_2025_T2.pdf / BNC-2025-T2.pdf / BNC2025T2.pdf
        re.compile(
            rf"{re.escape(bank)}[_\-]?{year}[_\-]?[tTqQ]{q_num}",
            re.IGNORECASE,
        ),
        # t2-rapport-actionnaire-2025.pdf  or  T2_2025q2_report_fr.pdf
        re.compile(
            rf"[tTqQ]{q_num}[_\-]",
            re.IGNORECASE,
        ),
    ]
    for pdf in sorted(directory.glob("*.pdf")):
        for pat in patterns:
            if pat.search(pdf.name):
                return pdf
    return None



def find_pdf_pair(
    bank: str,
    year_current: int,
    quarter_current: str,
    *,
    inputs_root: Path | None = None,
    legacy_data_root: Path | None = None,
) -> Tuple[Path, Path]:
    """Locate the *current* and *previous* PDF files.

    Search order per quarter:
    1. ``inputs_root / BANK / YEAR / *.pdf``   (the new Inputs/ tree)
    2. ``legacy_data_root / BANK / *.pdf``      (the flat data/ tree)

    Returns ``(previous_pdf, current_pdf)``.
    Raises ``FileNotFoundError`` when either file cannot be found.
    """
    q_cur = normalize_quarter(quarter_current)
    year_prev, q_prev = resolve_previous_quarter(year_current, q_cur)

    search_dirs_current: list[Path] = []
    search_dirs_previous: list[Path] = []

    if inputs_root is not None:
        search_dirs_current.append(inputs_root / bank.upper() / str(year_current))
        search_dirs_previous.append(inputs_root / bank.upper() / str(year_prev))

    if legacy_data_root is not None:
        search_dirs_current.append(legacy_data_root / bank.lower())
        search_dirs_previous.append(legacy_data_root / bank.lower())

    current_pdf: Path | None = None
    for d in search_dirs_current:
        current_pdf = _find_pdf_in_dir(d, bank, year_current, q_cur)
        if current_pdf:
            break

    previous_pdf: Path | None = None
    for d in search_dirs_previous:
        previous_pdf = _find_pdf_in_dir(d, bank, year_prev, q_prev)
        if previous_pdf:
            break

    if current_pdf is None:
        raise FileNotFoundError(
            f"PDF introuvable pour {bank.upper()} {q_cur.upper()}-{year_current}. "
            f"Répertoires explorés: {[str(d) for d in search_dirs_current]}"
        )
    if previous_pdf is None:
        raise FileNotFoundError(
            f"PDF introuvable pour {bank.upper()} {q_prev.upper()}-{year_prev}. "
            f"Répertoires explorés: {[str(d) for d in search_dirs_previous]}"
        )

    return previous_pdf, current_pdf
