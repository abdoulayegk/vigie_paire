"""Deduction de trimestres et decouverte de PDF pour le pipeline batch.

Reutilise le ``resolve_reference_period`` canonique de compare_gpt mais ajoute
des helpers de parcours de repertoire qui localisent les paires de PDF dans
l'arborescence standardisee ``Inputs/{BANK}/{YEAR}/`` ou l'ancienne
arborescence ``data/{bank}/``.
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
    """Retourner la forme canonique ``t1``..``t4``."""
    m = _QUARTER_RE.search(str(value or "").strip())
    if m:
        return f"t{m.group(1)}"
    raise ValueError(f"Trimestre invalide: {value!r}. Attendu: t1, t2, t3 ou t4.")


def resolve_previous_quarter(year: int, quarter: str) -> Tuple[int, str]:
    """Deduire le trimestre precedent a partir du trimestre courant.

    Regles de resolution : T2 -> T1 meme annee, T3 -> T2 meme annee,
    T1 -> T3 annee N-1, T4 -> T3 meme annee.
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
    """Chercher dans *directory* un PDF correspondant a la banque/annee/trimestre.

    Ordre de priorite :
    1. Nom canonique strict : ``{BANK}_{YEAR}_{Q}.pdf`` (ex. BNC_2025_T2.pdf)
    2. Repli par regex pour les noms anciens.
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
    """Localiser les fichiers PDF du trimestre courant et precedent.

    Ordre de recherche par trimestre :
    1. ``inputs_root / BANK / YEAR / *.pdf`` (nouvelle arborescence Inputs/)
    2. ``legacy_data_root / BANK / *.pdf`` (ancienne arborescence data/)

    Returns:
        Tuple ``(previous_pdf, current_pdf)``.

    Raises:
        FileNotFoundError: Si l'un des deux fichiers est introuvable.
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
