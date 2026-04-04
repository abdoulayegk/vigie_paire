"""Logique d'exclusion de pages basee sur des patterns regex configures en YAML."""

from __future__ import annotations

import re

from vigilance.extract.pdf_text import extract_page_text

import pdfplumber


def compile_exclusion_patterns(cfg: dict) -> list[re.Pattern]:
    """Compile les patterns regex depuis le bloc de configuration des exclusions.

    Args:
        cfg: Dictionnaire de configuration bancaire contenant
            ``cfg["exclusions"]["block_title_patterns"]``.

    Returns:
        Patterns compiles (insensibles a la casse).
    """
    raw_patterns: list[str] = cfg["exclusions"]["block_title_patterns"]
    return [re.compile(p, re.IGNORECASE) for p in raw_patterns]


def get_skipped_pages(
    pdf_path: str,
    cfg: dict,
    max_pages: int | None = None,
) -> list[int]:
    """Retourne la liste triee des indices de pages (0-indexe) a ignorer.

    Une page est ignoree lorsque le nombre de patterns d'exclusion distincts
    qui correspondent a son texte atteint ou depasse le seuil
    ``min_hits_to_skip`` configure.

    Args:
        pdf_path: Chemin du fichier PDF.
        cfg: Dictionnaire de configuration bancaire avec bloc ``exclusions``.
        max_pages: Si fourni, n'inspecte que les *max_pages* premieres pages.

    Returns:
        Indices de pages 0-indexes a ignorer, tries.
    """
    patterns = compile_exclusion_patterns(cfg)
    min_hits = cfg["exclusions"]["page_skip_rules"]["min_hits_to_skip"]

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))

    skipped: list[int] = []
    for page_idx in range(total):
        text = extract_page_text(pdf_path, page_idx)
        hits = sum(1 for p in patterns if p.search(text))
        if hits >= min_hits:
            skipped.append(page_idx)

    return sorted(skipped)
