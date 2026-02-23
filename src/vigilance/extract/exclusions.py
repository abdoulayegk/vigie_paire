"""Page exclusion logic based on YAML-configured regex patterns."""

from __future__ import annotations

import re

from vigilance.extract.pdf_text import extract_page_text

import pdfplumber


def compile_exclusion_patterns(cfg: dict) -> list[re.Pattern]:
    """Compile regex patterns from the exclusions config block.

    Parameters
    ----------
    cfg : dict
        Bank configuration dict that must contain
        ``cfg["exclusions"]["block_title_patterns"]``.

    Returns
    -------
    list[re.Pattern]
        Compiled patterns (case-insensitive).
    """
    raw_patterns: list[str] = cfg["exclusions"]["block_title_patterns"]
    return [re.compile(p, re.IGNORECASE) for p in raw_patterns]


def get_skipped_pages(
    pdf_path: str,
    cfg: dict,
    max_pages: int | None = None,
) -> list[int]:
    """Return sorted list of 0-based page indices that should be skipped.

    A page is skipped when the number of distinct exclusion patterns
    that match its text meets or exceeds the configured
    ``min_hits_to_skip`` threshold.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    cfg : dict
        Bank configuration dict with ``exclusions`` block.
    max_pages : int | None
        If given, only inspect the first *max_pages* pages.

    Returns
    -------
    list[int]
        Sorted 0-based page indices to skip.
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
