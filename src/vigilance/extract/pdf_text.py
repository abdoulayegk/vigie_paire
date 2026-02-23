"""Extract text from PDF pages using pdfplumber."""

from __future__ import annotations

import pdfplumber


def extract_page_text(pdf_path: str, page_index: int) -> str:
    """Extract the text content of a single PDF page.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_index : int
        0-based page index.

    Returns
    -------
    str
        Extracted text, or ``""`` if pdfplumber returns ``None``.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        text = page.extract_text()
        return text if text is not None else ""


def extract_pages_text(pdf_path: str, page_indices: list[int]) -> dict[int, str]:
    """Extract text from multiple PDF pages in a single open call.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_indices : list[int]
        0-based page indices to extract.

    Returns
    -------
    dict[int, str]
        Mapping of ``{page_index: extracted_text}``.
    """
    result: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for idx in page_indices:
            page = pdf.pages[idx]
            text = page.extract_text()
            result[idx] = text if text is not None else ""
    return result
