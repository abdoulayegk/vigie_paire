import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)


def find_text_bboxes_in_region(
    pdf_path: str,
    page_number: int,
    text_to_find: str,
    region_bbox_norm: list[float],
) -> list[list[float]]:
    """
    Search for a specific text within a constrained region of a PDF page
    and return the normalized bounding boxes of the matches.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        text_to_find: The exact string to locate (or part of it).
        region_bbox_norm: Normalized [l, t, r, b] of the table/region to restrict the search.

    Returns:
        List of normalized bounding boxes [l, t, r, b] where the text was found.
        Returns an empty list if not found.
    """
    if fitz is None:
        logger.warning(
            "PyMuPDF (fitz) is not installed. Text search for highlights will be disabled."
        )
        return []

    if not text_to_find or not text_to_find.strip():
        return []

    # Clean up the search string (remove newlines, extra spaces, footnote markers if desired)
    # PyMuPDF's search_for handles basic spaces, but exact matches on long multi-line
    # strings might break. So we take the first chunk if it's too long.
    search_text = text_to_find.strip().split("\n")[0]
    # Restrict length to improve match rate on split lines
    if len(search_text) > 40:
        search_text = search_text[:40].strip()

    try:
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            return []

        page = doc[page_number - 1]
        rect = page.rect

        # Convert normalized region bbox to absolute coordinates
        rx0 = rect.x0 + region_bbox_norm[0] * rect.width
        ry0 = rect.y0 + region_bbox_norm[1] * rect.height
        rx1 = rect.x0 + region_bbox_norm[2] * rect.width
        ry1 = rect.y0 + region_bbox_norm[3] * rect.height
        clip_rect = fitz.Rect(rx0, ry0, rx1, ry1)

        # Search for the text strictly inside the region
        matches = page.search_for(search_text, clip=clip_rect)

        # Convert absolute matches back to normalized coords
        normalized_matches = []
        for match_rect in matches:
            nl = (match_rect.x0 - rect.x0) / rect.width
            nt = (match_rect.y0 - rect.y0) / rect.height
            nr = (match_rect.x1 - rect.x0) / rect.width
            nb = (match_rect.y1 - rect.y0) / rect.height
            normalized_matches.append(
                [max(0.0, nl), max(0.0, nt), min(1.0, nr), min(1.0, nb)]
            )

        return normalized_matches

    except Exception as e:
        logger.error(f"Error searching text in PDF: {e}")
        return []
