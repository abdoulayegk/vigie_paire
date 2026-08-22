"""Localisation structurelle des tableaux via Docling (comportement historique)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from vigie.extraction.docling_bbox_helpers import _build_indicator_reference_text
from vigie.extraction.table_locator.models import TableAnchor, TableLocationResult

logger = logging.getLogger("vigie.extraction.docling.table_locator")

SOURCE = "docling"


class DoclingTableLocator:
    """Detecter les tableaux avec Docling et produire des ``TableAnchor``."""

    def __init__(self, converter: Any) -> None:
        """Initialiser avec le ``DocumentConverter`` Docling deja configure.

        Args:
            converter: Convertisseur Docling pret a l'emploi.
        """
        self._converter = converter

    def locate(
        self,
        pdf_path: Path,
        page_ranges: list[tuple[int, int]] | None = None,
        *,
        reference_text_max_chars: int = 6000,
        is_page_in_ranges: Callable[[int, list[tuple[int, int]] | None], bool] | None = None,
        docling_page_range: tuple[int, int] | None = None,
    ) -> TableLocationResult:
        """Convertir le PDF Docling et extraire les ancres de tableaux.

        Args:
            pdf_path: Chemin du PDF.
            page_ranges: Plages 1-indexees a conserver (filtrage applicatif).
            reference_text_max_chars: Plafond du texte de reference Vision.
            is_page_in_ranges: Predicat de filtrage ; defaut = inclusion simple.
            docling_page_range: Plage unique passee a ``converter.convert``.

        Returns:
            Ancres + markdown Docling + pages pour inventaire Vision.
        """
        convert_kwargs: dict[str, Any] = {}
        if docling_page_range is not None:
            convert_kwargs["page_range"] = docling_page_range

        result = self._converter.convert(str(pdf_path), **convert_kwargs)
        doc = result.document

        page_filter = is_page_in_ranges or _default_page_in_ranges
        anchors: list[TableAnchor] = []
        for idx, table in enumerate(doc.tables):
            page_num = table.prov[0].page_no if table.prov else 0
            table_bbox: list[float] | None = None
            try:
                if table.prov and hasattr(table.prov[0], "bbox") and table.prov[0].bbox is not None:
                    raw_bbox = table.prov[0].bbox
                    page_obj = doc.pages.get(page_num) if hasattr(doc, "pages") else None
                    if page_obj and hasattr(page_obj, "size") and page_obj.size:
                        norm = raw_bbox.to_top_left_origin(page_height=page_obj.size.height)
                        norm = norm.normalized(page_obj.size)
                        table_bbox = [norm.l, norm.t, norm.r, norm.b]
                    elif hasattr(raw_bbox, "as_tuple"):
                        table_bbox = list(raw_bbox.as_tuple())
            except Exception:
                table_bbox = None
            if not page_filter(page_num, page_ranges):
                continue
            table_id = f"tableau_{idx}"
            reference_text: str | None = None
            try:
                if hasattr(table, "text") and table.text:
                    _ref_raw = str(table.text).strip()
                    if len(_ref_raw) > 20 and reference_text_max_chars > 0:
                        reference_text = _build_indicator_reference_text(
                            _ref_raw,
                            max_chars=reference_text_max_chars,
                        )
            except Exception:
                pass
            anchors.append(
                TableAnchor(
                    table_id=table_id,
                    page_number=int(page_num),
                    bbox=table_bbox,
                    reference_text=reference_text,
                    source=SOURCE,
                )
            )

        inventory_pages: list[int] = []
        if hasattr(doc, "pages"):
            try:
                inventory_pages = sorted(int(page_number) for page_number in doc.pages.keys())
            except Exception:
                inventory_pages = []

        text_content = ""
        try:
            text_content = doc.export_to_markdown()
        except Exception as exc:
            logger.warning("Docling export_to_markdown failed (non-fatal): %s", exc)

        total_pages = len(doc.pages) if hasattr(doc, "pages") else 0
        logger.info(
            "Docling table locator: %d table(s) kept (inventory_pages=%d)",
            len(anchors),
            len(inventory_pages),
        )
        return TableLocationResult(
            anchors=anchors,
            text_content=text_content,
            total_pages=total_pages,
            inventory_pages=inventory_pages,
        )


def _default_page_in_ranges(page_num: int, page_ranges: list[tuple[int, int]] | None) -> bool:
    """Inclure la page si aucune plage n'est fournie, sinon tester l'appartenance."""
    if page_ranges is None:
        return True
    return any(start <= page_num <= end for start, end in page_ranges)
