"""Export des résultats de comparaison texte vers un classeur Excel analyste."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SECTION_DISPLAY: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_IMPACT_SORT_ORDER: dict[str, int] = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}

EXCEL_COLUMNS = [
    "Titre",
    "Page T1",
    "Page T2",
    "Texte exact T1",
    "Texte exact T2",
    "Type de changement",
    "Nouvelle idée ?",
    "Justification IA",
    "Commentaire analyste",
    "Décision analyste",
]


def _is_explicitly_cosmetic(block_comp: dict[str, Any]) -> bool:
    triage = block_comp.get("genai_triage") or {}
    return str(triage.get("category") or "").upper() == "COSMETIQUE"


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str, str]:
    return (
        0 if row.get("nouvelle_idee_bool", False) else 1,
        _IMPACT_SORT_ORDER.get(str(row.get("impact_level") or "").upper(), 99),
        str(row.get("section_title") or ""),
        str(row.get("page_t2") or row.get("page_t1") or ""),
        str(row.get("diff_type") or ""),
    )


def generate_text_comparison_excel(
    text_comparison: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path | bytes:
    """Génère le classeur Excel analyste à partir de text_comparison.json.

    Args:
        text_comparison: Dictionnaire text_comparison.json complet.
        output_path: Chemin de sortie du fichier .xlsx. Si None, retourne les
            bytes du classeur (pour envoi HTTP / dcc.send_bytes).

    Returns:
        Path du fichier créé, ou bytes si output_path est None.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Synthese"
    ws_all = wb.create_sheet("Tous_les_changements")

    # ---------- styles ----------
    header_font = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )
    cell_alignment = Alignment(vertical="top", wrap_text=True)

    def _setup_sheet(sheet) -> None:
        for col_idx, col_name in enumerate(EXCEL_COLUMNS, 1):
            cell = sheet.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border

    def _collect_rows(mode: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        field_name = "all_block_comparisons"
        for section_comp in text_comparison.get("section_comparisons", []):
            section_key = section_comp.get("section_key", "")
            section_title = section_comp.get("section_title") or _SECTION_DISPLAY.get(section_key, section_key)
            for block_comp in section_comp.get(field_name, []):
                diff_type = block_comp.get("diff_type", "")
                if diff_type == "unchanged":
                    continue

                triage = block_comp.get("genai_triage") or {}
                if mode != "all" and _is_explicitly_cosmetic(block_comp):
                    continue

                evidence_t2 = block_comp.get("evidence_t2") or {}
                evidence_t1 = block_comp.get("evidence_t1") or {}
                page_t1 = ", ".join(str(p) for p in (evidence_t1.get("pages") or []) if p)
                page_t2 = ", ".join(str(p) for p in (evidence_t2.get("pages") or []) if p)

                explanation = (triage.get("explanation") or "").strip()
                impact_desc = (triage.get("impact_description") or "").strip()
                header = (
                    f"Priorité: {(triage.get('impact_level') or 'MINEUR').upper()} | "
                    f"Action suggérée: {(triage.get('action_requise') or 'aucune').lower()}"
                )
                if impact_desc and impact_desc not in explanation:
                    justification = f"{header}\n{explanation}\n\n{impact_desc}".strip()
                else:
                    justification = f"{header}\n{explanation}".strip()

                rows.append(
                    {
                        "section_title": section_title,
                        "page_t1": page_t1,
                        "page_t2": page_t2,
                        "source_text_t1": (block_comp.get("source_text_t1") or "").strip(),
                        "source_text_t2": (block_comp.get("source_text_t2") or "").strip(),
                        "diff_type": diff_type,
                        "impact_level": (triage.get("impact_level") or "MINEUR").upper(),
                        "nouvelle_idee_bool": bool(triage.get("nouvelle_idee", False)),
                        "nouvelle_idee": "Oui" if triage.get("nouvelle_idee", False) else "Non",
                        "justification": justification,
                    }
                )
        rows.sort(key=_row_sort_key)
        return rows

    def _write_rows(sheet, rows: list[dict[str, Any]]) -> int:
        row_num = 2
        for row in rows:
            row_data = [
                row["section_title"],
                row["page_t1"],
                row["page_t2"],
                row["source_text_t1"],
                row["source_text_t2"],
                row["diff_type"],
                row["nouvelle_idee"],
                row["justification"],
                "",
                "",
            ]
            for col_idx, value in enumerate(row_data, 1):
                cell = sheet.cell(row=row_num, column=col_idx, value=value)
                cell.alignment = cell_alignment
                cell.border = thin_border
            row_num += 1

        col_widths = {
            1: 30,
            2: 10,
            3: 10,
            4: 70,
            5: 70,
            6: 18,
            7: 14,
            8: 65,
            9: 22,
            10: 22,
        }
        for col_idx, width in col_widths.items():
            sheet.column_dimensions[get_column_letter(col_idx)].width = width
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:J{max(row_num - 1, 1)}"
        return row_num - 2

    _setup_sheet(ws)
    _setup_sheet(ws_all)
    strict_rows = _collect_rows("strict")
    all_rows = _collect_rows("all")
    strict_count = _write_rows(ws, strict_rows)
    all_count = _write_rows(ws_all, all_rows)

    # ---------- save ----------
    if output_path is None:
        buf = io.BytesIO()
        wb.save(buf)
        logger.info(
            "text_comparison_excel: %d changements synthese, %d changements complets → BytesIO",
            strict_count,
            all_count,
        )
        return buf.getvalue()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))
    logger.info(
        "text_comparison_excel: %d changements synthese, %d changements complets → %s",
        strict_count,
        all_count,
        out,
    )
    return out
