"""Export des résultats de comparaison texte vers un classeur Excel analyste.

Filtre : uniquement les changements retenus par le pipeline canonique.
Cela correspond à:
- MAJEUR pertinent
- MODERE pertinent avec nouveauté métier explicite

Colonnes :
  A — Titre           : section ou sous-section du rapport
  B — Page            : numéro de page dans le rapport T2
  C — Phrase          : texte sémantique retenu
  D — Justification   : explication IA
  E — Nouvelle idée ? : Oui / Non
  F — Commentaires    : vide, réservé à l'analyste
"""

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

EXCEL_COLUMNS = [
    "Titre",
    "Page",
    "Impact",
    "Action",
    "Phrase",
    "Justification",
    "Nouvelle idée ?",
    "Commentaires",
]

_IMPACT_COLORS = {
    "MAJEUR": "FFCCCC",
    "MODERE": "FFE5CC",
}


def generate_text_comparison_excel(
    text_comparison: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path | bytes:
    """Génère le classeur Excel analyste à partir de text_comparison.json.

    Seuls les changements retenus par la politique métier finale sont inclus.

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
    ws_expert = wb.create_sheet("Expert")

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
        field_name = "expert_block_comparisons" if mode == "expert" else "block_comparisons"
        for section_comp in text_comparison.get("section_comparisons", []):
            section_key = section_comp.get("section_key", "")
            section_title = section_comp.get("section_title") or _SECTION_DISPLAY.get(section_key, section_key)
            for block_comp in section_comp.get(field_name, []):
                diff_type = block_comp.get("diff_type", "")
                if diff_type == "unchanged":
                    continue

                triage = block_comp.get("genai_triage") or {}
                if not triage.get("is_relevant", False):
                    continue

                phrase = (block_comp.get("semantic_text_t2") or "").strip()
                if not phrase and diff_type == "removed":
                    phrase = f"[SUPPRIMÉ] {(block_comp.get('semantic_text_t1') or '').strip()}"

                evidence_t2 = block_comp.get("evidence_t2") or {}
                page = ", ".join(str(p) for p in (evidence_t2.get("pages") or []) if p)
                if not page and diff_type == "removed":
                    evidence_t1 = block_comp.get("evidence_t1") or {}
                    page = ", ".join(str(p) for p in (evidence_t1.get("pages") or []) if p)

                explanation = (triage.get("explanation") or "").strip()
                impact_desc = (triage.get("impact_description") or "").strip()
                if impact_desc and impact_desc not in explanation:
                    justification = f"{explanation}\n\n{impact_desc}"
                else:
                    justification = explanation

                rows.append(
                    {
                        "section_title": section_title,
                        "page": page if page else "",
                        "impact": (triage.get("impact_level") or "MINEUR").upper(),
                        "action": (triage.get("action_requise") or "aucune").lower(),
                        "phrase": phrase,
                        "justification": justification,
                        "nouvelle_idee": "Oui" if triage.get("nouvelle_idee", False) else "Non",
                    }
                )
        return rows

    def _write_rows(sheet, rows: list[dict[str, Any]]) -> int:
        row_num = 2
        for row in rows:
            row_data = [
                row["section_title"],
                row["page"],
                row["impact"],
                row["action"],
                row["phrase"],
                row["justification"],
                row["nouvelle_idee"],
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
            4: 16,
            5: 80,
            6: 60,
            7: 14,
            8: 30,
        }
        for col_idx, width in col_widths.items():
            sheet.column_dimensions[get_column_letter(col_idx)].width = width
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:H{max(row_num - 1, 1)}"
        return row_num - 2

    _setup_sheet(ws)
    _setup_sheet(ws_expert)
    strict_rows = _collect_rows("strict")
    expert_rows = _collect_rows("expert")
    strict_count = _write_rows(ws, strict_rows)
    expert_count = _write_rows(ws_expert, expert_rows)

    # ---------- save ----------
    if output_path is None:
        buf = io.BytesIO()
        wb.save(buf)
        logger.info(
            "text_comparison_excel: %d changements synthese, %d changements expert → BytesIO",
            strict_count,
            expert_count,
        )
        return buf.getvalue()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))
    logger.info(
        "text_comparison_excel: %d changements synthese, %d changements expert → %s",
        strict_count,
        expert_count,
        out,
    )
    return out
