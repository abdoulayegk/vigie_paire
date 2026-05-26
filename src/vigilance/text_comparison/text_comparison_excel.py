"""Export des résultats de comparaison texte vers un classeur Excel analyste."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

from vigilance.text_comparison.justification import build_text_triage_justification

logger = logging.getLogger(__name__)

_SECTION_DISPLAY: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_IMPACT_SORT_ORDER: dict[str, int] = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}

_CATEGORY_SORT_ORDER: dict[str, int] = {
    "REGLEMENTAIRE": 0,
    "RISQUE": 1,
    "CAPITAL": 2,
    "STRUCTURE": 3,
    "NON_PERTINENT": 4,
    "INCONNU": 5,
}

EXCEL_COLUMNS = [
    "Titre",
    "Sous-section",
    "Page T1",
    "Page T2",
    "Type de changement",
    "Texte exact T1",
    "Texte exact T2",
    "Nouvelle idée ?",
    "Justification IA",
    "Commentaire analyste",
]

_DIFF_TYPE_FR: dict[str, str] = {
    "added": "Ajout",
    "removed": "Suppression",
    "modified": "Modification",
    "renamed": "Renommage",
    "unchanged": "Inchangé",
}

# ---------------------------------------------------------------------------
# Heuristiques d'exclusion (dates pures et reformulations strictes)
# ---------------------------------------------------------------------------

_DATE_UPDATE_RE = re.compile(
    r"(mise à jour|changement|modification|passé).{0,40}(date|période|trimestre|semestre|référence de clôture)|"
    r"(date|période de référence).{0,20}(mise à jour|changé|modifié|passé)|"
    r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b"
    r".{0,25}→.{0,25}"
    r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b|"
    r"^(la )?date (de référence|de clôture|des échéances) (a été|est) (mis|modif|chang)",
    flags=re.IGNORECASE,
)

_REFORMULATION_RE = re.compile(
    r"\b(légère|simple|même|pure).{0,20}(reformulation|reformulé|reformulée)\b|"
    r"\breformulation.{0,20}(légère|sans changement|sans nouveau fond)\b|"
    r"\bmême (idée|information|sens|contenu).{0,30}(reformul|formulat différente)\b",
    flags=re.IGNORECASE,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _excel_safe(value: Any) -> Any:
    """Nettoie les chaînes avant écriture dans openpyxl."""
    if isinstance(value, str):
        return _CONTROL_CHAR_RE.sub("", value)
    return value


def _is_pure_date_update(change: dict[str, Any]) -> bool:
    """Vrai uniquement pour les mises à jour de dates sans autre contenu."""
    triage = change.get("genai_triage") or {}
    if str(triage.get("impact_level") or "").upper() != "MINEUR":
        return False
    if str(triage.get("category") or "").upper() != "NON_PERTINENT":
        return False
    summary = change.get("change_summary") or ""
    return bool(_DATE_UPDATE_RE.search(summary))


def _is_pure_reformulation(change: dict[str, Any]) -> bool:
    """Vrai uniquement pour les reformulations strictement identiques."""
    triage = change.get("genai_triage") or {}
    if str(triage.get("impact_level") or "").upper() != "MINEUR":
        return False
    if str(triage.get("category") or "").upper() != "NON_PERTINENT":
        return False
    summary = change.get("change_summary") or ""
    return bool(_REFORMULATION_RE.search(summary))


def _should_exclude(change: dict[str, Any]) -> bool:
    """Indique si le changement doit être exclu de l'export (date ou reformulation)."""
    return _is_pure_date_update(change) or _is_pure_reformulation(change)


# ---------------------------------------------------------------------------
# Extraction de la sous-section
# ---------------------------------------------------------------------------

def _subsection_label(change: dict[str, Any]) -> str:
    """Retourne le nom lisible de la sous-section."""
    heading = change.get("subsection_heading") or ""
    if heading and heading not in ("__intro__", "full", ""):
        return heading
    # Fallback: extraire du change_id
    change_id = change.get("change_id") or ""
    section_key = change.get("section_key") or ""
    prefix = section_key + "_"
    if change_id.startswith(prefix):
        slug = re.sub(r"_change_\d+$", "", change_id[len(prefix):])
        if slug and slug != "full":
            return slug.replace("_", " ").strip()
    return ""


# ---------------------------------------------------------------------------
# Tri
# ---------------------------------------------------------------------------

def _row_sort_key(row: dict[str, Any]) -> tuple:
    """Cle de tri analyste : pertinence -> impact -> nouvelle idee."""
    return (
        0 if row.get("is_relevant") else 1,
        _IMPACT_SORT_ORDER.get(str(row.get("impact_level") or "").upper(), 99),
        0 if row.get("nouvelle_idee_bool") else 1,
        _CATEGORY_SORT_ORDER.get(str(row.get("category") or "").upper(), 5),
        str(row.get("section_title") or ""),
        str(row.get("diff_type") or ""),
    )


# ---------------------------------------------------------------------------
# Couleurs par niveau d'impact
# ---------------------------------------------------------------------------

_FILL_NOUVELLE_IDEE = "D6E4F0"   # bleu clair — nouvelle idée
_FILL_MAJEUR        = "FADADD"   # rouge clair
_FILL_MODERE        = "FDEBD0"   # orange clair
_FILL_MINEUR_REL    = "FEF9E7"   # jaune très pâle — MINEUR mais pertinent
_FILL_MINEUR        = "FFFFFF"   # blanc — non pertinent ou MINEUR standard


def _row_fill_color(row: dict[str, Any]) -> str | None:
    """Retourne la couleur de remplissage Excel selon nouvelle_idee et impact."""
    if row.get("nouvelle_idee_bool"):
        return _FILL_NOUVELLE_IDEE
    level = str(row.get("impact_level") or "").upper()
    if level == "MAJEUR":
        return _FILL_MAJEUR
    if level == "MODERE":
        return _FILL_MODERE
    if level == "MINEUR" and row.get("is_relevant"):
        return _FILL_MINEUR_REL
    return None


# ---------------------------------------------------------------------------
# Collecte des lignes
# ---------------------------------------------------------------------------

def _collect_rows(text_comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrait et aplatit toutes les lignes de changement depuis ``text_comparison.json``."""
    rows: list[dict[str, Any]] = []
    for section_comp in text_comparison.get("section_comparisons", []):
        section_key = section_comp.get("section_key", "")
        section_title = (
            section_comp.get("section_title")
            or _SECTION_DISPLAY.get(section_key, section_key)
        )
        for block_comp in section_comp.get("all_block_comparisons", []):
            if block_comp.get("diff_type") == "unchanged":
                continue
            if _should_exclude(block_comp):
                continue

            triage = block_comp.get("genai_triage") or {}
            evidence_t1 = block_comp.get("evidence_t1") or {}
            evidence_t2 = block_comp.get("evidence_t2") or {}
            page_t1 = ", ".join(str(p) for p in (evidence_t1.get("pages") or []) if p)
            page_t2 = ", ".join(str(p) for p in (evidence_t2.get("pages") or []) if p)

            justification = build_text_triage_justification(block_comp)
            analyst_review = block_comp.get("_analyst_review") or {}
            analyst_status = str(analyst_review.get("status") or "").strip().lower()
            ai_nouvelle_idee = bool(triage.get("nouvelle_idee", False))
            nouvelle_idee = "Oui" if ai_nouvelle_idee else "Non"
            analyst_comment = ""
            if analyst_status == "rejected":
                nouvelle_idee = "Non"
                analyst_comment = str(analyst_review.get("comment") or "").strip()
            elif analyst_status == "approved":
                analyst_comment = str(analyst_review.get("comment") or "").strip()

            rows.append(
                {
                    "change_id": block_comp.get("change_id", ""),
                    "section_key": section_key,
                    "section_title": section_title,
                    "subsection": _subsection_label(block_comp),
                    "page_t1": page_t1,
                    "page_t2": page_t2,
                    "source_text_t1": (block_comp.get("source_text_t1") or "").strip(),
                    "source_text_t2": (block_comp.get("source_text_t2") or "").strip(),
                    "diff_type": block_comp.get("diff_type", ""),
                    "impact_level": str(triage.get("impact_level") or "MINEUR").upper(),
                    "category": str(triage.get("category") or "INCONNU").upper(),
                    "is_relevant": bool(triage.get("is_relevant", False)),
                    "nouvelle_idee_bool": nouvelle_idee == "Oui",
                    "nouvelle_idee": nouvelle_idee,
                    "justification": justification,
                    "commentaire_analyste": analyst_comment,
                }
            )

    rows.sort(key=_row_sort_key)
    return rows


# ---------------------------------------------------------------------------
# Export principal
# ---------------------------------------------------------------------------

def generate_text_comparison_excel(
    text_comparison: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path | bytes:
    """Génère le classeur Excel analyste à partir de text_comparison.json.

    Affiche tous les changements détectés (hors unchanged, dates pures et
    reformulations strictes), triés par nouvelle idée puis par priorité.
    L'analyste valide chaque ligne via la colonne Statut.

    Args:
        text_comparison: Dictionnaire text_comparison.json complet.
        output_path: Chemin de sortie .xlsx. Si None, retourne les bytes.

    Returns:
        Path du fichier créé, ou bytes si output_path est None.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Analyse complète"

    # ---------- styles ----------
    header_font = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(style="thin")
    thin_border = Border(left=thin, right=thin, top=thin, bottom=thin)
    cell_align = Alignment(vertical="top", wrap_text=True)

    # ---------- en-têtes ----------
    for col_idx, col_name in enumerate(EXCEL_COLUMNS, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # ---------- données ----------
    rows = _collect_rows(text_comparison)

    for row_num, row in enumerate(rows, start=2):
        fill_hex = _row_fill_color(row)
        fill = PatternFill(start_color=fill_hex, end_color=fill_hex, fill_type="solid") if fill_hex else None

        row_data = [
            row["section_title"],
            row["subsection"],
            row["page_t1"],
            row["page_t2"],
            _DIFF_TYPE_FR.get(str(row["diff_type"]).lower(), row["diff_type"]),
            row["source_text_t1"],
            row["source_text_t2"],
            row["nouvelle_idee"],
            row["justification"],
            row["commentaire_analyste"],
        ]
        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_num, column=col_idx, value=_excel_safe(value))
            cell.alignment = cell_align
            cell.border = thin_border
            if fill:
                cell.fill = fill

    last_row = len(rows) + 1

    # ---------- largeurs de colonnes ----------
    col_widths = {
        1: 30,   # Titre
        2: 35,   # Sous-section
        3: 10,   # Page T1
        4: 10,   # Page T2
        5: 18,   # Type de changement
        6: 70,   # Texte exact T1
        7: 70,   # Texte exact T2
        8: 14,   # Nouvelle idée ?
        9: 70,   # Justification IA
        10: 25,  # Commentaire analyste
    }
    for col_idx, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(EXCEL_COLUMNS))}{max(last_row, 1)}"

    # ---------- sauvegarde ----------
    if output_path is None:
        buf = io.BytesIO()
        wb.save(buf)
        logger.info("text_comparison_excel: %d changements → BytesIO", len(rows))
        return buf.getvalue()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))
    logger.info("text_comparison_excel: %d changements → %s", len(rows), out)
    return out
