"""Export helpers for review items. Excel-ready CSV for Excel FR."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Iterator

from app.review_models import ReviewItem

CSV_SCHEMA_VERSION = "csv_review_v1"
CSV_SEPARATOR = ";"
CSV_BOM = "\ufeff"

# Schema validation superviseur (12 colonnes, ordre fixe)
VALIDATION_CSV_COLUMNS = [
    "banque",
    "section",
    "type_élément",
    "type_changement",
    "page_T1",
    "page_T2",
    "indicateur_T1",
    "indicateur_T2",
    "résumé_automatique",
    "score_confiance",
    "suspect",
    "validation_finale",
    "pertinence_genai",
    "niveau_risque_genai",
]

# Mapping type_changement anglais -> francais
_TYPE_CHANGEMENT_MAP = {
    "added": "ajouté",
    "removed": "retiré",
    "renamed": "renommé",
    "table_added": "ajouté",
    "table_removed": "retiré",
    "modified": "modifié",
    "modifie": "modifié",
    "displaced": "déplacé",
    "uncertain": "incertain",
    "needs_review": "incertain",
    "structure_change": "fusion",
    "footnote": "note modifiee",
}

# Match methods consideres suspects
_SUSPECT_MATCH_METHODS = frozenset({
    "split_merge_rescue",
    "single_rescue",
    "uncertain_competition",
    "unknown_section_penalized",
    "low_label_overlap_reject",
})

# Mapping section (code interne) -> libellé français
_SECTION_FR = {
    "gestion_capital": "Gestion du capital",
    "capital_management": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "risk_management": "Gestion des risques",
    "gestion_reglementation": "Réglementation",
    "regulatory_updates": "Réglementation",
    "reglementation": "Réglementation",
}

# Mapping review_status -> validation_finale
_VALIDATION_FINALE_MAP = {
    "pending": "En attente",
    "approved": "Validé",
    "rejected": "Rejeté",
}

_CSV_COLUMNS = [
    "run_id",
    "date_generation",
    "version_schema",
    "bank_code",
    "quarter_from",
    "quarter_to",
    "year",
    "change_id",
    "section",
    "table_name",
    "table_id_t1",
    "table_id_t2",
    "page_t1",
    "page_t2",
    "table_status",
    "indicator_name",
    "indicator_type",
    "indicator_from",
    "indicator_to",
    "resume_court",
    "confidence",
    "match_method",
    "review_status",
    "idee",
    "sujet",
    "validation_finale",
    "comment",
]


def _sanitize_cell(val: Any) -> str:
    """Nettoie retours ligne pour CSV Excel. Les guillemets/delimiteurs sont geres par csv.DictWriter."""
    if val is None:
        return ""
    s = str(val).strip()
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return s


def _format_cell(val: Any) -> str:
    """Formate une valeur pour CSV (None, int, float). Pages en entiers si possible."""
    if val is None:
        return ""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return str(int(val)) if val == int(val) else str(val)
    return _sanitize_cell(val)


def _build_resume_court(
    indicator_type: str,
    indicator_name: str,
    from_val: str,
    to_val: str,
    table_status: str = "",
    suspect: bool = False,
) -> str:
    """Genere une phrase interpretative courte pour le superviseur (FR, 1 phrase max)."""
    if indicator_type == "added":
        base = "Nouvel indicateur détecté."
    elif indicator_type == "removed":
        base = "Indicateur présent en T1 mais absent en T2."
    elif indicator_type == "renamed" and (from_val or to_val):
        base = "Renommage probable d'indicateur."
    elif indicator_type == "renamed":
        base = "Renommage probable d'indicateur."
    elif indicator_type == "table_added":
        base = "Nouveau tableau détecté."
    elif indicator_type == "table_removed":
        base = "Tableau présent en T1 mais absent en T2."
    elif indicator_type == "modified":
        base = "Modification de tableau détectée."
    elif indicator_type == "uncertain":
        base = "Aucun match clair trouvé — revue manuelle requise."
    elif indicator_type == "structure_change":
        base = "Fusion ou split probable — revue manuelle requise."
    elif indicator_type == "footnote":
        base = "Note de bas de tableau modifiee -- verifier impact reglementaire."
    elif "note" in indicator_type.lower() or table_status == "note":
        base = "Texte de note modifié — vérifier impact réglementaire."
    else:
        base = _sanitize_cell(indicator_name) or "Changement détecté."
    if suspect:
        base = f"{base} Revue manuelle recommandée."
    return base


def _to_type_changement(ind_type: str, table_status: str) -> str:
    """Convertit indicator_type + table_status en type_changement francais."""
    if table_status == "structure_change":
        return "fusion"
    return _TYPE_CHANGEMENT_MAP.get(ind_type, "incertain")


def _to_type_element(ind_type: str, item_type: str = "indicator") -> str:
    """Retourne type_élément: tableau, indicateur, note ou texte."""
    if item_type == "footnote" or ind_type == "footnote":
        return "note"
    if ind_type in ("table_added", "table_removed", "modified", "uncertain", "structure_change"):
        return "tableau"
    if ind_type in ("note", "texte"):
        return ind_type
    return "indicateur"


def _to_suspect(match_method: str) -> str:
    """Convertit match_method en suspect Oui/Non."""
    if not match_method:
        return "Non"
    normalized = str(match_method).strip().lower()
    for suspect in _SUSPECT_MATCH_METHODS:
        if suspect.lower() in normalized:
            return "Oui"
    return "Non"


def _section_to_fr(section: str) -> str:
    """Convertit le code section en libellé français pour le CSV."""
    if not section:
        return ""
    key = str(section).strip().lower()
    return _SECTION_FR.get(key, section)


def _to_validation_finale(review_status: str) -> str:
    """Convertit review_status en validation_finale."""
    return _VALIDATION_FINALE_MAP.get(
        str(review_status or "").strip().lower(),
        "En attente",
    )


def _iter_validation_rows(
    review_items: list[ReviewItem],
    indicator_result: dict[str, Any] | None,
) -> Iterator[dict[str, str]]:
    """Itere sur les lignes du schema validation (12 colonnes) pour CSV ou Excel."""
    ir = indicator_result or {}
    banque = _sanitize_cell(ir.get("bank_code", ""))

    for item in review_items:
        base = item.to_dict()
        indicators = base.get("indicators", [])
        table_status = str(base.get("table_status", ""))
        confidence_raw = base.get("confidence")
        if confidence_raw is not None:
            try:
                score = round(float(confidence_raw), 2)
            except (TypeError, ValueError):
                score = 0.0
        else:
            score = 0.0
        score_str = f"{score:.2f}"
        match_method = str(base.get("match_method", ""))
        suspect = _to_suspect(match_method)
        validation = _to_validation_finale(base.get("review_status", ""))

        item_type = str(base.get("item_type", "indicator"))
        ga = base.get("genai_analysis") or {}
        pertinence_genai = _sanitize_cell(ga.get("relevance", ""))
        niveau_risque_genai = _sanitize_cell(ga.get("risk_level", ""))

        if not indicators:
            ind_type = str(base.get("change_type", ""))
            ind_name = _sanitize_cell(base.get("indicator", ""))
            type_elem = _to_type_element(ind_type, item_type)
            type_chg = _to_type_changement(ind_type, table_status)
            resume = _build_resume_court(
                ind_type, ind_name, "", "", table_status, suspect=(suspect == "Oui")
            )

            if ind_type in ("added", "table_added"):
                ind_t1, ind_t2 = "", ind_name
            elif ind_type in ("removed", "table_removed"):
                ind_t1, ind_t2 = ind_name, ""
            else:
                ind_t1, ind_t2 = ind_name, ind_name

            row = {
                "banque": banque,
                "section": _section_to_fr(base.get("section", "")),
                "type_élément": type_elem,
                "type_changement": type_chg,
                "page_T1": _format_cell(base.get("page_t1")),
                "page_T2": _format_cell(base.get("page_t2")),
                "indicateur_T1": ind_t1,
                "indicateur_T2": ind_t2,
                "résumé_automatique": resume,
                "score_confiance": score_str,
                "suspect": suspect,
                "validation_finale": validation,
                "pertinence_genai": pertinence_genai,
                "niveau_risque_genai": niveau_risque_genai,
            }
            yield row
        else:
            item_change_type = str(base.get("change_type", ""))
            for ind in indicators:
                ind_type = str(ind.get("type", ""))
                ind_name = str(ind.get("name", ""))
                from_val = str(ind.get("from", ""))
                to_val = str(ind.get("to", ""))

                type_elem = _to_type_element(item_change_type or ind_type, item_type)
                type_chg = _to_type_changement(ind_type, table_status)
                resume_type = item_change_type if item_change_type in ("table_added", "table_removed") else ind_type
                resume = _build_resume_court(
                    resume_type, ind_name, from_val, to_val, table_status, suspect=(suspect == "Oui")
                )

                if ind_type == "added":
                    ind_t1, ind_t2 = "", _sanitize_cell(ind_name)
                elif ind_type == "removed":
                    ind_t1, ind_t2 = _sanitize_cell(ind_name), ""
                elif ind_type == "renamed":
                    ind_t1 = _sanitize_cell(from_val) if from_val else _sanitize_cell(ind_name)
                    ind_t2 = _sanitize_cell(to_val) if to_val else _sanitize_cell(ind_name)
                else:
                    ind_t1, ind_t2 = _sanitize_cell(ind_name), _sanitize_cell(ind_name)

                row = {
                    "banque": banque,
                    "section": _section_to_fr(base.get("section", "")),
                    "type_élément": type_elem,
                    "type_changement": type_chg,
                    "page_T1": _format_cell(base.get("page_t1")),
                    "page_T2": _format_cell(base.get("page_t2")),
                    "indicateur_T1": ind_t1,
                    "indicateur_T2": ind_t2,
                    "résumé_automatique": resume,
                    "score_confiance": score_str,
                    "suspect": suspect,
                    "validation_finale": validation,
                    "pertinence_genai": pertinence_genai,
                    "niveau_risque_genai": niveau_risque_genai,
                }
                yield row


def generate_validation_csv(
    review_items: list[ReviewItem],
    indicator_result: dict[str, Any] | None,
) -> str:
    """Genere le CSV de validation unique (12 colonnes, Excel FR, conformite bancaire).

    Schema exact:
    banque | section | type_élément | type_changement | page_T1 | page_T2 |
    indicateur_T1 | indicateur_T2 | résumé_automatique |
    score_confiance | suspect | validation_finale
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=VALIDATION_CSV_COLUMNS,
        delimiter=CSV_SEPARATOR,
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for row in _iter_validation_rows(review_items, indicator_result):
        writer.writerow(row)
    return CSV_BOM + buffer.getvalue()


def generate_validation_excel(
    review_items: list[ReviewItem],
    indicator_result: dict[str, Any] | None,
) -> bytes:
    """Genere le fichier Excel (.xlsx) de validation (meme schema que le CSV)."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    if ws is None:
        raise RuntimeError("openpyxl: no active sheet")
    ws.title = "Revue"
    ws.append(list(VALIDATION_CSV_COLUMNS))
    for row in _iter_validation_rows(review_items, indicator_result):
        ws.append([row[col] for col in VALIDATION_CSV_COLUMNS])
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def export_review_items_csv(
    items: list[ReviewItem],
    *,
    metadata: dict[str, Any] | None = None,
    separator: str = CSV_SEPARATOR,
) -> str:
    """Serialize review items to CSV text (schema technique, colonnes en anglais).

    DEPRECATED: Pour l'export superviseur en français, utiliser generate_validation_csv()
    qui produit le schema 12 colonnes (banque, section, type_élément, etc.).
    """
    now = datetime.now()
    meta = metadata or {}
    run_id = str(meta.get("run_id", "") or now.strftime("%Y%m%d_%H%M"))
    date_gen = meta.get("date_generation", "") or now.isoformat(timespec="seconds")
    bank = _sanitize_cell(meta.get("bank_code", ""))
    q_from = _sanitize_cell(meta.get("quarter_from", "t1"))
    q_to = _sanitize_cell(meta.get("quarter_to", "t2"))
    year_val = _format_cell(meta.get("year", ""))

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=_CSV_COLUMNS,
        delimiter=separator,
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()

    for item in items:
        base = item.to_dict()
        indicators = base.get("indicators", [])

        if not indicators:
            row = {
                "run_id": run_id,
                "date_generation": date_gen,
                "version_schema": CSV_SCHEMA_VERSION,
                "bank_code": bank,
                "quarter_from": q_from,
                "quarter_to": q_to,
                "year": year_val,
                "change_id": _sanitize_cell(base.get("change_id", "")),
                "section": _sanitize_cell(base.get("section", "")),
                "table_name": _sanitize_cell(base.get("table_name", "")),
                "table_id_t1": _sanitize_cell(base.get("table_id_t1", "")),
                "table_id_t2": _sanitize_cell(base.get("table_id_t2", "")),
                "page_t1": _format_cell(base.get("page_t1")),
                "page_t2": _format_cell(base.get("page_t2")),
                "table_status": _sanitize_cell(base.get("table_status", "")),
                "indicator_name": _sanitize_cell(base.get("indicator", "")),
                "indicator_type": _sanitize_cell(base.get("change_type", "")),
                "indicator_from": "",
                "indicator_to": "",
                "resume_court": _build_resume_court(
                    base.get("change_type", ""),
                    base.get("indicator", ""),
                    "",
                    "",
                    base.get("table_status", ""),
                ),
                "confidence": _format_cell(base.get("confidence", "")),
                "match_method": _sanitize_cell(base.get("match_method", "")),
                "review_status": _sanitize_cell(base.get("review_status", "")),
                "idee": "",
                "sujet": "",
                "validation_finale": "",
                "comment": _sanitize_cell(base.get("comment", "")),
            }
            writer.writerow(row)
        else:
            for ind in indicators:
                ind_type = str(ind.get("type", ""))
                ind_name = str(ind.get("name", ""))
                from_val = str(ind.get("from", ""))
                to_val = str(ind.get("to", ""))

                row = {
                    "run_id": run_id,
                    "date_generation": date_gen,
                    "version_schema": CSV_SCHEMA_VERSION,
                    "bank_code": bank,
                    "quarter_from": q_from,
                    "quarter_to": q_to,
                    "year": year_val,
                    "change_id": _sanitize_cell(base.get("change_id", "")),
                    "section": _sanitize_cell(base.get("section", "")),
                    "table_name": _sanitize_cell(base.get("table_name", "")),
                    "table_id_t1": _sanitize_cell(base.get("table_id_t1", "")),
                    "table_id_t2": _sanitize_cell(base.get("table_id_t2", "")),
                    "page_t1": _format_cell(base.get("page_t1")),
                    "page_t2": _format_cell(base.get("page_t2")),
                    "table_status": _sanitize_cell(base.get("table_status", "")),
                    "indicator_name": _sanitize_cell(ind_name),
                    "indicator_type": _sanitize_cell(ind_type),
                    "indicator_from": _sanitize_cell(from_val),
                    "indicator_to": _sanitize_cell(to_val),
                    "resume_court": _build_resume_court(
                        ind_type, ind_name, from_val, to_val,
                        base.get("table_status", ""),
                    ),
                    "confidence": _format_cell(base.get("confidence", "")),
                    "match_method": _sanitize_cell(base.get("match_method", "")),
                    "review_status": _sanitize_cell(base.get("review_status", "")),
                    "idee": "",
                    "sujet": "",
                    "validation_finale": "",
                    "comment": _sanitize_cell(base.get("comment", "")),
                }
                writer.writerow(row)

    return CSV_BOM + buffer.getvalue()


def export_review_items_json_fr(
    items: list[ReviewItem],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Serialize review items to JSON (French-oriented schema)."""
    payload = {
        "metadata": metadata or {},
        "total": len(items),
        "items": [item.to_dict() for item in items],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
