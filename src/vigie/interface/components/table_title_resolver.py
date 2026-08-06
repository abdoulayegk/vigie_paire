"""UI only: titres d'affichage Dash. Logique PDF dans ``vigie.extraction.table_title_resolver``."""

from __future__ import annotations

import re
from typing import Any

_RAW_ID_PATTERN = re.compile(r"^tbl_p\d+_i\d+$", re.IGNORECASE)


def resolve_display_table_title(table: dict[str, Any]) -> str:
    """Résout et retourne un titre clair et lisible pour l'affichage dans l'interface Dash.

    Si le titre extrait est vide ou correspond à un ID brut (ex: tbl_p082_i01),
    génère un titre sémantique à partir des analyses GenAI ou de la section/indicateurs.
    """
    if not isinstance(table, dict):
        return "Tableau"

    title_candidates = [
        str(table.get("table_name", "") or "").strip(),
        str(table.get("table_title_raw", "") or "").strip(),
        str(table.get("title", "") or "").strip(),
    ]

    for title in title_candidates:
        if title and not _RAW_ID_PATTERN.match(title) and title != "Tableau":
            return title

    # Si le titre est absent ou est un ID brut (ex. tbl_p082_i01), extraire le sujet GenAI
    genai = table.get("genai_analysis")
    if isinstance(genai, dict) and genai:
        # Essayer d'extraire le sujet dans la justification ou l'explication
        just = str(genai.get("nouvelle_idee_justification", "") or "").strip()
        changement = str(genai.get("changement_constate", "") or "").strip()

        sujet_match = re.search(r"Sujet d[eé]tect[eé]\s*:\s*([^.\n]+)", just, re.IGNORECASE)
        if sujet_match:
            sujet = sujet_match.group(1).strip().rstrip(".").strip()
            return f"[Tableau : {sujet}]"

        changement_match = re.search(
            r"RBC (?:supprime|ajoute|modifie|a supprimé|a ajouté)\s+([^.\n]+)", changement, re.IGNORECASE
        )
        if changement_match:
            sujet = changement_match.group(1).strip().rstrip(".").strip()
            if len(sujet) > 45:
                sujet = sujet[:42].rstrip() + "..."
            return f"[Tableau : {sujet}]"

        if genai.get("resume_metier"):
            resume = str(genai.get("resume_metier", "")).strip()
            first_sentence = resume.split(".")[0]
            if len(first_sentence) > 55:
                first_sentence = first_sentence[:52] + "..."
            return f"[Tableau : {first_sentence}]"

    section = str(table.get("section", "") or "").strip()
    if section:
        return f"[Tableau : {section}]"

    raw_id = (
        str(table.get("table_id_t2", "") or "").strip()
        or str(table.get("table_id_t1", "") or "").strip()
        or str(table.get("table_name", "") or "").strip()
    )
    return raw_id or "Tableau"
