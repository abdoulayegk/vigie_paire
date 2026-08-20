"""Traductions francaises -- terminologie professionnelle bancaire."""

from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# A) UI Labels
# -----------------------------------------------------------------------------
UI_LABELS: dict[str, str] = {
    "app_title": "Comparateur de Rapports Bancaires",
    "upload": "Charger",
    "analyze": "Analyser",
    "btn_analyze": "Analyser",
    "btn_load": "Charger",
    "btn_refresh": "Actualiser",
    "btn_reset": "Nouvelle Analyse",
    "results": "Résultats",
    "filters": "Filtres",
    "section": "Section",
    "page": "Page",
    "table": "Tableau",
    "tables": "Tableaux",
    "indicators": "Indicateurs",
    "added": "Ajout",
    "removed": "Retrait",
    "renamed": "Renommage",
    "matched": "Apparié",
    "uncertain": "Appariement incertain",
    "probable": "Probable",
    "rescued": "Sauvegardé",
    "export": "Exporter",
    "load": "Charger",
    "refresh": "Actualiser",
    "page_t1": "Page précédente",
    "page_t2": "Page courante",
    "reason": "Raison",
    "score": "Score",
    "statut": "Statut",
    "review": "Revue",
    "all_sections": "Toutes les sections",
    "validation_time": "Temps de validation",
    "file_review": "File de revue",
    "file_review_total": "File de revue (total)",
    "validated": "Valides",
    "rejected": "Rejetés",
    "pending": "En attente",
    "table_added": "Tableau ajouté",
    "table_added_plural": "Tableaux ajoutés",
    "table_removed": "Tableau retiré",
    "table_removed_plural": "Tableaux retirés",
    "table_entire_added": "Tableau entier ajouté",
    "table_entire_removed": "Tableau entier supprimé",
    "table_no_prefix": "Tableau n°",
    "indicator_add": "Ajout",
    "indicator_removal": "Suppression",
    "indicator_rename": "Renommage",
    "fusion_split": "Fusion/scission",
    "notes_bas_tableau": "Notes de bas de tableau",
    "no_changes_review": "Aucun changement à revoir.",
    "nouvelle_analyse": "Nouvelle Analyse",
    "analyse_comparative": "Analyse Comparative des Indicateurs",
    "statistiques_validation": "Statistiques de Validation",
    "kpi_matched": "Tableaux appariés",
    "kpi_added": "Indicateurs ajoutés",
    "kpi_removed": "Indicateurs retirés",
    "kpi_renamed": "Renommages",
    "kpi_compared_pairs": "Paires comparées",
    "kpi_removed_tables": "Tableaux supprimés",
    "kpi_notes_modified": "Notes modifiées",
    "kpi_priority_tables": "Tableaux prioritaires",
    "kpi_low_confidence_tables": "Tableaux à faible confiance",
    "kpi_changed_t1": "Tableaux changés (précédent)",
    "kpi_changed_t2": "Tableaux changés (courant)",
    "btn_approve": "Valider",
    "btn_reject": "Rejeter",
    "btn_apply": "Appliquer",
    "btn_prev": "Précédent",
    "btn_next": "Suivant",
    "detail_changement": "Détail du changement",
    "no_indicators": "Aucun indicateur",
    "image_unavailable": "Image non disponible",
    "no_table_added_t2": "Aucun tableau dans le trimestre précédent",
    "no_table_removed_t2": "Aucun tableau dans le trimestre courant",
    "decision_analyst": "Décision de l'analyste",
    "comment_optional": "Commentaire (optionnel)",
    "option_force_reextract": "Forcer la ré-extraction (ignorer le cache)",
}

# -----------------------------------------------------------------------------
# B) Status mapping (codes -> FR)
# -----------------------------------------------------------------------------
TABLE_STATUS: dict[str, str] = {
    "ajoute": "Tableau ajouté",
    "retire": "Tableau retiré",
    "supprime": "Tableau retiré",
    "match": "Tableau apparié",
    "matched": "Tableau apparié",
    "incertain": "Appariement incertain",
    "structure_change": "Fusion/scission",
    "modifie": "Modifié",
    "stable": "Stable",
    "needs_review": "À revoir",
}


def status_fr(code: str) -> str:
    """Convertit un code de statut de tableau ou d'indicateur en libelle francais."""
    if not code:
        return ""
    c = str(code).strip().lower()
    return TABLE_STATUS.get(c, code)


# Indicator change types (used in UI badges/labels)
INDICATOR_CHANGE_TYPE: dict[str, str] = {
    "added": "Ajout",
    "removed": "Retrait",
    "renamed": "Renommage",
}

# -----------------------------------------------------------------------------
# C) Reason mapping (reason_code -> FR)
# -----------------------------------------------------------------------------
REASON_MAP: dict[str, str] = {
    "table_number_low_overlap_rescue": "Appariement sauvé par numéro de tableau (chevauchement faible)",
    "cross_section_forbidden": "Sections différentes (appariement interdit)",
    "table_number_conflict": "Conflit de numéro de tableau",
    "low_label_overlap_reject": "Chevauchement d'indicateurs insuffisant",
    "size_mismatch_reject": "Dimensions incompatibles",
    "title_match": "Titres compatibles",
    "anchor_match": "Indicateurs compatibles",
    "rescued": "Appariement sauvé",
    "hungarian": "Affectation optimale",
    "indicator_set_hash_exact": "Indicateurs identiques (hash exact)",
    "indicator_overlap_match": "Chevauchement d'indicateurs suffisant",
    "multi_signal_match": "Plusieurs signaux concordants",
    "few_indicators_header_footer_match": "Peu d'indicateurs, en-tête/pied compatibles",
    "title_override_match": "Correspondance forcée par le titre",
    "date_title_structure_rescue": "Sauvetage par titre date + structure",
    "generic_title_insufficient_signals": "Titre générique, signaux insuffisants",
    "low_containment": "Contenu insuffisant",
    "weak_signals": "Signaux faibles",
    "unknown_section_penalized": "Section inconnue (pénalisée)",
    "unknown_section": "Section inconnue",
    "no_candidate_same_section": "Aucun candidat dans la même section",
    "uncertain_competition": "Compétition incertaine",
    "single_rescue": "Sauvetage unique",
    "split_merge_rescue": "Sauvetage fusion/scission",
    "split_probable": "Fusion/scission probable",
    "removed_table": "Tableau retiré",
    "added_table": "Tableau ajouté",
    "unmatched": "Non apparié",
    "id": "Correspondance par identifiant",
    "score": "Appariement par score",
    "rescue_split_merge": "Sauvetage fusion/scission",
    "rescue_high_jaccard": "Sauvetage par Jaccard élevé",
}


def reason_fr(code: str) -> str:
    """Convertit un code de raison en libelle francais. Les codes inconnus sont retournes tels quels."""
    if not code:
        return ""
    c = str(code).strip()
    return REASON_MAP.get(c, c)


# -----------------------------------------------------------------------------
# D) Source method mapping
# -----------------------------------------------------------------------------
SOURCE_METHOD_MAP: dict[str, str] = {
    "docling": "Extraction Docling",
    "vision_fallback_gpt4o": "Extraction vision (GPT-4o)",
    "vision_fallback_gpt-5.4": "Extraction vision (GPT-5.4)",
    "vision_fallback": "Extraction vision (GPT-4o)",
    "vector": "Extraction vectorielle",
}


def source_method_fr(method: str) -> str:
    """Convertit une methode source en libelle francais."""
    if not method:
        return ""
    m = str(method).strip()
    return SOURCE_METHOD_MAP.get(m, m)


# -----------------------------------------------------------------------------
# E) Generic translation helper
# -----------------------------------------------------------------------------
def t(key: str, default: str | None = None) -> str:
    """Retourne la traduction francaise pour une cle, ou la valeur par defaut, ou la cle elle-meme."""
    if not key:
        return ""
    val = UI_LABELS.get(key)
    if val is not None:
        return val
    if default is not None:
        return default
    return key


# -----------------------------------------------------------------------------
# F) Analyst-facing French normalization (Dash / Excel)
# -----------------------------------------------------------------------------
_IMPACT_LABEL_FR: dict[str, str] = {
    "MAJEUR": "Majeur",
    "MODERE": "Modéré",
    "MINEUR": "Mineur",
}

_ANALYST_ENGLISH_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\badded\b", re.IGNORECASE), "ajout"),
    (re.compile(r"\bremoved\b", re.IGNORECASE), "suppression"),
    (re.compile(r"\brenamed\b", re.IGNORECASE), "renommage"),
    (re.compile(r"\bmodified\b", re.IGNORECASE), "modification"),
    (re.compile(r"\binvestigation\b", re.IGNORECASE), "analyse approfondie"),
    (re.compile(r"\bchunks?\b", re.IGNORECASE), "passage"),
    (re.compile(r"\bfragments?\b", re.IGNORECASE), "passage"),
)

_T1_RE = re.compile(r"\bT1\b")
_T2_RE = re.compile(r"\bT2\b")
_META_FRAGMENTS_RE = re.compile(
    r"^Les deux fragments\b",
    re.IGNORECASE,
)


def impact_label_fr(code: str) -> str:
    """Convertit un code d'impact AMF en libellé français pour l'analyste."""
    normalized = str(code or "").strip().upper()
    return _IMPACT_LABEL_FR.get(normalized, str(code or "").strip().capitalize() or "Mineur")


def sanitize_analyst_french(text: str) -> str:
    """Normalise un texte destiné à l'analyste : français soutenu, sans jargon pipeline.

    Remplace T1/T2, fragment/chunk et termes anglais résiduels. Ne modifie pas
    le sens métier ; sert uniquement l'affichage Dash et l'export Excel.
    """
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ""

    if _META_FRAGMENTS_RE.match(value):
        value = _META_FRAGMENTS_RE.sub("Les deux passages", value, count=1)

    value = _T1_RE.sub("rapport précédent", value)
    value = _T2_RE.sub("rapport courant", value)

    for pattern, replacement in _ANALYST_ENGLISH_REPLACEMENTS:
        value = pattern.sub(replacement, value)

    return value
