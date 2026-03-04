"""
Comparateur structurel de tableaux - Detection des changements de structure uniquement.

Ce module detecte UNIQUEMENT les changements structurels entre tableaux:
- Lignes ajoutees (indicateur present dans T2 mais pas dans T1)
- Lignes supprimees (indicateur present dans T1 mais pas dans T2)
- Tableaux ajoutes (tableau entier present dans T2 mais pas dans T1)
- Tableaux supprimes (tableau entier present dans T1 mais pas dans T2)

EXCLUSIONS (par design):
- Valeurs numeriques (montants, pourcentages, ratios)
- Colonnes (entetes de dates, periodes)
- Contenu textuel des cellules
- Notes de bas de page
- Modifications de formatage

La premiere colonne de chaque tableau est utilisee comme INDICATEUR
pour identifier les lignes de maniere unique.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None

try:
    from .orphan_matcher import GenAIOrphanMatcher
except ImportError:
    GenAIOrphanMatcher = None

from vigilance.utils.indicator_cleaner import (
    normalize_indicator_for_comparison,
    strip_dates_from_table_title,
    strip_note_refs_from_title,
)
from vigilance.utils.matching_normalizer import is_date_only_line

from .displacement_detector import (
    AddedItem,
    RemovedItem,
    detect_cross_table_displacements,
)
from .noise_filter import NoiseFilter

try:
    from vigilance.extraction.section_taxonomy import canonicalize_section
except Exception:
    canonicalize_section = None

# Import paresseux pour eviter cycle: content_filters -> analysis.__init__ -> table_comparator
_filter_changes_cache = None


def _get_filter_changes():
    """Import filter_changes a la demande (evite import circulaire)."""
    global _filter_changes_cache
    if _filter_changes_cache is None:
        try:
            from vigilance.analysis.content_filters import filter_changes

            _filter_changes_cache = filter_changes
        except ImportError:
            _filter_changes_cache = False
    return _filter_changes_cache if _filter_changes_cache is not False else None


try:
    from ..output.visual_proofs import VisualProof, generate_visual_proofs

    VISUAL_PROOFS_AVAILABLE = True
except ImportError:
    VISUAL_PROOFS_AVAILABLE = False
    VisualProof = None

logger = logging.getLogger(__name__)
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}

# Section labels for display
SECTION_LABELS = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "capital_management": "Gestion du capital",
    "risk_management": "Gestion des risques",
    "unknown_section": "unknown_section",
}


def _canonical_section(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if canonicalize_section is None:
        return raw.lower()
    try:
        return canonicalize_section(raw)
    except Exception:
        return raw.lower()


def _sections_strict_match(table_t1: dict[str, Any], table_t2: dict[str, Any]) -> bool:
    section_t1 = _canonical_section(str(table_t1.get("section", "")))
    section_t2 = _canonical_section(str(table_t2.get("section", "")))
    if section_t1 in UNKNOWN_SECTIONS or section_t2 in UNKNOWN_SECTIONS:
        return False
    return section_t1 == section_t2


@dataclass
class StructuralChange:
    """Represente un changement structurel detecte."""

    table_title: str
    change_type: (
        str  # "row_added", "row_removed", "row_renamed", "table_added", "table_removed"
    )
    indicator: str  # Texte de la premiere colonne (label de ligne)

    # Metadata optionnelle
    page_t1: int | None = None
    page_t2: int | None = None
    section: str | None = None  # "gestion_capital" ou "gestion_risques"

    # Type metier EDTF/AMF: IFC, RG, PB
    type_metier: str | None = None

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour serialisation."""
        from vigilance.utils.type_metier import compute_type_metier

        type_metier_val = self.type_metier or compute_type_metier(
            self.section, self.change_type
        )
        return {
            "table_title": self.table_title,
            "change_type": self.change_type,
            "indicator": self.indicator,
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
            "section": self.section,
            "type_metier": type_metier_val,
        }


@dataclass
class StructuralChangeAnalyzed:
    """
    Changement structurel avec analyse GenAI.

    Format de sortie conforme au tableau demande:
    | Titre | Page | Phrase | Nouvelle idee | Justification |
    """

    titre: str  # Section (Gestion du capital, Gestion des risques)
    page: int
    phrase: str  # Indicateur complet (premiere colonne)
    change_type: str  # "ajoute" ou "supprime"
    nouvelle_idee: str  # "Oui" ou "Non"
    justification: str
    pertinence: str  # "pertinent", "non_pertinent", "nouvelle_idee"

    # Metadata optionnelle
    table_title: str | None = None
    table_number: str | None = None
    page_t1: int | None = None
    page_t2: int | None = None

    # Type metier EDTF/AMF: IFC, RG, PB
    type_metier: str | None = None

    # Verification humaine
    verified: bool = False  # Verifie manuellement par l'utilisateur
    verified_by: str | None = None  # Utilisateur/timestamp
    verified_at: str | None = None  # Timestamp de validation

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour serialisation."""
        from vigilance.utils.type_metier import compute_type_metier

        type_metier_val = self.type_metier or compute_type_metier(
            self.titre, self.change_type
        )
        return {
            "titre": self.titre,
            "page": self.page,
            "phrase": self.phrase,
            "change_type": self.change_type,
            "nouvelle_idee": self.nouvelle_idee,
            "justification": self.justification,
            "pertinence": self.pertinence,
            "table_title": self.table_title,
            "table_number": self.table_number,
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
            "type_metier": type_metier_val,
            "verified": self.verified,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
        }

    def to_table_row(self) -> list[str]:
        """Convertir en ligne de tableau."""
        return [
            self.titre,
            str(self.page),
            self.phrase,
            self.nouvelle_idee,
            self.justification,
        ]


@dataclass
class AnalyzedComparisonResult:
    """
    Resultat de comparaison structurelle avec analyse GenAI.

    Format de sortie JSON complet avec legende et resume.
    """

    comparison_date: str = field(
        default_factory=lambda: datetime.now().isoformat()[:10]
    )
    mode: str = "structural_analyzed"
    bank_code: str | None = None

    # Changements analyses
    changes: list[StructuralChangeAnalyzed] = field(default_factory=list)
    table_matching: dict[str, Any] = field(default_factory=dict)

    # Legende
    legende: dict[str, str] = field(
        default_factory=lambda: {
            "vert": "Pertinent, a garder",
            "rouge": "Non pertinent, a retirer",
            "jaune": "Nouvelle idee",
        }
    )

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour serialisation JSON."""
        summary = {
            "total": len(self.changes),
            "nouvelles_idees": sum(1 for c in self.changes if c.nouvelle_idee == "Oui"),
            "pertinents": sum(1 for c in self.changes if c.pertinence == "pertinent"),
            "non_pertinents": sum(
                1 for c in self.changes if c.pertinence == "non_pertinent"
            ),
            "ajouts": sum(1 for c in self.changes if c.change_type == "ajoute"),
            "suppressions": sum(1 for c in self.changes if c.change_type == "supprime"),
            "verifies": sum(1 for c in self.changes if c.verified),
        }

        if self.table_matching:
            summary["total_tables_matched_strong"] = len(
                self.table_matching.get("strong_matches", [])
            )
            summary["total_tables_matched_probable"] = len(
                self.table_matching.get("probable_matches", [])
            )
            summary["total_tables_unmatched_t1"] = len(
                self.table_matching.get("unmatched_t1", [])
            )
            summary["total_tables_unmatched_t2"] = len(
                self.table_matching.get("unmatched_t2", [])
            )

        changements_par_tableau = group_changes_by_table(self.changes)

        return {
            "comparison_date": self.comparison_date,
            "mode": self.mode,
            "bank_code": self.bank_code,
            "legende": self.legende,
            "changes": [c.to_dict() for c in self.changes],
            "changements_par_tableau": changements_par_tableau,
            "table_matching": self.table_matching,
            "summary": summary,
        }


@dataclass
class TableStructuralChanges:
    """Changements structurels pour un tableau specifique."""

    table_title: str
    table_number: str | None = None
    rows_added: list[str] = field(default_factory=list)
    rows_removed: list[str] = field(default_factory=list)
    rows_renamed: list[dict[str, Any]] = field(
        default_factory=list
    )  # {from: str, to: str, confidence: str, reasoning: str}
    rows_displaced: list[dict[str, Any]] = field(
        default_factory=list
    )  # {indicator, from_table, from_page, to_table, to_page}
    page_t1: int | None = None
    page_t2: int | None = None

    @property
    def has_changes(self) -> bool:
        """Verifier si le tableau a des changements structurels."""
        return (
            len(self.rows_added) > 0
            or len(self.rows_removed) > 0
            or len(self.rows_renamed) > 0
            or len(self.rows_displaced) > 0
        )

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour serialisation."""
        return {
            "table_title": self.table_title,
            "table_number": self.table_number,
            "rows_added": self.rows_added,
            "rows_removed": self.rows_removed,
            "rows_renamed": self.rows_renamed,
            "rows_displaced": self.rows_displaced,
            "rows_added_count": len(self.rows_added),
            "rows_removed_count": len(self.rows_removed),
            "rows_renamed_count": len(self.rows_renamed),
            "rows_displaced_count": len(self.rows_displaced),
            "page_t1": self.page_t1,
            "page_t2": self.page_t2,
        }


@dataclass
class StructuralComparisonResult:
    """Resultat complet de la comparaison structurelle."""

    comparison_date: str = field(
        default_factory=lambda: datetime.now().isoformat()[:10]
    )
    mode: str = "structural_only"

    # Changements par tableau
    table_changes: list[TableStructuralChanges] = field(default_factory=list)

    # Tableaux entiers ajoutes/supprimes (objets complets avec metadonnees)
    tables_added: list[dict[str, Any]] = field(default_factory=list)
    tables_removed: list[dict[str, Any]] = field(default_factory=list)
    matched_tables_strong: list[dict[str, Any]] = field(default_factory=list)
    matched_tables_probable: list[dict[str, Any]] = field(default_factory=list)
    unmatched_tables_t1: list[dict[str, Any]] = field(default_factory=list)
    unmatched_tables_t2: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    bank_code: str | None = None
    sections_analyzed: list[str] = field(default_factory=list)

    # Visual proofs paths (PNG files)
    visual_proofs_dir: str | None = None
    visual_proofs: list[Any] = field(default_factory=list)  # List of VisualProof

    @property
    def tables_with_changes(self) -> list[TableStructuralChanges]:
        """Retourner seulement les tableaux avec des changements."""
        return [tc for tc in self.table_changes if tc.has_changes]

    @property
    def total_rows_added(self) -> int:
        """Nombre total de lignes ajoutees."""
        return sum(len(tc.rows_added) for tc in self.table_changes)

    @property
    def total_rows_removed(self) -> int:
        """Nombre total de lignes supprimees."""
        return sum(len(tc.rows_removed) for tc in self.table_changes)

    @property
    def total_rows_renamed(self) -> int:
        """Nombre total de lignes renommees."""
        return sum(len(tc.rows_renamed) for tc in self.table_changes)

    @property
    def total_rows_displaced(self) -> int:
        """Nombre total de lignes deplacees (meme contenu, autre tableau)."""
        return sum(len(tc.rows_displaced) for tc in self.table_changes)

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour serialisation JSON."""

        # Helper pour extraire le titre d'un tableau
        def get_table_title(table: Any) -> str:
            if isinstance(table, str):
                return table
            return (
                table.get("title", "")
                or table.get("table_title", "")
                or table.get("name", "")
                or ""
            )

        def get_table_number(table: Any) -> str | None:
            if isinstance(table, dict):
                return table.get("table_number")
            return None

        def format_match_entry(entry: dict[str, Any]) -> dict[str, Any]:
            t1 = entry.get("t1", {})
            t2 = entry.get("t2", {})
            return {
                "score": entry.get("score", 0.0),
                "match_level": entry.get("match_level", "unknown"),
                "match_basis": entry.get("match_basis", "unknown"),
                "t1": {
                    "title": get_table_title(t1),
                    "table_number": get_table_number(t1),
                    "page": t1.get("page_number", 0) if isinstance(t1, dict) else 0,
                    "table_id": t1.get("table_id", "") if isinstance(t1, dict) else "",
                },
                "t2": {
                    "title": get_table_title(t2),
                    "table_number": get_table_number(t2),
                    "page": t2.get("page_number", 0) if isinstance(t2, dict) else 0,
                    "table_id": t2.get("table_id", "") if isinstance(t2, dict) else "",
                },
            }

        return {
            "comparison_date": self.comparison_date,
            "mode": self.mode,
            "bank_code": self.bank_code,
            "sections_analyzed": self.sections_analyzed,
            "changes": [tc.to_dict() for tc in self.tables_with_changes],
            "tables_added": [
                get_table_title(t) for t in self.tables_added
            ],  # Titres pour compatibilite
            "tables_removed": [
                get_table_title(t) for t in self.tables_removed
            ],  # Titres pour compatibilite
            "tables_added_full": [
                {
                    "title": get_table_title(t),
                    "table_number": get_table_number(t),
                    "page": t.get("page_number", 0) if isinstance(t, dict) else 0,
                }
                for t in self.tables_added
            ],
            "tables_removed_full": [
                {
                    "title": get_table_title(t),
                    "table_number": get_table_number(t),
                    "page": t.get("page_number", 0) if isinstance(t, dict) else 0,
                }
                for t in self.tables_removed
            ],
            "table_matching": {
                "strong_matches": [
                    format_match_entry(m) for m in self.matched_tables_strong
                ],
                "probable_matches": [
                    format_match_entry(m) for m in self.matched_tables_probable
                ],
                "unmatched_t1": [
                    {
                        "title": get_table_title(t),
                        "table_number": get_table_number(t),
                        "page": t.get("page_number", 0) if isinstance(t, dict) else 0,
                        "table_id": t.get("table_id", "")
                        if isinstance(t, dict)
                        else "",
                    }
                    for t in self.unmatched_tables_t1
                ],
                "unmatched_t2": [
                    {
                        "title": get_table_title(t),
                        "table_number": get_table_number(t),
                        "page": t.get("page_number", 0) if isinstance(t, dict) else 0,
                        "table_id": t.get("table_id", "")
                        if isinstance(t, dict)
                        else "",
                    }
                    for t in self.unmatched_tables_t2
                ],
            },
            "summary": {
                "total_tables_compared": len(self.table_changes),
                "tables_with_changes": len(self.tables_with_changes),
                "total_rows_added": self.total_rows_added,
                "total_rows_removed": self.total_rows_removed,
                "total_rows_renamed": self.total_rows_renamed,
                "total_rows_displaced": self.total_rows_displaced,
                "total_tables_added": len(self.tables_added),
                "total_tables_removed": len(self.tables_removed),
                "total_tables_matched_strong": len(self.matched_tables_strong),
                "total_tables_matched_probable": len(self.matched_tables_probable),
                "total_tables_unmatched_t1": len(self.unmatched_tables_t1),
                "total_tables_unmatched_t2": len(self.unmatched_tables_t2),
            },
            "visual_proofs": {
                "enabled": bool(self.visual_proofs),
                "output_dir": self.visual_proofs_dir,
                "count": len(self.visual_proofs),
                "files": [
                    str(p.image_path)
                    for p in self.visual_proofs
                    if hasattr(p, "image_path") and p.image_path
                ],
            },
        }


class StructuralTableComparator:
    """
    Comparateur structurel de tableaux.

    Compare UNIQUEMENT la structure des tableaux (premiere colonne = indicateurs).
    Ignore completement les valeurs numeriques et le contenu des autres colonnes.

    Utilisation:
        comparator = StructuralTableComparator()
        result = comparator.compare_tables(tables_t1, tables_t2)
    """

    # Seuil de similarite pour considerer une table comme renommee
    TABLE_RENAME_THRESHOLD = 0.85
    TABLE_PROBABLE_THRESHOLD = 0.65

    # Seuil de similarite pour considerer deux indicateurs comme identiques
    # 93% evite faux positifs (ex: "Indicateur C" vs "Indicateur D" = 91.67)
    # tout en gardant "Ratio de levier" vs "Ratio du levier" (93.33)
    INDICATOR_SIMILARITY_THRESHOLD = 0.93

    def __init__(
        self,
        table_rename_threshold: float | None = None,
        table_probable_threshold: float | None = None,
        indicator_similarity_threshold: float | None = None,
        normalize_indicators: bool = True,
        use_genai_matching: bool = False,
        genai_api_key: str | None = None,
        orphan_matcher_max_orphans: int = 50,
        bank_code: str | None = None,
    ):
        """
        Initialiser le comparateur structurel.

        Args:
            table_rename_threshold: Seuil Jaccard pour detecter les tables renommees
            table_probable_threshold: Seuil Jaccard pour match probable (revue humaine)
            indicator_similarity_threshold: Seuil pour considerer 2 indicateurs identiques
            normalize_indicators: Normaliser les indicateurs avant comparaison
            use_genai_matching: Utiliser GenAI pour apparier les lignes orphelines
            genai_api_key: Cle API pour GenAI
            orphan_matcher_max_orphans: Nombre max d'orphelins pour appeler l'Orphan Matcher
            bank_code: Code banque pour seuils specifiques (ex. td -> matching_overrides)
        """
        try:
            from vigilance.config import get_matching_thresholds

            thresholds = get_matching_thresholds(bank_code=bank_code)
        except ImportError:
            thresholds = {}
        self.table_rename_threshold = (
            table_rename_threshold
            if table_rename_threshold is not None
            else float(thresholds.get("table_rename_threshold", 0.85))
        )
        raw_probable = (
            table_probable_threshold
            if table_probable_threshold is not None
            else float(thresholds.get("table_probable_threshold", 0.65))
        )
        self.table_probable_threshold = min(raw_probable, self.table_rename_threshold)
        self.indicator_similarity_threshold = (
            indicator_similarity_threshold
            if indicator_similarity_threshold is not None
            else float(thresholds.get("indicator_similarity_threshold", 0.93))
        )
        self.indicator_fuzzy_token_threshold = float(
            thresholds.get("indicator_fuzzy_token_threshold", 0.85)
        )
        self.normalize_indicators = normalize_indicators
        self.orphan_matcher_max_orphans = orphan_matcher_max_orphans

        self.orphan_matcher = None
        if use_genai_matching and GenAIOrphanMatcher:
            self.orphan_matcher = GenAIOrphanMatcher(api_key=genai_api_key)

    def compare_tables(
        self,
        tables_t1: list[dict[str, Any]],
        tables_t2: list[dict[str, Any]],
        bank_code: str | None = None,
        section: str | None = None,
        pdf_t1_path: str | Path | None = None,
        pdf_t2_path: str | Path | None = None,
        visual_proofs_dir: str | Path | None = None,
    ) -> StructuralComparisonResult:
        """
        Comparer les tableaux entre T1 et T2 pour detecter les changements structurels.

        Args:
            tables_t1: Liste des tableaux du rapport T1 (ancien)
            tables_t2: Liste des tableaux du rapport T2 (nouveau)
            bank_code: Code de la banque (optionnel)
            section: Section analysee (optionnel)
            pdf_t1_path: Chemin du PDF T1 pour les preuves visuelles
            pdf_t2_path: Chemin du PDF T2 pour les preuves visuelles
            visual_proofs_dir: Repertoire de sortie pour les PNG (si None, desactive)

        Returns:
            StructuralComparisonResult avec tous les changements detectes
        """
        result = StructuralComparisonResult(
            bank_code=bank_code, sections_analyzed=[section] if section else []
        )

        # Etape 1: Matcher les tables entre T1 et T2
        (
            strong_matches,
            probable_matches,
            added_tables,
            removed_tables,
            unmatched_t1,
            unmatched_t2,
        ) = self._match_tables(tables_t1, tables_t2)

        matched_tables = [(m["t1"], m["t2"]) for m in strong_matches]
        result.matched_tables_strong = strong_matches
        result.matched_tables_probable = probable_matches
        result.unmatched_tables_t1 = unmatched_t1
        result.unmatched_tables_t2 = unmatched_t2

        # Etape 2: Pour chaque paire de tables matchees, comparer les indicateurs
        for t1, t2 in matched_tables:
            table_changes = self._compare_table_indicators(t1, t2)
            result.table_changes.append(table_changes)

        # Etape 2b: Detection des deplacements cross-tableaux
        self._apply_displacement_filter(result.table_changes)

        # Etape 3: Enregistrer les tables entierement ajoutees/supprimees (objets complets)
        result.tables_added = (
            added_tables  # Stocker les objets complets avec page_number
        )
        result.tables_removed = (
            removed_tables  # Stocker les objets complets avec page_number
        )

        # Etape 4: Generer les preuves visuelles si demande
        if (
            visual_proofs_dir
            and pdf_t1_path
            and pdf_t2_path
            and VISUAL_PROOFS_AVAILABLE
        ):
            result.visual_proofs_dir = str(visual_proofs_dir)
            result.visual_proofs = self._generate_visual_proofs(
                matched_tables=matched_tables,
                tables_with_changes=result.tables_with_changes,
                pdf_t1_path=pdf_t1_path,
                pdf_t2_path=pdf_t2_path,
                output_dir=visual_proofs_dir,
                bank_code=bank_code,
                section=section,
            )
            logger.info(
                f"Preuves visuelles generees: {len(result.visual_proofs)} fichiers PNG"
            )

        # Log du resume
        logger.info(
            f"Comparaison structurelle terminee: "
            f"{len(result.matched_tables_strong)} matchs forts, "
            f"{len(result.matched_tables_probable)} matchs probables, "
            f"{len(result.tables_with_changes)} tableaux avec changements, "
            f"{result.total_rows_added} lignes ajoutees, "
            f"{result.total_rows_removed} lignes supprimees, "
            f"{result.total_rows_displaced} lignes deplacees, "
            f"{len(result.tables_added)} tableaux ajoutes, "
            f"{len(result.tables_removed)} tableaux supprimes"
        )

        return result

    def _generate_visual_proofs(
        self,
        matched_tables: list[tuple],
        tables_with_changes: list[TableStructuralChanges],
        pdf_t1_path: str | Path,
        pdf_t2_path: str | Path,
        output_dir: str | Path,
        bank_code: str | None = None,
        section: str | None = None,
    ) -> list[Any]:
        """
        Generer les preuves visuelles PNG pour les tables avec changements.

        Args:
            matched_tables: Paires de tables matchees (t1, t2)
            tables_with_changes: Tables qui ont des changements
            pdf_t1_path: Chemin du PDF T1
            pdf_t2_path: Chemin du PDF T2
            output_dir: Repertoire de sortie
            bank_code: Code banque pour nommage
            section: Section pour nommage

        Returns:
            Liste de VisualProof
        """
        if not VISUAL_PROOFS_AVAILABLE:
            return []

        # Construire la liste des match_results pour generate_visual_proofs
        match_results = []

        # Creer un set des titres avec changements pour filtrage
        changed_titles = {tc.table_title for tc in tables_with_changes}

        for i, (t1, t2) in enumerate(matched_tables):
            title_t1 = self._get_table_title(t1)
            title_t2 = self._get_table_title(t2)

            # Verifier si cette table a des changements
            has_changes = title_t1 in changed_titles or title_t2 in changed_titles

            # Generer la cle unique
            safe_title = (
                re.sub(r"[^\w\-]", "_", title_t1[:50]) if title_t1 else f"table_{i}"
            )
            prefix = f"{bank_code}_{section}_" if bank_code and section else ""
            match_key = f"{prefix}{safe_title}"

            match_results.append(
                {
                    "match_key": match_key,
                    "table_a": {
                        "page_num": t1.get("page_number", 0),
                        "table_id": t1.get("table_id", ""),
                        "title": title_t1,
                        "table_number": t1.get("table_number"),
                        "bbox": t1.get("bbox"),
                    },
                    "table_b": {
                        "page_num": t2.get("page_number", 0),
                        "table_id": t2.get("table_id", ""),
                        "title": title_t2,
                        "table_number": t2.get("table_number"),
                        "bbox": t2.get("bbox"),
                    },
                    "score": 1.0,  # Tables deja matchees
                    "is_ambiguous": False,
                    "has_changes": has_changes,
                }
            )

        # Generer les preuves visuelles uniquement pour les tables avec changements
        proofs = generate_visual_proofs(
            match_results=match_results,
            pdf_a_path=pdf_t1_path,
            pdf_b_path=pdf_t2_path,
            output_dir=output_dir,
            only_changes=True,  # Seulement les tables avec changements
        )

        return proofs

    def _match_tables(
        self, tables_t1: list[dict[str, Any]], tables_t2: list[dict[str, Any]]
    ) -> tuple:
        """
        Matcher les tables entre T1 et T2 par titre ou par similarite des indicateurs.

        Returns:
            Tuple (
                strong_matches,
                probable_matches,
                added_tables,
                removed_tables,
                unmatched_t1,
                unmatched_t2,
            )
        """
        strong_matches = []
        probable_matches = []
        matched_t1_indices: set[int] = set()
        matched_t2_indices: set[int] = set()

        # Index par titre normalise (refs notes et dates retirees pour le matching)
        t1_by_title = {}
        for i, t in enumerate(tables_t1):
            if _canonical_section(str(t.get("section", ""))) in UNKNOWN_SECTIONS:
                continue
            raw_title = self._get_table_title(t)
            title_clean = strip_note_refs_from_title(raw_title)
            title_no_dates = strip_dates_from_table_title(title_clean)
            title = self._normalize_indicator(title_no_dates)
            if title:
                t1_by_title[title] = (i, t)

        t2_by_title = {}
        for j, t in enumerate(tables_t2):
            if _canonical_section(str(t.get("section", ""))) in UNKNOWN_SECTIONS:
                continue
            raw_title = self._get_table_title(t)
            title_clean = strip_note_refs_from_title(raw_title)
            title_no_dates = strip_dates_from_table_title(title_clean)
            title = self._normalize_indicator(title_no_dates)
            if title:
                t2_by_title[title] = (j, t)

        # Match exact par titre
        for title, (i, t1) in t1_by_title.items():
            if title in t2_by_title:
                j, t2 = t2_by_title[title]
                if not _sections_strict_match(t1, t2):
                    continue
                strong_matches.append(
                    self._build_match_entry(
                        t1=t1,
                        t2=t2,
                        score=1.0,
                        match_level="strong",
                        match_basis="exact_title",
                    )
                )
                matched_t1_indices.add(i)
                matched_t2_indices.add(j)
                logger.debug(f"Match exact: '{title}'")

        # Pour les non-matches, essayer par similarite des indicateurs
        unmatched_t1 = [
            (i, t) for i, t in enumerate(tables_t1) if i not in matched_t1_indices
        ]
        unmatched_t2 = [
            (j, t) for j, t in enumerate(tables_t2) if j not in matched_t2_indices
        ]

        for i, t1 in unmatched_t1:
            best_match = None
            best_similarity = 0.0

            indicators_t1 = self._extract_first_column(t1)
            if not indicators_t1:
                continue

            for j, t2 in unmatched_t2:
                if j in matched_t2_indices:
                    continue
                if not _sections_strict_match(t1, t2):
                    continue

                indicators_t2 = self._extract_first_column(t2)
                if not indicators_t2:
                    continue

                # Calculer similarite Jaccard
                similarity = self._jaccard_similarity(indicators_t1, indicators_t2)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (j, t2)

            # Si similarite >= seuil fort, considerer comme meme table (renommee)
            if best_match and best_similarity >= self.table_rename_threshold:
                j, t2 = best_match
                strong_matches.append(
                    self._build_match_entry(
                        t1=t1,
                        t2=t2,
                        score=best_similarity,
                        match_level="strong",
                        match_basis="indicator_similarity",
                    )
                )
                matched_t1_indices.add(i)
                matched_t2_indices.add(j)

                t1_title = self._get_table_title(t1)
                t2_title = self._get_table_title(t2)
                logger.info(
                    f"Table renommee detectee ({best_similarity:.1%}): '{t1_title}' -> '{t2_title}'"
                )
            elif best_match and best_similarity >= self.table_probable_threshold:
                j, t2 = best_match
                probable_matches.append(
                    self._build_match_entry(
                        t1=t1,
                        t2=t2,
                        score=best_similarity,
                        match_level="probable",
                        match_basis="indicator_similarity",
                    )
                )
                matched_t1_indices.add(i)
                matched_t2_indices.add(j)
                logger.info(
                    "Table probable (revue humaine) detectee "
                    f"({best_similarity:.1%}): '{self._get_table_title(t1)}' -> "
                    f"'{self._get_table_title(t2)}'"
                )

        # Tables non matchees
        added_tables = [
            t for j, t in enumerate(tables_t2) if j not in matched_t2_indices
        ]
        removed_tables = [
            t for i, t in enumerate(tables_t1) if i not in matched_t1_indices
        ]
        unmatched_t1 = [
            t for i, t in enumerate(tables_t1) if i not in matched_t1_indices
        ]
        unmatched_t2 = [
            t for j, t in enumerate(tables_t2) if j not in matched_t2_indices
        ]

        return (
            strong_matches,
            probable_matches,
            added_tables,
            removed_tables,
            unmatched_t1,
            unmatched_t2,
        )

    def _build_match_entry(
        self,
        t1: dict[str, Any],
        t2: dict[str, Any],
        score: float,
        match_level: str,
        match_basis: str,
    ) -> dict[str, Any]:
        """Construire une entree standard pour les matches de tableaux."""
        return {
            "t1": t1,
            "t2": t2,
            "score": score,
            "match_level": match_level,
            "match_basis": match_basis,
        }

    def _compare_table_indicators(
        self, table_t1: dict[str, Any], table_t2: dict[str, Any]
    ) -> TableStructuralChanges:
        """
        Comparer les indicateurs (premiere colonne) entre deux tables matchees.

        Args:
            table_t1: Table du rapport T1
            table_t2: Table du rapport T2

        Returns:
            TableStructuralChanges avec lignes ajoutees/supprimees
        """
        title = self._get_table_title(table_t2) or self._get_table_title(table_t1)

        # Extraire les indicateurs (premiere colonne)
        indicators_t1 = self._extract_first_column(table_t1)
        indicators_t2 = self._extract_first_column(table_t2)

        # Normaliser pour comparaison
        normalized_t1 = {self._normalize_indicator(ind): ind for ind in indicators_t1}
        normalized_t2 = {self._normalize_indicator(ind): ind for ind in indicators_t2}

        # Detecter les ajouts et suppressions (exact match d'abord)
        rows_added = []
        rows_removed = []

        # Lignes ajoutees: dans T2 mais pas dans T1
        for norm_ind, original in normalized_t2.items():
            if norm_ind and norm_ind not in normalized_t1:
                rows_added.append(original)

        # Lignes supprimees: dans T1 mais pas dans T2
        for norm_ind, original in normalized_t1.items():
            if norm_ind and norm_ind not in normalized_t2:
                rows_removed.append(original)

        # Fuzzy matching: retirer les paires dont la similarite >= seuil
        if rapidfuzz_fuzz and rows_added and rows_removed:
            threshold_pct = int(self.indicator_similarity_threshold * 100)
            matched_pairs = self._fuzzy_match_indicators(
                rows_added, rows_removed, threshold_pct
            )
            for added_text, removed_text in matched_pairs:
                if added_text in rows_added and removed_text in rows_removed:
                    rows_added.remove(added_text)
                    rows_removed.remove(removed_text)
                    logger.debug(
                        f"Fuzzy match: '{removed_text}' ~ '{added_text}' "
                        f"(seuil {self.indicator_similarity_threshold})"
                    )

        rows_renamed = []

        # --- PASSE 2 : Orphan Matching (GenAI) ---
        # Si activé, on essaie de trouver des renommages sémantiques parmi les orphelins
        # Garde-fou: ne pas appeler si trop d'orphelins (coût API)
        orphan_count = len(rows_added) + len(rows_removed)
        if (
            self.orphan_matcher
            and rows_added
            and rows_removed
            and orphan_count <= self.orphan_matcher_max_orphans
        ):
            try:
                matches = self.orphan_matcher.match_orphans(
                    orphans_t1=rows_removed,
                    orphans_t2=rows_added,
                    context_description=f"Table: {title}",
                )

                for match in matches:
                    # Vérifier que les lignes sont toujours disponibles (non traitées)
                    if (
                        match.t1_indicator in rows_removed
                        and match.t2_indicator in rows_added
                    ):
                        # Retirer des listes d'ajout/suppression
                        rows_removed.remove(match.t1_indicator)
                        rows_added.remove(match.t2_indicator)

                        # Ajouter à la liste des renommés
                        rows_renamed.append(
                            {
                                "from": match.t1_indicator,
                                "to": match.t2_indicator,
                                "confidence": match.confidence,
                                "reasoning": match.reasoning,
                            }
                        )
                        logger.info(
                            f"Orphan match found: '{match.t1_indicator}' -> '{match.t2_indicator}'"
                        )

            except Exception as e:
                logger.error(f"Error in orphan matching: {e}")
                # En cas d'erreur, on continue sans rien changer (fallback gracefull)

        table_number = table_t2.get("table_number") or table_t1.get("table_number")

        return TableStructuralChanges(
            table_title=title,
            table_number=table_number,
            rows_added=rows_added,
            rows_removed=rows_removed,
            rows_renamed=rows_renamed,
            page_t1=table_t1.get("page_number"),
            page_t2=table_t2.get("page_number"),
        )

    def _apply_displacement_filter(
        self, table_changes: list[TableStructuralChanges]
    ) -> None:
        """
        Detecte les indicateurs deplaces entre tableaux et met a jour table_changes.

        Un indicateur present comme supprime dans un tableau et ajoute dans un autre
        est marque comme deplacement (rows_displaced) et retire de added/removed.
        """
        removed_items: list[RemovedItem] = []
        added_items: list[AddedItem] = []

        def _tc_key(tc: TableStructuralChanges) -> tuple:
            return (
                tc.table_title or "",
                tc.page_t1 or 0,
                tc.page_t2 or 0,
            )

        def _tc_key_str(tc: TableStructuralChanges) -> str:
            k = _tc_key(tc)
            return f"{k[0]}|{k[1]}|{k[2]}"

        for tc in table_changes:
            key = _tc_key(tc)
            key_str = _tc_key_str(tc)
            section = None
            # Pre-calcul normalisation pour voisinage
            rem_list = [
                (text, self._normalize_indicator(text)) for text in tc.rows_removed
            ]
            for i, (text, canonical) in enumerate(rem_list):
                if not canonical:
                    continue

                prev_n = rem_list[i - 1][1] if i > 0 else None
                next_n = rem_list[i + 1][1] if i < len(rem_list) - 1 else None

                removed_items.append(
                    RemovedItem(
                        text=text,
                        canonical=canonical,
                        table_id_t1=key_str,
                        page_t1=int(key[1]),
                        table_id_t2=key_str,
                        page_t2=int(key[2]),
                        section=section,
                        comparison_key=key,
                        neighbor_prev=prev_n,
                        neighbor_next=next_n,
                    )
                )

            add_list = [
                (text, self._normalize_indicator(text)) for text in tc.rows_added
            ]
            for i, (text, canonical) in enumerate(add_list):
                if not canonical:
                    continue

                prev_n = add_list[i - 1][1] if i > 0 else None
                next_n = add_list[i + 1][1] if i < len(add_list) - 1 else None

                added_items.append(
                    AddedItem(
                        text=text,
                        canonical=canonical,
                        table_id_t1=key_str,
                        page_t1=int(key[1]),
                        table_id_t2=key_str,
                        page_t2=int(key[2]),
                        section=section,
                        comparison_key=key,
                        neighbor_prev=prev_n,
                        neighbor_next=next_n,
                    )
                )

        displaced_canonicals, displaced_list = detect_cross_table_displacements(
            removed_items, added_items, self._normalize_indicator
        )

        for d in displaced_list:
            for tc in table_changes:
                if _tc_key_str(tc) == d.from_table_id:
                    tc.rows_removed = [
                        t
                        for t in tc.rows_removed
                        if self._normalize_indicator(t) != d.canonical
                    ]
                    break
            for tc in table_changes:
                if _tc_key_str(tc) == d.to_table_id:
                    tc.rows_added = [
                        t
                        for t in tc.rows_added
                        if self._normalize_indicator(t) != d.canonical
                    ]
                    tc.rows_displaced.append(d.to_dict())
                    break

    def _extract_first_column(self, table: dict[str, Any]) -> list[str]:
        """
        Extraire les valeurs de la premiere colonne (indicateurs).

        Prefere first_column_indicators si present (ex. labels_only), sinon rows.
        Filtre unites, dates, notes via is_date_only_line.

        Args:
            table: Dictionnaire representant un tableau

        Returns:
            Liste des indicateurs (valeurs de la premiere colonne)
        """
        # Prefer first_column_indicators si disponible (labels_only, vigie_extract)
        precomputed = table.get("first_column_indicators", [])
        if isinstance(precomputed, list) and precomputed:
            indicators = []
            for item in precomputed:
                cell = str(item).strip() if item else ""
                if (
                    cell
                    and not self._is_numeric_only(cell)
                    and not is_date_only_line(cell)
                ):
                    indicators.append(cell)
            if indicators:
                return indicators

        # Fallback: extraire depuis rows
        indicators = []
        rows = table.get("rows", [])
        for row in rows:
            if not row or len(row) == 0:
                continue

            indicator = str(row[0]).strip() if row[0] else ""

            if (
                indicator
                and not self._is_numeric_only(indicator)
                and not is_date_only_line(indicator)
            ):
                indicators.append(indicator)

        return indicators

    def _normalize_indicator(self, text: str) -> str:
        """Delegate to shared canonical key (comparison_runner uses the same)."""
        return normalize_indicator_for_comparison(text)

    def _fuzzy_match_indicators(
        self,
        rows_added: list[str],
        rows_removed: list[str],
        threshold_pct: int,
    ) -> list[tuple]:
        """
        Apparier 1-1 les indicateurs ajoutes/supprimes par similarite fuzzy.

        Retourne les paires (added, removed) dont la similarite >= threshold_pct.
        Affectation gourmande par score decroissant pour eviter les conflits.

        Uses normalized text for scoring to avoid false renames from
        accent/note-marker/series-suffix differences.
        """
        if not rows_added or not rows_removed:
            return []

        # Pre-normalize all texts for fuzzy scoring (avoids OCR noise like g/9, accents, notes)
        norm_added = {a: self._normalize_indicator(a) for a in rows_added}
        norm_removed = {r: self._normalize_indicator(r) for r in rows_removed}

        # Toutes les paires (added, removed, score)
        # Utilise ratio et token_set_ratio pour restructurations (ex. Dépôts X vs Dépôts Y)
        token_threshold_pct = int(
            getattr(self, "indicator_fuzzy_token_threshold", 0.85) * 100
        )
        candidates: list[tuple] = []
        for added in rows_added:
            na = norm_added[added]
            if not na:
                continue
            for removed in rows_removed:
                nr = norm_removed[removed]
                if not nr:
                    continue
                # Score on normalized text to ignore accents, notes, series suffixes
                ratio_score = rapidfuzz_fuzz.ratio(na, nr)
                token_score = rapidfuzz_fuzz.token_set_ratio(na, nr)
                score = max(ratio_score, token_score)
                if score >= threshold_pct or token_score >= token_threshold_pct:
                    candidates.append((added, removed, score))

        # Trier par score decroissant pour affectation gourmande
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Affectation 1-1
        used_added: set[str] = set()
        used_removed: set[str] = set()
        result: list[tuple] = []
        for added, removed, _ in candidates:
            if added not in used_added and removed not in used_removed:
                result.append((added, removed))
                used_added.add(added)
                used_removed.add(removed)

        return result

    def _get_table_title(self, table: dict[str, Any]) -> str:
        """
        Obtenir le titre d'un tableau.

        Args:
            table: Dictionnaire representant un tableau

        Returns:
            Titre du tableau
        """
        return (
            table.get("title", "")
            or table.get("table_title", "")
            or table.get("name", "")
            or ""
        )

    def _jaccard_similarity(self, set1: list[str], set2: list[str]) -> float:
        """
        Calculer la similarite Jaccard entre deux listes d'indicateurs.

        Args:
            set1: Premier ensemble d'indicateurs
            set2: Deuxieme ensemble d'indicateurs

        Returns:
            Score de similarite entre 0 et 1
        """
        # Normaliser les indicateurs
        norm_set1 = {self._normalize_indicator(s) for s in set1 if s}
        norm_set2 = {self._normalize_indicator(s) for s in set2 if s}

        if not norm_set1 and not norm_set2:
            return 0.0

        intersection = len(norm_set1 & norm_set2)
        union = len(norm_set1 | norm_set2)

        return intersection / union if union > 0 else 0.0

    def _is_numeric_only(self, text: str) -> bool:
        """
        Verifier si un texte est purement numerique (a ignorer comme indicateur).

        Args:
            text: Texte a verifier

        Returns:
            True si le texte est purement numerique
        """
        if not text:
            return True

        # Supprimer les caracteres de formatage numerique
        cleaned = re.sub(r"[\s,.$%€()\-+]", "", text)

        # Verifier si ce qui reste est vide ou numerique
        if not cleaned:
            return True

        try:
            float(cleaned)
            return True
        except ValueError:
            return False


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def compare_tables_structural(
    tables_t1: list[dict[str, Any]],
    tables_t2: list[dict[str, Any]],
    bank_code: str | None = None,
    pdf_t1_path: str | Path | None = None,
    pdf_t2_path: str | Path | None = None,
    visual_proofs_dir: str | Path | None = None,
) -> StructuralComparisonResult:
    """
    Fonction utilitaire pour comparer les tableaux en mode structurel.

    Args:
        tables_t1: Tableaux du rapport T1
        tables_t2: Tableaux du rapport T2
        bank_code: Code de la banque (optionnel)
        pdf_t1_path: Chemin PDF T1 pour preuves visuelles (optionnel)
        pdf_t2_path: Chemin PDF T2 pour preuves visuelles (optionnel)
        visual_proofs_dir: Repertoire sortie PNG (optionnel, si fourni active les preuves)

    Returns:
        StructuralComparisonResult avec les changements detectes
    """
    comparator = StructuralTableComparator(bank_code=bank_code)
    return comparator.compare_tables(
        tables_t1,
        tables_t2,
        bank_code,
        pdf_t1_path=pdf_t1_path,
        pdf_t2_path=pdf_t2_path,
        visual_proofs_dir=visual_proofs_dir,
    )


def get_structural_changes_summary(result: StructuralComparisonResult) -> str:
    """
    Generer un resume textuel des changements structurels.

    Args:
        result: Resultat de la comparaison structurelle

    Returns:
        Resume en texte
    """
    lines = []
    lines.append(f"=== Comparaison Structurelle ({result.comparison_date}) ===")
    lines.append("")

    # Resume global
    summary = result.to_dict()["summary"]
    lines.append(f"Tableaux analyses: {summary['total_tables_compared']}")
    lines.append(f"Tableaux avec changements: {summary['tables_with_changes']}")
    lines.append(f"Total lignes ajoutees: {summary['total_rows_added']}")
    lines.append(f"Total lignes supprimees: {summary['total_rows_removed']}")
    lines.append(f"Total lignes renommees (GenAI): {summary['total_rows_renamed']}")
    lines.append(f"Tableaux entiers ajoutes: {summary['total_tables_added']}")
    lines.append(f"Tableaux entiers supprimes: {summary['total_tables_removed']}")
    lines.append(f"Matchs forts: {summary.get('total_tables_matched_strong', 0)}")
    lines.append(
        f"Matchs probables (revue): {summary.get('total_tables_matched_probable', 0)}"
    )
    lines.append("")

    # Details par tableau
    if result.tables_with_changes:
        lines.append("--- Changements par tableau ---")
        for tc in result.tables_with_changes:
            table_label = tc.table_title
            if tc.table_number:
                table_label = (
                    f"Tableau {tc.table_number} - {table_label}"
                    if table_label
                    else f"Tableau {tc.table_number}"
                )
            lines.append(f"\n[{table_label}]")
            if tc.rows_added:
                lines.append(f"  + Lignes ajoutees ({len(tc.rows_added)}):")
                for row in tc.rows_added[:5]:  # Max 5 exemples
                    lines.append(f"    - {row}")
                if len(tc.rows_added) > 5:
                    lines.append(f"    ... et {len(tc.rows_added) - 5} autres")
            if tc.rows_removed:
                lines.append(f"  - Lignes supprimees ({len(tc.rows_removed)}):")
                for row in tc.rows_removed[:5]:  # Max 5 exemples
                    lines.append(f"    - {row}")
                if len(tc.rows_removed) > 5:
                    lines.append(f"    ... et {len(tc.rows_removed) - 5} autres")
            if tc.rows_renamed:
                lines.append(f"  ~ Lignes renommees ({len(tc.rows_renamed)}):")
                for ren in tc.rows_renamed[:5]:
                    lines.append(f"    - '{ren['from']}' -> '{ren['to']}'")
                if len(tc.rows_renamed) > 5:
                    lines.append(f"    ... et {len(tc.rows_renamed) - 5} autres")

    # Tableaux entiers ajoutes/supprimes
    def get_table_title(table: Any) -> str:
        if isinstance(table, str):
            return table
        return (
            table.get("title", "")
            or table.get("table_title", "")
            or table.get("name", "")
            or ""
        )

    def get_table_number(table: Any) -> str | None:
        if isinstance(table, dict):
            return table.get("table_number")
        return None

    if result.tables_added:
        lines.append("\n--- Tableaux entiers ajoutes ---")
        for table in result.tables_added:
            title = get_table_title(table)
            table_number = get_table_number(table)
            page = table.get("page_number", 0) if isinstance(table, dict) else 0
            if table_number:
                title = (
                    f"Tableau {table_number} - {title}"
                    if title
                    else f"Tableau {table_number}"
                )
            lines.append(f"  + {title} (page {page})")

    if result.tables_removed:
        lines.append("\n--- Tableaux entiers supprimes ---")
        for table in result.tables_removed:
            title = get_table_title(table)
            table_number = get_table_number(table)
            page = table.get("page_number", 0) if isinstance(table, dict) else 0
            if table_number:
                title = (
                    f"Tableau {table_number} - {title}"
                    if title
                    else f"Tableau {table_number}"
                )
            lines.append(f"  - {title} (page {page})")

    return "\n".join(lines)


# =============================================================================
# ANALYSEUR GenAI POUR CHANGEMENTS STRUCTURELS
# =============================================================================


class StructuralChangeAnalyzer:
    """
    Analyse les changements structurels avec GenAI pour determiner
    si chaque changement est une "nouvelle idee" avec justification.
    """

    # Prompt pour l'analyse GenAI
    ANALYSIS_PROMPT = """Analyse ce changement structurel detecte dans un tableau de rapport bancaire trimestriel.

Indicateur: {phrase}
Type de changement: {change_type}
Section: {section}
Tableau: {table_title}
Page: {page}

Contexte: Ce changement a ete detecte en comparant deux rapports trimestriels consecutifs (T1 vs T2).
- "ajoute" = l'indicateur est present dans T2 mais absent de T1
- "supprime" = l'indicateur est present dans T1 mais absent de T2

REGLE IMPORTANTE: En cas de doute, prefere TOUJOURS "non_pertinent". Il vaut mieux manquer un changement que generer un faux positif.

Exemples de "non_pertinent" (a exclure systématiquement):
- Reformulation du meme concept: "Ratio de levier" vs "Ratio du levier"
- Deplacement de contenu sans changement semantique
- Variation mineure de libelle: tirets, abreviations, ponctuation ("CET1" vs "CET-1")
- Reorganisation de tableaux sans nouvel indicateur
- Changement de presentation (ordre, formatage)
- Ligne de total/sous-total deja presente sous autre forme

Criteria pour "nouvelle_idee" (reserve aux vrais changements):
- Nouveau concept reglementaire (ex: nouvelle exigence Bale IV, TLAC)
- Nouvelle categorie d'exposition ou de risque
- Nouvelle divulgation obligatoire (BSIF, AMF, OSFI)
- Nouveau risque emergent (IA, cyber, climat) introduit explicitement

Criteria pour "pertinent" (changement reel mais pas vraiment nouveau):
- Indicateur existant deplace vers un autre tableau avec sens conserve
- Modification de nomenclature significative mais sans nouveau concept

Determine:
1. Est-ce une "Nouvelle idee"? Oui ou Non
2. Justification: Explique en 1-2 phrases
3. Pertinence: "nouvelle_idee" ou "pertinent" ou "non_pertinent"

Reponds UNIQUEMENT en JSON valide (sans markdown):
{{"nouvelle_idee": "Oui" ou "Non", "justification": "...", "pertinence": "nouvelle_idee" ou "pertinent" ou "non_pertinent"}}"""

    # Prompt pour verification (2e appel - reduction faux positifs)
    VERIFICATION_PROMPT = """OBJECTIF: Comparer deux rapports trimestriels (T1 vs T2) en se concentrant sur la premiere colonne des tableaux (indicateurs). Detecter UNIQUEMENT les changements structurels reels. Priorite: reduire les faux positifs.

ANALYSE PRECEDENTE a verifier:
- Indicateur: "{phrase}"
- Type: {change_type}
- Classification: pertinence={pertinence}, nouvelle_idee={nouvelle_idee}
- Justification: {justification}

IGNORER (non_pertinent): notes de bas de page, numeros isoles, ponctuation, variations purement numeriques, deplacements de position, reformulations mineures ("Ratio de levier" vs "Ratio du levier"), variantes (CET1 vs CET-1).

QUESTION: Cette classification est-elle correcte? En cas de DOUTE, reclasse en non_pertinent.

Reponds UNIQUEMENT en JSON:
{{"downgrade_to_non_pertinent": true ou false, "reasoning": "..."}}"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        use_genai: bool = True,
        use_content_filter: bool = True,
        use_noise_filter: bool = True,
        use_verification: bool = True,
        genai_max_workers: int = 6,
    ):
        """
        Initialiser l'analyseur.

        Args:
            api_key: Cle API OpenAI (ou variable OPENAI_API_KEY)
            model: Modele a utiliser (gpt-4o, gpt-4o-mini)
            use_genai: Utiliser GenAI ou fallback sur regles simples
            use_content_filter: Appliquer ContentFilter avant GenAI
            use_noise_filter: Appliquer NoiseFilter avant GenAI
            use_verification: 2e appel GenAI pour verifier pertinent/nouvelle_idee (reduction faux positifs)
            genai_max_workers: Nombre max de workers pour parallellisation GenAI
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.use_genai = use_genai and bool(self.api_key)
        self.use_content_filter = use_content_filter
        self.use_noise_filter = use_noise_filter
        self.use_verification = use_verification
        self.genai_max_workers = min(genai_max_workers, 8)
        self._client = None
        self._genai_consecutive_failures = 0
        self._genai_circuit_open = False

    def _get_client(self):
        """Obtenir le client OpenAI (lazy loading)."""
        if self._client is None and self.use_genai:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed, falling back to rules")
                self.use_genai = False
        return self._client

    def _get_table_title_from_dict(self, table: Any) -> str:
        """Obtenir le titre d'un tableau depuis un dictionnaire ou string."""
        if isinstance(table, str):
            return table
        return (
            table.get("title", "")
            or table.get("table_title", "")
            or table.get("name", "")
            or ""
        )

    def _get_table_number_from_dict(self, table: Any) -> str | None:
        """Obtenir le numero d'un tableau depuis un dictionnaire."""
        if isinstance(table, dict):
            return table.get("table_number")
        return None

    def analyze_structural_result(
        self, result: StructuralComparisonResult, section_type: str | None = None
    ) -> AnalyzedComparisonResult:
        """
        Analyser tous les changements structurels d'un resultat.

        Args:
            result: Resultat de comparaison structurelle brut
            section_type: Type de section (optionnel)

        Returns:
            AnalyzedComparisonResult avec analyse GenAI
        """
        analyzed_result = AnalyzedComparisonResult(
            comparison_date=result.comparison_date, bank_code=result.bank_code
        )

        # Collecter tous les changements
        raw_changes = []

        # Changements par tableau
        for tc in result.tables_with_changes:
            section = section_type or "unknown_section"
            section_label = SECTION_LABELS.get(section, section)
            page = tc.page_t2 or tc.page_t1 or 0
            table_number = tc.table_number
            table_title_display = tc.table_title
            if table_number:
                table_title_display = (
                    f"Tableau {table_number} - {table_title_display}"
                    if table_title_display
                    else f"Tableau {table_number}"
                )

            for indicator in tc.rows_added:
                raw_changes.append(
                    {
                        "phrase": indicator,
                        "change_type": "ajoute",
                        "section": section_label,
                        "table_title": table_title_display,
                        "table_number": table_number,
                        "page": page,
                    }
                )

            for indicator in tc.rows_removed:
                raw_changes.append(
                    {
                        "phrase": indicator,
                        "change_type": "supprime",
                        "section": section_label,
                        "table_title": table_title_display,
                        "table_number": table_number,
                        "page": page,
                    }
                )

            for disp in getattr(tc, "rows_displaced", []) or []:
                text = disp.get("text_display", disp.get("canonical", ""))
                from_p = disp.get("from_page", "")
                to_p = disp.get("to_page", "")
                raw_changes.append(
                    {
                        "phrase": f"{text} (deplace page {from_p} -> page {to_p})",
                        "change_type": "deplacement",
                        "section": section_label,
                        "table_title": table_title_display,
                        "table_number": table_number,
                        "page": to_p or page,
                    }
                )

        # Tableaux entiers ajoutes
        for table in result.tables_added:
            # Extraire le titre et la page du tableau
            title = self._get_table_title_from_dict(table)
            table_number = self._get_table_number_from_dict(table)
            page = table.get("page_number", 0) if isinstance(table, dict) else 0
            section_label = SECTION_LABELS.get(
                section_type or "unknown_section", "unknown_section"
            )
            if table_number:
                title = (
                    f"Tableau {table_number} - {title}"
                    if title
                    else f"Tableau {table_number}"
                )
            raw_changes.append(
                {
                    "phrase": _format_whole_table_phrase(
                        change_type="ajoute",
                        section_label=section_label,
                        table_title=title,
                        page=page,
                        page_source="T2",
                    ),
                    "change_type": "ajoute",
                    "section": section_label,
                    "table_title": title,
                    "table_number": table_number,
                    "page": page,
                }
            )

        # Tableaux entiers supprimes
        for table in result.tables_removed:
            # Extraire le titre et la page du tableau
            title = self._get_table_title_from_dict(table)
            table_number = self._get_table_number_from_dict(table)
            page = table.get("page_number", 0) if isinstance(table, dict) else 0
            section_label = SECTION_LABELS.get(
                section_type or "unknown_section", "unknown_section"
            )
            if table_number:
                title = (
                    f"Tableau {table_number} - {title}"
                    if title
                    else f"Tableau {table_number}"
                )
            raw_changes.append(
                {
                    "phrase": _format_whole_table_phrase(
                        change_type="supprime",
                        section_label=section_label,
                        table_title=title,
                        page=page,
                        page_source="T1",
                    ),
                    "change_type": "supprime",
                    "section": section_label,
                    "table_title": title,
                    "table_number": table_number,
                    "page": page,
                }
            )

        # Pre-filtrage avant GenAI
        changes_to_analyze = raw_changes
        filter_changes_fn = _get_filter_changes()
        if self.use_content_filter and filter_changes_fn:
            to_analyze, excluded, filter_stats = filter_changes_fn(raw_changes)
            for row in excluded:
                filter_result = row.get("filter_result")
                reason = (
                    getattr(filter_result, "exclusion_reason", None)
                    if filter_result
                    else None
                ) or "Regle ContentFilter"
                analyzed_result.changes.append(
                    self._make_non_pertinent_change(row, f"Exclu par regle: {reason}")
                )
            changes_to_analyze = to_analyze
            logger.info(
                f"ContentFilter: {filter_stats.get('excluded', 0)} exclus, "
                f"{filter_stats.get('to_analyze', 0)} a analyser"
            )

        if self.use_noise_filter:
            noise_filter = NoiseFilter()
            to_genai = []
            for change in changes_to_analyze:
                result = noise_filter.is_noise(change)
                if result.is_noise:
                    analyzed_result.changes.append(
                        self._make_non_pertinent_change(
                            change, f"Bruit detecte: {result.reason or 'noise_filter'}"
                        )
                    )
                else:
                    to_genai.append(change)
            changes_to_analyze = to_genai

        # Analyser chaque changement restant (parallellise)
        max_workers = min(self.genai_max_workers, len(changes_to_analyze), 8)
        if max_workers <= 1 or len(changes_to_analyze) <= 1:
            for i, change in enumerate(changes_to_analyze):
                logger.info(
                    f"Analyse changement {i + 1}/{len(changes_to_analyze)}: "
                    f"{change['phrase'][:50]}..."
                )
                analyzed = self._analyze_single_change(change)
                analyzed_result.changes.append(analyzed)
        else:
            results_by_idx = [None] * len(changes_to_analyze)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_single_change, change): i
                    for i, change in enumerate(changes_to_analyze)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        analyzed = future.result()
                        results_by_idx[idx] = analyzed
                    except Exception as e:
                        logger.warning(f"Erreur analyse changement {idx}: {e}")
                        results_by_idx[idx] = self._analyze_with_rules(
                            changes_to_analyze[idx]
                        )
            for analyzed in results_by_idx:
                if analyzed is not None:
                    analyzed_result.changes.append(analyzed)

        analyzed_result.changes = _sort_analyzed_changes(analyzed_result.changes)

        logger.info(
            f"Analyse terminee: {len(analyzed_result.changes)} changements, "
            f"{sum(1 for c in analyzed_result.changes if c.nouvelle_idee == 'Oui')} nouvelles idees"
        )

        return analyzed_result

    def _make_non_pertinent_change(
        self, change: dict[str, Any], justification: str
    ) -> StructuralChangeAnalyzed:
        """Creer un StructuralChangeAnalyzed marque non_pertinent sans appel GenAI."""
        from vigilance.utils.type_metier import compute_type_metier

        return StructuralChangeAnalyzed(
            titre=change.get("section", ""),
            page=change.get("page", 0),
            phrase=change.get("phrase", ""),
            change_type=change.get("change_type", ""),
            nouvelle_idee="Non",
            justification=justification,
            pertinence="non_pertinent",
            table_title=change.get("table_title"),
            table_number=change.get("table_number"),
            page_t1=change.get("page_t1"),
            page_t2=change.get("page_t2"),
            type_metier=compute_type_metier(
                change.get("section"), change.get("change_type")
            ),
        )

    def _analyze_single_change(
        self, change: dict[str, Any]
    ) -> StructuralChangeAnalyzed:
        """
        Analyser un seul changement avec GenAI ou regles.

        Args:
            change: Dictionnaire avec phrase, change_type, section, etc.

        Returns:
            StructuralChangeAnalyzed avec nouvelle_idee et justification
        """
        if change.get("change_type") == "deplacement":
            from vigilance.utils.type_metier import compute_type_metier

            return StructuralChangeAnalyzed(
                titre=change.get("section", ""),
                page=change.get("page", 0),
                phrase=change.get("phrase", ""),
                change_type="deplacement",
                nouvelle_idee="Non",
                justification="Deplacement de contenu sans modification (reorganisation)",
                pertinence="non_pertinent",
                table_title=change.get("table_title"),
                table_number=change.get("table_number"),
                page_t1=change.get("page_t1"),
                page_t2=change.get("page_t2"),
                type_metier=compute_type_metier(change.get("section"), "deplacement"),
            )
        if change.get("change_type") == "renomme":
            from vigilance.utils.type_metier import compute_type_metier

            return StructuralChangeAnalyzed(
                titre=change.get("section", ""),
                page=change.get("page", 0),
                phrase=change.get("phrase", ""),
                change_type="renomme",
                nouvelle_idee="Non",
                justification="Renommage d'indicateur sans changement de contenu",
                pertinence="pertinent",
                table_title=change.get("table_title"),
                table_number=change.get("table_number"),
                page_t1=change.get("page_t1"),
                page_t2=change.get("page_t2"),
                type_metier=compute_type_metier(change.get("section"), "renomme"),
            )
        if self.use_genai:
            return self._analyze_with_genai(change)
        else:
            return self._analyze_with_rules(change)

    def _analyze_with_genai(self, change: dict[str, Any]) -> StructuralChangeAnalyzed:
        """Analyser avec GenAI (GPT-4)."""
        # Circuit-breaker : apres 3 echecs consecutifs, skip GenAI
        if self._genai_circuit_open:
            return self._analyze_with_rules(change)

        client = self._get_client()

        if not client:
            return self._analyze_with_rules(change)

        try:
            prompt = self.ANALYSIS_PROMPT.format(
                phrase=change["phrase"],
                change_type=change["change_type"],
                section=change["section"],
                table_title=change.get("table_title", "N/A"),
                page=change.get("page", 0),
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un analyste specialise dans les rapports bancaires reglementaires. Reponds uniquement en JSON valide.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=300,
            )

            # Parser la reponse JSON
            response_text = response.choices[0].message.content.strip()

            # Nettoyer la reponse (enlever markdown si present)
            if response_text.startswith("```"):
                response_text = re.sub(r"^```json?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

            analysis = json.loads(response_text)

            # Succes → reset circuit-breaker
            self._genai_consecutive_failures = 0

            from vigilance.utils.type_metier import compute_type_metier

            pertinence = analysis.get("pertinence", "pertinent")
            nouvelle_idee = analysis.get("nouvelle_idee", "Non")
            justification = analysis.get("justification", "")

            # Verification 2e etape: reduire faux positifs
            if self.use_verification and pertinence in ("pertinent", "nouvelle_idee"):
                downgrade, verify_reason = self._verify_structural_analysis(
                    change, pertinence, nouvelle_idee, justification
                )
                if downgrade:
                    pertinence = "non_pertinent"
                    nouvelle_idee = "Non"
                    justification = verify_reason or justification
                    logger.debug(
                        f"Verification: downgrade non_pertinent - {change['phrase'][:40]}..."
                    )

            return StructuralChangeAnalyzed(
                titre=change["section"],
                page=change.get("page", 0),
                phrase=change["phrase"],
                change_type=change["change_type"],
                nouvelle_idee=nouvelle_idee,
                justification=justification,
                pertinence=pertinence,
                table_title=change.get("table_title"),
                table_number=change.get("table_number"),
                page_t1=change.get("page_t1"),
                page_t2=change.get("page_t2"),
                type_metier=compute_type_metier(
                    change.get("section"), change.get("change_type")
                ),
            )

        except Exception as e:
            self._genai_consecutive_failures += 1
            if self._genai_consecutive_failures >= 3 and not self._genai_circuit_open:
                self._genai_circuit_open = True
                logger.warning(
                    f"Circuit-breaker GenAI ouvert apres {self._genai_consecutive_failures} "
                    f"echecs consecutifs. Fallback sur regles pour le reste de la session."
                )
            else:
                logger.warning(f"Erreur GenAI: {e}, fallback sur regles")
            return self._analyze_with_rules(change)

    def _verify_structural_analysis(
        self,
        change: dict[str, Any],
        pertinence: str,
        nouvelle_idee: str,
        justification: str,
    ) -> tuple[bool, str | None]:
        """
        Verifier une analyse avec un 2e appel GenAI. Peut downgrader en non_pertinent.

        Returns:
            (downgrade_to_non_pertinent, reason_if_downgraded)
        """
        client = self._get_client()
        if not client:
            return False, None

        try:
            prompt = self.VERIFICATION_PROMPT.format(
                phrase=change.get("phrase", ""),
                change_type=change.get("change_type", ""),
                pertinence=pertinence,
                nouvelle_idee=nouvelle_idee,
                justification=justification,
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un verificateur strict. En cas de doute, reclasse en non_pertinent. Reponds uniquement en JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=150,
            )

            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```json?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            data = json.loads(text)
            downgrade = data.get("downgrade_to_non_pertinent", False)
            reason = data.get("reasoning", "") if downgrade else None
            return bool(downgrade), reason

        except Exception as e:
            logger.debug(f"Verification echouee ({e}), conserver classification")
            return False, None

    def _analyze_with_rules(self, change: dict[str, Any]) -> StructuralChangeAnalyzed:
        """Analyser avec regles simples (fallback)."""
        phrase = change["phrase"].lower()
        change_type = change["change_type"]

        # Mots-cles pour nouvelles idees
        nouvelle_idee_keywords = [
            "esg",
            "climat",
            "carbone",
            "environnement",
            "ia",
            "intelligence artificielle",
            "cyber",
            "bsif",
            "bale",
            "basel",
            "reglementaire",
            "nouveau",
            "nouvelle",
            "emergent",
        ]

        # Verifier si c'est une nouvelle idee
        is_nouvelle_idee = any(kw in phrase for kw in nouvelle_idee_keywords)

        # Par defaut: ajout = nouvelle idee, suppression = non
        if change_type == "ajoute":
            if is_nouvelle_idee:
                nouvelle_idee = "Oui"
                pertinence = "nouvelle_idee"
                justification = "Nouvel indicateur potentiellement lie a une exigence reglementaire ou risque emergent"
            else:
                nouvelle_idee = "Oui"
                pertinence = "pertinent"
                justification = "Nouvel indicateur ajoute dans le rapport"
        else:  # supprime
            nouvelle_idee = "Non"
            pertinence = "pertinent"
            justification = "Indicateur supprime du rapport"

        from vigilance.utils.type_metier import compute_type_metier

        return StructuralChangeAnalyzed(
            titre=change["section"],
            page=change.get("page", 0),
            phrase=change["phrase"],
            change_type=change_type,
            nouvelle_idee=nouvelle_idee,
            justification=justification,
            pertinence=pertinence,
            table_title=change.get("table_title"),
            table_number=change.get("table_number"),
            page_t1=change.get("page_t1"),
            page_t2=change.get("page_t2"),
            type_metier=compute_type_metier(change.get("section"), change_type),
        )


# =============================================================================
# UTILITAIRES POUR VALIDATION HUMAINE
# =============================================================================


def group_changes_by_table(
    changes: list[StructuralChangeAnalyzed | dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Regrouper les changements par paire de tableaux (titre, table_title, table_number).

    Pour chaque groupe: indicateurs_ajoutes, indicateurs_supprimes, deplacements.
    Les indicateurs_renommes sont vides dans cette iteration (phase 2).

    Args:
        changes: Liste de StructuralChangeAnalyzed ou dicts

    Returns:
        Liste de dicts avec section, table_title, table_number, page_t1, page_t2,
        indicateurs_ajoutes, indicateurs_supprimes, indicateurs_renommes, deplacements
    """
    groups: dict[tuple, dict[str, Any]] = {}

    for change in changes:
        if isinstance(change, StructuralChangeAnalyzed):
            d = change.to_dict()
        else:
            d = change

        key = (
            d.get("titre") or "",
            d.get("table_title") or "",
            d.get("table_number") or "",
        )
        if key not in groups:
            groups[key] = {
                "section": key[0],
                "table_title": key[1],
                "table_number": key[2],
                "page_t1": d.get("page_t1"),
                "page_t2": d.get("page_t2"),
                "indicateurs_ajoutes": [],
                "indicateurs_supprimes": [],
                "indicateurs_renommes": [],
                "deplacements": [],
            }

        g = groups[key]
        phrase = d.get("phrase", "").strip()
        if not phrase:
            continue

        # Deduire page_t1/page_t2 du premier changement du groupe si absents
        if g["page_t1"] is None and d.get("page_t1") is not None:
            g["page_t1"] = d.get("page_t1")
        if g["page_t2"] is None and d.get("page_t2") is not None:
            g["page_t2"] = d.get("page_t2")

        ct = d.get("change_type", "")
        if ct == "ajoute":
            if phrase not in g["indicateurs_ajoutes"]:
                g["indicateurs_ajoutes"].append(phrase)
        elif ct == "supprime":
            if phrase not in g["indicateurs_supprimes"]:
                g["indicateurs_supprimes"].append(phrase)
        elif ct == "deplacement":
            depl = {
                "indicateur": phrase,
                "from_page": d.get("page_t1"),
                "to_page": d.get("page_t2"),
            }
            g["deplacements"].append(depl)
        elif ct == "renomme":
            if phrase not in g["indicateurs_renommes"]:
                g["indicateurs_renommes"].append(phrase)

    return list(groups.values())


def group_changes_by_page(
    changes: list[StructuralChangeAnalyzed | dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """
    Grouper les changements par numero de page.

    Args:
        changes: Liste de StructuralChangeAnalyzed ou dicts

    Returns:
        Dict[page_number, List[changes]]
    """
    changes_by_page: dict[int, list[dict[str, Any]]] = {}

    for change in changes:
        # Convertir en dict si c'est un objet
        if isinstance(change, StructuralChangeAnalyzed):
            change_dict = change.to_dict()
        else:
            change_dict = change

        page = change_dict.get("page", 0)
        if page > 0:
            if page not in changes_by_page:
                changes_by_page[page] = []
            changes_by_page[page].append(change_dict)

    return changes_by_page


# =============================================================================
# GENERATEUR DE RAPPORT MARKDOWN
# =============================================================================


def generate_markdown_report(result: AnalyzedComparisonResult) -> str:
    """
    Generer un rapport Markdown avec legende et tableau formate.

    Args:
        result: Resultat analyse avec GenAI

    Returns:
        Rapport en format Markdown
    """
    lines = []

    # Legende
    lines.append("## Legende")
    lines.append("- Vert : Pertinent, a garder")
    lines.append("- Rouge : Non pertinent, a retirer")
    lines.append("- Jaune : Nouvelle idee")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Tableau: Titre | Type metier | Page | Phrase | Nouvelle idee | Justification
    lines.append("## Tableau - Analyse")
    lines.append("")
    lines.append(
        "| Titre | Type metier | Page | Phrase | Nouvelle idee | Justification |"
    )
    lines.append(
        "|-------|-------------|------|--------|---------------|---------------|"
    )

    for change in result.changes:
        from vigilance.utils.type_metier import compute_type_metier

        type_metier = change.type_metier or compute_type_metier(
            change.titre, change.change_type
        )
        phrase = change.phrase.replace("|", "\\|")
        justification = change.justification.replace("|", "\\|")
        lines.append(
            f"| {change.titre} | {type_metier} | {change.page} | {phrase} | "
            f"{change.nouvelle_idee} | {justification} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Resume
    summary = result.to_dict()["summary"]
    lines.append("## Resume")
    lines.append(f"- Total changements: {summary['total']}")
    lines.append(f"- Nouvelles idees: {summary['nouvelles_idees']}")
    lines.append(f"- Pertinents: {summary['pertinents']}")
    lines.append(f"- Non pertinents: {summary['non_pertinents']}")
    lines.append(f"- Ajouts: {summary['ajouts']}")
    lines.append(f"- Suppressions: {summary['suppressions']}")

    return "\n".join(lines)


def analyze_and_format_structural_changes(
    result: StructuralComparisonResult,
    api_key: str | None = None,
    use_genai: bool = True,
    section_type: str | None = None,
) -> AnalyzedComparisonResult:
    """
    Analyser les changements structurels et retourner le resultat formate.

    Args:
        result: Resultat de comparaison structurelle brut
        api_key: Cle API OpenAI (optionnel)
        use_genai: Utiliser GenAI pour l'analyse
        section_type: Type de section

    Returns:
        AnalyzedComparisonResult avec analyse complete
    """
    analyzer = StructuralChangeAnalyzer(api_key=api_key, use_genai=use_genai)
    return analyzer.analyze_structural_result(result, section_type)


def _format_whole_table_phrase(
    change_type: str,
    section_label: str,
    table_title: str,
    page: int,
    page_source: str,
) -> str:
    """Construire une phrase explicite pour ajout/suppression de tableau entier."""
    action = "ajouté" if change_type == "ajoute" else "supprimé"
    title_display = (
        table_title.strip()
        if table_title and table_title.strip()
        else "(titre non disponible)"
    )
    page_display = str(page) if isinstance(page, int) and page > 0 else "non disponible"
    return (
        f"[Tableau entier {action}] "
        f"Section: {section_label} | "
        f"Titre: {title_display} | "
        f"Page {page_source}: {page_display}"
    )


def _sort_analyzed_changes(
    changes: list[StructuralChangeAnalyzed],
) -> list[StructuralChangeAnalyzed]:
    """Trier les changements pour revue/export: nouvelle idée Oui en premier."""

    def sort_key(change: StructuralChangeAnalyzed) -> tuple:
        nouvelle_rank = 0 if str(change.nouvelle_idee).strip().lower() == "oui" else 1
        change_rank = (
            0
            if change.change_type == "ajoute"
            else (2 if change.change_type == "deplacement" else 1)
        )
        section = (change.titre or "").lower()
        page = (
            change.page if isinstance(change.page, int) and change.page > 0 else 10**9
        )
        phrase = (change.phrase or "").lower()
        return (nouvelle_rank, change_rank, section, page, phrase)

    return sorted(changes, key=sort_key)


def analyze_and_format_structural_changes_multi_section(
    result: StructuralComparisonResult,
    page_to_section_t1: dict[int, str],
    page_to_section_t2: dict[int, str],
    api_key: str | None = None,
    use_genai: bool = True,
) -> AnalyzedComparisonResult:
    """
    Analyser les changements structurels avec sections determinees par page.

    Cette fonction determine la section (Gestion du capital, Gestion des risques)
    de chaque tableau en fonction de son numero de page.

    Args:
        result: Resultat de comparaison structurelle brut
        page_to_section_t1: Mapping page -> section pour T1
        page_to_section_t2: Mapping page -> section pour T2
        api_key: Cle API OpenAI (optionnel)
        use_genai: Utiliser GenAI pour l'analyse

    Returns:
        AnalyzedComparisonResult avec sections correctes
    """
    analyzer = StructuralChangeAnalyzer(api_key=api_key, use_genai=use_genai)

    analyzed_result = AnalyzedComparisonResult(
        comparison_date=result.comparison_date, bank_code=result.bank_code
    )
    analyzed_result.table_matching = result.to_dict().get("table_matching", {})

    # Collecter tous les changements avec section correcte
    raw_changes = []

    def get_table_number(table: Any) -> str | None:
        if isinstance(table, dict):
            return table.get("table_number")
        return None

    # Changements par tableau
    for tc in result.tables_with_changes:
        page_t1 = tc.page_t1 or 0
        page_t2 = tc.page_t2 or 0

        # Determiner la section basee sur la page
        section_type = (
            page_to_section_t2.get(page_t2)
            or page_to_section_t1.get(page_t1)
            or "unknown_section"
        )
        section_label = SECTION_LABELS.get(section_type, section_type)
        table_number = tc.table_number
        table_title_display = tc.table_title
        if table_number:
            table_title_display = (
                f"Tableau {table_number} - {table_title_display}"
                if table_title_display
                else f"Tableau {table_number}"
            )

        for indicator in tc.rows_added:
            raw_changes.append(
                {
                    "phrase": indicator,
                    "change_type": "ajoute",
                    "section": section_label,
                    "table_title": table_title_display,
                    "table_number": table_number,
                    "page": page_t2 or page_t1,
                    "page_t1": page_t1,
                    "page_t2": page_t2,
                }
            )

        for indicator in tc.rows_removed:
            raw_changes.append(
                {
                    "phrase": indicator,
                    "change_type": "supprime",
                    "section": section_label,
                    "table_title": table_title_display,
                    "table_number": table_number,
                    "page": page_t1 or page_t2,
                    "page_t1": page_t1,
                    "page_t2": page_t2,
                }
            )

        for disp in getattr(tc, "rows_displaced", []) or []:
            text = disp.get("text_display", disp.get("canonical", ""))
            from_p = disp.get("from_page", "")
            to_p = disp.get("to_page", "")
            raw_changes.append(
                {
                    "phrase": f"{text} (deplace page {from_p} -> page {to_p})",
                    "change_type": "deplacement",
                    "section": section_label,
                    "table_title": table_title_display,
                    "table_number": table_number,
                    "page": to_p or page_t2 or page_t1,
                    "page_t1": page_t1,
                    "page_t2": page_t2,
                }
            )

        for ren in getattr(tc, "rows_renamed", []) or []:
            from_val = ren.get("from", ren.get("from_indicator", ""))
            to_val = ren.get("to", ren.get("to_indicator", ""))
            phrase = f"{from_val} -> {to_val}" if from_val or to_val else str(ren)
            raw_changes.append(
                {
                    "phrase": phrase,
                    "change_type": "renomme",
                    "section": section_label,
                    "table_title": table_title_display,
                    "table_number": table_number,
                    "page": page_t2 or page_t1,
                    "page_t1": page_t1,
                    "page_t2": page_t2,
                }
            )

    # Helper pour extraire le titre d'un tableau
    def get_table_title(table: Any) -> str:
        if isinstance(table, str):
            return table
        return (
            table.get("title", "")
            or table.get("table_title", "")
            or table.get("name", "")
            or ""
        )

    # Tableaux entiers ajoutes (nouveaux dans T2)
    for table in result.tables_added:
        title = get_table_title(table)
        table_number = get_table_number(table)
        page = table.get("page_number", 0) if isinstance(table, dict) else 0

        # Determiner section depuis le mapping page -> section
        section_type = page_to_section_t2.get(page) or "unknown_section"
        section_label = SECTION_LABELS.get(section_type, "unknown_section")

        if table_number:
            title = (
                f"Tableau {table_number} - {title}"
                if title
                else f"Tableau {table_number}"
            )

        raw_changes.append(
            {
                "phrase": _format_whole_table_phrase(
                    change_type="ajoute",
                    section_label=section_label,
                    table_title=title,
                    page=page,
                    page_source="T2",
                ),
                "change_type": "ajoute",
                "section": section_label,
                "table_title": title,
                "table_number": table_number,
                "page": page,
                "page_t1": 0,
                "page_t2": page,
            }
        )

    # Tableaux entiers supprimes (absents de T2)
    for table in result.tables_removed:
        title = get_table_title(table)
        table_number = get_table_number(table)
        page = table.get("page_number", 0) if isinstance(table, dict) else 0

        # Determiner section depuis le mapping page -> section
        section_type = page_to_section_t1.get(page) or "unknown_section"
        section_label = SECTION_LABELS.get(section_type, "unknown_section")

        if table_number:
            title = (
                f"Tableau {table_number} - {title}"
                if title
                else f"Tableau {table_number}"
            )

        raw_changes.append(
            {
                "phrase": _format_whole_table_phrase(
                    change_type="supprime",
                    section_label=section_label,
                    table_title=title,
                    page=page,
                    page_source="T1",
                ),
                "change_type": "supprime",
                "section": section_label,
                "table_title": title,
                "table_number": table_number,
                "page": page,
                "page_t1": page,
                "page_t2": 0,
            }
        )

    # Pre-filtrage avant GenAI (ContentFilter + NoiseFilter)
    changes_to_analyze = raw_changes
    filter_changes_fn = _get_filter_changes()
    if analyzer.use_content_filter and filter_changes_fn:
        to_analyze, excluded, filter_stats = filter_changes_fn(raw_changes)
        for row in excluded:
            filter_result = row.get("filter_result")
            reason = (
                getattr(filter_result, "exclusion_reason", None)
                if filter_result
                else None
            ) or "Regle ContentFilter"
            analyzed_result.changes.append(
                analyzer._make_non_pertinent_change(row, f"Exclu par regle: {reason}")
            )
        changes_to_analyze = to_analyze
        logger.info(
            f"ContentFilter: {filter_stats.get('excluded', 0)} exclus, "
            f"{filter_stats.get('to_analyze', 0)} a analyser"
        )

    if analyzer.use_noise_filter:
        noise_filter = NoiseFilter()
        to_genai = []
        for change in changes_to_analyze:
            result = noise_filter.is_noise(change)
            if result.is_noise:
                analyzed_result.changes.append(
                    analyzer._make_non_pertinent_change(
                        change, f"Bruit detecte: {result.reason or 'noise_filter'}"
                    )
                )
            else:
                to_genai.append(change)
        changes_to_analyze = to_genai

    # Analyser chaque changement restant
    for i, change in enumerate(changes_to_analyze):
        logger.info(
            f"Analyse changement {i + 1}/{len(changes_to_analyze)}: {change['phrase'][:50]}..."
        )
        analyzed = analyzer._analyze_single_change(change)
        analyzed_result.changes.append(analyzed)

    analyzed_result.changes = _sort_analyzed_changes(analyzed_result.changes)

    logger.info(
        f"Analyse terminee: {len(analyzed_result.changes)} changements, "
        f"{sum(1 for c in analyzed_result.changes if c.nouvelle_idee == 'Oui')} nouvelles idees"
    )

    return analyzed_result
