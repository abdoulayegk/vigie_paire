"""Structures de donnees d'extraction : tableau, section et document.

Extrait de ``docling_processor.py`` sans modification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class ExtractedTable:
    """Représente un tableau extrait d'un PDF avec toutes ses métadonnées.

    Attributs principaux
    --------------------
    table_id:
        Identifiant unique du tableau dans le document (ex. ``"table_3_p12"``).
    page_number:
        Numéro de page (1-indexé) où le tableau a été détecté.
    title:
        Titre du tableau résolu (peut provenir de la légende Docling, d'une
        ligne de texte au-dessus, ou du titre de page).
    headers:
        Liste des en-têtes de colonnes extraits par Docling.
    rows:
        Contenu du tableau sous forme de liste de listes (lignes × colonnes).
    first_column_indicators:
        Indicateurs de la première colonne après normalisation pour la comparaison.
        C'est le "fingerprint" du tableau utilisé pour le matching.
    first_column_indicators_raw:
        Indicateurs bruts avant normalisation (tels que retournés par Vision GPT-4o).
        ``None`` si Vision n'a pas été utilisée.
    footnotes:
        Liste de dictionnaires ``{"ref": str, "text": str}`` pour les notes de
        bas de page détectées.
    section:
        Nom de section canonique assigné selon les plages de pages
        (ex. ``"gestion_capital"``).
    table_number:
        Numéro logique extrait du titre (ex. ``"28"``, ``"5a"``).
    title_clean:
        Titre sans le numéro de tableau ni les unités.
    table_summary:
        Résumé court du sujet métier du tableau généré par GPT.
    title_raw:
        Titre brut avant tout nettoyage ou matching.
    bbox:
        Boîte englobante normalisée ``[left, top, right, bottom]`` dans [0, 1],
        extraite depuis les métadonnées de provenance Docling.
    extraction_method:
        Méthode d'extraction utilisée : ``"docling"`` ou ``"vision_fallback_gpt4o"``.
    debug_metrics:
        Métriques de débogage (nombre de lignes, fusions, statut Vision, etc.).
    extraction_status:
        Etat canonique de l'extraction du tableau: ``ok``, ``rescued``,
        ``suspect_unresolved`` ou ``confirmed_no_table``.
    fragmentation_detected:
        ``True`` si des artefacts de fragmentation ont été détectés lors de
        l'extraction (lignes coupées, cellules fusionnées incorrectement).
    """

    table_id: str
    page_number: int
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    first_column_indicators: list[str] = field(default_factory=list)
    footnotes: list[dict[str, str]] = field(default_factory=list)
    section: str | None = None
    section_phase: int | None = None
    table_number: str | None = None
    title_clean: str | None = None
    table_summary: str | None = None
    title_raw: str | None = None
    unit_context: str | None = None
    title_resolution_method: str | None = None
    context_before: str = ""
    context_after: str = ""
    bbox: list[float] | None = None
    table_index_on_page: int | None = None
    tables_on_page: int | None = None
    bbox_top: float | None = None
    page_local_role: str | None = None
    first_column_indicators_raw: list[str] | None = None
    first_column_indicators_spatial: list[dict[str, Any]] | None = None
    first_column_groups: list[str] | None = None
    hierarchical_indicator_signature: list[str] | None = None
    title_reliability: str | None = None
    debug_metrics: dict[str, Any] = field(default_factory=dict)
    extraction_method: str | None = None
    extraction_status: str = "ok"
    fragmentation_detected: bool = False

    def to_dict(self) -> dict:
        """Convertir le tableau extrait en dictionnaire serialisable.

        Returns:
            Dictionnaire contenant tous les attributs du tableau.
        """
        return asdict(self)


@dataclass
class ExtractedSection:
    """Représente une section extraite d'un PDF avec ses tableaux et son contenu textuel.

    Attributs
    ---------
    section_id:
        Identifiant canonique de la section (ex. ``"gestion_capital"``).
    title:
        Titre de la section tel qu'il apparaît dans le PDF.
    start_page:
        Première page de la section (1-indexé).
    end_page:
        Dernière page de la section (incluse).
    text_content:
        Contenu textuel brut de la section (paragraphes, titres, etc.).
    tables:
        Liste des tableaux détectés dans cette section.
    phase:
        Phase de la section (1, 2, 3) pour les sections multi-phases.
    """

    section_id: str
    title: str
    start_page: int
    end_page: int
    text_content: str
    tables: list[ExtractedTable] = field(default_factory=list)
    phase: int | None = None

    def to_dict(self) -> dict:
        """Convertir la section extraite en dictionnaire serialisable.

        Returns:
            Dictionnaire contenant tous les attributs de la section et ses tableaux.
        """
        result = asdict(self)
        result["tables"] = [t.to_dict() for t in self.tables]
        return result


@dataclass
class ExtractedDocument:
    """Représente un document PDF entièrement extrait avec toutes ses sections et tableaux.

    Attributs
    ---------
    file_path:
        Chemin absolu vers le fichier PDF source.
    bank_code:
        Code identifiant la banque (ex. ``"rbc"``).
    quarter:
        Libellé du trimestre (ex. ``"Q1-2025"``).
    year:
        Année numérique du trimestre.
    total_pages:
        Nombre total de pages du document.
    sections:
        Liste des sections détectées avec leurs tableaux et contenu textuel.
    all_tables:
        Liste plate de tous les tableaux du document (toutes sections confondues),
        pour un accès direct sans parcourir les sections.
    metadata:
        Métadonnées supplémentaires (version Docling, durée d'extraction, etc.).
    """

    file_path: str
    bank_code: str
    quarter: str
    year: int
    total_pages: int
    sections: list[ExtractedSection] = field(default_factory=list)
    all_tables: list[ExtractedTable] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertir le document extrait en dictionnaire serialisable.

        Returns:
            Dictionnaire contenant tous les attributs du document, ses sections
            et ses tableaux.
        """
        return {
            "file_path": self.file_path,
            "bank_code": self.bank_code,
            "quarter": self.quarter,
            "year": self.year,
            "total_pages": self.total_pages,
            "sections": [s.to_dict() for s in self.sections],
            "all_tables": [t.to_dict() for t in self.all_tables],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialiser le document extrait en chaine JSON.

        Args:
            indent: Niveau d'indentation pour le formatage JSON.

        Returns:
            Chaine JSON representant le document complet.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
