"""Modeles de tables utilises par l'extraction, le stockage et la comparaison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Canonical footnote format: [{"id": str, "text": str}, ...]
# Legacy list[str] payloads are normalized at ingestion boundaries.
FootnoteList = list[dict[str, str]]

VISION_CONTENT_SOURCE = "vision_gpt4o"
UNKNOWN_CONTENT_SOURCE = "unknown"
NON_VISION_FATAL_BLOCKERS = frozenset(
    {
        "missing_vision_indicators",
        "non_vision_content_source",
    }
)

# Confidence telemetry thresholds kept for observability/debug only.
EXTRACTION_CONFIDENCE_CERTIFIED_MIN = 0.5
EXTRACTION_CONFIDENCE_REVIEW_MIN = 0.7
RECROP_CERTIFIED_MIN = 0.85
EXTRACTION_CONFIDENCE_REVIEW_FLOOR = 0.35

TABLE_EXTRACTION_STATUS_OK = "ok"
TABLE_EXTRACTION_STATUS_RESCUED = "rescued"
TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED = "suspect_unresolved"
TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE = "confirmed_no_table"
TABLE_EXTRACTION_STATUSES = frozenset(
    {
        TABLE_EXTRACTION_STATUS_OK,
        TABLE_EXTRACTION_STATUS_RESCUED,
        TABLE_EXTRACTION_STATUS_SUSPECT_UNRESOLVED,
        TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE,
    }
)


def normalize_extraction_status(value: Any) -> str:
    """Retourner une valeur canonique d'extraction_status."""
    status = str(value or "").strip().lower()
    if status in TABLE_EXTRACTION_STATUSES:
        return status
    return TABLE_EXTRACTION_STATUS_OK


def infer_content_source(
    extraction_method: str | None, explicit: str | None = None
) -> str:
    """Inferer la source de contenu depuis la methode d'extraction ou une surcharge explicite.

    Args:
        extraction_method: Chaine de la methode d'extraction (ex. "vision_full_gpt4o").
        explicit: Surcharge explicite optionnelle de la source de contenu ; prioritaire.

    Returns:
        Chaine de la source de contenu : VISION_CONTENT_SOURCE pour l'extraction
        par Vision, UNKNOWN_CONTENT_SOURCE sinon.
    """
    value = str(explicit or "").strip()
    if value:
        return value

    method = str(extraction_method or "").strip().lower()
    if method == "vision_full_gpt4o":
        return VISION_CONTENT_SOURCE
    if method.startswith("vision_"):
        return VISION_CONTENT_SOURCE
    return UNKNOWN_CONTENT_SOURCE


def derive_comparison_blockers(
    *,
    extraction_status: str,
) -> list[str]:
    """Deriver les codes bloquants de matching depuis l'extraction_status canonique.

    Seul ``confirmed_no_table`` bloque l'inclusion dans le matching GPT.
    ``suspect_unresolved`` n'est pas bloquant ici ; la qualite est signalee
    via ``extraction_status`` et le portail qualite.

    Args:
        extraction_status: Valeur canonique du statut d'extraction.

    Returns:
        Liste non vide uniquement pour ``confirmed_no_table`` ; sinon vide.
    """
    status = normalize_extraction_status(extraction_status)
    if status == TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE:
        return [TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE]
    return []


def is_comparison_eligible(
    extraction_status: str,
) -> bool:
    """Determiner si la table est eligible pour le matching de rapports (cartes GPT).

    Eligible pour tous les statuts sauf ``confirmed_no_table``. Les tables
    suspectes participent au matching ; ``extraction_status`` reste l'indicateur
    de qualite pour la revue et la certification.

    Args:
        extraction_status: Valeur canonique du statut d'extraction.

    Returns:
        False uniquement pour ``confirmed_no_table`` apres normalisation.
    """
    status = normalize_extraction_status(extraction_status)
    return status != TABLE_EXTRACTION_STATUS_CONFIRMED_NO_TABLE


def get_vision_raw_indicators(table: Any) -> list[str]:
    """Retourner les libelles bruts Vision de la premiere colonne.

    Args:
        table: Objet table avec attribut vision_raw_indicators ou
            first_column_indicators_raw.

    Returns:
        Liste de chaines d'indicateurs non vides ; liste vide si absent.
    """
    if table is None:
        return []
    values = getattr(table, "vision_raw_indicators", None)
    if values is None:
        values = getattr(table, "first_column_indicators_raw", None)
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def get_comparison_indicators(table: Any) -> list[str]:
    """Retourner les libelles normalises de la premiere colonne pour la comparaison.

    Args:
        table: Objet table avec attribut comparison_normalized_indicators ou
            first_column_indicators.

    Returns:
        Liste de chaines d'indicateurs non vides ; liste vide si absent.
    """
    if table is None:
        return []
    values = getattr(table, "comparison_normalized_indicators", None)
    if values is None:
        values = getattr(table, "first_column_indicators", None)
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def get_canonical_footnotes(table: Any) -> FootnoteList:
    """Retourner les notes de bas de page canoniques d'un objet table.

    Args:
        table: Objet table avec attribut canonical_footnotes ou footnotes.

    Returns:
        Liste canonique ``[{"id": str, "text": str}, ...]`` ; les formats
        anciens sont normalises a l'ingestion.
    """
    if table is None:
        return []
    values = getattr(table, "canonical_footnotes", None)
    if values is None:
        values = getattr(table, "footnotes", None)
    if not values:
        return []
    from vigie.support.utils.footnotes_utils import normalize_footnotes_to_canonical

    return normalize_footnotes_to_canonical(list(values))


def get_extraction_confidence(table: Any) -> float:
    """Retourner la confiance d'extraction depuis les debug_metrics (0.0--1.0).

    Privilegie vision_extraction_confidence ; repli sur vision_primary_confidence.
    Quand les metriques sont absentes mais la table a du contenu Vision et des
    indicateurs bruts, assume une confiance certifiee pour ne pas bloquer.

    Args:
        table: Objet table avec debug_metrics et content_source optionnel.

    Returns:
        Score de confiance dans [0.0, 1.0] ; 0.0 si table est None ou metriques invalides.
    """
    if table is None:
        return 0.0
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return 0.0
    v = dm.get("vision_extraction_confidence")
    if v is None:
        v = dm.get("vision_primary_confidence")
    if v is None:
        content_source = getattr(table, "content_source", None)
        if content_source != VISION_CONTENT_SOURCE:
            content_source = infer_content_source(
                getattr(table, "extraction_method", None), None
            )
        if content_source == VISION_CONTENT_SOURCE and get_vision_raw_indicators(table):
            return max(EXTRACTION_CONFIDENCE_CERTIFIED_MIN, 0.7)
        return 0.0
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def get_extraction_quality_flags(table: Any) -> dict[str, bool]:
    """Retourner un ensemble normalise de drapeaux qualite depuis les debug_metrics.

    Quand les metriques sont absentes mais la table a du contenu Vision et des
    indicateurs bruts, assume vision_extraction_applied=True pour ne pas bloquer.

    Args:
        table: Objet table avec debug_metrics et content_source optionnel.

    Returns:
        Dictionnaire de noms de drapeaux vers booleens : recrop_attempted,
        recrop_used, recrop_failed_incomplete, vision_extraction_applied,
        crop_rejected, partial_result, rows_missing_from_fallback.
    """
    if table is None:
        return {}
    dm = getattr(table, "debug_metrics", None) or {}
    if not isinstance(dm, dict):
        return {}
    warning_codes = {
        str(code).strip()
        for code in list(dm.get("vision_warning_codes") or dm.get("warnings") or [])
        if str(code).strip()
    }
    vision_status = str(dm.get("vision_status") or "").strip().lower()
    applied = dm.get("vision_extraction_applied")
    if applied is None:
        applied = dm.get("vision_primary_applied")
    if applied is None or applied is False:
        content_source = getattr(table, "content_source", None)
        if content_source != VISION_CONTENT_SOURCE:
            content_source = infer_content_source(
                getattr(table, "extraction_method", None), None
            )
        if content_source == VISION_CONTENT_SOURCE and get_vision_raw_indicators(table):
            applied = True
    return {
        "recrop_attempted": bool(dm.get("recrop_attempted", False)),
        "recrop_used": bool(dm.get("recrop_used", False)),
        "recrop_failed_incomplete": bool(dm.get("recrop_failed_incomplete", False)),
        "vision_extraction_applied": bool(applied if applied is not None else False),
        "crop_rejected": bool(dm.get("crop_reject_reason")),
        "partial_result": vision_status == "partial",
        "rows_missing_from_fallback": "vision_rows_missing_from_fallback"
        in warning_codes,
    }


@dataclass(slots=True)
class TableArtifact:
    """Representation canonique en memoire d'une table extraite.

    Attributes:
        bank_code: Identifiant de la banque.
        section: Section du document (ex. bilan, compte de resultat).
        page_pdf: Numero de page PDF (base 1).
        table_id: Identifiant unique de la table.
        title: Titre de la table ; peut contenir des montants.
        headers: En-tetes de colonnes.
        rows: Lignes de la table sous forme de listes de cellules.
        first_column_indicators: Indicateurs normalises de la premiere colonne.
        extraction_method: Methode utilisee (ex. vision_full_gpt4o).
        title_clean: Titre nettoye sans montants ; pour affichage et appariement.
        table_summary: Resume semantique court utilise comme signal d'appariement.
        title_raw: Titre original pour tracabilite.
        table_number: Numero de table si present.
        bbox: Boite englobante (dict ou liste de floats).
        table_index_on_page: Index de cette table sur la page.
        tables_on_page: Nombre total de tables sur la page.
        bbox_top: Coordonnee superieure de la boite englobante.
        page_local_role: Role de cette table sur la page.
        quarter: Trimestre du rapport si applicable.
        pdf_path: Chemin vers le PDF source.
        first_column_indicators_raw: Indicateurs bruts Vision de la premiere colonne.
        first_column_groups: Groupes d'indicateurs.
        hierarchical_indicator_signature: Structure hierarchique des indicateurs.
        title_reliability: Fiabilite de l'extraction du titre.
        footnotes: Liste canonique de notes de bas de page.
        fragmentation_detected: Table detectee comme fragmentee.
        fragment_near_merge_hint: Indices pour la fusion de fragments.
        debug_metrics: Metriques d'extraction et drapeaux qualite.
        content_source: Source de contenu (vision_gpt4o ou unknown).
        comparison_eligible: Eligible pour le matching GPT.
        comparison_blockers: Non vide uniquement pour confirmed_no_table.
        extraction_status: Classification de l'etat de rescue.
    """

    bank_code: str
    section: str
    page_pdf: int
    table_id: str
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    first_column_indicators: list[str]
    extraction_method: str
    title_clean: str | None = (
        None  # Cleaned title (no amounts); use for display/pairing when set
    )
    table_summary: str | None = None
    title_raw: str | None = None  # Original title for traceability
    row_count: int | None = None
    table_number: str | None = None
    bbox: dict[str, Any] | list[float] | None = None
    table_index_on_page: int | None = None
    tables_on_page: int | None = None
    bbox_top: float | None = None
    page_local_role: str | None = None
    quarter: str | None = None
    pdf_path: str | None = None
    first_column_indicators_raw: list[str] | None = None
    first_column_groups: list[str] | None = None
    hierarchical_indicator_signature: list[str] | None = None
    title_reliability: str | None = None
    footnotes: FootnoteList | None = None
    fragmentation_detected: bool = False
    fragment_near_merge_hint: dict[str, Any] | None = None
    debug_metrics: dict[str, Any] | None = None
    content_source: str = UNKNOWN_CONTENT_SOURCE
    comparison_eligible: bool = False
    comparison_blockers: list[str] = field(default_factory=list)
    extraction_status: str = TABLE_EXTRACTION_STATUS_OK

    def __post_init__(self) -> None:
        """Initialiser les champs derives depuis l'etat d'extraction."""
        self.content_source = infer_content_source(
            self.extraction_method,
            self.content_source,
        )
        self.extraction_status = normalize_extraction_status(self.extraction_status)
        if self.title_clean is None:
            self.title_clean = self.title
        if self.row_count is None:
            if self.first_column_indicators_raw:
                self.row_count = len(
                    [item for item in self.first_column_indicators_raw if str(item).strip()]
                )
            else:
                self.row_count = len(list(self.rows or []))
        # Always recompute blockers from canonical extraction_status.
        self.comparison_blockers = derive_comparison_blockers(
            extraction_status=self.extraction_status,
        )
        self.comparison_eligible = is_comparison_eligible(
            self.extraction_status,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialiser l'artefact en dictionnaire pour le stockage ou l'export JSON."""
        return asdict(self)

    @property
    def vision_raw_indicators(self) -> list[str]:
        """Retourner les libelles bruts Vision GPT-4o de la premiere colonne."""
        return [
            str(item)
            for item in (self.first_column_indicators_raw or [])
            if str(item).strip()
        ]

    @property
    def comparison_normalized_indicators(self) -> list[str]:
        """Retourner les libelles normalises de la premiere colonne pour la comparaison."""
        return [
            str(item)
            for item in (self.first_column_indicators or [])
            if str(item).strip()
        ]

    @property
    def canonical_footnotes(self) -> FootnoteList:
        """Retourner les notes de bas de page sous forme canonique ``[{id, text}]``."""
        return list(self.footnotes or [])

    @property
    def is_vision_sourced(self) -> bool:
        """Retourner True quand le contenu de la table provient de Vision."""
        return self.content_source == VISION_CONTENT_SOURCE


# Kept for naming compatibility with downstream code.
TableCandidate = TableArtifact
