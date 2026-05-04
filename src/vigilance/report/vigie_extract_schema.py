"""Schema vigie_extract_v1 : format de sortie JSON unique par PDF.

Definit les structures typees, les utilitaires de transformation,
le constructeur, l'ecrivain et le chargeur (JSON -> TableArtifact).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from vigilance.models.table_models import (
    TABLE_EXTRACTION_STATUS_OK,
    VISION_CONTENT_SOURCE,
    TableArtifact,
)

SCHEMA_VERSION = "vigie_extract_v1"

# ---------------------------------------------------------------------------
# Canonical → output slug mapping
# ---------------------------------------------------------------------------

CANONICAL_TO_SLUG: dict[str, str] = {
    "capital_management": "gestion_capital_fonds_propres",
    "risk_management": "gestion_risques",
    "regulatory_updates": "reglementation",
}

SLUG_TO_DEFAULT_TITLE: dict[str, str] = {
    "gestion_capital_fonds_propres": "Gestion des fonds propres",
    "gestion_risques": "Gestion du risque",
    "reglementation": "Faits nouveaux en matière de réglementation",
}

SLUG_TO_CANONICAL: dict[str, str] = {v: k for k, v in CANONICAL_TO_SLUG.items()}


# ---------------------------------------------------------------------------
# TypedDict definitions (for documentation / IDE completion)
# ---------------------------------------------------------------------------


class FirstColumnEntry(TypedDict):
    """Entree de la premiere colonne d'un tableau."""

    row_idx: int
    text: str
    text_norm: str
    note_refs: list[str]


class FootnoteEntry(TypedDict):
    """Entree de note de bas de page extraite."""

    marker: str
    raw_text: str
    text: str
    text_norm: str
    scope: str


class TableFeatures(TypedDict):
    """Caracteristiques calculees d'un tableau extrait."""

    n_indicators: int
    indicator_set_hash: str
    anchors: list[str]


class TablePayload(TypedDict, total=False):
    """Payload complet d'un tableau extrait pour le schema Vigie."""

    table_uid: str
    table_id: str
    page_number: int
    table_number: str | None
    table_title: str | None
    unit_context: str | None
    headers: list[str]
    first_column: list[FirstColumnEntry]
    footnotes: list[FootnoteEntry]
    features: TableFeatures
    quality_flags: list[str]
    bbox: list[float] | None
    content_source: str
    comparison_eligible: bool
    comparison_blockers: list[str]


class SectionPayload(TypedDict):
    """Payload d'une section contenant ses tableaux."""

    section_title_pdf: str
    start_page: int
    end_page: int
    tables: list[TablePayload]


class ExtractionMeta(TypedDict, total=False):
    """Metadonnees de l'extraction (source, banque, trimestre, etc.)."""

    source_pdf: str
    pdf_hash: str
    bank_code: str
    quarter: str
    year: int
    language: str
    extracted_at: str
    extraction_method: str
    docling_version: str


class VigieExtractPayload(TypedDict):
    """Payload racine du schema d'extraction Vigie."""

    schema_version: str
    extraction_meta: ExtractionMeta
    sections: dict[str, SectionPayload]


# ---------------------------------------------------------------------------
# Text normalisation (mirrors section_taxonomy logic)
# ---------------------------------------------------------------------------


# Strip variant suffix after "Série/Series YYYY" to avoid OCR confusion (e.g. g vs 9)
# "Série 2023-g(7)" and "Série 2023-9(7)" -> "serie 2023"
_SERIES_SUFFIX_RE = re.compile(
    r"(s[eé]ries?\s+\d{4})[-\s]*[a-z0-9]+(?:\(\d+\))?",
    re.IGNORECASE,
)


def normalize_text(raw: str) -> str:
    """Normalisation sans accents, en minuscules, avec espaces reduits."""
    text = unicodedata.normalize("NFD", raw or "")
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    # Strip digits glued after closing paren (footnote ref from OCR)
    # e.g. "amorti)3" -> "amorti)"
    text = re.sub(r"\)\s*\d{1,3}", ")", text)
    # Strip 1-2 bare trailing digits glued to a letter (footnote ref)
    # e.g. "region2" -> "region", but "cet1", "tier1" kept (< 5 letters)
    text = re.sub(r"(?<=[a-z]{5})\d{1,2}(?=\W|$)", "", text)
    # Strip note reference patterns: (1), [2]
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    # Strip series variant suffixes before general cleanup
    text = _SERIES_SUFFIX_RE.sub(r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# Footnote-reference detection in indicator labels
# ---------------------------------------------------------------------------

# Extended pattern to capture all footnote marker formats from Vision extraction:
# - Numeric parentheses: (1), (2), (3)
# - Superscript digits: ¹, ², ³, ⁴, ⁵, ⁶, ⁷, ⁸, ⁹
# - Asterisk/dagger: *, †, ‡
# - Letter markers: (a), (b), a), b)
_NOTE_REF_RE = re.compile(
    r"\(([\d]+)\)"  # (1), (2), (12)
    r"|([¹²³⁴⁵⁶⁷⁸⁹⁰]+)"  # superscript digits
    r"|([\*†‡])"  # *, †, ‡
    r"|\(([a-zA-Z])\)"  # (a), (b)
    r"|\b([a-z])\)"  # a), b) at word boundary
    r"|(?<=\))(\d{1,3})"  # digits glued after ): amorti)3
)

# Map superscript digits to normal digits for normalization
_SUPERSCRIPT_MAP = str.maketrans("¹²³⁴⁵⁶⁷⁸⁹⁰", "1234567890")


def _extract_note_refs(label: str) -> tuple[str, list[str]]:
    """Extrait les references de notes d'un libelle d'indicateur.

    Gere les formats de marqueurs issus de l'extraction Vision :
    ``(1)``, ``(2)`` ; ``superscripts`` ; ``*``, ``dagger``, ``double-dagger`` ; ``(a)``, ``a)``.

    Args:
        label: Libelle brut contenant potentiellement des references.

    Returns:
        Tuple ``(texte_nettoye, [marqueurs_references])``.
    """
    refs: list[str] = []
    cleaned = label

    # Pre-pass: strip bare trailing digits glued to letters (min 5 letters).
    # e.g. "Région2" -> "Région" with ref '2', but "CET1"/"Tier1" stay.
    _bare_trailing = re.compile(r"(?<=[a-zA-Z]{5})(\d{1,2})$")
    _bt = _bare_trailing.search(cleaned)
    if _bt:
        refs.append(_bt.group(1))
        cleaned = cleaned[: _bt.start()] + cleaned[_bt.end() :]

    for m in _NOTE_REF_RE.finditer(cleaned):
        # Find which group matched (groups 1-6)
        ref = (
            m.group(1)
            or m.group(2)
            or m.group(3)
            or m.group(4)
            or m.group(5)
            or m.group(6)
        )
        if ref:
            # Normalize superscript digits to regular digits
            if m.group(2):  # superscript group
                ref = ref.translate(_SUPERSCRIPT_MAP)
            refs.append(ref)
    if refs:
        cleaned = _NOTE_REF_RE.sub("", cleaned).strip()
        # Also strip the bare trailing digit (already removed from cleaned above)
    return cleaned, refs


# ---------------------------------------------------------------------------
# First-column parsing
# ---------------------------------------------------------------------------


def parse_first_column(indicators: list[str]) -> list[FirstColumnEntry]:
    """Transforme les indicateurs bruts de premiere colonne en entrees structurees.

    Args:
        indicators: Liste de libelles bruts d'indicateurs.

    Returns:
        Liste de ``FirstColumnEntry`` avec texte nettoye et references de notes.
    """
    entries: list[FirstColumnEntry] = []
    for idx, raw in enumerate(indicators):
        text, note_refs = _extract_note_refs(raw.strip())
        # Use local aggressive normalize_text to guarantee accents,
        # punctuation, and extra spaces are stripped for matching.
        text_norm = normalize_text(text)
        entries.append(
            FirstColumnEntry(
                row_idx=idx,
                text=text,
                text_norm=text_norm,
                note_refs=note_refs,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Footnote parsing
# ---------------------------------------------------------------------------

_FOOTNOTE_MARKER_RE = re.compile(r"^\s*\(?(\d+)\)?\s*[.:\-–—]?\s*")


def parse_footnotes(raw_footnotes: list[str]) -> list[FootnoteEntry]:
    """Transforme les chaines brutes de footnotes en entrees structurees.

    Args:
        raw_footnotes: Liste de chaines brutes de footnotes.

    Returns:
        Liste de ``FootnoteEntry`` avec marqueur et texte normalise.
    """
    entries: list[FootnoteEntry] = []
    for raw in raw_footnotes:
        raw_stripped = raw.strip()
        if not raw_stripped:
            continue
        m = _FOOTNOTE_MARKER_RE.match(raw_stripped)
        marker = m.group(1) if m else ""
        text = raw_stripped[m.end() :].strip() if m else raw_stripped
        entries.append(
            FootnoteEntry(
                marker=marker,
                raw_text=raw_stripped,
                text=text,
                text_norm=normalize_text(text),
                scope="table",
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Table UID generation
# ---------------------------------------------------------------------------


def make_table_uid(
    bank_code: str,
    year: int,
    quarter: str,
    section_slug: str,
    table_number: str | None,
    page_number: int,
    table_index: int,
) -> str:
    """Genere un identifiant unique deterministe pour un tableau dans une extraction PDF."""
    q = quarter.lower().replace("-", "")
    tbl = f"tbl{table_number}" if table_number else f"idx{table_index}"
    return f"{bank_code}_{year}_{q}_{section_slug}_{tbl}_p{page_number}"


# ---------------------------------------------------------------------------
# Features computation
# ---------------------------------------------------------------------------


def compute_features(first_column: list[FirstColumnEntry]) -> TableFeatures:
    """Calcule les caracteristiques d'un tableau a partir de la premiere colonne parsee.

    Args:
        first_column: Liste de ``FirstColumnEntry`` parsees.

    Returns:
        ``TableFeatures`` avec hash d'indicateurs et ancres.
    """
    norms = sorted(entry["text_norm"] for entry in first_column if entry["text_norm"])
    hash_input = "|".join(norms)
    indicator_hash = "sha1:" + hashlib.sha1(
        hash_input.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return TableFeatures(
        n_indicators=len(first_column),
        indicator_set_hash=indicator_hash,
        anchors=[entry["text_norm"] for entry in first_column if entry["text_norm"]],
    )


# ---------------------------------------------------------------------------
# Section slug resolution
# ---------------------------------------------------------------------------


def canonical_to_slug(canonical: str) -> str:
    """Convertit une cle de section canonique en slug de sortie."""
    return CANONICAL_TO_SLUG.get(canonical, canonical)


def section_title_for_slug(slug: str, evidence_title: str | None = None) -> str:
    """Retourne un titre de section lisible, en privilegiant l'evidence PDF."""
    if evidence_title:
        return evidence_title
    return SLUG_TO_DEFAULT_TITLE.get(slug, slug.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Extraction metadata helpers
# ---------------------------------------------------------------------------


def compute_pdf_hash(pdf_path: str | Path) -> str:
    """Calcule le hash SHA-256 du fichier PDF."""
    return "sha256:" + hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()


def get_docling_version() -> str:
    """Retourne la version de docling installee, ou ``"unknown"``."""
    try:
        import docling

        return getattr(docling, "__version__", "unknown")
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Builder: assemble the full vigie_extract_v1 payload
# ---------------------------------------------------------------------------


def build_vigie_extract(
    *,
    pdf_path: str | Path,
    bank_code: str,
    quarter: str,
    year: int,
    language: str = "fr",
    section_ranges: list[dict[str, Any]],
    tables: list[Any],
) -> VigieExtractPayload:
    """Construit le payload vigie_extract_v1 a partir des sorties d'extraction.

    Args:
        pdf_path: Chemin du PDF source.
        bank_code: Code de la banque (ex. ``"cibc"``).
        quarter: Trimestre (ex. ``"t2-2025"`` ou ``"t2_2025"``).
        year: Annee fiscale.
        language: Code langue ISO (defaut ``"fr"``).
        section_ranges: Liste de dictionnaires avec cles ``"section"``, ``"start"``,
            ``"end"`` et optionnellement ``"evidence"``.
        tables: Liste d'objets ExtractedTable ou equivalents.

    Returns:
        Payload complet ``VigieExtractPayload``.
    """
    pdf_p = Path(pdf_path)
    extraction_meta = ExtractionMeta(
        source_pdf=str(pdf_p),
        pdf_hash=compute_pdf_hash(pdf_p) if pdf_p.exists() else "",
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        language=language,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        extraction_method="docling",
        docling_version=get_docling_version(),
    )

    range_lookup: dict[str, dict[str, Any]] = {}
    for sr in section_ranges:
        canon = sr.get("section", "")
        if canon:
            range_lookup[canon] = sr

    tables_by_section: dict[str, list[Any]] = {}
    for t in tables:
        sec = getattr(t, "section", None) or "unknown"
        tables_by_section.setdefault(sec, []).append(t)

    sections: dict[str, SectionPayload] = {}
    all_canonical_keys = set(range_lookup.keys()) | set(tables_by_section.keys())

    for canonical in sorted(all_canonical_keys):
        slug = canonical_to_slug(canonical)
        sr = range_lookup.get(canonical, {})
        evidence = sr.get("evidence", {})
        evidence_title: str | None = None
        if isinstance(evidence, dict):
            evidence_title = evidence.get("title_found")

        start_page = int(sr.get("start", sr.get("start_page_pdf", 0)) or 0)
        end_page = int(sr.get("end", sr.get("end_page_pdf", start_page)) or start_page)

        section_tables: list[TablePayload] = []
        for t_idx, t in enumerate(tables_by_section.get(canonical, [])):
            indicators_raw = list(getattr(t, "first_column_indicators_raw", None) or [])
            indicators_clean = list(getattr(t, "first_column_indicators", []) or [])
            first_col = parse_first_column(indicators_raw)
            footnotes = parse_footnotes(list(getattr(t, "footnotes", []) or []))
            features = compute_features(first_col)

            t_page = int(getattr(t, "page_number", 0) or 0)
            t_number = getattr(t, "table_number", None)
            t_uid = make_table_uid(
                bank_code=bank_code,
                year=year,
                quarter=quarter,
                section_slug=slug,
                table_number=t_number,
                page_number=t_page,
                table_index=t_idx,
            )

            table_payload = TablePayload(
                table_uid=t_uid,
                table_id=str(getattr(t, "table_id", f"tableau_{t_idx}")),
                page_number=t_page,
                table_number=t_number,
                table_title=getattr(t, "title", None),
                unit_context=getattr(t, "unit_context", None),
                headers=list(getattr(t, "headers", []) or []),
                first_column=first_col,
                footnotes=footnotes,
                features=features,
                quality_flags=[],
                bbox=getattr(t, "bbox", None),
                content_source=str(getattr(t, "content_source", "") or ""),
                comparison_eligible=bool(getattr(t, "comparison_eligible", False)),
                comparison_blockers=list(getattr(t, "comparison_blockers", None) or []),
            )
            if not first_col and indicators_clean:
                table_payload["quality_flags"] = [
                    *list(table_payload.get("quality_flags", []) or []),
                    "missing_first_column_raw",
                ]
            section_tables.append(table_payload)

        sections[slug] = SectionPayload(
            section_title_pdf=section_title_for_slug(slug, evidence_title),
            start_page=start_page,
            end_page=end_page,
            tables=section_tables,
        )

    return VigieExtractPayload(
        schema_version=SCHEMA_VERSION,
        extraction_meta=extraction_meta,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_vigie_extract(
    out_dir: str | Path,
    payload: VigieExtractPayload,
    filename: str | None = None,
) -> Path:
    """Ecrit le JSON vigie_extract_v1 dans *out_dir* et retourne son chemin.

    Args:
        out_dir: Repertoire de sortie.
        payload: Payload vigie_extract_v1 a ecrire.
        filename: Nom de fichier optionnel (genere automatiquement si absent).

    Returns:
        Chemin du fichier JSON ecrit.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    if filename is None:
        meta = payload.get("extraction_meta", {})
        bank = meta.get("bank_code", "unknown")
        quarter = meta.get("quarter", "q0").replace("-", "_")
        yr = meta.get("year", "0000")
        filename = f"{bank}_{quarter}_{yr}_extract.json"

    out_path = target / filename
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Loader: vigie_extract_v1 JSON -> list[TableArtifact]
# ---------------------------------------------------------------------------


def _slug_to_canonical(slug: str) -> str:
    """Convertit un slug de sortie vers la cle de section canonique."""
    return SLUG_TO_CANONICAL.get(slug, slug)


def load_artifacts_from_vigie_extract(
    source: str | Path | dict[str, Any],
) -> list[TableArtifact]:
    """Charge un JSON vigie_extract_v1 et retourne une liste de TableArtifact.

    Args:
        source: Chemin d'un fichier JSON, ou dictionnaire deja parse.

    Returns:
        Liste de ``TableArtifact`` reconstruits depuis le payload.
    """
    if isinstance(source, dict):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))

    meta = payload.get("extraction_meta", {})
    bank_code = meta.get("bank_code", "unknown")
    quarter = meta.get("quarter", "")
    pdf_path = meta.get("source_pdf", "")
    extraction_method = meta.get("extraction_method", "docling")

    artifacts: list[TableArtifact] = []
    sections = payload.get("sections", {})

    for slug, section_data in sections.items():
        canonical = _slug_to_canonical(slug)
        for table in section_data.get("tables", []):
            indicators = [
                entry.get("text", "") for entry in table.get("first_column", [])
            ]
            artifacts.append(
                TableArtifact(
                    bank_code=bank_code,
                    section=canonical,
                    page_pdf=int(table.get("page_number", 0) or 0),
                    table_id=table.get("table_id", ""),
                    title=table.get("table_title"),
                    headers=list(table.get("headers", [])),
                    rows=[],
                    first_column_indicators=indicators,
                    extraction_method=extraction_method,
                    table_number=table.get("table_number"),
                    bbox=table.get("bbox"),
                    quarter=quarter,
                    pdf_path=pdf_path,
                    first_column_indicators_raw=indicators,
                    footnotes=[
                        {
                            "id": str(item.get("marker", "") or ""),
                            "text": str(item.get("text", "") or ""),
                        }
                        for item in table.get("footnotes", [])
                        if str(item.get("text", "") or "").strip()
                    ],
                    content_source=str(
                        table.get("content_source") or VISION_CONTENT_SOURCE
                    ),
                    extraction_status=(
                        str(table.get("extraction_status") or "").strip()
                        or TABLE_EXTRACTION_STATUS_OK
                    ),
                    comparison_eligible=bool(
                        table.get("comparison_eligible", bool(indicators))
                    ),
                    comparison_blockers=list(table.get("comparison_blockers") or []),
                )
            )

    return artifacts
