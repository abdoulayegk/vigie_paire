"""Pipeline texte — .md comme source de vérité.

Ce module orchestre la comparaison textuelle inter-trimestrielle en trois étapes:

1. localise les sections texte dans les deux PDFs,
2. extrait les blocs narratifs via Docling + classification heuristique,
   puis écrit les fichiers ``text_extraction_*.md`` (source de vérité auditée),
3. compare les sections T1 vs T2 via GPT-4o directement sur le contenu .md,
4. trie les changements métiers et ne conserve que les changements vraiment majeurs.

Le .md est à la fois l'artefact d'audit lisible par l'humain
et l'entrée directe du GPT de comparaison — garantissant leur cohérence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.cli.quarter_logic import normalize_quarter, resolve_previous_quarter
from vigilance.extraction.section_locator import locate_sections_in_pdf
from vigilance.extraction.section_taxonomy import canonicalize_section
from vigilance.text_comparison.text_comparison_writer import (
    get_text_comparison_path,
    write_text_comparison,
)
from vigilance.text_extraction.text_extraction_markdown_writer import (
    get_text_extraction_markdown_path,
    write_text_extraction_markdown,
)
from vigilance.utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)

UNIFIED_TEXT_SCHEMA_VERSION = 3

_SECTION_LABELS: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_CANONICAL_TO_TEXT_KEY: dict[str, str] = {
    "capital_management": "gestion_capital",
    "capital": "gestion_capital",
    "risk_management": "gestion_risques",
    "risk": "gestion_risques",
    "regulatory_updates": "gestion_reglementation",
    "regulatory": "gestion_reglementation",
}

_THEME_BY_SECTION: dict[str, str] = {
    "gestion_capital": "capital",
    "gestion_risques": "risque",
    "gestion_reglementation": "changement",
}

_TARGET_SECTIONS_BY_BANK: dict[str, set[str]] = {
    "bnc": {"gestion_capital", "gestion_risques"},
    "rbc": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
    "bns": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
    "scotia": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
    "cibc": {"gestion_capital", "gestion_risques"},
    "td": {"gestion_capital", "gestion_risques"},
    "bmo": {"gestion_capital", "gestion_risques", "gestion_reglementation"},
}

_MODEL_MAX_OUTPUT_TOKENS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^gpt-4o(?:$|-)", flags=re.IGNORECASE), 16_384),
    (re.compile(r"^gpt-4\.1(?:$|-)", flags=re.IGNORECASE), 32_768),
    (re.compile(r"^gpt-4(?:$|-)", flags=re.IGNORECASE), 8_192),
]

_REGULATORY_REF_RE = re.compile(
    r"\b(?:OSFI|BSIF|Bâle|Basel|TLAC|LCR|NSFR|CET1|Tier\s*1|Tier\s*2|Pilier\s*[123]|IFRS|IAS|NIIF|BISM|VaR)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\S*\d\S*\b")
_ROMAN_NUMERAL_RE = re.compile(r"\b[IVX]{1,4}\b")
_PERCENT_RE = re.compile(r"[%‰]+")
_BPS_RE = re.compile(r"\b(?:pb|pbs|bp|bps|point(?:s)?\s+de\s+base)\b", flags=re.IGNORECASE)
_PUNCT_SPACING_RE = re.compile(r"\s+([,;:.])")
_MULTISPACE_RE = re.compile(r"\s+")
_TABLE_VALUE_RE = re.compile(r"\b\d+(?:[\s.,]\d+)*(?:\s*[%$])?\b")
_TABLE_ROW_MARKER_RE = re.compile(
    r"\b(?:tableau|table|total|totaux|s[ée]rie|series|moody's|s&p|fitch|dbrs|en millions de dollars|"
    r"en milliers de dollars|valeur pond[ée]r[ée]e|valeur non pond[ée]r[ée]e|aux 31|au 31)\b",
    flags=re.IGNORECASE,
)
_FOOTNOTE_MARKER_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:\(?\d+\)|\d+\)|[¹²³⁴⁵⁶⁷⁸⁹]+|\*{1,3}|[a-z]\))|(?:^|\n)\s*(?:note|source)\b)",
    flags=re.IGNORECASE,
)
_TABLE_HEADING_RE = re.compile(r"^\s*(?:tableau|table)\b", flags=re.IGNORECASE)
_SUBSECTION_SPLIT_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_SEMANTIC_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bcadre de capacité totale d[’']absorption des pertes\b", flags=re.IGNORECASE),
        "un cadre renforcé d'absorption des pertes",
    ),
    (
        re.compile(r"\bligne directrice sur le levier\b", flags=re.IGNORECASE),
        "des exigences de levier",
    ),
    (
        re.compile(r"\bréformes de\s+[IVX]{1,4}\b", flags=re.IGNORECASE),
        "des réformes prudentielles",
    ),
    (
        re.compile(r"\bexigences?\s+réglementaires?\b", flags=re.IGNORECASE),
        "des exigences prudentielles",
    ),
    (
        re.compile(r"\bexigence\s+réglementaire\s+minimale\b", flags=re.IGNORECASE),
        "exigence minimale",
    ),
    (
        re.compile(r"\bBISM\b", flags=re.IGNORECASE),
        "les banques d'importance systémique",
    ),
    (
        re.compile(r"\bVaR\b", flags=re.IGNORECASE),
        "la mesure de risque de marché",
    ),
]


class TextAnalysisQualityError(RuntimeError):
    """Raised when a targeted text section cannot yield analyzable semantic units."""


@dataclass(slots=True)
class SemanticUnit:
    """Unité sémantique extraite d'une section narratif d'un rapport bancaire."""

    unit_id: str
    section_key: str
    theme: str
    semantic_text: str
    source_text: str
    source_block_ids: list[str]
    source_resolution: str
    evidence_pages: list[int]
    evidence_snippet: str


@dataclass(slots=True)
class ResolvedSection:
    """Section narrative résolue avec ses coordonnées de pages."""

    section_key: str
    title: str
    start_page: int
    end_page: int
    anchor_page: int | None = None
    anchor_text: str | None = None
    anchor_bbox_norm: list[float] | None = None

    @property
    def pages(self) -> list[int]:
        """Retourne la liste des pages couvertes par la section."""
        return list(range(self.start_page, self.end_page + 1))


@dataclass(slots=True)
class PDFBlock:
    """Bloc de texte extrait d'un PDF avec ses métadonnées de position."""

    block_id: str
    page: int
    bbox_norm: list[float]
    text: str
    line_number: int
    block_type: str = "other"
    included: bool = False
    exclusion_reason: str = ""
    source_label: str = ""

    @property
    def y0(self) -> float:
        """Coordonnée Y du bord supérieur du bloc (normalisée)."""
        return float(self.bbox_norm[1])

    @property
    def y1(self) -> float:
        """Coordonnée Y du bord inférieur du bloc (normalisée)."""
        return float(self.bbox_norm[3])


@dataclass(slots=True)
class SectionAudit:
    """Audit complet d'une section narrative : blocs inclus, exclus et unités sémantiques."""

    section_key: str
    section_title: str
    start_page: int
    end_page: int
    anchor_page: int | None
    anchor_text: str | None
    anchor_bbox_norm: list[float] | None
    included_blocks: list[PDFBlock]
    excluded_blocks: list[PDFBlock]
    semantic_units: list[SemanticUnit] = field(default_factory=list)
    table_regions: list[dict[str, Any]] = field(default_factory=list)


def _json_dumps(data: Any) -> str:
    """Sérialise ``data`` en JSON indenté avec support complet des caractères UTF-8."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def _sanitize_semantic_text(text: str) -> str:
    """Normalise un texte pour la comparaison sémantique inter-trimestrielle.

    Supprime les éléments non sémantiques — chiffres, pourcentages, points de base,
    références réglementaires, numéros romains — afin que deux paragraphes exprimant
    la même idée avec des valeurs différentes soient reconnus comme identiques.
    Utilisée pour peupler ``semantic_text_t1`` / ``semantic_text_t2`` dans les changements.
    """
    value = (text or "").strip()
    if not value:
        return ""
    for pattern, replacement in _SEMANTIC_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    value = _REGULATORY_REF_RE.sub("", value)
    value = _NUMERIC_TOKEN_RE.sub("", value)
    value = _ROMAN_NUMERAL_RE.sub("", value)
    value = _PERCENT_RE.sub("", value)
    value = _BPS_RE.sub("", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\([^)]*\d[^)]*\)", "", value)
    value = re.sub(r"\s*[-–—]\s*", " ", value)
    value = re.sub(r"\b(?:Le|La|Les)\s+a\b", "La banque a", value)
    value = re.sub(r"\bLa Banque\b", "La banque", value)
    value = re.sub(r"\bLe Groupe\b", "La banque", value)
    value = re.sub(r"\bConseil d'administration\b", "gouvernance", value, flags=re.IGNORECASE)
    value = _PUNCT_SPACING_RE.sub(r"\1", value)
    value = _MULTISPACE_RE.sub(" ", value).strip(" ,;:.")
    return value.strip()


def _normalized_block_text(text: str) -> str:
    """Normalise un texte pour les comparaisons de correspondance (matching).

    Passe en minuscules, réduit les espaces multiples, retire la ponctuation et
    les caractères spéciaux. Conserve lettres accentuées et chiffres.
    Utilisée pour détecter les doublons, les en-têtes déjà vus et pour
    retrouver la page exacte d'un fragment GPT dans les blocs PDF.
    """
    value = (text or "").lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9 ]+", "", value)
    return value.strip()


def _sanitize_explanation(text: str) -> str:
    """Nettoie et tronque une explication GPT à 1 200 caractères maximum."""
    value = _sanitize_semantic_text(text)
    return value[:1200]


def _count_numeric_values(text: str) -> int:
    """Compte le nombre de valeurs numériques dans ``text`` (entiers, décimaux, négatifs)."""
    return len(_TABLE_VALUE_RE.findall(text or ""))


def _contains_dense_numeric_line(text: str) -> bool:
    """Retourne True si au moins une ligne contient plus de 3 valeurs numériques.

    Signe caractéristique d'une ligne de tableau financier (colonnes de chiffres).
    """
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _count_numeric_values(line) > 3:
            return True
    return False


def _looks_like_table_or_financial_grid(text: str) -> bool:
    """Détecte si un bloc ressemble à un tableau ou une grille financière.

    Utilisée pour exclure les blocs non narratifs (tableaux, grilles de notation,
    listes de chiffres) lors de la classification des blocs PDF.
    Critères : en-têtes de tableau, marqueurs de ligne + chiffres, densité numérique
    élevée, ratings, tabulations, colonnes séparées par des espaces multiples.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _TABLE_HEADING_RE.search(value):
        return True
    if _TABLE_ROW_MARKER_RE.search(value) and _count_numeric_values(value) >= 2:
        return True
    if _contains_dense_numeric_line(value) and len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)) <= 20:
        return True
    if re.search(r"\b(?:AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|Aa[123]|A[123]|Baa[123]|FPUNV)\b", value):
        return True
    if "\t" in value or "|" in value:
        return True
    if re.search(r"(?:\b\S+\s+\d+(?:[.,]\d+)?\s*){4,}", value):
        return True
    return False


def _looks_like_footnote(text: str) -> bool:
    """Détecte si un bloc ressemble à une note de bas de page.

    Reconnaît les marqueurs courants : ``(1)``, ``1)``, ``*``, exposants Unicode,
    numéros suivis d'un texte court. Ces blocs sont exclus du contenu narratif
    pour ne pas polluer la comparaison de sections textuelles.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _FOOTNOTE_MARKER_RE.search(value):
        return True
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[¹²³⁴⁵⁶⁷⁸⁹]+\s*", line):
            return True
        if re.match(r"^\d{1,2}\s+", line):
            words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", line)
            if len(words) <= 30:
                return True
    return False


def _looks_like_table_footnote_text(text: str) -> bool:
    """Détecte si un bloc est une légende ou note annexée à un tableau.

    Variante de ``_looks_like_footnote`` pour les blocs situés juste sous un
    tableau (zone inférée par ``_infer_table_footnote_bboxes``). Reconnaît en
    plus les préfixes « s.o. », « note », « source » et les courtes phrases
    mêlant quelques chiffres et parenthèses.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _looks_like_footnote(value):
        return True
    lower_value = value.lower()
    if lower_value.startswith(("s.o.", "n.s.", "sans objet", "note", "source")):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)
    return len(words) <= 30 and _count_numeric_values(value) >= 1 and "(" in value


def _looks_like_narrative_paragraph(text: str) -> bool:
    """Détecte si un bloc est un paragraphe narratif (texte continu à analyser).

    Un paragraphe narratif valide doit contenir ≥ 18 mots, ≥ 120 caractères,
    une forte proportion de lettres, peu de chiffres, et au moins un connecteur
    grammatical ou une ponctuation de fin de phrase. Les tableaux et grilles
    financières sont explicitement rejetés.
    """
    value = str(text or "").strip()
    if not value:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", value)
    if len(words) < 18 or len(value) < 120:
        return False
    if "\t" in value or "|" in value:
        return False
    if _TABLE_HEADING_RE.search(value):
        return False
    connectors = re.findall(
        r"\b(?:la|le|les|une|un|des|afin|ainsi|mais|et|ou|que|qui|dont|pour|avec|alors|toutefois|cependant|de plus|enfin)\b",
        value,
        flags=re.IGNORECASE,
    )
    sentence_marks = len(re.findall(r"[.;:?!]", value))
    alpha_chars = sum(1 for ch in value if ch.isalpha())
    digit_chars = sum(1 for ch in value if ch.isdigit())
    alpha_ratio = alpha_chars / max(1, len(value))
    digit_ratio = digit_chars / max(1, len(value))
    short_word_ratio = sum(1 for word in words if len(word) <= 3) / max(1, len(words))
    return (
        alpha_ratio >= 0.45
        and digit_ratio <= 0.2
        and short_word_ratio <= 0.5
        and (sentence_marks >= 1 or len(connectors) >= 4)
    )


def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    """Calcule le ratio de chevauchement de la bounding box ``a`` avec ``b``.

    Retourne l'aire de l'intersection divisée par l'aire de ``a`` (valeur entre 0 et 1).
    Un résultat de 1.0 signifie que ``a`` est entièrement contenu dans ``b``.
    Les coordonnées sont normalisées [x0, y0, x1, y1] dans l'espace [0, 1].
    """
    if len(a) < 4 or len(b) < 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area = max(1e-9, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    return inter / area


def _block_overlaps_table(block: PDFBlock, table_bboxes: list[list[float]]) -> bool:
    """Retourne True si le bloc chevauche d'au moins 5 % une des bounding boxes de tableau."""
    return any(_bbox_overlap_ratio(block.bbox_norm, bbox) >= 0.05 for bbox in table_bboxes)


def _infer_table_footnote_bboxes(
    table_bboxes_by_page: dict[int, list[list[float]]],
    *,
    max_height: float = 0.05,
) -> dict[int, list[list[float]]]:
    """Infère les zones de notes de bas de tableau à partir des bounding boxes des tableaux.

    Pour chaque tableau, génère une zone candidate juste en-dessous (hauteur ≤ ``max_height``
    en coordonnées normalisées). Ces zones sont ensuite utilisées par
    ``_classify_block_type`` pour identifier les blocs de légende ou de notes
    annexés aux tableaux et les exclure du contenu narratif.

    Args:
        table_bboxes_by_page: Bounding boxes des tableaux détectés par Docling, par page.
        max_height: Hauteur maximale (normalisée) de la zone note inférée.

    Returns:
        Dictionnaire page → liste de bounding boxes de zones notes potentielles.
    """
    footnote_bboxes_by_page: dict[int, list[list[float]]] = {}
    for page, boxes in table_bboxes_by_page.items():
        ordered = sorted((list(box) for box in boxes if len(box) == 4), key=lambda bbox: (float(bbox[1]), float(bbox[0])))
        page_regions: list[list[float]] = []
        for idx, bbox in enumerate(ordered):
            top = max(0.0, min(1.0, float(bbox[3])))
            next_top = float(ordered[idx + 1][1]) if idx + 1 < len(ordered) else 1.0
            bottom = min(1.0, next_top, top + max_height)
            if bottom - top < 0.01:
                continue
            page_regions.append([0.0, top, 1.0, bottom])
        if page_regions:
            footnote_bboxes_by_page[page] = page_regions
    return footnote_bboxes_by_page


def _is_new_major_or_allowed_moderate(triage: dict[str, Any]) -> bool:
    """Retourne True si le triage indique un changement majeur ou un changement modéré significatif.

    Un changement modéré est retenu uniquement s'il introduit une nouvelle idée
    ou s'il contient un signal réglementaire ou méthodologique fort.
    Utilisée pour filtrer les changements à escalader en priorité.
    """
    if not triage.get("is_relevant", False):
        return False
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    if impact == "MAJEUR":
        return True
    if impact != "MODERE":
        return False
    if triage.get("nouvelle_idee", False):
        return True
    signals = triage.get("signals") or {}
    return bool(signals.get("regulatory_reference_added") or signals.get("methodology_change"))


def _compute_conservative_new_idea(change: dict[str, Any], triage: dict[str, Any]) -> bool:
    """Détermine de façon conservatrice si un changement constitue une nouvelle idée.

    Une nouvelle idée est retenue uniquement si le triage la juge pertinente ET
    que le diff_type est ``added`` avec un texte T2 non vide. Cela évite de
    qualifier comme nouvelles idées les simples reformulations ou suppressions.
    """
    if not triage.get("is_relevant", False):
        return False

    diff_type = str(change.get("diff_type") or "").lower()
    return diff_type == "added" and bool(change.get("semantic_text_t2"))


def _is_non_cosmetic_change(triage: dict[str, Any]) -> bool:
    """Retourne True si le changement n'est pas classé COSMETIQUE par le triage GPT."""
    return str(triage.get("category") or "").upper() != "COSMETIQUE"


def _retained_change_sort_key(change: dict[str, Any]) -> tuple[int, int, str, str, str]:
    """Clé de tri pour ordonner les changements retenus dans le rapport final.

    Ordre de priorité : nouvelles idées en premier, puis impact décroissant
    (MAJEUR → MODERE → MINEUR), puis section, puis page, puis type de diff.
    """
    triage = change.get("genai_triage") or {}
    impact = str(triage.get("impact_level") or "MINEUR").upper()
    diff_type = str(change.get("diff_type") or "").lower()
    pages = change.get("pages_t2") or change.get("pages_t1") or []
    first_page = ""
    if pages:
        try:
            first_page = f"{int(pages[0]):06d}"
        except (TypeError, ValueError):
            first_page = str(pages[0])
    impact_order = {"MAJEUR": 0, "MODERE": 1, "MINEUR": 2}.get(impact, 99)
    return (
        0 if triage.get("nouvelle_idee", False) else 1,
        impact_order,
        str(change.get("section_key") or ""),
        first_page,
        diff_type,
    )


def _sorted_sections(sections: dict[str, ResolvedSection]) -> list[ResolvedSection]:
    """Retourne les sections triées par ordre d'apparition dans le PDF.

    Tri par page de début, puis par position verticale de l'ancre, puis par clé
    de section pour garantir un ordre stable lors des itérations.
    """
    return sorted(
        sections.values(),
        key=lambda sec: (
            sec.start_page,
            float(sec.anchor_bbox_norm[1]) if sec.anchor_bbox_norm else 0.0,
            sec.section_key,
        ),
    )


def _next_section_by_key(sections: dict[str, ResolvedSection]) -> dict[str, ResolvedSection | None]:
    """Construit un mapping section_key → section suivante dans le PDF.

    Utilisé pour délimiter la fenêtre basse d'une section quand deux sections
    partagent la même page de fin/début.
    """
    ordered = _sorted_sections(sections)
    next_map: dict[str, ResolvedSection | None] = {section.section_key: None for section in ordered}
    for current, nxt in zip(ordered, ordered[1:]):
        next_map[current.section_key] = nxt
    return next_map


def _section_window_for_page(
    section: ResolvedSection,
    page_number: int,
    next_section: ResolvedSection | None = None,
) -> tuple[float, float]:
    """Calcule la fenêtre verticale (top, bottom) d'une section sur une page donnée.

    Sur la première page de la section, la fenêtre commence sous l'ancre pour
    éviter d'inclure le titre de section lui-même. Sur la dernière page partagée
    avec la section suivante, la fenêtre se ferme au-dessus de l'ancre suivante.
    Les coordonnées sont normalisées dans [0, 1].

    Returns:
        Tuple ``(top, bottom)`` délimitant la zone de la section sur cette page.
    """
    top = 0.0
    bottom = 1.0
    if (
        page_number == section.start_page
        and section.anchor_page == section.start_page
        and section.anchor_bbox_norm
    ):
        top = max(top, float(section.anchor_bbox_norm[3]))
    if (
        next_section is not None
        and page_number == section.end_page == next_section.start_page
        and next_section.anchor_page == next_section.start_page
        and next_section.anchor_bbox_norm
    ):
        bottom = min(bottom, float(next_section.anchor_bbox_norm[1]))
    if bottom <= top:
        return top, 1.0
    return top, bottom


def _docling_bbox_to_norm(docling_doc: Any, prov: Any) -> list[float] | None:
    """Convertit une bounding box Docling en coordonnées normalisées [0, 1].

    Applique ``to_top_left_origin`` (système Docling en bas-gauche) puis
    ``normalized`` pour obtenir [x0, y0, x1, y1] dans l'espace page normalisé.
    Retourne None si la conversion échoue (page manquante, bbox invalide).
    """
    try:
        page_obj = docling_doc.pages[prov.page_no]
        page_height = page_obj.size.height
        norm = prov.bbox.to_top_left_origin(page_height=page_height).normalized(page_obj.size)
        return [
            max(0.0, min(1.0, float(norm.l))),
            max(0.0, min(1.0, float(norm.t))),
            max(0.0, min(1.0, float(norm.r))),
            max(0.0, min(1.0, float(norm.b))),
        ]
    except Exception:
        return None


def _extract_docling_page_blocks(
    pdf_path: Path,
    page_numbers: list[int],
) -> tuple[dict[int, list[PDFBlock]], dict[int, list[list[float]]], dict[int, list[list[float]]]]:
    """Extrait tous les blocs de texte d'un PDF via Docling pour les pages demandées.

    Lance Docling sans OCR sur la plage ``[min(pages), max(pages)]``, puis filtre
    les blocs par page. Retourne trois structures indexées par numéro de page :

    - ``page_blocks`` : liste de PDFBlock triés par position (y, x)
    - ``table_bboxes_by_page`` : bounding boxes des tableaux détectés
    - ``footnote_bboxes_by_page`` : zones de notes inférées sous les tableaux

    Args:
        pdf_path: Chemin vers le fichier PDF source.
        page_numbers: Pages à extraire (numérotation 1-based Docling).

    Returns:
        Tuple ``(page_blocks, table_bboxes_by_page, footnote_bboxes_by_page)``.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if not page_numbers:
        return {}, {}, {}

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    start_page = min(page_numbers)
    end_page = max(page_numbers)
    result = converter.convert(str(pdf_path), page_range=(start_page, end_page))
    docling_doc = result.document

    table_bboxes_by_page: dict[int, list[list[float]]] = {}
    for table in getattr(docling_doc, "tables", []) or []:
        try:
            if not getattr(table, "prov", None):
                continue
            page = int(getattr(table.prov[0], "page_no", 0) or 0)
            if page not in page_numbers:
                continue
            bbox = _docling_bbox_to_norm(docling_doc, table.prov[0])
            if not bbox:
                continue
            table_bboxes_by_page.setdefault(page, []).append(bbox)
        except Exception:
            continue
    footnote_bboxes_by_page = _infer_table_footnote_bboxes(table_bboxes_by_page)

    page_blocks: dict[int, list[PDFBlock]] = {page: [] for page in page_numbers}
    line_numbers: dict[int, int] = {page: 0 for page in page_numbers}
    for page in page_numbers:
        for item, _level in docling_doc.iterate_items(page_no=page, with_groups=False):
            text = _MULTISPACE_RE.sub(" ", str(getattr(item, "text", "") or "").replace("\n", " ").strip()).strip()
            if not text:
                continue
            label = str(getattr(item, "label", "") or "").lower()
            item_provs = [prov for prov in list(getattr(item, "prov", []) or []) if int(getattr(prov, "page_no", 0) or 0) == page]
            if not item_provs:
                continue
            bbox_norm = _docling_bbox_to_norm(docling_doc, item_provs[0])
            if not bbox_norm:
                continue
            line_numbers[page] += 1
            initial_block_type = {
                "footnote": "footnote",
                "page_header": "header_footer",
                "page_footer": "header_footer",
                "caption": "table",
            }.get(label, "other")
            page_blocks[page].append(
                PDFBlock(
                    block_id=f"p{page:03d}_d{line_numbers[page]:03d}",
                    page=page,
                    bbox_norm=bbox_norm,
                    text=text,
                    line_number=line_numbers[page],
                    block_type=initial_block_type,
                    source_label=label,
                )
            )
        page_blocks[page].sort(key=lambda block: (round(block.y0, 4), round(block.bbox_norm[0], 4)))
    return page_blocks, table_bboxes_by_page, footnote_bboxes_by_page


def _classify_block_type(
    block: PDFBlock,
    repeated_text_counts: dict[str, int],
    table_bboxes: list[list[float]] | None = None,
    footnote_bboxes: list[list[float]] | None = None,
) -> str:
    """Classifie un bloc PDF en l'une des catégories : narrative, table, footnote, header_footer, other.

    Priorités d'application :
    1. Type déjà assigné par Docling (table, footnote, header_footer) → conservé tel quel.
    2. Texte répété en haut/bas de page → header_footer.
    3. Bas de page avec marqueur de note → footnote.
    4. Chevauchement géométrique avec un tableau → table.
    5. Chevauchement avec zone de note + texte de légende → footnote.
    6. Heuristiques textuelles : narrative, table ou other.

    Args:
        block: Bloc PDF à classifier.
        repeated_text_counts: Nombre d'occurrences normalisées de chaque texte (toutes pages).
        table_bboxes: Bounding boxes des tableaux sur la même page.
        footnote_bboxes: Zones de notes inférées sur la même page.

    Returns:
        Chaîne parmi ``"narrative"``, ``"table"``, ``"footnote"``, ``"header_footer"``, ``"other"``.
    """
    if block.block_type in {"table", "footnote", "header_footer"}:
        return block.block_type
    norm = _normalized_block_text(block.text)
    if not norm:
        return "other"
    text = block.text.strip()
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    digits = re.findall(r"\d", text)
    numeric_tokens = re.findall(r"\b\S*\d\S*\b", text)
    rating_tokens = re.findall(r"\b(?:[A-Z]{1,4}[+-]?|[A-Z][a-z]\d|[A-Z]{1,3}\s*\(hyb\)|FPUNV)\b", text)
    short_word_ratio = (
        sum(1 for word in words if len(word) <= 4) / max(1, len(words))
        if words
        else 0.0
    )
    digit_ratio = len(digits) / max(1, len(text))
    upper_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    repeated = repeated_text_counts.get(norm, 0)
    table_bboxes = table_bboxes or []
    footnote_bboxes = footnote_bboxes or []

    if repeated >= 2 and (block.y1 <= 0.12 or block.y0 >= 0.88):
        return "header_footer"
    if block.y0 >= 0.75 and re.match(r"^\s*(?:\(?\d+\)?|[*†‡]|note\b|source\b)", text, flags=re.IGNORECASE):
        return "footnote"
    if _block_overlaps_table(block, table_bboxes):
        return "table"
    if _block_overlaps_table(block, footnote_bboxes) and _looks_like_table_footnote_text(text):
        return "footnote"
    if _looks_like_footnote(text):
        return "footnote"
    if _looks_like_narrative_paragraph(text):
        return "narrative"
    if (
        _looks_like_table_or_financial_grid(text)
        or
        (digit_ratio >= 0.12 and len(words) <= 16)
        or (len(numeric_tokens) >= 10 and len(words) <= 20)
        or len(rating_tokens) >= 6
        or (len(numeric_tokens) >= 4 and short_word_ratio >= 0.45)
        or (short_word_ratio >= 0.62 and upper_ratio >= 0.18 and len(words) >= 18)
        or "\t" in text
        or "  " in text
        or ("|" in text)
        or (len(words) <= 8 and len(digits) >= 4)
        or (upper_ratio >= 0.7 and len(words) <= 12)
    ):
        return "table"
    if len(words) >= 8 and len(text) >= 45:
        return "narrative"
    return "other"


def _exclusion_reason_for_block(block_type: str, in_window: bool) -> str:
    """Retourne la raison d'exclusion d'un bloc non narratif.

    Un bloc hors fenêtre de section reçoit ``outside_target_section``.
    Un bloc dans la fenêtre mais non narratif reçoit une raison selon son type.
    Un bloc narratif inclus reçoit une chaîne vide (pas d'exclusion).
    """
    if not in_window:
        return "outside_target_section"
    return {
        "table": "table_like_block",
        "footnote": "footnote",
        "header_footer": "header_footer",
        "other": "non_narrative_block",
    }.get(block_type, "")


def _table_regions_for_pages(
    section_key: str,
    table_bboxes_by_page: dict[int, list[list[float]]],
    footnote_bboxes_by_page: dict[int, list[list[float]]],
    pages: list[int],
) -> list[dict[str, Any]]:
    """Construit la liste des régions de tableaux et de notes pour un ensemble de pages.

    Chaque région est un dictionnaire avec ``table_id``, ``page``, ``region_type``
    (``"table"`` ou ``"footnote"``) et ``bbox``. Stockée dans ``SectionAudit.table_regions``
    pour la traçabilité et l'audit de l'extraction.
    """
    regions: list[dict[str, Any]] = []
    for page in pages:
        for idx, bbox in enumerate(table_bboxes_by_page.get(page, []), start=1):
            regions.append(
                {
                    "table_id": f"{section_key}_p{page:03d}_tbl_{idx:02d}",
                    "page": page,
                    "region_type": "table",
                    "bbox": [round(v, 6) for v in bbox],
                }
            )
        for idx, bbox in enumerate(footnote_bboxes_by_page.get(page, []), start=1):
            regions.append(
                {
                    "table_id": f"{section_key}_p{page:03d}_ftn_{idx:02d}",
                    "page": page,
                    "region_type": "footnote",
                    "bbox": [round(v, 6) for v in bbox],
                }
            )
    return regions


def _repeated_text_counts(page_blocks: dict[int, list[PDFBlock]]) -> dict[str, int]:
    """Compte le nombre de pages distinctes où chaque texte normalisé apparaît.

    Un texte qui apparaît sur ≥ 2 pages est un candidat en-tête/pied de page.
    Utilisé par ``_classify_block_type`` pour détecter et exclure ces répétitions.
    """
    counts: dict[str, int] = {}
    for blocks in page_blocks.values():
        seen_on_page: set[str] = set()
        for block in blocks:
            norm = _normalized_block_text(block.text)
            if not norm or norm in seen_on_page:
                continue
            seen_on_page.add(norm)
            counts[norm] = counts.get(norm, 0) + 1
    return counts


def _build_section_audit(
    *,
    section: ResolvedSection,
    next_section: ResolvedSection | None,
    page_blocks: dict[int, list[PDFBlock]],
    repeated_text_counts: dict[str, int],
    table_bboxes_by_page: dict[int, list[list[float]]],
    footnote_bboxes_by_page: dict[int, list[list[float]]],
) -> SectionAudit:
    """Construit l'audit complet d'une section : blocs inclus, exclus et régions de tableaux.

    Pour chaque bloc de la section, applique la fenêtre verticale de la section,
    classifie le type de bloc et décide de son inclusion. Seuls les blocs
    ``narrative`` dans la fenêtre sont inclus ; les autres sont gardés dans
    ``excluded_blocks`` pour la traçabilité.

    Args:
        section: Section résolue avec ses pages et son ancre.
        next_section: Section suivante dans le PDF (pour délimiter la fenêtre basse).
        page_blocks: Blocs extraits par Docling, indexés par page.
        repeated_text_counts: Comptages de textes répétés pour détection en-têtes/pieds.
        table_bboxes_by_page: Bounding boxes des tableaux, par page.
        footnote_bboxes_by_page: Zones de notes inférées, par page.

    Returns:
        ``SectionAudit`` avec ``included_blocks``, ``excluded_blocks`` et ``table_regions``.
    """
    included_blocks: list[PDFBlock] = []
    excluded_blocks: list[PDFBlock] = []
    for page in section.pages:
        blocks = page_blocks.get(page, [])
        page_tables = table_bboxes_by_page.get(page, [])
        page_footnotes = footnote_bboxes_by_page.get(page, [])
        top_cutoff, bottom_cutoff = _section_window_for_page(section, page, next_section)
        for block in blocks:
            section_block = PDFBlock(
                block_id=block.block_id,
                page=block.page,
                bbox_norm=list(block.bbox_norm),
                text=block.text,
                line_number=block.line_number,
                block_type=block.block_type,
                source_label=block.source_label,
            )
            midpoint = (block.y0 + block.y1) / 2.0
            in_window = top_cutoff <= midpoint < bottom_cutoff
            block_type = _classify_block_type(section_block, repeated_text_counts, page_tables, page_footnotes)
            section_block.block_type = block_type
            section_block.included = in_window and block_type == "narrative"
            section_block.exclusion_reason = "" if section_block.included else _exclusion_reason_for_block(block_type, in_window)
            if section_block.included:
                included_blocks.append(section_block)
            else:
                excluded_blocks.append(section_block)
    return SectionAudit(
        section_key=section.section_key,
        section_title=section.title,
        start_page=section.start_page,
        end_page=section.end_page,
        anchor_page=section.anchor_page,
        anchor_text=section.anchor_text,
        anchor_bbox_norm=section.anchor_bbox_norm,
        included_blocks=included_blocks,
        excluded_blocks=excluded_blocks,
        table_regions=_table_regions_for_pages(
            section.section_key,
            table_bboxes_by_page,
            footnote_bboxes_by_page,
            section.pages,
        ),
    )


def _markdown_blocks_for_section(section: SectionAudit) -> list[PDFBlock]:
    """Retourne les blocs à inclure dans le markdown source de vérité d'une section.

    Sélectionne les blocs narratifs inclus, plus les titres/en-têtes exclus pour
    raison ``non_narrative_block`` (afin de préserver la structure ``###`` dans le
    markdown). Les doublons sont dédupliqués par ``block_id``. Le résultat est
    trié par (page, line_number, y0) pour respecter l'ordre de lecture.
    """
    selected: list[PDFBlock] = []
    seen_ids: set[str] = set()
    for block in section.included_blocks:
        if block.block_id not in seen_ids:
            selected.append(block)
            seen_ids.add(block.block_id)
    for block in section.excluded_blocks:
        if block.block_id in seen_ids:
            continue
        if block.exclusion_reason != "non_narrative_block":
            continue
        if block.source_label not in {"title", "section_header"}:
            continue
        if _looks_like_table_or_financial_grid(block.text) or _looks_like_footnote(block.text):
            continue
        selected.append(block)
        seen_ids.add(block.block_id)
    return sorted(selected, key=lambda block: (block.page, block.line_number, block.y0))


def _build_text_extraction_markdown(section_audits: list[SectionAudit]) -> str:
    """Convertit une liste d'audits de sections en markdown source de vérité.

    Chaque section devient un bloc ``## Titre``. Les sous-titres détectés deviennent
    des blocs ``### Sous-titre`` positionnés juste avant le premier paragraphe qui
    les suit. Les doublons de titres et les titres identiques à la section parente
    sont supprimés. Le markdown produit est la seule entrée des appels GPT de
    comparaison ; il est aussi sauvegardé comme artefact d'audit lisible.
    """
    lines: list[str] = []
    for section in section_audits:
        lines.append(f"## {section.section_title}")
        lines.append("")
        seen_heading_norms: set[str] = set()
        section_title_norm = _normalized_block_text(section.section_title)
        pending_heading: str | None = None
        for block in _markdown_blocks_for_section(section):
            text = str(block.text or "").strip()
            if not text:
                continue
            norm = _normalized_block_text(text)
            if block.source_label in {"title", "section_header"}:
                if not norm or norm == section_title_norm or norm in seen_heading_norms:
                    continue
                pending_heading = text
                seen_heading_norms.add(norm)
                continue
            if pending_heading is not None:
                lines.append(f"### {pending_heading}")
                lines.append("")
                pending_heading = None
            lines.append(text)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_block_page_index(section: SectionAudit) -> list[tuple[int, str]]:
    """Retourne la liste ordonnée (page, texte) des blocs narratifs d'une section.

    Utilisée pour retrouver la page exacte d'un fragment de texte retourné par GPT.
    Les blocs titres/en-têtes sont exclus car GPT ne les retourne pas comme corps de texte.
    """
    index: list[tuple[int, str]] = []
    for block in _markdown_blocks_for_section(section):
        if block.source_label in {"title", "section_header"}:
            continue
        text = str(block.text or "").strip()
        if text:
            index.append((block.page, text))
    return index


def _find_page_for_fragment(fragment: str, block_index: list[tuple[int, str]]) -> int | None:
    """Retrouve la page du bloc qui correspond le mieux au fragment de texte GPT.

    Stratégie en deux passes :
    1. Correspondance substring normalisée (exact ou inclusion)
    2. Score de recouvrement de mots (retenu si ≥ 30 %)
    """
    if not fragment or not block_index:
        return None
    frag_norm = _normalized_block_text(fragment)
    if not frag_norm:
        return None

    # Pass 1 — substring
    for page, text in block_index:
        text_norm = _normalized_block_text(text)
        if frag_norm in text_norm or text_norm in frag_norm:
            return page

    # Pass 2 — word overlap
    frag_words = set(frag_norm.split())
    best_score = 0.0
    best_page: int | None = None
    for page, text in block_index:
        text_words = set(_normalized_block_text(text).split())
        if not text_words:
            continue
        overlap = len(frag_words & text_words)
        score = overlap / max(len(frag_words), len(text_words))
        if score > best_score:
            best_score = score
            best_page = page

    return best_page if best_score >= 0.30 else None


def _build_openai_client():
    """Instancie le client OpenAI avec la clé API du projet.

    Lève ``RuntimeError`` si la clé est absente — le pipeline texte ne peut pas
    fonctionner sans accès à l'API OpenAI.
    """
    from openai import OpenAI

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY absent: le pipeline texte GPT-first ne peut pas s'exécuter.")
    return OpenAI(api_key=api_key)


def _strip_markdown_fences(text: str) -> str:
    """Retire les clotures Markdown et isole l'objet JSON si present."""
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]
    return stripped


def _parse_json_object_response(raw: str) -> dict[str, Any]:
    """Parse un objet JSON reponse OpenAI et valide son type racine."""
    cleaned = _strip_markdown_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("OpenAI response is not a JSON object")
    return data


def _preview_response_text(raw: str, limit: int = 500) -> str:
    """Retourne un apercu compact de la reponse brute pour les logs."""
    text = (raw or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head} ... {tail}"


def _build_json_repair_messages(raw_response: str) -> list[dict[str, str]]:
    """Construit un mini-prompt de reparation pour un JSON mal forme."""
    return [
        {
            "role": "system",
            "content": (
                "Tu reçois une tentative de réponse JSON invalide. "
                "Réécris-la en un objet JSON valide uniquement, sans markdown, "
                "sans commentaire, sans texte avant ou après."
            ),
        },
        {
            "role": "user",
            "content": (
                "Répare cet objet JSON sans changer sa structure ni son sens. "
                "Si une chaîne est tronquée, ferme-la et ferme correctement les objets/listes.\n\n"
                f"{raw_response[:12000]}"
            ),
        },
    ]


def _allowed_target_sections(bank_code: str) -> set[str]:
    """Retourne l'ensemble des clés de sections autorisées pour une banque donnée.

    Si la banque n'a pas de configuration spécifique dans ``_TARGET_SECTIONS_BY_BANK``,
    toutes les sections du catalogue ``_SECTION_LABELS`` sont autorisées.
    """
    return set(_TARGET_SECTIONS_BY_BANK.get(str(bank_code or "").strip().lower(), set(_SECTION_LABELS)))


def _max_output_tokens_for_model(model: str, fallback: int = 16_384) -> int:
    """Retourne le plafond de sortie connu du modele quand il est connu.

    Ce plafond n'est pas utilise comme limite par defaut dans le flux texte:
    les appels normaux laissent le modele s'arreter naturellement. Il sert
    seulement de borne de retry quand un appel explicitement cappe finit en
    ``finish_reason="length"``.
    """
    normalized = str(model or "").strip()
    for pattern, limit in _MODEL_MAX_OUTPUT_TOKENS:
        if pattern.match(normalized):
            return limit
    return fallback


def _call_json_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Execute un appel JSON OpenAI robuste pour le flux texte.

    Politique du pipeline texte:
    - pas de limite de completion explicite par defaut, afin de privilegier un
      arret naturel et de reduire les JSON tronques;
    - retry au plafond connu du modele seulement si un plafond explicite plus
      bas a ete fourni et que la reponse finit en ``length``;
    - tentative de reparation du JSON avant de lever une erreur finale.

    Returns:
        Objet JSON racine valide sous forme de dictionnaire.
    """
    model_max_tokens = _max_output_tokens_for_model(model)
    initial_max_tokens = None if max_tokens is None else min(int(max_tokens), model_max_tokens)

    def _request(request_messages: list[dict[str, Any]], token_budget: int | None) -> tuple[str, str | None]:
        """Exécute un appel brut à l'API OpenAI et retourne ``(contenu, finish_reason)``."""
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if token_budget is not None:
            request_kwargs["max_completion_tokens"] = token_budget

        response = client.chat.completions.create(
            **request_kwargs,
        )
        choice = response.choices[0]
        payload = choice.message.content or "{}"
        finish_reason = getattr(choice, "finish_reason", None)
        return payload, finish_reason

    raw_payload, finish_reason = _request(messages, initial_max_tokens)
    try:
        return _parse_json_object_response(raw_payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "OpenAI returned invalid JSON (finish_reason=%s): %s | preview=%s",
            finish_reason,
            exc,
            _preview_response_text(raw_payload),
        )

    if finish_reason == "length" and initial_max_tokens is not None and initial_max_tokens < model_max_tokens:
        retry_max_tokens = model_max_tokens
        raw_retry, retry_finish_reason = _request(messages, retry_max_tokens)
        try:
            logger.warning(
                "Retrying OpenAI JSON call after truncation with max_completion_tokens=%s (model_max=%s)",
                retry_max_tokens,
                model_max_tokens,
            )
            return _parse_json_object_response(raw_retry)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Retry after truncation still returned invalid JSON (finish_reason=%s): %s | preview=%s",
                retry_finish_reason,
                exc,
                _preview_response_text(raw_retry),
            )
            raw_payload = raw_retry
            finish_reason = retry_finish_reason
    elif finish_reason == "length":
        logger.warning(
            "OpenAI response hit the effective output ceiling (%s) and remained truncated",
            model_max_tokens,
        )

    repair_messages = _build_json_repair_messages(raw_payload)
    raw_repair, repair_finish_reason = _request(repair_messages, None)
    try:
        return _parse_json_object_response(raw_repair)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "OpenAI JSON parse failed after repair attempt "
            f"(finish_reason={repair_finish_reason or finish_reason or 'unknown'}): {exc}. "
            f"Preview: {_preview_response_text(raw_repair)}"
        ) from exc


def _resolve_sections(pdf_path: Path, bank_code: str) -> dict[str, ResolvedSection]:
    """Localise et normalise les sections textuelles cibles d'un PDF.

    Le resultat est deja filtre selon les sections autorisees pour la banque,
    ce qui stabilise le flux de comparaison inter-trimestrielle.
    """
    mapping = locate_sections_in_pdf(str(pdf_path), bank_code.lower())
    allowed_sections = _allowed_target_sections(bank_code)
    sections: dict[str, ResolvedSection] = {}
    for item in getattr(mapping, "sections", []) or []:
        canonical = canonicalize_section(getattr(item, "section_type", ""))
        section_key = _CANONICAL_TO_TEXT_KEY.get(canonical)
        if not section_key or section_key not in allowed_sections or section_key in sections:
            continue
        start_page = int(getattr(item, "start_page", 0) or 0)
        end_page = int(getattr(item, "end_page", 0) or 0)
        if start_page <= 0 or end_page < start_page:
            continue
        sections[section_key] = ResolvedSection(
            section_key=section_key,
            title=_SECTION_LABELS[section_key],
            start_page=start_page,
            end_page=end_page,
            anchor_page=int(getattr(item, "anchor_page", 0) or 0) or None,
            anchor_text=str(getattr(item, "anchor_text", "") or "") or None,
            anchor_bbox_norm=list(getattr(item, "anchor_bbox_norm", []) or []) or None,
        )
    return sections




def _extract_audits_for_pdf(
    *,
    pdf_path: Path,
    sections: dict[str, ResolvedSection],
) -> list[SectionAudit]:
    """Extrait les blocs narratifs de chaque section via Docling + heuristiques.

    Cette etape construit les audits qui seront convertis en markdown
    ``source of truth``. Les appels GPT de comparaison/triage relisent ensuite
    exclusivement ce markdown, pas les PDFs directement.
    """
    if not sections:
        return []
    section_order = _next_section_by_key(sections)
    unique_pages = sorted({page for section in sections.values() for page in section.pages})
    page_blocks, table_bboxes_by_page, footnote_bboxes_by_page = _extract_docling_page_blocks(pdf_path, unique_pages)
    repeated_counts = _repeated_text_counts(page_blocks)
    audits: list[SectionAudit] = []
    for section_key, section in sections.items():
        audit = _build_section_audit(
            section=section,
            next_section=section_order.get(section_key),
            page_blocks=page_blocks,
            repeated_text_counts=repeated_counts,
            table_bboxes_by_page=table_bboxes_by_page,
            footnote_bboxes_by_page=footnote_bboxes_by_page,
        )
        audits.append(audit)
    return audits


def _extract_section_text_from_markdown(md_content: str, section_key: str) -> str:
    """Extrait le texte d'une section depuis le markdown auditable.

    Le markdown ecrit sur disque est la source de verite fonctionnelle du flux
    texte; ce helper redecoupe exactement ce meme contenu avant comparaison.
    """
    title = _SECTION_LABELS.get(section_key, section_key)
    lines = md_content.splitlines()
    in_section = False
    section_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if in_section:
                break
            if line.strip() == f"## {title}":
                in_section = True
                continue
        if in_section:
            section_lines.append(line)
    return "\n".join(section_lines).strip()


def _normalize_heading(heading: str) -> str:
    """Normalise un heading ### pour le pairing T1/T2 (insensible à la casse et aux préfixes de tableaux)."""
    h = heading.lower()
    h = re.sub(r"\b[tT]\d{2,3}\b\s*", "", h)  # strip T22, T25, T125, etc.
    h = re.sub(r"[^\w\s]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _parse_subsections(md_text: str) -> list[tuple[str, str]]:
    """Découpe un texte markdown en paires (heading, body).

    Le texte avant le premier ### devient (``__intro__``, body).
    Les headings ## de section ne sont pas inclus.
    """
    parts = _SUBSECTION_SPLIT_RE.split(md_text)
    result: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        result.append(("__intro__", intro))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if heading:
            result.append((heading, body))
    return result


def _pair_subsections(
    subs_t1: list[tuple[str, str]],
    subs_t2: list[tuple[str, str]],
) -> list[tuple[str | None, str, str | None, str]]:
    """Paire les sous-sections T1 et T2 par heading normalisé.

    Retourne une liste de ``(heading_t1, body_t1, heading_t2, body_t2)``.
    ``None`` pour un heading signifie qu'il n'a pas de contrepartie dans l'autre trimestre.
    """
    norm_to_t2: dict[str, tuple[str, str]] = {
        _normalize_heading(h): (h, body) for h, body in subs_t2
    }
    matched_t2_norms: set[str] = set()
    pairs: list[tuple[str | None, str, str | None, str]] = []
    for h1, body1 in subs_t1:
        norm = _normalize_heading(h1)
        if norm in norm_to_t2:
            h2, body2 = norm_to_t2[norm]
            pairs.append((h1, body1, h2, body2))
            matched_t2_norms.add(norm)
        else:
            pairs.append((h1, body1, None, ""))
    for h2, body2 in subs_t2:
        if _normalize_heading(h2) not in matched_t2_norms:
            pairs.append((None, "", h2, body2))
    return pairs


def _gpt_match_orphan_headings(
    *,
    client: Any,
    model: str,
    section_key: str,
    orphans_t1: list[str],
    orphans_t2: list[str],
) -> list[dict[str, Any]]:
    """Identifie via GPT les sous-sections renommées entre T1 et T2 (1→1 uniquement).

    Retourne une liste de dicts ``{"heading_t1": ..., "heading_t2": ...,
    "confidence": "high|medium", "reason": ...}`` dont les deux côtés
    appartiennent bien aux listes orphelines (anti-hallucination).
    En cas d'erreur, retourne [] pour ne pas bloquer le pipeline.
    """
    if not orphans_t1 or not orphans_t2:
        return []
    try:
        raw = _call_json_completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es expert en rapports bancaires réglementaires canadiens. "
                        "Tu identifies les correspondances entre sous-sections renommées "
                        "d'un trimestre à l'autre."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        'Format de réponse: {"matches": [{"heading_t1": "...", "heading_t2": "...", '
                        '"confidence": "high|medium|low", "reason": "..."}]}\n'
                        "Règles strictes:\n"
                        "- Correspondances 1→1 uniquement (un heading T1 ↔ un heading T2)\n"
                        "- N'inclure que confidence high ou medium\n"
                        "- Si tu n'es pas sûr, ne pas inclure la paire\n"
                        "- Retourner les headings EXACTEMENT comme fournis\n\n"
                        f"Section: {section_key}\n\n"
                        "Sous-sections T1 sans correspondance exacte:\n"
                        + "\n".join(f"- {h}" for h in orphans_t1)
                        + "\n\nSous-sections T2 sans correspondance exacte:\n"
                        + "\n".join(f"- {h}" for h in orphans_t2)
                    ),
                },
            ],
        )
        orphans_t1_set = set(orphans_t1)
        orphans_t2_set = set(orphans_t2)
        used_t1: set[str] = set()
        used_t2: set[str] = set()
        matches = []
        for m in raw.get("matches") or []:
            conf = str(m.get("confidence") or "").lower()
            h1 = m.get("heading_t1") or ""
            h2 = m.get("heading_t2") or ""
            if conf not in {"high", "medium"}:
                continue
            if h1 not in orphans_t1_set or h2 not in orphans_t2_set:
                continue
            if h1 in used_t1 or h2 in used_t2:
                continue
            matches.append(m)
            used_t1.add(h1)
            used_t2.add(h2)
        return matches
    except Exception:
        logger.warning("Orphan heading GPT match failed for %s — skipping", section_key)
        return []


def _synthetic_subsection_change(
    *,
    section_key: str,
    diff_type: str,
    heading: str,
    body_t1: str,
    body_t2: str,
    idx: int,
) -> dict[str, Any]:
    """Crée un enregistrement de changement pour une sous-section entièrement ajoutée ou supprimée."""
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(heading))[:40].strip("_")
    label = "ajoutée" if diff_type == "added" else "supprimée"
    return {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": heading,
        "diff_type": diff_type,
        "semantic_text_t1": _sanitize_semantic_text(body_t1),
        "semantic_text_t2": _sanitize_semantic_text(body_t2),
        "source_text_t1": body_t1,
        "source_text_t2": body_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown",
        "source_resolution_t2": "markdown",
        "evidence_t1": {"pages": [], "snippet": body_t1[:400]},
        "evidence_t2": {"pages": [], "snippet": body_t2[:400]},
        "change_summary": f"Sous-section {label}: {heading}",
    }


def _compare_texts_single_call(
    *,
    client: Any,
    model: str,
    section_key: str,
    heading_label: str,
    heading_slug: str,
    text_t1: str,
    text_t2: str,
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Appel GPT unique pour comparer deux corps de texte.

    Extrait la logique de comparaison GPT de ``_compare_section_texts`` pour
    permettre son appel répété par sous-section.
    """
    try:
        raw = _call_json_completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu compares deux versions d'une section de rapport bancaire. "
                        "Identifie les changements réels paragraphe par paragraphe. "
                        "Ignore les reformulations purement cosmétiques ou rédactionnelles."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Compare ces deux versions et retourne uniquement du JSON.\n"
                        'Format: {"changes":[{"diff_type":"unchanged|modified|added|removed",'
                        '"text_t1":"texte du paragraphe en T1, vide si added",'
                        '"text_t2":"texte du paragraphe en T2, vide si removed",'
                        '"change_summary":"explication concise du changement"}]}.\n'
                        "unchanged = même idée, légère reformulation sans évolution réelle.\n"
                        "modified = même idée mais vraie évolution du contenu ou de la nuance.\n"
                        "added = idée nouvelle présente uniquement en T2.\n"
                        "removed = idée présente en T1, absente en T2.\n"
                        f"Section: {section_key}\n\n"
                        f"=== T1 ===\n{text_t1}\n\n"
                        f"=== T2 ===\n{text_t2}\n"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Section comparison failed for {section_key}/{heading_slug}: {exc}") from exc

    validated: list[dict[str, Any]] = []
    for local_idx, item in enumerate(raw.get("changes") or [], start=1):
        diff_type = str(item.get("diff_type") or "").strip().lower()
        if diff_type not in {"unchanged", "modified", "added", "removed"}:
            continue
        text_t1_item = str(item.get("text_t1") or "").strip()
        text_t2_item = str(item.get("text_t2") or "").strip()
        if diff_type in {"unchanged", "modified"} and not (text_t1_item and text_t2_item):
            continue
        if diff_type == "added" and not text_t2_item:
            continue
        if diff_type == "removed" and not text_t1_item:
            continue
        global_idx = idx_offset + local_idx
        validated.append(
            {
                "change_id": f"{section_key}_{heading_slug}_change_{global_idx:03d}",
                "section_key": section_key,
                "subsection_heading": heading_label,
                "diff_type": diff_type,
                "semantic_text_t1": _sanitize_semantic_text(text_t1_item),
                "semantic_text_t2": _sanitize_semantic_text(text_t2_item),
                "source_text_t1": text_t1_item,
                "source_text_t2": text_t2_item,
                "source_block_ids_t1": [],
                "source_block_ids_t2": [],
                "source_refs_t1": [],
                "source_refs_t2": [],
                "pages_t1": [],
                "pages_t2": [],
                "source_resolution_t1": "markdown",
                "source_resolution_t2": "markdown",
                "evidence_t1": {"pages": [], "snippet": text_t1_item[:400]},
                "evidence_t2": {"pages": [], "snippet": text_t2_item[:400]},
                "change_summary": _sanitize_explanation(str(item.get("change_summary") or "")),
            }
        )
    return validated


def _compare_section_texts(
    *,
    client: Any,
    model: str,
    section_key: str,
    text_t1: str,
    text_t2: str,
) -> list[dict[str, Any]]:
    """Compare deux sections markdown T1/T2 sous-section par sous-section.

    Le texte est découpé selon les headings ### existants. Chaque paire de
    sous-sections fait l'objet d'un appel GPT séparé, évitant les dépassements
    de contexte sur les grandes sections comme ``Gestion des risques``.

    Les sous-sections sans contrepartie sont marquées ajoutées ou supprimées
    sans appel GPT. Si le texte ne contient aucune sous-section, un seul appel
    GPT est lancé sur le texte entier (comportement précédent).
    """
    if not text_t1.strip() and not text_t2.strip():
        return []

    subs_t1 = _parse_subsections(text_t1)
    subs_t2 = _parse_subsections(text_t2)

    # No subsections on either side — fall back to single call (legacy behaviour)
    if len(subs_t1) <= 1 and len(subs_t2) <= 1:
        return _compare_texts_single_call(
            client=client,
            model=model,
            section_key=section_key,
            heading_label="",
            heading_slug="full",
            text_t1=text_t1,
            text_t2=text_t2,
            idx_offset=0,
        )

    pairs = _pair_subsections(subs_t1, subs_t2)

    # Phase 2 — Fuzzy rename resolution via GPT
    orphans_t1 = [h1 for h1, _b1, h2, _b2 in pairs if h2 is None and h1 is not None]
    orphans_t2 = [h2 for _h1, _b1, h2, _b2 in pairs if _h1 is None and h2 is not None]
    rename_matches = _gpt_match_orphan_headings(
        client=client,
        model=model,
        section_key=section_key,
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    # Build lookup: orphan_t1_heading → matched orphan_t2_heading (and vice versa)
    rename_t1_to_t2: dict[str, str] = {m["heading_t1"]: m["heading_t2"] for m in rename_matches}
    renamed_as_t2: set[str] = set(rename_t1_to_t2.values())

    # Augment pairs: replace (h1, body1, None, "") with (h1, body1, h2_matched, body2_matched)
    body_by_t2_heading: dict[str, str] = {h2: b2 for _, _, h2, b2 in pairs if h2 is not None}
    resolved_pairs: list[tuple[str | None, str, str | None, str]] = []
    for h1, body1, h2, body2 in pairs:
        if h2 is None and h1 is not None and h1 in rename_t1_to_t2:
            matched_h2 = rename_t1_to_t2[h1]
            matched_body2 = body_by_t2_heading.get(matched_h2, "")
            resolved_pairs.append((h1, body1, matched_h2, matched_body2))
        elif h1 is None and h2 is not None and h2 in renamed_as_t2:
            # Skip — already consumed by the T1 side above
            continue
        else:
            resolved_pairs.append((h1, body1, h2, body2))
    pairs = resolved_pairs

    all_changes: list[dict[str, Any]] = []
    global_idx = 1
    renamed_pairs: set[tuple[str, str]] = {(m["heading_t1"], m["heading_t2"]) for m in rename_matches}

    for h1, body1, h2, body2 in pairs:
        heading_label = h1 or h2 or "unknown"
        heading_slug = re.sub(r"[^\w]+", "_", _normalize_heading(heading_label))[:40].strip("_")

        # For renamed pairs, display as "T1 heading → T2 heading"
        if h1 is not None and h2 is not None and (h1, h2) in renamed_pairs:
            heading_label = f"{h1} → {h2}"
            heading_slug = re.sub(r"[^\w]+", "_", _normalize_heading(h1))[:40].strip("_")

        if h2 is None:
            assert h1 is not None
            all_changes.append(
                _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type="removed",
                    heading=h1,
                    body_t1=body1,
                    body_t2="",
                    idx=global_idx,
                )
            )
            global_idx += 1
            continue

        if h1 is None:
            assert h2 is not None
            all_changes.append(
                _synthetic_subsection_change(
                    section_key=section_key,
                    diff_type="added",
                    heading=h2,
                    body_t1="",
                    body_t2=body2,
                    idx=global_idx,
                )
            )
            global_idx += 1
            continue

        if not body1.strip() and not body2.strip():
            continue

        subsection_changes = _compare_texts_single_call(
            client=client,
            model=model,
            section_key=section_key,
            heading_label=heading_label,
            heading_slug=heading_slug,
            text_t1=body1,
            text_t2=body2,
            idx_offset=global_idx - 1,
        )
        all_changes.extend(subsection_changes)
        global_idx += len(subsection_changes)

    return all_changes


def _default_triage() -> dict[str, Any]:
    """Retourne un triage par défaut conservateur (non pertinent, cosmétique, mineur).

    Utilisé comme valeur de repli quand le modèle ne retourne pas de triage pour
    un changement donné, ou quand l'appel GPT de triage échoue partiellement.
    """
    return {
        "is_relevant": False,
        "category": "COSMETIQUE",
        "impact_level": "MINEUR",
        "risk_type": "autre",
        "relevance_score": "FAIBLE",
        "risk_level": "FAIBLE",
        "explanation": "",
        "impact_description": "",
        "action_requise": "aucune",
        "reference_reglementaire": "",
        "nouvelle_idee": False,
        "confidence": 0.0,
        "source": "gpt4o_triage",
        "signals": {
            "regulatory_reference_added": False,
            "methodology_change": False,
            "tone_changed": False,
            "forward_looking": False,
            "quantitative_changed": False,
        },
    }


def _triage_section_changes(
    *,
    client: Any,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Qualifie metier les changements detectes et fusionne le triage.

    Le triage ne recalcule pas la diff textuelle: il prend les changements deja
    identifies, demande une qualification selective au modele, puis rattache le
    resultat a chaque changement pour la retention finale et le resume global.
    """
    if not changes:
        return []
    triage_inputs = []
    for idx, change in enumerate(changes, start=1):
        triage_inputs.append(
            {
                "change_index": idx,
                "diff_type": change["diff_type"],
                "semantic_text_t1": change.get("semantic_text_t1", ""),
                "semantic_text_t2": change.get("semantic_text_t2", ""),
                "source_snippet_t1": (change.get("source_text_t1") or "")[:300],
                "source_snippet_t2": (change.get("source_text_t2") or "")[:300],
                "change_summary": change.get("change_summary", ""),
            }
        )
    try:
        raw = _call_json_completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un analyste senior spécialisé dans la surveillance des rapports bancaires réglementaires. "
                        "Tu classifies et qualifies les changements détectés entre deux trimestres. "
                        "Tu as accès au texte source complet (avec les chiffres réels) pour chaque changement."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Retourne uniquement du JSON.\n"
                        'Format: {"triages":[{"change_index":1,"is_relevant":true,'
                        ' "category":"REGLEMENTAIRE|RISQUE|CAPITAL|STRUCTURE|COSMETIQUE",'
                        ' "impact_level":"MAJEUR|MODERE|MINEUR",'
                        ' "action_requise":"escalade|investigation|confirmation|information|aucune",'
                        ' "nouvelle_idee":true, "explanation":"...", "impact_description":"...",'
                        ' "risk_type":"credit|marche|liquidite|capital|conformite|autre",'
                        ' "signals":{"regulatory_reference_added":false,"methodology_change":false,'
                        ' "tone_changed":false,"forward_looking":false,"quantitative_changed":false}}]}.\n\n'
                        "Règles de classification:\n"
                        "- COSMETIQUE uniquement pour: mises à jour de dates pures (janvier → avril), "
                        "reformulations strictement identiques sans nouvel élément, références de pages.\n"
                        "- REGLEMENTAIRE ou CAPITAL pour tout changement de montant ou ratio lié à la "
                        "réglementation prudentielle (fonds propres, ratios, expositions, provisions, "
                        "seuils réglementaires) — même si le changement est purement quantitatif.\n"
                        "- RISQUE pour toute évolution de tendance, d'exposition ou de posture face à un risque.\n"
                        "- Une inversion de tendance ('diminué au lieu d'augmenté', 'baisse → hausse') "
                        "est au minimum MODERE.\n"
                        "- N'utilise pas d'acronymes prudentiels dans l'explication; reformule en langage "
                        "métier générique.\n\n"
                        "Exigence pour `explanation` (obligatoire si is_relevant=true):\n"
                        "Rédige exactement 3 phrases complètes en français:\n"
                        "1. Ce qui a changé concrètement entre T1 et T2 (faits précis, chiffres si pertinents).\n"
                        "2. Pourquoi ce changement est substantiel et non cosmétique.\n"
                        "3. Ce que cela implique pour la surveillance de cette banque.\n\n"
                        f"Section: {section_key}\n{_json_dumps(triage_inputs)}"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Section triage failed for {section_key}: {exc}") from exc

    triage_map: dict[int, dict[str, Any]] = {}
    for item in raw.get("triages") or []:
        try:
            idx = int(item.get("change_index"))
        except (TypeError, ValueError):
            continue
        triage = _default_triage()
        triage.update(
            {
                "is_relevant": bool(item.get("is_relevant", False)),
                "category": str(item.get("category") or "COSMETIQUE").upper(),
                "impact_level": str(item.get("impact_level") or "MINEUR").upper(),
                "risk_type": str(item.get("risk_type") or "autre").lower(),
                "explanation": _sanitize_explanation(str(item.get("explanation") or "")),
                "impact_description": _sanitize_explanation(str(item.get("impact_description") or "")),
                "action_requise": str(item.get("action_requise") or "aucune").lower(),
                "nouvelle_idee": bool(item.get("nouvelle_idee", False)),
                "source": "gpt4o_triage",
                "signals": {
                    "regulatory_reference_added": bool(
                        (item.get("signals") or {}).get("regulatory_reference_added", False)
                    ),
                    "methodology_change": bool((item.get("signals") or {}).get("methodology_change", False)),
                    "tone_changed": bool((item.get("signals") or {}).get("tone_changed", False)),
                    "forward_looking": bool((item.get("signals") or {}).get("forward_looking", False)),
                    "quantitative_changed": bool((item.get("signals") or {}).get("quantitative_changed", False)),
                },
            }
        )
        triage_map[idx] = triage

    enriched: list[dict[str, Any]] = []
    for idx, change in enumerate(changes, start=1):
        triage = triage_map.get(idx, _default_triage())
        triage["nouvelle_idee"] = _compute_conservative_new_idea(change, triage)
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return enriched


def _build_global_summary(section_comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrège les statistiques de toutes les sections en un résumé global.

    Calcule les comptages par impact, catégorie et action requise, extrait
    les cinq premiers résumés de changements comme points saillants, et détermine
    la pertinence globale (FAIBLE / MOYENNE / ELEVEE) selon le nombre de changements
    majeurs. Utilisé pour les deux champs ``global_summary`` et ``all_changes_summary``
    du payload final.
    """
    all_changes = [
        block
        for section in section_comparisons
        for block in (section.get("block_comparisons") or [])
    ]
    by_impact: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_action: dict[str, int] = {}
    highlights: list[str] = []

    for change in all_changes:
        triage = change.get("genai_triage") or {}
        impact = str(triage.get("impact_level") or "MINEUR").upper()
        category = str(triage.get("category") or "INCONNU").upper()
        action = str(triage.get("action_requise") or "aucune").lower()
        by_impact[impact] = by_impact.get(impact, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
        summary = str(change.get("change_summary") or "").strip()
        if summary and len(highlights) < 5:
            highlights.append(summary)

    overview = (
        "Aucun changement textuel non cosmétique retenu."
        if not all_changes
        else f"{len(all_changes)} changement(s) textuel(s) non cosmétique(s) retenu(s) pour revue experte."
    )
    pertinence = "FAIBLE"
    if by_impact.get("MAJEUR", 0) >= 3:
        pertinence = "ELEVEE"
    elif all_changes:
        pertinence = "MOYENNE"

    return {
        "executive_overview": overview,
        "key_highlights": highlights,
        "pertinence_globale": pertinence,
        "counts": {
            "total": len(all_changes),
            "total_relevant": len(all_changes),
            "by_impact": by_impact,
            "by_category": by_category,
            "by_action": by_action,
        },
    }


def run_text_analysis_pipeline(
    *,
    bank_code: str,
    year_current: int,
    quarter_current: str,
    pdf_previous: Path,
    pdf_current: Path,
    out_root: Path,
    model: str = "gpt-4o",
    allowed_section_keys: set[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Execute le pipeline texte complet avec le markdown comme source de verite.

    Flow:
    1. Locate sections in both PDFs.
    2. Extract narrative blocks via Docling + heuristics → build .md (source of truth).
    3. Feed .md sections directly to GPT-4o for comparison (added/modified/removed).
    4. Triage and retain non-cosmetic changes sorted by business impact.

    Notes:
    - les fichiers ``text_extraction_*.md`` ecrits en sortie sont exactement les
      artefacts relus par le flux GPT de comparaison;
    - les appels GPT du flux texte n'envoient pas de limite de completion
      explicite par defaut, afin de privilegier l'arret naturel du modele.
    """
    quarter_current = normalize_quarter(quarter_current)
    year_previous, quarter_previous = resolve_previous_quarter(year_current, quarter_current)
    bank_code = bank_code.lower()

    sections_previous = _resolve_sections(pdf_previous, bank_code)
    sections_current = _resolve_sections(pdf_current, bank_code)
    section_keys = sorted(set(sections_previous) | set(sections_current))
    if allowed_section_keys is not None:
        section_keys = [key for key in section_keys if key in allowed_section_keys]
    if not section_keys:
        raise TextAnalysisQualityError("Aucune section texte ciblée localisée dans les rapports.")

    # Step 1: Docling extraction + heuristic classification → audits
    audits_previous = _extract_audits_for_pdf(
        pdf_path=pdf_previous,
        sections={key: sections_previous[key] for key in section_keys if key in sections_previous},
    )
    audits_current = _extract_audits_for_pdf(
        pdf_path=pdf_current,
        sections={key: sections_current[key] for key in section_keys if key in sections_current},
    )

    # Step 2: Build .md — source of truth (human-auditable + GPT input)
    md_previous = _build_text_extraction_markdown(audits_previous)
    md_current = _build_text_extraction_markdown(audits_current)

    # Validate: at least one period must have content per section
    for section_key in section_keys:
        text_t1 = _extract_section_text_from_markdown(md_previous, section_key)
        text_t2 = _extract_section_text_from_markdown(md_current, section_key)
        if not text_t1.strip() and not text_t2.strip():
            raise TextAnalysisQualityError(
                f"Section vide dans les deux rapports: {section_key}"
            )

    # Step 3: GPT comparison on .md sections
    client = _build_openai_client()
    audit_by_key_prev: dict[str, SectionAudit] = {a.section_key: a for a in audits_previous}
    audit_by_key_curr: dict[str, SectionAudit] = {a.section_key: a for a in audits_current}
    section_comparisons: list[dict[str, Any]] = []
    for section_key in section_keys:
        text_t1 = _extract_section_text_from_markdown(md_previous, section_key)
        text_t2 = _extract_section_text_from_markdown(md_current, section_key)
        changes = _compare_section_texts(
            client=client,
            model=model,
            section_key=section_key,
            text_t1=text_t1,
            text_t2=text_t2,
        )
        # Inject exact page numbers by matching GPT text fragments back to PDFBlocks.
        # Falls back to section start_page if no block match is found.
        _audit_prev = audit_by_key_prev.get(section_key)
        _audit_curr = audit_by_key_curr.get(section_key)
        _block_idx_t1 = _build_block_page_index(_audit_prev) if _audit_prev else []
        _block_idx_t2 = _build_block_page_index(_audit_curr) if _audit_curr else []
        _fallback_t1 = _audit_prev.start_page if _audit_prev else None
        _fallback_t2 = _audit_curr.start_page if _audit_curr else None
        for _ch in changes:
            _dt = _ch.get("diff_type", "")
            _txt_t1 = _ch.get("source_text_t1") or ""
            _txt_t2 = _ch.get("source_text_t2") or ""
            if _dt in {"modified", "removed", "unchanged"} and _txt_t1:
                _pg = _find_page_for_fragment(_txt_t1, _block_idx_t1)
                _p_t1: list[int] = [_pg] if _pg else ([_fallback_t1] if _fallback_t1 else [])
            else:
                _p_t1 = []
            if _dt in {"modified", "added", "unchanged"} and _txt_t2:
                _pg = _find_page_for_fragment(_txt_t2, _block_idx_t2)
                _p_t2: list[int] = [_pg] if _pg else ([_fallback_t2] if _fallback_t2 else [])
            else:
                _p_t2 = []
            _ch["pages_t1"] = _p_t1
            _ch["pages_t2"] = _p_t2
            if isinstance(_ch.get("evidence_t1"), dict):
                _ch["evidence_t1"]["pages"] = _p_t1
            if isinstance(_ch.get("evidence_t2"), dict):
                _ch["evidence_t2"]["pages"] = _p_t2

        non_unchanged = [change for change in changes if change.get("diff_type") != "unchanged"]
        enriched = _triage_section_changes(
            client=client,
            model=model,
            section_key=section_key,
            changes=non_unchanged,
        )
        all_changes = [dict(change) for change in enriched]
        all_changes.sort(key=_retained_change_sort_key)
        retained = [change for change in enriched if _is_non_cosmetic_change(change["genai_triage"])]
        retained.sort(key=_retained_change_sort_key)
        section_comparisons.append(
            {
                "section_key": section_key,
                "section_title": _SECTION_LABELS.get(section_key, section_key),
                "block_comparisons": retained,
                "all_block_comparisons": all_changes,
                "summary": {
                    "retained_changes": len(retained),
                    "all_changes": len(all_changes),
                    "pages_previous": [s.start_page for s in [sections_previous.get(section_key)] if s]
                    + [s.end_page for s in [sections_previous.get(section_key)] if s],
                    "pages_current": [s.start_page for s in [sections_current.get(section_key)] if s]
                    + [s.end_page for s in [sections_current.get(section_key)] if s],
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": UNIFIED_TEXT_SCHEMA_VERSION,
        "artifact_type": "text_comparison",
        "pipeline": "gpt4o_markdown_source_of_truth",
        "bank_code": bank_code,
        "year_previous": year_previous,
        "quarter_previous": f"{year_previous}_{quarter_previous}",
        "year_current": year_current,
        "quarter_current": f"{year_current}_{quarter_current}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "extraction_artifact_t1": f"text_extraction_{year_previous}_{quarter_previous}.md",
        "extraction_artifact_t2": f"text_extraction_{year_current}_{quarter_current}.md",
        "section_comparisons": section_comparisons,
    }
    payload["global_summary"] = _build_global_summary(section_comparisons)
    payload["all_changes_summary"] = _build_global_summary(
        [
            {
                "section_key": section["section_key"],
                "block_comparisons": section.get("all_block_comparisons") or [],
            }
            for section in section_comparisons
        ]
    )

    out_path = get_text_comparison_path(
        out_root=out_root,
        bank_code=bank_code,
        year_t2=year_current,
        quarter_t2=quarter_current,
        year_t1=year_previous,
        quarter_t1=quarter_previous,
    )
    out_dir = out_path.parent
    write_text_extraction_markdown(
        md_previous,
        get_text_extraction_markdown_path(out_dir, f"{year_previous}_{quarter_previous}"),
    )
    write_text_extraction_markdown(
        md_current,
        get_text_extraction_markdown_path(out_dir, f"{year_current}_{quarter_current}"),
    )
    write_text_comparison(payload, out_path)
    return payload, out_path
