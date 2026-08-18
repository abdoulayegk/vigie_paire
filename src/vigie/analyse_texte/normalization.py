"""Normalisation et classification déterministes des blocs textuels.

Le module nettoie le texte et détecte notamment les tableaux, notes de bas de
page, éléments répétitifs et chevauchements géométriques. Ces contrôles
centralisés précèdent la production du Markdown et les appels sémantiques.
"""

from __future__ import annotations

import json
import re
from typing import Any

from vigie.analyse_texte.canonical_cleanup import is_quarterly_running_chrome
from vigie.analyse_texte.constants import (
    _BPS_RE,
    _FOOTNOTE_MARKER_RE,
    _MULTISPACE_RE,
    _NUMERIC_TOKEN_RE,
    _PERCENT_RE,
    _PUNCT_SPACING_RE,
    _REGULATORY_REF_RE,
    _ROMAN_NUMERAL_RE,
    _SEMANTIC_REPLACEMENTS,
    _TABLE_HEADING_RE,
    _TABLE_ROW_MARKER_RE,
    _TABLE_VALUE_RE,
)
from vigie.analyse_texte.models import PDFBlock

_NOT_APPLICABLE_MARKER_RE = re.compile(
    r"^\s*(?:(?:\[\s*(?:x|X)?\s*\]|[-*•‰])\s*)?s\.?\s*o\.?\s*$",
    flags=re.IGNORECASE,
)
_TABLE_UNIT_LABEL_RE = re.compile(
    r"^\s*\(?\s*en\s+(?:milliers|millions|milliards)\s+de\s+"
    r"dollars(?:\s+canadiens)?\s*\)?\s*$",
    flags=re.IGNORECASE,
)
_TABLE_CAPTION_UNIT_RE = re.compile(
    r"\(\s*en\s+(?:milliers|millions|milliards)\s+de\s+dollars",
    flags=re.IGNORECASE,
)
_NUMBERED_TABLE_CAPTION_RE = re.compile(
    r"^\s*T\d+\b.+" + _TABLE_CAPTION_UNIT_RE.pattern,
    flags=re.IGNORECASE,
)
_REPORT_BANK_NAME_PATTERN = (
    r"(?:"
    r"banque\s+(?:nationale\s+du\s+canada|scotia|royale\s+du\s+canada|td)"
    r"|bmo(?:\s+groupe\s+financier)?"
    r"|groupe\s+banque\s+td"
    r"|cibc"
    r"|rbc"
    r"|bns"
    r"|bnc"
    r"|td"
    r")"
)
_REPORT_TITLE_PATTERN = (
    r"(?:\d{1,3}e\s+)?rapport\s+"
    r"(?:annuel|du\s+(?:premier|deuxi[eè]me|troisi[eè]me|quatri[eè]me)\s+"
    r"trimestre(?:\s+de)?)"
)
_REPORT_PAGE_CHROME_SUFFIX_PATTERN = r"(?:\s+rapport\s+de\s+gestion)?"
_BANK_FIRST_REPORT_PAGE_CHROME_RE = re.compile(
    r"^\s*\d{1,3}\s*(?:\|\s*)?"
    + _REPORT_BANK_NAME_PATTERN
    + r"\s*(?:[-–—]\s*)?"
    + _REPORT_TITLE_PATTERN
    + r"\s*(?:[-–—]\s*)?20\d{2}"
    + _REPORT_PAGE_CHROME_SUFFIX_PATTERN
    + r"\s*$",
    flags=re.IGNORECASE,
)
_REPORT_FIRST_BANK_PAGE_CHROME_RE = re.compile(
    r"^\s*\d{1,3}\s*(?:\|\s*)?"
    + _REPORT_TITLE_PATTERN
    + r"\s*(?:[-–—]\s*)?"
    + _REPORT_BANK_NAME_PATTERN
    + r"\s*(?:[-–—]\s*)?20\d{2}"
    + _REPORT_PAGE_CHROME_SUFFIX_PATTERN
    + r"\s*$",
    flags=re.IGNORECASE,
)
_BNC_ANNUAL_REPORT_CHROME_RE = re.compile(
    r"^\s*banque\s+nationale\s+du\s+canada\s+rapport\s+annuel\s+"
    r"20\d{2}(?:\s+\d{1,3})?\s*$",
    flags=re.IGNORECASE,
)
_BNC_MANAGEMENT_RUNNING_HEADER_RE = re.compile(
    r"^\s*rapport\s+de\s+gestion(?:\s+gestion\s+"
    r"(?:du\s+capital|des\s+risques))?\s*$",
    flags=re.IGNORECASE,
)
_CHART_AXIS_LABEL_ROW_RE = re.compile(r"^\s*(?:\(?-?\d+(?:[,.]\d+)?\)?\s+){11,}\(?-?\d+(?:[,.]\d+)?\)?\s*$")


def _json_dumps(data: Any) -> str:
    """Sérialise ``data`` en JSON indenté avec support complet des caractères UTF-8."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def _sanitize_semantic_text(text: str) -> str:
    """Normalise un texte pour la comparaison sémantique inter-trimestrielle.

    Remplace les éléments non sémantiques — chiffres, pourcentages, points de base,
    références réglementaires — par des placeholders afin que deux paragraphes
    exprimant la même idée avec des valeurs différentes soient reconnus comme
    identiques, sans casser la grammaire du squelette (« le BSIF a annoncé »).
    Utilisée pour peupler ``semantic_text_t1`` / ``semantic_text_t2`` dans les changements.
    """
    value = (text or "").strip()
    if not value:
        return ""
    for pattern, replacement in _SEMANTIC_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    value = _REGULATORY_REF_RE.sub("<regulateur>", value)
    value = _NUMERIC_TOKEN_RE.sub("<nombre>", value)
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


def _is_not_applicable_marker(text: str) -> bool:
    """Indique qu'un bloc ne contient que le marqueur autonome « s.o. »."""
    return bool(_NOT_APPLICABLE_MARKER_RE.fullmatch(str(text or "")))


def _is_table_unit_label(text: str) -> bool:
    """Indique qu'un bloc est le libellé d'unité d'un tableau ou graphique."""
    return bool(_TABLE_UNIT_LABEL_RE.fullmatch(str(text or "")))


def _looks_like_table_caption_title(text: str) -> bool:
    """Indique un titre/légende de tableau, pas du narratif comparable.

    Docling étiquette souvent ces légendes ``section_header`` (ex. ``T33 État
    des flux … (en millions de dollars)``). Elles précèdent une grille et doivent
    être exclues du markdown narratif, comme le font déjà les chaînes de titres
    immédiatement suivies d'une table Markdown.
    """
    value = str(text or "").strip()
    if not value:
        return False
    if _NUMBERED_TABLE_CAPTION_RE.search(value):
        return True
    return bool(_TABLE_HEADING_RE.match(value) and _TABLE_CAPTION_UNIT_RE.search(value))


def _is_running_report_chrome(text: str) -> bool:
    """Indique un en-tête ou pied de page récurrent d'un rapport bancaire."""
    value = str(text or "").strip()
    return bool(
        is_quarterly_running_chrome(value)
        or _BANK_FIRST_REPORT_PAGE_CHROME_RE.fullmatch(value)
        or _REPORT_FIRST_BANK_PAGE_CHROME_RE.fullmatch(value)
        or _BNC_ANNUAL_REPORT_CHROME_RE.fullmatch(value)
        or _BNC_MANAGEMENT_RUNNING_HEADER_RE.fullmatch(value)
    )


def _is_chart_axis_label_row(text: str) -> bool:
    """Indique une suite de graduations numériques issue d'un graphique."""
    return bool(_CHART_AXIS_LABEL_ROW_RE.fullmatch(str(text or "")))


def _sanitize_explanation(text: str) -> str:
    """Nettoie et tronque une explication GPT à 1 200 caractères maximum.

    Ne réutilise pas ``_sanitize_semantic_text`` : cette dernière retire
    volontairement chiffres et sigles réglementaires (BSIF, IFRS, Bâle...)
    pour l'appariement sémantique interne, ce qui casse la grammaire et
    prive l'analyste d'informations utiles dans un texte qui lui est destiné.
    """
    value = " ".join(str(text or "").split())
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
    if lower_value.startswith(("s.o.", "n.s.", "sans objet", "note", "source", "consulter")):
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
    max_height: float = 0.14,
) -> dict[int, list[list[float]]]:
    """Infère les zones de notes de bas de tableau à partir des bounding boxes des tableaux.

    Pour chaque tableau, génère une zone candidate juste en-dessous (hauteur ≤ ``max_height``
    en coordonnées normalisées). Ces zones sont ensuite utilisées par
    ``_classify_block_type`` pour identifier les blocs de légende ou de notes
    annexés aux tableaux et les exclure du contenu narratif.

    Args:
        table_bboxes_by_page: Bounding boxes des tableaux détectés par Docling, par page.
        max_height: Hauteur maximale (normalisée) de la zone note inférée. La
            marge par défaut de 14 % couvre les notes BNC séparées du tableau
            par un léger espace visuel, sans supprimer un paragraphe ordinaire
            (qui doit aussi présenter une forme explicite de note).

    Returns:
        Dictionnaire page → liste de bounding boxes de zones notes potentielles.
    """
    footnote_bboxes_by_page: dict[int, list[list[float]]] = {}
    for page, boxes in table_bboxes_by_page.items():
        ordered = sorted(
            (list(box) for box in boxes if len(box) == 4), key=lambda bbox: (float(bbox[1]), float(bbox[0]))
        )
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
