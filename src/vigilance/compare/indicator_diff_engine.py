"""
Moteur de diff au niveau des indicateurs pour les tableaux déjà appariés.

Ce module détecte les changements d'indicateurs (lignes de la première colonne)
entre deux tableaux appariés T1 et T2. Il est intentionnellement séparé du
moteur de pairing (``table_pairing_engine.py``) pour que les paramètres de
l'algorithme hongrois, de l'alignement par ordre et des embeddings n'interfèrent
pas avec le matching de tableaux.

Responsabilités
---------------
Ce module prend en entrée une paire de tableaux déjà appariés et retourne :
- La liste des indicateurs **ajoutés** (présents en T2, absents en T1).
- La liste des indicateurs **supprimés** (présents en T1, absents en T2).
- Un flag ``had_fusion_split`` indiquant si une fusion/scission d'indicateurs
  a été détectée.
- Un dictionnaire de compteurs d'exclusions (indicateurs structurels, doublons,
  etc.).

Pipeline de détection (fonction ``_indicator_diff``)
-----------------------------------------------------
1. **Exclusion des tables de référence de pages** : si les deux tableaux sont
   des tables d'index de pages, retourne immédiatement un diff vide.
2. **Normalisation** : construction des ensembles canoniques d'indicateurs pour
   T1 et T2, en excluant les en-têtes structurels (lignes sans valeurs numériques),
   les en-têtes de rollforward et les lignes à valeurs dupliquées.
3. **Alignement par ordre** (optionnel) : utilise la plus longue sous-séquence
   commune (LCS) pour identifier les indicateurs stables malgré de légères
   différences textuelles.
4. **Résolution near-stable** : les paires d'indicateurs avec une similarité
   très élevée (≥ 0.92–0.95) sont considérées comme stables (renommages mineurs).
5. **Garde court** (short indicator guard) : supprime les indicateurs courts
   (≤ 3 tokens) qui sont un sous-ensemble d'un indicateur stable long.
6. **Détection fusion/scission** : détecte si un indicateur a été scindé en deux
   ou si deux indicateurs ont été fusionnés en un seul.
7. **Filtre d'alignement voisin** : supprime les faux positifs causés par des
   artefacts de découpage de lignes lors de l'extraction PDF.
8. **Appariement des renames** : utilise l'algorithme hongrois (scipy) ou un
   appariement glouton (fallback) pour associer les indicateurs supprimés aux
   indicateurs ajoutés qui leur ressemblent (score ≥ seuil configurable).

Fonctions principales
---------------------
- ``_indicator_diff`` : point d'entrée principal du diff.
- ``_hungarian_pair_added_removed`` : appariement optimal des renames via
  l'algorithme hongrois avec scoring lexical + embedding optionnel.
- ``_detect_fusion_split`` : détection des fusions et scissions d'indicateurs.
- ``_build_indicator_diff_debug`` : construction d'une trace d'audit par
  indicateur (statut : stable / renommé / ajouté / supprimé).
- ``_fuzzy_pair_added_removed`` : appariement glouton de fallback pour les
  renames quand scipy n'est pas disponible.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None  # type: ignore[assignment]

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]

from vigilance.config import get_indicator_diff_config
from vigilance.models.table_models import TableArtifact, get_comparison_indicators, get_vision_raw_indicators
from vigilance.utils.indicator_cleaner import normalize_indicator_for_comparison
from vigilance.utils.indicator_normalizer import (
    get_canonical_text,
    get_token_sorted_text,
    strip_footnote_markers_from_indicator,
)
from vigilance.utils.matching_normalizer import _classify_excluded_line

logger = logging.getLogger(__name__)

_INDICATOR_DEFAULTS = {
    "indicator_rename_min_score": 0.86,
    "indicator_gate_min_len_ratio": 0.55,
    "indicator_gate_min_token_overlap": 1,
}


def _canonical_indicator_key(text: str) -> str:
    """Produire une clé canonique pour la comparaison d'indicateurs.

    Applique la normalisation standard (minuscules, suppression des espaces
    superflus, etc.) pour obtenir une représentation stable utilisée dans
    tout le moteur de diff.

    Args:
        text: Texte brut de l'indicateur (libellé de la première colonne).

    Returns:
        Clé canonique normalisée pour la comparaison.
    """
    return normalize_indicator_for_comparison(text)


def _structural_header_keys_from_rows(table: TableArtifact) -> set[str]:
    """Identifier les en-têtes structurels sans valeurs numériques dans un tableau.

    Une ligne est considérée comme un en-tête structurel si sa première cellule
    contient du texte mais que toutes les autres cellules sont vides. Ces lignes
    servent de séparateurs visuels dans le PDF et ne correspondent pas à des
    indicateurs financiers réels.

    Retourne l'ensemble des clés canoniques de ces lignes, qui seront exclues
    du diff pour éviter les faux positifs.
    """
    result: set[str] = set()
    for row in getattr(table, "rows", None) or []:
        if not isinstance(row, (list, tuple)) or not row:
            continue
        label = str(row[0]).strip()
        if not label or _classify_excluded_line(label):
            continue
        other_cells = [str(cell).strip() for cell in row[1:]]
        if any(cell for cell in other_cells):
            continue
        cleaned = strip_footnote_markers_from_indicator(label)
        key = _canonical_indicator_key(cleaned)
        if key:
            result.add(key)
    return result


_DONT_STRUCTURAL_RE = re.compile(r"\bdont\b\s*:?\s*$", re.IGNORECASE)
_ROLLFORWARD_HEADER_CHILD_PREFIXES = (
    "solde debut",
    "nouvelle emission d instrument admissible a titre de fonds propre",
    "rachat de fonds propre",
    "autre y compri",
    "solde a la fin",
)


def _normalize_value_cells_for_structure(row: list[Any] | tuple[Any, ...]) -> list[str]:
    """Extraire et normaliser les cellules de valeur d'une ligne (hors première colonne).

    Utilisé pour comparer les valeurs numériques entre lignes afin de détecter
    les doublons structurels (ex. lignes ``dont :`` avec valeurs identiques
    à la ligne suivante).

    Args:
        row: Ligne du tableau (liste ou tuple de cellules).

    Returns:
        Liste des valeurs non vides, espaces normalisés, pour les cellules
        à partir de l'index 1.
    """
    values: list[str] = []
    for cell in list(row)[1:]:
        text = re.sub(r"\s+", " ", str(cell or "").strip())
        if text:
            values.append(text)
    return values


def _structural_rollforward_header_keys_from_rows(table: TableArtifact) -> set[str]:
    """Identifier les en-têtes de tableaux de rollforward (variation de fonds propres).

    Les tableaux de rollforward présentent une structure caractéristique :
    un en-tête suivi de lignes commençant par ``"solde début"``, ``"nouvelle
    émission"``, ``"rachat"``, etc. Ces en-têtes ne sont pas des indicateurs
    financiers comparables et doivent être exclus du diff.

    Détecte cette structure en cherchant une ligne dont les 4 lignes suivantes
    correspondent à au moins 3 des préfixes caractéristiques d'un rollforward.
    """
    rows = [
        row
        for row in (getattr(table, "rows", None) or [])
        if isinstance(row, (list, tuple)) and row
    ]
    result: set[str] = set()
    for idx in range(len(rows) - 2):
        current = list(rows[idx])
        label = str(current[0] if current else "").strip()
        if not label or _classify_excluded_line(label):
            continue
        current_key = _canonical_indicator_key(
            strip_footnote_markers_from_indicator(label)
        )
        if not current_key or current_key.startswith("solde "):
            continue

        next_keys: list[str] = []
        for lookahead in rows[idx + 1 : idx + 6]:
            next_label = str(list(lookahead)[0] if lookahead else "").strip()
            if not next_label or _classify_excluded_line(next_label):
                continue
            next_key = _canonical_indicator_key(
                strip_footnote_markers_from_indicator(next_label)
            )
            if next_key:
                next_keys.append(next_key)
        if not next_keys or not next_keys[0].startswith("solde debut"):
            continue
        child_signal_count = sum(
            1
            for next_key in next_keys[:4]
            if any(next_key.startswith(prefix) for prefix in _ROLLFORWARD_HEADER_CHILD_PREFIXES)
        )
        if child_signal_count >= 3:
            result.add(current_key)
    return result


def _structural_duplicate_value_keys_from_rows(table: TableArtifact) -> set[str]:
    """Identifier les lignes ``dont :`` avec valeurs identiques à la ligne suivante.

    Dans certains tableaux bancaires, une ligne ``"dont : [sous-catégorie]"``
    répète exactement les mêmes valeurs numériques que la ligne précédente.
    Ces lignes sont des artefacts de mise en forme et ne représentent pas de
    nouveaux indicateurs. Elles sont exclues du diff pour éviter les faux positifs.

    Détecte ce pattern en cherchant les lignes dont le libellé correspond à
    ``_DONT_STRUCTURAL_RE`` et dont les valeurs numériques sont identiques à
    celles de la ligne suivante.
    """
    rows = [
        row
        for row in (getattr(table, "rows", None) or [])
        if isinstance(row, (list, tuple)) and row
    ]
    result: set[str] = set()
    for idx in range(len(rows) - 1):
        current = list(rows[idx])
        nxt = list(rows[idx + 1])
        label = str(current[0] if current else "").strip()
        next_label = str(nxt[0] if nxt else "").strip()
        if not label or not next_label:
            continue
        if _classify_excluded_line(label) or _classify_excluded_line(next_label):
            continue
        if not _DONT_STRUCTURAL_RE.search(label):
            continue
        current_values = _normalize_value_cells_for_structure(current)
        next_values = _normalize_value_cells_for_structure(nxt)
        if len(current_values) < 2 or current_values != next_values:
            continue
        key = _canonical_indicator_key(strip_footnote_markers_from_indicator(label))
        if key:
            result.add(key)
    return result


_PAGE_REF_HEADER_RE = re.compile(r"\bpages?\b", re.IGNORECASE)
_LEADING_ORDINAL_RE = re.compile(r"^\s*\d{1,3}\s+\S")
_PAGE_REF_ALLOWED_RE = re.compile(r"^(?:notes?\s+)?[\d,\s\-àaet]+$", re.IGNORECASE)
_PAGE_REF_TEXT_RE = re.compile(
    r"\b(?:page|pages|voir page|voir pages|renvoi|référence|reference|notes?)\b",
    re.IGNORECASE,
)


def _looks_like_page_reference_cell(text: str) -> bool:
    """Déterminer si une cellule ressemble à une référence de page (table des matières).

    Les cellules de type ``"28"``, ``"voir page 45"``, ``"notes 1 et 2"`` sont
    typiques des tables d'index de pages. Seuls les tokens ``note``, ``notes``,
    ``et``, ``a``, ``à`` sont autorisés en plus des chiffres et tirets.

    Args:
        text: Contenu brut de la cellule.

    Returns:
        True si la cellule ressemble à une référence de page, False sinon.
    """
    value = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not value:
        return False
    value = re.sub(r"\(\d+\)\s*$", "", value).strip()
    if not value or len(value) > 80:
        return False
    alpha_tokens = re.findall(r"[a-zà-ÿ]+", value, flags=re.IGNORECASE)
    if any(token not in {"note", "notes", "et", "a", "à"} for token in alpha_tokens):
        return False
    return bool(_PAGE_REF_ALLOWED_RE.fullmatch(value))


def _is_page_reference_table(table: TableArtifact) -> bool:
    """Détecter si un tableau est une table d'index de pages (table des matières).

    Ces tables listent des indicateurs avec des références de pages (ex. ``"28"``,
    ``"voir page 45"``) plutôt que des valeurs financières. Elles ne sont pas
    comparables entre trimestres et sont exclues du diff.

    Critères de détection :
    - En-tête ou titre contenant des mots-clés de référence de pages.
    - Au moins 10 lignes dans le tableau.
    - Au moins 50 % des cellules de valeur contiennent des références de pages.
    """
    title = str(getattr(table, "title", "") or "").strip()
    headers = [str(h or "").strip() for h in (getattr(table, "headers", None) or [])]
    raw_indicators = [
        str(value or "").strip()
        for value in (getattr(table, "first_column_indicators_raw", None) or [])
        if str(value or "").strip()
    ]
    page_ref_header = any(
        _PAGE_REF_HEADER_RE.search(header) or _PAGE_REF_TEXT_RE.search(header)
        for header in headers
    )
    page_ref_title = bool(_PAGE_REF_TEXT_RE.search(title))
    page_ref_labels = sum(1 for value in raw_indicators[:10] if _PAGE_REF_TEXT_RE.search(value))
    if not (page_ref_header or page_ref_title or page_ref_labels >= 2):
        return False
    rows = [
        list(row)
        for row in (getattr(table, "rows", None) or [])
        if isinstance(row, (list, tuple)) and row
    ]
    if len(rows) < 10:
        return False
    ordinal_rows = 0
    populated_value_cells = 0
    page_ref_cells = 0
    for row in rows:
        label = str(row[0] if row else "").strip()
        second_cell = str(row[1] if len(row) > 1 else "").strip()
        if (label and label.replace(" ", "").isdigit()) or (
            label and _LEADING_ORDINAL_RE.match(label)
        ):
            ordinal_rows += 1
        label_col_idx = (
            1
            if len(row) >= 3
            and (
                (label and label.replace(" ", "").isdigit())
                or (not label and second_cell)
            )
            else 0
        )
        value_cells = row[label_col_idx + 1 :] if label_col_idx + 1 < len(row) else row[1:]
        for cell in value_cells:
            cell_text = str(cell or "").strip()
            if not cell_text:
                continue
            populated_value_cells += 1
            if _looks_like_page_reference_cell(cell_text) or _PAGE_REF_TEXT_RE.search(cell_text):
                page_ref_cells += 1
    if populated_value_cells == 0:
        return False
    if ordinal_rows < 2 and page_ref_labels < 2 and not page_ref_title and not page_ref_header:
        return False
    return (page_ref_cells / populated_value_cells) >= 0.5


def _neighbor_alignment_anchor(
    key: str,
    *,
    source_order: list[str],
    target_order: list[str],
    candidate_keys: set[str],
) -> bool:
    """Vérifier si une clé candidate est un ancrage d'alignement voisin valide.

    Une clé est un ancrage valide si ses voisins immédiats (hors candidats)
    dans source_order conservent leur ordre relatif dans target_order, avec
    au plus 3 positions d'écart. Utilisé pour filtrer les faux positifs dus
    aux artefacts de découpage de lignes lors de l'extraction PDF.

    Args:
        key: Clé candidate à évaluer.
        source_order: Ordre des indicateurs dans le tableau source.
        target_order: Ordre des indicateurs dans le tableau cible.
        candidate_keys: Ensemble des clés candidates (added ou removed).

    Returns:
        True si la clé est un ancrage d'alignement voisin valide, False sinon.
    """
    try:
        idx = source_order.index(key)
    except ValueError:
        return False
    target_pos = {value: pos for pos, value in enumerate(target_order)}
    prev_key = next(
        (
            source_order[j]
            for j in range(idx - 1, -1, -1)
            if source_order[j] not in candidate_keys
        ),
        None,
    )
    next_key = next(
        (
            source_order[j]
            for j in range(idx + 1, len(source_order))
            if source_order[j] not in candidate_keys
        ),
        None,
    )
    if not prev_key or not next_key:
        return False
    if prev_key not in target_pos or next_key not in target_pos:
        return False
    return target_pos[prev_key] < target_pos[next_key] and (
        target_pos[next_key] - target_pos[prev_key]
    ) <= 3


def _lcs_pair_indices(
    left_seq: list[str],
    right_seq: list[str],
    *,
    band_window: int | None = None,
) -> list[tuple[int, int]]:
    """Calculer la plus longue sous-séquence commune (LCS) entre deux séquences.

    Retourne la liste des paires d'indices ``(i, j)`` tels que
    ``left_seq[i] == right_seq[j]`` et que ces paires forment une sous-séquence
    croissante optimale.

    Paramètres
    ----------
    left_seq:
        Séquence de gauche (indicateurs T1 dans l'ordre d'apparition).
    right_seq:
        Séquence de droite (indicateurs T2 dans l'ordre d'apparition).
    band_window:
        Si fourni, utilise un algorithme LCS en bande (complexité réduite) pour
        les grandes matrices (n*m > 10 000). Légèrement moins précis mais
        beaucoup plus rapide.
    """
    if not left_seq or not right_seq:
        return []
    rows, cols = len(left_seq), len(right_seq)
    if band_window is not None:
        band = band_window
        dp: dict[tuple[int, int], int] = {}
        for i in range(rows + 1):
            for j in range(max(0, i - band), min(cols + 1, i + band + 2)):
                if i == 0 or j == 0:
                    dp[i, j] = 0
                elif left_seq[i - 1] == right_seq[j - 1]:
                    dp[i, j] = dp.get((i - 1, j - 1), 0) + 1
                else:
                    dp[i, j] = max(dp.get((i - 1, j), 0), dp.get((i, j - 1), 0))
        pairs: list[tuple[int, int]] = []
        i, j = rows, cols
        while i > 0 and j > 0 and (i, j) in dp:
            if left_seq[i - 1] == right_seq[j - 1]:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            else:
                v_prev_i = dp.get((i - 1, j), -1)
                v_prev_j = dp.get((i, j - 1), -1)
                if v_prev_i >= v_prev_j and (i - 1, j) in dp:
                    i -= 1
                elif (i, j - 1) in dp:
                    j -= 1
                else:
                    break
        pairs.reverse()
        return pairs
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if left_seq[i - 1] == right_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    pairs: list[tuple[int, int]] = []
    i, j = rows, cols
    while i > 0 and j > 0:
        if left_seq[i - 1] == right_seq[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _order_aware_stable_pairs(
    left_order: list[str],
    right_order: list[str],
    removed_keys: set[str],
    added_keys: set[str],
    *,
    th: dict[str, Any],
) -> set[tuple[str, str]]:
    """Identifier les paires d'indicateurs stables en tenant compte de l'ordre.

    Utilise la LCS pour trouver les indicateurs communs dans le même ordre
    relatif dans T1 et T2. Les indicateurs non alignés par la LCS mais avec
    une haute similarité textuelle (≥ ``indicator_order_aware_min_ratio``) et
    dans la même position relative sont considérés comme stables (renommages
    mineurs qui ne doivent pas apparaître dans le diff).

    Retourne un ensemble de paires ``(removed_key, added_key)`` à traiter comme
    stables (à exclure du diff).
    """
    if not removed_keys or not added_keys:
        return set()
    min_ratio = float(th.get("indicator_order_aware_min_ratio", 0.85))
    band = int(th.get("indicator_order_aware_band_window", 50))
    n, m = len(left_order), len(right_order)
    use_band = n * m > 10000
    pair_indices = _lcs_pair_indices(left_order, right_order, band_window=band if use_band else None)
    lcs_left = {i for i, _ in pair_indices}
    lcs_right = {j for _, j in pair_indices}
    left_unmatched_keys = [left_order[i] for i in range(len(left_order)) if i not in lcs_left]
    right_unmatched_keys = [right_order[j] for j in range(len(right_order)) if j not in lcs_right]
    left_candidates = [k for k in left_unmatched_keys if k in removed_keys]
    right_candidates = [k for k in right_unmatched_keys if k in added_keys]
    stable: set[tuple[str, str]] = set()
    if rapidfuzz_fuzz is None:
        return stable
    for idx in range(min(len(left_candidates), len(right_candidates))):
        lk, rk = left_candidates[idx], right_candidates[idx]
        score = rapidfuzz_fuzz.ratio(lk, rk) / 100.0
        if score >= min_ratio:
            stable.add((lk, rk))
    return stable


def _ordered_indicator_keys(
    values: list[str],
    *,
    excluded_keys: set[str] | None = None,
) -> list[str]:
    """Extraire les clés canoniques d'indicateurs dans l'ordre d'apparition.

    Exclut les lignes classées comme structurelles (``_classify_excluded_line``),
    les doublons et les clés explicitement exclues. Conserve l'ordre de première
    occurrence pour chaque indicateur.

    Args:
        values: Liste des libellés bruts de la première colonne.
        excluded_keys: Ensemble optionnel de clés à exclure (ex. structurelles).

    Returns:
        Liste des clés canoniques uniques, dans l'ordre d'apparition.
    """
    result: list[str] = []
    seen: set[str] = set()
    excluded = excluded_keys or set()
    for value in values:
        kind = _classify_excluded_line(value)
        if kind:
            continue
        cleaned = strip_footnote_markers_from_indicator(value)
        key = _canonical_indicator_key(cleaned)
        if not key or key in seen or key in excluded:
            continue
        seen.add(key)
        result.append(key)
    return result


def _is_likely_extraction_split(added_key: str, prev_key: str, next_key: str) -> bool:
    """Déterminer si un indicateur ajouté ressemble à un artefact de découpage.

    Lors de l'extraction PDF, une ligne peut être scindée en plusieurs. Un
    indicateur ajouté est suspect s'il contient peu de tokens absents de ses
    voisins (prev_key, next_key). Moins de 2 tokens « nouveaux » suggère un
    découpage plutôt qu'un vrai nouvel indicateur.

    Args:
        added_key: Clé canonique de l'indicateur ajouté.
        prev_key: Clé du voisin précédent dans l'ordre source.
        next_key: Clé du voisin suivant dans l'ordre source.

    Returns:
        True si l'indicateur ressemble à un artefact de découpage, False sinon.
    """
    atokens = set(added_key.split())
    ptokens = set(prev_key.split())
    ntokens = set(next_key.split())
    tokens_in_neither = atokens - ptokens - ntokens
    return len(tokens_in_neither) < 2


def _filter_neighbor_aligned_candidates(
    candidate_keys: set[str],
    *,
    source_order: list[str],
    target_order: list[str],
) -> set[str]:
    """Filtrer les candidats qui sont des artefacts d'alignement voisin.

    Ne conserve que les candidats qui sont des singletons (pas dans un bloc
    contigu de candidats), qui passent le test d'ancrage voisin
    (``_neighbor_alignment_anchor``) et qui ressemblent à un découpage
    d'extraction (``_is_likely_extraction_split``). Ces candidats sont
    considérés comme des faux positifs et seront exclus du diff.

    Args:
        candidate_keys: Ensemble des clés candidates (added ou removed).
        source_order: Ordre des indicateurs dans le tableau source.
        target_order: Ordre des indicateurs dans le tableau cible.

    Returns:
        Sous-ensemble des candidats identifiés comme artefacts à exclure.
    """
    if not candidate_keys:
        return set()
    filtered: set[str] = set()
    for idx, key in enumerate(source_order):
        if key not in candidate_keys:
            continue
        block_start = idx
        while block_start > 0 and source_order[block_start - 1] in candidate_keys:
            block_start -= 1
        block_end = idx
        while block_end + 1 < len(source_order) and source_order[block_end + 1] in candidate_keys:
            block_end += 1
        if block_end > block_start:
            continue
        prev_key = next(
            (
                source_order[j]
                for j in range(idx - 1, -1, -1)
                if source_order[j] not in candidate_keys
            ),
            None,
        )
        next_key = next(
            (
                source_order[j]
                for j in range(idx + 1, len(source_order))
                if source_order[j] not in candidate_keys
            ),
            None,
        )
        if (
            prev_key
            and next_key
            and _neighbor_alignment_anchor(
                key,
                source_order=source_order,
                target_order=target_order,
                candidate_keys=candidate_keys,
            )
            and _is_likely_extraction_split(key, prev_key, next_key)
        ):
            filtered.add(key)
    return filtered


_INDICATOR_STOPWORDS = frozenset({"de", "du", "des", "la", "le", "les", "et", "ou", "and", "the", "of", "to", "en", "au", "aux", "a", "an"})
_INDICATOR_UNIT_TOKENS = frozenset({"%", "million", "millions", "milliard", "milliards", "dollars", "cad", "usd"})
_INDICATOR_ACRONYM_RE = re.compile(
    r"\b(cet[-]?1|at[-]?1|tlac|rwa|ifrs[-]?9|tier[-]?\s*1|tier[-]?\s*2|bale[-]?\s*iii|pillar[-]?\s*3)\b",
    re.IGNORECASE,
)


def _indicator_strong_tokens(text: str) -> set[str]:
    """Extraire les tokens « forts » d'un indicateur pour le matching lexical.

    Exclut les stopwords (de, du, des, la, le, les, et, ou...), les tokens
    d'unité (million, milliard, %, cad...) et la plupart des chiffres.
    Conserve les chiffres courts (1, 2, 3, 9) et les acronymes réglementaires.

    Args:
        text: Texte de l'indicateur (déjà normalisé ou brut).

    Returns:
        Ensemble des tokens significatifs pour la comparaison.
    """
    if not text:
        return set()
    normalized = re.sub(r"[-/]", " ", (text or "").lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    tokens: set[str] = set()
    for token in normalized.split():
        if not token:
            continue
        if token in _INDICATOR_STOPWORDS or token in _INDICATOR_UNIT_TOKENS:
            continue
        if token.isdigit():
            if len(token) > 1:
                continue
            if token in ("1", "2", "3", "9"):
                tokens.add(token)
            continue
        tokens.add(token)
    return tokens


def _indicator_acronyms(text: str) -> set[str]:
    """Extraire les acronymes réglementaires présents dans un indicateur.

    Détecte les patterns tels que CET-1, AT1, TLAC, RWA, IFRS-9, Tier 1,
    Bâle III, Pillar 3, etc. Utilisé pour le gate de chevauchement lexical
    lors de l'appariement hongrois (un acronyme commun suffit à qualifier).

    Args:
        text: Texte de l'indicateur.

    Returns:
        Ensemble des acronymes trouvés (normalisés en minuscules, sans espaces).
    """
    if not text:
        return set()
    return {
        match.group(1).lower().replace(" ", "").replace("-", "")
        for match in _INDICATOR_ACRONYM_RE.finditer(text or "")
    }


_PREFILTER_MATRIX_CAP = 25_000
_PREFILTER_TOP_K_PER_REMOVED = 50
_INDICATOR_EMB_MIN = 0.45
_INDICATOR_MIN_TOKENS = 6
_INDICATOR_MIN_ALPHA_RATIO = 0.40
_PARENT_CHILD_UNIT_QUALIFIER_TOKENS = frozenset(
    {
        "en", "de", "des", "du", "million", "millions", "milliard", "milliers",
        "dollars", "canadiens", "cad", "usd", "total", "sous", "net", "brut", "montant",
    }
)


def _is_parent_child_pair(removed_label: str, added_label: str, norm_fn: Any) -> bool:
    """Déterminer si une paire removed/added est une relation parent-enfant.

    Un indicateur est parent de l'autre si l'un est un sous-ensemble strict
    de tokens de l'autre, et que la différence dépasse les qualifiants
    d'unité (million, dollars, total, net, etc.). Ces paires sont rejetées
    car elles ne représentent pas un simple renommage.

    Args:
        removed_label: Libellé de l'indicateur supprimé (T1).
        added_label: Libellé de l'indicateur ajouté (T2).
        norm_fn: Fonction de normalisation (ex. ``_canonical_indicator_key``).

    Returns:
        True si la paire est une relation parent-enfant à rejeter, False sinon.
    """
    a = norm_fn(removed_label)
    b = norm_fn(added_label)
    if not a or not b:
        return False
    ta, tb = set(a.split()), set(b.split())
    if len(ta) <= 1 or len(tb) <= 1:
        return False
    if ta < tb:
        return (tb - ta) > _PARENT_CHILD_UNIT_QUALIFIER_TOKENS
    if tb < ta:
        return (ta - tb) > _PARENT_CHILD_UNIT_QUALIFIER_TOKENS
    return False


def _hungarian_pair_added_removed(
    removed_items: list[str],
    added_items: list[str],
    *,
    th: dict[str, Any] | None = None,
    embedding_service: Any = None,
) -> tuple[list[str], list[str], list[tuple[str, str]], dict[str, Any]]:
    """Apparier les indicateurs supprimés et ajoutés comme renames via l'algorithme hongrois.

    Construit une matrice de scores de similarité entre tous les indicateurs
    supprimés (T1) et tous les indicateurs ajoutés (T2), puis utilise
    l'algorithme d'assignation hongroise (``scipy.optimize.linear_sum_assignment``)
    pour trouver l'appariement optimal qui maximise la similarité globale.

    Les paires dont le score est inférieur au seuil ``indicator_rename_min_score``
    (défaut : 0.86) sont rejetées. Les paires parent-enfant (un indicateur est
    un sous-ensemble de l'autre) sont également rejetées.

    Filtres de pré-qualification (gate)
    ------------------------------------
    Avant le calcul de la matrice, chaque paire est filtrée par :
    - **Ratio de longueur** : les indicateurs dont les longueurs sont trop
      différentes (ratio < ``indicator_gate_min_len_ratio``) sont exclus.
    - **Chevauchement de tokens** : au moins un token fort (hors stopwords)
      doit être commun entre les deux indicateurs.

    Scoring
    -------
    Le score final est une combinaison de :
    - Similarité lexicale canonique (``rapidfuzz.ratio`` + ``token_set_ratio``).
    - Similarité lexicale sur texte trié par tokens (pour les indicateurs
      dont les mots sont dans un ordre différent).
    - Similarité sémantique via embeddings (optionnel, poids configurable).

    Paramètres
    ----------
    removed_items:
        Liste des indicateurs présents en T1 mais absents en T2 (après diff initial).
    added_items:
        Liste des indicateurs présents en T2 mais absents en T1 (après diff initial).
    th:
        Dictionnaire de seuils et paramètres de configuration.
    embedding_service:
        Service d'embedding optionnel pour le scoring sémantique.

    Retourne
    --------
    Tuple ``(added_restants, removed_restants, renamed_pairs, debug_info)`` où :
    - ``added_restants`` : indicateurs ajoutés non appariés (vrais ajouts).
    - ``removed_restants`` : indicateurs supprimés non appariés (vraies suppressions).
    - ``renamed_pairs`` : liste de tuples ``(removed, added)`` représentant les renames.
    - ``debug_info`` : dictionnaire de métriques de débogage.
    """
    th = th or {}
    min_score = float(th.get("indicator_rename_min_score", _INDICATOR_DEFAULTS["indicator_rename_min_score"]))
    min_len_ratio = float(th.get("indicator_gate_min_len_ratio", _INDICATOR_DEFAULTS["indicator_gate_min_len_ratio"]))
    min_token_overlap = int(th.get("indicator_gate_min_token_overlap", _INDICATOR_DEFAULTS["indicator_gate_min_token_overlap"]))
    weights_raw = th.get("indicator_similarity_weights")
    weights: dict[str, float] | None = weights_raw if isinstance(weights_raw, dict) else None
    if not removed_items or not added_items or rapidfuzz_fuzz is None:
        return list(added_items), list(removed_items), [], {"gated_out_pairs": 0, "accepted_renames": 0}

    large_matrix_cap = int(th.get("indicator_hungarian_large_matrix_cap", 500))
    large_matrix_min_score = float(th.get("indicator_hungarian_large_matrix_min_score", 0.88))
    matrix_size_for_cap = len(removed_items) * len(added_items)
    is_large_matrix = matrix_size_for_cap > large_matrix_cap
    effective_min_score = max(min_score, large_matrix_min_score) if is_large_matrix else min_score
    min_score_pct = int(effective_min_score * 100)

    def _norm_for_sort(s: str) -> str:
        return _canonical_indicator_key(strip_footnote_markers_from_indicator(s))

    removed = sorted(removed_items, key=_norm_for_sort)
    added = sorted(added_items, key=_norm_for_sort)

    gate_len_ratio = float(th.get("indicator_hungarian_large_matrix_min_len_ratio", min_len_ratio)) if is_large_matrix else min_len_ratio
    gate_token_overlap = int(th.get("indicator_hungarian_large_matrix_min_token_overlap", min_token_overlap)) if is_large_matrix else min_token_overlap

    def _length_ratio_ok(a: str, r: str) -> bool:
        la, lr = len(_norm_for_sort(a)), len(_norm_for_sort(r))
        if max(la, lr) <= 0:
            return True
        return (min(la, lr) / max(la, lr)) >= gate_len_ratio

    def _token_overlap_ok(a: str, r: str) -> bool:
        na, nr = _norm_for_sort(a), _norm_for_sort(r)
        ta = _indicator_strong_tokens(na)
        tr = _indicator_strong_tokens(nr)
        if len(ta & tr) >= gate_token_overlap:
            return True
        return len(_indicator_acronyms(na) & _indicator_acronyms(nr)) > 0

    def _similarity(a: str, r: str) -> float:
        ratio_score = rapidfuzz_fuzz.ratio(a, r)
        token_score = rapidfuzz_fuzz.token_set_ratio(a, r)
        if weights:
            return weights.get("ratio", 0.4) * ratio_score + weights.get("token_set", 0.6) * token_score
        return max(ratio_score, token_score)

    use_token_sorted = bool(th.get("use_indicator_token_sorted_matching", True))
    min_tokens = int(th.get("indicator_embed_min_tokens", _INDICATOR_MIN_TOKENS))
    emb_min = float(th.get("indicator_embed_min_sim", _INDICATOR_EMB_MIN))
    min_alpha_ratio = float(th.get("indicator_embed_min_alpha_ratio", _INDICATOR_MIN_ALPHA_RATIO))

    def _lex_similarity_both_forms(a: str, r: str) -> tuple[float, float, float]:
        lex_canon = _similarity(a, r)
        if not use_token_sorted:
            return lex_canon, 0.0, lex_canon
        ts_a = get_token_sorted_text(a)
        ts_r = get_token_sorted_text(r)
        lex_ts = 0.0
        if ts_a and ts_r:
            lex_ts = max(
                rapidfuzz_fuzz.ratio(ts_a, ts_r),
                rapidfuzz_fuzz.token_set_ratio(ts_a, ts_r),
            )
        return lex_canon, lex_ts, max(lex_canon, lex_ts)

    def _embed_gate_ok(a: str, r: str, emb_sim: float) -> bool:
        if emb_sim < emb_min:
            return False
        ts_a, ts_r = get_token_sorted_text(a), get_token_sorted_text(r)
        tokens_a = [t for t in ts_a.split() if t] if ts_a else []
        tokens_r = [t for t in ts_r.split() if t] if ts_r else []
        if len(tokens_a) < min_tokens or len(tokens_r) < min_tokens:
            return False
        for text in (a, r):
            alpha = sum(1 for c in text if c.isalpha())
            if text and (alpha / len(text)) < min_alpha_ratio:
                return False
        return True

    n_rem, n_add = len(removed), len(added)
    use_emb = bool(th.get("use_embeddings", False)) and embedding_service and getattr(embedding_service, "available", False)
    embed_weight = float(th.get("embedding_weight_indicator", 0.35)) if use_emb else 0.0

    def _norm_for_embed(s: str) -> str:
        c = _canonical_indicator_key(strip_footnote_markers_from_indicator(s))
        return c if c else (s or " ").strip()[:200]

    embed_matrix_canon = None
    embed_matrix_ts = None
    if use_emb and n_rem > 0 and n_add > 0:
        try:
            import numpy as np  # noqa: F401

            texts_rem_canon = [_norm_for_embed(removed[i]) for i in range(n_rem)]
            texts_add_canon = [_norm_for_embed(added[j]) for j in range(n_add)]
            embed_matrix_canon = embedding_service.get_pairwise_cosine(texts_rem_canon, texts_add_canon)
            if use_token_sorted:
                texts_rem_ts = [get_token_sorted_text(removed[i]) or " " for i in range(n_rem)]
                texts_add_ts = [get_token_sorted_text(added[j]) or " " for j in range(n_add)]
                embed_matrix_ts = embedding_service.get_pairwise_cosine(texts_rem_ts, texts_add_ts)
        except Exception as exc:
            logger.debug("Indicator embedding batch failed: %s", exc)
            embed_matrix_canon = None
            embed_matrix_ts = None

    matrix_size = n_rem * n_add
    prefilter_used = matrix_size > _PREFILTER_MATRIX_CAP
    candidate_set: set[tuple[int, int]] | None = None
    if prefilter_used:
        candidate_set = set()
        for i in range(n_rem):
            scored: list[tuple[float, int]] = []
            for j in range(n_add):
                if _length_ratio_ok(added[j], removed[i]) and _token_overlap_ok(added[j], removed[i]):
                    scored.append((rapidfuzz_fuzz.token_set_ratio(added[j], removed[i]), j))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, j in scored[:_PREFILTER_TOP_K_PER_REMOVED]:
                candidate_set.add((i, j))

    gated_out = 0
    accepted_scores: list[float] = []
    if linear_sum_assignment is not None:
        import numpy as np

        scores = np.full((n_rem, n_add), -1e9, dtype=np.float64)
        for i in range(n_rem):
            for j in range(n_add):
                if candidate_set is not None and (i, j) not in candidate_set:
                    gated_out += 1
                    continue
                if _length_ratio_ok(added[j], removed[i]) and _token_overlap_ok(added[j], removed[i]):
                    lex_canon, lex_ts, lex = _lex_similarity_both_forms(added[j], removed[i])
                    emb_sim_canon = float(embed_matrix_canon[i, j]) if embed_matrix_canon is not None else 0.0
                    emb_sim_ts = float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
                    emb_sim = max(emb_sim_canon, emb_sim_ts) if (embed_matrix_canon is not None or embed_matrix_ts is not None) else 0.0
                    embed_ok = embed_weight > 0 and _embed_gate_ok(added[j], removed[i], emb_sim)
                    w_eff = embed_weight if embed_ok else 0.0
                    calibrated_embed = (emb_sim * 100.0) if emb_sim >= emb_min else 0.0
                    scores[i, j] = (1.0 - w_eff) * lex + w_eff * calibrated_embed
                else:
                    gated_out += 1
        cost = -scores
        row_ind, col_ind = linear_sum_assignment(cost)
        renamed_pairs: list[tuple[str, str]] = []
        renamed_indices: list[tuple[int, int]] = []
        used_rem: set[int] = set()
        used_add: set[int] = set()
        for k in range(len(row_ind)):
            i, j = int(row_ind[k]), int(col_ind[k])
            if i >= n_rem or j >= n_add:
                continue
            sc = float(scores[i, j])
            if sc >= min_score_pct and not _is_parent_child_pair(removed[i], added[j], _norm_for_sort):
                renamed_pairs.append((removed[i], added[j]))
                renamed_indices.append((i, j))
                accepted_scores.append(sc)
                used_rem.add(i)
                used_add.add(j)
        added_restant = [added[j] for j in range(n_add) if j not in used_add]
        removed_restant = [removed[i] for i in range(n_rem) if i not in used_rem]

        asc = sorted(accepted_scores) if accepted_scores else []
        unmatched_candidates: list[dict[str, Any]] = []
        for i in range(n_rem):
            if i in used_rem:
                continue
            cand: list[tuple[str, float]] = []
            for j in range(n_add):
                if j in used_add:
                    continue
                sc = float(scores[i, j])
                if sc > -1e8:
                    cand.append((added[j], sc))
            cand.sort(key=lambda x: x[1], reverse=True)
            unmatched_candidates.append({"removed": removed[i], "top3": cand[:3]})
        rename_pair_debug: list[dict[str, Any]] = []
        for (r, a), (i, j) in zip(renamed_pairs, renamed_indices):
            lc, lts, _ = _lex_similarity_both_forms(a, r)
            ec = float(embed_matrix_canon[i, j]) if embed_matrix_canon is not None else 0.0
            ets = float(embed_matrix_ts[i, j]) if embed_matrix_ts is not None else 0.0
            reasons: list[str] = []
            if not _embed_gate_ok(a, r, max(ec, ets)):
                reasons.append("embed_gated")
            rename_pair_debug.append(
                {
                    "lex_canonical": round(lc, 2),
                    "lex_token_sorted": round(lts, 2),
                    "embed_canonical": round(ec, 3),
                    "embed_token_sorted": round(ets, 3),
                    "final_score": round(float(scores[i, j]), 2),
                    "reasons": reasons or ["ok"],
                }
            )
        return added_restant, removed_restant, renamed_pairs, {
            "gated_out_pairs": gated_out,
            "accepted_renames": len(renamed_pairs),
            "prefilter_used": prefilter_used,
            "rename_pair_debug": rename_pair_debug,
            "score_distribution": {
                "min": min(asc) if asc else None,
                "max": max(asc) if asc else None,
                "mean": sum(asc) / len(asc) if asc else None,
                "median": asc[len(asc) // 2] if asc else None,
            },
            "unmatched_removed_with_candidates": unmatched_candidates,
        }

    used_add_f: set[int] = set()
    used_rem_f: set[int] = set()
    renamed_pairs = []
    for i, removed_value in enumerate(removed):
        best_j = -1
        best_score = -1.0
        for j, added_value in enumerate(added):
            if j in used_add_f:
                continue
            if not _length_ratio_ok(added_value, removed_value) or not _token_overlap_ok(added_value, removed_value):
                gated_out += 1
                continue
            _, _, lex_final = _lex_similarity_both_forms(added_value, removed_value)
            if lex_final >= min_score_pct and lex_final > best_score:
                best_score = lex_final
                best_j = j
        if best_j >= 0 and not _is_parent_child_pair(removed_value, added[best_j], _norm_for_sort):
            renamed_pairs.append((removed_value, added[best_j]))
            _, _, lex_f = _lex_similarity_both_forms(added[best_j], removed_value)
            accepted_scores.append(lex_f)
            used_rem_f.add(i)
            used_add_f.add(best_j)
    added_restant = [added[j] for j in range(n_add) if j not in used_add_f]
    removed_restant = [removed[i] for i in range(n_rem) if i not in used_rem_f]
    asc = sorted(accepted_scores) if accepted_scores else []
    return added_restant, removed_restant, renamed_pairs, {
        "gated_out_pairs": gated_out,
        "accepted_renames": len(renamed_pairs),
        "prefilter_used": prefilter_used,
        "score_distribution": {
            "min": min(asc) if asc else None,
            "max": max(asc) if asc else None,
            "mean": sum(asc) / len(asc) if asc else None,
            "median": asc[len(asc) // 2] if asc else None,
        },
        "unmatched_removed_with_candidates": [{"removed": removed[i], "top3": []} for i in range(n_rem) if i not in used_rem_f],
    }


def _adaptive_fusion_threshold(concat_token_count: int) -> float:
    """Calculer le seuil de similarité adaptatif pour la détection fusion/scission.

    Plus la concaténation est longue, plus le seuil est bas (tolérance accrue).
    Pour les concaténations courtes (< 5 tokens), un seuil élevé (0.94) évite
    les faux positifs.

    Args:
        concat_token_count: Nombre de tokens dans la concaténation des deux
            indicateurs (k1 + k2 ou k2 + k1).

    Returns:
        Seuil de similarité (entre 0.88 et 0.94) pour valider une fusion/scission.
    """
    if concat_token_count >= 8:
        return 0.88
    if concat_token_count < 5:
        return 0.94
    return 0.92


def _fusion_split_score(single_norm: str, k1: str, k2: str) -> tuple[float, int]:
    """Calculer le score de similarité entre un indicateur et la concaténation de deux.

    Compare single_norm à ``k1 + k2`` et ``k2 + k1`` via token_set_ratio.
    Tente aussi des réparations quand le dernier token d'un fragment est un
    préfixe d'un mot du single (ex. découpage mal placé).

    Args:
        single_norm: Indicateur unique normalisé (T1 ou T2).
        k1: Premier fragment (clé canonique).
        k2: Second fragment (clé canonique).

    Returns:
        Tuple (meilleur score de similarité, nombre de tokens de la concaténation).
    """
    c_fwd = f"{k1} {k2}".strip()
    c_rev = f"{k2} {k1}".strip()
    ntok = max(len(c_fwd.split()), 1)
    if not c_fwd:
        return 0.0, ntok
    if rapidfuzz_fuzz is not None:
        best = max(
            rapidfuzz_fuzz.token_set_ratio(single_norm, c_fwd) / 100.0,
            rapidfuzz_fuzz.token_set_ratio(single_norm, c_rev) / 100.0,
        )
    else:
        best = max(
            SequenceMatcher(None, single_norm, c_fwd).ratio(),
            SequenceMatcher(None, single_norm, c_rev).ratio(),
        )
    toks_single = single_norm.split()
    for frag_key in (k1, k2):
        ft = frag_key.split()
        if not ft or len(ft[-1]) < 4:
            continue
        last = ft[-1]
        for word in toks_single:
            if len(word) >= len(last) and word.startswith(last) and word != last:
                repaired = " ".join(ft[:-1] + [word])
                other = k2 if frag_key == k1 else k1
                c1 = f"{repaired} {other}".strip()
                c2 = f"{other} {repaired}".strip()
                if rapidfuzz_fuzz is not None:
                    best = max(
                        best,
                        rapidfuzz_fuzz.token_set_ratio(single_norm, c1) / 100.0,
                        rapidfuzz_fuzz.token_set_ratio(single_norm, c2) / 100.0,
                    )
                else:
                    best = max(
                        best,
                        SequenceMatcher(None, single_norm, c1).ratio(),
                        SequenceMatcher(None, single_norm, c2).ratio(),
                    )
    return best, ntok


def _detect_fusion_split(added: list[str], removed: list[str]) -> tuple[list[str], list[str], bool]:
    """Détecter les fusions et scissions d'indicateurs entre T1 et T2.

    Deux cas sont détectés de manière itérative :

    - **Scission** (split) : un indicateur de T1 a été divisé en deux indicateurs
      de T2. Détecté si la concaténation des deux indicateurs T2 est très similaire
      à l'indicateur T1 (score ≥ seuil adaptatif).
    - **Fusion** (merge) : deux indicateurs de T1 ont été fusionnés en un seul
      indicateur de T2. Détecté si la concaténation des deux indicateurs T1 est
      très similaire à l'indicateur T2.

    Le seuil de similarité est adaptatif : plus la concaténation est longue
    (beaucoup de tokens), plus le seuil est bas (0.88 pour ≥ 8 tokens, 0.94
    pour < 5 tokens).

    Paramètres
    ----------
    added:
        Liste des indicateurs ajoutés (candidats à la fusion depuis T1).
    removed:
        Liste des indicateurs supprimés (candidats à la scission vers T2).

    Retourne
    --------
    Tuple ``(added_restants, removed_restants, had_fusion_split)`` où les
    listes restantes excluent les indicateurs impliqués dans une fusion/scission.
    """
    added = list(added)
    removed = list(removed)
    had_fusion_split = False

    def _merge_added_from_removed() -> None:
        nonlocal added, removed, had_fusion_split
        for added_value in added[:]:
            a_norm = _canonical_indicator_key(added_value)
            if not a_norm:
                continue
            for j, r1 in enumerate(removed):
                k1 = _canonical_indicator_key(r1)
                if not k1:
                    continue
                for k, r2 in enumerate(removed):
                    if j >= k:
                        continue
                    k2 = _canonical_indicator_key(r2)
                    if not k2:
                        continue
                    score, ntok = _fusion_split_score(a_norm, k1, k2)
                    thresh = _adaptive_fusion_threshold(ntok)
                    if score >= thresh or f"{k1} {k2}".strip() == a_norm or f"{k2} {k1}".strip() == a_norm:
                        added.remove(added_value)
                        removed.remove(r2)
                        removed.remove(r1)
                        had_fusion_split = True
                        return

    def _merge_removed_from_added() -> None:
        nonlocal added, removed, had_fusion_split
        for removed_value in removed[:]:
            r_norm = _canonical_indicator_key(removed_value)
            if not r_norm:
                continue
            for j, a1 in enumerate(added):
                k1 = _canonical_indicator_key(a1)
                if not k1:
                    continue
                for k, a2 in enumerate(added):
                    if j >= k:
                        continue
                    k2 = _canonical_indicator_key(a2)
                    if not k2:
                        continue
                    score, ntok = _fusion_split_score(r_norm, k1, k2)
                    thresh = _adaptive_fusion_threshold(ntok)
                    if score >= thresh or f"{k1} {k2}".strip() == r_norm or f"{k2} {k1}".strip() == r_norm:
                        removed.remove(removed_value)
                        added.remove(a2)
                        added.remove(a1)
                        had_fusion_split = True
                        return

    changed = True
    while changed:
        changed = False
        before_a, before_r = len(added), len(removed)
        _merge_added_from_removed()
        if len(added) != before_a or len(removed) != before_r:
            changed = True
            continue
        _merge_removed_from_added()
        if len(added) != before_a or len(removed) != before_r:
            changed = True
    return added, removed, had_fusion_split


def _apply_short_indicator_guard(
    added_keys: set[str],
    removed_keys: set[str],
    stable_keys: set[str],
    th: dict[str, Any],
    excluded_counts: dict[str, int],
) -> None:
    """Supprimer les indicateurs courts qui sont un sous-ensemble d'un indicateur stable.

    Certains artefacts d'extraction produisent des fragments courts (2-3 tokens)
    qui correspondent à une partie d'un indicateur stable plus long. Ces fragments
    ne représentent pas de vrais changements et sont supprimés des ensembles
    ``added_keys`` et ``removed_keys``.

    Exemple : si ``"fonds propres"`` est stable, l'indicateur ajouté ``"propres"``
    (sous-ensemble de tokens) est supprimé.

    Paramètres
    ----------
    added_keys:
        Ensemble des clés d'indicateurs ajoutés (modifié en place).
    removed_keys:
        Ensemble des clés d'indicateurs supprimés (modifié en place).
    stable_keys:
        Ensemble des clés d'indicateurs stables (présents dans T1 et T2).
    th:
        Dictionnaire de seuils (``indicator_short_guard_max_tokens``,
        ``indicator_short_guard_min_stable_tokens``,
        ``indicator_short_guard_enabled``).
    excluded_counts:
        Dictionnaire de compteurs mis à jour avec le nombre d'exclusions.
    """
    if not th.get("indicator_short_guard_enabled", True):
        return
    max_t = int(th.get("indicator_short_guard_max_tokens", 3))
    min_stable = int(th.get("indicator_short_guard_min_stable_tokens", 5))
    stable_list = [sk for sk in stable_keys if len(sk.split()) >= min_stable]

    def _maybe_drop(key: str, which: str) -> bool:
        parts = key.split()
        if len(parts) < 2 or len(parts) > max_t:
            return False
        tk = frozenset(parts)
        for stable in stable_list:
            if tk < set(stable.split()):
                if which == "added":
                    added_keys.discard(key)
                else:
                    removed_keys.discard(key)
                excluded_counts["short_indicator_guard"] = excluded_counts.get("short_indicator_guard", 0) + 1
                return True
        return False

    for key in list(added_keys):
        _maybe_drop(key, "added")
    for key in list(removed_keys):
        _maybe_drop(key, "removed")


def _indicator_diff(
    t1: TableArtifact,
    t2: TableArtifact,
    *,
    neighbor_aligned_filter_enabled: bool = True,
    return_debug: bool = False,
    th: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], bool, dict[str, int], dict[str, Any] | None]:
    """Calculer le diff d'indicateurs entre deux tableaux appariés T1 et T2.

    C'est le point d'entrée principal de ce module. Il orchestre toutes les
    étapes du pipeline de détection décrites dans la docstring du module.

    Paramètres
    ----------
    t1:
        Tableau du trimestre précédent.
    t2:
        Tableau du trimestre courant.
    neighbor_aligned_filter_enabled:
        Si ``True``, applique le filtre d'alignement voisin pour supprimer les
        faux positifs causés par des artefacts de découpage de lignes.
    return_debug:
        Si ``True``, inclut les maps d'indicateurs dans le résultat de débogage.
    th:
        Dictionnaire de seuils de configuration. Si ``None``, utilise les
        valeurs par défaut de ``_INDICATOR_DEFAULTS``.

    Retourne
    --------
    Tuple ``(added, removed, had_fusion_split, excluded_counts, debug_info)`` :
    - ``added`` : liste triée des indicateurs ajoutés en T2.
    - ``removed`` : liste triée des indicateurs supprimés depuis T1.
    - ``had_fusion_split`` : ``True`` si une fusion ou scission a été détectée.
    - ``excluded_counts`` : dictionnaire des compteurs d'exclusions par catégorie.
    - ``debug_info`` : dictionnaire de débogage (``None`` si ``return_debug=False``).
    """
    if _is_page_reference_table(t1) and _is_page_reference_table(t2):
        return [], [], False, {"page_reference_table": 1}, None

    left = get_comparison_indicators(t1)
    right = get_comparison_indicators(t2)
    left_all_keys = set(_ordered_indicator_keys(left))
    right_all_keys = set(_ordered_indicator_keys(right))
    left_structural_keys = (
        _structural_header_keys_from_rows(t1)
        | _structural_rollforward_header_keys_from_rows(t1)
        | _structural_duplicate_value_keys_from_rows(t1)
    ) - right_all_keys
    right_structural_keys = (
        _structural_header_keys_from_rows(t2)
        | _structural_rollforward_header_keys_from_rows(t2)
        | _structural_duplicate_value_keys_from_rows(t2)
    ) - left_all_keys

    def _norm(values: list[str], *, structural_keys: set[str]) -> tuple[dict[str, str], dict[str, int]]:
        mapped: dict[str, str] = {}
        excluded: dict[str, int] = {}
        for value in values:
            kind = _classify_excluded_line(value)
            if kind:
                excluded[kind] = excluded.get(kind, 0) + 1
                continue
            value_clean = strip_footnote_markers_from_indicator(value)
            key = _canonical_indicator_key(value_clean)
            if key in structural_keys:
                excluded["structural"] = excluded.get("structural", 0) + 1
                continue
            if key and key not in mapped:
                mapped[key] = value_clean
        return mapped, excluded

    left_map, left_excluded = _norm(left, structural_keys=left_structural_keys)
    right_map, right_excluded = _norm(right, structural_keys=right_structural_keys)
    excluded_counts: dict[str, int] = {}
    for key in set(left_excluded) | set(right_excluded):
        excluded_counts[key] = left_excluded.get(key, 0) + right_excluded.get(key, 0)

    left_order = _ordered_indicator_keys(left, excluded_keys=left_structural_keys)
    right_order = _ordered_indicator_keys(right, excluded_keys=right_structural_keys)
    added_keys = set(right_map.keys() - left_map.keys())
    removed_keys = set(left_map.keys() - right_map.keys())
    th = th or {}
    if th.get("indicator_order_aware_alignment_enabled", False):
        order_stable = _order_aware_stable_pairs(left_order, right_order, removed_keys, added_keys, th=th)
        added_keys -= {r for _l, r in order_stable}
        removed_keys -= {l for l, _r in order_stable}
        if order_stable:
            excluded_counts["order_aware_stable"] = len(order_stable)

    total_keys = len(left_map) + len(right_map)
    large_table_min = int(th.get("indicator_near_stable_large_table_min_indicators", 40))
    if total_keys >= large_table_min:
        near_stable_threshold = float(th.get("indicator_near_stable_large_table_threshold", 0.92))
    else:
        near_stable_threshold = float(th.get("indicator_near_stable_threshold", 0.95))
    use_token_set = bool(th.get("indicator_near_stable_use_token_set", False))
    resolved_removed: set[str] = set()
    resolved_added: set[str] = set()
    singleton_near_stable = len(added_keys) == 1 and len(removed_keys) == 1
    for removed_key in list(removed_keys):
        best_match = None
        best_score = 0.0
        for added_key in list(added_keys - resolved_added):
            if rapidfuzz_fuzz is not None:
                ratio_score = rapidfuzz_fuzz.ratio(removed_key, added_key) / 100.0
                score = max(ratio_score, rapidfuzz_fuzz.token_set_ratio(removed_key, added_key) / 100.0) if use_token_set else ratio_score
            else:
                left_tokens = set(removed_key.split())
                right_tokens = set(added_key.split())
                score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if (left_tokens | right_tokens) else 0.0
            if score > best_score:
                best_score = score
                best_match = added_key
        if best_score >= near_stable_threshold and best_match:
            if singleton_near_stable and not (
                _neighbor_alignment_anchor(
                    removed_key,
                    source_order=left_order,
                    target_order=right_order,
                    candidate_keys=removed_keys,
                )
                or _neighbor_alignment_anchor(
                    best_match,
                    source_order=right_order,
                    target_order=left_order,
                    candidate_keys=added_keys,
                )
            ):
                continue
            resolved_removed.add(removed_key)
            resolved_added.add(best_match)
    added_keys -= resolved_added
    removed_keys -= resolved_removed
    if resolved_added:
        excluded_counts["near_stable"] = len(resolved_added)

    stable_keys = set(left_map.keys()) & set(right_map.keys())
    _apply_short_indicator_guard(added_keys, removed_keys, stable_keys, th, excluded_counts)

    added = sorted(right_map[key] for key in added_keys)
    removed = sorted(left_map[key] for key in removed_keys)
    added, removed, had_fusion_split = _detect_fusion_split(added, removed)

    remaining_added_keys = {
        key for value in added if (key := _canonical_indicator_key(strip_footnote_markers_from_indicator(value)))
    }
    remaining_removed_keys = {
        key for value in removed if (key := _canonical_indicator_key(strip_footnote_markers_from_indicator(value)))
    }
    filtered_added: set[str] = set()
    filtered_removed: set[str] = set()
    if neighbor_aligned_filter_enabled and (len(remaining_added_keys) + len(remaining_removed_keys) > 2):
        filtered_added = _filter_neighbor_aligned_candidates(remaining_added_keys, source_order=right_order, target_order=left_order)
        filtered_removed = _filter_neighbor_aligned_candidates(remaining_removed_keys, source_order=left_order, target_order=right_order)
    if filtered_added:
        excluded_counts["neighbor_aligned"] = excluded_counts.get("neighbor_aligned", 0) + len(filtered_added)
        added = [value for value in added if _canonical_indicator_key(strip_footnote_markers_from_indicator(value)) not in filtered_added]
    if filtered_removed:
        excluded_counts["neighbor_aligned"] = excluded_counts.get("neighbor_aligned", 0) + len(filtered_removed)
        removed = [value for value in removed if _canonical_indicator_key(strip_footnote_markers_from_indicator(value)) not in filtered_removed]

    diff_debug_info: dict[str, Any] | None = None
    if return_debug:
        diff_debug_info = {"left_map": left_map, "right_map": right_map}
    return sorted(added), sorted(removed), had_fusion_split, excluded_counts, diff_debug_info


def _build_indicator_diff_debug(
    table_t1: TableArtifact,
    table_t2: TableArtifact,
    left_map: dict[str, str],
    right_map: dict[str, str],
    added: list[str],
    removed: list[str],
    renamed_pairs: list[tuple[str, str]],
    t1_clean_to_raw: dict[str, str],
    t2_clean_to_raw: dict[str, str],
    indicator_debug: dict[str, Any] | None,
    th: dict[str, Any],
) -> list[dict[str, Any]]:
    """Construire une trace d'audit détaillée par indicateur pour le débogage.

    Pour chaque indicateur de T1 et T2, produit un enregistrement avec :
    - ``side`` : ``"t1"`` ou ``"t2"``.
    - ``raw`` : texte brut de l'indicateur (avant normalisation).
    - ``clean`` : texte nettoyé.
    - ``canonical_key`` : clé canonique utilisée pour la comparaison.
    - ``status`` : ``"stable"``, ``"renamed"``, ``"added"`` ou ``"removed"``.
    - ``matched_to`` : indicateur correspondant dans l'autre trimestre (si applicable).
    - ``score`` : score de similarité du rename (si applicable).
    - ``reason`` : raison de la décision (ex. ``"exact_canonical_match"``,
      ``"fuzzy_rename"``, ``"no_match_after_rename_pairing"``).

    Ces enregistrements sont écrits dans ``indicator_diff_debug.jsonl`` pour
    permettre l'audit et le débogage des décisions du moteur de diff.
    """
    decisions: list[dict[str, Any]] = []
    rename_pair_scores: dict[tuple[str, str], float] = {}
    if indicator_debug:
        for pair, dbg in zip(renamed_pairs, (indicator_debug.get("rename_pair_debug") or [])[: len(renamed_pairs)]):
            try:
                score = float(dbg.get("final_score", 0.0))
                if score > 1.0:
                    score = score / 100.0
                rename_pair_scores[pair] = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                pass
    min_score = float(th.get("indicator_rename_min_score", _INDICATOR_DEFAULTS["indicator_rename_min_score"]))
    for key, value_clean in left_map.items():
        raw = t1_clean_to_raw.get(key) or value_clean
        if key in right_map:
            status, matched_to, reason, score = "stable", right_map[key], "exact_canonical_match", 100.0
        else:
            pair = next(((r, a) for (r, a) in renamed_pairs if r == value_clean), None)
            if pair:
                _, matched_to = pair
                score = rename_pair_scores.get(pair)
                status = "renamed"
                reason = "fuzzy_rename" if score is None or score >= min_score else f"fuzzy_rename_below_threshold_{min_score}"
            else:
                status, matched_to, score, reason = "removed", None, None, "no_match_after_rename_pairing"
        decisions.append(
            {
                "side": "t1",
                "raw": raw[:200] if raw else "",
                "clean": value_clean[:200] if value_clean else "",
                "canonical_key": key[:200] if key else "",
                "status": status,
                "matched_to": (matched_to[:200] if matched_to else None) if matched_to else None,
                "score": round(score, 4) if score is not None else None,
                "reason": reason,
                "threshold_used": min_score if status == "renamed" else None,
            }
        )
    for key, value_clean in right_map.items():
        raw = t2_clean_to_raw.get(key) or value_clean
        if key in left_map:
            status, matched_to, reason, score = "stable", left_map[key], "exact_canonical_match", 100.0
        else:
            pair = next(((r, a) for (r, a) in renamed_pairs if a == value_clean), None)
            if pair:
                matched_to, _ = pair
                score = rename_pair_scores.get(pair)
                status = "renamed"
                reason = "fuzzy_rename" if score is None or score >= min_score else f"fuzzy_rename_below_threshold_{min_score}"
            else:
                status, matched_to, score, reason = "added", None, None, "no_match_after_rename_pairing"
        decisions.append(
            {
                "side": "t2",
                "raw": raw[:200] if raw else "",
                "clean": value_clean[:200] if value_clean else "",
                "canonical_key": key[:200] if key else "",
                "status": status,
                "matched_to": (matched_to[:200] if matched_to else None) if matched_to else None,
                "score": round(score, 4) if score is not None else None,
                "reason": reason,
                "threshold_used": min_score if status == "renamed" else None,
            }
        )
    return decisions


def _fuzzy_pair_added_removed(
    added: list[str],
    removed: list[str],
    bank_code: str | None,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Apparier les indicateurs supprimés et ajoutés comme renames (appariement glouton).

    Fallback utilisé quand ``scipy`` n'est pas disponible ou comme alternative
    à l'algorithme hongrois. Utilise un appariement glouton : trie toutes les
    paires candidates par score décroissant et sélectionne les meilleures sans
    réutiliser un indicateur déjà apparié.

    Moins optimal que l'algorithme hongrois (peut manquer des appariements
    globalement meilleurs) mais plus simple et sans dépendance scipy.

    Paramètres
    ----------
    added:
        Liste des indicateurs ajoutés.
    removed:
        Liste des indicateurs supprimés.
    bank_code:
        Code banque pour charger les seuils de similarité spécifiques.

    Retourne
    --------
    Tuple ``(added_restants, removed_restants, renamed_pairs)``.
    """
    if not added or not removed or rapidfuzz_fuzz is None:
        return added, removed, []
    th = get_indicator_diff_config(bank_code=bank_code)
    threshold = float(th.get("indicator_similarity_threshold", 0.88))
    token_threshold = float(th.get("indicator_fuzzy_token_threshold", 0.85))
    threshold_pct = int(threshold * 100)
    token_threshold_pct = int(token_threshold * 100)
    candidates: list[tuple[str, str, float]] = []
    for added_value in added:
        for removed_value in removed:
            ratio_score = rapidfuzz_fuzz.ratio(added_value, removed_value)
            token_score = rapidfuzz_fuzz.token_set_ratio(added_value, removed_value)
            score = max(ratio_score, token_score)
            if score >= threshold_pct or token_score >= token_threshold_pct:
                candidates.append((added_value, removed_value, float(score)))
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_added: set[str] = set()
    used_removed: set[str] = set()
    renamed_pairs: list[tuple[str, str]] = []
    for added_value, removed_value, _score in candidates:
        if added_value in used_added or removed_value in used_removed:
            continue
        used_added.add(added_value)
        used_removed.add(removed_value)
        renamed_pairs.append((removed_value, added_value))
    return (
        [value for value in added if value not in used_added],
        [value for value in removed if value not in used_removed],
        renamed_pairs,
    )
