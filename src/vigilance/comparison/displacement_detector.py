"""
Detection des deplacements d'indicateurs entre tableaux.

Detecte quand un indicateur presente le meme contenu (apres normalisation canonique)
apparait comme "supprime" dans un tableau et "ajoute" dans un autre tableau.
Dans ce cas, il s'agit d'un deplacement (autre page, autre tableau) et non d'un
vrai ajout + suppression.

Supporte deux modes de matching :
- exact : canonical identique (comportement original)
- fuzzy : similarite canonique >= seuil configurable (nouveau)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}

from vigilance.comparison.scoring_engine import compute_candidate_score


@dataclass
class DisplacedIndicator:
    """Indicateur deplace d'un tableau a un autre (meme contenu, autre page/tableau)."""

    canonical: str
    to_canonical: str
    text_display: str
    from_table_id: str
    from_page: int
    to_table_id: str
    to_page: int
    section: str | None = None
    match_type: str = "exact"
    similarity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "to_canonical": self.to_canonical,
            "text_display": self.text_display,
            "from_table_id": self.from_table_id,
            "from_page": self.from_page,
            "to_table_id": self.to_table_id,
            "to_page": self.to_page,
            "section": self.section,
            "match_type": self.match_type,
            "similarity": self.similarity,
        }


@dataclass
class RemovedItem:
    """Item supprime avec son contexte (tableau, page, cle de comparaison)."""

    text: str
    canonical: str
    table_id_t1: str
    page_t1: int
    table_id_t2: str
    page_t2: int
    section: str | None = None
    comparison_key: tuple[str, ...] = field(default_factory=tuple)
    neighbor_prev: str | None = None
    neighbor_next: str | None = None


@dataclass
class AddedItem:
    """Item ajoute avec son contexte."""

    text: str
    canonical: str
    table_id_t1: str
    page_t1: int
    table_id_t2: str
    page_t2: int
    section: str | None = None
    comparison_key: tuple[str, ...] = field(default_factory=tuple)
    neighbor_prev: str | None = None
    neighbor_next: str | None = None


def _sections_compatible(
    section_a: str | None, section_b: str | None, section_strict: bool
) -> bool:
    """Verifie si deux sections sont compatibles pour un deplacement."""
    if not section_strict:
        return True
    value_a = (section_a or "").strip().lower()
    value_b = (section_b or "").strip().lower()
    if value_a in UNKNOWN_SECTIONS or value_b in UNKNOWN_SECTIONS:
        return False
    return value_a == value_b


def detect_cross_table_displacements(
    removed_items: list[RemovedItem],
    added_items: list[AddedItem],
    canonical_fn: Callable[[str], str],
    # similarity_fn: OBSOLETE (remplace par scoring_engine)
    fuzzy_threshold: float = 0.90,
    section_strict: bool = True,
) -> tuple[set[str], list[DisplacedIndicator]]:
    """
    Detecte les indicateurs deplaces (meme contenu dans removed et added, tableaux differents).

    Un deplacement = canonical X apparait dans un removed (tableau A) ET dans un added
    (tableau B), avec A et B etant des paires de tableaux differentes.

    Passe 1 : matching exact canonique (comportement original)
    Passe 2 : matching flou pour les residuels (si similarity_fn fourni)

    Args:
        removed_items: Liste des items supprimes avec contexte
        added_items: Liste des items ajoutes avec contexte
        canonical_fn: Utilise pour normaliser si .canonical absent (legacy)
        similarity_fn: Fonction de similarite (retourne dict avec cle 'score')
        fuzzy_threshold: Seuil minimum pour le matching flou
        section_strict: Si True, restreint les deplacements a la meme section

    Returns:
        Tuple (set des canonicals deplaces, liste des DisplacedIndicator)
    """
    displaced_canonicals: set[str] = set()
    displaced_list: list[DisplacedIndicator] = []

    # Indexer removed par canonical -> liste d'items
    removed_by_canonical: dict[str, list[RemovedItem]] = {}
    for item in removed_items:
        c = (
            item.canonical
            if hasattr(item, "canonical") and item.canonical
            else canonical_fn(item.text)
        )
        if not c:
            continue
        removed_by_canonical.setdefault(c, []).append(item)

    added_by_canonical: dict[str, list[AddedItem]] = {}
    for item in added_items:
        c = (
            item.canonical
            if hasattr(item, "canonical") and item.canonical
            else canonical_fn(item.text)
        )
        if not c:
            continue
        added_by_canonical.setdefault(c, []).append(item)

    # ── Passe 1 : matching exact canonique ──
    used_rem_ids: set[int] = set()
    used_add_ids: set[int] = set()

    for canonical in set(removed_by_canonical.keys()) & set(added_by_canonical.keys()):
        rem_list = list(removed_by_canonical[canonical])
        add_list = list(added_by_canonical[canonical])

        for rem in rem_list:
            if id(rem) in used_rem_ids:
                continue
            for add in add_list:
                if id(add) in used_add_ids:
                    continue
                if not rem.comparison_key or not add.comparison_key:
                    continue
                if rem.comparison_key == add.comparison_key:
                    continue
                if not _sections_compatible(rem.section, add.section, section_strict):
                    continue
                displaced_canonicals.add(canonical)
                displaced_list.append(
                    DisplacedIndicator(
                        canonical=canonical,
                        to_canonical=canonical,
                        text_display=rem.text,
                        from_table_id=rem.table_id_t1,
                        from_page=rem.page_t1,
                        to_table_id=add.table_id_t2,
                        to_page=add.page_t2,
                        section=rem.section or add.section,
                        match_type="exact",
                        similarity=1.0,
                    )
                )
                used_rem_ids.add(id(rem))
                used_add_ids.add(id(add))
                break

    # ── Passe 2 : matching flou sur les residuels ──
    # if similarity_fn is None: -> On utilise toujours scoring_engine maintenant
    #    return displaced_canonicals, displaced_list

    residual_removed = [item for item in removed_items if id(item) not in used_rem_ids]
    residual_added = [item for item in added_items if id(item) not in used_add_ids]

    if not residual_removed or not residual_added:
        return displaced_canonicals, displaced_list

    # Construire toutes les paires candidates avec score
    candidates: list[tuple[float, RemovedItem, AddedItem]] = []
    for rem in residual_removed:
        rem_canonical = (
            rem.canonical if hasattr(rem, "canonical") and rem.canonical else canonical_fn(rem.text)
        )
        if not rem_canonical:
            continue
        for add in residual_added:
            if not rem.comparison_key or not add.comparison_key:
                continue
            if rem.comparison_key == add.comparison_key:
                continue
            if not _sections_compatible(rem.section, add.section, section_strict):
                continue
            add_canonical = (
                add.canonical
                if hasattr(add, "canonical") and add.canonical
                else canonical_fn(add.text)
            )
            if not add_canonical:
                continue
            if rem_canonical == add_canonical:
                continue  # deja traite en passe 1

            # Appel au moteur unifie
            cand_ctx = {
                "section": add.section,
                "table_id": add.table_id_t2,
                "page": add.page_t2,
                "group": "unknown",  # Pas dispo ici
                "neighbor_prev": add.neighbor_prev,
                "neighbor_next": add.neighbor_next,
            }
            ref_ctx = {
                "section": rem.section,
                "table_id": rem.table_id_t1,
                "page": rem.page_t1,
                "group": "unknown",
                "neighbor_prev": rem.neighbor_prev,
                "neighbor_next": rem.neighbor_next,
            }

            result = compute_candidate_score(
                candidate_text=add.canonical or canonical_fn(add.text),  # Fallback safety
                reference_text=rem.canonical or canonical_fn(rem.text),
                candidate_context=cand_ctx,
                reference_context=ref_ctx,
                robust_mode=False,
                section_strict=False,  # Deja filtre par section, laisser la logique de score faire
            )

            # Note: Le moteur fait aussi check_context_compatibility mais on l'a deja fait (section_strict)
            # On utilise le score composite
            score = result["composite_score"]

            if score >= fuzzy_threshold:
                candidates.append((score, rem, add))

    # Greedy bijective : meilleur score d'abord
    candidates.sort(key=lambda x: x[0], reverse=True)
    fuzzy_used_rem: set[int] = set()
    fuzzy_used_add: set[int] = set()

    for score, rem, add in candidates:
        if id(rem) in fuzzy_used_rem or id(add) in fuzzy_used_add:
            continue
        rem_canonical = (
            rem.canonical if hasattr(rem, "canonical") and rem.canonical else canonical_fn(rem.text)
        )
        add_canonical = (
            add.canonical if hasattr(add, "canonical") and add.canonical else canonical_fn(add.text)
        )
        displaced_canonicals.add(rem_canonical)
        displaced_list.append(
            DisplacedIndicator(
                canonical=rem_canonical,
                to_canonical=add_canonical,
                text_display=rem.text,
                from_table_id=rem.table_id_t1,
                from_page=rem.page_t1,
                to_table_id=add.table_id_t2,
                to_page=add.page_t2,
                section=rem.section or add.section,
                match_type="fuzzy",
                similarity=round(score, 3),
            )
        )
        fuzzy_used_rem.add(id(rem))
        fuzzy_used_add.add(id(add))

    return displaced_canonicals, displaced_list
