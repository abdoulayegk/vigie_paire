"""
Moteur de scoring unifie pour la comparaison d'indicateurs (lignes).

Centralise la logique de calcul de score et de decision pour garantir une uniformite
entre les differentes phases (near-exact, ambiguous, rename, orphans).
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None


# Poids par defaut (Mode Normal)
WEIGHTS_NORMAL = {
    "text": 0.85,
    "position": 0.10,
    "neighborhood": 0.05,
}

# Poids pour le mode Robuste (structure change, review needed)
# On privilegie encore plus le texte car la structure est instable
WEIGHTS_ROBUST = {
    "text": 0.95,
    "position": 0.03,
    "neighborhood": 0.02,
}

# Seuils de decision (bases sur text_score)
THRESHOLD_MATCH = 0.90
THRESHOLD_PROBABLE = 0.75  # Pour le mode ambigu/rename


def compute_candidate_score(
    candidate_text: str,
    reference_text: str,
    candidate_context: dict[str, Any],
    reference_context: dict[str, Any],
    robust_mode: bool = False,
    section_strict: bool = True,
) -> dict[str, Any]:
    """
    Calculer le score composite et la decision pour une paire candidat/reference.

    Args:
        candidate_text: Texte du candidat (T2)
        reference_text: Texte de reference (T1)
        candidate_context: Dict avec {section, table_id, group, page, neighbor_prev, neighbor_next}
        reference_context: Dict avec {section, table_id, group, page, neighbor_prev, neighbor_next}
        robust_mode: Si True, utilise des poids plus stricts sur le texte
        section_strict: Si True, exige une correspondance exacte de section (sauf si unknown)

    Returns:
        Dict contenant tous les scores et la decision:
        {
            "text_score": float,
            "position_score": float,
            "neighborhood_score": float,
            "composite_score": float,
            "length_ratio": float,
            "is_context_compatible": bool,
            "decision": str,  # MATCH, PROBABLE, NO_MATCH
            "reason": str,
            "details": dict  # Sous-scores detailles
        }
    """
    # 0. Verification contexte (Garde-fous)
    is_compatible = _check_context_compatibility(
        candidate_context, reference_context, section_strict
    )
    if not is_compatible:
        return _build_result(0.0, 0.0, 0.0, 0.0, 0.0, False, "NO_MATCH", "Contexte incompatible")

    # 1. Score Textuel
    text_score, text_details = _compute_text_score(candidate_text, reference_text)
    length_ratio = text_details["length_ratio"]

    # 2. Score Position
    pos_score = _compute_position_score(candidate_context, reference_context)

    # 3. Score Voisinage
    neigh_score = _compute_neighborhood_score(candidate_context, reference_context)

    # 4. Score Composite
    weights = WEIGHTS_ROBUST if robust_mode else WEIGHTS_NORMAL
    composite_score = (
        (text_score * weights["text"])
        + (pos_score * weights["position"])
        + (neigh_score * weights["neighborhood"])
    )

    # 5. Decision
    decision, reason = _make_decision(text_score, composite_score, robust_mode)

    return _build_result(
        text_score,
        pos_score,
        neigh_score,
        composite_score,
        length_ratio,
        True,
        decision,
        reason,
        text_details,
    )


def _check_context_compatibility(
    cand_ctx: dict[str, Any], ref_ctx: dict[str, Any], section_strict: bool
) -> bool:
    """Verifier si le contexte est compatible (section, groupe, table)."""
    # Section
    if section_strict:
        sec_c = cand_ctx.get("section", "unknown")
        sec_r = ref_ctx.get("section", "unknown")
        if sec_c != "unknown" and sec_r != "unknown" and sec_c != sec_r:
            return False

    # Groupe (ex: Actif, Passif, etc.)
    grp_c = cand_ctx.get("group", "unknown")
    grp_r = ref_ctx.get("group", "unknown")
    if grp_c != "unknown" and grp_r != "unknown" and grp_c != grp_r:
        return False

    return True


def _compute_text_score(text1: str, text2: str) -> tuple[float, dict]:
    """Calculer la similarite textuelle (max de plusieurs methodes)."""
    if not text1 or not text2:
        return 0.0, {"seq": 0.0, "token": 0.0, "jaccard": 0.0, "length_ratio": 0.0}

    # Rapports de longueur
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2)

    # 1. SequenceMatcher (sensible a l'ordre)
    seq_score = SequenceMatcher(None, text1, text2).ratio()

    # 2. Token Sort (Rapidfuzz si dispo)
    token_score = 0.0
    if rapidfuzz_fuzz:
        token_score = rapidfuzz_fuzz.token_sort_ratio(text1, text2) / 100.0

    # 3. Jaccard (ensemble de mots)
    set1 = set(text1.split())
    set2 = set(text2.split())
    jaccard_score = 0.0
    if set1 or set2:
        jaccard_score = len(set1 & set2) / len(set1 | set2)

    # Max des scores
    final_score = max(seq_score, token_score, jaccard_score)

    return final_score, {
        "seq": seq_score,
        "token": token_score,
        "jaccard": jaccard_score,
        "length_ratio": length_ratio,
    }


def _compute_position_score(cand_ctx: dict[str, Any], ref_ctx: dict[str, Any]) -> float:
    """Calculer le score de position (proximite page/ordre)."""
    # Page
    page_c = cand_ctx.get("page", 0)
    page_r = ref_ctx.get("page", 0)

    if page_c == 0 or page_r == 0:
        return 0.5  # Neutre si info manquante

    diff = abs(page_c - page_r)
    # Penalite exponentielle : 0->1.0, 1->0.8, 2->0.5, 5+->0.0
    if diff == 0:
        return 1.0
    elif diff <= 1:
        return 0.8
    elif diff <= 2:
        return 0.5
    else:
        return 0.1


def _compute_neighborhood_score(cand_ctx: dict[str, Any], ref_ctx: dict[str, Any]) -> float:
    """Calculer le score de voisinage (overlap des voisins prev/next)."""
    # Pour l'instant, implementation simple basique sur presence
    # Idealement: comparer le texte des voisins
    score = 0.0
    count = 0

    # Voisin precedent
    prev_c = cand_ctx.get("neighbor_prev")
    prev_r = ref_ctx.get("neighbor_prev")
    if prev_c and prev_r:
        score += _simple_similarity(prev_c, prev_r)
        count += 1

    # Voisin suivant
    next_c = cand_ctx.get("neighbor_next")
    next_r = ref_ctx.get("neighbor_next")
    if next_c and next_r:
        score += _simple_similarity(next_c, next_r)
        count += 1

    if count == 0:
        return 0.5  # Neutre

    return score / count


def _simple_similarity(t1: str, t2: str) -> float:
    return SequenceMatcher(None, t1, t2).ratio()


def _make_decision(text_score: float, composite_score: float, robust_mode: bool) -> tuple[str, str]:
    """Prendre une decision basee sur le score."""
    # En mode normal, on est plus tolerant
    thresh_match = THRESHOLD_MATCH
    thresh_prob = THRESHOLD_PROBABLE

    if robust_mode:
        # En mode robuste, on exige plus de certitude textuelle
        thresh_match = 0.95
        thresh_prob = 0.85

    if text_score >= thresh_match:
        return "MATCH", "Score textuel excellent"
    elif text_score >= thresh_prob:
        if composite_score > text_score:
            return "MATCH", "Score composite eleve (contexte favorable)"
        return "PROBABLE", "Score textuel moyen, a verifier"

    return "NO_MATCH", "Score insuffisant"


def _build_result(
    text: float,
    pos: float,
    neigh: float,
    comp: float,
    lr: float,
    compatible: bool,
    dec: str,
    reason: str,
    details: dict | None = None,
) -> dict[str, Any]:
    return {
        "text_score": round(text, 4),
        "position_score": round(pos, 4),
        "neighborhood_score": round(neigh, 4),
        "composite_score": round(comp, 4),
        "length_ratio": round(lr, 4),
        "is_context_compatible": compatible,
        "decision": decision_to_label(dec),
        "reason": reason,
        "details": details or {},
    }


def decision_to_label(decision: str) -> str:
    """Normaliser la decision."""
    decision = decision.upper()
    if decision in ("MATCH", "EXACT", "NEAR_EXACT"):
        return "MATCH"
    if decision in ("PROBABLE", "AMBIGUOUS", "RENAME"):
        return "PROBABLE"
    return "NO_MATCH"
