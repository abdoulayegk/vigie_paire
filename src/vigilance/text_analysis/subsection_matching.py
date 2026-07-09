"""Composants modulaires du pipeline texte."""

from __future__ import annotations

import logging
import re
from typing import Any

from vigilance.text_analysis.constants import _SUBSECTION_SPLIT_RE
from vigilance.text_analysis.normalization import _sanitize_semantic_text
from vigilance.text_analysis.openai_client import _call_json_completion

logger = logging.getLogger(__name__)


def _normalize_heading(heading: str) -> str:
    """Normalise un heading ### pour le pairing T1/T2 (insensible à la casse et aux préfixes de tableaux)."""
    h = heading.lower()
    h = re.sub(r"\s*\[(?:pdf|p)\.?\s*\d+(?:\s*[-–]\s*\d+)?\]\s*", " ", h, flags=re.IGNORECASE)
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
    norm_to_t2: dict[str, tuple[str, str]] = {_normalize_heading(h): (h, body) for h, body in subs_t2}
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


def _synthetic_subsection_rename_change(
    *,
    section_key: str,
    heading_t1: str,
    heading_t2: str,
    idx: int,
) -> dict[str, Any]:
    """Crée un changement explicite pour une sous-section renommée."""
    slug_source = f"{heading_t1}_{heading_t2}"
    slug = re.sub(r"[^\w]+", "_", _normalize_heading(slug_source))[:40].strip("_")
    summary = f"Sous-section renommée: {heading_t1} -> {heading_t2}"
    return {
        "change_id": f"{section_key}_{slug}_change_{idx:03d}",
        "section_key": section_key,
        "subsection_heading": f"{heading_t1} → {heading_t2}",
        "previous_subsection_heading": heading_t1,
        "current_subsection_heading": heading_t2,
        "diff_type": "renamed",
        "semantic_text_t1": _sanitize_semantic_text(heading_t1),
        "semantic_text_t2": _sanitize_semantic_text(heading_t2),
        "source_text_t1": heading_t1,
        "source_text_t2": heading_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown_heading",
        "source_resolution_t2": "markdown_heading",
        "evidence_t1": {"pages": [], "snippet": heading_t1},
        "evidence_t2": {"pages": [], "snippet": heading_t2},
        "change_summary": summary,
    }
