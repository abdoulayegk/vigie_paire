"""Lecture et écriture du fichier text_comparison.json."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from vigie.analyse_texte.text_comparison.change_segments import build_change_segments
from vigie.analyse_texte.text_comparison.justification import (
    build_text_triage_justification,
    is_structured_text_triage_justification,
)

logger = logging.getLogger(__name__)

TEXT_COMPARISON_SCHEMA_VERSION = 3
_ACCEPTED_TEXT_COMPARISON_SCHEMAS = {TEXT_COMPARISON_SCHEMA_VERSION}
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text_value(value: Any) -> Any:
    """Retire les caractères de contrôle illégaux pour JSON/Dash/Excel."""
    if isinstance(value, str):
        return _CONTROL_CHAR_RE.sub("", value)
    if isinstance(value, list):
        return [_sanitize_text_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_text_value(item) for key, item in value.items()}
    return value


def _fallback_change_segments(change: dict[str, Any]) -> list[dict[str, str]]:
    """Construit les segments de preuve quand le triage n'en contient pas."""
    return build_change_segments(change)


def _normalize_text_change(change: dict[str, Any]) -> None:
    """Garantit les champs nécessaires au rendu analyste de l'analyse textuelle."""
    triage = change.get("genai_triage")
    if not isinstance(triage, dict) or not triage:
        return

    justification = build_text_triage_justification(change)
    if justification and not is_structured_text_triage_justification(triage.get("nouvelle_idee_justification")):
        triage["nouvelle_idee_justification"] = justification

    if bool(triage.get("is_relevant", False)):
        segments = triage.get("change_segments")
        if not isinstance(segments, list) or not segments:
            fallback_segments = _fallback_change_segments(change)
            if fallback_segments:
                triage["change_segments"] = fallback_segments
    else:
        triage["change_segments"] = []


def extract_term_replacement_pattern(summary: str) -> str:
    """Extrait un motif de remplacement de termes ou de sous-section."""
    m = re.search(
        r"\"([^\"]+)\"\s*(?:par|en|à|remplacé par|modifié en)\s*\"([^\"]+)\"",
        summary,
        re.IGNORECASE,
    )
    if m:
        return f"term_replace:{m.group(1).strip().lower()}->{m.group(2).strip().lower()}"

    m_del = re.search(
        r"(?:passage(?: de sous-section)? supprimé|suppression(?: de sous-section)?)\s*:\s*([^\n\.]+)",
        summary,
        re.IGNORECASE,
    )
    if m_del:
        return f"section_del:{m_del.group(1).strip().lower()}"

    m_add = re.search(
        r"(?:passage(?: de sous-section)? ajouté|ajout(?: de sous-section)?)\s*:\s*([^\n\.]+)",
        summary,
        re.IGNORECASE,
    )
    if m_add:
        return f"section_add:{m_add.group(1).strip().lower()}"

    return ""


def deduplicate_and_group_section_changes(
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dédoublonne et regroupe les changements d'une section par liste de pages.

    Conserve 1 seule entrée par changement distinct et agrège les numéros de page
    (ex. pages_t1: [98, 99, 101, 110]).
    """
    grouped: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}

    for change in changes:
        if not isinstance(change, dict):
            continue
        dt = str(change.get("diff_type") or "").strip().lower()
        summary = str(change.get("change_summary") or "").strip()
        term_pattern = extract_term_replacement_pattern(summary)
        t1 = str(change.get("source_text_t1") or change.get("semantic_text_t1") or "").strip()
        t2 = str(change.get("source_text_t2") or change.get("semantic_text_t2") or "").strip()
        sub = str(change.get("subsection_heading") or "").strip().lower()

        if term_pattern:
            key = (dt, f"term:{term_pattern}")
        elif dt in ("added", "removed") and (t1 or t2):
            key = (dt, f"{dt}:{sub}:{(t1 or t2)[:120].lower()}")
        elif summary:
            key = (dt, f"sum:{summary.lower()}")
        elif t1 and t2:
            key = (dt, f"t1_t2:{t1[:120].lower()}||{t2[:120].lower()}")
        elif t1:
            key = (dt, f"t1:{t1[:120].lower()}")
        elif t2:
            key = (dt, f"t2:{t2[:120].lower()}")
        else:
            key = (dt, f"raw:{len(grouped)}")

        if key in seen:
            idx = seen[key]
            existing = grouped[idx]

            p1 = list(existing.get("pages_t1") or [])
            for p in change.get("pages_t1") or []:
                if p and p not in p1:
                    p1.append(p)
            existing["pages_t1"] = sorted(p1)

            p2 = list(existing.get("pages_t2") or [])
            for p in change.get("pages_t2") or []:
                if p and p not in p2:
                    p2.append(p)
            existing["pages_t2"] = sorted(p2)

            if "evidence_t1" in existing and isinstance(existing["evidence_t1"], dict):
                existing["evidence_t1"]["pages"] = existing["pages_t1"]
            if "evidence_t2" in existing and isinstance(existing["evidence_t2"], dict):
                existing["evidence_t2"]["pages"] = existing["pages_t2"]
        else:
            c_copy = dict(change)
            c_copy["pages_t1"] = list(change.get("pages_t1") or [])
            c_copy["pages_t2"] = list(change.get("pages_t2") or [])
            if "evidence_t1" in c_copy and isinstance(c_copy["evidence_t1"], dict):
                c_copy["evidence_t1"]["pages"] = list(c_copy["pages_t1"])
            if "evidence_t2" in c_copy and isinstance(c_copy["evidence_t2"], dict):
                c_copy["evidence_t2"]["pages"] = list(c_copy["pages_t2"])
            seen[key] = len(grouped)
            grouped.append(c_copy)

    return grouped


def _normalize_text_comparison_payload(payload: dict[str, Any]) -> None:
    """Normalise tous les changements avant écriture du JSON final."""
    for section in payload.get("section_comparisons") or []:
        if not isinstance(section, dict):
            continue
        for bucket in ("block_comparisons", "all_block_comparisons"):
            raw_changes = section.get(bucket) or []
            if raw_changes:
                deduped = deduplicate_and_group_section_changes(raw_changes)
                section[bucket] = deduped
            for change in section.get(bucket) or []:
                if isinstance(change, dict):
                    _normalize_text_change(change)


def get_text_comparison_path(
    out_root: Path,
    bank_code: str,
    year_t2: int,
    quarter_t2: str,
    year_t1: int,
    quarter_t1: str,
) -> Path:
    """Retourne le chemin canonique du fichier text_comparison.json.

    Pattern : out_root/{bank}/{year_t2}_{qt2}_vs_{year_t1}_{qt1}/text_comparison.json
    Exemple  : outputs/resultats/bns/2025_t2_vs_2025_t1/text_comparison.json
    """
    folder = f"{year_t2}_{quarter_t2.lower()}_vs_{year_t1}_{quarter_t1.lower()}"
    return out_root / bank_code.lower() / folder / "text_comparison.json"


def write_text_comparison(
    payload: dict[str, Any],
    out_path: Path,
) -> Path:
    """Sérialise et écrit text_comparison.json.

    Args:
        payload: Dictionnaire texte canonique.
        out_path: Chemin complet du fichier de sortie.

    Returns:
        Path du fichier écrit.
    """
    _normalize_text_comparison_payload(payload)
    sanitized_payload = _sanitize_text_value(payload)
    if isinstance(sanitized_payload, dict):
        payload.clear()
        payload.update(sanitized_payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total_changes = sum(len(sc.get("block_comparisons", [])) for sc in payload.get("section_comparisons", []))
    logger.info(
        "text_comparison.json écrit : %s (%d sections, %d changements)",
        out_path,
        len(payload.get("section_comparisons", [])),
        total_changes,
    )
    return out_path


def load_text_comparison(comparison_path: Path) -> dict[str, Any]:
    """Charge text_comparison.json et valide le schema_version.

    Args:
        comparison_path: Chemin vers text_comparison.json.

    Returns:
        Dictionnaire chargé.

    Raises:
        FileNotFoundError: Si le fichier est absent.
        ValueError: Si le schema_version est incompatible.
    """
    if not comparison_path.exists():
        raise FileNotFoundError(f"text_comparison.json introuvable : {comparison_path}")

    data = json.loads(comparison_path.read_text(encoding="utf-8"))

    version = data.get("schema_version")
    if version not in _ACCEPTED_TEXT_COMPARISON_SCHEMAS:
        raise ValueError(
            f"schema_version incompatible : accepté {_ACCEPTED_TEXT_COMPARISON_SCHEMAS}, "
            f"trouvé {version} dans {comparison_path}"
        )
    return data
