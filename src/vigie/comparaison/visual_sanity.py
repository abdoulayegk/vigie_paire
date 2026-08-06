"""Verification visuelle de coherence (Visual Sanity Check) via GPT-4o Vision.

Ce module verifie les changements de comparaison deja calcules en les confrontant
aux rendus finaux (epreuves) utilises par le produit : images pleine page avec la
zone du tableau cible mise en evidence. Il agit comme un filtre conservateur de
derniere passe et ne retire que les faux positifs visuels.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Callable

from vigie.comparaison.noise_filter import recompute_table_level_change
from vigie.support.models.comparison_models import VisualSanityCheckResponse
from vigie.support.utils.proof_rendering import PROOF_RENDER_DPI, render_full_proof_bytes

logger = logging.getLogger(__name__)

VISUAL_SANITY_SCOPE = ["indicators", "footnotes", "tables"]

VISUAL_SANITY_SYSTEM_PROMPT = """\
You are a Senior Risk Analyst performing a FINAL VISUAL verification of computed
changes between two quarterly banking table proof renders.

You receive:
1. The final proof render for the Previous Quarter (PQ): full page with the
   target table region highlighted when an exact table anchor exists.
2. The final proof render for the Current Quarter (CQ): full page with the
   target table region highlighted when an exact table anchor exists.
3. A typed list of alleged changes.

For an added or removed table, one proof may intentionally show the inferred
opposite-quarter page without a highlighted region because no corresponding
table anchor exists. In that case, inspect the entire inferred page for the
alleged table or a clearly equivalent table.
The highlighted region, when present, indicates the target table.
Use the proof renders as the final source of truth.

Your mission:
- For EACH item, decide whether it is visually confirmed or should be rejected
  as a false positive.

Rules by item_type:
- indicator_added: visible in CQ and absent in PQ.
- indicator_removed: visible in PQ and absent in CQ.
- indicator_renamed: previous label visible in PQ, current label visible in CQ.
- footnote_added: note visible below the target table in CQ and absent in PQ.
- footnote_removed: note visible below the target table in PQ and absent in CQ.
- footnote_modified: the logical note exists on both sides AND the text is
  materially changed. Pure renumbering or cosmetic wording changes must be rejected.
- table_added: a table is visibly present in the highlighted CQ region and absent
  from the highlighted PQ region.
- table_removed: a table is visibly present in the highlighted PQ region and absent
  from the highlighted CQ region.

Conservative policy:
- Reject an item only when the proof renders clearly contradict the computed diff.
- If the proof is ambiguous or unreadable, keep the item confirmed.
- Reject pure numbering, footnote marker, OCR, or formatting-only differences.
- Never invent new changes. Evaluate only the provided items.

Return JSON following the response_schema exactly. Each verdict MUST echo the
exact item_id provided in the input.
"""


def _default_meta(
    *,
    applied: bool,
    rejected_count: int,
    render_status: str,
) -> dict[str, Any]:
    """Construit le dictionnaire de metadonnees par defaut pour le controle visuel."""
    return {
        "visual_sanity_applied": applied,
        "visual_sanity_rejected_count": int(rejected_count),
        "visual_sanity_scope": list(VISUAL_SANITY_SCOPE),
        "visual_sanity_render_mode": "full",
        "visual_sanity_render_status": render_status,
    }


def render_visual_sanity_proof(
    pdf_path: str | None,
    *,
    page: Any,
    bbox: Any,
    dpi: int = PROOF_RENDER_DPI,
    allow_full_page_fallback: bool = False,
) -> tuple[bytes | None, str]:
    """Genere une image d'epreuve finale pour le controle visuel de coherence.

    Args:
        pdf_path: Chemin vers le fichier PDF source. ``None`` si indisponible.
        page: Numero ou reference de la page cible.
        bbox: Coordonnees de la zone du tableau (bounding box).
        dpi: Resolution de rendu PDF (defaut :data:`PROOF_RENDER_DPI`).
        allow_full_page_fallback: Rendre la page entiere quand la bbox manque.

    Returns:
        Tuple ``(image_bytes, status)`` ou *status* est l'un de :
        ``ok``, ``skipped_missing_pdf``, ``skipped_missing_bbox`` ou
        ``skipped_render_failed``.
    """
    raw, status, _ = render_full_proof_bytes(
        pdf_path,
        page=page,
        bbox=bbox,
        dpi=dpi,
        allow_full_page_fallback=allow_full_page_fallback,
    )
    if status == "ok":
        return raw, "ok"
    if status == "pdf_missing":
        return None, "skipped_missing_pdf"
    if status == "bbox_missing":
        return None, "skipped_missing_bbox"
    return None, "skipped_render_failed"


def _build_item_id(prefix: str, *parts: str) -> str:
    """Construit un identifiant unique pour un element de verification."""
    clean = [str(part or "").strip() for part in parts]
    return prefix + "::" + " | ".join(clean)


def _build_sanity_items_from_diff(technical_diff: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrait les elements a verifier visuellement depuis le diff technique."""
    items: list[dict[str, Any]] = []

    for entry in list(technical_diff.get("indicators_added", []) or []):
        value = str(entry.get("value", "") or "").strip()
        if value:
            items.append(
                {
                    "item_id": _build_item_id("indicator_added", value),
                    "item_type": "indicator_added",
                    "value": value,
                }
            )
    for entry in list(technical_diff.get("indicators_removed", []) or []):
        value = str(entry.get("value", "") or "").strip()
        if value:
            items.append(
                {
                    "item_id": _build_item_id("indicator_removed", value),
                    "item_type": "indicator_removed",
                    "value": value,
                }
            )
    for entry in list(technical_diff.get("indicators_renamed", []) or []):
        previous = str(entry.get("previous", "") or "").strip()
        current = str(entry.get("current", "") or "").strip()
        if previous or current:
            items.append(
                {
                    "item_id": _build_item_id("indicator_renamed", previous, current),
                    "item_type": "indicator_renamed",
                    "previous": previous,
                    "current": current,
                }
            )

    for entry in list(technical_diff.get("footnotes_added", []) or []):
        fid = str(entry.get("id", "") or "").strip()
        text = str(entry.get("text", "") or "").strip()
        if fid or text:
            items.append(
                {
                    "item_id": _build_item_id("footnote_added", fid, text),
                    "item_type": "footnote_added",
                    "id": fid,
                    "text": text,
                }
            )
    for entry in list(technical_diff.get("footnotes_removed", []) or []):
        fid = str(entry.get("id", "") or "").strip()
        text = str(entry.get("text", "") or "").strip()
        if fid or text:
            items.append(
                {
                    "item_id": _build_item_id("footnote_removed", fid, text),
                    "item_type": "footnote_removed",
                    "id": fid,
                    "text": text,
                }
            )
    for entry in list(technical_diff.get("footnotes_renamed", []) or []):
        previous_id = str(entry.get("previous_id", "") or "").strip()
        current_id = str(entry.get("current_id", "") or "").strip()
        previous_text = str(entry.get("previous_text", "") or "").strip()
        current_text = str(entry.get("current_text", "") or "").strip()
        if previous_id or current_id or previous_text or current_text:
            items.append(
                {
                    "item_id": _build_item_id(
                        "footnote_modified",
                        previous_id,
                        current_id,
                        previous_text,
                        current_text,
                    ),
                    "item_type": "footnote_modified",
                    "previous_id": previous_id,
                    "current_id": current_id,
                    "previous_text": previous_text,
                    "current_text": current_text,
                }
            )
    return items


def _build_table_event_item(
    *,
    event_type: str,
    table_id: str,
    table_title: str,
) -> list[dict[str, Any]]:
    """Construit l'element de verification pour un evenement tableau (ajout/suppression)."""
    normalized_type = str(event_type or "").strip().lower()
    if normalized_type not in {"table_added", "table_removed"}:
        return []
    return [
        {
            "item_id": _build_item_id(normalized_type, table_id, table_title),
            "item_type": normalized_type,
            "table_id": str(table_id or "").strip(),
            "table_title": str(table_title or "").strip(),
        }
    ]


def _call_visual_sanity_llm(
    previous_render_bytes: bytes,
    current_render_bytes: bytes,
    *,
    items: list[dict[str, Any]],
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Appelle le LLM Vision pour verifier les elements contre les epreuves."""
    pq_b64 = base64.standard_b64encode(previous_render_bytes).decode("ascii")
    cq_b64 = base64.standard_b64encode(current_render_bytes).decode("ascii")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VISUAL_SANITY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Verify every listed item against the final proof renders.\n\n"
                        f"Items to verify:\n{json.dumps(items, ensure_ascii=False)}"
                    ),
                },
                {"type": "text", "text": "Previous Quarter (PQ) proof render:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{pq_b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": "Current Quarter (CQ) proof render:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{cq_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    try:
        return call_openai_json(
            model=model,
            messages=messages,
            temperature=0.0,
            usage_recorder=usage_recorder,
            call_kind="visual_sanity_check",
            response_model=VisualSanityCheckResponse,
        )
    except Exception as exc:
        logger.warning("Visual sanity check: API call failed: %s", exc)
        return None


def _rejected_item_ids(data: dict[str, Any] | None) -> set[str]:
    """Extrait les identifiants des elements rejetes par le LLM."""
    rejected: set[str] = set()
    if not isinstance(data, dict):
        return rejected
    for verdict in list(data.get("verdicts", []) or []):
        if not isinstance(verdict, dict):
            continue
        if str(verdict.get("verdict", "")).strip().lower() != "rejected":
            continue
        item_id = str(verdict.get("item_id", "") or "").strip()
        if item_id:
            rejected.add(item_id)
    return rejected


def visual_sanity_check(
    previous_render_bytes: bytes | None,
    current_render_bytes: bytes | None,
    diff_result: dict[str, Any],
    *,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Filtre les faux positifs du diff de paire en utilisant les epreuves finales.

    Args:
        previous_render_bytes: Image PNG du rendu du trimestre precedent.
        current_render_bytes: Image PNG du rendu du trimestre courant.
        diff_result: Resultat de comparaison contenant le ``technical_diff``.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Fonction injectable pour les appels OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer la consommation API.

    Returns:
        Dictionnaire de resultat enrichi avec les metadonnees de verification
        visuelle et le diff filtre.
    """
    technical_diff = dict(diff_result.get("technical_diff", {}) or {})
    items = _build_sanity_items_from_diff(technical_diff)
    if not items:
        return {
            **diff_result,
            **_default_meta(applied=False, rejected_count=0, render_status="ok"),
        }
    if not previous_render_bytes or not current_render_bytes:
        return {
            **diff_result,
            **_default_meta(
                applied=False,
                rejected_count=0,
                render_status="skipped_render_failed",
            ),
        }

    data = _call_visual_sanity_llm(
        previous_render_bytes,
        current_render_bytes,
        items=items,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    if data is None:
        return {
            **diff_result,
            **_default_meta(applied=False, rejected_count=0, render_status="ok"),
        }

    rejected = _rejected_item_ids(data)
    filtered_added = [
        entry
        for entry in list(technical_diff.get("indicators_added", []) or [])
        if _build_item_id("indicator_added", str(entry.get("value", "") or "").strip()) not in rejected
    ]
    filtered_removed = [
        entry
        for entry in list(technical_diff.get("indicators_removed", []) or [])
        if _build_item_id("indicator_removed", str(entry.get("value", "") or "").strip()) not in rejected
    ]
    filtered_renamed = [
        entry
        for entry in list(technical_diff.get("indicators_renamed", []) or [])
        if _build_item_id(
            "indicator_renamed",
            str(entry.get("previous", "") or "").strip(),
            str(entry.get("current", "") or "").strip(),
        )
        not in rejected
    ]
    filtered_fn_added = [
        entry
        for entry in list(technical_diff.get("footnotes_added", []) or [])
        if _build_item_id(
            "footnote_added",
            str(entry.get("id", "") or "").strip(),
            str(entry.get("text", "") or "").strip(),
        )
        not in rejected
    ]
    filtered_fn_removed = [
        entry
        for entry in list(technical_diff.get("footnotes_removed", []) or [])
        if _build_item_id(
            "footnote_removed",
            str(entry.get("id", "") or "").strip(),
            str(entry.get("text", "") or "").strip(),
        )
        not in rejected
    ]
    filtered_fn_renamed = [
        entry
        for entry in list(technical_diff.get("footnotes_renamed", []) or [])
        if _build_item_id(
            "footnote_modified",
            str(entry.get("previous_id", "") or "").strip(),
            str(entry.get("current_id", "") or "").strip(),
            str(entry.get("previous_text", "") or "").strip(),
            str(entry.get("current_text", "") or "").strip(),
        )
        not in rejected
    ]

    updated_diff = {
        **technical_diff,
        "indicators_added": filtered_added,
        "indicators_removed": filtered_removed,
        "indicators_renamed": filtered_renamed,
        "footnotes_added": filtered_fn_added,
        "footnotes_removed": filtered_fn_removed,
        "footnotes_renamed": filtered_fn_renamed,
    }
    updated_diff["table_level_change"] = recompute_table_level_change(updated_diff)

    rejected_count = max(
        0,
        len(items)
        - (
            len(filtered_added)
            + len(filtered_removed)
            + len(filtered_renamed)
            + len(filtered_fn_added)
            + len(filtered_fn_removed)
            + len(filtered_fn_renamed)
        ),
    )
    if rejected_count:
        logger.info(
            "Visual sanity check: rejected %d false-positive change(s).",
            rejected_count,
        )
    return {
        **diff_result,
        "technical_diff": updated_diff,
        **_default_meta(applied=True, rejected_count=rejected_count, render_status="ok"),
    }


def visual_sanity_check_table_event(
    previous_render_bytes: bytes | None,
    current_render_bytes: bytes | None,
    *,
    event_type: str,
    table_id: str,
    table_title: str,
    model: str,
    call_openai_json: Callable[..., dict[str, Any]],
    usage_recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Valide un evenement autonome d'ajout ou de suppression de tableau.

    Args:
        previous_render_bytes: Image PNG du rendu du trimestre precedent.
        current_render_bytes: Image PNG du rendu du trimestre courant.
        event_type: Type d'evenement (``table_added`` ou ``table_removed``).
        table_id: Identifiant unique du tableau.
        table_title: Titre du tableau.
        model: Identifiant du modele OpenAI a utiliser.
        call_openai_json: Fonction injectable pour les appels OpenAI.
        usage_recorder: Liste optionnelle pour enregistrer la consommation API.

    Returns:
        Dictionnaire contenant ``confirmed`` (bool) et les metadonnees de
        verification visuelle.
    """
    items = _build_table_event_item(
        event_type=event_type,
        table_id=table_id,
        table_title=table_title,
    )
    if not items:
        return {
            "confirmed": True,
            **_default_meta(applied=False, rejected_count=0, render_status="ok"),
        }
    if not previous_render_bytes or not current_render_bytes:
        return {
            "confirmed": True,
            **_default_meta(
                applied=False,
                rejected_count=0,
                render_status="skipped_render_failed",
            ),
        }

    data = _call_visual_sanity_llm(
        previous_render_bytes,
        current_render_bytes,
        items=items,
        model=model,
        call_openai_json=call_openai_json,
        usage_recorder=usage_recorder,
    )
    if data is None:
        return {
            "confirmed": True,
            **_default_meta(applied=False, rejected_count=0, render_status="ok"),
        }

    rejected = _rejected_item_ids(data)
    confirmed = items[0]["item_id"] not in rejected
    return {
        "confirmed": confirmed,
        **_default_meta(
            applied=True,
            rejected_count=0 if confirmed else 1,
            render_status="ok",
        ),
    }
