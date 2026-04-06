"""Pass 2 — Diff sémantique de sections textuelles via GPT-4o.

Pour chaque section, les blocs sont découpés en batches de ~10 blocs.
Chaque batch fait un appel GPT-4o séparé, puis les résultats sont fusionnés.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_SECTION_DISPLAY: dict[str, str] = {
    "gestion_capital": "Gestion du capital",
    "gestion_risques": "Gestion des risques",
    "gestion_reglementation": "Faits nouveaux en matière de réglementation",
}

_BATCH_SIZE: int = 10

# ---------------------------------------------------------------------------
# System prompt Pass 2
# ---------------------------------------------------------------------------

_DIFF_SYSTEM_PROMPT = """\
Tu es un analyste senior en réglementation bancaire canadienne (BSIF/OSFI). \
On te soumet le contenu textuel d'une même section de rapport bancaire \
pour deux trimestres consécutifs (PRÉCÉDENT et COURANT).

Ton rôle : identifier les paragraphes qui ont été ajoutés, supprimés ou \
modifiés de façon substantielle entre les deux versions.

RÈGLES STRICTES :
- IGNORER les changements de dates de clôture de rapport \
  (ex : "31 janvier 2025" → "30 avril 2025").
- IGNORER les variations de valeurs numériques propres à la banque (chiffres \
  d'affaires, rendements, nombres d'actions) qui ne sont pas des seuils \
  réglementaires.
- INCLURE les ajouts ou suppressions de paragraphes entiers.
- INCLURE les reformulations substantielles (changement de sens, d'engagement \
  ou de méthodologie).
- INCLURE tout changement touchant la réglementation, les méthodologies de \
  calcul, les risques identifiés, ou le capital réglementaire.
- CONSERVER les dates d'événements (ex : "le 3 février 2025") dans le texte.
- Pour les blocs "unchanged", inclure UNIQUEMENT les paires strictement identiques \
  ou quasi-identiques (cosmétique seulement).

NOTE : Tu reçois un sous-ensemble (batch) des paragraphes de la section. \
Certains paragraphes côté PRÉCÉDENT n'auront pas de correspondance côté COURANT \
(et vice-versa) car leur correspondance peut être dans un autre batch. \
Dans ce cas, marque-les comme "added" ou "removed". \
Ne force PAS un appariement entre des paragraphes qui ne traitent pas du même sujet.

RÉPONDRE UNIQUEMENT en JSON valide, sans markdown, selon ce schéma exact :
{
  "changes": [
    {
      "change_type": "added" | "removed" | "modified" | "unchanged",
      "text_t1": "<texte du paragraphe PRÉCÉDENT, chaîne vide si ajout>",
      "text_t2": "<texte du paragraphe COURANT, chaîne vide si suppression>",
      "change_summary": "<1 phrase décrivant le changement, vide si unchanged>"
    }
  ]
}

RÈGLES DE FORMAT :
- Un bloc "added"   : text_t1 = "" et text_t2 rempli.
- Un bloc "removed" : text_t1 rempli et text_t2 = "".
- Un bloc "modified": text_t1 et text_t2 tous deux remplis.
- Un bloc "unchanged": text_t1 == text_t2 (ou quasi-identique).
- Retourner le texte EXACT des paragraphes (pas de résumé), tel qu'il apparaît \
  dans le contenu fourni.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_blocks_by_section(blocks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        section = str(block.get("section", ""))
        if section:
            grouped.setdefault(section, []).append(block)
    return grouped


def _chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _format_blocks_for_prompt(blocks: list[dict[str, Any]], start_index: int = 1) -> str:
    parts = []
    for i, block in enumerate(blocks, start_index):
        text = str(block.get("text", "")).strip()
        if text:
            parts.append(f"[§{i}] {text}")
    return "\n\n".join(parts)


def _build_diff_user_prompt(
    section_key: str,
    blocks_t1: list[dict[str, Any]],
    blocks_t2: list[dict[str, Any]],
    quarter_t1: str,
    quarter_t2: str,
    batch_index: int = 1,
    total_batches: int = 1,
    start_t1: int = 1,
    start_t2: int = 1,
) -> str:
    section_title = _SECTION_DISPLAY.get(section_key, section_key)
    t1_text = _format_blocks_for_prompt(blocks_t1, start_t1)
    t2_text = _format_blocks_for_prompt(blocks_t2, start_t2)

    batch_info = f"  — Batch {batch_index}/{total_batches}" if total_batches > 1 else ""

    return (
        f"Section : {section_title} ({section_key}){batch_info}\n\n"
        f"=== {quarter_t1.upper()} (PRÉCÉDENT) ===\n"
        f"{t1_text or '(aucun bloc dans ce batch)'}\n\n"
        f"=== {quarter_t2.upper()} (COURANT) ===\n"
        f"{t2_text or '(aucun bloc dans ce batch)'}"
    )


def _validate_diff_response(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Valide et normalise la réponse JSON du diff.

    Retourne liste vide si data est None ou invalide.
    """
    if not data or not isinstance(data, dict):
        return []

    raw_changes = data.get("changes")
    if not isinstance(raw_changes, list):
        return []

    valid_types = {"added", "removed", "modified", "unchanged"}
    validated: list[dict[str, Any]] = []

    for item in raw_changes:
        if not isinstance(item, dict):
            continue
        change_type = str(item.get("change_type", "")).lower()
        if change_type not in valid_types:
            continue
        validated.append(
            {
                "change_type": change_type,
                "text_t1": str(item.get("text_t1", "") or "").strip(),
                "text_t2": str(item.get("text_t2", "") or "").strip(),
                "change_summary": str(item.get("change_summary", "") or "").strip(),
            }
        )

    return validated


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------


async def _diff_one_batch_async(
    client: Any,
    section_key: str,
    blocks_t1: list[dict[str, Any]],
    blocks_t2: list[dict[str, Any]],
    quarter_t1: str,
    quarter_t2: str,
    batch_index: int,
    total_batches: int,
    start_t1: int,
    start_t2: int,
    model: str = "gpt-4o",
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, Any]]:
    if not blocks_t1 and not blocks_t2:
        return []

    prompt = _build_diff_user_prompt(
        section_key,
        blocks_t1,
        blocks_t2,
        quarter_t1,
        quarter_t2,
        batch_index,
        total_batches,
        start_t1,
        start_t2,
    )

    sem = semaphore or asyncio.Semaphore(1)
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _DIFF_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw)
            changes = _validate_diff_response(data)
            logger.info(
                "text_diff: section=%s batch=%d/%d → %d changements (t1=%d blocs, t2=%d blocs)",
                section_key,
                batch_index,
                total_batches,
                len(changes),
                len(blocks_t1),
                len(blocks_t2),
            )
            if not changes and (blocks_t1 or blocks_t2):
                logger.warning(
                    "text_diff: section=%s batch=%d → 0 changements malgré %d+%d blocs. Réponse brute: %.200s",
                    section_key,
                    batch_index,
                    len(blocks_t1),
                    len(blocks_t2),
                    raw,
                )
            return changes
        except Exception as exc:
            logger.error(
                "text_diff: ÉCHEC section=%s batch=%d — %s",
                section_key,
                batch_index,
                exc,
            )
            return []


async def _run_all_section_diffs(
    extraction_t1: dict[str, Any],
    extraction_t2: dict[str, Any],
    sections_to_compare: list[str],
    model: str = "gpt-4o",
    max_concurrency: int = 6,
    batch_size: int = _BATCH_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    from openai import AsyncOpenAI

    from vigilance.utils.genai import get_openai_api_key

    api_key = get_openai_api_key()
    if not api_key:
        logger.warning("text_diff: OPENAI_API_KEY non définie — aucun diff possible.")
        return {}

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    blocks_t1_by_section = _group_blocks_by_section(extraction_t1.get("blocks", []))
    blocks_t2_by_section = _group_blocks_by_section(extraction_t2.get("blocks", []))

    quarter_t1 = str(extraction_t1.get("quarter", "T1"))
    quarter_t2 = str(extraction_t2.get("quarter", "T2"))

    # Build batch tasks: (section_key, batch_index, task)
    batch_tasks: list[tuple[str, int, asyncio.Task]] = []

    for section_key in sections_to_compare:
        t1_blocks = blocks_t1_by_section.get(section_key, [])
        t2_blocks = blocks_t2_by_section.get(section_key, [])

        if not t1_blocks and not t2_blocks:
            logger.info("text_diff: section %s absente des deux trimestres — skip.", section_key)
            continue

        batches_t1 = _chunk_list(t1_blocks, batch_size)
        batches_t2 = _chunk_list(t2_blocks, batch_size)
        total_batches = max(len(batches_t1), len(batches_t2))

        logger.info(
            "text_diff: section=%s → %d batches (t1=%d blocs, t2=%d blocs)",
            section_key,
            total_batches,
            len(t1_blocks),
            len(t2_blocks),
        )

        for i in range(total_batches):
            chunk_t1 = batches_t1[i] if i < len(batches_t1) else []
            chunk_t2 = batches_t2[i] if i < len(batches_t2) else []
            start_t1 = i * batch_size + 1
            start_t2 = i * batch_size + 1

            task = asyncio.create_task(
                _diff_one_batch_async(
                    client=client,
                    section_key=section_key,
                    blocks_t1=chunk_t1,
                    blocks_t2=chunk_t2,
                    quarter_t1=quarter_t1,
                    quarter_t2=quarter_t2,
                    batch_index=i + 1,
                    total_batches=total_batches,
                    start_t1=start_t1,
                    start_t2=start_t2,
                    model=model,
                    semaphore=semaphore,
                )
            )
            batch_tasks.append((section_key, i + 1, task))

    if batch_tasks:
        await asyncio.gather(*(t[2] for t in batch_tasks), return_exceptions=True)

    # Merge batch results by section
    results: dict[str, list[dict[str, Any]]] = {}
    for section_key, batch_idx, task in batch_tasks:
        try:
            batch_changes = task.result()
        except Exception as exc:
            logger.warning(
                "text_diff: récupération résultat section=%s batch=%d — %s",
                section_key,
                batch_idx,
                exc,
            )
            batch_changes = []
        results.setdefault(section_key, []).extend(batch_changes)

    total = sum(len(v) for v in results.values())
    logger.info(
        "text_diff: terminé — %d sections, %d batches, %d changements totaux",
        len(results),
        len(batch_tasks),
        total,
    )

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_section_diff(
    extraction_t1: dict[str, Any],
    extraction_t2: dict[str, Any],
    sections_to_compare: list[str] | None = None,
    model: str = "gpt-4o",
    max_concurrency: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    if sections_to_compare is None:
        all_blocks_t1 = {b.get("section") for b in extraction_t1.get("blocks", [])}
        all_blocks_t2 = {b.get("section") for b in extraction_t2.get("blocks", [])}
        sections_to_compare = sorted((all_blocks_t1 | all_blocks_t2) - {None, ""})  # type: ignore[operator]

    if not sections_to_compare:
        logger.info("run_section_diff: aucune section à comparer.")
        return {}

    return asyncio.run(
        _run_all_section_diffs(
            extraction_t1=extraction_t1,
            extraction_t2=extraction_t2,
            sections_to_compare=list(sections_to_compare),
            model=model,
            max_concurrency=max_concurrency,
        )
    )
