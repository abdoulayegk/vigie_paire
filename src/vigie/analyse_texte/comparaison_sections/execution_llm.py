"""Execution des lots et appels LLM de comparaison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from vigie.comparaison.analyst_change_presentation import bank_subject
from vigie.analyse_texte.chunk_alignment import _format_alignments_for_prompt
from vigie.analyse_texte.comparaison_sections.modeles import (
    ChunkComparisonLLMResponse,
    ComparisonBatch,
    _CHUNK_COMPARISON_VALIDATION_RETRY_MESSAGE,
    _MAX_COMPARISON_LLM_WORKERS,
)
from vigie.analyse_texte.comparaison_sections.resolution_alignements import (
    _attach_alignment_metadata,
    _materialize_semantic_alignment_decisions,
)
from vigie.analyse_texte.normalization import (
    _sanitize_explanation,
    _sanitize_semantic_text,
)
from vigie.analyse_texte.openai_client import (
    _call_structured_completion_with_correction,
)


def _compare_alignment_batch(
    *,
    client: Any,
    model: str,
    section_key: str,
    batch: ComparisonBatch,
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Compare un lot d'alignements via un appel LLM."""
    text_t1, text_t2 = _format_alignments_for_prompt(batch.alignments)
    try:
        changes = _compare_texts_single_call(
            client=client,
            model=model,
            section_key=section_key,
            heading_label=batch.heading_label,
            heading_slug=batch.heading_slug,
            text_t1=text_t1,
            text_t2=text_t2,
            idx_offset=batch.idx_offset,
            bank_code=bank_code,
        )
        scoped = _attach_alignment_metadata(changes, batch.alignments)
        return _materialize_semantic_alignment_decisions(scoped)
    except Exception as exc:
        raise RuntimeError(
            f"Batch comparison failed for {section_key}/{batch.heading_label}/{batch.batch_id}: {exc}"
        ) from exc


def _compare_alignment_batches(
    *,
    client: Any,
    model: str,
    section_key: str,
    batches: list[ComparisonBatch],
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Compare les lots d'alignements en parallèle puis restitue l'ordre source."""
    if not batches:
        return []
    max_workers = min(_MAX_COMPARISON_LLM_WORKERS, len(batches))
    results_by_index: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _compare_alignment_batch,
                client=client,
                model=model,
                section_key=section_key,
                batch=batch,
                bank_code=bank_code,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results_by_index[index] = future.result()

    merged: list[dict[str, Any]] = []
    for index in range(len(batches)):
        merged.extend(results_by_index.get(index, []))
    return merged


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
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Appel GPT unique pour comparer deux corps de texte.

    Extrait la logique de comparaison GPT de ``_compare_section_texts`` pour
    permettre son appel répété par sous-section.
    """
    subject = bank_subject(bank_code)
    try:
        raw = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu compares deux versions d'une section de rapport bancaire. "
                        "Identifie tous les changements observables paragraphe par paragraphe. "
                        "Ne masque pas les reformulations, les mises à jour de dates "
                        "ou les variations chiffrées : retourne-les comme changements, "
                        "le triage métier décidera ensuite de leur pertinence. "
                        "Rédige pour une analyste en vigie prudentielle : français soutenu, "
                        "phrases courtes, vocabulaire métier bancaire. "
                        f"La banque analysée est {subject}. "
                        f"Chaque change_summary doit commencer par « {subject} » suivi "
                        "d'un verbe d'action direct et du changement précis. "
                        "Cette règle s'applique aussi aux éléments inchangés, avec un verbe "
                        "tel que « maintient ». "
                        "Interdit dans change_summary et alignment_rationale : fragment, chunk, "
                        "T1, T2, termes anglais, et formulations meta du type "
                        "« Les deux fragments traitent… ». "
                        "Les expressions « rapport précédent » et « rapport courant » "
                        "peuvent seulement servir de contexte de comparaison; elles ne doivent "
                        "jamais être le sujet grammatical de change_summary. "
                        "N'inscris aucun trimestre, libellé de période ou commentaire sur le "
                        "processus de comparaison dans change_summary. "
                        "Lorsque le texte contient des blocs [c00], [c01], etc., "
                        "utilise ces bornes pour aligner les idées comparables, "
                        "mais ne recopie pas ces balises dans text_t1 ou text_t2. "
                        "Dans un bloc enumeration_item ou list_item, les marqueurs "
                        "i), ii), 1), 2) ou les puces sont structurels : une simple "
                        "renumérotation ou un déplacement ne constitue pas un changement "
                        "de divulgation si le contenu demeure le même. "
                        "Lorsque le texte contient des blocs [a00 | matched_strong], "
                        "[a00 | matched_grouped], [a00 | matched_weak], [a00 | ambiguous], [a00 | possible_added] "
                        "ou [a00 | possible_removed], ces alignements hybrides "
                        "(TF-IDF + embeddings) sont des indices locaux dans la même "
                        "sous-section, pas des verdicts. "
                        "Valide les cas faibles, ambigus, ajoutés ou supprimés possibles "
                        "avec les candidats fournis avant de décider added/removed/modified. "
                        "Pour chaque alignment, rends aussi une décision sémantique : "
                        "same_disclosure si les passages décrivent la même divulgation, "
                        "distinct_disclosures s'ils décrivent des événements, faits ou "
                        "obligations différents malgré un vocabulaire commun, moved_text "
                        "si la même information a seulement été déplacée, ou uncertain si "
                        "les éléments fournis ne permettent réellement pas de trancher."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Compare ces deux versions et retourne uniquement du JSON.\n"
                        'Format: {"changes":[{"alignment_id":"a00",'
                        '"diff_type":"unchanged|modified|added|removed",'
                        '"text_t1":"texte du paragraphe dans le rapport précédent, vide si added",'
                        '"text_t2":"texte du paragraphe dans le rapport courant, vide si removed",'
                        '"change_summary":"1 ou 2 phrases factuelles en français décrivant le '
                        f"changement; commencer exactement par {subject} suivi d'un verbe "
                        "d'action direct; nommer le BSIF, les montants ou dates exacts lorsque "
                        "pertinent; interdire fragment/chunk/T1/T2, les trimestres et les "
                        'formulations meta",'
                        '"alignment_decision":"same_disclosure|distinct_disclosures|moved_text|uncertain",'
                        '"alignment_confidence":"high|medium|low",'
                        '"alignment_rationale":"justification concise en français de la décision"}]}.\n'
                        "Chaque changement doit référencer exactement un alignment_id fourni "
                        "dans les blocs [a00 | ...]. Ne fusionne jamais plusieurs alignments "
                        "dans un seul changement. Ne recopie jamais les balises [a00] ou [c00] "
                        "dans text_t1/text_t2.\n"
                        "unchanged = texte substantiellement identique, sans changement observable.\n"
                        "modified = texte correspondant changé, y compris reformulation, "
                        "mise à jour de date, variation chiffrée, changement de nuance "
                        "ou évolution substantielle. modified exige TOUJOURS text_t1 et "
                        "text_t2 non vides.\n"
                        "added = idée nouvelle présente uniquement dans le rapport courant "
                        "(text_t1 vide, text_t2 non vide).\n"
                        "removed = idée présente dans le rapport précédent, absente du rapport "
                        "courant (text_t2 vide, text_t1 non vide).\n"
                        "Si une seule face est remplie, utilise added ou removed — jamais "
                        "modified. Si alignment_decision=distinct_disclosures et qu'un seul "
                        "côté a du texte, choisis aussi added ou removed.\n"
                        "La décision sémantique est obligatoire pour CHAQUE alignment, y compris "
                        "les alignments forts. Ne choisis uncertain qu'après examen des candidats. "
                        "Si deux passages décrivent deux événements distincts (par exemple deux "
                        "émissions différentes), choisis distinct_disclosures même si la structure "
                        "des phrases se ressemble.\n"
                        "Important : si le texte change mais que le sens semble identique, "
                        "retourne quand même diff_type='modified' avec un résumé indiquant "
                        "qu'il s'agit probablement d'une reformulation.\n"
                        f"Banque analysée : {subject}\n"
                        f"Section: {section_key}\n\n"
                        f"=== Rapport précédent ===\n{text_t1}\n\n"
                        f"=== Rapport courant ===\n{text_t2}\n"
                    ),
                },
            ],
            response_format=ChunkComparisonLLMResponse,
            max_retries=1,
            validation_retry_message=(
                f"{_CHUNK_COMPARISON_VALIDATION_RETRY_MESSAGE} "
                f'Chaque change_summary doit commencer exactement par "{subject} " suivi '
                "d'un verbe d'action direct; n'utilise jamais rapport courant, rapport "
                "précédent, T1, T2 ou un trimestre comme sujet."
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Section comparison failed for {section_key}/{heading_slug}: {exc}") from exc

    validated: list[dict[str, Any]] = []
    for local_idx, item in enumerate(raw.changes, start=1):
        global_idx = idx_offset + local_idx
        validated.append(
            {
                "change_id": f"{section_key}_{heading_slug}_change_{global_idx:03d}",
                "section_key": section_key,
                "subsection_heading": heading_label,
                "diff_type": item.diff_type,
                "alignment_id": item.alignment_id,
                "semantic_text_t1": _sanitize_semantic_text(item.text_t1),
                "semantic_text_t2": _sanitize_semantic_text(item.text_t2),
                "source_text_t1": item.text_t1,
                "source_text_t2": item.text_t2,
                "source_block_ids_t1": [],
                "source_block_ids_t2": [],
                "source_refs_t1": [],
                "source_refs_t2": [],
                "pages_t1": [],
                "pages_t2": [],
                "source_resolution_t1": "markdown",
                "source_resolution_t2": "markdown",
                "evidence_t1": {"pages": [], "snippet": item.text_t1[:400]},
                "evidence_t2": {"pages": [], "snippet": item.text_t2[:400]},
                "change_summary": _sanitize_explanation(item.change_summary),
                "alignment_decision": item.alignment_decision,
                "alignment_confidence": item.alignment_confidence,
                "alignment_rationale": _sanitize_explanation(item.alignment_rationale),
            }
        )
    return validated
