"""Composants modulaires du pipeline texte."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from vigilance.analyst_change_presentation import bank_subject
from vigilance.text_analysis.chunk_alignment import (
    _align_chunks_hybrid,
    _format_alignments_for_prompt,
    _sequence_similarity,
)
from vigilance.text_analysis.chunk_alignment import ChunkAlignment
from vigilance.text_analysis.chunking import TextChunk, _chunk_subsection_text
from vigilance.text_analysis.constants import _SECTION_LABELS
from vigilance.text_analysis.models import TextAnalysisQualityError
from vigilance.text_analysis.normalization import _sanitize_explanation, _sanitize_semantic_text
from vigilance.text_analysis.openai_client import _call_structured_completion_with_correction
from vigilance.text_analysis.subsection_matching import (
    OrphanSubsection,
    _normalize_heading,
    _pair_subsections,
    _parse_subsections,
    _resolve_orphan_subsections,
    _synthetic_subsection_rename_change,
)


_MAX_COMPARISON_LLM_WORKERS = 6
_EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD = 0.98
_COMPARISON_BATCH_SIZES = {
    "matched_strong": 5,
    "matched_grouped": 2,
    "matched_weak": 3,
    "ambiguous": 2,
    "possible_added": 3,
    "possible_removed": 3,
}
_CHUNK_COMPARISON_VALIDATION_RETRY_MESSAGE = (
    "Corrige la réponse et renvoie le batch COMPLET en respectant strictement le schéma. "
    "Chaque changement doit inclure alignment_id obligatoire, diff_type parmi "
    "unchanged|modified|added|removed, text_t1/text_t2 sans balises [a00]/[c00], "
    "alignment_decision parmi same_disclosure|distinct_disclosures|moved_text|uncertain, "
    "alignment_confidence parmi high|medium|low et alignment_rationale non vide, "
    "modified et unchanged doivent avoir text_t1 et text_t2 non vides, added doit "
    "avoir text_t2 non vide, removed doit avoir text_t1 non vide. Ne fusionne jamais "
    "plusieurs alignments dans un même changement."
)


class ChunkComparisonLLMChange(BaseModel):
    """Changement brut validé à la frontière LLM pour un seul alignment."""

    model_config = ConfigDict(extra="forbid")

    alignment_id: str
    diff_type: Literal["unchanged", "modified", "added", "removed"]
    text_t1: str
    text_t2: str
    change_summary: str
    # Kept optional for compatibility with existing cached responses.  The
    # prompt requires all three fields; missing values are handled
    # conservatively from the deterministic alignment type below.
    alignment_decision: Literal[
        "same_disclosure", "distinct_disclosures", "moved_text", "uncertain", ""
    ] = ""
    alignment_confidence: Literal["high", "medium", "low", ""] = ""
    alignment_rationale: str = ""

    @field_validator(
        "alignment_id",
        "text_t1",
        "text_t2",
        "change_summary",
        "alignment_rationale",
        mode="before",
    )
    @classmethod
    def _coerce_string(cls, value: Any) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_by_diff_type(self) -> "ChunkComparisonLLMChange":
        if not self.alignment_id:
            raise ValueError("alignment_id est obligatoire pour chaque changement chunké")
        if self.diff_type in {"unchanged", "modified"} and not (self.text_t1 and self.text_t2):
            raise ValueError("unchanged/modified exigent text_t1 et text_t2 non vides")
        if self.diff_type == "added" and not self.text_t2:
            raise ValueError("added exige text_t2 non vide")
        if self.diff_type == "removed" and not self.text_t1:
            raise ValueError("removed exige text_t1 non vide")
        return self


class ChunkComparisonLLMResponse(BaseModel):
    """Réponse structurée du LLM pour un batch d'alignements chunkés."""

    model_config = ConfigDict(extra="forbid")

    changes: list[ChunkComparisonLLMChange]


@dataclass(slots=True)
class ComparisonBatch:
    """Lot d'alignements envoyé dans un appel LLM de comparaison."""

    batch_id: str
    alignment_type: str
    alignments: list[ChunkAlignment]
    heading_label: str
    heading_slug: str
    idx_offset: int


def _prepare_subsection_alignments(
    *,
    section_key: str,
    subsection_heading_t1: str,
    subsection_heading_t2: str,
    body_t1: str,
    body_t2: str,
    client: Any | None = None,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-4o",
) -> list[ChunkAlignment]:
    """Prépare une paire de sous-sections en alignements hybrides locaux."""
    section_title = _SECTION_LABELS.get(section_key, section_key)
    chunks_t1 = _chunk_subsection_text(
        body_t1,
        subsection_heading=subsection_heading_t1,
        section_title=section_title,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )
    chunks_t2 = _chunk_subsection_text(
        body_t2,
        subsection_heading=subsection_heading_t2,
        section_title=section_title,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )
    # Après exclusion des tableaux, cellules et renvois non narratifs, une
    # sous-section peut légitimement ne plus avoir de contenu comparable. Ce
    # n'est pas une erreur de qualité : elle ne doit simplement pas produire
    # un ajout/retrait artificiel.
    if not chunks_t1 or not chunks_t2:
        return []
    return _align_chunks_hybrid(
        chunks_t1,
        chunks_t2,
        client=client,
        embedding_model=embedding_model,
    )


def _atomic_unit_metadata(
    chunk_t1: TextChunk | None,
    chunk_t2: TextChunk | None,
) -> dict[str, Any]:
    """Expose la filiation des unités sans modifier leur preuve source."""
    return {
        "unit_role_t1": chunk_t1.unit_role if chunk_t1 else None,
        "unit_role_t2": chunk_t2.unit_role if chunk_t2 else None,
        "parent_chunk_id_t1": chunk_t1.parent_chunk_id if chunk_t1 else None,
        "parent_chunk_id_t2": chunk_t2.parent_chunk_id if chunk_t2 else None,
        "atomic_marker_t1": chunk_t1.atomic_marker if chunk_t1 else None,
        "atomic_marker_t2": chunk_t2.atomic_marker if chunk_t2 else None,
        "parent_context_t1": chunk_t1.parent_context if chunk_t1 else "",
        "parent_context_t2": chunk_t2.parent_context if chunk_t2 else "",
    }


def _exact_diff_change_for_strong_alignment(
    *,
    alignment: ChunkAlignment,
    section_key: str,
    heading_label: str,
    heading_slug: str,
    change_index: int,
) -> dict[str, Any] | None:
    """Compare localement un alignement très solide sans arbitrage GPT supplémentaire."""
    if alignment.alignment_type != "matched_strong":
        return None
    if alignment.chunk_t1 is None or alignment.chunk_t2 is None:
        return None
    text_t1 = alignment.chunk_t1.text
    text_t2 = alignment.chunk_t2.text
    comparison_t1 = alignment.chunk_t1.comparison_text or text_t1
    comparison_t2 = alignment.chunk_t2.comparison_text or text_t2
    similarity = _sequence_similarity(comparison_t1, comparison_t2)
    if similarity < _EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD:
        return None
    normalized_t1 = re.sub(r"\s+", " ", comparison_t1).strip()
    normalized_t2 = re.sub(r"\s+", " ", comparison_t2).strip()
    if normalized_t1 == normalized_t2:
        diff_type = "unchanged"
        summary = "Passages alignés identiques après normalisation."
    else:
        diff_type = "modified"
        summary = "Passages fortement alignés avec une différence locale exacte."
    return {
        "change_id": f"{section_key}_{heading_slug}_change_{change_index:03d}",
        "section_key": section_key,
        "subsection_heading": heading_label,
        "diff_type": diff_type,
        "source_scope": "chunk",
        "alignment_id": alignment.alignment_id,
        "alignment_type": alignment.alignment_type,
        "chunk_id_t1": alignment.chunk_t1.chunk_id,
        "chunk_id_t2": alignment.chunk_t2.chunk_id,
        "semantic_text_t1": _sanitize_semantic_text(text_t1),
        "semantic_text_t2": _sanitize_semantic_text(text_t2),
        "source_text_t1": text_t1,
        "source_text_t2": text_t2,
        "source_block_ids_t1": [],
        "source_block_ids_t2": [],
        "source_refs_t1": [],
        "source_refs_t2": [],
        "pages_t1": [],
        "pages_t2": [],
        "source_resolution_t1": "markdown",
        "source_resolution_t2": "markdown",
        "evidence_t1": {"pages": [], "snippet": text_t1[:400]},
        "evidence_t2": {"pages": [], "snippet": text_t2[:400]},
        "change_summary": summary,
        "alignment_decision": "same_disclosure",
        "alignment_confidence": "high",
        "alignment_rationale": (
            f"Alignement hybride fort (tfidf={alignment.tfidf_score:.2f}, "
            f"embedding={alignment.embedding_score:.2f}, sequence={similarity:.2f}); "
            "diff exacte locale sans arbitrage GPT supplémentaire."
        ),
        "alignment_reason": alignment.reason,
        "tfidf_score": alignment.tfidf_score,
        "embedding_score": alignment.embedding_score,
        **_atomic_unit_metadata(alignment.chunk_t1, alignment.chunk_t2),
    }


def _batch_size_for_alignment_type(alignment_type: str) -> int:
    return _COMPARISON_BATCH_SIZES.get(alignment_type, 1)


def _build_comparison_batches(
    *,
    alignments: list[ChunkAlignment],
    heading_label: str,
    heading_slug: str,
) -> list[ComparisonBatch]:
    """Découpe les alignements en lots homogènes et ordonnés."""
    batches: list[ComparisonBatch] = []
    current_type = ""
    current: list[ChunkAlignment] = []

    def flush_current() -> None:
        nonlocal current, current_type
        if not current:
            return
        batch_index = len(batches)
        batches.append(
            ComparisonBatch(
                batch_id=f"b{batch_index:02d}",
                alignment_type=current_type,
                alignments=current,
                heading_label=heading_label,
                heading_slug=heading_slug,
                idx_offset=batch_index * 1000,
            )
        )
        current = []
        current_type = ""

    for alignment in alignments:
        alignment_type = alignment.alignment_type
        max_size = _batch_size_for_alignment_type(alignment_type)
        if current and (alignment_type != current_type or len(current) >= max_size):
            flush_current()
        current_type = alignment_type
        current.append(alignment)
    flush_current()
    return batches


def _split_exact_diff_alignments(
    alignments: list[ChunkAlignment],
) -> tuple[list[ChunkAlignment], list[ChunkAlignment]]:
    """Sépare les alignements hybrides assez solides pour un diff exact local."""
    exact: list[ChunkAlignment] = []
    remaining: list[ChunkAlignment] = []
    for alignment in alignments:
        # matched_strong may come from TF-IDF-only fallback (embedding_score=0)
        # or from embeddings; both deserve the local exact-diff fast path when
        # the sequences are near-identical.
        strong_signal = (
            alignment.embedding_score >= 0.85 or alignment.tfidf_score >= 0.85
        )
        if (
            alignment.alignment_type == "matched_strong"
            and strong_signal
            and alignment.chunk_t1 is not None
            and alignment.chunk_t2 is not None
            and _sequence_similarity(alignment.chunk_t1.text, alignment.chunk_t2.text)
            >= _EXACT_DIFF_STRONG_SEQUENCE_THRESHOLD
        ):
            exact.append(alignment)
        else:
            remaining.append(alignment)
    return exact, remaining


def _reindex_changes(
    changes: list[dict[str, Any]],
    *,
    section_key: str,
    heading_slug: str,
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Réindexe les changements fusionnés après appels LLM parallèles."""
    reindexed: list[dict[str, Any]] = []
    for local_idx, change in enumerate(changes, start=1):
        updated = dict(change)
        updated["change_id"] = f"{section_key}_{heading_slug}_change_{idx_offset + local_idx:03d}"
        reindexed.append(updated)
    return reindexed


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
                        "les éléments fournis ne permettent réellement pas de trancher. "
                        "Cette étape établit uniquement les faits et la relation documentaire : "
                        "n'évalue ni la pertinence AMF, ni la matérialité métier, ni le niveau "
                        "MAJEUR, MODÉRÉ ou MINEUR. same_disclosure signifie seulement que les "
                        "passages portent sur la même divulgation; ce n'est jamais une preuve "
                        "d'équivalence métier. Dans change_summary et alignment_rationale, "
                        "n'écris pas que le changement est sans effet, sans changement de fond "
                        "ou qu'il n'altère pas le sens. Décris plutôt les termes, responsabilités, "
                        "périmètres, méthodes, contrôles, obligations ou statuts effectivement "
                        "ajoutés, retirés ou remplacés."
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
                        "formulations meta\","
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
                        "ou évolution substantielle.\n"
                        "added = idée nouvelle présente uniquement dans le rapport courant.\n"
                        "removed = idée présente dans le rapport précédent, absente du rapport courant.\n"
                        "La décision sémantique est obligatoire pour CHAQUE alignment, y compris "
                        "les alignments forts. Ne choisis uncertain qu'après examen des candidats. "
                        "Si deux passages décrivent deux événements distincts (par exemple deux "
                        "émissions différentes), choisis distinct_disclosures même si la structure "
                        "des phrases se ressemble.\n"
                        "Important : si le texte change mais paraît porter sur le même référent, "
                        "retourne quand même diff_type='modified' et décris factuellement les "
                        "mots ou éléments remplacés. Réserve toute conclusion d'équivalence "
                        "métier ou de matérialité à l'étape de triage suivante.\n"
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


def _normalize_for_alignment_contains(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _coerce_text_to_chunk(text: str, chunk: TextChunk | None) -> str | None:
    """Ramène le texte LLM au périmètre exact d'un chunk, sinon invalide."""
    value = str(text or "").strip()
    if chunk is None:
        return "" if not value else None
    if not value:
        return ""
    chunk_text = chunk.text.strip()
    if value in chunk_text:
        return value

    normalized_value = _normalize_for_alignment_contains(value)
    normalized_chunk = _normalize_for_alignment_contains(chunk_text)
    if not normalized_value:
        return ""
    if normalized_value == normalized_chunk:
        return chunk_text
    if normalized_chunk and normalized_chunk in normalized_value:
        return chunk_text
    if normalized_value in normalized_chunk:
        return chunk_text
    return None


_SEMANTIC_ALIGNMENT_DECISIONS = frozenset(
    {"same_disclosure", "distinct_disclosures", "moved_text", "uncertain"}
)


def _resolved_alignment_decision(change: dict[str, Any], alignment: ChunkAlignment) -> str:
    """Normalizes the first GPT call's semantic decision conservatively."""
    decision = str(change.get("alignment_decision") or "").strip().lower()
    if decision in _SEMANTIC_ALIGNMENT_DECISIONS:
        return decision
    # Cached / legacy responses do not have the new field.  Preserve the
    # previous conservative handling only for genuinely ambiguous matches.
    if alignment.alignment_type == "ambiguous":
        return "uncertain"
    return "same_disclosure"


def _resolved_alignment_confidence(change: dict[str, Any], decision: str) -> str:
    confidence = str(change.get("alignment_confidence") or "").strip().lower()
    if confidence in {"high", "medium", "low"}:
        return confidence
    return "low" if decision == "uncertain" else "medium"


def _attach_alignment_metadata(
    changes: list[dict[str, Any]],
    alignments: list[ChunkAlignment],
) -> list[dict[str, Any]]:
    """Valide les changements LLM et les borne à leur alignment/chunk source."""
    alignment_by_id = {alignment.alignment_id: alignment for alignment in alignments}
    scoped: list[dict[str, Any]] = []
    for change in changes:
        alignment_id = str(change.get("alignment_id") or "").strip()
        alignment = alignment_by_id.get(alignment_id)
        if alignment is None:
            continue

        text_t1 = _coerce_text_to_chunk(str(change.get("source_text_t1") or ""), alignment.chunk_t1)
        text_t2 = _coerce_text_to_chunk(str(change.get("source_text_t2") or ""), alignment.chunk_t2)
        if text_t1 is None or text_t2 is None:
            continue

        diff_type = str(change.get("diff_type") or "").lower()
        if diff_type in {"unchanged", "modified"} and not (text_t1 and text_t2):
            continue
        if diff_type == "added" and not text_t2:
            continue
        if diff_type == "removed" and not text_t1:
            continue

        scoped_change = dict(change)
        alignment_decision = _resolved_alignment_decision(scoped_change, alignment)
        scoped_change.update(
            {
                "source_scope": "chunk",
                "alignment_id": alignment.alignment_id,
                "alignment_type": alignment.alignment_type,
                "chunk_id_t1": alignment.chunk_t1.chunk_id if alignment.chunk_t1 else None,
                "chunk_id_t2": alignment.chunk_t2.chunk_id if alignment.chunk_t2 else None,
                "source_text_t1": text_t1,
                "source_text_t2": text_t2,
                "semantic_text_t1": _sanitize_semantic_text(text_t1),
                "semantic_text_t2": _sanitize_semantic_text(text_t2),
                "evidence_t1": {"pages": [], "snippet": text_t1[:400]},
                "evidence_t2": {"pages": [], "snippet": text_t2[:400]},
                "alignment_decision": alignment_decision,
                "alignment_confidence": _resolved_alignment_confidence(
                    scoped_change, alignment_decision
                ),
                "alignment_rationale": str(scoped_change.get("alignment_rationale") or "").strip(),
                "alignment_reason": alignment.reason,
                "tfidf_score": alignment.tfidf_score,
                "embedding_score": alignment.embedding_score,
                **_atomic_unit_metadata(alignment.chunk_t1, alignment.chunk_t2),
            }
        )
        scoped.append(scoped_change)
    return scoped


def _materialize_semantic_alignment_decisions(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turns a GPT-confirmed distinct pairing into separate source changes.

    A lexical matcher can put two distinct events in one provisional pair.  If
    the comparison model explicitly confirms that they are distinct, retaining
    a single ``modified`` card would still imply a false correspondence.  Two
    one-sided records preserve the original evidence for the AMF triage.
    """
    materialized: list[dict[str, Any]] = []
    for change in changes:
        decision = str(change.get("alignment_decision") or "").strip().lower()
        text_t1 = str(change.get("source_text_t1") or "").strip()
        text_t2 = str(change.get("source_text_t2") or "").strip()
        if decision != "distinct_disclosures" or not text_t1 or not text_t2:
            materialized.append(change)
            continue

        rationale = str(change.get("alignment_rationale") or "").strip()
        base_summary = str(change.get("change_summary") or "").strip()
        removed = dict(change)
        removed.update(
            {
                "diff_type": "removed",
                "alignment_id": f"{change.get('alignment_id')}:removed",
                "alignment_type": "semantic_distinct",
                "semantic_alignment_group_id": str(change.get("alignment_id") or ""),
                "source_text_t2": "",
                "semantic_text_t2": "",
                "evidence_t2": {"pages": [], "snippet": ""},
                "change_summary": (
                    f"Divulgation distincte retirée. {rationale or base_summary}".strip()
                ),
            }
        )
        added = dict(change)
        added.update(
            {
                "diff_type": "added",
                "alignment_id": f"{change.get('alignment_id')}:added",
                "alignment_type": "semantic_distinct",
                "semantic_alignment_group_id": str(change.get("alignment_id") or ""),
                "source_text_t1": "",
                "semantic_text_t1": "",
                "evidence_t1": {"pages": [], "snippet": ""},
                "change_summary": (
                    f"Divulgation distincte ajoutée. {rationale or base_summary}".strip()
                ),
            }
        )
        materialized.extend([removed, added])
    return materialized


def _deduplicate_alignment_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conserve une carte par alignment, même si le LLM liste plusieurs détails.

    Les détails restent dans le résumé concaténé ; le texte source demeure le
    chunk unique auquel ils se rapportent. Cela évite de répéter la même paire
    de paragraphes dans plusieurs cartes Dash.
    """
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    ordered_keys: list[tuple[str, str, str, str, str]] = []
    for change in changes:
        key = (
            str(change.get("section_key") or ""),
            str(change.get("subsection_heading") or ""),
            str(change.get("alignment_id") or ""),
            str(change.get("chunk_id_t1") or ""),
            str(change.get("chunk_id_t2") or ""),
        )
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(change)

    deduplicated: list[dict[str, Any]] = []
    for key in ordered_keys:
        group = grouped[key]
        if len(group) == 1 or not key[2]:
            deduplicated.extend(group)
            continue

        representative = next(
            (
                change
                for change in group
                if str(change.get("source_text_t1") or "").strip()
                and str(change.get("source_text_t2") or "").strip()
            ),
            group[0],
        )
        merged = dict(representative)
        summaries: list[str] = []
        for change in group:
            summary = str(change.get("change_summary") or "").strip()
            if summary and summary not in summaries:
                summaries.append(summary)
        merged["change_summary"] = " ; ".join(summaries)
        if str(merged.get("source_text_t1") or "").strip() and str(merged.get("source_text_t2") or "").strip():
            merged["diff_type"] = "modified"
        deduplicated.append(merged)
    return deduplicated


def _heading_slug(heading: str) -> str:
    return re.sub(r"[^\w]+", "_", _normalize_heading(heading))[:40].strip("_") or "unknown"


def _display_heading_for_alignment(alignment: ChunkAlignment) -> str:
    """Heading affiché : H1 → H2 quand le match croise deux sous-sections."""
    heading_t1 = str(alignment.chunk_t1.subsection_heading if alignment.chunk_t1 else "").strip()
    heading_t2 = str(alignment.chunk_t2.subsection_heading if alignment.chunk_t2 else "").strip()
    if heading_t1 and heading_t2 and heading_t1 != heading_t2:
        return f"{heading_t1} → {heading_t2}"
    return heading_t1 or heading_t2 or "unknown"


def _annotate_section_rescue(alignment: ChunkAlignment) -> ChunkAlignment:
    """Marque un match Phase B comme récupération cross-sous-section."""
    alignment.reason = "section_rescue"
    return alignment


def _is_matched_alignment(alignment: ChunkAlignment) -> bool:
    return alignment.alignment_type not in {"possible_added", "possible_removed"}


def _chunk_subsection_bodies(
    *,
    section_key: str,
    heading: str,
    body: str,
    client: Any,
    embedding_model: str,
    semantic_model: str,
) -> list[TextChunk]:
    if not str(body or "").strip():
        return []
    section_title = _SECTION_LABELS.get(section_key, section_key)
    return _chunk_subsection_text(
        body,
        subsection_heading=heading,
        section_title=section_title,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )


def _process_alignment_group(
    *,
    client: Any,
    model: str,
    section_key: str,
    heading_label: str,
    heading_slug: str,
    alignments: list[ChunkAlignment],
    idx_offset: int,
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Exact-diff + LLM pour un groupe d'alignements déjà résolus."""
    if not alignments:
        return []
    exact_alignments, llm_alignments = _split_exact_diff_alignments(alignments)
    exact_changes: list[dict[str, Any]] = []
    for index, alignment in enumerate(exact_alignments, start=1):
        change = _exact_diff_change_for_strong_alignment(
            alignment=alignment,
            section_key=section_key,
            heading_label=heading_label,
            heading_slug=heading_slug,
            change_index=index,
        )
        if change is None:
            llm_alignments.append(alignment)
            continue
        exact_changes.append(change)

    batches = _build_comparison_batches(
        alignments=llm_alignments,
        heading_label=heading_label,
        heading_slug=heading_slug,
    )
    llm_changes = _compare_alignment_batches(
        client=client,
        model=model,
        section_key=section_key,
        batches=batches,
        bank_code=bank_code,
    )
    group_changes = _deduplicate_alignment_changes([*exact_changes, *llm_changes])
    return _reindex_changes(
        group_changes,
        section_key=section_key,
        heading_slug=heading_slug,
        idx_offset=idx_offset,
    )


def _changes_from_orphan_chunks(
    *,
    section_key: str,
    diff_type: str,
    chunks: list[TextChunk],
    idx_offset: int,
) -> list[dict[str, Any]]:
    """Ajouts/retraits déterministes pour les orphelins restants après Phase B."""
    changes: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        heading = chunk.subsection_heading or "unknown"
        slug = _heading_slug(heading)
        text_t1 = chunk.text if diff_type == "removed" else ""
        text_t2 = chunk.text if diff_type == "added" else ""
        change_index = idx_offset + chunk_index
        changes.append(
            {
                "change_id": f"{section_key}_{slug}_change_{change_index:03d}",
                "section_key": section_key,
                "subsection_heading": heading,
                "diff_type": diff_type,
                "source_scope": "chunk",
                "alignment_id": f"unmatched_{chunk.chunk_id}",
                "alignment_type": f"unmatched_{diff_type}",
                "alignment_reason": "section_orphan_after_rescue",
                "chunk_id_t1": chunk.chunk_id if diff_type == "removed" else None,
                "chunk_id_t2": chunk.chunk_id if diff_type == "added" else None,
                "semantic_text_t1": _sanitize_semantic_text(text_t1),
                "semantic_text_t2": _sanitize_semantic_text(text_t2),
                "source_text_t1": text_t1,
                "source_text_t2": text_t2,
                "source_block_ids_t1": [],
                "source_block_ids_t2": [],
                "source_refs_t1": [],
                "source_refs_t2": [],
                "pages_t1": [],
                "pages_t2": [],
                "source_resolution_t1": "markdown",
                "source_resolution_t2": "markdown",
                "evidence_t1": {"pages": [], "snippet": text_t1[:400]},
                "evidence_t2": {"pages": [], "snippet": text_t2[:400]},
                "change_summary": (
                    f"Passage de sous-section "
                    f"{'ajouté' if diff_type == 'added' else 'supprimé'}: {heading}"
                ),
                **_atomic_unit_metadata(
                    chunk if diff_type == "removed" else None,
                    chunk if diff_type == "added" else None,
                ),
            }
        )
    return changes


def _unmatched_subsection_chunk_changes(
    *,
    section_key: str,
    diff_type: str,
    heading: str,
    body: str,
    idx_offset: int,
    client: Any,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Produit des ajouts/retraits par chunk pour une sous-section sans paire."""
    chunks = _chunk_subsection_bodies(
        section_key=section_key,
        heading=heading,
        body=body,
        client=client,
        embedding_model=embedding_model,
        semantic_model=semantic_model,
    )
    return _changes_from_orphan_chunks(
        section_key=section_key,
        diff_type=diff_type,
        chunks=chunks,
        idx_offset=idx_offset,
    )


def _compare_section_texts(
    *,
    client: Any,
    model: str,
    section_key: str,
    text_t1: str,
    text_t2: str,
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Compare deux sections markdown T1/T2 avec alignement cascade.

    Phase A aligne sous-section par sous-section. Les orphelins sont ensuite
    ré-alignés une fois sur toute la section (Phase B) avant tout add/remove
    définitif, ce qui récupère les passages déplacés entre rubriques.
    """
    # Safety: the .md canonique contient des marqueurs ``[p.N]`` pour la
    # reconstruction de l'index page→texte. Ils DOIVENT avoir été strippés
    # avant d'arriver ici par ``_extract_section_text_from_markdown``.
    if "[p." in text_t1 or "[p." in text_t2:
        raise TextAnalysisQualityError("Fuite de marqueurs de page vers le prompt GPT — strip manquant ?")
    if not text_t1.strip() and not text_t2.strip():
        return []

    subs_t1 = _parse_subsections(text_t1)
    subs_t2 = _parse_subsections(text_t2)

    has_subsections_t1 = any(heading != "__intro__" for heading, _body in subs_t1)
    has_subsections_t2 = any(heading != "__intro__" for heading, _body in subs_t2)

    if not has_subsections_t1 and not has_subsections_t2:
        raise TextAnalysisQualityError(
            f"Section non vide sans sous-sections ###: {section_key}"
        )

    pairs = _pair_subsections(subs_t1, subs_t2)

    # Heading-level orphan rename resolution (unchanged).
    orphans_t1 = [
        OrphanSubsection(heading=h1, body=body1)
        for h1, body1, h2, _body2 in pairs
        if h2 is None and h1 is not None
    ]
    orphans_t2 = [
        OrphanSubsection(heading=h2, body=body2)
        for h1, _body1, h2, body2 in pairs
        if h1 is None and h2 is not None
    ]
    rename_matches = _resolve_orphan_subsections(
        client=client,
        model=model,
        section_key=section_key,
        orphans_t1=orphans_t1,
        orphans_t2=orphans_t2,
    )
    rename_t1_to_t2: dict[str, str] = {m["heading_t1"]: m["heading_t2"] for m in rename_matches}
    renamed_as_t2: set[str] = set(rename_t1_to_t2.values())

    body_by_t2_heading: dict[str, str] = {h2: b2 for _, _, h2, b2 in pairs if h2 is not None}
    resolved_pairs: list[tuple[str | None, str, str | None, str]] = []
    for h1, body1, h2, body2 in pairs:
        if h2 is None and h1 is not None and h1 in rename_t1_to_t2:
            matched_h2 = rename_t1_to_t2[h1]
            matched_body2 = body_by_t2_heading.get(matched_h2, "")
            resolved_pairs.append((h1, body1, matched_h2, matched_body2))
        elif h1 is None and h2 is not None and h2 in renamed_as_t2:
            continue
        else:
            resolved_pairs.append((h1, body1, h2, body2))
    pairs = resolved_pairs

    all_changes: list[dict[str, Any]] = []
    global_idx = 1
    renamed_pairs: set[tuple[str, str]] = {(m["heading_t1"], m["heading_t2"]) for m in rename_matches}

    # heading_label -> matched alignments from Phase A (same subsection).
    matched_by_heading: dict[str, list[ChunkAlignment]] = {}
    orphan_chunks_t1: list[TextChunk] = []
    orphan_chunks_t2: list[TextChunk] = []
    embedding_model = "text-embedding-3-small"

    for h1, body1, h2, body2 in pairs:
        heading_label = h1 or h2 or "unknown"

        is_renamed_pair = h1 is not None and h2 is not None and (h1, h2) in renamed_pairs
        if is_renamed_pair:
            all_changes.append(
                _synthetic_subsection_rename_change(
                    section_key=section_key,
                    heading_t1=h1,
                    heading_t2=h2,
                    idx=global_idx,
                )
            )
            global_idx += 1
            heading_label = f"{h1} → {h2}"

        if h2 is None:
            assert h1 is not None
            orphan_chunks_t1.extend(
                _chunk_subsection_bodies(
                    section_key=section_key,
                    heading=h1,
                    body=body1,
                    client=client,
                    embedding_model=embedding_model,
                    semantic_model=model,
                )
            )
            continue

        if h1 is None:
            assert h2 is not None
            orphan_chunks_t2.extend(
                _chunk_subsection_bodies(
                    section_key=section_key,
                    heading=h2,
                    body=body2,
                    client=client,
                    embedding_model=embedding_model,
                    semantic_model=model,
                )
            )
            continue

        if not body1.strip() and not body2.strip():
            continue
        if not body1.strip():
            orphan_chunks_t2.extend(
                _chunk_subsection_bodies(
                    section_key=section_key,
                    heading=h2,
                    body=body2,
                    client=client,
                    embedding_model=embedding_model,
                    semantic_model=model,
                )
            )
            continue
        if not body2.strip():
            orphan_chunks_t1.extend(
                _chunk_subsection_bodies(
                    section_key=section_key,
                    heading=h1,
                    body=body1,
                    client=client,
                    embedding_model=embedding_model,
                    semantic_model=model,
                )
            )
            continue

        chunks_t1 = _chunk_subsection_bodies(
            section_key=section_key,
            heading=h1,
            body=body1,
            client=client,
            embedding_model=embedding_model,
            semantic_model=model,
        )
        chunks_t2 = _chunk_subsection_bodies(
            section_key=section_key,
            heading=h2,
            body=body2,
            client=client,
            embedding_model=embedding_model,
            semantic_model=model,
        )
        if not chunks_t1 and not chunks_t2:
            continue
        if not chunks_t1:
            orphan_chunks_t2.extend(chunks_t2)
            continue
        if not chunks_t2:
            orphan_chunks_t1.extend(chunks_t1)
            continue

        # Phase A — local hybrid alignment inside the paired subsection.
        alignments = _align_chunks_hybrid(
            chunks_t1,
            chunks_t2,
            client=client,
            embedding_model=embedding_model,
        )
        matched = [alignment for alignment in alignments if _is_matched_alignment(alignment)]
        for alignment in alignments:
            if alignment.alignment_type == "possible_removed" and alignment.chunk_t1 is not None:
                orphan_chunks_t1.append(alignment.chunk_t1)
            elif alignment.alignment_type == "possible_added" and alignment.chunk_t2 is not None:
                orphan_chunks_t2.append(alignment.chunk_t2)
        if matched:
            matched_by_heading.setdefault(heading_label, []).extend(matched)

    # Phase B — section-wide rescue among remaining orphans only.
    rescued_by_heading: dict[str, list[ChunkAlignment]] = {}
    if orphan_chunks_t1 and orphan_chunks_t2:
        rescue_alignments = _align_chunks_hybrid(
            orphan_chunks_t1,
            orphan_chunks_t2,
            client=client,
            embedding_model=embedding_model,
        )
        remaining_t1: list[TextChunk] = []
        remaining_t2: list[TextChunk] = []
        for alignment in rescue_alignments:
            if not _is_matched_alignment(alignment):
                if alignment.alignment_type == "possible_removed" and alignment.chunk_t1 is not None:
                    remaining_t1.append(alignment.chunk_t1)
                elif alignment.alignment_type == "possible_added" and alignment.chunk_t2 is not None:
                    remaining_t2.append(alignment.chunk_t2)
                continue
            rescued = _annotate_section_rescue(alignment)
            rescue_heading = _display_heading_for_alignment(rescued)
            rescued_by_heading.setdefault(rescue_heading, []).append(rescued)
        orphan_chunks_t1 = remaining_t1
        orphan_chunks_t2 = remaining_t2

    # Emit matched groups (Phase A + Phase B) through exact-diff / LLM.
    for heading_label, alignments in [*matched_by_heading.items(), *rescued_by_heading.items()]:
        heading_slug = _heading_slug(heading_label.split(" → ", 1)[0] if " → " in heading_label else heading_label)
        group_changes = _process_alignment_group(
            client=client,
            model=model,
            section_key=section_key,
            heading_label=heading_label,
            heading_slug=heading_slug,
            alignments=alignments,
            idx_offset=global_idx - 1,
            bank_code=bank_code,
        )
        all_changes.extend(group_changes)
        global_idx += len(group_changes)

    # True add/remove only after section rescue failed to pair them.
    removed_changes = _changes_from_orphan_chunks(
        section_key=section_key,
        diff_type="removed",
        chunks=orphan_chunks_t1,
        idx_offset=global_idx - 1,
    )
    all_changes.extend(removed_changes)
    global_idx += len(removed_changes)
    added_changes = _changes_from_orphan_chunks(
        section_key=section_key,
        diff_type="added",
        chunks=orphan_chunks_t2,
        idx_offset=global_idx - 1,
    )
    all_changes.extend(added_changes)

    return all_changes
