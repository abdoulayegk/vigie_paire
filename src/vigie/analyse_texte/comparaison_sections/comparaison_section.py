"""Orchestration de la comparaison semantique d une section."""

from __future__ import annotations

from typing import Any

from vigie.analyse_texte.chunk_alignment import ChunkAlignment, _align_chunks_hybrid
from vigie.analyse_texte.chunking import TextChunk
from vigie.analyse_texte.comparaison_sections.traitement_fragments_orphelins import (
    _annotate_section_rescue,
    _changes_from_orphan_chunks,
    _chunk_subsection_bodies,
    _display_heading_for_alignment,
    _heading_slug,
    _is_matched_alignment,
    _process_alignment_group,
)
from vigie.analyse_texte.markdown import _first_page_marker
from vigie.analyse_texte.models import TextAnalysisQualityError
from vigie.analyse_texte.subsection_matching import (
    OrphanSubsection,
    _pair_subsections,
    _parse_subsections,
    _resolve_orphan_subsections,
    _synthetic_subsection_rename_change,
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
    # Safety: le .md canonique porte des marqueurs ``[pdf.N]`` sur ses titres
    # pour la reconstruction de l'index page→texte. Ils DOIVENT avoir été
    # strippés avant d'arriver ici par ``_extract_section_text_from_markdown``.
    # Le motif vient de ``markdown.py``: le redéfinir ici l'avait fait diverger
    # du format réel, au point de se déclencher sur « [p. ex., ...] ».
    for period, text in (("T1", text_t1), ("T2", text_t2)):
        marker = _first_page_marker(text)
        if marker is not None:
            raise TextAnalysisQualityError(
                f"Fuite du marqueur de page {marker} vers le prompt GPT "
                f"(section {section_key}, période {period}) — strip manquant ?"
            )
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
