from __future__ import annotations

from vigie.analyse_texte.chunk_alignment import (
    _align_chunks_tfidf,
    _embedding_similarity_matrix,
)
from vigie.analyse_texte.chunking import _chunk_subsection_text
from vigie.analyse_texte.comparaison_sections import (
    _attach_alignment_metadata,
    _deduplicate_alignment_changes,
    _exact_diff_change_for_strong_alignment,
)


def _items(chunks):
    return [chunk for chunk in chunks if chunk.unit_role == "item"]


def test_inline_roman_enumeration_creates_parent_and_atomic_children() -> None:
    text = (
        "Les facteurs suivants peuvent affecter les titres de la Banque : "
        "i) les résultats financiers et opérationnels; "
        "ii) la capacité à respecter la résolution globale; "
        "iii) l'incidence de la résolution sur les activités; et "
        "iv) les changements réglementaires applicables."
    )

    chunks = _chunk_subsection_text(
        text,
        subsection_heading="Valeur des titres",
        section_title="Gestion des risques",
    )

    assert [chunk.kind for chunk in chunks] == [
        "enumeration_context",
        "enumeration_item",
        "enumeration_item",
        "enumeration_item",
        "enumeration_item",
    ]
    assert [chunk.atomic_marker for chunk in _items(chunks)] == [
        "i)",
        "ii)",
        "iii)",
        "iv)",
    ]
    assert chunks[0].parent_chunk_id is None
    assert {
        chunk.parent_chunk_id for chunk in _items(chunks)
    } == {chunks[0].chunk_id}
    assert chunks[0].text.endswith(":")
    assert _items(chunks)[0].text.startswith("i)")
    assert _items(chunks)[0].comparison_text.startswith("les résultats")
    assert all(
        not chunk.comparison_text.startswith(str(chunk.atomic_marker))
        for chunk in _items(chunks)
    )


def test_nonsequential_markers_do_not_trigger_atomic_split() -> None:
    text = (
        "Le rapport renvoie à i) une première définition et "
        "iii) une référence non consécutive."
    )

    chunks = _chunk_subsection_text(text)

    assert len(chunks) == 1
    assert chunks[0].kind == "paragraph"
    assert chunks[0].unit_role == "standalone"


def test_numeric_and_alphabetic_sequences_are_supported() -> None:
    numeric = _chunk_subsection_text(
        "Les mesures comprennent : 1) renforcer les contrôles; 2) revoir la gouvernance."
    )
    alphabetic = _chunk_subsection_text(
        "Les mesures comprennent : a) renforcer les contrôles; b) revoir la gouvernance."
    )

    assert [chunk.atomic_marker for chunk in _items(numeric)] == ["1)", "2)"]
    assert [chunk.atomic_marker for chunk in _items(alphabetic)] == ["a)", "b)"]


def test_td_style_roman_enumeration_supports_fifteen_factors() -> None:
    markers = [
        "i)",
        "ii)",
        "iii)",
        "iv)",
        "v)",
        "vi)",
        "vii)",
        "viii)",
        "ix)",
        "x)",
        "xi)",
        "xii)",
        "xiii)",
        "xiv)",
        "xv)",
    ]
    factors = "; ".join(
        f"{marker} le facteur de risque bancaire numéro {index}"
        for index, marker in enumerate(markers, start=1)
    )

    chunks = _chunk_subsection_text(
        f"Les facteurs suivants peuvent affecter les titres de la Banque : {factors}."
    )

    assert [chunk.atomic_marker for chunk in _items(chunks)] == markers


def test_bmo_appetite_markdown_list_creates_one_parent_and_five_units() -> None:
    text = (
        "### Cadre d'appétit pour le risque [pdf.74]\n\n"
        "Nous jugeons que la responsabilité de la gestion des risques incombe à chacun "
        "de nos employés et notre approche en gestion des risques s'articule autour de "
        "cinq objectifs clés, qui orientent toutes nos activités en ce domaine et "
        "s'inscrivent dans notre énoncé d'appétit pour le risque:\n\n"
        "- Comprendre et gérer en n'assumant que les risques qui sont transparents et "
        "clairement définis.\n\n"
        "- Préserver la réputation de BMO en adhérant à des principes d'honnêteté, "
        "d'intégrité et de respect, ainsi qu'à des normes éthiques élevées.\n\n"
        "- Diversifier et restreindre les risques extrêmes en visant une diversification "
        "de nos activités.\n\n"
        "- Maintenir une situation enviable en matière de capital et de liquidité.\n\n"
        "- Optimiser le rapport risque-rendement en gérant les expositions ajustées "
        "en fonction des risques."
    )

    chunks = _chunk_subsection_text(
        text,
        subsection_heading="Cadre d'appétit pour le risque",
        section_title="Gestion des risques",
    )

    assert len(chunks) == 6
    assert chunks[0].kind == "list_context"
    assert chunks[0].unit_role == "context"
    assert [chunk.kind for chunk in chunks[1:]] == ["list_item"] * 5
    assert {chunk.parent_chunk_id for chunk in chunks[1:]} == {
        chunks[0].chunk_id
    }
    assert all(chunk.atomic_marker == "-" for chunk in chunks[1:])
    assert all(not chunk.comparison_text.startswith("- ") for chunk in chunks[1:])


def test_marker_is_only_a_tiebreaker_after_content_similarity() -> None:
    previous = (
        "Les principaux risques comprennent : "
        "i) le risque de crédit associé aux entreprises; "
        "ii) le risque technologique associé aux systèmes; "
        "iii) le risque opérationnel associé aux processus."
    )
    current = (
        "Les principaux risques comprennent : "
        "i) le nouveau risque climatique associé aux activités; "
        "ii) le risque de crédit associé aux entreprises; "
        "iii) le risque technologique associé aux systèmes; "
        "iv) le risque opérationnel associé aux processus."
    )

    alignments = _align_chunks_tfidf(
        _chunk_subsection_text(previous),
        _chunk_subsection_text(current),
    )
    matched_markers = {
        (alignment.chunk_t1.atomic_marker, alignment.chunk_t2.atomic_marker)
        for alignment in alignments
        if alignment.chunk_t1
        and alignment.chunk_t2
        and alignment.chunk_t1.unit_role == "item"
    }
    added = [
        alignment
        for alignment in alignments
        if alignment.alignment_type == "possible_added"
    ]

    assert matched_markers == {("i)", "ii)"), ("ii)", "iii)"), ("iii)", "iv)")}
    assert len(added) == 1
    assert added[0].chunk_t2 is not None
    assert added[0].chunk_t2.atomic_marker == "i)"
    assert "climatique" in added[0].chunk_t2.comparison_text


def test_renumbered_identical_item_is_unchanged_in_exact_diff() -> None:
    previous = (
        "Les principaux risques comprennent : "
        "i) le risque de crédit associé aux entreprises; "
        "ii) le risque opérationnel associé aux processus."
    )
    current = (
        "Les principaux risques comprennent : "
        "i) le nouveau risque climatique associé aux activités; "
        "ii) le risque de crédit associé aux entreprises; "
        "iii) le risque opérationnel associé aux processus."
    )
    alignment = next(
        alignment
        for alignment in _align_chunks_tfidf(
            _chunk_subsection_text(previous),
            _chunk_subsection_text(current),
        )
        if alignment.chunk_t1
        and alignment.chunk_t2
        and alignment.chunk_t1.atomic_marker == "i)"
    )

    change = _exact_diff_change_for_strong_alignment(
        alignment=alignment,
        section_key="gestion_risques",
        heading_label="Principaux risques",
        heading_slug="principaux_risques",
        change_index=1,
    )

    assert change is not None
    assert change["diff_type"] == "unchanged"
    assert change["atomic_marker_t1"] == "i)"
    assert change["atomic_marker_t2"] == "ii)"


def test_reordered_markdown_bullets_are_aligned_by_content() -> None:
    previous = (
        "Les objectifs sont les suivants:\n\n"
        "- Préserver la réputation de la Banque.\n\n"
        "- Maintenir une solide position de liquidité."
    )
    current = (
        "Les objectifs sont les suivants:\n\n"
        "- Maintenir une solide position de liquidité.\n\n"
        "- Préserver la réputation de la Banque."
    )

    alignments = _align_chunks_tfidf(
        _chunk_subsection_text(previous),
        _chunk_subsection_text(current),
    )
    item_alignments = [
        alignment
        for alignment in alignments
        if alignment.chunk_t1
        and alignment.chunk_t2
        and alignment.chunk_t1.unit_role == "item"
    ]

    assert len(item_alignments) == 2
    assert all(alignment.similarity_score == 1.0 for alignment in item_alignments)
    assert {
        (
            alignment.chunk_t1.comparison_text,
            alignment.chunk_t2.comparison_text,
        )
        for alignment in item_alignments
    } == {
        (
            "Préserver la réputation de la Banque.",
            "Préserver la réputation de la Banque.",
        ),
        (
            "Maintenir une solide position de liquidité.",
            "Maintenir une solide position de liquidité.",
        ),
    }


def test_same_marker_with_distinct_content_is_not_forced_into_a_pair() -> None:
    previous = (
        "Les risques comprennent : "
        "i) le risque de crédit commercial; "
        "ii) le risque de liquidité structurelle."
    )
    current = (
        "Les risques comprennent : "
        "i) les attaques de cybersécurité externes; "
        "ii) le risque de liquidité structurelle."
    )

    alignments = _align_chunks_tfidf(
        _chunk_subsection_text(previous),
        _chunk_subsection_text(current),
    )

    removed = [
        alignment
        for alignment in alignments
        if alignment.alignment_type == "possible_removed"
    ]
    added = [
        alignment
        for alignment in alignments
        if alignment.alignment_type == "possible_added"
    ]
    assert len(removed) == 1
    assert removed[0].chunk_t1 is not None
    assert removed[0].chunk_t1.atomic_marker == "i)"
    assert len(added) == 1
    assert added[0].chunk_t2 is not None
    assert added[0].chunk_t2.atomic_marker == "i)"


def test_embedding_input_excludes_atomic_markers(monkeypatch) -> None:
    chunks = _chunk_subsection_text(
        "Les risques suivants sont surveillés : "
        "i) le risque de crédit commercial; "
        "ii) le risque opérationnel interne."
    )
    captured: list[str] = []

    def fake_embed(client, texts, model):
        captured.extend(texts)
        return [[1.0, float(index)] for index, _text in enumerate(texts)]

    monkeypatch.setattr(
        "vigie.analyse_texte.chunk_alignment._embed_texts",
        fake_embed,
    )

    _embedding_similarity_matrix(
        chunks,
        client=object(),
        embedding_model="text-embedding-3-small",
    )

    assert all(not text.startswith(("i)", "ii)")) for text in captured)
    assert any(text.startswith("le risque de crédit") for text in captured)


def test_atomic_metadata_is_attached_to_each_comparison_change() -> None:
    previous_chunks = _chunk_subsection_text(
        "Les risques suivants sont surveillés : "
        "i) le risque de crédit commercial; "
        "ii) le risque opérationnel interne."
    )
    current_chunks = _chunk_subsection_text(
        "Les risques suivants sont surveillés : "
        "i) le risque de crédit commercial renforcé; "
        "ii) le risque opérationnel interne."
    )
    alignment = next(
        alignment
        for alignment in _align_chunks_tfidf(previous_chunks, current_chunks)
        if alignment.chunk_t1
        and alignment.chunk_t2
        and alignment.chunk_t1.atomic_marker == "i)"
    )

    scoped = _attach_alignment_metadata(
        [
            {
                "alignment_id": alignment.alignment_id,
                "diff_type": "modified",
                "source_text_t1": alignment.chunk_t1.text,
                "source_text_t2": alignment.chunk_t2.text,
                "alignment_decision": "same_disclosure",
                "alignment_confidence": "high",
            }
        ],
        [alignment],
    )

    assert len(scoped) == 1
    assert scoped[0]["unit_role_t1"] == "item"
    assert scoped[0]["unit_role_t2"] == "item"
    assert scoped[0]["atomic_marker_t1"] == "i)"
    assert scoped[0]["atomic_marker_t2"] == "i)"
    assert scoped[0]["parent_chunk_id_t1"]
    assert scoped[0]["parent_chunk_id_t2"]


def test_changes_from_two_children_are_not_deduplicated_by_parent() -> None:
    changes = [
        {
            "section_key": "gestion_risques",
            "subsection_heading": "Valeur des titres",
            "alignment_id": "a03",
            "chunk_id_t1": "c03",
            "chunk_id_t2": "c03",
            "parent_chunk_id_t1": "c00",
            "parent_chunk_id_t2": "c00",
            "diff_type": "modified",
            "change_summary": "La limite de l'actif est ajoutée.",
        },
        {
            "section_key": "gestion_risques",
            "subsection_heading": "Valeur des titres",
            "alignment_id": "a06",
            "chunk_id_t1": "c06",
            "chunk_id_t2": "c06",
            "parent_chunk_id_t1": "c00",
            "parent_chunk_id_t2": "c00",
            "diff_type": "modified",
            "change_summary": "L'incapacité à atteindre les cibles est ajoutée.",
        },
    ]

    deduplicated = _deduplicate_alignment_changes(changes)

    assert len(deduplicated) == 2
    assert {change["alignment_id"] for change in deduplicated} == {"a03", "a06"}
