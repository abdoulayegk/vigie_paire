"""Tests for page-local structure derivation (table_index_on_page, role, etc.)."""

from __future__ import annotations

from vigie.support.utils.table_page_structure import derive_page_local_structure


def test_derive_page_local_structure_empty() -> None:
    assert derive_page_local_structure([]) == {}


def test_derive_page_local_structure_single_table() -> None:
    class T:
        table_id = "a"
        page_pdf = 1
        bbox = [0.1, 0.2, 0.9, 0.5]

    out = derive_page_local_structure([T()])
    assert out == {
        ("a", 1): {"table_index_on_page": 1, "tables_on_page": 1, "bbox_top": 0.2, "page_local_role": "single"}
    }


def test_derive_page_local_structure_two_tables_same_page() -> None:
    class T1:
        table_id = "top"
        page_pdf = 1
        bbox = [0.1, 0.2, 0.9, 0.45]

    class T2:
        table_id = "lower"
        page_pdf = 1
        bbox = [0.1, 0.5, 0.9, 0.85]

    out = derive_page_local_structure([T1(), T2()])
    assert out[("top", 1)]["table_index_on_page"] == 1
    assert out[("top", 1)]["tables_on_page"] == 2
    assert out[("top", 1)]["page_local_role"] == "first"
    assert out[("lower", 1)]["table_index_on_page"] == 2
    assert out[("lower", 1)]["tables_on_page"] == 2
    assert out[("lower", 1)]["page_local_role"] == "last"
    assert out[("top", 1)]["bbox_top"] == 0.2
    assert out[("lower", 1)]["bbox_top"] == 0.5


def test_derive_page_local_structure_two_pages() -> None:
    class T1:
        table_id = "a"
        page_pdf = 1
        bbox = [0.0, 0.1, 1.0, 0.5]

    class T2:
        table_id = "b"
        page_pdf = 2
        bbox = [0.0, 0.2, 1.0, 0.6]

    out = derive_page_local_structure([T1(), T2()])
    assert out[("a", 1)]["table_index_on_page"] == 1
    assert out[("a", 1)]["tables_on_page"] == 1
    assert out[("b", 2)]["table_index_on_page"] == 1
    assert out[("b", 2)]["tables_on_page"] == 1


def test_derive_page_local_structure_skips_missing_bbox() -> None:
    class TNoBbox:
        table_id = "x"
        page_pdf = 1
        bbox = None

    class TWithBbox:
        table_id = "y"
        page_pdf = 1
        bbox = [0.1, 0.2, 0.9, 0.5]

    out = derive_page_local_structure([TNoBbox(), TWithBbox()])
    assert ("x", 1) not in out
    assert ("y", 1) in out
    assert out[("y", 1)]["table_index_on_page"] == 1


def test_derive_page_local_structure_page_number_fallback() -> None:
    class T:
        table_id = "p"
        page_number = 3
        bbox = [0.0, 0.0, 1.0, 0.3]

    out = derive_page_local_structure([T()])
    assert ("p", 3) in out
    assert out[("p", 3)]["table_index_on_page"] == 1


def test_derive_page_local_structure_three_tables_roles() -> None:
    class T1:
        table_id = "first"
        page_pdf = 1
        bbox = [0.0, 0.1, 1.0, 0.3]

    class T2:
        table_id = "mid"
        page_pdf = 1
        bbox = [0.0, 0.35, 1.0, 0.55]

    class T3:
        table_id = "last"
        page_pdf = 1
        bbox = [0.0, 0.6, 1.0, 0.9]

    out = derive_page_local_structure([T1(), T2(), T3()])
    assert out[("first", 1)]["page_local_role"] == "first"
    assert out[("mid", 1)]["page_local_role"] == "middle"
    assert out[("last", 1)]["page_local_role"] == "last"
    assert out[("first", 1)]["table_index_on_page"] == 1
    assert out[("mid", 1)]["table_index_on_page"] == 2
    assert out[("last", 1)]["table_index_on_page"] == 3
    assert out[("first", 1)]["tables_on_page"] == 3


def test_derive_page_local_structure_stable_sort_same_top() -> None:
    class T1:
        table_id = "left"
        page_pdf = 1
        bbox = [0.0, 0.2, 0.4, 0.5]

    class T2:
        table_id = "right"
        page_pdf = 1
        bbox = [0.5, 0.2, 0.9, 0.5]

    out = derive_page_local_structure([T1(), T2()])
    assert out[("left", 1)]["table_index_on_page"] == 1
    assert out[("right", 1)]["table_index_on_page"] == 2
