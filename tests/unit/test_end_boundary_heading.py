"""Tests de l'égalité exacte des ancres de fin de section."""

from __future__ import annotations

from vigie.analyse_texte.docling_markdown import DoclingSegment, _is_end_boundary_heading
from vigie.analyse_texte.models import SectionAudit


def _risk_audit(*, end_anchor_text: str = "Gestion des fonds propres") -> SectionAudit:
    return SectionAudit(
        section_key="gestion_risques",
        section_title="Gestion des risques",
        start_page=76,
        end_page=126,
        anchor_page=76,
        anchor_text="Gestion du risque",
        anchor_bbox_norm=[0.1, 0.1, 0.8, 0.14],
        included_blocks=[],
        excluded_blocks=[],
        end_anchor_page=127,
        end_anchor_text=end_anchor_text,
        end_anchor_bbox_norm=[0.1, 0.25, 0.8, 0.28],
    )


def test_exact_end_anchor_closes_section() -> None:
    audits = [_risk_audit()]
    assert _is_end_boundary_heading(
        DoclingSegment(kind="heading", text="Gestion des fonds propres", heading_level=2),
        audits,
    )


def test_capital_subsection_does_not_close_risks() -> None:
    audits = [_risk_audit()]
    assert not _is_end_boundary_heading(
        DoclingSegment(
            kind="heading",
            text="Cadre de gestion des fonds propres",
            heading_level=3,
        ),
        audits,
    )


def test_cross_reference_does_not_close_risks() -> None:
    audits = [_risk_audit()]
    assert not _is_end_boundary_heading(
        DoclingSegment(
            kind="heading",
            text="se reporter à la rubrique Gestion des fonds propres",
            heading_level=3,
        ),
        audits,
    )
