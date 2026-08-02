"""Generation du rapport PDF du tableau de bord (reportlab).

Extrait de ``vigie_dashboard_flow.py`` sans modification.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from vigilance.vigie_columns import build_text_vigie_display_row

from .formatting import _format_number, _pdf_text
from .metrics import _text_changes


_PDF_NAVY = colors.HexColor("#0b1725")
_PDF_PANEL = colors.HexColor("#101f31")
_PDF_TEXT = colors.HexColor("#172033")
_PDF_MUTED = colors.HexColor("#5f6b7a")
_PDF_BLUE = colors.HexColor("#4b74f2")
_PDF_RED = colors.HexColor("#e45142")
_PDF_AMBER = colors.HexColor("#f3b23c")
_PDF_GREEN = colors.HexColor("#68b976")


class _DashboardCharts(Flowable):
    """Dessine les graphiques principaux du rapport PDF."""

    def __init__(self, *, text_total: int, indicator_total: int, bars: dict[str, int]) -> None:
        """Initialise les compteurs et la taille du flowable cockpit."""
        super().__init__()
        self.text_total = text_total
        self.indicator_total = indicator_total
        self.bars = bars
        self.width = 7.2 * inch
        self.height = 3.45 * inch

    def draw(self) -> None:
        """Dessine le donut combiné et l'histogramme de répartition sur la page PDF."""
        canvas = self.canv
        canvas.saveState()
        self._draw_donut(canvas, 0.05 * inch, 0.2 * inch)
        self._draw_bars(canvas, 3.1 * inch, 0.2 * inch)
        canvas.restoreState()

    def _draw_donut(self, canvas, x: float, y: float) -> None:
        """Dessine le donut texte / indicateurs aux coordonnées ``(x, y)``."""
        total = max(1, self.text_total + self.indicator_total)
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(x, y + 2.95 * inch, "APERÇU COMBINÉ")
        cx = x + 1.05 * inch
        cy = y + 1.55 * inch
        radius = 0.82 * inch
        start = 90
        for value, color in ((self.text_total, _PDF_RED), (self.indicator_total, _PDF_BLUE)):
            extent = -360 * (value / total)
            canvas.setFillColor(color)
            canvas.wedge(cx - radius, cy - radius, cx + radius, cy + radius, start, extent, stroke=0, fill=1)
            start += extent
        canvas.setFillColor(colors.white)
        inner = 0.43 * inch
        canvas.circle(cx, cy, inner, stroke=0, fill=1)
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawCentredString(cx, cy + 0.05 * inch, str(self.text_total + self.indicator_total))
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_PDF_MUTED)
        canvas.drawCentredString(cx, cy - 0.18 * inch, "changements")
        legend_x = x + 2.15 * inch
        canvas.setFont("Helvetica", 8)
        for idx, (label, value, color) in enumerate(
            (
                ("Texte narratif", self.text_total, _PDF_RED),
                ("Indicateurs / notes", self.indicator_total, _PDF_BLUE),
            )
        ):
            yy = y + 2.15 * inch - idx * 0.25 * inch
            canvas.setFillColor(color)
            canvas.rect(legend_x, yy, 0.11 * inch, 0.11 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.drawString(legend_x + 0.18 * inch, yy, f"{label}: {value}")

    def _draw_bars(self, canvas, x: float, y: float) -> None:
        """Dessine l'histogramme horizontal « Répartition par nature » aux coordonnées ``(x, y)``."""
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(x, y + 2.95 * inch, "RÉPARTITION PAR NATURE")
        max_value = max(1, *(self.bars.values() or [1]))
        colors_by_label = {
            "Ajouts": _PDF_GREEN,
            "Suppressions": _PDF_RED,
            "Modifications": _PDF_AMBER,
            "Renommages": _PDF_BLUE,
        }
        bar_width = 2.75 * inch
        for idx, (label, value) in enumerate(self.bars.items()):
            yy = y + 2.35 * inch - idx * 0.48 * inch
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(_PDF_TEXT)
            canvas.drawString(x, yy + 0.03 * inch, label)
            canvas.setFillColor(colors.HexColor("#edf2f7"))
            canvas.rect(x + 1.15 * inch, yy, bar_width, 0.16 * inch, stroke=0, fill=1)
            canvas.setFillColor(colors_by_label.get(label, _PDF_BLUE))
            canvas.rect(x + 1.15 * inch, yy, bar_width * (value / max_value), 0.16 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(x + 4.0 * inch, yy + 0.02 * inch, str(value))


class _ReportBarChart(Flowable):
    """Dessine un graphique a barres pleine page pour le rapport PDF."""

    def __init__(
        self,
        *,
        title: str,
        values: list[tuple[str, int, colors.Color]],
        orientation: str = "vertical",
        width: float = 7.0 * inch,
        height: float = 4.8 * inch,
    ) -> None:
        """Initialise le titre, les valeurs et l'orientation du graphique à barres PDF."""
        super().__init__()
        self.title = title
        self.values = values
        self.orientation = orientation
        self.width = width
        self.height = height

    def draw(self) -> None:
        """Dessine le titre puis l'histogramme selon l'orientation choisie."""
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(0, self.height - 0.15 * inch, self.title)
        if self.orientation == "horizontal":
            self._draw_horizontal(canvas)
        else:
            self._draw_vertical(canvas)
        canvas.restoreState()

    def _draw_vertical(self, canvas) -> None:
        """Dessine un histogramme vertical avec axes et étiquettes."""
        chart_x = 0.45 * inch
        chart_y = 0.55 * inch
        chart_w = self.width - 0.9 * inch
        chart_h = self.height - 1.25 * inch
        values = [value for _, value, _ in self.values] or [0]
        max_value = max(1, max(values))
        canvas.setStrokeColor(colors.HexColor("#d7dee9"))
        canvas.setLineWidth(0.5)
        canvas.line(chart_x, chart_y, chart_x + chart_w, chart_y)
        canvas.line(chart_x, chart_y, chart_x, chart_y + chart_h)
        for step in range(1, 5):
            yy = chart_y + chart_h * step / 4
            canvas.setStrokeColor(colors.HexColor("#edf2f7"))
            canvas.line(chart_x, yy, chart_x + chart_w, yy)
            canvas.setFillColor(_PDF_MUTED)
            canvas.setFont("Helvetica", 7)
            canvas.drawRightString(chart_x - 0.08 * inch, yy - 0.03 * inch, str(round(max_value * step / 4)))
        gap = 0.18 * inch
        bar_w = min(0.75 * inch, (chart_w - gap * (len(self.values) + 1)) / max(1, len(self.values)))
        for idx, (label, value, color) in enumerate(self.values):
            x = chart_x + gap + idx * (bar_w + gap)
            h = chart_h * value / max_value
            canvas.setFillColor(color)
            canvas.rect(x, chart_y, bar_w, h, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawCentredString(x + bar_w / 2, chart_y + h + 0.08 * inch, str(value))
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica", 8)
            canvas.drawCentredString(x + bar_w / 2, chart_y - 0.22 * inch, _pdf_text(label, 18))

    def _draw_horizontal(self, canvas) -> None:
        """Dessine un histogramme horizontal avec étiquettes à gauche."""
        chart_x = 1.65 * inch
        chart_y = 0.55 * inch
        chart_w = self.width - 2.0 * inch
        values = [value for _, value, _ in self.values] or [0]
        max_value = max(1, max(values))
        row_h = min(0.48 * inch, (self.height - 1.35 * inch) / max(1, len(self.values)))
        for idx, (label, value, color) in enumerate(self.values):
            y = chart_y + (len(self.values) - idx - 1) * row_h
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(chart_x - 0.12 * inch, y + 0.11 * inch, _pdf_text(label, 26))
            canvas.setFillColor(colors.HexColor("#edf2f7"))
            canvas.rect(chart_x, y + 0.06 * inch, chart_w, 0.18 * inch, stroke=0, fill=1)
            canvas.setFillColor(color)
            canvas.rect(chart_x, y + 0.06 * inch, chart_w * value / max_value, 0.18 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(chart_x + chart_w + 0.08 * inch, y + 0.08 * inch, str(value))


class _ReportDonutChart(Flowable):
    """Dessine un graphique anneau pleine page pour le rapport PDF."""

    def __init__(
        self,
        *,
        title: str,
        values: list[tuple[str, int, colors.Color]],
        width: float = 7.0 * inch,
        height: float = 4.8 * inch,
    ) -> None:
        """Initialise le titre et les valeurs du graphique en anneau PDF."""
        super().__init__()
        self.title = title
        self.values = values
        self.width = width
        self.height = height

    def draw(self) -> None:
        """Dessine le titre, l'anneau et la légende du donut sur la page PDF."""
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(0, self.height - 0.15 * inch, self.title)
        total = max(1, sum(value for _, value, _ in self.values))
        cx = 1.75 * inch
        cy = 2.25 * inch
        radius = 1.3 * inch
        start = 90
        for _, value, color in self.values:
            extent = -360 * value / total
            canvas.setFillColor(color)
            canvas.wedge(cx - radius, cy - radius, cx + radius, cy + radius, start, extent, stroke=0, fill=1)
            start += extent
        canvas.setFillColor(colors.white)
        canvas.circle(cx, cy, 0.68 * inch, stroke=0, fill=1)
        canvas.setFillColor(_PDF_TEXT)
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawCentredString(cx, cy + 0.08 * inch, str(sum(value for _, value, _ in self.values)))
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(_PDF_MUTED)
        canvas.drawCentredString(cx, cy - 0.22 * inch, "changements")
        legend_x = 3.55 * inch
        for idx, (label, value, color) in enumerate(self.values):
            yy = 3.1 * inch - idx * 0.42 * inch
            pct = value / total
            canvas.setFillColor(color)
            canvas.rect(legend_x, yy, 0.16 * inch, 0.16 * inch, stroke=0, fill=1)
            canvas.setFillColor(_PDF_TEXT)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(legend_x + 0.28 * inch, yy, f"{label}: {value} ({pct:.0%})")


def _pdf_styles() -> dict[str, ParagraphStyle]:
    """Retourne le dictionnaire des styles ReportLab utilisés dans le rapport PDF."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "VigieTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=_PDF_NAVY,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "VigieH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=_PDF_NAVY,
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "VigieH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=_PDF_TEXT,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "VigieBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=_PDF_TEXT,
            spaceAfter=4,
        ),
        "muted": ParagraphStyle(
            "VigieMuted",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=_PDF_MUTED,
        ),
        "center": ParagraphStyle(
            "VigieCenter",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=_PDF_TEXT,
            alignment=TA_CENTER,
        ),
    }


def _pdf_table(data: list[list[Any]], widths: list[float] | None = None, *, header: bool = True) -> Table:
    """Construit un ``reportlab.Table`` stylisé pour le rapport PDF (grille, alternance de couleurs)."""
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7dee9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _PDF_NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _build_pdf_report(
    *,
    bank: str,
    current_label: str,
    previous_label: str,
    updated_at: str,
    text_metrics: dict[str, Any],
    indicator_metrics: dict[str, Any],
    review_counts: dict[str, int],
    bars: dict[str, int],
    priority_rows: list[dict[str, Any]],
    text_data: dict | None,
) -> bytes:
    """Construit le rapport PDF complet du cockpit (résumé, graphiques, priorités, annexe)."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"Rapport de vigie {bank}",
    )
    styles = _pdf_styles()
    story: list[Any] = []

    def p(text: Any, style: str = "body") -> Paragraph:
        """Construit un ``Paragraph`` ReportLab avec échappement HTML et style nommé."""
        return Paragraph(_pdf_text(text), styles[style])

    story.append(p("Rapport de vigie bancaire", "title"))
    story.append(p("Document interne - confidentiel", "h1"))
    story.append(p(f"{bank} - Comparaison {current_label} vs {previous_label}", "h1"))
    story.append(p(f"Sources: pipelines texte et indicateurs | Généré le {updated_at}", "muted"))
    story.append(Spacer(1, 0.18 * inch))
    total_changes = text_metrics["total"] + indicator_metrics["total_changes"]
    executive = (
        f"{total_changes} changement(s) détecté(s), dont {text_metrics['major']} majeur(s), "
        f"{text_metrics['moderate']} modéré(s), {indicator_metrics['priority']} tableau(x) prioritaire(s) "
        f"et {indicator_metrics['low_confidence']} appariement(s) à faible confiance."
    )
    story.append(p(executive, "body"))
    story.append(Spacer(1, 0.12 * inch))
    kpis = [
        ["Dimension", "Valeur", "Lecture analyste"],
        ["Total changements", _format_number(total_changes), "Volume global à couvrir"],
        ["Texte narratif", _format_number(text_metrics["total"]), "Tous changements auditables"],
        ["Indicateurs / notes", _format_number(indicator_metrics["total_changes"]), "Tables, indicateurs et notes"],
        ["Pertinents / analysés", f"{text_metrics['relevant']} / {text_metrics['analyzed']}", "Couverture de triage"],
        ["File de revue", _format_number(review_counts["total"]), "Cas à suivre côté revue"],
        ["En attente", _format_number(review_counts["pending"]), "Actions ouvertes"],
    ]
    story.append(_pdf_table(kpis, [1.55 * inch, 1.1 * inch, 4.15 * inch]))

    story.append(PageBreak())
    story.append(p("Vue de synthèse", "h1"))
    story.append(
        _ReportDonutChart(
            title="Répartition des changements par source",
            values=[
                ("Texte narratif", text_metrics["total"], _PDF_RED),
                ("Indicateurs et notes", indicator_metrics["total_changes"], _PDF_BLUE),
            ],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    source_rows = [
        ["Source", "Nombre", "Lecture analyste"],
        ["Texte narratif", _format_number(text_metrics["total"]), "Changements détectés dans les sections narratives"],
        ["Indicateurs et notes", _format_number(indicator_metrics["total_changes"]), "Changements détectés dans les tableaux et notes"],
    ]
    story.append(_pdf_table(source_rows, [1.7 * inch, 1.0 * inch, 4.0 * inch]))

    story.append(PageBreak())
    story.append(p("Répartition par nature", "h1"))
    story.append(
        _ReportBarChart(
            title="Types de changements détectés",
            values=[
                ("Ajouts", bars.get("Ajouts", 0), _PDF_GREEN),
                ("Suppressions", bars.get("Suppressions", 0), _PDF_RED),
                ("Modifications", bars.get("Modifications", 0), _PDF_AMBER),
                ("Renommages", bars.get("Renommages", 0), _PDF_BLUE),
            ],
            orientation="horizontal",
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    nature = [["Nature", "Nombre"], *[[key, _format_number(value)] for key, value in bars.items()]]
    story.append(_pdf_table(nature, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Évolution globale", "h1"))
    story.append(
        _ReportBarChart(
            title="Volumes consolidés par type",
            values=[
                ("Ajouts", bars.get("Ajouts", 0), _PDF_GREEN),
                ("Suppressions", bars.get("Suppressions", 0), _PDF_RED),
                ("Modifications", bars.get("Modifications", 0), _PDF_AMBER),
                ("Renommages", bars.get("Renommages", 0), _PDF_BLUE),
            ],
        )
    )

    story.append(PageBreak())
    story.append(p("Sévérité des changements textuels", "h1"))
    story.append(
        _ReportBarChart(
            title="Classification des changements narratifs",
            values=[
                ("Majeur", text_metrics.get("major", 0), _PDF_RED),
                ("Modéré", text_metrics.get("moderate", 0), _PDF_AMBER),
                ("Pertinent", text_metrics.get("relevant", 0), _PDF_BLUE),
                ("Analysé", text_metrics.get("analyzed", 0), _PDF_GREEN),
            ],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    text_summary_rows = [
        ["Indicateur texte", "Valeur"],
        ["Total changements", _format_number(text_metrics["total"])],
        ["Pertinents / analysés", f"{text_metrics['relevant']} / {text_metrics['analyzed']}"],
        ["Changements majeurs", _format_number(text_metrics["major"])],
        ["Changements modérés", _format_number(text_metrics["moderate"])],
    ]
    story.append(_pdf_table(text_summary_rows, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Indicateurs et notes", "h1"))
    story.append(
        _ReportBarChart(
            title="Changements structurés par catégorie",
            values=[
                ("Ind. ajoutés", indicator_metrics["indicator_added"], _PDF_GREEN),
                ("Ind. supprimés", indicator_metrics["indicator_removed"], _PDF_RED),
                ("Ind. renommés", indicator_metrics["indicator_renamed"], _PDF_BLUE),
                ("Notes ajoutées", indicator_metrics["footnote_added"], _PDF_GREEN),
                ("Notes supprimées", indicator_metrics["footnote_removed"], _PDF_RED),
                ("Notes modifiées", indicator_metrics["footnote_modified"], _PDF_AMBER),
                ("Notes renommées", indicator_metrics["footnote_renamed"], _PDF_BLUE),
            ],
            orientation="horizontal",
        )
    )

    story.append(PageBreak())
    story.append(p("File de revue", "h1"))
    story.append(
        _ReportBarChart(
            title="Statut de validation analyste",
            values=[
                ("Validés", review_counts.get("approved", 0), _PDF_GREEN),
                ("Rejetés", review_counts.get("rejected", 0), _PDF_RED),
                ("En attente", review_counts.get("pending", 0), _PDF_AMBER),
            ],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    review_rows = [
        ["Statut", "Nombre"],
        ["Total", _format_number(review_counts.get("total", 0))],
        ["Validés", _format_number(review_counts.get("approved", 0))],
        ["Rejetés", _format_number(review_counts.get("rejected", 0))],
        ["En attente", _format_number(review_counts.get("pending", 0))],
    ]
    story.append(_pdf_table(review_rows, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Priorités analyste", "h1"))
    priority_data = [["#", "Tableau", "Type", "Impact", "Confiance", "Statut"]]
    for idx, row in enumerate(priority_rows, start=1):
        priority_data.append(
            [
                str(idx),
                p(row["title"], "body"),
                p(row["change"], "body"),
                row["impact"],
                row["confidence"],
                row["status_label"],
            ]
        )
    if len(priority_data) == 1:
        priority_data.append(["-", "Aucun tableau prioritaire détecté.", "", "", "", ""])
    story.append(_pdf_table(priority_data, [0.3 * inch, 2.55 * inch, 1.35 * inch, 0.75 * inch, 0.75 * inch, 0.9 * inch]))

    story.append(PageBreak())
    story.append(p("Changements textuels principaux", "h1"))
    text_rows = [["#", "Section", "Impact", "Résumé"]]
    for idx, item in enumerate(text_metrics.get("top") or [], start=1):
        text_rows.append([str(idx), p(item["section"], "body"), item["impact"], p(item["summary"], "body")])
    if len(text_rows) == 1:
        text_rows.append(["-", "", "", "Aucun changement textuel prioritaire."])
    story.append(_pdf_table(text_rows, [0.3 * inch, 1.45 * inch, 0.85 * inch, 4.2 * inch]))

    story.append(PageBreak())
    story.append(p("Synthèse indicateurs et notes", "h1"))
    indicator_rows = [
        ["Catégorie", "Nombre"],
        ["Indicateurs ajoutés", _format_number(indicator_metrics["indicator_added"])],
        ["Indicateurs supprimés", _format_number(indicator_metrics["indicator_removed"])],
        ["Indicateurs renommés", _format_number(indicator_metrics["indicator_renamed"])],
        ["Notes ajoutées", _format_number(indicator_metrics["footnote_added"])],
        ["Notes supprimées", _format_number(indicator_metrics["footnote_removed"])],
        ["Notes modifiées", _format_number(indicator_metrics["footnote_modified"])],
        ["Notes renommées", _format_number(indicator_metrics["footnote_renamed"])],
        ["Tableaux ajoutés", _format_number(indicator_metrics["tables_added"])],
        ["Tableaux supprimés", _format_number(indicator_metrics["tables_removed"])],
    ]
    story.append(_pdf_table(indicator_rows, [2.4 * inch, 1.2 * inch]))

    story.append(PageBreak())
    story.append(p("Annexe - changements détectés", "h1"))
    annex_rows = [["#", "Pipeline", "Type", "Section / tableau", "Résumé"]]
    for idx, (change, section) in enumerate(_text_changes(text_data, relevant_only=False)[:30], start=1):
        display = build_text_vigie_display_row(
            change,
            section_title=section,
            bank_code=bank,
        )
        annex_rows.append(
            [
                str(idx),
                "Texte",
                str(change.get("diff_type") or ""),
                p(section, "body"),
                p(
                    display.get("what_changed")
                    or change.get("source_text_t2")
                    or change.get("source_text_t1"),
                    "body",
                ),
            ]
        )
    for idx, row in enumerate(priority_rows[:10], start=len(annex_rows)):
        annex_rows.append([str(idx), "Indicateurs", row["change"], p(row["title"], "body"), row["impact"]])
    if len(annex_rows) == 1:
        annex_rows.append(["-", "", "", "", "Aucun changement détecté."])
    story.append(_pdf_table(annex_rows, [0.3 * inch, 0.7 * inch, 0.85 * inch, 1.55 * inch, 3.4 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(p("Note: l'annexe est un extrait opérationnel; l'export Excel conserve le détail complet ligne par ligne.", "muted"))

    def _page(canvas, doc_obj) -> None:
        """Dessine le bandeau d'en-tête et de pied de page sur chaque page du rapport PDF."""
        canvas.saveState()
        canvas.setFillColor(_PDF_NAVY)
        canvas.rect(0, letter[1] - 0.22 * inch, letter[0], 0.22 * inch, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(0.45 * inch, letter[1] - 0.14 * inch, "Document interne - confidentiel")
        canvas.setFillColor(_PDF_MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(0.45 * inch, 0.25 * inch, "Usage interne - vigie bancaire")
        canvas.drawRightString(letter[0] - 0.45 * inch, 0.25 * inch, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return buffer.getvalue()
