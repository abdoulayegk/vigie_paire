"""UI only: affichage base64 Dash. Rendu pages PDF dans ``vigie.extraction.pdf_preview``."""

from __future__ import annotations

from dash import html


def pdf_images_from_base64(images: list[str], captions: list[str] | None = None) -> html.Div:
    """Affiche des images PDF encodees en base64 dans une grille verticale.

    Args:
        images: Liste de chaines base64 representant des images PNG.
        captions: Legendes optionnelles pour chaque image. Si absent,
            des legendes ``Page 1``, ``Page 2``, ... sont generees.

    Returns:
        Un ``Div`` contenant les images avec leurs legendes, ou un
        message de repli si *images* est vide.
    """
    if not images:
        return html.Div("Aucune image disponible", className="text-muted")

    if captions is None:
        captions = [f"Page {i + 1}" for i in range(len(images))]

    children = []
    for i, (b64, cap) in enumerate(zip(images, captions)):
        src = f"data:image/png;base64,{b64}" if b64 else ""
        children.append(
            html.Div(
                [
                    html.Img(src=src, style={"maxWidth": "100%", "border": "1px solid #ddd"}),
                    html.P(cap, className="small text-muted mt-1"),
                ],
                className="mb-3",
            )
        )

    return html.Div(children)
