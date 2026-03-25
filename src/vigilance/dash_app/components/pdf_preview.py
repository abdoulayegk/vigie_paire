"""Composant affichage PDF (images base64)."""

from __future__ import annotations

from dash import html


def pdf_images_from_base64(
    images: list[str], captions: list[str] | None = None
) -> html.Div:
    """
    Afficher des images PDF (base64) dans une grille.

    Args:
        images: Liste de chaines base64 (PNG)
        captions: Captions optionnels pour chaque image
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
                    html.Img(
                        src=src, style={"maxWidth": "100%", "border": "1px solid #ddd"}
                    ),
                    html.P(cap, className="small text-muted mt-1"),
                ],
                className="mb-3",
            )
        )

    return html.Div(children)
