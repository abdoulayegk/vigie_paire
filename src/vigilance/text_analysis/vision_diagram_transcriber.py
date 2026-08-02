"""Transcription sémantique de schémas, diagrammes et organigrammes via GPT-4o Vision.

Ce module permet de transformer les éléments visuels non textuels (organigrammes RH,
architectures de gestion des risques, diagrammes de ventilation) en paragraphes
de texte canonique structurés et comparables dans le pipeline de diff.
"""

from __future__ import annotations

import base64
import logging

from vigilance.utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)

_DIAGRAM_VISION_SYSTEM_PROMPT = """\
Tu es un analyste senior en réglementation bancaire canadienne expert en analyse visuelle.
On te fournit une image représentant un schéma, un organigramme, une figure ou un diagramme extrait d'un rapport bancaire.

Retranscris l'intégralité du contenu de ce schéma de façon structurée et complète sous forme de Markdown propre :
1. Titre du schéma / diagramme.
2. Structure hiérarchique et lignes de responsabilité (ex: 1ère, 2ème, 3ème ligne de défense, comités, équipes).
3. Intitulés exacts des divisions, départements, politiques ou fonctions mentionnés.
4. Notes ou explications textuelles figurant dans le schéma.

RÈGLES STRICTES :
- Ne résume pas : conserve tous les mots clés métiers et intitulés exacts.
- Rédige en français sous forme de liste structurée lisible et directement comparable.
- Réponds UNIQUEMENT avec le texte Markdown du schéma, sans introduction ni conclusion.
"""


def transcribe_diagram_image_bytes_with_vision(
    image_bytes: bytes,
    *,
    diagram_title_context: str = "",
    model: str = "gpt-4o",
) -> str:
    """Envoie l'image d'un schéma/diagramme à GPT-4o Vision et retourne la transcription sémantique Markdown.

    Args:
        image_bytes: Bytes de l'image (PNG / JPEG).
        diagram_title_context: Contexte optionnel du titre du schéma.
        model: Nom du modèle OpenAI Vision (défaut 'gpt-4o').

    Returns:
        Texte Markdown structuré décrivant le diagramme, ou chaîne vide en cas d'échec.
    """
    if not image_bytes:
        return ""

    api_key = get_openai_api_key()
    if not api_key:
        logger.warning("Clé OpenAI manquante pour la transcription Vision de diagramme.")
        return ""

    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("OpenAI SDK non disponible pour la transcription Vision de diagramme.")
        return ""

    try:
        client = OpenAI(api_key=api_key)
        b64_img = base64.b64encode(image_bytes).decode("ascii")

        prompt_text = "Retranscris ce diagramme bancaire sous forme de texte structuré Markdown."
        if diagram_title_context:
            prompt_text += f" Titre contextuel du schéma : {diagram_title_context}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _DIAGRAM_VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        content = (response.choices[0].message.content or "").strip()
        if content:
            logger.info("Transcription GPT-4o Vision réussie pour le diagramme (%d caractères).", len(content))
        return content
    except Exception as exc:
        logger.warning("Échec de la transcription Vision du diagramme : %s", exc)
        return ""
