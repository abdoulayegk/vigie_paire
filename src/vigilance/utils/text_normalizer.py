"""Normaliseur de texte utilise par les modules de comparaison historiques."""

from __future__ import annotations

import re

from vigilance.utils.matching_normalizer import normalize_for_matching


class TextNormalizer:
    """Wrapper de compatibilite pour les utilitaires de normalisation statique historiques."""

    @staticmethod
    def normalize(
        text: str,
        *,
        aggressive: bool = False,
        remove_notes: bool = False,
        lowercase: bool = True,
    ) -> str:
        """Normalise un texte pour la comparaison.

        Args:
            text: Texte brut a normaliser.
            aggressive: Si ``True``, supprime les mots-outils francais courants.
            remove_notes: Si ``True``, supprime les marqueurs de notes de bas de page.
            lowercase: Si ``False``, retourne en majuscules au lieu de minuscules.

        Returns:
            Texte normalise.
        """
        value = text or ""
        if remove_notes:
            # Extended: strip all footnote marker formats
            value = re.sub(
                r"\s*(?:\(\d+\)|\[\d+\]|\([a-zA-Z]\)|[a-zA-Z]\)|[\*†‡]+|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)\s*",
                " ",
                value,
            )
        value = normalize_for_matching(value)
        if not lowercase:
            value = value.upper()
        if aggressive:
            value = re.sub(r"\b(?:de|des|du|la|le|les|et|ou)\b", " ", value)
            value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def normalize_indicator(text: str) -> str:
        """Normalise un indicateur de maniere agressive (notes supprimees, mots-outils retires)."""
        return TextNormalizer.normalize(
            text,
            aggressive=True,
            remove_notes=True,
            lowercase=True,
        )
