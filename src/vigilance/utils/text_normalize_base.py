"""Normalisation de base partagee pour la comparaison de texte entre indicateurs et notes.

Source unique de verite pour : suppression des accents, canonicalisation des
apostrophes, reduction elision+espace, et normalisation des espaces. Utilise
par indicator_cleaner, footnote_comparator, row_bbox_extractor,
vision_indicator_added_validator et footnotes_utils pour eliminer les faux
positifs lies a la variance de caracteres/encodages.
"""

from __future__ import annotations

import re
import unicodedata

# Apostrophe variants (typographic, modifier letters, fullwidth) -> ASCII '
_APOSTROPHE_VARIANTS_RE = re.compile(r"[\u2019\u2018\u0060\u00b4\u02bc\u02bb\uff07]")
# Elision: "d' impôt" -> "d'impôt" (collapse space after apostrophe for French elision)
_ELISION_SPACE_RE = re.compile(r"\b([dljscnmt])'\s+", re.IGNORECASE)


def normalize_text_base(text: str, *, lowercase: bool = True) -> str:
    """Normalise le texte pour une comparaison stable : accents, apostrophes, elision, espaces.

    Quand ``lowercase=True`` (defaut), met aussi en minuscules (matching, notes, bbox).
    Quand ``lowercase=False``, preserve la casse pour la passe initiale d'indicator_cleaner
    afin que les regex en aval (dates, unites, numeros de lignes) conservent le comportement attendu.

    Ordre des operations :
    1. NFD + encode ascii ignore (suppression des accents)
    2. Normalisation des variantes d'apostrophes vers ASCII ``'``
    3. Reduction elision+espace (ex. ``"d' actions"`` -> ``"d'actions"``)
    4. Reduction de tous les espaces (y compris U+00A0) en espace simple, strip
    5. Mise en minuscules si ``lowercase=True``

    Args:
        text: Texte brut a normaliser.
        lowercase: Si ``True``, convertit en minuscules.

    Returns:
        Texte normalise.
    """
    if not text:
        return ""
    # NBSP and typographic apostrophes are not ASCII; normalize before ascii strip
    s = str(text).replace("\u00a0", " ")
    s = _APOSTROPHE_VARIANTS_RE.sub("'", s)
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    s = _ELISION_SPACE_RE.sub(r"\1'", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if lowercase else s
