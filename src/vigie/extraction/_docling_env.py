"""Valeurs booleennes acceptees dans les variables d'environnement.

Extrait pour que ``docling/config.py`` puisse les importer sans dependre de
``docling_bbox_helpers``, qui porte des responsabilites sans rapport.
"""

from __future__ import annotations

_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"0", "false", "no", "off"}
