"""Modules issus du decoupage de ``page_text_analysis.py``.

Chaque module regroupe une responsabilite de rendu extraite du monolithe, sans
changement de comportement. ``page_text_analysis`` reste la facade publique.

Sans ce fichier, le sous-package echapperait a ``setuptools.packages.find`` et
serait absent des artefacts installes.
"""
