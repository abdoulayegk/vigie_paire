"""Modules issus du decoupage de ``triage.py``.

Chaque module regroupe une responsabilite extraite du monolithe, sans changement
de comportement. ``triage`` reste la facade publique et re-exporte tout ce qui
etait accessible avant, y compris les symboles prives utilises par les tests.

Sans ce fichier, le sous-package echapperait a ``setuptools.packages.find`` et
serait absent des artefacts installes.
"""
