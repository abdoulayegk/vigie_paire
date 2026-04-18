"""Fonctions utilitaires de contexte pour les callbacks Dash.

Fournit la resolution des chemins PDF, le contexte trimestriel et les noms
d'export. Extrait de ``dash_app/app.py`` ; ``app.py`` reexporte tous les
noms de ce module afin que les monkey-patches existants continuent de
fonctionner.
"""

from __future__ import annotations

import json
from pathlib import Path

from vigilance.quarter_utils import build_quarter_context


def _comparison_path_from_meta(
    indicator_meta: dict | None, indicator_result: dict | None = None
) -> str:
    """Retourne le chemin de comparaison persiste a partir de l'etat Dash disponible."""
    meta = indicator_meta if isinstance(indicator_meta, dict) else {}
    compare_path = str(meta.get("compare_path") or "").strip()
    if compare_path:
        return compare_path
    if isinstance(indicator_result, dict):
        result_meta = indicator_result.get("meta", {}) or {}
        return str(result_meta.get("compare_path") or "").strip()
    return ""


def _quarter_context_from_store(data: dict | None) -> dict:
    """Reconstitue le contexte trimestriel depuis le store Dash ou utilise un defaut."""
    if isinstance(data, dict):
        current = data.get("current")
        previous = data.get("previous")
        if isinstance(current, dict) and isinstance(previous, dict):
            return data
    return build_quarter_context("T2", year=2025)


def _pdf_paths_from_comparison_meta(
    indicator_meta: dict | None,
    indicator_result: dict | None = None,
) -> dict[str, str]:
    """Resout les chemins PDF (precedent/courant) depuis les metadonnees de comparaison.

    Cascade de resolution — le premier chemin qui existe sur le disque gagne :
    1. Stores Dash (``pdf_paths.pdf_previous`` / ``pdf_current``)
    2. Voisins archives ``previous_report.pdf`` / ``current_report.pdf`` dans le
       repertoire du run (portable entre OS, priorise pour la portabilite).
    3. Chemins absolus stockes (``archived_pdf_*`` puis ``source_pdf_*``).
    4. Chemins du ``manifest.json`` voisin si present.

    Si aucun candidat n'existe, on retourne le premier non vide afin que le
    message d'avertissement utilisateur reste coherent.
    """
    meta: dict[str, object] = {}
    top_level: dict[str, object] = {}
    if isinstance(indicator_result, dict):
        top_level = indicator_result
        result_meta = indicator_result.get("meta", {})
        if isinstance(result_meta, dict):
            meta.update(result_meta)
    if isinstance(indicator_meta, dict):
        meta.update(indicator_meta)

    raw_paths = meta.get("pdf_paths") if isinstance(meta.get("pdf_paths"), dict) else {}
    store_previous = str(
        raw_paths.get("pdf_previous") or raw_paths.get("pdf_t1") or ""
    ).strip()
    store_current = str(
        raw_paths.get("pdf_current") or raw_paths.get("pdf_t2") or ""
    ).strip()

    archived_previous = str(
        meta.get("archived_pdf_previous") or top_level.get("archived_pdf_previous") or ""
    ).strip()
    archived_current = str(
        meta.get("archived_pdf_current") or top_level.get("archived_pdf_current") or ""
    ).strip()
    source_previous = str(
        meta.get("source_pdf_previous") or top_level.get("source_pdf_previous") or ""
    ).strip()
    source_current = str(
        meta.get("source_pdf_current") or top_level.get("source_pdf_current") or ""
    ).strip()

    sibling_previous = ""
    sibling_current = ""
    manifest_previous = ""
    manifest_current = ""
    compare_path_raw = str(
        meta.get("compare_path") or top_level.get("compare_path") or ""
    ).strip()
    if compare_path_raw:
        compare_path = Path(compare_path_raw)
        run_dir = compare_path.parent if compare_path.suffix else compare_path
        sibling_p = run_dir / "previous_report.pdf"
        sibling_c = run_dir / "current_report.pdf"
        if sibling_p.exists():
            sibling_previous = str(sibling_p)
        if sibling_c.exists():
            sibling_current = str(sibling_c)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_previous = str(
                    (manifest.get("previous") or {}).get("pdf_path") or ""
                ).strip()
                manifest_current = str(
                    (manifest.get("current") or {}).get("pdf_path") or ""
                ).strip()
            except (OSError, ValueError):
                pass

    def _pick(*candidates: str) -> str:
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        for candidate in candidates:
            if candidate:
                return candidate
        return ""

    previous = _pick(
        store_previous,
        sibling_previous,
        archived_previous,
        source_previous,
        manifest_previous,
    )
    current = _pick(
        store_current,
        sibling_current,
        archived_current,
        source_current,
        manifest_current,
    )

    return {
        "pdf_t1": previous,
        "pdf_t2": current,
        "pdf_previous": previous,
        "pdf_current": current,
    }


def _normalize_pdf_paths_store(paths: dict | None) -> dict[str, str]:
    """Normalise un dictionnaire de chemins PDF en cles canoniques ``pdf_t1``/``pdf_t2``."""
    source = paths if isinstance(paths, dict) else {}
    previous = str(source.get("pdf_previous") or source.get("pdf_t1") or "").strip()
    current = str(source.get("pdf_current") or source.get("pdf_t2") or "").strip()
    return {
        "pdf_t1": previous,
        "pdf_t2": current,
        "pdf_previous": previous,
        "pdf_current": current,
    }


def _missing_pdf_warning(paths: dict[str, str] | None) -> str:
    """Retourne un message d'avertissement si des PDF de preuve sont introuvables."""
    if not isinstance(paths, dict):
        return (
            "Comparaison chargée, mais les PDF archivés de preuve sont indisponibles."
        )
    missing: list[str] = []
    previous = str(paths.get("pdf_previous") or paths.get("pdf_t1") or "").strip()
    current = str(paths.get("pdf_current") or paths.get("pdf_t2") or "").strip()
    if not previous or not Path(previous).exists():
        missing.append("rapport précédent")
    if not current or not Path(current).exists():
        missing.append("rapport courant")
    if not missing:
        return ""
    joined = " et ".join(missing)
    return (
        "Comparaison chargée, mais la preuve PDF archivée est indisponible pour le "
        f"{joined}."
    )
