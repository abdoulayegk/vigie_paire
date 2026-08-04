"""Contexte comparaison pour les callbacks Dash (PDF, trimestres, noms d'export)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from vigie.support.quarter_utils import build_quarter_context


def _quarter_number(value: object) -> int | None:
    """Extrait un numero de trimestre entre 1 et 4."""
    match = re.search(r"[tq]?\s*([1-4])", str(value or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _comparison_period(
    meta: dict[str, object],
    top_level: dict[str, object],
    role: str,
) -> tuple[int | None, int | None]:
    """Retourne l'annee et le trimestre precedent ou courant."""
    quarter_context = meta.get("quarter_context")
    role_context = (
        quarter_context.get(role, {})
        if isinstance(quarter_context, dict)
        else {}
    )
    if not isinstance(role_context, dict):
        role_context = {}

    year_raw = (
        role_context.get("year")
        or top_level.get(f"year_{role}")
        or meta.get(f"year_{role}")
    )
    quarter_raw = (
        role_context.get("code")
        or role_context.get("quarter")
        or role_context.get("label")
        or top_level.get(f"quarter_{role}")
        or meta.get(f"quarter_{role}")
    )
    try:
        year = int(year_raw) if year_raw else None
    except (TypeError, ValueError):
        year = None
    return year, _quarter_number(quarter_raw)


def _case_insensitive_child(directory: Path, name: str) -> Path | None:
    """Trouve un sous-repertoire sans dependre de la casse du systeme."""
    try:
        return next(
            child
            for child in directory.iterdir()
            if child.is_dir() and child.name.casefold() == name.casefold()
        )
    except (OSError, StopIteration):
        return None


def _inputs_roots_from_comparison(compare_path_raw: str) -> list[Path]:
    """Trouve les repertoires Inputs ancetres du fichier de comparaison."""
    if not compare_path_raw:
        return []

    compare_path = Path(compare_path_raw)
    run_dir = compare_path.parent if compare_path.suffix else compare_path
    roots: list[Path] = []
    for ancestor in (run_dir, *run_dir.parents):
        candidate = ancestor / "Inputs"
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def _portable_basename(path_raw: str) -> str:
    """Extrait un nom de fichier depuis un chemin Windows ou POSIX."""
    return path_raw.replace("\\", "/").rsplit("/", 1)[-1]


def _find_input_pdf(
    inputs_root: Path,
    *,
    bank_code: str,
    year: int | None,
    quarter: int | None,
    source_hint: str,
) -> str:
    """Trouve le rapport source correspondant dans Inputs."""
    if not bank_code or year is None or quarter is None:
        return ""

    bank_dir = _case_insensitive_child(inputs_root, bank_code)
    if bank_dir is None:
        return ""
    year_dir = _case_insensitive_child(bank_dir, str(year))
    if year_dir is None:
        return ""

    try:
        pdfs = sorted(
            (
                path
                for path in year_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        return ""

    hint_name = _portable_basename(source_hint).casefold()
    if hint_name:
        for path in pdfs:
            if path.name.casefold() == hint_name:
                return str(path)

    canonical_name = f"{bank_code}_{year}_T{quarter}.pdf".casefold()
    for path in pdfs:
        if path.name.casefold() == canonical_name:
            return str(path)

    quarter_pattern = re.compile(
        rf"(?:^|[^a-z0-9])[tq][ _-]*{quarter}(?=$|[^0-9])",
        flags=re.IGNORECASE,
    )
    matching = [path for path in pdfs if quarter_pattern.search(path.stem)]
    if not matching:
        return ""

    bank_token = bank_code.casefold()
    year_token = str(year)
    matching.sort(
        key=lambda path: (
            year_token not in path.stem,
            bank_token not in path.stem.casefold(),
            path.name.casefold(),
        )
    )
    return str(matching[0])


def _input_pdf_fallbacks(
    compare_path_raw: str,
    *,
    meta: dict[str, object],
    top_level: dict[str, object],
    source_previous: str,
    source_current: str,
) -> tuple[str, str]:
    """Resout les deux rapports depuis le dossier Inputs du depot."""
    bank_code = str(
        meta.get("bank_code") or top_level.get("bank_code") or ""
    ).strip()
    previous_year, previous_quarter = _comparison_period(
        meta, top_level, "previous"
    )
    current_year, current_quarter = _comparison_period(meta, top_level, "current")

    previous = ""
    current = ""
    for inputs_root in _inputs_roots_from_comparison(compare_path_raw):
        if not previous:
            previous = _find_input_pdf(
                inputs_root,
                bank_code=bank_code,
                year=previous_year,
                quarter=previous_quarter,
                source_hint=source_previous,
            )
        if not current:
            current = _find_input_pdf(
                inputs_root,
                bank_code=bank_code,
                year=current_year,
                quarter=current_quarter,
                source_hint=source_current,
            )
        if previous and current:
            break
    return previous, current


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
    4. Rapports correspondants sous ``Inputs/{BANQUE}/{ANNEE}``.
    5. Chemins du ``manifest.json`` voisin si present.

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
    input_previous = ""
    input_current = ""
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

    input_previous, input_current = _input_pdf_fallbacks(
        compare_path_raw,
        meta=meta,
        top_level=top_level,
        source_previous=source_previous,
        source_current=source_current,
    )

    def _pick(*candidates: str) -> str:
        """Retourne le premier candidat existant sur disque, sinon le premier non vide."""
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
        input_previous,
        manifest_previous,
    )
    current = _pick(
        store_current,
        sibling_current,
        archived_current,
        source_current,
        input_current,
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
