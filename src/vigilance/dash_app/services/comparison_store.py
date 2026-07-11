"""Magasin de comparaisons (fichier d'abord) utilise par Dash.

Ce module centralise tous les acces Dash aux artefacts de comparaison
sauvegardes afin que l'interface utilisateur ne se couple plus directement
aux helpers du systeme de fichiers. Le backend initial est l'arborescence
locale ``comparison.json``, mais l'interface est concue pour qu'un futur
backend base de donnees puisse le remplacer sans reecrire chaque callback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, TypedDict

from vigilance.comparison_canonical import (
    is_canonical_comparison,
    new_empty_ui_comparison_payload,
    to_canonical_payload,
)
from vigilance.dash_app.services.comparison_context import (
    _missing_pdf_warning,
    _normalize_pdf_paths_store,
    _pdf_paths_from_comparison_meta,
)
from vigilance.quarter_utils import build_quarter_context
from vigilance.review_storage import load_review_state, save_review_state
from vigilance.ui_config import INDICATOR_COMPARISON_DIR
from vigilance.ui_io import load_comparison_result


def _text_only_ui_payload(text_payload: dict[str, Any], compare_path: Path) -> dict[str, Any]:
    """Construit un payload Dash minimal pour une analyse texte sans indicateurs."""
    bank_code = str(
        text_payload.get("bank_code")
        or compare_path.parent.parent.name
        or ""
    ).lower()
    current_period = str(text_payload.get("quarter_current") or "").strip()
    previous_period = str(text_payload.get("quarter_previous") or "").strip()
    if not current_period or not previous_period:
        folder = compare_path.parent.name
        if "_vs_" in folder:
            current_period, previous_period = folder.split("_vs_", 1)

    try:
        ctx = build_quarter_context(current_period, previous_quarter=previous_period)
    except Exception:
        ctx = build_quarter_context("T2", year=text_payload.get("year_current") or 2025)

    current = ctx.get("current") or {}
    previous = ctx.get("previous") or {}
    created_at = str(
        text_payload.get("created_at")
        or text_payload.get("generated_at")
        or ""
    )

    payload = new_empty_ui_comparison_payload()
    payload["bank_code"] = bank_code
    payload["year"] = int(current.get("year") or text_payload.get("year_current") or payload["year"])
    payload["quarter_from"] = str(previous.get("label") or previous_period)
    payload["quarter_to"] = str(current.get("label") or current_period)
    payload["previous_quarter"] = str(previous.get("label") or previous_period)
    payload["current_quarter"] = str(current.get("label") or current_period)
    payload["created_at"] = created_at
    payload["meta"].update(
        {
            "generated_at": created_at or payload["meta"].get("generated_at"),
            "provenance": "text_comparison",
            "source_format": "text_only",
            "bank_code": bank_code,
            "quarter_context": ctx,
            "text_comparison_path": str(compare_path),
            "compare_path": str(compare_path),
            "executive_summary": {
                "content": "Analyse textuelle disponible; aucun résultat indicateurs dans ce run."
            },
        }
    )
    return payload


class ComparisonPayloadBundle(TypedDict):
    """Payload canonique et contexte systeme de fichiers resolu pour Dash."""

    compare_path: str
    raw_data: dict[str, Any]
    indicator_result: dict[str, Any]
    indicator_meta: dict[str, Any]
    pdf_paths: dict[str, str]
    warning: str


class ComparisonStore(Protocol):
    """Interface de persistence des comparaisons utilisee par Dash."""

    def list_comparison_options(self) -> list[dict[str, str]]:
        """Lister les options de comparaison disponibles."""
        ...

    def list_saved_run_options(
        self,
        *,
        bank_code: str,
        year: int | str,
        current_quarter: str,
    ) -> list[dict[str, Any]]:
        """Lister les executions sauvegardees pour une banque et un trimestre."""
        ...

    def load_dash_payload(
        self,
        target: str | Path,
        *,
        source: str,
        source_label: str,
    ) -> ComparisonPayloadBundle | None:
        """Charger le payload de comparaison depuis un artefact cible."""
        ...

    def load_review_state(
        self,
        compare_path: str | Path | None,
        *,
        username: str | None = None,
    ) -> dict[str, Any] | None:
        """Charger l'etat de revue associe a une comparaison."""
        ...

    def save_review_state(
        self,
        compare_path: str | Path | None,
        **kwargs: Any,
    ) -> Path | None:
        """Persister l'etat de revue pour une comparaison."""
        ...


class FileComparisonStore:
    """Magasin de comparaisons supporte par les artefacts locaux ``comparison.json``."""

    def __init__(self, *, root_dir: str | Path | None = None) -> None:
        """Initialise le magasin avec un repertoire racine optionnel.

        Args:
            root_dir: Repertoire racine des artefacts de comparaison.
                Par defaut, utilise ``INDICATOR_COMPARISON_DIR``.
        """
        self.root_dir = Path(root_dir) if root_dir else INDICATOR_COMPARISON_DIR

    def list_comparison_options(self) -> list[dict[str, str]]:
        """Liste toutes les comparaisons disponibles sous forme d'options ``{label, value}``.

        Returns:
            Liste de dictionnaires triee par date de modification decroissante.
        """
        if not self.root_dir.exists():
            return []

        files = sorted(
            self.root_dir.glob("**/*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        options: list[dict[str, str]] = []
        for path in files:
            if path.name == "text_comparison.json" and (path.parent / "comparison.json").exists():
                continue
            if path.name not in {"comparison.json", "text_comparison.json"}:
                continue
            try:
                rel = path.relative_to(self.root_dir)
            except ValueError:
                rel = path
            options.append({"label": rel.as_posix(), "value": rel.as_posix()})
        return options

    def list_saved_run_options(
        self,
        *,
        bank_code: str,
        year: int | str,
        current_quarter: str,
    ) -> list[dict[str, Any]]:
        """Liste les executions sauvegardees filtrees par banque, annee et trimestre.

        Args:
            bank_code: Code de la banque (ex. ``bnc``, ``rbc``).
            year: Annee de l'analyse.
            current_quarter: Trimestre courant selectionne (ex. ``T2``).

        Returns:
            Liste d'options ``{label, value}`` triee par valeur decroissante.
        """
        try:
            ctx = build_quarter_context(current_quarter, year=year)
            curr_y = ctx["current"]["year"]
            curr_q_t = ctx["current"]["code"]
            prev_y = ctx["previous"]["year"]
            prev_q_t = ctx["previous"]["code"]
        except Exception:
            curr_y, curr_q_t = year, "t2"
            prev_y, prev_q_t = year, "t1"

        expected_dir = f"{bank_code}/{curr_y}_{curr_q_t}_vs_{prev_y}_{prev_q_t}"
        filtered_options: list[dict[str, Any]] = []
        for option in self.list_comparison_options():
            value = str(option.get("value", "") or "")
            if not value.startswith(expected_dir):
                continue
            payload = self.load_dash_payload(
                value,
                source="analyse_enregistree",
                source_label="Analyse enregistrée",
            )
            timestamp_label = "Analyse enregistrée"
            if payload is not None:
                created_at = str(payload["raw_data"].get("created_at", "") or "")
                if created_at and "T" in created_at:
                    date_part, time_part = created_at.split("T", 1)
                    time_part = time_part[:5]
                    day_parts = date_part.split("-")
                    if len(day_parts) == 3:
                        timestamp_label = f"Le {day_parts[2]}/{day_parts[1]} à {time_part}"

            compare_path = self.resolve_path(value)
            run_dir = compare_path.parent
            # Check for PDFs: either archived locally (legacy) or referenced via source paths.
            has_pdfs = (run_dir / "previous_report.pdf").exists() and (run_dir / "current_report.pdf").exists()
            if not has_pdfs and payload is not None:
                pdf_paths = payload.get("pdf_paths") or {}
                src_prev = str(pdf_paths.get("pdf_previous") or "").strip()
                src_cur = str(pdf_paths.get("pdf_current") or "").strip()
                has_pdfs = bool(src_prev and Path(src_prev).exists() and src_cur and Path(src_cur).exists())
            pdf_icon = "✅" if has_pdfs else "⚠️"
            try:
                relative_parent = compare_path.relative_to(self.root_dir).parent.as_posix()
            except ValueError:
                relative_parent = compare_path.parent.as_posix()
            pretty_parent = relative_parent.replace("_", " ").replace("vs", " vs ")
            label = f"{pdf_icon} {bank_code.upper()} - {pretty_parent} ({timestamp_label})"
            filtered_options.append({"label": label, "value": value})

        filtered_options.sort(key=lambda item: str(item["value"]), reverse=True)
        return filtered_options

    def resolve_path(self, target: str | Path) -> Path:
        """Resout un chemin relatif ou absolu vers un chemin absolu.

        Args:
            target: Chemin relatif au repertoire racine ou chemin absolu.

        Returns:
            Chemin absolu resolu.
        """
        candidate = Path(target)
        if candidate.is_absolute():
            return candidate
        return self.root_dir / candidate

    def load_dash_payload(
        self,
        target: str | Path,
        *,
        source: str,
        source_label: str,
    ) -> ComparisonPayloadBundle | None:
        """Charge un artefact de comparaison et construit le bundle Dash.

        Args:
            target: Chemin relatif ou absolu vers le fichier ``comparison.json``.
            source: Identifiant interne de la source (ex. ``analyse_enregistree``).
            source_label: Libelle affiche a l'utilisateur pour la source.

        Returns:
            Bundle contenant le payload canonique, les metadonnees et les
            chemins PDF resolus, ou ``None`` si le fichier est invalide.
        """
        compare_path = self.resolve_path(target)
        if compare_path.name == "text_comparison.json":
            try:
                text_payload = json.loads(compare_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if not isinstance(text_payload, dict):
                return None
            canonical = _text_only_ui_payload(text_payload, compare_path)
            indicator_meta = dict(canonical.get("meta", {}))
            indicator_meta["compare_path"] = str(compare_path)
            indicator_meta["source"] = source
            indicator_meta["source_label"] = source_label
            indicator_meta["storage_backend"] = "fichier_local"
            pdf_paths = _normalize_pdf_paths_store(
                _pdf_paths_from_comparison_meta(indicator_meta, canonical)
            )
            indicator_meta["pdf_paths"] = pdf_paths
            canonical["meta"]["pdf_paths"] = pdf_paths
            warning = _missing_pdf_warning(pdf_paths)
            return {
                "compare_path": str(compare_path),
                "raw_data": canonical,
                "indicator_result": canonical,
                "indicator_meta": indicator_meta,
                "pdf_paths": pdf_paths,
                "warning": warning,
            }

        raw_data = load_comparison_result(compare_path)
        if not isinstance(raw_data, dict):
            return None

        if raw_data.get("result_type") == "metier_tableaux":
            pdf_paths = _normalize_pdf_paths_store(
                _pdf_paths_from_comparison_meta(
                    {
                        "compare_path": str(compare_path),
                        "source": source,
                        "source_label": source_label,
                    },
                    raw_data,
                )
            )
            indicator_result = raw_data
            indicator_meta = {
                "compare_path": str(compare_path),
                "source": source,
                "source_label": source_label,
                "storage_backend": "fichier_local",
                "pdf_paths": pdf_paths,
            }
            warning = _missing_pdf_warning(pdf_paths)
            return {
                "compare_path": str(compare_path),
                "raw_data": raw_data,
                "indicator_result": indicator_result,
                "indicator_meta": indicator_meta,
                "pdf_paths": pdf_paths,
                "warning": warning,
            }

        canonical = to_canonical_payload(raw_data)
        if not is_canonical_comparison(canonical):
            canonical = raw_data

        indicator_meta = dict(canonical.get("meta", {})) if isinstance(canonical, dict) else {}
        indicator_meta["compare_path"] = str(compare_path)
        indicator_meta["source"] = source
        indicator_meta["source_label"] = source_label
        indicator_meta["storage_backend"] = "fichier_local"
        pdf_paths = _normalize_pdf_paths_store(_pdf_paths_from_comparison_meta(indicator_meta, canonical))
        indicator_meta["pdf_paths"] = pdf_paths
        warning = _missing_pdf_warning(pdf_paths)
        return {
            "compare_path": str(compare_path),
            "raw_data": raw_data,
            "indicator_result": canonical,
            "indicator_meta": indicator_meta,
            "pdf_paths": pdf_paths,
            "warning": warning,
        }

    def load_review_state(
        self,
        compare_path: str | Path | None,
        *,
        username: str | None = None,
    ) -> dict[str, Any] | None:
        """Charge l'etat de revue persiste pour une comparaison donnee.

        Args:
            compare_path: Chemin vers le fichier de comparaison.
            username: Identifiant analyste pour la strategie multi-utilisateurs.

        Returns:
            Dictionnaire d'etat de revue ou ``None`` si introuvable.
        """
        return load_review_state(compare_path, username=username)

    def save_review_state(
        self,
        compare_path: str | Path | None,
        **kwargs: Any,
    ) -> Path | None:
        """Persiste l'etat de revue a cote du fichier de comparaison.

        Args:
            compare_path: Chemin vers le fichier de comparaison.
            **kwargs: Donnees d'etat de revue a sauvegarder.

        Returns:
            Chemin du fichier d'etat sauvegarde ou ``None`` en cas d'echec.
        """
        return save_review_state(compare_path, **kwargs)


def build_file_comparison_store(*, root_dir: str | Path | None = None) -> FileComparisonStore:
    """Retourne le magasin local supporte par fichiers utilise par Dash.

    Args:
        root_dir: Repertoire racine optionnel. Par defaut, utilise la
            configuration globale ``INDICATOR_COMPARISON_DIR``.

    Returns:
        Instance de ``FileComparisonStore`` configuree.
    """
    return FileComparisonStore(root_dir=root_dir)
