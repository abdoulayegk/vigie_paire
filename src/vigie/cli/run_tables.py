r"""Point d'entree CLI pour l'extraction de tableaux sur des plages de pages pre-detectees.

Ce module constitue la deuxième étape du pipeline d'extraction en ligne de
commande. Il prend en entrée un fichier ``section_ranges.json`` (produit par
``run_ranges.py``) et extrait les tableaux financiers des pages correspondantes
via Docling (et optionnellement GPT-4o Vision).

Les tableaux extraits sont convertis en ``TableArtifact`` canoniques et écrits
dans un fichier ``tables_docling.json``. Avec ``--save-extraction``, les memes
artefacts que l'app (``tables.json``, ``indicators.json``, ``footnotes.json``,
etc.) sont aussi ecrits sous ``--extraction-root``. Optionnellement, un fichier
``vigie_extract_v1.json`` peut egalement etre produit (format d'export enrichi).

Usage typique
-------------
.. code-block:: bash

    python -m vigie.cli.run_tables \\
        --banque rbc \\
        --pdf /data/rbc_q1_2025.pdf \\
        --trimestre t1-2025 \\
        --ranges_json outputs/runs/t1-2025/rbc/section_ranges.json \\
        --sortie outputs/runs \\
        --vigie_extract

Le chemin du fichier JSON produit est affiché sur la sortie standard.

Formats JSON de plages acceptés
--------------------------------
La fonction ``_load_section_ranges`` reconnaît trois formats différents pour
le fichier ``section_ranges.json`` (rétrocompatibilité) :
- Clé ``section_ranges`` avec champs ``section``, ``start``, ``end``.
- Clé ``ranges`` avec champs ``section``, ``start_page_pdf``, ``end_page_pdf``.
- Clé ``sections`` (dictionnaire) avec champs ``section``, ``start_page``, ``end_page``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from vigie.support.config.loader import get_bank_cfg, load_config
from vigie.support.models.table_models import (
    VISION_CONTENT_SOURCE,
    TableArtifact,
    infer_content_source,
)
from vigie.support.report.export_json import write_tables_docling
from vigie.support.utils.footnotes_utils import normalize_footnotes_to_canonical
from vigie.support.utils.indicator_cleaner import (
    dedupe_indicators,
    merge_line_split_indicators,
    normalize_indicator_for_comparison,
    post_normalize_indicator,
)
from vigie.support.utils.rbc_table_signals import (
    build_rbc_first_column_signals,
    classify_rbc_title_reliability,
    is_rbc_bank,
)

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/runs"


def _canonicalize_section(raw: str | None) -> str | None:
    """Normaliser un nom de section brut pour l'export des tableaux extraits.

    Tente d'utiliser ``canonicalize_section`` depuis le module de taxonomie.
    En cas d'échec, convertit la chaîne en identifiant ``snake_case``.
    Retourne ``None`` si la chaîne d'entrée est ``None`` ou vide après nettoyage.
    """
    if raw is None:
        return None
    try:
        from vigie.extraction.section_taxonomy import canonicalize_section

        return canonicalize_section(raw)
    except Exception:
        fallback = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        return fallback or None


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI d'extraction des tableaux sur plages ciblées.

    Arguments obligatoires
    ----------------------
    --banque:
        Code identifiant la banque (ex. ``rbc``).
    --pdf:
        Chemin vers le fichier PDF à extraire.
    --trimestre:
        Libellé du trimestre (ex. ``t1-2025``).
    --ranges_json:
        Chemin vers le fichier ``section_ranges.json`` produit par ``run_ranges``.

    Arguments optionnels
    --------------------
    --config:
        Chemin vers le fichier YAML de configuration. Défaut :
        ``configs/bank_profiles.yaml``.
    --sortie:
        Répertoire racine pour les fichiers de sortie. Défaut : ``outputs/runs``.
    --vigie_extract:
        Si présent, produit également un fichier ``vigie_extract_v1.json`` en
        plus du fichier ``tables_docling.json``.
    --language:
        Code de langue pour le fichier ``vigie_extract_v1`` (défaut : ``fr``).
    """
    parser = argparse.ArgumentParser(
        description="Extraire les tableaux sur des plages de pages selectionnees."
    )
    parser.add_argument("--banque", required=True, help="Code banque (ex: rbc)")
    parser.add_argument("--pdf", required=True, help="Chemin du PDF d'entree")
    parser.add_argument("--trimestre", required=True, help="Libelle trimestre (ex: t1-2025)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Chemin YAML de configuration")
    parser.add_argument(
        "--ranges_json", required=True, help="Chemin vers section_ranges.json"
    )
    parser.add_argument(
        "--sortie", default=DEFAULT_OUT_ROOT, help="Racine des sorties"
    )
    parser.add_argument(
        "--vigie_extract",
        action="store_true",
        default=False,
        help="Also produce a single vigie_extract_v1 JSON per PDF",
    )
    parser.add_argument(
        "--language", default="fr", help="Language code for vigie_extract (default: fr)"
    )
    parser.add_argument(
        "--save-extraction",
        action="store_true",
        default=False,
        help=(
            "Also call save_extraction: writes tables.json, indicators.json, "
            "footnotes.json, report_summary.* under --racine-extraction (same layout as the app)."
        ),
    )
    parser.add_argument(
        "--racine-extraction",
        default="outputs/extractions",
        help="Racine lors de --save-extraction (defaut: outputs/extractions)",
    )
    return parser


def _infer_year(quarter: str) -> int:
    r"""Déduire l'année depuis le libellé de trimestre, avec repli sur 2025.

    Recherche un motif ``(19|20)\\d{2}`` dans la chaîne fournie.
    Retourne 2025 si aucune année n'est trouvée.

    Exemple : ``_infer_year("t1-2025")`` → ``2025``.
    """
    match = re.search(r"(19|20)\d{2}", quarter)
    return int(match.group(0)) if match else 2025


def _load_section_ranges(path: str | Path) -> list[dict[str, Any]]:
    """Charger des plages de sections depuis les formats JSON reconnus par le projet.

    Supporte trois formats de fichier pour assurer la rétrocompatibilité :

    1. **Format ``section_ranges``** : liste sous la clé ``section_ranges`` avec
       les champs ``section``, ``start``, ``end``.
    2. **Format ``ranges``** : liste sous la clé ``ranges`` avec les champs
       ``section``, ``start_page_pdf``, ``end_page_pdf``.
    3. **Format ``sections``** : dictionnaire sous la clé ``sections`` avec les
       champs ``section``, ``start_page``, ``end_page``.

    Les entrées invalides (section vide, page de début ≤ 0, fin < début) sont
    silencieusement ignorées.

    Lève ``ValueError`` si aucune plage valide n'est trouvée dans le fichier.

    Paramètres
    ----------
    path:
        Chemin vers le fichier JSON des plages de sections.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    section_ranges: list[dict[str, Any]] = []

    ranges = data.get("section_ranges")
    if isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, dict):
                continue
            section = _canonicalize_section(str(item.get("section", "")).strip())
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", start) or start)
            if section and start > 0 and end >= start:
                entry: dict[str, Any] = {"section": section, "start": start, "end": end}
                if item.get("evidence"):
                    entry["evidence"] = item["evidence"]
                section_ranges.append(entry)

    ranges = data.get("ranges")
    if not section_ranges and isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, dict):
                continue
            section = _canonicalize_section(str(item.get("section", "")).strip())
            start = int(item.get("start_page_pdf", 0) or 0)
            end = int(item.get("end_page_pdf", start) or start)
            if section and start > 0 and end >= start:
                entry = {"section": section, "start": start, "end": end}
                if item.get("evidence"):
                    entry["evidence"] = item["evidence"]
                section_ranges.append(entry)

    if not section_ranges and isinstance(data.get("sections"), dict):
        for section_name, item in data["sections"].items():
            if not isinstance(item, dict):
                continue
            section = _canonicalize_section(
                str(item.get("section", section_name)).strip()
            )
            start = int(item.get("start_page", 0) or 0)
            end = int(item.get("end_page", start) or start)
            if section and start > 0 and end >= start:
                entry = {"section": section, "start": start, "end": end}
                if item.get("evidence"):
                    entry["evidence"] = item["evidence"]
                section_ranges.append(entry)

    if not section_ranges:
        raise ValueError(f"No valid section ranges found in {path}")
    return section_ranges


def _to_artifacts(
    raw_tables: list[Any], bank: str, quarter: str, pdf_path: str
) -> list[TableArtifact]:
    """Convertir les tables extraites (Docling) en ``TableArtifact`` canoniques d'export.

    Pour chaque table brute, cette fonction :
    1. Détermine la source de contenu (Docling ou Vision GPT-4o).
    2. Pour les banques RBC avec indicateurs Vision, calcule les signaux de
       hiérarchie de la première colonne (``build_rbc_first_column_signals``).
    3. Fusionne les indicateurs coupés sur plusieurs lignes et déduplique.
    4. Normalise chaque indicateur pour la comparaison.
    5. Normalise les notes de bas de page vers le format canonique.
    6. Construit et retourne un ``TableArtifact`` avec toutes les métadonnées.

    Paramètres
    ----------
    raw_tables:
        Liste d'objets ``ExtractedTable`` retournés par Docling.
    bank:
        Code banque (ex. ``"rbc"``).
    quarter:
        Libellé du trimestre (ex. ``"t1-2025"``).
    pdf_path:
        Chemin vers le PDF source, inclus dans chaque artefact pour traçabilité.
    """
    artifacts: list[TableArtifact] = []
    for index, table in enumerate(raw_tables, start=1):
        extraction_method = getattr(table, "extraction_method", None) or "docling"
        content_source = infer_content_source(
            extraction_method,
            getattr(table, "content_source", None),
        )
        raw_values = getattr(table, "first_column_indicators_raw", None)
        vision_raw_indicators = (
            [str(x).strip() for x in raw_values if str(x).strip()]
            if raw_values is not None
            else []
        )
        if content_source != VISION_CONTENT_SOURCE:
            vision_raw_indicators = []
        rows = [list(row) for row in (getattr(table, "rows", []) or [])]
        first_column_groups: list[str] | None = None
        hierarchical_indicator_signature: list[str] | None = None
        if is_rbc_bank(bank) and vision_raw_indicators:
            rbc_signals = build_rbc_first_column_signals(
                rows=rows,
                raw_indicators=vision_raw_indicators,
            )
            first_column_groups = list(rbc_signals.groups_raw)
            hierarchical_indicator_signature = list(
                rbc_signals.hierarchical_indicator_signature
            )

        vision_raw_indicators, _, _ = dedupe_indicators(
            merge_line_split_indicators(vision_raw_indicators)[0]
        )
        comparison_normalized_indicators: list[str] = []
        for item in vision_raw_indicators:
            normalized, _, _ = post_normalize_indicator(
                normalize_indicator_for_comparison(item)
            )
            if normalized:
                comparison_normalized_indicators.append(normalized)

        # comparison_blockers are recomputed by TableArtifact.__post_init__
        footnotes_raw = getattr(table, "footnotes", None)
        if footnotes_raw is None:
            canonical_footnotes = None
        else:
            canonical_footnotes = normalize_footnotes_to_canonical(footnotes_raw)

        section = (
            _canonicalize_section(getattr(table, "section", None)) or "unknown_section"
        )
        artifacts.append(
            TableArtifact(
                bank_code=bank,
                section=section,
                page_pdf=int(getattr(table, "page_number", 0) or 0),
                table_id=str(getattr(table, "table_id", f"table_{index}")),
                title=getattr(table, "title", None),
                headers=list(getattr(table, "headers", []) or []),
                rows=rows,
                first_column_indicators=comparison_normalized_indicators,
                first_column_indicators_raw=vision_raw_indicators,
                first_column_groups=first_column_groups,
                hierarchical_indicator_signature=hierarchical_indicator_signature,
                table_number=getattr(table, "table_number", None),
                bbox=getattr(table, "bbox", None),
                table_index_on_page=getattr(table, "table_index_on_page", None),
                tables_on_page=getattr(table, "tables_on_page", None),
                bbox_top=getattr(table, "bbox_top", None),
                page_local_role=getattr(table, "page_local_role", None),
                extraction_method=extraction_method,
                quarter=quarter,
                pdf_path=pdf_path,
                title_clean=getattr(table, "title_clean", None),
                table_summary=getattr(table, "table_summary", None),
                title_raw=getattr(table, "title_raw", None),
                row_count=len(vision_raw_indicators) or len(rows),
                title_reliability=classify_rbc_title_reliability(
                    getattr(table, "title_clean", None)
                    or getattr(table, "title", None),
                    bank_code=bank,
                ),
                footnotes=canonical_footnotes,
                debug_metrics=dict(getattr(table, "debug_metrics", {}) or {}),
                content_source=content_source,
                extraction_status=str(
                    getattr(table, "extraction_status", "ok") or "ok"
                ),
            )
        )
    return artifacts


def main(argv: list[str] | None = None) -> None:
    """Extraire les tableaux sur les plages fournies puis écrire les artefacts JSON.

    Étapes exécutées
    ----------------
    1. Chargement et validation de la configuration YAML.
    2. Chargement des plages de sections depuis ``--ranges_json``.
    3. Import du backend Docling (``extract_tables_docling_by_sections``). Lève
       ``NotImplementedError`` si Docling n'est pas disponible.
    4. Extraction des tableaux sur les plages de pages ciblées.
    5. Conversion des tables brutes en ``TableArtifact`` canoniques.
    6. Écriture du fichier ``tables_docling.json`` dans
       ``<out_root>/<quarter>/<bank>/``.
    7. Si ``--vigie_extract`` est activé, production d'un fichier
       ``vigie_extract_v1.json`` supplémentaire.
    8. Affichage du/des chemin(s) des fichiers produits sur la sortie standard.

    Paramètres
    ----------
    argv:
        Liste d'arguments CLI. Si ``None``, utilise ``sys.argv[1:]``.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.banque)
    section_ranges = _load_section_ranges(args.ranges_json)

    try:
        from vigie.extraction.docling.processor import (
            extract_tables_docling_by_sections,
        )
    except Exception as exc:
        raise NotImplementedError(
            "Docling extraction backend from extraction/ is not importable in this environment."
        ) from exc

    year = _infer_year(args.trimestre)
    raw_tables = extract_tables_docling_by_sections(
        pdf_path=args.pdf,
        bank_code=args.banque,
        quarter=args.trimestre,
        year=year,
        section_ranges=section_ranges,
    )
    artifacts = _to_artifacts(
        raw_tables, bank=args.banque, quarter=args.trimestre, pdf_path=args.pdf
    )
    out_dir = Path(args.sortie) / args.trimestre / args.banque
    out_path = write_tables_docling(out_dir=out_dir, tables=artifacts)
    print(out_path)

    if args.vigie_extract:
        from vigie.support.report.vigie_extract_schema import (
            build_vigie_extract,
            write_vigie_extract,
        )

        payload = build_vigie_extract(
            pdf_path=args.pdf,
            bank_code=args.banque,
            quarter=args.trimestre,
            year=year,
            language=args.language,
            section_ranges=section_ranges,
            tables=raw_tables,
        )
        vigie_path = write_vigie_extract(out_dir=out_dir, payload=payload)
        print(vigie_path)

    if args.save_extraction:
        from vigie.extraction.extraction_storage import (
            get_extraction_artifact_paths,
            save_extraction,
        )

        extraction_root = Path(args.racine_extraction)
        extraction_method = "docling"
        for art in artifacts:
            em = getattr(art, "extraction_method", None)
            if em:
                extraction_method = str(em)
                break
        meta: dict[str, Any] = {
            "pdf_path": str(Path(args.pdf).resolve()),
            "section_ranges": section_ranges,
            "extraction_method": extraction_method,
        }
        save_extraction(
            bank_code=args.banque,
            year=year,
            quarter=args.trimestre,
            tables=artifacts,
            meta=meta,
            base_dir=extraction_root,
        )
        paths = get_extraction_artifact_paths(
            args.banque, year, args.trimestre, extraction_root
        )
        print(paths["indicators"])
        print(paths["footnotes"])


if __name__ == "__main__":
    main()
