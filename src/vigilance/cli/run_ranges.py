"""
Façade CLI pour la détection des plages de sections dans un PDF bancaire.

Ce module constitue la première étape du pipeline d'extraction en ligne de
commande. Il analyse un rapport PDF pour identifier les pages correspondant
à chaque section réglementaire (gestion du capital, gestion des risques, etc.)
et écrit le résultat dans un fichier ``section_ranges.json`` canonique.

Ce fichier JSON est ensuite utilisé comme entrée par ``run_tables.py`` pour
cibler l'extraction Docling sur les bonnes pages.

Usage typique
-------------
.. code-block:: bash

    python -m vigilance.cli.run_ranges \\
        --bank rbc \\
        --pdf /data/rbc_q1_2025.pdf \\
        --quarter t1-2025 \\
        --out_root outputs/runs

Le chemin du fichier JSON produit est affiché sur la sortie standard.

Pipeline complet
----------------
1. ``run_ranges`` → ``section_ranges.json``
2. ``run_tables`` (utilise ``section_ranges.json``) → ``tables_docling.json``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from vigilance.config.loader import get_bank_cfg, load_config
from vigilance.models.section_models import SectionRange, SectionRangesResult
from vigilance.report.export_json import write_section_ranges

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/runs"


def _canonicalize_section(raw: str) -> str:
    """Normaliser un nom de section brut vers la taxonomie canonique de sortie.

    Tente d'utiliser ``canonicalize_section`` depuis le module de taxonomie.
    En cas d'échec, convertit la chaîne en identifiant ``snake_case`` en
    remplaçant les caractères non alphanumériques par des underscores.

    Exemple : ``"Gestion du Capital"`` → ``"gestion_capital"``
    """
    try:
        from vigilance.extraction.section_taxonomy import canonicalize_section

        return canonicalize_section(raw)
    except Exception:
        fallback = re.sub(r"[^a-z0-9]+", "_", (raw or "").lower()).strip("_")
        return fallback


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de détection des plages de sections.

    Arguments obligatoires
    ----------------------
    --bank:
        Code identifiant la banque (ex. ``rbc``).
    --pdf:
        Chemin vers le fichier PDF à analyser.
    --quarter:
        Libellé du trimestre (ex. ``t1-2025``).

    Arguments optionnels
    --------------------
    --config:
        Chemin vers le fichier YAML de configuration. Défaut :
        ``configs/bank_profiles.yaml``.
    --out_root:
        Répertoire racine pour les fichiers de sortie. Le fichier JSON sera
        écrit dans ``<out_root>/<quarter>/<bank>/section_ranges.json``.
        Défaut : ``outputs/runs``.
    """
    parser = argparse.ArgumentParser(description="Detect section ranges for one PDF.")
    parser.add_argument("--bank", required=True, help="Bank code (e.g. rbc)")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--quarter", required=True, help="Quarter label (e.g. t1-2025)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument(
        "--out_root", default=DEFAULT_OUT_ROOT, help="Output root directory"
    )
    return parser


def _to_result(
    mapping: Any, bank_code: str, quarter: str, pdf_path: str
) -> SectionRangesResult:
    """Transformer la sortie brute du locator en ``SectionRangesResult`` exportable.

    Parcourt les sections détectées par ``locate_sections_in_pdf``, filtre les
    entrées sans page de début valide (≤ 0), et construit des objets ``SectionRange``
    avec les métadonnées de détection (méthode, confiance, titre trouvé, etc.).

    Paramètres
    ----------
    mapping:
        Objet retourné par ``locate_sections_in_pdf``, avec un attribut
        ``sections`` contenant la liste des sections détectées.
    bank_code:
        Code banque (ex. ``"rbc"``).
    quarter:
        Libellé du trimestre (ex. ``"t1-2025"``).
    pdf_path:
        Chemin vers le PDF source, inclus dans le résultat pour traçabilité.

    Retourne
    --------
    Un ``SectionRangesResult`` prêt à être sérialisé en JSON par
    ``write_section_ranges``.
    """
    ranges: list[SectionRange] = []
    for located in getattr(mapping, "sections", []):
        start_page = int(getattr(located, "start_page", 0) or 0)
        if start_page <= 0:
            continue
        end_page = int(getattr(located, "end_page", start_page) or start_page)
        ranges.append(
            SectionRange(
                section=_canonicalize_section(
                    str(getattr(located, "section_type", ""))
                ),
                start_page_pdf=start_page,
                end_page_pdf=end_page,
                method=str(getattr(located, "detection_method", "")),
                confidence=float(getattr(located, "confidence", 0.0) or 0.0),
                evidence={
                    "title_found": getattr(located, "title_found", ""),
                    "end_detection_method": getattr(
                        located, "end_detection_method", ""
                    ),
                    "detected_span": getattr(located, "detected_span", None),
                    "final_span": getattr(located, "final_span", None),
                    "constraint_applied": getattr(located, "constraint_applied", False),
                    "constraint_reason": getattr(located, "constraint_reason", ""),
                },
            )
        )
    return SectionRangesResult(
        bank_code=bank_code, quarter=quarter, pdf_path=pdf_path, ranges=ranges
    )


def main(argv: list[str] | None = None) -> None:
    """Détecter les sections d'un PDF puis écrire le JSON canonique des plages.

    Étapes exécutées
    ----------------
    1. Chargement et validation de la configuration YAML (banque reconnue).
    2. Import du backend de détection (``locate_sections_in_pdf``). Lève
       ``NotImplementedError`` si le module d'extraction n'est pas disponible
       dans l'environnement courant.
    3. Exécution de la détection de sections sur le PDF.
    4. Conversion du résultat brut en ``SectionRangesResult`` canonique.
    5. Écriture du fichier ``section_ranges.json`` dans
       ``<out_root>/<quarter>/<bank>/``.
    6. Affichage du chemin du fichier produit sur la sortie standard.

    Paramètres
    ----------
    argv:
        Liste d'arguments CLI. Si ``None``, utilise ``sys.argv[1:]``.
    """
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.bank)

    try:
        from vigilance.extraction.section_locator import locate_sections_in_pdf
    except Exception as exc:
        raise NotImplementedError(
            "Section detection backend from extraction/ is not importable in this environment."
        ) from exc

    mapping = locate_sections_in_pdf(
        args.pdf, bank_code=args.bank, quarter=args.quarter
    )
    result = _to_result(
        mapping, bank_code=args.bank, quarter=args.quarter, pdf_path=args.pdf
    )
    out_dir = Path(args.out_root) / args.quarter / args.bank
    out_path = write_section_ranges(out_dir=out_dir, result=result)
    print(out_path)


if __name__ == "__main__":
    main()
