"""CLI pour l'extraction de blocs de texte par section depuis un PDF.

Utilise Docling pour extraire les paragraphes de texte des sections ciblées
(gestion_capital, gestion_risques, gestion_reglementation) en excluant
tous les tableaux et leurs zones de proximité.

Usage:
    python -m vigilance.cli.run_text_extract --bank BNS --year 2025 --T2 --pdf path/to/report.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/bank_profiles.yaml"
DEFAULT_OUT_ROOT = "outputs/text_extractions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text blocks (paragraphs) from targeted sections of a bank report PDF."
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. bns, bnc, rbc)")
    parser.add_argument("--year", required=True, type=int, help="Report year (e.g. 2025)")
    quarter_group = parser.add_mutually_exclusive_group(required=True)
    quarter_group.add_argument("--T1", dest="quarter_flag", action="store_const", const="T1", help="Current report quarter T1")
    quarter_group.add_argument("--T2", dest="quarter_flag", action="store_const", const="T2", help="Current report quarter T2")
    quarter_group.add_argument("--T3", dest="quarter_flag", action="store_const", const="T3", help="Current report quarter T3")
    quarter_group.add_argument("--T4", dest="quarter_flag", action="store_const", const="T4", help="Current report quarter T4")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML bank profiles config")
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return parser


# Mapping from canonical section keys (section_taxonomy) → text pipeline keys
_CANONICAL_TO_TEXT_KEY: dict[str, str] = {
    "capital_management": "gestion_capital",
    "risk_management": "gestion_risques",
    "regulatory_updates": "gestion_reglementation",
    # French names pass through unchanged
    "gestion_capital": "gestion_capital",
    "gestion_risques": "gestion_risques",
    "gestion_reglementation": "gestion_reglementation",
}


def _get_section_ranges_from_locator(
    pdf_path: Path,
    bank_code: str,
    year: int,
    quarter: str,
) -> list[dict]:
    """Utilise section_locator pour obtenir les plages de pages par section.

    Convertit les clés canoniques (capital_management, etc.) vers les clés
    du pipeline texte (gestion_capital, etc.).
    """
    try:
        from vigilance.extraction.section_locator import locate_sections_in_pdf
        mapping = locate_sections_in_pdf(
            pdf_path=pdf_path,
            bank_code=bank_code,
            quarter=quarter,
            year=year,
        )
        ranges = []
        for located in getattr(mapping, "sections", []) or []:
            start = int(getattr(located, "start_page", 0) or 0)
            if start <= 0:
                continue
            end = int(getattr(located, "end_page", start) or start)
            if end < start:
                end = start
            section_raw = str(getattr(located, "section_type", "") or "")
            # Canonicalize via taxonomy first
            try:
                from vigilance.extraction.section_taxonomy import canonicalize_section
                section_raw = canonicalize_section(section_raw) or section_raw
            except Exception:
                pass
            # Map to text pipeline key
            section = _CANONICAL_TO_TEXT_KEY.get(section_raw, section_raw)
            if section:
                ranges.append({
                    "section": section,
                    "start": start,
                    "end": end,
                    "anchor_page": getattr(located, "anchor_page", None),
                    "anchor_text": getattr(located, "anchor_text", None),
                    "anchor_bbox_norm": getattr(located, "anchor_bbox_norm", None),
                    "anchor_found": bool(getattr(located, "anchor_found", False)),
                })
        return ranges
    except Exception as exc:
        logger.error("Erreur lors de la localisation des sections : %s", exc)
        return []


def _get_target_sections_for_bank(bank_code: str, config_path: str) -> set[str]:
    """Retourne les sections texte autorisées pour cette banque depuis bank_profiles.yaml.

    Mappe les clés canoniques (capital_management, etc.) vers les clés texte
    (gestion_capital, etc.). Si le config est inaccessible, retourne les 3 sections.
    """
    try:
        import yaml
        cfg = yaml.safe_load(open(config_path))
        bank_cfg = cfg.get("banks", {}).get(bank_code.lower(), {})
        canonical_sections = set(bank_cfg.get("sections", {}).keys())
        if not canonical_sections:
            raise ValueError("Sections vides")
        return {
            _CANONICAL_TO_TEXT_KEY[s]
            for s in canonical_sections
            if s in _CANONICAL_TO_TEXT_KEY
        }
    except Exception as exc:
        logger.warning(
            "Impossible de lire les sections pour %s depuis %s (%s) — utilisation des 3 sections par défaut.",
            bank_code.upper(), config_path, exc,
        )
        return {"gestion_capital", "gestion_risques", "gestion_reglementation"}


def main(argv: list[str] | None = None) -> int:
    """Extraire les blocs de texte d'un rapport bancaire."""
    args = build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from vigilance.cli.quarter_logic import normalize_quarter

    bank_code = args.bank.lower()
    year = args.year
    quarter = normalize_quarter(args.quarter_flag)
    pdf_path = Path(args.pdf)
    out_root = Path(args.out_root)

    if not pdf_path.exists():
        logger.error("PDF introuvable : %s", pdf_path)
        return 1

    # Locate sections
    logger.info("Localisation des sections dans %s ...", pdf_path.name)
    section_ranges = _get_section_ranges_from_locator(pdf_path, bank_code, year, quarter)

    if not section_ranges:
        logger.warning(
            "Aucune section détectée dans %s. "
            "Vérifier que bank_profiles.yaml contient les sections cibles pour %s.",
            pdf_path.name, bank_code.upper(),
        )
        return 1

    # Lire les sections autorisées pour cette banque depuis bank_profiles.yaml
    target_sections = _get_target_sections_for_bank(bank_code, args.config)
    section_ranges = [r for r in section_ranges if r.get("section") in target_sections]

    if not section_ranges:
        logger.warning(
            "Aucune section cible trouvée pour %s %s %s. Sections autorisées : %s",
            bank_code.upper(), year, quarter.upper(), sorted(target_sections),
        )
        return 1

    logger.info(
        "Sections détectées : %s",
        [f"{r['section']} (p{r['start']}-{r['end']})" for r in section_ranges],
    )

    # Extract text blocks
    from vigilance.text_extraction import extract_text_blocks_by_sections
    from vigilance.text_extraction.text_extraction_writer import (
        get_text_extraction_path,
        write_text_extraction,
    )

    logger.info("Extraction des blocs de texte depuis %s ...", pdf_path.name)
    blocks = extract_text_blocks_by_sections(
        pdf_path=pdf_path,
        bank_code=bank_code,
        quarter=quarter,
        year=year,
        section_ranges=section_ranges,
    )

    if not blocks:
        logger.warning("Aucun bloc de texte extrait depuis %s.", pdf_path.name)
        return 1

    # Write output
    out_dir = get_text_extraction_path(out_root, bank_code, year, quarter).parent
    out_path = write_text_extraction(
        blocks=blocks,
        out_dir=out_dir,
        bank_code=bank_code,
        year=year,
        quarter=quarter,
        source_pdf=str(pdf_path),
    )

    logger.info("Extraction terminée : %d blocs → %s", len(blocks), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
