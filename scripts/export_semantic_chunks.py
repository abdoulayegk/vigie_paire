#!/usr/bin/env python
"""Exporte un audit Markdown des chunks sémantiques d'une extraction canonique."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vigilance.text_analysis.chunking import TextChunk, _chunk_subsection_text
from vigilance.text_analysis.constants import _SECTION_LABELS
from vigilance.text_analysis.markdown import _extract_section_text_from_markdown
from vigilance.text_analysis.openai_client import _build_openai_client
from vigilance.text_analysis.subsection_matching import _parse_subsections


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère un Markdown lisible contenant les chunks sémantiques d'un rapport."
    )
    parser.add_argument("--markdown", required=True, type=Path, help="Extraction Markdown source.")
    parser.add_argument("--output", required=True, type=Path, help="Fichier d'audit à écrire.")
    parser.add_argument(
        "--section-key",
        action="append",
        choices=sorted(_SECTION_LABELS),
        help="Section à exporter; répétable. Toutes les sections présentes par défaut.",
    )
    parser.add_argument(
        "--subsection",
        help="Sous-section exacte à exporter. Nécessite une seule --section-key.",
    )
    parser.add_argument("--model", default="gpt-4o", help="Modèle pour les frontières ambiguës.")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Modèle d'embeddings.",
    )
    return parser


def _format_chunk(chunk: TextChunk) -> str:
    path = f" | {chunk.hierarchy_path}" if chunk.hierarchy_path else ""
    return f"[{chunk.chunk_id} | {chunk.kind}{path}]\n\n{chunk.text}"


def _export_chunks(
    *,
    markdown: str,
    section_keys: list[str],
    subsection_filter: str | None,
    client,
    semantic_model: str,
    embedding_model: str,
) -> tuple[str, int]:
    blocks: list[str] = []
    chunk_count = 0
    for section_key in section_keys:
        section_text = _extract_section_text_from_markdown(markdown, section_key)
        if not section_text:
            continue
        section_title = _SECTION_LABELS[section_key]
        section_blocks: list[str] = []
        for subsection_heading, body in _parse_subsections(section_text):
            display_heading = "Introduction" if subsection_heading == "__intro__" else subsection_heading
            if subsection_filter and display_heading != subsection_filter:
                continue
            chunks = _chunk_subsection_text(
                body,
                subsection_heading=display_heading,
                section_title=section_title,
                client=client,
                embedding_model=embedding_model,
                semantic_model=semantic_model,
            )
            if not chunks:
                continue
            section_blocks.append(f"## {display_heading}\n\n" + "\n\n".join(map(_format_chunk, chunks)))
            chunk_count += len(chunks)
        if section_blocks:
            blocks.append(f"# {section_title}\n\n" + "\n\n".join(section_blocks))
    return "\n\n".join(blocks).rstrip() + "\n", chunk_count


def main() -> int:
    """Exécute l'export demandé par la ligne de commande."""
    args = _build_parser().parse_args()
    if args.subsection and (not args.section_key or len(args.section_key) != 1):
        raise SystemExit("--subsection exige exactement une valeur --section-key.")
    if not args.markdown.is_file():
        raise FileNotFoundError(f"Extraction Markdown introuvable: {args.markdown}")

    section_keys = args.section_key or list(_SECTION_LABELS)
    client = _build_openai_client()
    audit, chunk_count = _export_chunks(
        markdown=args.markdown.read_text(encoding="utf-8"),
        section_keys=section_keys,
        subsection_filter=args.subsection,
        client=client,
        semantic_model=args.model,
        embedding_model=args.embedding_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(audit, encoding="utf-8")
    print(f"Audit écrit: {args.output} ({chunk_count} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
