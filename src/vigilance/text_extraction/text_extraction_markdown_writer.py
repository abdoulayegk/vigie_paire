from __future__ import annotations

from pathlib import Path


def get_text_extraction_markdown_path(out_dir: Path, quarter_label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"text_extraction_{quarter_label.lower()}.md"


def write_text_extraction_markdown(content: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path
