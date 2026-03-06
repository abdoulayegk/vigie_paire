"""I/O helpers for Dash workflows."""

from __future__ import annotations

import json
from pathlib import Path

from app.ui_config import INDICATOR_COMPARISON_DIR


def save_pdfs_to_temp(
    pdf_bytes_t1: bytes, pdf_bytes_t2: bytes, *, temp_dir: Path
) -> tuple[str, str]:
    """Persist uploaded PDF bytes into a temporary working directory."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_bytes_t1 or not pdf_bytes_t2:
        raise ValueError(
            "Les deux fichiers PDF (trimestre precedent et trimestre courant) sont requis."
        )

    path_t1 = temp_dir / "previous_upload.pdf"
    path_t2 = temp_dir / "current_upload.pdf"

    path_t1.write_bytes(pdf_bytes_t1)
    path_t2.write_bytes(pdf_bytes_t2)

    return str(path_t1), str(path_t2)


def load_comparison_result(path: str | Path) -> dict | None:
    """Load a comparison JSON payload from disk."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_available_indicator_comparison_options() -> list[dict[str, str]]:
    """Return dropdown options for available comparison JSON files."""
    if not INDICATOR_COMPARISON_DIR.exists():
        return []

    files = sorted(
        INDICATOR_COMPARISON_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [{"label": path.name, "value": path.name} for path in files]


def get_available_comparisons() -> list[dict[str, str]]:
    """Backward-compatible alias used by older UI variants."""
    return get_available_indicator_comparison_options()
