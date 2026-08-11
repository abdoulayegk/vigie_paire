"""Tests for strict intra-section audit report generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_build_report():
    script_path = Path("scripts/audit_intra_section.py")
    spec = importlib.util.spec_from_file_location("audit_intra_section", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load audit_intra_section script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_report


def test_build_report_detects_clean_payload(tmp_path: Path) -> None:
    build_report = _load_build_report()
    payload = json.loads((Path("tests/fixtures/strict_intra_section_sample.json")).read_text(encoding="utf-8"))
    report = build_report(payload)
    assert report["summary"]["cross_section_pairs"] == 0
    assert report["summary"]["unknown_matched_pairs"] == 0
    assert report["status"]["strict_intra_section_ok"] is True
    assert report["status"]["unknown_never_matched_ok"] is True
