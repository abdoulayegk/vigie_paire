from __future__ import annotations

import pytest

from vigie.pipelines import indicateurs


def test_indicateurs_parser_accepts_forcer_extraction() -> None:
    parser = indicateurs.build_parser()
    args = parser.parse_args(["--banque", "BNC", "--annee", "2025", "--T2", "--forcer-extraction"])
    assert args.forcer_extraction is True
    assert args.sans_extraction is False


def test_indicateurs_rejects_forcer_and_sans_extraction() -> None:
    with pytest.raises(SystemExit):
        indicateurs.main(
            [
                "--banque",
                "BNC",
                "--annee",
                "2025",
                "--T2",
                "--sans-extraction",
                "--forcer-extraction",
            ]
        )
