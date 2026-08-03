from __future__ import annotations

from vigilance.utils import pymupdf_utils


class _FakeTools:
    def __init__(self) -> None:
        self.warning_calls: list[bool] = []
        self.reset_calls = 0

    def mupdf_display_warnings(self, on=None) -> None:
        self.warning_calls.append(bool(on))

    def reset_mupdf_warnings(self) -> None:
        self.reset_calls += 1


class _FakePyMuPDF:
    def __init__(self) -> None:
        self.TOOLS = _FakeTools()


def test_configure_mupdf_runtime_disables_warnings_once() -> None:
    pymupdf_module = _FakePyMuPDF()
    original = pymupdf_utils._MUPDF_CONFIGURED
    pymupdf_utils._MUPDF_CONFIGURED = False
    try:
        pymupdf_utils.configure_mupdf_runtime(pymupdf_module)
        pymupdf_utils.configure_mupdf_runtime(pymupdf_module)

        assert pymupdf_module.TOOLS.warning_calls == [False]
        assert pymupdf_module.TOOLS.reset_calls == 1
    finally:
        pymupdf_utils._MUPDF_CONFIGURED = original
