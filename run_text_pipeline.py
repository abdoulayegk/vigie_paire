#!/usr/bin/env python
"""Shim de compatibilite : pipeline texte."""

from __future__ import annotations

from vigie.pipelines.texte import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
