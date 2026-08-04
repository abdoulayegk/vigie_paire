#!/usr/bin/env python
"""Shim de compatibilite : pipeline indicateurs."""

from __future__ import annotations

from vigie.pipelines.indicateurs import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
