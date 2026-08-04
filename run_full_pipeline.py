#!/usr/bin/env python
"""Shim de compatibilite : pipeline complet."""

from __future__ import annotations

from vigie.pipelines.complet import main

if __name__ == "__main__":
    raise SystemExit(main())
