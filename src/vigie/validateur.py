"""Facade publique du validateur multiplateforme sans LLM."""

from __future__ import annotations

from vigilance.dash_app.validator import main


if __name__ == "__main__":
    raise SystemExit(main())
