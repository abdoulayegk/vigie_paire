"""Dash entrypoint compatibility module.

This keeps ``python -m app.app`` working while the code lives in
``vigilance.dash_app``.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vigilance.dash_app.app import app


if __name__ == "__main__":
    debug = os.getenv("DASH_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("DASH_PORT", "8050"))
    app.run(debug=debug, use_reloader=debug, port=port)
