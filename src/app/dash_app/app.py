"""Compatibility Dash entrypoint wrapper."""

from __future__ import annotations

import os

from vigilance.dash_app.app import app


if __name__ == "__main__":
    debug = os.getenv("DASH_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("DASH_PORT", "8050"))
    app.run(debug=debug, use_reloader=debug, port=port)
