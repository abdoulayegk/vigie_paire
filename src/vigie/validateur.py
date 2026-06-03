"""Lance le validateur Dash Vigie sans exposer les modules internes."""

from __future__ import annotations

from vigilance.dash_app.reader import main


if __name__ == "__main__":
    main()
