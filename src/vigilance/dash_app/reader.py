r"""Point d'entree VigieRegDesjardins (.exe Windows en lecture seule).

Demarre l'application Dash en mode "reader" :

- aucune dependance LLM (docling, openai) requise — pipeline non disponible ;
- lit les comparaisons deja generees dans ``outputs/resultats/<banque>/<periode>/``
  (typiquement un dossier SharePoint synchronise via OneDrive) ;
- l'analyste valide/annote -> ecrit dans ``comparison.review_state.<username>.json``
  pour eviter les conflits multi-utilisateurs (consolidation faite plus tard
  cote pipeline) ;
- demande au premier lancement le dossier ``resultats``, memorise dans
  ``%APPDATA%\\VigieRegDesjardins\\config.json``.

Pour lancer en dev :
    uv run python -m vigilance.dash_app.reader

Empaquete avec PyInstaller (cible : Windows .exe), ce module est l'``entry``
du ``.spec``. Voir ``packaging/`` (a venir en Phase 2).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "VigieRegDesjardins"
DEFAULT_PORT = 8050


def _config_dir() -> Path:
    """Retourne le repertoire de configuration utilisateur (cree si besoin)."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    target = base / APP_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def _config_path() -> Path:
    return _config_dir() / "config.json"


def _load_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(payload: dict) -> None:
    path = _config_path()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ask_resultats_dir(initial: str | None = None) -> str | None:
    """Ouvre une boite de dialogue (tkinter) pour selectionner le dossier resultats."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        logger.error("tkinter indisponible — impossible de demander le dossier resultats")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(
        title=f"{APP_NAME} — Selectionnez le dossier 'resultats' (synchronise SharePoint)",
        initialdir=initial or str(Path.home()),
        mustexist=True,
    )
    root.destroy()
    return chosen or None


def _resolve_resultats_dir() -> Path:
    """Determine le dossier resultats : env var, config, ou dialogue utilisateur."""
    env_override = os.environ.get("VIGIE_RESULTATS_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()

    cfg = _load_config()
    saved = cfg.get("resultats_dir")
    if saved and Path(saved).exists():
        return Path(saved).expanduser().resolve()

    chosen = _ask_resultats_dir(initial=saved)
    if not chosen:
        sys.stderr.write(
            f"[{APP_NAME}] Aucun dossier 'resultats' selectionne — arret.\n"
        )
        sys.exit(1)

    chosen_path = Path(chosen).expanduser().resolve()
    cfg["resultats_dir"] = str(chosen_path)
    _save_config(cfg)
    return chosen_path


def _find_free_port(preferred: int = DEFAULT_PORT) -> int:
    """Retourne un port libre, en commencant par ``preferred``."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", candidate))
                return sock.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("Aucun port libre disponible")


def main() -> None:
    """Initialise reader_config + env, importe l'app Dash, lance le serveur."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    resultats_dir = _resolve_resultats_dir()
    os.environ["VIGIE_RESULTATS_DIR"] = str(resultats_dir)
    logger.info("Dossier resultats : %s", resultats_dir)

    # Activer le mode reader AVANT d'importer l'app Dash, pour que
    # ``build_sidebar()`` rende sa version reduite.
    from vigilance.dash_app import reader_config

    reader_config.set_reader_mode(True)
    username = reader_config.current_username()
    logger.info("Identifiant analyste : %s", username)

    # Import de l'app apres configuration du mode reader.
    from vigilance.dash_app.app import app

    port = _find_free_port(int(os.environ.get("VIGIE_PORT", DEFAULT_PORT)))
    url = f"http://127.0.0.1:{port}"
    logger.info("Demarrage de %s sur %s", APP_NAME, url)

    try:
        webbrowser.open(url)
    except Exception:
        logger.warning("Impossible d'ouvrir le navigateur automatiquement — ouvrez %s", url)

    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
