"""Lance le validateur Dash leger sur Windows, macOS ou Linux.

Ce point d'entree charge uniquement des resultats existants. Il n'execute ni
extraction PDF, ni comparaison, ni appel LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "VigieValidateur"
DEFAULT_PORT = 8050


def _platform_config_base(platform: str | None = None) -> Path:
    """Retourner la racine de configuration conventionnelle pour l'OS courant."""
    effective_platform = platform or sys.platform
    if effective_platform.startswith("win"):
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    if effective_platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _config_dir() -> Path:
    """Creer et retourner le dossier de configuration propre a l'OS."""
    target = _platform_config_base() / APP_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def _config_path() -> Path:
    """Retourner le chemin du fichier memorisant le dossier de resultats."""
    return _config_dir() / "config.json"


def _load_config() -> dict:
    """Charger la configuration utilisateur si elle est exploitable."""
    path = _config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_config(payload: dict) -> None:
    """Sauvegarder la configuration utilisateur sans donnee sensible."""
    _config_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _ask_resultats_dir(initial: str | None = None) -> str | None:
    """Afficher le selecteur de dossier natif quand une interface est disponible."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(
            title="Selectionnez le dossier de resultats Vigie",
            initialdir=initial or str(Path.home()),
            mustexist=True,
        )
        root.destroy()
        return chosen or None
    except Exception as exc:
        logger.info("Selecteur graphique indisponible: %s", exc)
        return None


def _existing_directory(raw_path: str | Path | None) -> Path | None:
    """Normaliser un chemin uniquement s'il designe un dossier existant."""
    if raw_path is None or not str(raw_path).strip():
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_dir():
        return None
    return candidate.resolve()


def _resolve_resultats_dir(cli_path: str | None = None) -> Path:
    """Resoudre le dossier via CLI, environnement, configuration puis dialogue."""
    explicit_sources = (
        ("--resultats", cli_path),
        ("VIGIE_RESULTATS_DIR", os.environ.get("VIGIE_RESULTATS_DIR")),
    )
    for source, raw_path in explicit_sources:
        if raw_path and not (resolved := _existing_directory(raw_path)):
            raise ValueError(f"Dossier invalide fourni par {source}: {raw_path}")
        if raw_path and resolved:
            return resolved

    config = _load_config()
    saved = config.get("resultats_dir")
    if resolved := _existing_directory(saved):
        return resolved

    chosen = _ask_resultats_dir(initial=str(saved or "") or None)
    if not (resolved := _existing_directory(chosen)):
        raise ValueError(
            "Aucun dossier de resultats selectionne. Utilisez --resultats ou "
            "VIGIE_RESULTATS_DIR."
        )
    config["resultats_dir"] = str(resolved)
    _save_config(config)
    return resolved


def _find_free_port(preferred: int = DEFAULT_PORT) -> int:
    """Trouver un port local disponible en privilegiant la valeur demandee."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", candidate))
                return int(sock.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError("Aucun port local disponible")


def build_parser() -> argparse.ArgumentParser:
    """Construire les options du validateur multiplateforme."""
    parser = argparse.ArgumentParser(
        description="Consulter et valider des resultats Vigie sans appels LLM."
    )
    parser.add_argument("--resultats", help="Dossier contenant les resultats existants")
    parser.add_argument("--analyste", help="Identifiant utilise pour les validations")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DASH_PORT", DEFAULT_PORT)),
        help=f"Port Dash prefere (defaut: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--sans-navigateur",
        action="store_true",
        help="Ne pas ouvrir automatiquement le navigateur",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Configurer le mode sans LLM et lancer l'application Dash locale."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        resultats_dir = _resolve_resultats_dir(args.resultats)
    except ValueError as exc:
        build_parser().error(str(exc))

    os.environ["VIGIE_RESULTATS_DIR"] = str(resultats_dir)

    from vigie.interface import validator_config

    validator_config.set_validator_mode(True)
    validator_config.set_username(args.analyste)

    # Import tardif indispensable : ui_config doit lire VIGIE_RESULTATS_DIR
    # avant la construction de l'application et de ses stores.
    from vigie.interface.app import app

    port = _find_free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    logger.info("Mode validateur sans LLM: %s", resultats_dir)
    logger.info("Analyste: %s", validator_config.current_username())

    if not args.sans_navigateur:
        try:
            webbrowser.open(url)
        except Exception:
            logger.warning("Ouvrez manuellement %s", url)

    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
