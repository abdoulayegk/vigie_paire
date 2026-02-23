"""Minimal CLI entry-point for bank-peer-vigilance."""

from __future__ import annotations

import argparse
import sys

from vigilance.config.loader import get_bank_cfg, load_config

DEFAULT_CONFIG = "configs/bank_profiles.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bank Peer Vigilance – section extraction CLI",
    )
    parser.add_argument("--bank", required=True, help="Bank code (e.g. rbc)")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--quarter", required=True, help="Quarter label (e.g. Q1-2025)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to the YAML config (default: {DEFAULT_CONFIG})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    get_bank_cfg(cfg, args.bank)
    print(f"OK config loaded for {args.bank}, pdf={args.pdf}, quarter={args.quarter}")


if __name__ == "__main__":
    main()
