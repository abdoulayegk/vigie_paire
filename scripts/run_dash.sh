#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export DASH_DEBUG="${DASH_DEBUG:-0}"
export DASH_PORT="${DASH_PORT:-8050}"
export DOCLING_NUM_THREADS="${DOCLING_NUM_THREADS:-4}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache}"

exec uv run python -m vigilance.dash_app.app "$@"
