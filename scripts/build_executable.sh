#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == Linux ]] && ! command -v objdump >/dev/null 2>&1; then
  echo "PyInstaller on Linux needs objdump (from the binutils package)." >&2
  echo "Install it, then re-run this script. Examples:" >&2
  echo "  Debian/Ubuntu: sudo apt install binutils" >&2
  echo "  Fedora:        sudo dnf install binutils" >&2
  echo "  Arch:          sudo pacman -S binutils" >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/pyinstaller" ]]; then
  "$ROOT/.venv/bin/pyinstaller" --noconfirm "$ROOT/graph_traffic_dataset_creator.spec"
elif command -v uv >/dev/null 2>&1; then
  uv run pyinstaller --noconfirm "$ROOT/graph_traffic_dataset_creator.spec"
else
  pyinstaller --noconfirm "$ROOT/graph_traffic_dataset_creator.spec"
fi
