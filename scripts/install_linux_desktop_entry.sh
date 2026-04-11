#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="graph-traffic-dataset-creator"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_DIR="$DATA_HOME/applications"
ICON_DIR="$DATA_HOME/icons/hicolor/256x256/apps"
ICON_SRC="$ROOT/assets/icon.png"
if [ ! -f "$ICON_SRC" ]; then
  ICON_SRC="$ROOT/assets/app_icon_square_rgba.png"
fi
ICON_INSTALLED="$ICON_DIR/${APP_ID}.png"
RUN_SCRIPT="$ROOT/scripts/run_gui.sh"

if [ ! -f "$ICON_SRC" ]; then
  echo "Missing icon: add $ROOT/assets/icon.png (or app_icon_square_rgba.png)." >&2
  exit 1
fi

chmod +x "$RUN_SCRIPT"
mkdir -p "$APP_DIR" "$ICON_DIR"

# Center-crop to a square so theme/dock icons are not stretched (source may be widescreen).
SQUARE_TMP="$(mktemp --suffix=.png 2>/dev/null || mktemp)"
cleanup_icon_temp() { rm -f "$SQUARE_TMP"; }
trap cleanup_icon_temp EXIT
ICON_TO_INSTALL="$ICON_SRC"
# Square masters: copy as-is. Widescreen sources: optional ffmpeg center-crop.
if [ "$(basename "$ICON_SRC")" != "app_icon_square_rgba.png" ] && [ "$(basename "$ICON_SRC")" != "icon.png" ] && command -v ffmpeg >/dev/null 2>&1; then
  if ffmpeg -y -loglevel error -i "$ICON_SRC" \
    -vf "crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'(ih-min(iw,ih))/2'" \
    -frames:v 1 "$SQUARE_TMP" 2>/dev/null && [ -f "$SQUARE_TMP" ]; then
    ICON_TO_INSTALL="$SQUARE_TMP"
  fi
fi
cp "$ICON_TO_INSTALL" "$ICON_INSTALLED"

cat > "$APP_DIR/${APP_ID}.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Graph Traffic Dataset Creator
Comment=SUMO simulations and graph traffic datasets
Exec=$RUN_SCRIPT %u
Icon=$ICON_INSTALLED
Categories=Science;Education;
Terminal=false
StartupNotify=true
StartupWMClass=graph-traffic-dataset-creator
EOF

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "$DATA_HOME/icons/hicolor" 2>/dev/null || true
fi
update-desktop-database "$APP_DIR" 2>/dev/null || true

echo "Installed: $APP_DIR/${APP_ID}.desktop"
echo "Start the app from the app grid or run: $RUN_SCRIPT"
