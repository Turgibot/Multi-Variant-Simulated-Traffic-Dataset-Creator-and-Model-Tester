#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="graph-traffic-dataset-creator"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_DIR="$DATA_HOME/applications"
ICON_DIR="$DATA_HOME/icons/hicolor/256x256/apps"
ICON_SRC="$ROOT/assets/traffic_app_icon.png"
RUN_SCRIPT="$ROOT/scripts/run_gui.sh"

if [ ! -f "$ICON_SRC" ]; then
  echo "Missing icon: $ICON_SRC" >&2
  exit 1
fi

chmod +x "$RUN_SCRIPT"
mkdir -p "$APP_DIR" "$ICON_DIR"
cp "$ICON_SRC" "$ICON_DIR/${APP_ID}.png"

cat > "$APP_DIR/${APP_ID}.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Graph Traffic Dataset Creator
Comment=SUMO simulations and graph traffic datasets
Exec=$RUN_SCRIPT %u
Icon=$APP_ID
Categories=Science;Education;
Terminal=false
StartupNotify=true
EOF

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "$DATA_HOME/icons/hicolor" 2>/dev/null || true
fi
update-desktop-database "$APP_DIR" 2>/dev/null || true

echo "Installed: $APP_DIR/${APP_ID}.desktop"
echo "Start the app from the app grid or run: $RUN_SCRIPT"
