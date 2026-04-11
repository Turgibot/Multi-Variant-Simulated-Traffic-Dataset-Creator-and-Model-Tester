#!/usr/bin/env bash
# Build the GUI binary and assemble a portable folder:
#   <bundle>/
#     graph-traffic-dataset-creator   (or .exe on Windows — use the .bat on Windows)
#     sumo/                           (optional: full SUMO install; see README inside)
#
# When frozen, the app looks for SUMO at <directory of executable>/sumo (see sumo_detector.py).
#
# Optional: copy SUMO into the bundle automatically:
#   SUMO_HOME=/opt/sumo ./scripts/package_portable_bundle.sh
#   PORTABLE_BUNDLE_SUMO_HOME=/path/to/sumo ./scripts/package_portable_bundle.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

"$ROOT/scripts/build_executable.sh"

BUNDLE="$ROOT/dist/portable-graph-traffic-dataset-creator"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/sumo"

if [[ -f "$ROOT/dist/graph-traffic-dataset-creator.exe" ]]; then
  cp "$ROOT/dist/graph-traffic-dataset-creator.exe" "$BUNDLE/"
else
  cp "$ROOT/dist/graph-traffic-dataset-creator" "$BUNDLE/"
fi

SUMO_SRC="${PORTABLE_BUNDLE_SUMO_HOME:-${SUMO_HOME:-}}"
if [[ -n "$SUMO_SRC" ]] && [[ -d "$SUMO_SRC/bin" ]] && [[ -d "$SUMO_SRC/tools" ]]; then
  echo "Copying SUMO from: $SUMO_SRC"
  cp -a "$SUMO_SRC/." "$BUNDLE/sumo/"
else
  cat > "$BUNDLE/sumo/README.txt" << 'EOF'
Portable SUMO layout
====================

Put a complete SUMO installation in this folder so that:

  • Linux/macOS: sumo/bin/sumo exists
  • Windows:     sumo/bin/sumo.exe exists
  • TraCI/sumolib: sumo/tools/ exists

Download builds: https://sumo.dlr.de/docs/Downloads.php

If the archive unpacks to a versioned directory (e.g. sumo-1.20.0/),
copy the *contents* of that directory here — this directory should be SUMO_HOME.

The application (when run as a PyInstaller binary) checks for ./sumo next to the executable
before falling back to SUMO_HOME, PATH, and common install locations.
EOF
  echo "No SUMO copy: created sumo/README.txt (set SUMO_HOME or PORTABLE_BUNDLE_SUMO_HOME to bundle SUMO)."
fi

ARCHIVE="$ROOT/dist/portable-graph-traffic-dataset-creator-$(uname -s | tr '[:upper:]' '[:lower:]').tar.gz"
tar -C "$ROOT/dist" -czf "$ARCHIVE" "$(basename "$BUNDLE")"
echo "Portable bundle: $BUNDLE"
echo "Archive:         $ARCHIVE"
