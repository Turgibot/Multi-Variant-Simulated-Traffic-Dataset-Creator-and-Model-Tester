# SUMO_HOME and Standalone Executables

## Overview

When creating a standalone executable with PyInstaller, **SUMO itself is NOT bundled** with the executable. SUMO is a large C++ application (typically 100-500MB) and needs to be installed separately on the target system.

## How SUMO_HOME Works in Executables

### 1. **Executable Contents**
- ✅ Your Python application code
- ✅ PySide6/Qt libraries
- ✅ Python dependencies (numpy, pandas, etc.)
- ❌ SUMO installation (NOT included)

### 2. **SUMO_HOME Path Storage**
- SUMO_HOME path is stored in `project_folder/sumo_config.json`
- Each project can have its own SUMO_HOME path
- Path is stored as an **absolute path** (e.g., `/opt/sumo` or `C:\Program Files\SUMO`)

### 3. **Path Resolution in Executable**
When the executable runs:
1. Loads SUMO_HOME from project config
2. Sets `SUMO_HOME` environment variable
3. Adds `SUMO_HOME/tools` to Python path (for TraCI)
4. Uses `SUMO_HOME/bin/sumo` to start simulations

## User Requirements

### For End Users:
1. **Install SUMO separately** on their system
2. **Set SUMO_HOME** in the application (via settings page)
3. The executable will use the configured SUMO_HOME path

### Installation Options:
- **Option A**: Users install SUMO from official sources
  - Download from https://sumo.dlr.de/
  - Install to standard location (e.g., `/opt/sumo` or `C:\Program Files\SUMO`)
  - Set SUMO_HOME in the app

- **Option B**: Bundle SUMO with installer (Advanced)
  - Create platform-specific installers (Windows/macOS/Linux)
  - Include SUMO in installer package
  - Installer sets SUMO_HOME automatically
  - **Note**: This makes the installer very large (500MB+)

## Implementation Details

### Current Implementation:
- SUMO_HOME is stored per-project in `sumo_config.json`
- When starting simulation, the app:
  1. Reads SUMO_HOME from config
  2. Sets `os.environ['SUMO_HOME']`
  3. Adds `SUMO_HOME/tools` to `sys.path`
  4. Uses `SUMO_HOME/bin/sumo` binary

### Path Handling:
- **Absolute paths**: Stored as absolute paths (works across systems)
- **Relative paths**: Not recommended (breaks when project moves)
- **Validation**: Checks that path exists and has `bin/` directory

## Recommendations

### For Standalone Executable:
1. **Require SUMO installation**: Users must install SUMO separately
2. **First-run setup**: Guide users to set SUMO_HOME on first use
3. **Auto-detection**: Try to auto-detect SUMO if not set (optional)
4. **Clear error messages**: Show helpful messages if SUMO not found

### Alternative Approaches:
1. **Bundle SUMO**: Include SUMO in installer (large download, but self-contained)
2. **Portable SUMO**: Include portable SUMO version with executable
3. **Cloud SUMO**: Use remote SUMO service (requires internet)

## Current Code Behavior

The application:
- ✅ Stores SUMO_HOME per project
- ✅ Validates SUMO_HOME path
- ✅ Sets environment variable when starting simulation
- ✅ Adds tools directory to Python path
- ✅ Uses configured SUMO binary

This approach works well for executables where users install SUMO separately.

