# Qt/PySide6 Dependencies Installation

## Issue
If you see errors like:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
xcb-cursor0 or libxcb-cursor0 is needed
```

This means the Qt system dependencies are missing.

## Solution

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install libxcb-cursor0 libxcb-cursor-dev
```

### Or install all Qt dependencies:
```bash
sudo apt-get install libxcb-cursor0 libxcb-cursor-dev \
    libxcb-xinerama0 libxcb-xinerama0-dev \
    libxcb-xfixes0 libxcb-xfixes0-dev \
    libxcb-render0 libxcb-render0-dev \
    libxcb-shape0 libxcb-shape0-dev \
    libxcb-randr0 libxcb-randr0-dev \
    libxcb-icccm4 libxcb-icccm4-dev \
    libxcb-image0 libxcb-image0-dev \
    libxcb-keysyms1 libxcb-keysyms1-dev \
    libxcb-util1 libxcb-util-dev
```

### Alternative: Use Offscreen Mode
If you don't need GUI (e.g., for testing or headless servers), you can use offscreen mode:
```bash
export QT_QPA_PLATFORM=offscreen
python tests/test_pyside6_minimal.py
```

The test scripts will automatically fall back to offscreen mode if xcb is not available.

