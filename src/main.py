"""
Main application entry point.
"""

import os
import sys


# Add SUMO tools to path if available
def setup_sumo_path():
    """Add SUMO tools directory to Python path."""
    import subprocess

    # Try to find SUMO tools
    try:
        result = subprocess.run(['which', 'sumo'], capture_output=True, text=True)
        if result.returncode == 0:
            sumo_bin = result.stdout.strip()
            sumo_dir = os.path.dirname(os.path.dirname(sumo_bin))
            tools_path = os.path.join(sumo_dir, 'tools')
            if os.path.exists(tools_path) and tools_path not in sys.path:
                sys.path.insert(0, tools_path)
    except Exception:
        pass
    
    # Common paths
    common_paths = [
        '/usr/share/sumo/tools',
        '/opt/sumo/tools',
        '/usr/local/share/sumo/tools',
    ]
    
    for path in common_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)


def _resolve_libx11_for_preload():
    """Absolute path to libX11.so.6 for LD_PRELOAD (Linux)."""
    candidates = (
        "/lib/x86_64-linux-gnu/libX11.so.6",
        "/usr/lib/x86_64-linux-gnu/libX11.so.6",
        "/lib64/libX11.so.6",
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    try:
        import ctypes.util

        name = ctypes.util.find_library("X11")
        if name and os.path.isfile(name):
            return name
    except Exception:
        pass
    return None


def _reexec_with_libx11_preload_if_needed():
    """
    uv's CPython on Linux can resolve X11 symbols from the wrong DSO, so Qt's
    xcb plugin crashes in XOpenDisplay (SIGSEGV). Preloading libX11 fixes it.
    Must run before PySide6/Qt is imported; uses exec so LD_PRELOAD applies.
    """
    if sys.platform != "linux":
        return
    if not os.environ.get("DISPLAY"):
        return
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return
    ld = os.environ.get("LD_PRELOAD", "")
    if "libX11.so" in ld:
        return
    x11 = _resolve_libx11_for_preload()
    if not x11:
        return
    os.environ["LD_PRELOAD"] = f"{x11}:{ld}" if ld.strip() else x11
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _libxcb_cursor_loadable():
    """Qt 6.5+ xcb platform plugin links libxcb-cursor; missing .so aborts the process."""
    import ctypes

    try:
        ctypes.CDLL("libxcb-cursor.so.0")
        return True
    except OSError:
        return False


def check_qt_platform():
    """Pick a viable Qt platform: offscreen if headless, wayland if XCB cursor is missing."""
    import os

    explicit = os.environ.get("QT_QPA_PLATFORM")
    has_display = bool(os.environ.get("DISPLAY"))
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    has_gui_session = has_display or has_wayland

    if not has_gui_session and not explicit:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        print(
            "Note: Using offscreen platform (no DISPLAY or WAYLAND_DISPLAY; "
            "GUI will not appear on screen)"
        )
        return

    if explicit or not has_gui_session:
        return

    # Default platform is often xcb when DISPLAY is set; that requires libxcb-cursor.so.0.
    if not _libxcb_cursor_loadable():
        if has_wayland:
            os.environ["QT_QPA_PLATFORM"] = "wayland"
            print(
                "Note: libxcb-cursor not found; using Qt Wayland (QT_QPA_PLATFORM=wayland).",
                file=sys.stderr,
            )
        else:
            print(
                "PySide6 / Qt 6 needs libxcb-cursor for the X11 (xcb) platform plugin.\n"
                "Install on Debian/Ubuntu: sudo apt install libxcb-cursor0\n"
                "Then run this app again.",
                file=sys.stderr,
            )
            sys.exit(1)


def main():
    """Main application entry point."""
    _reexec_with_libx11_preload_if_needed()

    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Setup SUMO path
    setup_sumo_path()
    
    # Check Qt platform
    check_qt_platform()
    
    # Import Qt after platform is set
    from PySide6.QtCore import qInstallMessageHandler
    from PySide6.QtWidgets import QApplication

    from src.gui.main_window import MainWindow, load_app_icon

    # Suppress QPainter warnings (harmless but annoying)
    # These warnings come from Qt's internal rendering with cached graphics items
    original_handler = qInstallMessageHandler(None)  # Get default handler
    
    def message_handler(msg_type, context, message):
        """Filter out QPainter warnings."""
        # Suppress QPainter warnings about saved states
        if "QPainter::end" in message and "saved states" in message:
            return  # Suppress these warnings
        # Use original handler for other messages
        if original_handler:
            original_handler(msg_type, context, message)
    
    qInstallMessageHandler(message_handler)

    # Create application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Freedesktop: applicationName matches StartupWMClass in
    # scripts/install_linux_desktop_entry.sh so the shell uses the right .desktop icon.
    app.setApplicationName("graph-traffic-dataset-creator")
    app.setApplicationDisplayName("Graph Traffic Dataset Creator")
    app.setDesktopFileName("graph-traffic-dataset-creator")
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

