"""
Main application entry point.
"""

import sys
import os

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


def check_qt_platform():
    """Check and set Qt platform if needed."""
    import os
    
    # Check if xcb-cursor is available
    def check_xcb_available():
        import subprocess
        try:
            result = subprocess.run(
                ['ldconfig', '-p'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if 'xcb-cursor' in result.stdout:
                return True
        except Exception:
            pass
        return False
    
    # Use offscreen if no DISPLAY or xcb not available
    if not os.environ.get('DISPLAY') or not check_xcb_available():
        if not os.environ.get('QT_QPA_PLATFORM'):
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            print("Note: Using offscreen platform (GUI may not be visible)")


def main():
    """Main application entry point."""
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Setup SUMO path
    setup_sumo_path()
    
    # Check Qt platform
    check_qt_platform()
    
    # Import Qt after platform is set
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    
    # Create application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

