"""
Minimal PySide6 test - creates a simple window to verify installation.
Run this script to see a PySide6 window.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Test Window")
        self.setGeometry(100, 100, 400, 300)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add label
        label = QLabel("PySide6 is working correctly!")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(label)
        
        # Add version info
        from PySide6 import __version__
        version_label = QLabel(f"PySide6 Version: {__version__}")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)


def check_xcb_available():
    """Check if xcb-cursor library is available."""
    import subprocess
    try:
        # Try to find the library
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
    
    # Try checking with dpkg (Debian/Ubuntu)
    try:
        result = subprocess.run(
            ['dpkg', '-l'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if 'xcb-cursor' in result.stdout:
            return True
    except Exception:
        pass
    
    return False


def main():
    # Check if running in headless environment or if xcb is not available
    # Use offscreen platform if DISPLAY is not set or if xcb fails
    import os
    
    # Set platform BEFORE creating QApplication
    use_offscreen = False
    
    if not os.environ.get('DISPLAY'):
        print("No DISPLAY found, using offscreen platform...")
        use_offscreen = True
    elif os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        use_offscreen = True
    elif not check_xcb_available():
        print("xcb-cursor library not found, using offscreen platform...")
        print("To use GUI mode, install: sudo apt-get install libxcb-cursor0")
        use_offscreen = True
    
    if use_offscreen:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Now create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = TestWindow()
    
    # Only show window if not in offscreen mode
    if not use_offscreen:
        window.show()
        print("PySide6 test window opened. Close the window to exit.")
        sys.exit(app.exec())
    else:
        print("PySide6 test window created (offscreen mode).")
        print("Window components are working correctly!")
        window.close()
        return 0


if __name__ == '__main__':
    main()

