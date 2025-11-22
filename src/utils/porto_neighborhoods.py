"""
Porto quadrant (quadra) grid system.
Divides the map into a 4x4 grid (16 quadrants) named A through P.
"""

from typing import Dict, List, Tuple
from PySide6.QtGui import QColor
from PySide6.QtCore import QRectF
import math


# Generate colors for 16 quadrants using a color palette
def _generate_quadrant_colors() -> List[Tuple[int, int, int]]:
    """Generate distinct colors for 16 quadrants."""
    # Predefined distinct colors for 16 quadrants
    colors = [
        (255, 100, 100),   # A - Red
        (100, 255, 100),   # B - Green
        (100, 100, 255),   # C - Blue
        (255, 255, 100),   # D - Yellow
        (255, 100, 255),   # E - Magenta
        (100, 255, 255),   # F - Cyan
        (255, 165, 0),     # G - Orange
        (128, 0, 128),     # H - Purple
        (255, 192, 203),   # I - Pink
        (144, 238, 144),   # J - Light Green
        (173, 216, 230),   # K - Light Blue
        (255, 218, 185),   # L - Peach
        (221, 160, 221),   # M - Plum
        (152, 251, 152),   # N - Pale Green
        (255, 160, 122),   # O - Light Salmon
        (176, 224, 230),   # P - Powder Blue
    ]
    return colors


QUADRANT_COLORS = _generate_quadrant_colors()


def get_porto_quadrants(network_bounds: Dict[str, float]) -> List[Tuple[str, QRectF, QColor]]:
    """
    Generate 16 quadrants (4x4 grid) covering the network bounds.
    
    Args:
        network_bounds: Dict with 'x_min', 'y_min', 'x_max', 'y_max' in network coordinates
        
    Returns:
        List of tuples: (name, QRectF, QColor) where name is A-P
    """
    if not network_bounds:
        return []
    
    x_min = network_bounds.get('x_min', 0)
    y_min = network_bounds.get('y_min', 0)
    x_max = network_bounds.get('x_max', 10000)
    y_max = network_bounds.get('y_max', 10000)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Divide into 4x4 grid
    quadrant_width = width / 4
    quadrant_height = height / 4
    
    quadrants = []
    quadrant_names = [chr(ord('A') + i) for i in range(16)]  # A, B, C, ..., P
    
    quadrant_index = 0
    for row in range(4):
        for col in range(4):
            x = x_min + col * quadrant_width
            y = y_min + row * quadrant_height
            rect = QRectF(x, y, quadrant_width, quadrant_height)
            color = QColor(*QUADRANT_COLORS[quadrant_index])
            quadrants.append((quadrant_names[quadrant_index], rect, color))
            quadrant_index += 1
    
    return quadrants


def get_quadrant_color(name: str) -> QColor:
    """
    Get color for a quadrant by name (A-P).
    
    Args:
        name: Quadrant name (A-P)
        
    Returns:
        QColor for the quadrant
    """
    if len(name) == 1 and 'A' <= name.upper() <= 'P':
        index = ord(name.upper()) - ord('A')
        if 0 <= index < 16:
            return QColor(*QUADRANT_COLORS[index])
    # Default color if not found
    return QColor(200, 200, 200)


def get_quadrant_for_point(x: float, y: float, network_bounds: Dict[str, float]) -> str:
    """
    Determine which quadrant (A-P) a point belongs to.
    
    Args:
        x: X coordinate in network space
        y: Y coordinate in network space
        network_bounds: Dict with 'x_min', 'y_min', 'x_max', 'y_max'
        
    Returns:
        Quadrant name (A-P) or None if outside bounds
    """
    if not network_bounds:
        return None
    
    x_min = network_bounds.get('x_min', 0)
    y_min = network_bounds.get('y_min', 0)
    x_max = network_bounds.get('x_max', 10000)
    y_max = network_bounds.get('y_max', 10000)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Check if point is within bounds
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return None
    
    # Calculate grid position (0-3 for both x and y)
    # Handle boundary cases: if exactly on max boundary, assign to last quadrant
    if width > 0:
        col_float = (x - x_min) / (width / 4)
        col = min(3, int(col_float))
        # If exactly on the boundary between quadrants, use the right/lower one
        if col_float >= 4.0:
            col = 3
    else:
        col = 0
    
    if height > 0:
        row_float = (y - y_min) / (height / 4)
        row = min(3, int(row_float))
        # If exactly on the boundary between quadrants, use the right/lower one
        if row_float >= 4.0:
            row = 3
    else:
        row = 0
    
    # Convert to quadrant name (A-P)
    quadrant_index = row * 4 + col
    return chr(ord('A') + quadrant_index)

