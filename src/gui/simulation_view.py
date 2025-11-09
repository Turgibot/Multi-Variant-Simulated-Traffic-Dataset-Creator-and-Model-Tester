"""
Custom QGraphicsView widget for rendering SUMO simulation.
"""

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QWheelEvent
from typing import Dict, List, Tuple, Optional
import math


class NetworkEdgeItem(QGraphicsItem):
    """Graphics item for rendering a network edge."""
    
    def __init__(self, edge_data: Dict, parent=None):
        super().__init__(parent)
        self.edge_data = edge_data
        self.setZValue(0)  # Background layer
        
        # Calculate bounding rect from lanes
        if edge_data['lanes']:
            all_points = []
            for lane in edge_data['lanes']:
                all_points.extend(lane.get('shape', []))
            
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                self.bounding_rect = QRectF(
                    min(xs), min(ys),
                    max(xs) - min(xs), max(ys) - min(ys)
                )
            else:
                self.bounding_rect = QRectF(0, 0, 0, 0)
        else:
            self.bounding_rect = QRectF(0, 0, 0, 0)
    
    def boundingRect(self) -> QRectF:
        return self.bounding_rect
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the edge."""
        pen = QPen(QColor(100, 100, 100), 2)
        painter.setPen(pen)
        
        # Draw each lane
        for lane in self.edge_data.get('lanes', []):
            shape_points = lane.get('shape', [])
            if len(shape_points) >= 2:
                for i in range(len(shape_points) - 1):
                    p1 = QPointF(shape_points[i][0], shape_points[i][1])
                    p2 = QPointF(shape_points[i+1][0], shape_points[i+1][1])
                    painter.drawLine(p1, p2)


class VehicleItem(QGraphicsEllipseItem):
    """Graphics item for rendering a vehicle."""
    
    def __init__(self, vehicle_id: str, x: float, y: float, angle: float = 0, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.setPos(x, y)
        self.setRotation(angle)
        self.setZValue(10)  # Foreground layer
        
        # Vehicle size (meters to pixels approximation)
        size = 4.0
        self.setRect(-size/2, -size/2, size, size)
        
        # Vehicle color
        brush = QBrush(QColor(255, 100, 100))
        self.setBrush(brush)
        pen = QPen(QColor(200, 50, 50), 1)
        self.setPen(pen)
    
    def update_position(self, x: float, y: float, angle: float = 0):
        """Update vehicle position."""
        self.setPos(x, y)
        self.setRotation(angle)


class SimulationView(QGraphicsView):
    """Custom QGraphicsView for SUMO simulation rendering."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # View settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Zoom settings
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.current_zoom = 1.0
        
        # Network items
        self.edge_items = {}
        self.vehicle_items = {}
        
        # Background color
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    def load_network(self, network_parser):
        """Load network from parser."""
        # Clear existing items
        self.clear_network()
        
        # Add edges
        edges = network_parser.get_edges()
        for edge_id, edge_data in edges.items():
            edge_item = NetworkEdgeItem(edge_data)
            self.scene.addItem(edge_item)
            self.edge_items[edge_id] = edge_item
        
        # Fit view to network bounds
        bounds = network_parser.get_bounds()
        if bounds:
            rect = QRectF(
                bounds['x_min'],
                bounds['y_min'],
                bounds['x_max'] - bounds['x_min'],
                bounds['y_max'] - bounds['y_min']
            )
            self.fitInView(rect, Qt.KeepAspectRatio)
            self.current_zoom = 1.0
    
    def clear_network(self):
        """Clear all network items."""
        for item in list(self.edge_items.values()):
            self.scene.removeItem(item)
        self.edge_items.clear()
        self.clear_vehicles()
    
    def update_vehicle(self, vehicle_id: str, x: float, y: float, angle: float = 0):
        """Update or create a vehicle."""
        if vehicle_id in self.vehicle_items:
            self.vehicle_items[vehicle_id].update_position(x, y, angle)
        else:
            vehicle_item = VehicleItem(vehicle_id, x, y, angle)
            self.scene.addItem(vehicle_item)
            self.vehicle_items[vehicle_id] = vehicle_item
    
    def remove_vehicle(self, vehicle_id: str):
        """Remove a vehicle."""
        if vehicle_id in self.vehicle_items:
            self.scene.removeItem(self.vehicle_items[vehicle_id])
            del self.vehicle_items[vehicle_id]
    
    def clear_vehicles(self):
        """Clear all vehicles."""
        for item in list(self.vehicle_items.values()):
            self.scene.removeItem(item)
        self.vehicle_items.clear()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel zoom."""
        # Calculate zoom factor
        if event.angleDelta().y() > 0:
            zoom = self.zoom_factor
        else:
            zoom = 1 / self.zoom_factor
        
        # Check zoom limits
        new_zoom = self.current_zoom * zoom
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.scale(zoom, zoom)
            self.current_zoom = new_zoom
    
    def zoom_in(self):
        """Zoom in programmatically."""
        if self.current_zoom * self.zoom_factor <= self.max_zoom:
            self.scale(self.zoom_factor, self.zoom_factor)
            self.current_zoom *= self.zoom_factor
    
    def zoom_out(self):
        """Zoom out programmatically."""
        if self.current_zoom / self.zoom_factor >= self.min_zoom:
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
            self.current_zoom /= self.zoom_factor
    
    def zoom_fit(self, network_parser=None):
        """Fit network in view."""
        if network_parser:
            bounds = network_parser.get_bounds()
            if bounds:
                rect = QRectF(
                    bounds['x_min'],
                    bounds['y_min'],
                    bounds['x_max'] - bounds['x_min'],
                    bounds['y_max'] - bounds['y_min']
                )
                self.fitInView(rect, Qt.KeepAspectRatio)
                # Reset zoom tracking (fitInView resets transform)
                self.current_zoom = 1.0
        else:
            # Fit to all items
            if self.scene.items():
                rect = self.scene.itemsBoundingRect()
                if not rect.isEmpty():
                    self.fitInView(rect, Qt.KeepAspectRatio)
                    self.current_zoom = 1.0
    
    def reset_view(self, network_parser=None):
        """Reset view to default."""
        self.resetTransform()
        self.current_zoom = 1.0
        if network_parser:
            self.zoom_fit(network_parser)

