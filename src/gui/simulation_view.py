"""
Custom QGraphicsView widget for rendering SUMO simulation.
"""

import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QPointF, QRectF, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QWheelEvent, QPixmap, QImage


# ESRI World Imagery tile URL
ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# Tile cache directory
TILE_CACHE_DIR = Path.home() / ".cache" / "sumo_satellite_tiles"


class TileDownloadWorker(QThread):
    """Worker thread for downloading satellite tiles."""
    
    tile_ready = Signal(int, int, int, QPixmap)  # z, x, y, pixmap
    all_done = Signal()
    
    def __init__(self, tiles_to_download: List[Tuple[int, int, int]], parent=None):
        super().__init__(parent)
        self.tiles_to_download = tiles_to_download
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        """Download tiles."""
        TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        for z, x, y in self.tiles_to_download:
            if self._is_cancelled:
                break
            
            cache_path = TILE_CACHE_DIR / f"{z}_{x}_{y}.png"
            
            # Check cache first
            if cache_path.exists():
                pixmap = QPixmap(str(cache_path))
                if not pixmap.isNull():
                    self.tile_ready.emit(z, x, y, pixmap)
                    continue
            
            # Download tile
            url = ESRI_TILE_URL.format(z=z, y=y, x=x)
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'SUMO-Traffic-Simulator/1.0'
                })
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = response.read()
                    
                    # Save to cache
                    with open(cache_path, 'wb') as f:
                        f.write(data)
                    
                    # Create pixmap
                    image = QImage()
                    image.loadFromData(data)
                    pixmap = QPixmap.fromImage(image)
                    
                    if not pixmap.isNull():
                        self.tile_ready.emit(z, x, y, pixmap)
            except Exception as e:
                print(f"Failed to download tile {z}/{x}/{y}: {e}")
        
        self.all_done.emit()


class NetworkEdgeItem(QGraphicsItem):
    """Graphics item for rendering a network edge."""
    
    def __init__(self, edge_data: Dict, parent=None):
        super().__init__(parent)
        self.edge_data = edge_data
        self.setZValue(0)  # Background layer
        self.is_selected = False  # Whether this edge is selected/highlighted
        
        # Performance: Enable caching for static items
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        
        # Pre-compute line segments for faster drawing
        self.line_segments = []
        all_points = []
        
        if edge_data['lanes']:
            for lane in edge_data['lanes']:
                shape_points = lane.get('shape', [])
                all_points.extend(shape_points)
                if len(shape_points) >= 2:
                    for i in range(len(shape_points) - 1):
                        self.line_segments.append((
                            QPointF(shape_points[i][0], shape_points[i][1]),
                            QPointF(shape_points[i+1][0], shape_points[i+1][1])
                        ))
        
        # Calculate bounding rect
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            # Add small margin for pen width
            margin = 3
            self.bounding_rect = QRectF(
                min(xs) - margin, min(ys) - margin,
                max(xs) - min(xs) + 2*margin, max(ys) - min(ys) + 2*margin
            )
        else:
            self.bounding_rect = QRectF(0, 0, 0, 0)
        
        # Pre-create pens
        self.normal_pen = QPen(QColor(100, 100, 100), 2)
        self.selected_pen = QPen(QColor(50, 50, 50), 4)
    
    def set_selected(self, selected: bool):
        """Set whether this edge is selected/highlighted."""
        if self.is_selected != selected:
            self.is_selected = selected
            self.update()  # Trigger repaint
    
    def boundingRect(self) -> QRectF:
        return self.bounding_rect
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the edge."""
        painter.setPen(self.selected_pen if self.is_selected else self.normal_pen)
        
        # Draw pre-computed line segments
        for p1, p2 in self.line_segments:
            painter.drawLine(p1, p2)


class NetworkNodeItem(QGraphicsEllipseItem):
    """Graphics item for rendering a network node/junction."""
    
    def __init__(self, node_id: str, x: float, y: float, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.is_selected = False
        self.setPos(x, y)
        self.setZValue(1)  # Above edges, below vehicles
        
        # Performance: Enable caching for static items
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        
        # Default size (small circle)
        self.default_size = 3.0
        self.selected_size = 6.0
        self.setRect(-self.default_size/2, -self.default_size/2, 
                    self.default_size, self.default_size)
        
        # Pre-create brushes and pens
        self.default_brush = QBrush(QColor(150, 150, 150))
        self.default_pen = QPen(QColor(100, 100, 100), 1)
        self.selected_brush = QBrush(QColor(50, 50, 50))
        self.selected_pen = QPen(QColor(30, 30, 30), 2)
        
        # Default appearance
        self.setBrush(self.default_brush)
        self.setPen(self.default_pen)
    
    def set_selected(self, selected: bool):
        """Set whether this node is selected/highlighted."""
        if self.is_selected != selected:
            self.is_selected = selected
            if selected:
                # Bigger circle for selected nodes
                self.setRect(-self.selected_size/2, -self.selected_size/2,
                           self.selected_size, self.selected_size)
                self.setBrush(self.selected_brush)
                self.setPen(self.selected_pen)
            else:
                # Default size
                self.setRect(-self.default_size/2, -self.default_size/2,
                           self.default_size, self.default_size)
                self.setBrush(self.default_brush)
                self.setPen(self.default_pen)


class VehicleItem(QGraphicsEllipseItem):
    """Graphics item for rendering a vehicle."""
    
    def __init__(self, vehicle_id: str, x: float, y: float, angle: float = 0, parent=None):
        super().__init__(parent)
        self.vehicle_id = vehicle_id
        self.setPos(x, y)
        self.setRotation(angle)
        self.setZValue(10)  # Foreground layer
        
        # Vehicle size (meters to pixels approximation) - 20x bigger for visibility
        size = 80.0  # 4.0 * 20
        self.setRect(-size/2, -size/2, size, size)
        
        # Vehicle color
        brush = QBrush(QColor(255, 100, 100))
        self.setBrush(brush)
        pen = QPen(QColor(200, 50, 50), 3)  # Thicker pen for larger vehicle
        self.setPen(pen)
    
    def update_position(self, x: float, y: float, angle: float = 0):
        """Update vehicle position."""
        self.setPos(x, y)
        self.setRotation(angle)


class SimulationView(QGraphicsView):
    """Custom QGraphicsView for SUMO simulation rendering."""
    
    # Signals
    satellite_loading_finished = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Performance optimizations for large networks
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        
        # Disable antialiasing for better performance (can be enabled for smaller networks)
        # self.setRenderHint(QPainter.Antialiasing)
        # self.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Use ScrollHandDrag for panning (more efficient than RubberBandDrag)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Improve scene indexing for large networks
        self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self.scene.setBspTreeDepth(16)  # Deeper tree for large networks
        
        # Zoom settings
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.current_zoom = 1.0
        
        # Network items
        self.edge_items = {}
        self.node_items = {}  # For rendering nodes/junctions
        self.vehicle_items = {}
        
        # Satellite tile items
        self.tile_items = {}  # Dict of (z, x, y) -> QGraphicsPixmapItem
        self.tile_worker = None
        self.show_satellite = False
        self.network_parser = None  # Store reference for satellite tiles
        
        # Background color
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        
        # Store network bounds for Y-flip calculation
        self._network_y_min = 0
        self._network_y_max = 0
    
    def load_network(self, network_parser, roads_junctions_only: bool = False):
        """Load network from parser.
        
        Args:
            network_parser: The NetworkParser instance containing the network data
            roads_junctions_only: If True, only show edges that allow passenger vehicles
                                  and use junctions instead of all nodes
        """
        from PySide6.QtWidgets import QApplication
        import copy
        
        # Store network parser reference for satellite tiles
        self.network_parser = network_parser
        
        # Clear existing items
        self.clear_network()
        
        # Get bounds for Y-flip calculation
        # The network map needs to be flipped vertically to match satellite imagery
        bounds = network_parser.get_bounds()
        if bounds:
            self._network_y_min = bounds['y_min']
            self._network_y_max = bounds['y_max']
        
        def flip_y(y):
            """Flip Y coordinate to correct north/south orientation."""
            # Flip around the center: new_y = y_max + y_min - y
            return self._network_y_max + self._network_y_min - y
        
        # Disable updates during loading for better performance
        self.setUpdatesEnabled(False)
        
        # Temporarily disable indexing for faster batch insertion
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        
        try:
            # Add edges (optionally filtered to roads only)
            edges = network_parser.get_edges()
            edge_count = 0
            batch_size = 500  # Process events every N items
            
            for edge_id, edge_data in edges.items():
                # Filter by roads_junctions_only: only show edges that allow passenger vehicles
                if roads_junctions_only:
                    if not edge_data.get('allows_passenger', True):
                        continue
                
                # Create a copy of edge_data with flipped Y coordinates
                flipped_edge_data = copy.deepcopy(edge_data)
                if flipped_edge_data.get('lanes'):
                    for lane in flipped_edge_data['lanes']:
                        if lane.get('shape'):
                            lane['shape'] = [(x, flip_y(y)) for x, y in lane['shape']]
                
                edge_item = NetworkEdgeItem(flipped_edge_data)
                self.scene.addItem(edge_item)
                self.edge_items[edge_id] = edge_item
                
                # Process events periodically to keep UI responsive
                edge_count += 1
                if edge_count % batch_size == 0:
                    QApplication.processEvents()
            
            # Add nodes or junctions as circles
            node_count = 0
            if roads_junctions_only:
                # Use junctions instead of nodes when filtering
                junctions = network_parser.get_junctions()
                for junction_id, junction_data in junctions.items():
                    x = junction_data.get('x', 0)
                    y = flip_y(junction_data.get('y', 0))
                    node_item = NetworkNodeItem(junction_id, x, y)
                    self.scene.addItem(node_item)
                    self.node_items[junction_id] = node_item
                    
                    node_count += 1
                    if node_count % batch_size == 0:
                        QApplication.processEvents()
            else:
                # Use all nodes
                nodes = network_parser.get_nodes()
                for node_id, node_data in nodes.items():
                    x = node_data.get('x', 0)
                    y = flip_y(node_data.get('y', 0))
                    node_item = NetworkNodeItem(node_id, x, y)
                    self.scene.addItem(node_item)
                    self.node_items[node_id] = node_item
                    
                    node_count += 1
                    if node_count % batch_size == 0:
                        QApplication.processEvents()
        finally:
            # Re-enable BSP indexing after loading
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
            self.setUpdatesEnabled(True)
        
        # Fit view to network bounds (bounds stay the same, just content is flipped)
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
        for item in list(self.node_items.values()):
            self.scene.removeItem(item)
        self.node_items.clear()
        self.clear_vehicles()
    
    def set_satellite_visible(self, visible: bool):
        """Show or hide satellite imagery.
        
        Args:
            visible: True to show satellite imagery, False to hide
        """
        self.show_satellite = visible
        
        if visible:
            # Load satellite tiles
            self._load_satellite_tiles()
        else:
            # Clear satellite tiles
            self._clear_satellite_tiles()
            # Reset background color
            self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    def _clear_satellite_tiles(self):
        """Clear all satellite tile items."""
        # Cancel any ongoing download
        if self.tile_worker and self.tile_worker.isRunning():
            self.tile_worker.cancel()
            self.tile_worker.wait()
        
        for item in list(self.tile_items.values()):
            self.scene.removeItem(item)
        self.tile_items.clear()
    
    def _load_satellite_tiles(self):
        """Load satellite tiles for the current network area."""
        if not self.network_parser:
            return
        
        orig_boundary = self.network_parser.orig_boundary
        bounds = self.network_parser.get_bounds()
        
        if not orig_boundary or not bounds:
            print("Cannot load satellite tiles: missing boundary information")
            return
        
        # Cancel any previous download
        if self.tile_worker and self.tile_worker.isRunning():
            self.tile_worker.cancel()
            self.tile_worker.wait()
        
        # Clear existing tiles
        self._clear_satellite_tiles()
        
        # Set dark background while loading
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        
        # Determine zoom level based on area size
        lon_range = orig_boundary['lon_max'] - orig_boundary['lon_min']
        lat_range = orig_boundary['lat_max'] - orig_boundary['lat_min']
        
        # Choose zoom level (higher = more detail, more tiles)
        # For Porto area (~0.5 degrees), zoom 13-14 works well
        if lon_range > 0.3 or lat_range > 0.3:
            zoom = 13
        elif lon_range > 0.1 or lat_range > 0.1:
            zoom = 14
        else:
            zoom = 15
        
        # Get tile coordinates for the boundary
        tiles_to_download = []
        
        min_tile_x, max_tile_y = self._gps_to_tile(
            orig_boundary['lon_min'], orig_boundary['lat_min'], zoom
        )
        max_tile_x, min_tile_y = self._gps_to_tile(
            orig_boundary['lon_max'], orig_boundary['lat_max'], zoom
        )
        
        # Collect all tiles in range
        for tx in range(min_tile_x, max_tile_x + 1):
            for ty in range(min_tile_y, max_tile_y + 1):
                tiles_to_download.append((zoom, tx, ty))
        
        print(f"Loading {len(tiles_to_download)} satellite tiles at zoom {zoom}")
        
        # Start download worker
        self.tile_worker = TileDownloadWorker(tiles_to_download)
        self.tile_worker.tile_ready.connect(self._on_tile_ready)
        self.tile_worker.all_done.connect(self._on_tiles_done)
        self.tile_worker.start()
    
    def _gps_to_tile(self, lon: float, lat: float, zoom: int) -> Tuple[int, int]:
        """Convert GPS coordinates to tile coordinates.
        
        Args:
            lon: Longitude
            lat: Latitude
            zoom: Tile zoom level
            
        Returns:
            Tuple of (tile_x, tile_y)
        """
        n = 2 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (tile_x, tile_y)
    
    def _tile_to_gps(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to GPS (northwest corner of tile).
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Tile zoom level
            
        Returns:
            Tuple of (lon, lat)
        """
        n = 2 ** zoom
        lon = tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
        lat = math.degrees(lat_rad)
        return (lon, lat)
    
    def _on_tile_ready(self, z: int, x: int, y: int, pixmap: QPixmap):
        """Handle a downloaded tile."""
        if not self.show_satellite or not self.network_parser:
            return
        
        # Get the GPS bounds of this tile
        nw_lon, nw_lat = self._tile_to_gps(x, y, z)      # NW corner (top-left of tile)
        se_lon, se_lat = self._tile_to_gps(x + 1, y + 1, z)  # SE corner (bottom-right of tile)
        
        # Convert GPS corners to SUMO coordinates
        nw_result = self.network_parser.gps_to_sumo_coords(nw_lon, nw_lat)
        se_result = self.network_parser.gps_to_sumo_coords(se_lon, se_lat)
        
        if nw_result is None or se_result is None:
            return
        
        nw_x, nw_y = nw_result
        se_x, se_y = se_result
        
        # Apply the same Y-flip used for the network
        # flip_y(y) = y_max + y_min - y
        nw_y_flipped = self._network_y_max + self._network_y_min - nw_y
        se_y_flipped = self._network_y_max + self._network_y_min - se_y
        
        # Calculate tile dimensions
        tile_width = abs(se_x - nw_x)
        tile_height = abs(nw_y_flipped - se_y_flipped)
        
        if tile_width <= 0 or tile_height <= 0:
            return
        
        # Scale the pixmap to exact dimensions
        target_width = max(1, int(math.ceil(tile_width)))
        target_height = max(1, int(math.ceil(tile_height)))
        
        scaled_pixmap = pixmap.scaled(
            target_width, target_height,
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        
        # Create graphics item
        item = QGraphicsPixmapItem(scaled_pixmap)
        
        # Position at the top-left corner in flipped coordinates
        # After Y-flip: nw_y_flipped < se_y_flipped (north is now at lower Y)
        pos_x = min(nw_x, se_x)
        pos_y = min(nw_y_flipped, se_y_flipped)
        
        item.setPos(pos_x, pos_y)
        item.setZValue(-10)  # Behind everything else
        
        self.scene.addItem(item)
        self.tile_items[(z, x, y)] = item
    
    def _on_tiles_done(self):
        """Called when all tiles are downloaded."""
        print(f"Satellite tiles loaded: {len(self.tile_items)} tiles")
        self.satellite_loading_finished.emit()
    
    def set_selected_edges(self, edge_ids: set):
        """Set which edges are selected/highlighted."""
        for edge_id, edge_item in self.edge_items.items():
            edge_item.set_selected(edge_id in edge_ids)
    
    def set_selected_nodes(self, node_ids: set):
        """Set which nodes are selected/highlighted."""
        for node_id, node_item in self.node_items.items():
            node_item.set_selected(node_id in node_ids)
    
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

