"""
Custom QGraphicsView widget for rendering SUMO simulation.
"""

import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal, QTimer
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPen, QPixmap,
                           QPolygonF, QWheelEvent)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsPixmapItem, QGraphicsPolygonItem,
                               QGraphicsScene, QGraphicsView)

# OpenStreetMap tile URL
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# Tile cache directory
TILE_CACHE_DIR = Path.home() / ".cache" / "sumo_osm_tiles"


class TileDownloadWorker(QThread):
    """Worker thread for downloading OSM map tiles."""
    
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
            url = OSM_TILE_URL.format(z=z, x=x, y=y)
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


class NetworkPrepareWorker(QThread):
    """Worker thread for preparing network edge and node data (copy, flip Y)."""

    finished = Signal(object, object, object, object, float, float)
    # edges_data, nodes_data, conv_boundary, bounds, y_min, y_max

    def __init__(self, network_parser, roads_junctions_only: bool = False, parent=None):
        super().__init__(parent)
        self.network_parser = network_parser
        self.roads_junctions_only = roads_junctions_only

    def run(self):
        import copy
        conv_boundary = self.network_parser.conv_boundary
        bounds = self.network_parser.get_bounds()
        y_min = conv_boundary['y_min'] if conv_boundary else (bounds['y_min'] if bounds else 0)
        y_max = conv_boundary['y_max'] if conv_boundary else (bounds['y_max'] if bounds else 0)

        def flip_y(y):
            return y_max + y_min - y

        edges_data = []
        edges = self.network_parser.get_edges()
        for edge_id, edge_data in edges.items():
            if self.roads_junctions_only and not edge_data.get('allows_passenger', True):
                continue
            flipped_data = copy.deepcopy(edge_data)
            if flipped_data.get('lanes'):
                for lane in flipped_data['lanes']:
                    if lane.get('shape'):
                        lane['shape'] = [(x, flip_y(y)) for x, y in lane['shape']]
            edges_data.append((edge_id, flipped_data))

        nodes_data = []
        nodes = self.network_parser.get_nodes()
        junctions = self.network_parser.get_junctions()
        nodes_to_use = nodes if nodes else junctions
        for node_id, node_data in nodes_to_use.items():
            x = node_data.get('x', 0)
            y = flip_y(node_data.get('y', 0))
            shape_points = node_data.get('shape', [])
            if shape_points:
                flipped_shape = [(px, flip_y(py)) for px, py in shape_points]
            else:
                flipped_shape = None
            nodes_data.append((node_id, x, y, flipped_shape))

        self.finished.emit(edges_data, nodes_data, conv_boundary, bounds, y_min, y_max)


class NetworkEdgeItem(QGraphicsItem):
    """Graphics item for rendering a network edge."""
    
    def __init__(self, edge_data: Dict, parent=None):
        super().__init__(parent)
        self.edge_data = edge_data
        self.setZValue(0)  # Background layer
        self.is_selected = False  # Whether this edge is selected/highlighted
        
        # Performance: Enable caching for static items
        # Use ItemCoordinateCache instead of DeviceCoordinateCache to avoid painter state warnings
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        
        # Pre-compute line segments for faster drawing
        # Use the first lane's shape to represent the edge (all lanes typically follow same path)
        self.line_segments = []
        all_points = []
        num_lanes = len(edge_data.get('lanes', []))
        
        if edge_data['lanes']:
            # Use first lane's shape (representative of the road)
            first_lane = edge_data['lanes'][0]
            shape_points = first_lane.get('shape', [])
            all_points.extend(shape_points)
            if len(shape_points) >= 2:
                for i in range(len(shape_points) - 1):
                    self.line_segments.append((
                        QPointF(shape_points[i][0], shape_points[i][1]),
                        QPointF(shape_points[i+1][0], shape_points[i+1][1])
                    ))
        
        # Calculate line thickness based on number of lanes
        # Base thickness: 1.5, add 0.5 per lane (min 1.5, scales with lanes)
        base_thickness = 1.5
        lane_thickness = 0.5
        line_thickness = base_thickness + (num_lanes - 1) * lane_thickness
        # Cap at reasonable maximum (e.g., 8 for very wide roads)
        line_thickness = min(line_thickness, 8.0)
        
        # Calculate bounding rect
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            # Add margin based on line thickness
            margin = max(3, int(line_thickness) + 1)
            self.bounding_rect = QRectF(
                min(xs) - margin, min(ys) - margin,
                max(xs) - min(xs) + 2*margin, max(ys) - min(ys) + 2*margin
            )
        else:
            self.bounding_rect = QRectF(0, 0, 0, 0)
        
        # Store number of lanes for pen thickness
        self.num_lanes = num_lanes
        self.line_thickness = line_thickness
        
        # Pre-create pens with variable thickness
        self.normal_pen = QPen(QColor(100, 100, 100), line_thickness)
        self.selected_pen = QPen(QColor(50, 50, 50), line_thickness + 2)
    
    def set_selected(self, selected: bool):
        """Set whether this edge is selected/highlighted."""
        if self.is_selected != selected:
            self.is_selected = selected
            self.update()  # Trigger repaint
    
    def boundingRect(self) -> QRectF:
        return self.bounding_rect
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the edge."""
        # Save painter state to avoid warnings
        painter.save()
        try:
            painter.setPen(self.selected_pen if self.is_selected else self.normal_pen)
            
            # Draw pre-computed line segments
            for p1, p2 in self.line_segments:
                painter.drawLine(p1, p2)
        finally:
            painter.restore()


class NetworkNodeItem(QGraphicsPolygonItem):
    """Graphics item for rendering a network node/junction with its actual shape."""
    
    def __init__(self, node_id: str, x: float, y: float, shape_points: List[Tuple[float, float]] = None, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.is_selected = False
        self.setZValue(1)  # Above edges, below vehicles
        
        # Performance: Enable caching for static items
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        
        # Pre-create brushes and pens
        self.default_brush = QBrush(QColor(150, 150, 150))
        self.default_pen = QPen(QColor(100, 100, 100), 1)
        self.selected_brush = QBrush(QColor(50, 50, 50))
        self.selected_pen = QPen(QColor(30, 30, 30), 2)
        
        # Use actual junction shape if available, otherwise create a small circle
        if shape_points and len(shape_points) >= 3:
            # Create polygon from shape points (already in absolute coordinates)
            polygon_points = [QPointF(px, py) for px, py in shape_points]
            polygon = QPolygonF(polygon_points)
            self.setPolygon(polygon)
            # No need to set position - shape points are already absolute
        else:
            # Fallback: create a small circle for nodes without shape
            default_size = 3.0
            circle_points = []
            num_points = 16  # Smooth circle
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                px = default_size / 2 * math.cos(angle)
                py = default_size / 2 * math.sin(angle)
                circle_points.append(QPointF(px, py))
            polygon = QPolygonF(circle_points)
            self.setPolygon(polygon)
            # Center the circle at the node position
            self.setPos(x, y)
        
        # Default appearance
        self.setBrush(self.default_brush)
        self.setPen(self.default_pen)
    
    def set_selected(self, selected: bool):
        """Set whether this node is selected/highlighted."""
        if self.is_selected != selected:
            self.is_selected = selected
            if selected:
                self.setBrush(self.selected_brush)
                self.setPen(self.selected_pen)
            else:
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
    osm_map_loading_finished = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Performance optimizations for large networks
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        # Note: DontSavePainterState removed to avoid QPainter warnings
        # The flag can cause "Painter ended with N saved states" warnings
        # self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        
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
        
        # Zoom settings - increased max_zoom to allow deeper zooming
        self.zoom_factor = 1.15
        self.min_zoom = 0.01  # Allow zooming out more
        self.max_zoom = 100.0  # Allow much deeper zooming (was 10.0)
        self.current_zoom = 1.0
        
        # Network items
        self.edge_items = {}
        self.node_items = {}  # For rendering nodes/junctions
        self.vehicle_items = {}
        
        # OSM map tile items
        self.tile_items = {}  # Dict of (z, x, y) -> QGraphicsPixmapItem
        self.tile_worker = None
        self._network_prepare_worker = None
        self.show_osm_map = False
        self.network_parser = None  # Store reference for OSM tiles
        
        # Map offset for manual alignment adjustment
        self.map_offset_x = 0.0
        self.map_offset_y = 0.0
        
        # Map scaling factors to account for projection aspect ratio differences
        # Web Mercator (OSM) vs UTM (network) have different scaling
        self.map_scale_x = 1.0
        self.map_scale_y = 1.0
        
        # Background color
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        
        # Store network bounds for Y-flip calculation
        self._network_y_min = 0
        self._network_y_max = 0
    
    def load_network(self, network_parser, roads_junctions_only: bool = False):
        """Load network from parser asynchronously."""
        self.network_parser = network_parser
        self.clear_network()
        if self._network_prepare_worker and self._network_prepare_worker.isRunning():
            self._network_prepare_worker.terminate()
            self._network_prepare_worker.wait()
        self._network_prepare_worker = NetworkPrepareWorker(network_parser, roads_junctions_only)
        self._network_prepare_worker.finished.connect(self._on_network_prepare_finished)
        self._network_prepare_worker.start()

    def _on_network_prepare_finished(self, edges_data, nodes_data, conv_boundary, bounds, y_min, y_max):
        """Add prepared network items in batches (runs on main thread)."""
        self._network_prepare_worker = None
        self._network_y_min = y_min
        self._network_y_max = y_max
        self._network_prepare_edges = edges_data
        self._network_prepare_nodes = nodes_data
        self._network_prepare_conv_boundary = conv_boundary
        self._network_prepare_bounds = bounds
        self._network_prepare_edge_idx = 0
        self._network_prepare_node_idx = 0
        self._network_prepare_batch_size = 300
        try:
            self.setUpdatesEnabled(False)
            if self.scene is not None:
                self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        except RuntimeError:
            return
        QTimer.singleShot(0, self._add_network_batch)

    def _add_network_batch(self):
        """Add one batch of edges/nodes."""
        try:
            if self.scene is None:
                return
            edges = self._network_prepare_edges
            nodes = self._network_prepare_nodes
            batch_start = self._network_prepare_edge_idx
            batch_end = min(batch_start + self._network_prepare_batch_size, len(edges))
            for i in range(batch_start, batch_end):
                edge_id, flipped_data = edges[i]
                edge_item = NetworkEdgeItem(flipped_data)
                self.scene.addItem(edge_item)
                self.edge_items[edge_id] = edge_item
            self._network_prepare_edge_idx = batch_end
            if batch_end >= len(edges):
                node_start = self._network_prepare_node_idx
                node_end = min(node_start + self._network_prepare_batch_size, len(nodes))
                for i in range(node_start, node_end):
                    node_id, x, y, shape_points = nodes[i]
                    node_item = NetworkNodeItem(node_id, x, y, shape_points=shape_points)
                    self.scene.addItem(node_item)
                    self.node_items[node_id] = node_item
                self._network_prepare_node_idx = node_end
                if node_end >= len(nodes):
                    self._network_prepare_edges = None
                    self._network_prepare_nodes = None
                    if self.scene is not None:
                        self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
                    self.setUpdatesEnabled(True)
                    # Apply initial map offset to network (may have been set before load)
                    if self.map_offset_x != 0 or self.map_offset_y != 0:
                        for edge_item in self.edge_items.values():
                            pos = edge_item.pos()
                            edge_item.setPos(pos.x() + self.map_offset_x, pos.y() + self.map_offset_y)
                        for node_item in self.node_items.values():
                            pos = node_item.pos()
                            node_item.setPos(pos.x() + self.map_offset_x, pos.y() + self.map_offset_y)
                    conv_boundary = self._network_prepare_conv_boundary
                    bounds = self._network_prepare_bounds
                    ox, oy = self.map_offset_x, self.map_offset_y
                    if conv_boundary:
                        rect = QRectF(
                            conv_boundary['x_min'] + ox,
                            conv_boundary['y_min'] + oy,
                            conv_boundary['x_max'] - conv_boundary['x_min'],
                            conv_boundary['y_max'] - conv_boundary['y_min']
                        )
                        self.fitInView(rect, Qt.KeepAspectRatio)
                    elif bounds:
                        rect = QRectF(
                            bounds['x_min'] + ox,
                            bounds['y_min'] + oy,
                            bounds['x_max'] - bounds['x_min'],
                            bounds['y_max'] - bounds['y_min']
                        )
                        self.fitInView(rect, Qt.KeepAspectRatio)
                    self.current_zoom = 1.0
                    return
            QTimer.singleShot(0, self._add_network_batch)
        except RuntimeError:
            # View or scene deleted
            try:
                self.setUpdatesEnabled(True)
            except RuntimeError:
                pass
    
    def clear_network(self):
        """Clear all network items."""
        for item in list(self.edge_items.values()):
            self.scene.removeItem(item)
        self.edge_items.clear()
        for item in list(self.node_items.values()):
            self.scene.removeItem(item)
        self.node_items.clear()
        self.clear_vehicles()
    
    def set_network_visible(self, visible: bool):
        """Show or hide the SUMO network (edges and nodes).
        
        Args:
            visible: True to show network, False to hide
        """
        for edge_item in self.edge_items.values():
            edge_item.setVisible(visible)
        for node_item in self.node_items.values():
            node_item.setVisible(visible)
    
    def set_map_offset(self, x_offset: float, y_offset: float):
        """Set map offset for manual alignment adjustment.
        
        Args:
            x_offset: X offset in meters (positive = move map right)
            y_offset: Y offset in meters (positive = move map down)
        """
        old_x = self.map_offset_x
        old_y = self.map_offset_y
        self.map_offset_x = x_offset
        self.map_offset_y = y_offset
        delta_x = x_offset - old_x
        delta_y = y_offset - old_y

        # Always apply offset to network (edges and nodes) so it's visible regardless of OSM
        for edge_item in self.edge_items.values():
            pos = edge_item.pos()
            edge_item.setPos(pos.x() + delta_x, pos.y() + delta_y)
        for node_item in self.node_items.values():
            pos = node_item.pos()
            node_item.setPos(pos.x() + delta_x, pos.y() + delta_y)

        # If tiles are loaded, reposition them too (keeps alignment when both visible)
        if self.show_osm_map and self.tile_items:
            for item in self.tile_items.values():
                current_pos = item.pos()
                item.setPos(current_pos.x() + delta_x, current_pos.y() + delta_y)
    
    def set_osm_map_visible(self, visible: bool):
        """Show or hide OSM map tiles.
        
        Args:
            visible: True to show OSM map, False to hide
        """
        self.show_osm_map = visible
        
        if visible:
            # Load OSM tiles
            self._load_osm_tiles()
        else:
            # Clear OSM tiles
            self._clear_osm_tiles()
            # Reset background color
            self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    def _clear_osm_tiles(self):
        """Clear all OSM tile items."""
        # Cancel any ongoing download
        if self.tile_worker and self.tile_worker.isRunning():
            self.tile_worker.cancel()
            self.tile_worker.wait()
        
        for item in list(self.tile_items.values()):
            self.scene.removeItem(item)
        self.tile_items.clear()
    
    def _load_osm_tiles(self):
        """Load OSM map tiles for the current network area."""
        if not self.network_parser:
            return
        
        orig_boundary = self.network_parser.orig_boundary
        bounds = self.network_parser.get_bounds()
        conv_boundary = self.network_parser.conv_boundary
        
        if not orig_boundary or not bounds or not conv_boundary:
            print("Cannot load OSM tiles: missing boundary information")
            return
        
        # Note: Scaling is now handled in the coordinate conversion (gps_to_sumo_coords)
        # which uses pyproj to properly account for projection differences
        # The map_scale_x and map_scale_y are kept for potential manual fine-tuning if needed
        
        # Calculate actual network extent in GPS coordinates (not origBoundary which may be larger)
        # Convert network bounds (recalculated from edges) to GPS coordinates
        lon_norm_sw = (bounds['x_min'] - conv_boundary['x_min']) / (conv_boundary['x_max'] - conv_boundary['x_min']) if conv_boundary['x_max'] != conv_boundary['x_min'] else 0
        lat_norm_sw = (bounds['y_min'] - conv_boundary['y_min']) / (conv_boundary['y_max'] - conv_boundary['y_min']) if conv_boundary['y_max'] != conv_boundary['y_min'] else 0
        network_lon_min = orig_boundary['lon_min'] + lon_norm_sw * (orig_boundary['lon_max'] - orig_boundary['lon_min'])
        network_lat_min = orig_boundary['lat_min'] + lat_norm_sw * (orig_boundary['lat_max'] - orig_boundary['lat_min'])
        
        lon_norm_ne = (bounds['x_max'] - conv_boundary['x_min']) / (conv_boundary['x_max'] - conv_boundary['x_min']) if conv_boundary['x_max'] != conv_boundary['x_min'] else 1
        lat_norm_ne = (bounds['y_max'] - conv_boundary['y_min']) / (conv_boundary['y_max'] - conv_boundary['y_min']) if conv_boundary['y_max'] != conv_boundary['y_min'] else 1
        network_lon_max = orig_boundary['lon_min'] + lon_norm_ne * (orig_boundary['lon_max'] - orig_boundary['lon_min'])
        network_lat_max = orig_boundary['lat_min'] + lat_norm_ne * (orig_boundary['lat_max'] - orig_boundary['lat_min'])
        
        # Use actual network GPS bounds for tile requests (not origBoundary)
        network_gps_bounds = {
            'lon_min': network_lon_min,
            'lat_min': network_lat_min,
            'lon_max': network_lon_max,
            'lat_max': network_lat_max
        }
        
        # Cancel any previous download
        if self.tile_worker and self.tile_worker.isRunning():
            self.tile_worker.cancel()
            self.tile_worker.wait()
        
        # Clear existing tiles
        self._clear_osm_tiles()
        
        # Set light background while loading
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        
        # Determine zoom level based on actual network area size
        lon_range = network_gps_bounds['lon_max'] - network_gps_bounds['lon_min']
        lat_range = network_gps_bounds['lat_max'] - network_gps_bounds['lat_min']
        
        # Choose zoom level (higher = more detail, more tiles)
        # For Porto area (~0.5 degrees), zoom 13-14 works well
        if lon_range > 0.3 or lat_range > 0.3:
            zoom = 13
        elif lon_range > 0.1 or lat_range > 0.1:
            zoom = 14
        else:
            zoom = 15
        
        # Get tile coordinates for the boundary
        # #region agent log - Check what bounds are being used
        with open('/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix6","hypothesisId":"H","location":"simulation_view.py:_load_osm_tiles","message":"Tile request bounds comparison","data":{"orig_boundary":orig_boundary,"network_bounds":bounds,"conv_boundary":conv_boundary,"network_gps_bounds":network_gps_bounds,"zoom":zoom},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        tiles_to_download = []
        
        # Request tiles based on actual network GPS bounds (not origBoundary)
        min_tile_x, max_tile_y = self._gps_to_tile(
            network_gps_bounds['lon_min'], network_gps_bounds['lat_min'], zoom
        )
        max_tile_x, min_tile_y = self._gps_to_tile(
            network_gps_bounds['lon_max'], network_gps_bounds['lat_max'], zoom
        )
        
        # #region agent log
        with open('/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix6","hypothesisId":"H","location":"simulation_view.py:_load_osm_tiles","message":"Tile range","data":{"min_tile_x":min_tile_x,"max_tile_x":max_tile_x,"min_tile_y":min_tile_y,"max_tile_y":max_tile_y,"tile_count":(max_tile_x-min_tile_x+1)*(max_tile_y-min_tile_y+1)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        # Collect all tiles in range
        for tx in range(min_tile_x, max_tile_x + 1):
            for ty in range(min_tile_y, max_tile_y + 1):
                tiles_to_download.append((zoom, tx, ty))
        
        print(f"Loading {len(tiles_to_download)} OSM tiles at zoom {zoom}")
        
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
        if not self.show_osm_map or not self.network_parser:
            return
        
        # Get the GPS bounds of this tile
        nw_lon, nw_lat = self._tile_to_gps(x, y, z)      # NW corner (top-left of tile)
        se_lon, se_lat = self._tile_to_gps(x + 1, y + 1, z)  # SE corner (bottom-right of tile)
        
        # Clip tile GPS bounds to actual network GPS bounds (calculated from network bounds)
        # Calculate network GPS bounds from network bounds
        network_bounds = self.network_parser.get_bounds()
        conv_boundary = self.network_parser.conv_boundary
        orig_boundary = self.network_parser.orig_boundary
        
        # Clip tile GPS bounds to network GPS bounds if available
        # Use the network parser's gps_to_sumo_coords which now uses proper projection
        # Don't do reverse linear interpolation - let gps_to_sumo_coords handle it properly
        if network_bounds and conv_boundary and orig_boundary:
            # Calculate actual network GPS extent using reverse conversion
            # But use the same method as gps_to_sumo_coords for consistency
            # For now, use linear interpolation for bounds calculation (this is just for clipping)
            lon_norm_sw = (network_bounds['x_min'] - conv_boundary['x_min']) / (conv_boundary['x_max'] - conv_boundary['x_min']) if conv_boundary['x_max'] != conv_boundary['x_min'] else 0
            lat_norm_sw = (network_bounds['y_min'] - conv_boundary['y_min']) / (conv_boundary['y_max'] - conv_boundary['y_min']) if conv_boundary['y_max'] != conv_boundary['y_min'] else 0
            network_lon_min = orig_boundary['lon_min'] + lon_norm_sw * (orig_boundary['lon_max'] - orig_boundary['lon_min'])
            network_lat_min = orig_boundary['lat_min'] + lat_norm_sw * (orig_boundary['lat_max'] - orig_boundary['lat_min'])
            
            lon_norm_ne = (network_bounds['x_max'] - conv_boundary['x_min']) / (conv_boundary['x_max'] - conv_boundary['x_min']) if conv_boundary['x_max'] != conv_boundary['x_min'] else 1
            lat_norm_ne = (network_bounds['y_max'] - conv_boundary['y_min']) / (conv_boundary['y_max'] - conv_boundary['y_min']) if conv_boundary['y_max'] != conv_boundary['y_min'] else 1
            network_lon_max = orig_boundary['lon_min'] + lon_norm_ne * (orig_boundary['lon_max'] - orig_boundary['lon_min'])
            network_lat_max = orig_boundary['lat_min'] + lat_norm_ne * (orig_boundary['lat_max'] - orig_boundary['lat_min'])
            
            # Clip to network GPS bounds
            nw_lon = max(nw_lon, network_lon_min)
            nw_lat = min(nw_lat, network_lat_max)  # NW is top (max lat)
            se_lon = min(se_lon, network_lon_max)
            se_lat = max(se_lat, network_lat_min)  # SE is bottom (min lat)
        
        # Convert GPS corners to SUMO coordinates using proper projection transformation
        # This now uses pyproj transformer if available, which handles projection distortion correctly
        nw_result = self.network_parser.gps_to_sumo_coords(nw_lon, nw_lat)
        se_result = self.network_parser.gps_to_sumo_coords(se_lon, se_lat)
        
        if nw_result is None or se_result is None:
            return
        
        nw_x, nw_y = nw_result
        se_x, se_y = se_result
        
        # Clip to network bounds (recalculated from actual edges)
        network_bounds = self.network_parser.get_bounds()
        if network_bounds:
            nw_x = max(nw_x, network_bounds['x_min'])
            se_x = min(se_x, network_bounds['x_max'])
            nw_y = max(nw_y, network_bounds['y_min'])
            se_y = min(se_y, network_bounds['y_max'])
        
        # Apply the same Y-flip used for the network
        # flip_y(y) = y_max + y_min - y
        nw_y_flipped = self._network_y_max + self._network_y_min - nw_y
        se_y_flipped = self._network_y_max + self._network_y_min - se_y
        
        # Calculate tile dimensions in SUMO coordinates
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
        # Apply manual offset for alignment adjustment
        pos_x = min(nw_x, se_x) + self.map_offset_x
        pos_y = min(nw_y_flipped, se_y_flipped) + self.map_offset_y
        
        item.setPos(pos_x, pos_y)
        item.setZValue(-10)  # Behind everything else
        
        self.scene.addItem(item)
        self.tile_items[(z, x, y)] = item
    
    def _on_tiles_done(self):
        """Called when all tiles are downloaded."""
        print(f"OSM map tiles loaded: {len(self.tile_items)} tiles")
        self.osm_map_loading_finished.emit()
    
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
        
        # Check zoom limits (only min_zoom, no max_zoom limit for unlimited zoom in)
        new_zoom = self.current_zoom * zoom
        if new_zoom >= self.min_zoom:
            self.scale(zoom, zoom)
            self.current_zoom = new_zoom
    
    def zoom_in(self):
        """Zoom in programmatically - unlimited zoom in."""
        # No max_zoom limit - allow unlimited zooming in
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

