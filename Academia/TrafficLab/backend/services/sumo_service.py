import os
import json
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from models.entities import Junction, Road, Zone, Vehicle
from models.eta_inference import Inference
from sumolib import net as sumo_net
import traci
import traci.constants as tc

class SUMOSimulation:
    """
    Simplified SUMO simulation service for traffic simulation
    """
    
    def __init__(self, sumo_config_path: str, sim_config_path: str):
        """
        Initialize the SUMO simulation with configuration files
        """
        self.sumo_config_path = sumo_config_path
        self.sim_config_path = sim_config_path
        
        # Load configuration
        with open(sim_config_path) as f:
            self.config = json.load(f)
        
        # Extract network file path from SUMO config
        tree = ET.parse(sumo_config_path)
        root = tree.getroot()
        net_file_elem = root.find('.//net-file')
        if net_file_elem is not None:
            net_file_path = os.path.join(os.path.dirname(sumo_config_path), net_file_elem.get('value'))
            self.net = sumo_net.readNet(net_file_path)
        else:
            self.net = sumo_net.readNet(sumo_config_path)
        
        # Initialize simulation state
        self.simulation_running = False
        # Start at 5:00 AM (5*3600 = 18000 seconds)
        self.current_step = 18000  
        print(f"🔍 Starting simulation at step {self.current_step} which is {self.get_simulation_time(offset=0)}")
        self.data_loaded = False
        self.trips_added = 0  # Counter for total trips added
        
        # Initialize ETA inference model
        try:
            self.eta_inference = Inference(
                checkpoint_path="models/moe_best.pt",
                config_path="models/config.yaml",
                seed=42
            )
            print("✅ ETA Inference model initialized successfully")
        except Exception as e:
            print(f"⚠️  Failed to initialize ETA inference model: {e}")
            self.eta_inference = None
        
        # Thread safety for TraCI access
        self.traci_lock = threading.Lock()
        
        # Initialize data structures
        self.vehicles = {}
        self.junctions = {}
        self.roads = {}
        self.zones = {}
        self.vehicles_in_route = set()
        self.stagnant_vehicles = set()  # Track stagnant vehicles

        # Trips data for playback
        self.trips_data = {}
        self.max_step = 0
        
        # User-defined vehicle tracking
        self.user_defined_vehicles = {}  # vehicle_id -> {id, path, travel_distance, start_time, end_time}
        self.finished_vehicles = []  # List of completed vehicles for frontend events
        self.is_playing = False
        
        # Simulation thread
        self.simulation_thread = None
        self.simulation_running = False
        
        # Load all static data
        self.read_static_entities_from_sumo()
        
        # Load trips data
        self.load_trips_data()

        # Start TraCI connection
        try:
            traci.start(["sumo", "-c", "sumo/urban_three_zones.sumocfg", "--start"])
            print("✅ TraCI connection established")
        except Exception as e:
            print(f"❌ Failed to start TraCI: {e}")
            return
        
        
        # Start endless simulation automatically (commented out for now)
        self.start_endless_simulation(traci)
        
        # Network data for visualization
        self.network_data = None
        self.network_bounds = None
        
    
    def load_trips_data(self):
        """Load trips data from JSON file"""
        try:
            trips_file = os.path.join(os.path.dirname(__file__), '..', 'sumo', 'trips.json')
            with open(trips_file, 'r') as f:
                self.trips_data = json.load(f)
            
            # Get the maximum step from the data
            self.max_step = max(int(step) for step in self.trips_data.keys())
            print(f"📊 Loaded trips data: {len(self.trips_data)} time steps, max step: {self.max_step}")
            
        except Exception as e:
            print(f"❌ Error loading trips data: {e}")
            self.trips_data = {}
            self.max_step = 0
    
    def get_trips_for_current_step(self):
        """Get trips data for the current simulation step"""
        step_key = str(self.current_step % self.max_step)
        if step_key in self.trips_data:
            return self.trips_data[step_key]
        return []
    

    def start_trips_playback(self):
        """Start the trips playback"""
        self.is_playing = True
        self.current_step = 0
        print("▶️ Started trips playback")
    
    def stop_trips_playback(self):
        """Stop the trips playback"""
        self.is_playing = False
        print("⏹️ Stopped trips playback")
    
        """Get current playback status"""
    def get_playback_status(self):
                return {
            "is_playing": self.is_playing,
            "current_step": self.current_step,
            "max_step": self.max_step,
            "trips_count": len(self.get_trips_for_current_step())
        }
    
    def start_endless_simulation(self, traci):
        """Start the endless simulation in a background thread"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            return  # Already running
            
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self._endless_simulation_loop, args=(traci,), daemon=True)
        self.simulation_thread.start()
        print("🔄 Started endless simulation thread")
    
    def stop_endless_simulation(self):
        """Stop the endless simulation"""
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
        print("🛑 Stopped endless simulation thread")
    
    def _endless_simulation_loop(self, traci):
        """The main endless simulation loop with TraCI"""
        print("🚀 Starting endless simulation loop with TraCI...")
        
       
        while self.simulation_running:
            try:
                # Get trips for current step
                trips = self.get_trips_for_current_step()
                
                # Process trips (add vehicles to simulation using TraCI)
                self.process_trips_for_step(trips)
                
                # Advance simulation step (thread-safe)
                with self.traci_lock:
                    traci.simulationStep()

                # Debug: Show which vehicles were removed
                self.vehicles_in_route = set([v["id"] for v in self.get_active_vehicles().get("vehicles", [])])
                
                # Debug: Log vehicle tracking state
                if self.user_defined_vehicles:
                    # Check user-defined vehicles for completion
                    self._check_user_defined_vehicles(traci)
                
                    print(f"🔍 Step {self.current_step}: {len(self.vehicles_in_route)} vehicles in route")
                    print(f"🔍 User-defined vehicles: {list(self.user_defined_vehicles.keys())}")
                    for vid in self.user_defined_vehicles.keys():
                        in_route = vid in self.vehicles_in_route
                        print(f"🔍   {vid}: {'IN' if in_route else 'OUT'} of route")
                
                # Move to next step (with cycling)
                self.current_step += 1
                # Small sleep to prevent overwhelming TraCI connection
                time.sleep(0.05)  # 100ms sleep for stability
                
            except Exception as e:
                print(f"❌ Error in simulation loop: {e}")
                # Try to reconnect TraCI if connection is lost
                if "Connection already closed" in str(e) or "Not connected" in str(e):
                    print("🔄 Attempting to reconnect TraCI...")
                    try:
                        with self.traci_lock:
                            time.sleep(2)  # Wait before reconnecting
                            traci.start(["sumo", "-c", "sumo/urban_three_zones.sumocfg", "--start"])
                        print("✅ TraCI reconnected successfully")
                    except Exception as reconnect_error:
                        print(f"❌ Failed to reconnect TraCI: {reconnect_error}")
                        time.sleep(5)  # Wait longer before retrying
                else:
                    time.sleep(1.0)  # Wait before retrying
        
        # Keep TraCI connection open - only close on server shutdown
        print("🏁 Endless simulation loop ended")
    
    def _check_user_defined_vehicles(self, traci):
        """Check if user-defined vehicles have finished their journey"""
        vehicles_to_remove = []
        
        # Check user-defined vehicles for completion        
        for vehicle_id, vehicle_data in self.user_defined_vehicles.items():
            try:
                # Only check for completion after the delay period
                with self.traci_lock:
                    traci_time = traci.simulation.getTime()
                    # Check if enough time has passed since vehicle was added (use simulation step)
                    if self.current_step <= vehicle_data.get('start_time', 0):
                        continue
                    
                    # Also check if we're past the check_after_step
                    if self.current_step <= vehicle_data.get('check_after_step', 0):
                        continue
                
                # Check if vehicle is still in the simulation
                is_in_route = vehicle_id in self.vehicles_in_route
                print(f"🔍 Vehicle {vehicle_id} in route: {is_in_route}")
                
                if not is_in_route:
                    # Vehicle has finished its journey
                    print(f"🎯 Vehicle {vehicle_id} finished! Adding to finished vehicles...")
                    
                    vehicle_data['end_time'] = self.current_step
                    
                    # Calculate travel distance (simplified - could be enhanced)
                    vehicle_data['travel_distance'] = self._calculate_travel_distance(vehicle_data['path'])
                    
                    # Add to finished vehicles list for frontend events
                    self.finished_vehicles.append(vehicle_data.copy())
                    
                    # Mark for removal from tracking
                    vehicles_to_remove.append(vehicle_id)
                    
                    duration = vehicle_data['end_time'] - vehicle_data['start_time']
                    print(f"🎯 Vehicle {vehicle_id} finished journey at simulation step {self.current_step}")
                    print(f"📊 Travel distance: {vehicle_data['travel_distance']:.2f}m")
                    print(f"⏱️ Journey duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
                    
                    # Show prediction accuracy if prediction exists
                    if 'prediction' in vehicle_data:
                        prediction = vehicle_data['prediction']
                        predicted_duration = prediction['predicted_travel_time']
                        actual_duration = duration
                        
                        # Calculate accuracy, handling division by zero
                        if actual_duration > 0:
                            accuracy = 1.0 - abs(predicted_duration - actual_duration) / actual_duration
                        else:
                            accuracy = 0.0  # Can't calculate accuracy for zero duration
                        
                        print(f"🎯 Prediction Accuracy:")
                        print(f"   Predicted: {predicted_duration:.1f}s")
                        print(f"   Actual: {actual_duration:.1f}s")
                        if actual_duration > 0:
                            print(f"   Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")
                        else:
                            print(f"   Accuracy: N/A (actual duration was 0s)")
                        print(f"   Confidence: {prediction['confidence']:.2f}")
                    
                    print(f"📋 Finished vehicles count: {len(self.finished_vehicles)}")
                    print(f"🎯 DEBUG: Added vehicle {vehicle_id} to finished_vehicles list")
                    
            except Exception as e:
                print(f"❌ Error checking vehicle {vehicle_id}: {e}")
        
        # Remove finished vehicles from tracking
        for vehicle_id in vehicles_to_remove:
            del self.user_defined_vehicles[vehicle_id]
            print(f"🗑️ Removed finished vehicle {vehicle_id} from tracking")
    
    def _calculate_travel_distance(self, path_edges):
        """Calculate total travel distance for a path"""
        total_distance = 0.0
        for edge_id in path_edges:
            if edge_id in self.roads and hasattr(self.roads[edge_id], 'length'):
                total_distance += self.roads[edge_id].length
        return total_distance
            
    
    def predict_eta(self, vehicle_id, depart_pos, route_edges, route_distance, simulation_step_depart_time):
        """
        ETA prediction using ML model
        
        Args:
            vehicle_id: ID of the vehicle
            depart_pos: Departure position of the vehicle
            route_edges: List of edges in the route
            route_distance: Total distance of the route in meters
            simulation_step_depart_time: Current simulation step time
            
        Returns:
            dict: Prediction result with ETA and traffic impact
            
        Raises:
            RuntimeError: If ML model is unavailable or prediction fails
        """
        # Check if ML model is available (fatal error if not)
        if self.eta_inference is None:
            raise RuntimeError("ETA inference model is not available - this is a fatal error")
        
        try:
            # Get vehicle position and zone information
            current_road = self.roads[route_edges[0]]
            if current_road:
                current_junction_id = current_road.to_junction
                current_junction = self.junctions[current_junction_id]
                current_x, current_y = current_junction.x, current_junction.y   
            else:
                current_x, current_y = 0.0, 0.0
                print(f"⚠️ Could not get start poition for vehicle {vehicle_id}, using default position")
            dest_road = self.roads[route_edges[-1]]
            if dest_road:
                dest_junction_id = dest_road.to_junction
                dest_junction = self.junctions[dest_junction_id]
                dest_x, dest_y = dest_junction.x, dest_junction.y
                print(f"🔍 Destination position: {dest_x}, {dest_y} on edge {route_edges[-1]}")
            else:
                dest_x, dest_y = 0.0, 0.0
                print(f"⚠️ Could not get shape for edge {route_edges[-1]}, using default position")

            curr_road = self.roads[route_edges[0]]
            if curr_road:
                zone = curr_road.zone
                num_lanes = curr_road.num_lanes
                print(f"🔍 Current zone: {zone} on edge {route_edges[0]}")
            else:
                zone = "A"
                num_lanes = 2
                print(f"⚠️ Could not get zone and num_lanes for edge {route_edges[0]}, using default values")
            
            # Prepare vehicle info for ML model
            vehicle_info = {
                "veh_id": vehicle_id,
                "current_x": current_x,
                "current_y": current_y,
                "destination_x": dest_x,
                "destination_y": dest_y,
                "current_edge_num_lanes": num_lanes,
                "zone": zone,
                "route_length": route_distance,
                "current_edge_id": route_edges[0]
            }
            
            # Prepare route info for ML model
            route_info = {
                "route_edges": route_edges,
                "route_length": route_distance
            }
            
            # # Get ML prediction
            predicted_eta_seconds, avg_change = self.eta_inference.predict_eta(
                vehicle_info, route_info, simulation_step_depart_time
            )

            prediction_result = {
                'predicted_travel_time': predicted_eta_seconds,
                'predicted_eta': predicted_eta_seconds + simulation_step_depart_time,
                'traffic_impact': avg_change
            }
            
            print(f"🎯 ETA Prediction for {vehicle_id} (Trained Model):")
            print(f"   Predicted ETA: {predicted_eta_seconds:.1f}s ({predicted_eta_seconds/60:.1f} min)")
            print(f"   Traffic Impact: {avg_change:.2f}s average change")
            
            return prediction_result
            
        except Exception as e:
            print(f"❌ Trained model ETA prediction failed for {vehicle_id}: {e}")
            raise RuntimeError(f"ETA prediction failed for vehicle {vehicle_id}: {e}") from e
    
    
    def get_finished_vehicles(self):
        """Get finished user-defined vehicles"""
        finished = self.finished_vehicles.copy()
        # Don't clear immediately - let vehicles stay in the list for a while
        # The frontend will handle multiple notifications for the same vehicle
        return finished
    
    def clear_finished_vehicles(self):
        """Clear all finished vehicles"""
        self.finished_vehicles.clear()
        print("🧹 Cleared all finished vehicles")
    
    def get_vehicle_prediction(self, vehicle_id):
        """Get prediction for a specific vehicle"""
        if vehicle_id in self.user_defined_vehicles:
            vehicle_data = self.user_defined_vehicles[vehicle_id]
            if 'prediction' in vehicle_data:
                return vehicle_data['prediction']
        return None
    
    def process_trips_for_step(self, trips):
        """Process trips for the current step - add vehicles to simulation using TraCI"""
        if not trips:
            return
        
        # Increment trips counter
        self.trips_added += len(trips)
            
        for trip in trips:
            try:
                # Add vehicle to SUMO simulation using TraCI
                self._add_vehicle_to_traci(trip)
            except Exception as e:
                print(f"❌ Error adding vehicle {trip.get('vehicle_id', 'unknown')}: {e}")
    
    
    
    def _add_vehicle_to_traci(self, trip):
        """Add vehicle to SUMO simulation using TraCI"""
        # Generate unique vehicle ID to avoid collisions
        vehicle_unique_suffix = str(uuid.uuid4())[:8]
        base_vehicle_id = trip.get('vehicle_id', 'vehicle')
        vehicle_id = f"{base_vehicle_id}_{vehicle_unique_suffix}"
        base_route_id = trip.get('route_id', 'route')
        route_id = f"{base_route_id}_{vehicle_unique_suffix}"
        full_route_edges = trip.get('full_route_edges') 
        type = trip.get('type') 
        depart = trip.get('depart') 
        departPos = trip.get('departPos') 
        is_stagnant = trip.get('is_stagnant') 
        current_edge = trip.get('current_edge') 
        current_x = trip.get('current_x') 
        current_y = trip.get('current_y') 
        destination_edge = trip.get('destination_edge') 
        destination_x = trip.get('destination_x') 
        destination_y = trip.get('destination_y') 
            
        try:
            # Thread-safe TraCI access
            with self.traci_lock:
                traci.route.add(routeID=route_id, edges=full_route_edges)
                # Add vehicle to SUMO simulation
                # Use current simulation time instead of trip departure time
                current_time = traci.simulation.getTime()
                traci.vehicle.add(
                    vehID=vehicle_id,
                    routeID=route_id,
                    typeID=type,
                    depart=current_time,  # Use current time instead of trip depart time
                    departPos=departPos,
                    departSpeed=0,
                    departLane="0"
                )
                
                # Subscribe to vehicle updates
                traci.vehicle.subscribe(vehicle_id, [tc.VAR_ROAD_ID, tc.VAR_POSITION, tc.VAR_SPEED])
            
            # Add to tracking
            self.vehicles_in_route.add(vehicle_id)
            
            # Track stagnant vehicles based on JSON data
            if is_stagnant:
                self.stagnant_vehicles.add(vehicle_id)
            
        except Exception as e:
            print(f"❌ Failed to add vehicle {vehicle_id} to TraCI: {e}")
    

    def is_data_loaded(self) -> bool:
        """
        Check if all static data has been loaded
        """
        return self.data_loaded

    def read_static_entities_from_sumo(self):
        """
        Load entities from SUMO network into memory
        """
        zone_objects = {}

        # 1. Collect junctions by zone attribute
        print("📊 Processing junctions...")
        junction_count = 0
        for junction in self.net.getNodes():
            zone_attr = junction.getParam("zone")
            if not zone_attr:
                continue
            zone_id = zone_attr.upper()
            if zone_id not in zone_objects:
                zone_objects[zone_id] = Zone(zone_id)
                print(f"Zone {zone_id} created.")

            junction_count += 1
            if junction_count % 100 == 0:
                print(f"📊 Processed {junction_count} junctions...")

            # Create junction entity
            junc = Junction(
                junction_id=junction.getID(),
                x=junction.getCoord()[0],
                y=junction.getCoord()[1],
                junc_type=junction.getType(),
                zone=zone_id
            )
            self.junctions[junc.id] = junc
            zone_objects[zone_id].add_junction(junc.id)

        # 2. Collect edges by zone attribute and update junction connections
        print("🛣️  Processing roads...")
        road_count = 0
        for edge in self.net.getEdges():
            zone_attr = edge.getParam("zone")
            if not zone_attr:
                    continue
                    
            road_count += 1
            if road_count % 100 == 0:
                print(f"🛣️  Processed {road_count} roads...")
            zone_id = zone_attr.upper()
            if zone_id not in zone_objects:
                zone_objects[zone_id] = Zone(zone_id)
                print(f"Zone {zone_id} created.")

            # Create road entity
            road = Road(
                road_id=edge.getID(),
                from_junction=edge.getFromNode().getID(),
                to_junction=edge.getToNode().getID(),
                speed=edge.getSpeed(),
                length=edge.getLength(),
                num_lanes=edge.getLaneNumber(),
                zone=zone_id
            )
            
            # Add detailed shape information for express edges
            if self._is_express_edge(edge.getID()):
                road.shape_points = self._get_edge_shape_points(edge)
            self.roads[road.id] = road
            zone_objects[zone_id].add_edge(road.id)

            # Update junction connections
            from_junction = self.junctions.get(road.from_junction)
            to_junction = self.junctions.get(road.to_junction)
            if from_junction:
                from_junction.add_outgoing(road.id)
            if to_junction:
                to_junction.add_incoming(road.id)

        # 3. Store zones
        for zone in zone_objects.values():
            self.zones[zone.id] = zone
        
        print(f"✅ Successfully loaded {len(self.junctions)} junctions, {len(self.roads)} roads, {len(self.zones)} zones")
        
        # Mark data as loaded
        self.data_loaded = True
        print("✅ All static data loaded successfully!")
    
    def _is_express_edge(self, edge_id):
        """
        Check if an edge is an express edge based on naming patterns
        """
        # Express edges are the main connecting roads between zones
        # They typically start with -E or E
        if edge_id.startswith('-E') or edge_id.startswith('E'):
            return True
        
        # Also check for cross-zone connections (edges that connect different zones)
        express_patterns = [
            # Diagonal connections between zones
            'BI', 'BH', 'BG', 'BF', 'BE', 'BD', 'BC', 'BA', 'BB',
            # Cross-zone connections
            'AQ', 'AP', 'AO', 'AN', 'AM', 'AL', 'AK', 'AJ', 'AI', 'AH', 'AG', 'AF', 'AE', 'AD', 'AC', 'AB', 'AA'
        ]
        
        # Check if edge connects different zones (not internal to a zone)
        for pattern in express_patterns:
            if edge_id.startswith(pattern) and len(edge_id) > 3:
                return True
        return False
    
    def _get_edge_shape_points(self, edge):
        """
        Get detailed shape points for an edge from SUMO network
        """
        try:
            # Get the first lane of the edge (all lanes should have similar shape)
            lanes = edge.getLanes()
            if not lanes:
                print(f"⚠️  No lanes found for edge {edge.getID()}")
                return []
            
            first_lane = lanes[0]
            shape = first_lane.getShape()
            
            # Convert shape points to list of [x, y] coordinates
            shape_points = []
            for point in shape:
                shape_points.append([float(point[0]), float(point[1])])
            
            # Only log if there are issues with shape points
            if len(shape_points) < 2:
                print(f"⚠️  Edge {edge.getID()} has insufficient shape points: {len(shape_points)}")
            return shape_points
        except Exception as e:
            print(f"⚠️  Could not get shape points for edge {edge.getID()}: {e}")
            return []
        

    def get_vehicle(self, vehicle_id):
        """
        Get vehicle by ID
        """
        return self.vehicles.get(vehicle_id)

    def get_all_vehicles(self):
        """
        Get all vehicles
        """
        return list(self.vehicles.values())

    def get_zones(self):
        """
        Get all zones
        """
        return self.zones

    def start_simulation(self) -> bool:
        """
        Start SUMO simulation
        """
        if not self.is_data_loaded():
            print("❌ Cannot start simulation: Data not loaded yet")
            return False
        
        try:
            self.simulation_running = True
            print("🚀 Simulation started")
            return True
        
        except Exception as e:
            print(f"Error starting SUMO simulation: {e}")
        return False
    
    def stop_simulation(self):
        """
        Stop the SUMO simulation
        """
        self.simulation_running = False
        print("🛑 Simulation stopped")

    def get_simulation_status(self):
        """
        Get current simulation status
        """
        return {
            "is_running": self.simulation_running,
            "vehicles": len(self.vehicles),
            "vehicles_in_route": len(self.vehicles_in_route),
            "trips_added": self.trips_added,
            "current_step": self.current_step,
            "data_loaded": self.data_loaded,
            "simulation_type": "endless",
            "simulation_time": self.get_simulation_time()
        }
        
    def get_simulation_time(self, offset=0):
        """
        Convert simulation step to HH:MM:SS format
        """
        # Assuming 1 step = 1 second in simulation time
        total_seconds = int(self.current_step + offset)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    

    def get_active_vehicles(self):
        """
        Get currently active vehicles from TraCI
        """
        vehicles = []
        
        try:
            # Thread-safe TraCI access
            with self.traci_lock:
                # Get all vehicle IDs from TraCI
                vehicle_ids = traci.vehicle.getIDList()
                
                vehicles_data = []
                for vehicle_id in vehicle_ids:
                    try:
                        # Get vehicle data from TraCI
                        position = traci.vehicle.getPosition(vehicle_id)
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        road_id = traci.vehicle.getRoadID(vehicle_id)
                        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                        
                        # Check if vehicle is stagnant from JSON data
                        is_stagnant = vehicle_id in self.stagnant_vehicles
                        
                        vehicles_data.append({
                            "id": vehicle_id,
                            "x": position[0],
                            "y": position[1],
                            "speed": speed,
                            "edge": road_id,
                            "type": vehicle_type,
                            "status": "stagnant" if is_stagnant else "driving"
                        })
                        
                    except Exception as e:
                        print(f"❌ Error getting vehicle {vehicle_id}: {e}")
                        continue
                
                # Return vehicles data
                return {"vehicles": vehicles_data}
                    
        except Exception as e:
            print(f"❌ Error getting active vehicles from TraCI: {e}")
            # Try to reconnect if connection is lost
            if "Connection already closed" in str(e) or "Not connected" in str(e):
                print("🔄 Attempting to reconnect TraCI for vehicle retrieval...")
                try:
                    with self.traci_lock:
                        time.sleep(1)
                        traci.start(["sumo", "-c", "sumo/urban_three_zones.sumocfg", "--start"])
                    print("✅ TraCI reconnected for vehicle retrieval")
                except Exception as reconnect_error:
                    print(f"❌ Failed to reconnect TraCI: {reconnect_error}")
            # Fallback to empty list
            return {"vehicles": []}
        
        return {"vehicles": vehicles}
    
    def update_stagnant_vehicles(self, vehicle_id, is_stagnant):
        """
        Update the stagnant status of a vehicle
        """
        if is_stagnant:
            self.stagnant_vehicles.add(vehicle_id)
        else:
            self.stagnant_vehicles.discard(vehicle_id)
    
    def get_stagnant_vehicles(self):
        """
        Get list of currently stagnant vehicles
        """
        return list(self.stagnant_vehicles)

    def get_network_data(self):
        """
        Get network data for visualization
        """
        # Calculate bounds for the network
        if self.junctions:
            x_coords = [j.x for j in self.junctions.values()]
            y_coords = [j.y for j in self.junctions.values()]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
        else:
            min_x = max_x = min_y = max_y = 0
        
        # Debug: Check for express edges with invalid shape points
        express_edges_with_issues = []
        for road in self.roads.values():
            if hasattr(road, 'shape_points') and road.shape_points:
                # Check if shape points look like junction IDs instead of coordinates
                if (isinstance(road.shape_points[0], str) and 
                    not road.shape_points[0].includes(',')):
                    express_edges_with_issues.append(road.id)
        
        if express_edges_with_issues:
            print(f"⚠️  Express edges with junction ID shape points: {express_edges_with_issues[:3]}")
        
        return {
            "junctions": list(self.junctions.values()),
            "edges": list(self.roads.values()),  # Frontend expects "edges" not "roads"
            "roads": list(self.roads.values()),  # Keep both for compatibility
            "zones": list(self.zones.values()),
            "bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y
            }
        }

    # Placeholder methods for API compatibility
    def add_vehicle(self, route):
        """Add a vehicle to the simulation"""
        vehicle_unique_suffix = str(uuid.uuid4())[:8]
        vehicle_id = f"veh_{vehicle_unique_suffix}"
        return vehicle_id


    def get_route_distance(self, start_coords, end_coords):
        """Calculate route distance"""
        import math
        return math.sqrt((end_coords[0] - start_coords[0])**2 + (end_coords[1] - start_coords[1])**2)

    def get_vehicle_position(self, vehicle_id):
        """Get vehicle position"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle:
            return {"x": vehicle.current_x, "y": vehicle.current_y}
        return None

    def get_vehicle_speed(self, vehicle_id):
        """Get vehicle speed"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle:
            return vehicle.speed
            return 0.0
            
    def calculate_route(self, start_coords, end_coords):
        """Calculate route between two points"""
        return {
            "edges": ["edge1", "edge2", "edge3"],
            "distance": self.get_route_distance(start_coords, end_coords),
            "duration": 300.0
        }

    def calculate_route_by_edges(self, start_edge, end_edge):
        """Calculate route between two edges using SUMO routing"""
        try:
            print(f"🛣️ Calculating route from {start_edge} to {end_edge}")
            print(f"🔍 Available roads: {len(self.roads)}")
            print(f"🔍 Simulation running: {self.simulation_running}")
            
            # Validate that both edges exist in our network
            if start_edge not in self.roads:            
                print(f"❌ Start edge '{start_edge}' not found in {list(self.roads.keys())[:10]}...")
                return {
                    "error": f"Start edge '{start_edge}' not found in network"
                }
            if end_edge not in self.roads:
                print(f"❌ End edge '{end_edge}' not found in {list(self.roads.keys())[:10]}...")
                return {
                    "error": f"End edge '{end_edge}' not found in network"
                }
            
            # Use TraCI routing
            with self.traci_lock:
                # Check if TraCI is connected and simulation is running
                try:
                    traci.simulation.getMinExpectedNumber()
                except Exception as e:
                    print(f"❌ TraCI not connected for route calculation: {e}")
                    return {"error": f"TraCI not connected: {e}"}
                
                # Try to use TraCI routing
                try:
                    route_result = traci.simulation.findRoute(start_edge, end_edge)
                except Exception as route_error:
                    print(f"❌ TraCI route calculation failed: {route_error}")
                    return {"error": f"Route calculation failed: {route_error}"}
                
                if route_result and route_result.edges:
                    edge_list = list(route_result.edges)
                    distance = route_result.length
                    
                    print(f"✅ Route calculated via TraCI: {len(edge_list)} edges, {distance:.1f}m")
                    return {
                        "edges": edge_list,
                        "distance": distance,
                        "duration": distance / 13.89,  # Assume 50 km/h average speed
                        "start_edge": start_edge,
                        "end_edge": end_edge
                    }
                else:
                    print(f"❌ No route found between {start_edge} and {end_edge}")
                    return {"error": "No route found"}
                    
        except Exception as e:
            print(f"❌ Error in calculate_route_by_edges: {e}")
            return {"error": f"Route calculation failed: {str(e)}"}
    
    
    def debug_system_state(self):
        """
        Debug method to print current system state
        """
        print("\n" + "="*50)
        print("🐛 DEBUG: System State")
        print("="*50)
        print(f"📊 Junctions loaded: {len(self.junctions)}")
        print(f"🛣️  Roads loaded: {len(self.roads)}")
        print(f"🏘️  Zones loaded: {len(self.zones)}")
        print(f"🚗 Vehicles loaded: {len(self.vehicles)}")
        
        print("\n📍 Zone Details:")
        for zone_id, zone in self.zones.items():
            print(f"  Zone {zone_id}: {len(zone.edges)} edges, {len(zone.junctions)} junctions, {len(zone.current_vehicles)} vehicles")
        
        print("="*50)
    
    def add_journey_vehicle(self, start_edge, end_edge, route_edges):
        """Add a vehicle to the simulation for a specific journey"""
        try:
            # Generate unique vehicle ID
            vehicle_unique_suffix = str(uuid.uuid4())[:8]
            vehicle_id = f"journey_vehicle_{vehicle_unique_suffix}"
            
            # Create route from the calculated route edges
            route_str = " ".join(route_edges)
            
            print(f"🚗 Adding journey vehicle {vehicle_id}")
            print(f"🛣️ Route: {route_str}")
            
            # Validate route edges
            if not route_edges or not isinstance(route_edges, list):
                raise Exception(f"Invalid route edges: {route_edges}")
            
            # Check if all route edges exist in the network
            missing_edges = []
            for edge_id in route_edges:
                if edge_id not in self.roads:
                    missing_edges.append(edge_id)
                    print(f"⚠️ Route edge {edge_id} not found in network")
            
            if missing_edges:
                raise Exception(f"Route contains invalid edges: {missing_edges}")
            
            
            with self.traci_lock:
                # Check if TraCI is connected
                try:
                    traci.simulation.getMinExpectedNumber()
                    print(f"✅ TraCI connection is active")
                except Exception as e:
                    print(f"❌ TraCI connection error: {e}")
                    raise Exception(f"TraCI not connected: {e}")
                
                # Create route first with truly unique ID
                unique_suffix = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for uniqueness
                route_id = f"route_{vehicle_id}_{start_edge}_{end_edge}_{unique_suffix}"
                try:
                    traci.route.add(routeID=route_id, edges=route_edges)
                    print(f"✅ Route {route_id} created successfully")
                except Exception as route_error:
                    print(f"❌ Failed to create route {route_id}: {route_error}")
                    print(f"❌ Route edges: {route_edges}")
                    raise Exception(f"Could not add route '{route_id}': {route_error}")

                # Depart time needs to be in jumps of 30 steps for example if current step is 100, then depart time should be 120
                current_time = traci.simulation.getTime()
                depart_time = current_time + 30 - current_time % 30
                
                # Calculate simulation step equivalent for consistency
                simulation_step_depart_time = self.current_step + 30 - self.current_step % 30
                route_distance = self._calculate_travel_distance(route_edges)
                # Depart pos needs to be the beginning of the road
                depart_pos = 0 
                
                # Add vehicle to simulation FIRST
                try:
                    traci.vehicle.add(
                        vehID=vehicle_id,
                        routeID=route_id,
                        typeID="user_defined",
                        depart=depart_time,  # Use current time instead of trip depart time
                        departPos=depart_pos,
                        departSpeed=0,
                        departLane="0"
                    )
                    print(f"✅ Vehicle {vehicle_id} added to simulation successfully")
                except Exception as vehicle_error:
                    print(f"❌ Failed to add vehicle {vehicle_id}: {vehicle_error}")
                    raise Exception(f"Could not add vehicle '{vehicle_id}': {vehicle_error}")
                traci.vehicle.subscribe(vehicle_id, [tc.VAR_ROAD_ID, tc.VAR_POSITION, tc.VAR_SPEED])
                start_time_string = self.get_simulation_time(offset=30-self.current_step%30)
                print(f"🚗 Vehicle {vehicle_id} added to TraCI with route {route_id} at {start_time_string}")
                print(f"🚗 Depart time: {depart_time}, Depart pos: {depart_pos}")
            
            # Add to tracking
            self.vehicles_in_route.add(vehicle_id)
            # Track user-defined vehicle
            self.user_defined_vehicles[vehicle_id] = {
                'id': vehicle_id,
                'path': route_edges,
                'travel_distance': route_distance,
                'start_time': simulation_step_depart_time,  # Use simulation step for consistency with frontend
                'start_time_string': start_time_string,
                'end_time': None,
                'check_after_step': self.current_step + 10  # Don't check for completion until 10 steps later
            }
            
            # Generate ETA prediction
            print(f"🎯 Generating ETA prediction for {vehicle_id}...")
            prediction = self.predict_eta(vehicle_id, depart_pos, route_edges, route_distance, simulation_step_depart_time)
            self.user_defined_vehicles[vehicle_id]['prediction'] = prediction
            
            print(f"✅ Journey vehicle {vehicle_id} added to simulation and tracking")
            return vehicle_id
            
        except Exception as e:
            print(f"❌ Error adding journey vehicle: {e}")
            raise e