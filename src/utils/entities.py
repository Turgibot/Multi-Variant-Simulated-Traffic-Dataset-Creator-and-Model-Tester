import json
import logging
import os
import random
import re
from collections import defaultdict
from logging.handlers import RotatingFileHandler

import traci.constants as tc
from tqdm import tqdm


class Junction:
    """
    Represents a fixed junction point in the traffic network.
    Holds incoming and outgoing road connections and basic spatial metadata.
    """
    def __init__(self, junction_id, x=0.0, y=0.0, junc_type="priority", zone=None):
        self.id = junction_id
        self.x = x
        self.y = y
        self.type = junc_type
        self.zone = zone
        self.node_type = 0  # 0 for junction, 1 for vehicle

        self.incoming_roads = set()  # Set of incoming road IDs
        self.outgoing_roads = set()  # Set of outgoing road IDs

    def add_incoming(self, road_id):
        self.incoming_roads.add(road_id)

    def add_outgoing(self, road_id):
        self.outgoing_roads.add(road_id)

    def to_dict(self):
        return {
            "id": self.id,
            "node_type": self.node_type,
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "zone": self.zone,
            "incoming": sorted(self.incoming_roads),
            "outgoing": sorted(self.outgoing_roads)
        }


class Road:
    """
    Represents a road (edge) connecting two junctions.
    Includes static properties such as speed, length, and lane count.
    """
    def __init__(self, road_id, from_junction, to_junction, speed=13.89, length=100.0, num_lanes=1, zone=None):
        self.id = road_id
        self.from_junction = from_junction
        self.to_junction = to_junction
        self.speed = speed
        self.length = length
        self.num_lanes = num_lanes
        self.zone = zone  # Zone label (e.g. 'A', 'B', 'C', or 'H')
        self.vehicles_on_road = {}
        self.density = 0.0  # Computed as vehicles / (length * num_lanes)
        self.avg_speed = 0.0


    def set_density(self):
        if self.length > 0 and self.num_lanes > 0:
            self.density = len(self.vehicles_on_road) / (self.length * self.num_lanes)
        else:
            self.density = 0.0

    def add_vehicle_and_update(self, vehicle):
        """
        Adds a vehicle ID to the road and updates density.
        """
        self.vehicles_on_road[vehicle.id] = vehicle.speed
        self.set_density()
        self.update_avg_speed()

    def remove_vehicle_and_update(self, vehicle):
        """
        Removes a vehicle ID from the road and updates density.
        """
        del self.vehicles_on_road[vehicle.id]
        self.set_density()
        self.update_avg_speed()


    def get_density(self):
        """
        Returns the current density of the road.
        """
        return self.density
    def update_avg_speed(self):
        """
        Updates the average speed of vehicles on this road.
        """
        if not self.vehicles_on_road:
            self.avg_speed = 0.0
            return

        total_speed = sum(self.vehicles_on_road.values())
        self.avg_speed = total_speed / len(self.vehicles_on_road)
        
    

    def to_dict(self):
        return {
            "id": self.id,
            "from": self.from_junction,
            "to": self.to_junction,
            "speed": self.speed,
            "length": self.length,
            "num_lanes": self.num_lanes,
            "zone": self.zone,
            "density": self.density,
            "avg_speed": self.avg_speed,
            "vehicles_on_road": sorted(self.vehicles_on_road.keys())    
        }


class Vehicle:
    """
    Represents a dynamic vehicle in the simulation.
    Tracks position, movement, physical characteristics, and zone associations.
    """
    def __init__(
        self,
        vehicle_id,
        vehicle_type,
        current_edge,
        current_position=0.0,
        speed=0.0,
        acceleration=0.0,
        route=None,
        route_left=None,
        length=4.5,
        width=1.8,
        height=1.5,
        current_x=None,
        current_y=None,
        current_zone=None,
        color='green',
        status="parked",
        is_stagnant=False
    ):
        # static properties
        self.id = vehicle_id
        self.vehicle_type = vehicle_type
        self.width = width
        self.length = length
        self.height = height
        self.color = color
        
        # dynamic properties
        self.speed = speed
        self.acceleration = acceleration
        self.current_edge = current_edge
        self.current_position = current_position
        self.current_x = current_x
        self.current_y = current_y
        self.current_zone = current_zone
        

        self.node_type = 1  # 0 for junction, 1 for vehicle
        self.is_stagnant = is_stagnant  # True if vehicle is not tracked by the model

        # routing and scheduling properties
        self.status = status  # e.g., "moving", "parked"
        self.scheduled = [False, False, False, False]  # True if vehicle is already scheduled for dispatch for the current week
       
        self.route = route if route else []
        self.route_left = route_left if route_left else []
        self.route_length = 0.0
        self.route_length_left = 0.0
       
        self.origin_name = None
        self.origin_zone = None
        self.origin_x = None
        self.origin_y = None
        self.origin_edge = None
        self.origin_position = None
        self.origin_step = None

        self.destinations = {
            "home": {"edge": self.current_edge, "position": self.current_position},
            "work": None,
            "friend1": None,
            "friend2": None,
            "friend3": None,
            "park1": None,
            "park2": None,
            "park3": None,
            "park4": None,
            "stadium1": None,
            "stadium2": None,
            "restaurantA": None,
            "restaurantB": None,
            "restaurantC": None
        }
        self.destination_name = None
        self.destination_zone = None
        self.destination_x = None
        self.destination_y = None
        self.destination_edge = None
        self.destination_position = None
        self.destination_step = None

    def to_dict(self):
        return {
            # Static properties
            "id": self.id,
            "node_type": self.node_type,
            "vehicle_type": self.vehicle_type,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            # Dynamic properties
            "speed": self.speed,
            "acceleration": self.acceleration,
            "current_x": self.current_x,
            "current_y": self.current_y,
            "current_zone": self.current_zone,
            "current_edge": self.current_edge,
            "current_position": self.current_position,

            # Routing and scheduling properties
            
            "origin_name": self.origin_name,
            "origin_zone": self.origin_zone,
            "origin_edge": self.origin_edge,
            "origin_position": self.origin_position,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "origin_start_sec": self.origin_step,

            "route": self.route,
            "route_length": self.route_length,
            "route_left": self.route_left,
            "route_length_left": self.route_length_left,
            
            "destination_name": self.destination_name,
            "destination_edge": self.destination_edge,
            "destination_position": self.destination_position,
            "destination_x": self.destination_x,
            "destination_y": self.destination_y
        }


class Zone:
    """
    Represents a traffic zone (e.g., 'A', 'B', 'C', 'H').
    Tracks all edges and junctions belonging to the zone,
    as well as vehicles that originated or are currently located in the zone.
    """
    def __init__(self, zone_id, description=None):
        self.id = zone_id
        self.description = description  # Optional textual description of the zone
        self.edges = set()
        self.junctions = set()
        self.original_vehicles = set()  # Vehicles that originated here
        self.current_vehicles = set()   # Vehicles currently here

    def add_edge(self, edge_id):
        self.edges.add(edge_id)

    def add_junction(self, junction_id):
        self.junctions.add(junction_id)

    def add_original_vehicle(self, vehicle_id):
        self.original_vehicles.add(vehicle_id)

    def add_current_vehicle(self, vehicle_id):
        self.current_vehicles.add(vehicle_id)

    def remove_current_vehicle(self, vehicle_id):
        self.current_vehicles.discard(vehicle_id)

    def get_random_edge(self):
        import random
        return random.choice(list(self.edges)) if self.edges else None

    def get_random_junction(self):
        import random
        return random.choice(list(self.junctions)) if self.junctions else None

    def get_random_vehicle(self):
        import random
        return random.choice(list(self.original_vehicles)) if self.original_vehicles else None

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "edges": sorted(self.edges),
            "junctions": sorted(self.junctions),
            "original_vehicles": sorted(self.original_vehicles),
            "current_vehicles": sorted(self.current_vehicles)
        }


# Keep these at the bottom so that entity classes are defined first
class DataBase:
    """
    Centralized store for all simulation entities:
    roads, junctions, vehicles, and zones.
    SimManager uses this to read/write state.
    """
    def __init__(self):
        self.roads = {}       # edge_id -> Road
        self.junctions = {}   # junction_id -> Junction
        self.vehicles = {}    # vehicle_id -> Vehicle
        self.zones = {}       # zone_id -> Zone

    def add_road(self, road):
        self.roads[road.id] = road

    def add_junction(self, junction):
        self.junctions[junction.id] = junction

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.id] = vehicle

    def update_junction(self, junction_id, incoming_roads=None, outgoing_roads=None):
        junction = self.get_junction(junction_id)
        if not junction:
            return
        if incoming_roads is not None:
            junction.incoming_roads = set(incoming_roads)
        if outgoing_roads is not None:
            junction.outgoing_roads = set(outgoing_roads)

    def add_zone(self, zone):
        self.zones[zone.id] = zone

    def get_road(self, road_id):
        return self.roads.get(road_id)

    def get_junction(self, junction_id):
        return self.junctions.get(junction_id)

    def get_vehicle(self, vehicle_id):
        return self.vehicles.get(vehicle_id)

    def get_zone(self, zone_id):
        return self.zones.get(zone_id)
    
    def print_zone_statistics(self):
        print("\n--- Simulation Zone Statistics ---")
        for zid, zone in self.zones.items():
            print(f"\nZone {zid}:")

            # Roads by lane count
            lane_counts = {}
            for eid in zone.edges:
                road = self.get_road(eid)
                lane_counts[road.num_lanes] = lane_counts.get(road.num_lanes, 0) + 1
            total_roads = sum(lane_counts.values())
            print(f"  Roads: {total_roads}")
            for lanes, count in sorted(lane_counts.items()):
                print(f"    {count} with {lanes} lane(s)")

            # Junctions and traffic lights
            total_junctions = len(zone.junctions)
            num_tls = sum(1 for jid in zone.junctions if self.get_junction(jid).type == "traffic_light")
            print(f"  Junctions: {total_junctions} ({num_tls} traffic lights)")

            # Vehicles
            vehicle_type_counts = {}
            stagnant_count = 0
            for vid in zone.current_vehicles:
                vehicle = self.get_vehicle(vid)
                vehicle_type_counts[vehicle.vehicle_type] = vehicle_type_counts.get(vehicle.vehicle_type, 0) + 1
                if vehicle.is_stagnant:
                    stagnant_count += 1

            total_vehicles = len(zone.current_vehicles)
            print(f"  Vehicles: {total_vehicles} ({stagnant_count} stagnant)")
            for vtype, count in vehicle_type_counts.items():
                print(f"    {count} {vtype}")

        print("\n total vehicles in simulation:", len(self.vehicles))
        print("\n-----------------------------------\n")


class SUMOConnectionManager:
    """
    Singleton manager for shared TraCI connection and SUMO network object.
    These objects are expensive to create and should be reused throughout
    the entire application lifespan.
    """
    _instance = None
    _traci = None
    _sumo_net = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SUMOConnectionManager, cls).__new__(cls)
        return cls._instance
    
    def set_traci(self, traci_connection):
        """
        Set the TraCI connection object.
        
        Args:
            traci_connection: The TraCI connection object (from traci.start() or traci module)
        """
        self._traci = traci_connection
    
    def get_traci(self):
        """
        Get the TraCI connection object.
        
        Returns:
            The TraCI connection object, or None if not set
        """
        return self._traci
    
    def set_sumo_net(self, sumo_net):
        """
        Set the SUMO network object.
        
        Args:
            sumo_net: The SUMO network object (from sumolib.net.readNet())
        """
        self._sumo_net = sumo_net
    
    def get_sumo_net(self):
        """
        Get the SUMO network object.
        
        Returns:
            The SUMO network object, or None if not set
        """
        return self._sumo_net
    
    def is_traci_available(self):
        """
        Check if TraCI connection is available.
        
        Returns:
            True if TraCI is set, False otherwise
        """
        return self._traci is not None
    
    def is_sumo_net_available(self):
        """
        Check if SUMO network object is available.
        
        Returns:
            True if SUMO net is set, False otherwise
        """
        return self._sumo_net is not None
    
    def reset(self):
        """
        Reset both TraCI and SUMO net objects (e.g., when closing application).
        """
        self._traci = None
        self._sumo_net = None
