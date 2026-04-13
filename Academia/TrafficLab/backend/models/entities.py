
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
        self.shape_points = []  # Detailed shape points for express edges


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
        is_stagnant=False,
        scheduled=False
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
        self.scheduled = scheduled
        

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

