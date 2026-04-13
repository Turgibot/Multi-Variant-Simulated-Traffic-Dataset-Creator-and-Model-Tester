#!/usr/bin/env python3
"""
Test script for entity database persistence
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.database import create_tables, get_db
from models.entities import Junction, Road, Vehicle, Zone
from services.simulation_manager import simulation_manager

def test_entity_persistence():
    """Test entity creation and database persistence"""
    print("🧪 Testing Entity Database Persistence")
    print("=" * 50)
    
    # Create tables
    print("📋 Creating database tables...")
    create_tables()
    
    # Load simulation manager
    print("🔄 Loading simulation manager from database...")
    simulation_manager.load_from_database()
    
    # Create test entities
    print("\n🏗️ Creating test entities...")
    
    # Create junctions
    junction1 = Junction("J1", 100.0, 200.0, "priority", "A")
    junction2 = Junction("J2", 300.0, 200.0, "priority", "A")
    junction3 = Junction("J3", 200.0, 400.0, "traffic_light", "B")
    
    simulation_manager.add_junction(junction1)
    simulation_manager.add_junction(junction2)
    simulation_manager.add_junction(junction3)
    print(f"✅ Created {len(simulation_manager.junctions)} junctions")
    
    # Create roads
    road1 = Road("R1", "J1", "J2", 13.89, 200.0, 2, "A")
    road2 = Road("R2", "J2", "J3", 11.11, 250.0, 1, "B")
    road3 = Road("R3", "J3", "J1", 8.33, 300.0, 1, "B")
    
    simulation_manager.add_road(road1)
    simulation_manager.add_road(road2)
    simulation_manager.add_road(road3)
    print(f"✅ Created {len(simulation_manager.roads)} roads")
    
    # Create zones
    zone_a = Zone("A", "Residential Zone A")
    zone_b = Zone("B", "Commercial Zone B")
    
    zone_a.add_junction("J1")
    zone_a.add_junction("J2")
    zone_a.add_edge("R1")
    
    zone_b.add_junction("J3")
    zone_b.add_edge("R2")
    zone_b.add_edge("R3")
    
    simulation_manager.add_zone(zone_a)
    simulation_manager.add_zone(zone_b)
    print(f"✅ Created {len(simulation_manager.zones)} zones")
    
    # Create vehicles
    vehicle1 = Vehicle(
        vehicle_id="V1",
        vehicle_type="car",
        current_edge="R1",
        current_position=50.0,
        speed=13.89,
        current_x=150.0,
        current_y=200.0,
        current_zone="A",
        status="moving"
    )
    
    vehicle2 = Vehicle(
        vehicle_id="V2",
        vehicle_type="truck",
        current_edge="R2",
        current_position=100.0,
        speed=11.11,
        current_x=250.0,
        current_y=300.0,
        current_zone="B",
        status="moving"
    )
    
    simulation_manager.add_vehicle(vehicle1)
    simulation_manager.add_vehicle(vehicle2)
    print(f"✅ Created {len(simulation_manager.vehicles)} vehicles")
    
    # Test road density calculation
    print("\n📊 Testing road density calculation...")
    road1.add_vehicle_and_update(vehicle1)
    print(f"Road R1 density: {road1.get_density():.4f} vehicles/m²")
    print(f"Road R1 avg speed: {road1.avg_speed:.2f} m/s")
    
    # Test simulation state
    print("\n📈 Simulation State:")
    state = simulation_manager.get_simulation_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Test vehicle movement
    print("\n🚗 Testing vehicle movement...")
    simulation_manager.move_vehicle("V1", "R2", 25.0, 275.0, 250.0)
    print("Vehicle V1 moved to road R2")
    
    # Test zone tracking
    print("\n🗺️ Zone tracking:")
    for zone_id, zone in simulation_manager.zones.items():
        print(f"Zone {zone_id}: {len(zone.current_vehicles)} current vehicles")
    
    # Test database persistence
    print("\n💾 Testing database persistence...")
    simulation_manager.save_to_database()
    print("✅ All entities saved to database")
    
    # Test loading from database
    print("\n🔄 Testing database loading...")
    new_manager = simulation_manager.__class__()
    new_manager.load_from_database()
    
    print(f"Loaded {len(new_manager.junctions)} junctions")
    print(f"Loaded {len(new_manager.roads)} roads")
    print(f"Loaded {len(new_manager.vehicles)} vehicles")
    print(f"Loaded {len(new_manager.zones)} zones")
    
    # Verify data integrity
    assert len(new_manager.junctions) == 3, "Junction count mismatch"
    assert len(new_manager.roads) == 3, "Road count mismatch"
    assert len(new_manager.vehicles) == 2, "Vehicle count mismatch"
    assert len(new_manager.zones) == 2, "Zone count mismatch"
    
    print("\n✅ All tests passed! Database persistence is working correctly.")
    
    # Test simulation stepping
    print("\n🎮 Testing simulation stepping...")
    simulation_manager.start_simulation()
    simulation_manager.step_simulation(1.0)
    simulation_manager.step_simulation(1.0)
    simulation_manager.stop_simulation()
    print("✅ Simulation stepping completed")

if __name__ == "__main__":
    test_entity_persistence()
