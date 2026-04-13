#!/usr/bin/env python3
"""
TrafficLab System Debug Script
Run this to debug various aspects of the system
"""

import os
import sys
import json
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

def debug_environment():
    """Debug environment and dependencies"""
    print("🐛 DEBUG: Environment Check")
    print("="*50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Current working directory
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ ERROR: main.py not found. Are you in the backend directory?")
        return False
    
    # Check required files
    required_files = [
        "main.py",
        "models/database.py", 
        "services/sumo_service.py",
        "sumo/urban_three_zones.sumocfg",
        "sumo/sim.config.json"
    ]
    
    print("\n📁 Required files:")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
    
    return True

def debug_database():
    """Debug database connection"""
    print("\n🐛 DEBUG: Database Connection")
    print("="*50)
    
    try:
        from models.database import get_db, check_entities_exist
        from sqlalchemy import text
        
        # Get database session
        db = next(get_db())
        
        # Test connection
        result = db.execute(text("SELECT 1")).fetchone()
        print("✅ Database connection: OK")
        
        # Check entity status
        entity_status = check_entities_exist(db)
        print(f"📊 Entity status: {entity_status}")
        
        # Check tables
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        print(f"📋 Available tables: {tables}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def debug_sumo_config():
    """Debug SUMO configuration"""
    print("\n🐛 DEBUG: SUMO Configuration")
    print("="*50)
    
    # Check SUMO config file
    sumo_config_path = "sumo/urban_three_zones.sumocfg"
    if os.path.exists(sumo_config_path):
        print(f"✅ SUMO config file exists: {sumo_config_path}")
        
        # Parse config file
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(sumo_config_path)
            root = tree.getroot()
            
            # Find network file
            net_file_elem = root.find('.//net-file')
            if net_file_elem is not None:
                net_file = net_file_elem.get('value')
                net_path = os.path.join(os.path.dirname(sumo_config_path), net_file)
                print(f"📄 Network file: {net_path}")
                print(f"  Exists: {'✅' if os.path.exists(net_path) else '❌'}")
            else:
                print("❌ No network file found in SUMO config")
                
        except Exception as e:
            print(f"❌ Error parsing SUMO config: {e}")
    else:
        print(f"❌ SUMO config file not found: {sumo_config_path}")
        return False
    
    # Check simulation config
    sim_config_path = "sumo/sim.config.json"
    if os.path.exists(sim_config_path):
        print(f"✅ Simulation config exists: {sim_config_path}")
        
        try:
            with open(sim_config_path, 'r') as f:
                config = json.load(f)
            
            # Check vehicle generation config
            if "vehicle_generation" in config:
                vg = config["vehicle_generation"]
                print(f"🚗 Total vehicles: {vg.get('total_num_vehicles', 'N/A')}")
                print(f"📊 Dev fraction: {vg.get('dev_fraction', 'N/A')}")
                
                # Check zone allocation
                if "zone_allocation" in vg:
                    zones = vg["zone_allocation"]
                    print(f"🏘️  Zones configured: {list(zones.keys())}")
                    for zone_id, zone_config in zones.items():
                        if zone_id.lower() not in ["h", "stagnant"]:
                            percentage = zone_config.get("percentage", 0)
                            print(f"  Zone {zone_id}: {percentage}%")
            else:
                print("❌ No vehicle_generation config found")
                
        except Exception as e:
            print(f"❌ Error parsing simulation config: {e}")
    else:
        print(f"❌ Simulation config not found: {sim_config_path}")
        return False
    
    return True

def debug_system_initialization():
    """Debug system initialization"""
    print("\n🐛 DEBUG: System Initialization")
    print("="*50)
    
    try:
        from services.sumo_service import SUMOSimulation
        
        # Initialize simulation
        sumo_config_path = "sumo/urban_three_zones.sumocfg"
        sim_config_path = "sumo/sim.config.json"
        
        print("🚀 Initializing SUMO simulation...")
        sim = SUMOSimulation(sumo_config_path, sim_config_path)
        
        # Debug system state
        sim.debug_system_state()
        sim.debug_database_connection()
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🐛 TrafficLab System Debug Tool")
    print("="*50)
    
    # Run all debug checks
    checks = [
        ("Environment", debug_environment),
        ("Database", debug_database),
        ("SUMO Config", debug_sumo_config),
        ("System Init", debug_system_initialization)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 DEBUG SUMMARY")
    print("="*50)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All checks passed! System should be working correctly.")
    else:
        print("\n⚠️  Some checks failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
