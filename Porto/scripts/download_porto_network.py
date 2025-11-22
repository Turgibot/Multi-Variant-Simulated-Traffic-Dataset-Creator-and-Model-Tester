#!/usr/bin/env python3
"""
Script to download and convert Porto OSM data to SUMO network format.
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

# Get project root (assuming script is in Porto/scripts/)
SCRIPT_DIR = Path(__file__).parent
PORTO_DIR = SCRIPT_DIR.parent
CONFIG_DIR = PORTO_DIR / 'config'

# Porto bounding box coordinates (approximate city center)
PORTO_BBOX = {
    'north': 41.2000,
    'south': 41.1400,
    'east': -8.5800,
    'west': -8.6500
}


def check_sumo():
    """Check if SUMO is available."""
    try:
        result = subprocess.run(['which', 'netconvert'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except Exception:
        pass
    
    # Check SUMO_HOME
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home and Path(sumo_home).exists():
        netconvert = Path(sumo_home) / 'bin' / 'netconvert'
        if netconvert.exists():
            return str(netconvert)
    
    return False


def download_porto_osm(output_file):
    """Download Porto OSM data using Overpass API."""
    print("Downloading Porto OSM data...")
    
    # Overpass query for Porto bounding box
    query = f"""
    [out:xml][timeout:60];
    (
      way["highway"]({PORTO_BBOX['south']},{PORTO_BBOX['west']},{PORTO_BBOX['north']},{PORTO_BBOX['east']});
      relation["highway"]({PORTO_BBOX['south']},{PORTO_BBOX['west']},{PORTO_BBOX['north']},{PORTO_BBOX['east']});
    );
    (._;>;);
    out body;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    
    try:
        req = urllib.request.Request(url, data=query.encode('utf-8'))
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(output_file, 'wb') as f:
                f.write(response.read())
        print(f"✓ Downloaded OSM data to {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to download OSM data: {e}")
        print("\nAlternative: Download manually from:")
        print("  - https://extract.bbbike.org/ (select Porto, Portugal)")
        print("  - https://www.openstreetmap.org/export (select area manually)")
        return False


def convert_osm_to_sumo(osm_file, net_file, sumo_home=None):
    """Convert OSM file to SUMO network."""
    print(f"Converting {osm_file} to SUMO network...")
    
    # Find netconvert
    netconvert_path = None
    if sumo_home:
        netconvert_path = Path(sumo_home) / 'bin' / 'netconvert'
    else:
        result = subprocess.run(['which', 'netconvert'], capture_output=True, text=True)
        if result.returncode == 0:
            netconvert_path = Path(result.stdout.strip())
    
    if not netconvert_path or not netconvert_path.exists():
        print("✗ netconvert not found. Please set SUMO_HOME or ensure netconvert is in PATH.")
        return False
    
    # Build netconvert command
    cmd = [
        str(netconvert_path),
        '--osm-files', str(osm_file),
        '--output-file', str(net_file),
        '--geometry.remove',
        '--roundabouts.guess',
        '--ramps.guess',
        '--junctions.join',
        '--tls.guess-signals',
        '--tls.discard-simple',
        '--tls.join',
        '--no-turnarounds',
        '--no-internal-links',
        '--remove-edges.by-vclass', 'pedestrian,bicycle',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✓ Converted to SUMO network: {net_file}")
            return True
        else:
            print(f"✗ Conversion failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Conversion timed out (took more than 10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Conversion error: {e}")
        return False


def create_sumocfg(net_file, route_file, output_file):
    """Create SUMO configuration file."""
    print(f"Creating SUMO configuration file...")
    
    # Use relative paths
    net_rel = Path(net_file).name
    route_rel = Path(route_file).name
    
    config_xml = f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_rel}"/>
        <route-files value="{route_rel}"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    
    <processing>
        <lateral-resolution value="0.8"/>
    </processing>
    
    <report>
        <verbose value="true"/>
        <no-warnings value="false"/>
    </report>
</configuration>
"""
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(config_xml)
        print(f"✓ Created configuration file: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to create config file: {e}")
        return False


def create_empty_route_file(route_file):
    """Create an empty route file."""
    print(f"Creating empty route file...")
    
    route_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
</routes>
"""
    
    try:
        with open(route_file, 'w', encoding='utf-8') as f:
            f.write(route_xml)
        print(f"✓ Created route file: {route_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to create route file: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("Porto Network Download and Conversion Script")
    print("=" * 60)
    print()
    
    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # File paths
    osm_file = CONFIG_DIR / 'porto.osm'
    net_file = CONFIG_DIR / 'porto.net.xml'
    route_file = CONFIG_DIR / 'porto.rou.xml'
    sumocfg_file = CONFIG_DIR / 'porto.sumocfg'
    
    # Check SUMO
    sumo_home = os.environ.get('SUMO_HOME')
    if not check_sumo():
        print("⚠ Warning: SUMO not found. Please:")
        print("  1. Install SUMO")
        print("  2. Set SUMO_HOME environment variable")
        print("  3. Or ensure netconvert is in your PATH")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 1: Download OSM data
    if not osm_file.exists():
        if not download_porto_osm(osm_file):
            print("\nPlease download Porto OSM data manually and save it as:")
            print(f"  {osm_file}")
            return
    else:
        print(f"✓ OSM file already exists: {osm_file}")
    
    # Step 2: Convert to SUMO network
    if not net_file.exists():
        if not convert_osm_to_sumo(osm_file, net_file, sumo_home):
            print("\nPlease convert manually using:")
            print(f"  netconvert --osm-files {osm_file} --output-file {net_file}")
            return
    else:
        print(f"✓ Network file already exists: {net_file}")
    
    # Step 3: Create route file
    if not route_file.exists():
        create_empty_route_file(route_file)
    else:
        print(f"✓ Route file already exists: {route_file}")
    
    # Step 4: Create sumocfg
    if not sumocfg_file.exists():
        create_sumocfg(net_file, route_file, sumocfg_file)
    else:
        print(f"✓ Config file already exists: {sumocfg_file}")
        print("  (Skipping creation to preserve existing configuration)")
    
    print()
    print("=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)
    print()
    print("Files created:")
    print(f"  Network: {net_file}")
    print(f"  Routes:  {route_file}")
    print(f"  Config:  {sumocfg_file}")
    print()
    print("Next steps:")
    print("  1. Open the application")
    print("  2. Enable 'Use Porto Taxi Dataset' checkbox")
    print("  3. The network will be automatically loaded")


if __name__ == '__main__':
    main()

