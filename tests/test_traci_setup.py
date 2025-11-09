"""
Helper script to set up TraCI Python bindings.
This script helps find and add SUMO tools to Python path.
"""

import sys
import os
import subprocess


def find_sumo_tools():
    """Find SUMO tools directory."""
    try:
        # Try to get SUMO_HOME from environment
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home:
            tools_path = os.path.join(sumo_home, 'tools')
            if os.path.exists(tools_path):
                return tools_path
        
        # Try to find SUMO binary and work backwards
        result = subprocess.run(['which', 'sumo'], capture_output=True, text=True)
        if result.returncode == 0:
            sumo_bin = result.stdout.strip()
            # SUMO binary is typically in bin/, so go up to find tools
            sumo_dir = os.path.dirname(os.path.dirname(sumo_bin))
            tools_path = os.path.join(sumo_dir, 'tools')
            if os.path.exists(tools_path):
                return tools_path
        
        # Common installation paths
        common_paths = [
            '/usr/share/sumo/tools',
            '/opt/sumo/tools',
            '/usr/local/share/sumo/tools',
            os.path.expanduser('~/sumo/tools'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    except Exception as e:
        print(f"Error finding SUMO tools: {e}")
        return None


def setup_traci():
    """Set up TraCI by adding SUMO tools to Python path."""
    tools_path = find_sumo_tools()
    
    if tools_path is None:
        print("✗ Could not find SUMO tools directory")
        print("\nTo fix this:")
        print("1. Set SUMO_HOME environment variable:")
        print("   export SUMO_HOME=/path/to/sumo")
        print("2. Or add SUMO tools to PYTHONPATH:")
        print("   export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools")
        return False
    
    print(f"✓ Found SUMO tools at: {tools_path}")
    
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
        print(f"✓ Added to Python path")
    
    try:
        import traci
        print("✓ TraCI imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ TraCI import still failed: {e}")
        print(f"\nTry adding this to your shell profile:")
        print(f"export PYTHONPATH=$PYTHONPATH:{tools_path}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("TraCI Setup Helper")
    print("=" * 60)
    setup_traci()

