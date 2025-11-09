# How to Use OSM Web Wizard

## Overview

OSM Web Wizard is a web-based tool that creates SUMO simulation scenarios from OpenStreetMap (OSM) data. It provides a user-friendly interface to select an area on a map and generate a complete SUMO simulation.

## Prerequisites

- SUMO installed (you have it at `/usr/share/sumo`)
- Python 2.7+ or Python 3.x
- Web browser (opens automatically)

## How to Run

### Method 1: Direct Python Command
```bash
python /usr/share/sumo/tools/osmWebWizard.py
```

### Method 2: Using SUMO_HOME
```bash
export SUMO_HOME=/usr/share/sumo
python $SUMO_HOME/tools/osmWebWizard.py
```

### Method 3: If SUMO tools are in PATH
```bash
python osmWebWizard.py
```

## What Happens

1. **Web Browser Opens**: The tool automatically opens your default web browser
2. **Map Interface**: Shows an interactive map (centered on Berlin by default)
3. **Web Server**: Runs a local web server (usually on port 8080)

## Using the Web Interface

### 1. Navigate the Map
- **Zoom**: Use mouse wheel or zoom controls
- **Pan**: Click and drag the map
- **Search**: Use the search box to find specific locations

### 2. Select Simulation Area
- Check **"Select Area"** checkbox on the right panel
- A selection box will appear on the map
- **Drag the corners** to adjust the area
- **Be careful**: Large areas may take a long time to process

### 3. Configure Network Options
- **Add Polygon**: Include buildings and other polygons
- **Left-hand Traffic**: Enable for countries with left-hand traffic (UK, Japan, etc.)
- **Car-only Network**: Only include roads for cars (exclude pedestrian areas)
- **Import Public Transport**: Include bus/tram routes

### 4. Set Traffic Demand
- Click the **car icon** to open demand generation panel
- Enable/disable transport modes:
  - Cars
  - Bicycles
  - Pedestrians
  - Public transport
- Adjust parameters:
  - **Through Traffic Factor**: Amount of traffic passing through
  - **Count**: Number of vehicles
  - **Time Range**: Simulation duration

### 5. Generate Scenario
- Click **"Generate Scenario"** button
- The tool will:
  1. Download OSM data for the selected area
  2. Convert OSM to SUMO network (.net.xml)
  3. Generate routes (.rou.xml)
  4. Create configuration file (.sumocfg)
  5. Launch SUMO-GUI to visualize

### 6. Output Location
Generated files are saved in:
- Format: `yyyy-mm-dd-hh-mm-ss` (timestamp directory)
- Location: `~/SUMO/` or `tools/` directory
- Files created:
  - `*.net.xml` - Network file
  - `*.rou.xml` - Route file
  - `*.sumocfg` - Configuration file
  - `*.poly.xml` - Polygons (if enabled)

## Tips

1. **Start Small**: Begin with a small area (few city blocks) to test
2. **Check Processing Time**: Large areas can take 10-30 minutes
3. **Internet Required**: Needs internet to download OSM data
4. **Edit After Generation**: You can modify generated files manually
5. **Reuse Networks**: Save generated networks for future use

## Troubleshooting

### Web Browser Doesn't Open
- Check if port 8080 is available
- Manually open: `http://localhost:8080`

### Pink/Blank Screen
- May be due to OpenStreetMap tile loading issues
- Check internet connection
- Try refreshing the page

### Generation Fails
- Area might be too large
- Try selecting a smaller area
- Check console for error messages

## Using Generated Files in Your Application

After generating a scenario with OSM Web Wizard:

1. **Find the generated directory** (timestamp format)
2. **Locate the `.sumocfg` file**
3. **Use it in your application**:
   - Go to Dataset Generation Settings
   - Click "Browse..." for SUMO Configuration File
   - Select the `.sumocfg` file
   - The network and route files will be automatically detected

## Example Workflow

1. Run: `python /usr/share/sumo/tools/osmWebWizard.py`
2. Browser opens with map
3. Navigate to your city/area of interest
4. Select a small area (e.g., downtown)
5. Configure options (add polygons, traffic demand)
6. Click "Generate Scenario"
7. Wait for generation (may take a few minutes)
8. SUMO-GUI opens with the simulation
9. Find generated files in `~/SUMO/yyyy-mm-dd-hh-mm-ss/`
10. Use the `.sumocfg` file in your application

