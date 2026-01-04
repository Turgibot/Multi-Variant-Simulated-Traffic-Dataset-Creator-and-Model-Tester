# Complete Guide: Using OSMWebWizard to Generate a Complete SUMO Network

## Overview

This guide explains how to use OSMWebWizard to generate a complete SUMO network file that includes all necessary metadata, especially the `origBoundary` attribute which is required for GPS-to-SUMO coordinate conversion.

## Why `origBoundary` is Important

The `origBoundary` attribute in the SUMO network file contains the GPS coordinates (longitude, latitude) of the original OpenStreetMap area that was converted. This is essential for:

- Converting GPS coordinates to SUMO coordinates
- Displaying OSM map tiles aligned with the network
- Validating GPS trajectories against the network
- Accurate route calculation from GPS points

**Without `origBoundary`, the network is considered incomplete** because GPS-to-SUMO coordinate conversion cannot work properly.

## Prerequisites

1. **SUMO installed** (check with: `sumo --version`)
   - Location: Usually `/usr/share/sumo` or `$SUMO_HOME`
   - Verify: `echo $SUMO_HOME` or check `/usr/share/sumo/tools/osmWebWizard.py`

2. **Python 3.x** (Python 2.7+ also works but Python 3 is recommended)

3. **Web browser** (Chrome, Firefox, etc.)

4. **Internet connection** (required to download OSM data)

## Step-by-Step Instructions

### Step 1: Determine the Current Network Bounds

If you have an existing network file but it's missing `origBoundary`, you need to find the GPS bounds of the area it covers.

#### Method 1: Check if Network Has Partial Information

Open your network file and look for the `<location>` element:

```xml
<location netOffset="..." convBoundary="..." origBoundary="..." projParameter="..."/>
```

- If `origBoundary` is missing or empty (`origBoundary="0,0,0,0"`), you need to regenerate
- If `convBoundary` exists, you can estimate the GPS bounds (see Method 2)

#### Method 2: Extract Bounds from Network Edges

If you know the approximate geographic area (e.g., "Porto, Portugal"), you can:

1. **Use a mapping tool** to identify the bounding box:
   - Go to https://www.openstreetmap.org
   - Navigate to your area
   - Use the "Export" feature to get bounding box coordinates
   - Format: `min_lat, min_lon, max_lat, max_lon` (note: OSM uses lat,lon order)

2. **Or use Python to extract from existing network**:

```python
from src.utils.network_parser import NetworkParser

parser = NetworkParser("path/to/porto.net.xml")
bounds = parser.get_bounds()  # SUMO coordinates
print(f"SUMO bounds: {bounds}")

# If you know the approximate GPS area, you can estimate
# For Porto, Portugal, approximate bounds are:
# lon_min: -8.7, lat_min: 41.1, lon_max: -8.5, lat_max: 41.2
```

#### Method 3: Use Existing Network Statistics

If you have the original OSM file or know the city/region:
- Search for the city on OpenStreetMap
- Note the approximate bounding box
- Use these coordinates in OSMWebWizard

### Step 2: Launch OSMWebWizard

#### Option A: Direct Command

```bash
python /usr/share/sumo/tools/osmWebWizard.py
```

#### Option B: Using SUMO_HOME

```bash
export SUMO_HOME=/usr/share/sumo
python $SUMO_HOME/tools/osmWebWizard.py
```

#### Option C: If SUMO is in PATH

```bash
python osmWebWizard.py
```

**Expected behavior:**
- A web browser should open automatically
- URL: `http://localhost:8080` (or similar port)
- You'll see an interactive map (default: centered on Berlin)

### Step 3: Navigate to Your Target Area

1. **Use the search box** (top right) to find your location:
   - Example: "Porto, Portugal"
   - Example: "41.15,-8.61" (latitude, longitude)

2. **Zoom to appropriate level**:
   - Use mouse wheel or zoom controls
   - Zoom level should show the entire area you want
   - **Important**: The selected area should match your existing network size

3. **Pan the map** if needed:
   - Click and drag to move

### Step 4: Select the Exact Area

**Critical**: To match your existing network size, you need to select the same bounding box.

#### Option A: If You Know the GPS Bounds

1. **Check "Select Area"** checkbox (right panel)
2. A selection rectangle appears on the map
3. **Manually adjust corners** to match your bounds:
   - Click and drag corner handles
   - Use coordinates if available (see Option B)

#### Option B: Using Known Coordinates

If you have the exact GPS bounds (e.g., from the original network or OSM export):

1. **Note the coordinates** in format: `lon_min, lat_min, lon_max, lat_max`
   - Example for Porto: `-8.7, 41.1, -8.5, 41.2`

2. **Calculate center point**:
   - Center lon = (lon_min + lon_max) / 2
   - Center lat = (lat_min + lat_max) / 2
   - Example: `-8.6, 41.15`

3. **Search for center point** in OSMWebWizard search box:
   - Format: `41.15,-8.6` (lat,lon - note the order!)

4. **Zoom to appropriate level** (zoom out enough to see the full area)

5. **Enable "Select Area"** and adjust rectangle to match bounds

#### Option C: Matching Existing Network Size Visually

If you don't have exact coordinates:

1. **Load your existing network** in SUMO-GUI or your application
2. **Note the visual extent** (how much area it covers)
3. **In OSMWebWizard**, navigate to the same geographic area
4. **Select an area** that visually matches the size
5. **Compare edge counts** after generation (see Step 7)

### Step 5: Configure Network Generation Options

**Right panel options** (important settings):

1. **"Add Polygon"** (optional):
   - Includes buildings and other polygons
   - Increases file size but adds detail
   - Recommended: **Enabled** for complete networks

2. **"Left-hand Traffic"**:
   - Enable only for UK, Japan, Australia, etc.
   - Most countries: **Disabled**

3. **"Car-only Network"**:
   - If enabled: Only roads for cars (excludes pedestrian areas)
   - Recommended: **Disabled** (include all road types for completeness)

4. **"Import Public Transport"** (optional):
   - Includes bus/tram routes
   - Recommended: **Enabled** for complete networks

### Step 6: Generate the Network

1. **Click "Generate Scenario"** button (bottom of right panel)

2. **Wait for processing**:
   - Small areas (< 1 km²): 1-5 minutes
   - Medium areas (1-10 km²): 5-15 minutes
   - Large areas (> 10 km²): 15-60 minutes
   - Progress shown in browser and terminal

3. **What happens**:
   - Downloads OSM data for selected area
   - Converts OSM to SUMO network (`.net.xml`)
   - Generates routes (`.rou.xml`) - optional, can be skipped
   - Creates configuration (`.sumocfg`)
   - Launches SUMO-GUI (optional)

### Step 7: Verify the Generated Network

#### Check 1: File Location

Generated files are saved in:
- **Format**: `yyyy-mm-dd-hh-mm-ss/` (timestamp directory)
- **Location**: Usually `~/SUMO/` or current directory
- **Files**:
  - `*.net.xml` - **Network file** (this is what you need)
  - `*.rou.xml` - Route file (optional)
  - `*.sumocfg` - Configuration file
  - `*.poly.xml` - Polygons (if enabled)

#### Check 2: Verify `origBoundary` Exists

Open the `.net.xml` file and check the `<location>` element:

```xml
<net version="..." ...>
    <location netOffset="..." 
              convBoundary="..." 
              origBoundary="-8.7,41.1,-8.5,41.2" 
              projParameter="..."/>
    ...
</net>
```

**Required**: `origBoundary` should contain 4 numbers: `lon_min,lat_min,lon_max,lat_max`

**If `origBoundary` is missing or `0,0,0,0`**:
- The network generation failed or used an old method
- Try again with a smaller area or different options

#### Check 3: Compare Network Size

Compare with your existing network:

```python
from src.utils.network_parser import NetworkParser

# Old network
old_parser = NetworkParser("old/network.net.xml")
old_edges = len(old_parser.get_edges())
old_nodes = len(old_parser.get_nodes())

# New network
new_parser = NetworkParser("new/network.net.xml")
new_edges = len(new_parser.get_edges())
new_nodes = len(new_parser.get_nodes())

print(f"Old: {old_edges} edges, {old_nodes} nodes")
print(f"New: {new_edges} edges, {new_nodes} nodes")
print(f"Difference: {new_edges - old_edges} edges")
```

**Expected**: Similar edge/node counts (within 10-20% is acceptable due to OSM data updates)

#### Check 4: Test GPS-to-SUMO Conversion

```python
from src.utils.network_parser import NetworkParser

parser = NetworkParser("new/network.net.xml")

# Check if origBoundary exists
if parser.orig_boundary:
    print("✅ origBoundary found!")
    print(f"GPS bounds: {parser.orig_boundary}")
    
    # Test conversion
    # Use a known GPS point in the area
    lon, lat = -8.61, 41.15  # Example: Porto center
    sumo_coords = parser.gps_to_sumo_coords(lon, lat)
    if sumo_coords:
        print(f"✅ GPS conversion works: ({lon}, {lat}) -> {sumo_coords}")
    else:
        print("❌ GPS conversion failed")
else:
    print("❌ origBoundary missing - network is incomplete!")
```

### Step 8: Replace Your Network File

Once verified:

1. **Backup your old network**:
   ```bash
   cp porto.net.xml porto.net.xml.backup
   ```

2. **Copy the new network**:
   ```bash
   cp ~/SUMO/2025-01-15-10-30-45/*.net.xml porto.net.xml
   ```

3. **Update your project** to use the new network file

4. **Test your application** to ensure everything works

## Troubleshooting

### Problem: `origBoundary` is Still Missing After Generation

**Possible causes:**
1. **Old SUMO version**: Update SUMO to latest version
2. **Generation failed**: Check terminal/browser for errors
3. **Area too large**: Try a smaller area first

**Solution:**
- Check SUMO version: `sumo --version` (should be 1.0.0 or later)
- Try generating a very small area (few city blocks) to test
- Check browser console (F12) for JavaScript errors

### Problem: Generated Network is Different Size

**Causes:**
- Selected area doesn't match original bounds
- OSM data has changed since original generation
- Different options selected (car-only vs. all roads)

**Solution:**
- Use exact GPS bounds if available
- Compare edge counts and adjust selection
- Note: Some variation is normal due to OSM updates

### Problem: OSMWebWizard Won't Start

**Check:**
1. SUMO installation: `which sumo` or `sumo --version`
2. Python version: `python --version` (should be 2.7+ or 3.x)
3. Port availability: Try `netstat -an | grep 8080` (port should be free)

**Solution:**
- Install/update SUMO
- Try different port: `python osmWebWizard.py --port 8081`
- Check firewall settings

### Problem: Browser Shows Blank/Pink Screen

**Causes:**
- OpenStreetMap tile server issues
- Internet connection problems
- Browser compatibility

**Solution:**
- Check internet connection
- Try different browser (Chrome, Firefox)
- Refresh the page
- Check browser console (F12) for errors

### Problem: Generation Takes Too Long

**Causes:**
- Area too large
- Slow internet connection
- Complex road network

**Solution:**
- Start with smaller area (1-2 km²)
- Check internet speed
- Be patient for large areas (can take 30-60 minutes)

## Best Practices

1. **Always verify `origBoundary`** after generation
2. **Keep backups** of working networks
3. **Document the GPS bounds** used for future reference
4. **Test GPS conversion** before using in production
5. **Compare network statistics** (edge/node counts) to ensure similarity

## Quick Reference: GPS Bounds Format

- **OSMWebWizard search**: `lat,lon` (e.g., `41.15,-8.61`)
- **origBoundary in .net.xml**: `lon_min,lat_min,lon_max,lat_max` (e.g., `-8.7,41.1,-8.5,41.2`)
- **Bounding box (OSM export)**: `min_lat, min_lon, max_lat, max_lon`

**Note**: Pay attention to coordinate order - it varies by tool!

## Example: Regenerating Porto Network

Assuming you want to regenerate the Porto network with complete metadata:

1. **Launch OSMWebWizard**:
   ```bash
   python /usr/share/sumo/tools/osmWebWizard.py
   ```

2. **Search for Porto**:
   - Search box: `Porto, Portugal` or `41.15,-8.61`

3. **Select area**:
   - Approximate bounds: `-8.7, 41.1, -8.5, 41.2` (lon_min, lat_min, lon_max, lat_max)
   - Center: `41.15, -8.6` (lat, lon for search)
   - Adjust selection rectangle to cover Porto city area

4. **Configure options**:
   - Add Polygon: **Enabled**
   - Car-only: **Disabled**
   - Left-hand traffic: **Disabled**

5. **Generate** and wait

6. **Verify**:
   ```bash
   grep origBoundary ~/SUMO/*/osm.net.xml
   # Should show: origBoundary="-8.7,41.1,-8.5,41.2" (or similar)
   ```

7. **Replace old network**:
   ```bash
   cp ~/SUMO/YYYY-MM-DD-HH-MM-SS/osm.net.xml Porto/config/porto.net.xml
   ```

## Summary

To generate a complete SUMO network with `origBoundary`:

1. ✅ Use OSMWebWizard (not netconvert directly)
2. ✅ Select the exact geographic area you need
3. ✅ Enable appropriate options (polygons, all road types)
4. ✅ Verify `origBoundary` exists in generated `.net.xml`
5. ✅ Test GPS-to-SUMO coordinate conversion
6. ✅ Compare network size with original

**Key Point**: Only networks generated with OSMWebWizard (or netconvert with proper OSM input and location parameters) will have complete `origBoundary` metadata. Networks created manually or with older tools may be missing this critical information.





