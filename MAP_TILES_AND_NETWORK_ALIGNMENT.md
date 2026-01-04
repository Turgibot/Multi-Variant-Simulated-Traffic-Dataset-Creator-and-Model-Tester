# How Map Tiles and SUMO Networks Match

## Overview

This document explains how OpenStreetMap (OSM) tiles align with SUMO network coordinates, and how to download a new SUMO network using OSMWebWizard without breaking the alignment.

## The Coordinate System Chain

The alignment between map tiles and SUMO networks relies on a chain of coordinate conversions:

```
GPS Coordinates (WGS84) 
    ↓
origBoundary (in .net.xml)
    ↓
convBoundary (in .net.xml)
    ↓
SUMO Network Coordinates (x, y)
```

## Key Components

### 1. `origBoundary` - The GPS Anchor

**Location**: In the `.net.xml` file, inside the `<location>` element:
```xml
<location netOffset="..." 
          convBoundary="..." 
          origBoundary="-8.7,41.1,-8.5,41.2" 
          projParameter="..."/>
```

**Format**: `lon_min,lat_min,lon_max,lat_max` (longitude, latitude)

**Purpose**: This is the **GPS bounding box** of the original OpenStreetMap area that was converted to the SUMO network. It's the critical link between:
- The geographic area you selected in OSMWebWizard
- The SUMO network coordinates
- The OSM map tiles

**Why it matters**: Without `origBoundary`, the system cannot convert GPS coordinates to SUMO coordinates, and map tiles cannot be aligned with the network.

### 2. `convBoundary` - The SUMO Coordinate Bounds

**Format**: `x_min,y_min,x_max,y_max` (SUMO network coordinates)

**Purpose**: This defines the SUMO network coordinate bounds that correspond to the `origBoundary` GPS area. It's the projected coordinate system used by SUMO.

**Relationship**: The system uses **linear interpolation** to map between `origBoundary` (GPS) and `convBoundary` (SUMO):
- A GPS point at `(lon_min, lat_min)` maps to SUMO `(x_min, y_min)`
- A GPS point at `(lon_max, lat_max)` maps to SUMO `(x_max, y_max)`
- Points in between are interpolated proportionally

### 3. Map Tile Alignment Process

When displaying OSM tiles over the SUMO network:

1. **Tile Download**: Tiles are requested based on GPS coordinates (using `origBoundary`)
2. **GPS to SUMO Conversion**: Each tile's GPS corners are converted to SUMO coordinates using:
   ```python
   # Normalize GPS within origBoundary
   lon_norm = (lon - lon_min) / (lon_max - lon_min)
   lat_norm = (lat - lat_min) / (lat_max - lat_min)
   
   # Map to SUMO coordinates
   x = x_min + lon_norm * (x_max - x_min)
   y = y_min + lat_norm * (y_max - y_min)
   ```
3. **Y-Flip**: SUMO uses Y-up (north = higher Y), but screen coordinates use Y-down, so tiles are flipped vertically
4. **Positioning**: Tiles are positioned at their converted SUMO coordinates

## How to Download a New Network Correctly

### Critical Steps to Avoid Breaking Alignment

#### Step 1: Determine Your Required GPS Bounds

**Before** using OSMWebWizard, you need to know the GPS bounds of the area you want:

**Option A: From Existing Network**
```python
from src.utils.network_parser import NetworkParser

parser = NetworkParser("path/to/existing.net.xml")
if parser.orig_boundary:
    print(f"Current GPS bounds: {parser.orig_boundary}")
    # Format: {'lon_min': -8.7, 'lat_min': 41.1, 'lon_max': -8.5, 'lat_max': 41.2}
else:
    print("⚠️ Network missing origBoundary - you need to regenerate it!")
```

**Option B: From OpenStreetMap**
1. Go to https://www.openstreetmap.org
2. Navigate to your area
3. Use the "Export" feature to get bounding box coordinates
4. Format: `lon_min, lat_min, lon_max, lat_max`

**Option C: Estimate from City/Region**
- Use a mapping tool or search engine to find approximate bounds
- Example for Porto: `-8.7, 41.1, -8.5, 41.2` (lon_min, lat_min, lon_max, lat_max)

#### Step 2: Launch OSMWebWizard

```bash
# Method 1: Direct path
python /usr/share/sumo/tools/osmWebWizard.py

# Method 2: Using SUMO_HOME
export SUMO_HOME=/usr/share/sumo
python $SUMO_HOME/tools/osmWebWizard.py
```

**Expected**: Browser opens at `http://localhost:8080` with an interactive map.

#### Step 3: Navigate to Your Area

1. **Search for location**:
   - Use search box: `"Porto, Portugal"` or `"41.15,-8.61"` (lat,lon - note order!)
   - OSMWebWizard uses `lat,lon` format for search

2. **Calculate center point** (if you have bounds):
   ```python
   center_lon = (lon_min + lon_max) / 2
   center_lat = (lat_min + lat_max) / 2
   # Search with: f"{center_lat},{center_lon}"
   ```

3. **Zoom appropriately**:
   - Zoom out enough to see the entire area you need
   - The zoom level should match the size of your required bounds

#### Step 4: Select the Exact Area

**⚠️ CRITICAL**: The selected area in OSMWebWizard **must match** your required GPS bounds.

**Method A: Visual Selection**
1. Check "Select Area" checkbox
2. Drag the selection rectangle to cover your area
3. **Problem**: This is imprecise and may not match exactly

**Method B: Precise Selection (Recommended)**
1. If you know exact GPS bounds, you need to manually adjust the selection rectangle
2. The selection rectangle in OSMWebWizard corresponds to the area that will become `origBoundary`
3. **Tip**: Select slightly larger than needed (add 5-10% padding) to ensure coverage

**Method C: Use Known Coordinates**
- Unfortunately, OSMWebWizard doesn't allow direct coordinate input
- You must visually align the selection rectangle
- Use the map's coordinate display (if available) to verify bounds

#### Step 5: Configure Network Options

**Important settings**:

- ✅ **Add Polygon**: Enable (includes buildings, adds detail)
- ❌ **Left-hand Traffic**: Disable (unless UK, Japan, Australia, etc.)
- ❌ **Car-only Network**: Disable (include all road types for completeness)
- ✅ **Import Public Transport**: Enable (optional, adds bus/tram routes)

**Why these matter**: Different options can affect the network size and coverage, which might change the effective bounds.

#### Step 6: Generate the Network

1. Click **"Generate Scenario"**
2. Wait for processing (5-60 minutes depending on area size)
3. Files are saved in: `~/SUMO/YYYY-MM-DD-HH-MM-SS/`

#### Step 7: Verify `origBoundary` Exists

**Critical Check**: Open the generated `.net.xml` file:

```bash
grep origBoundary ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.net.xml
```

**Expected output**:
```xml
<location ... origBoundary="-8.7,41.1,-8.5,41.2" .../>
```

**If `origBoundary` is missing or `0,0,0,0`**:
- ❌ The network is **incomplete** and won't work with map tiles
- ❌ GPS-to-SUMO conversion will fail
- ✅ **Solution**: Regenerate with a smaller area or different options

#### Step 8: Verify Alignment

**Test GPS-to-SUMO conversion**:
```python
from src.utils.network_parser import NetworkParser

parser = NetworkParser("new/network.net.xml")

# Check origBoundary exists
if not parser.orig_boundary:
    print("❌ origBoundary missing - network incomplete!")
    exit(1)

print(f"✅ origBoundary: {parser.orig_boundary}")

# Test conversion with a known GPS point in the area
lon, lat = -8.61, 41.15  # Example: Porto center
sumo_coords = parser.gps_to_sumo_coords(lon, lat)

if sumo_coords:
    print(f"✅ GPS conversion works: ({lon}, {lat}) -> {sumo_coords}")
else:
    print("❌ GPS conversion failed")
```

**Test map tile alignment**:
1. Load the network in your application
2. Enable OSM map tiles
3. Verify tiles align with the network roads
4. If misaligned, check:
   - `origBoundary` matches the selected area
   - Network bounds are reasonable
   - No coordinate system errors

## Common Mistakes and How to Avoid Them

### Mistake 1: Selecting Wrong Area Size

**Problem**: Selected area in OSMWebWizard doesn't match your needs.

**Symptoms**:
- Network too small (missing roads you need)
- Network too large (includes unnecessary areas, slower processing)
- Map tiles don't cover the network area

**Solution**:
- Measure your required GPS bounds first
- Select area that matches those bounds (with small padding)
- Verify `origBoundary` after generation matches your requirements

### Mistake 2: Missing `origBoundary`

**Problem**: Generated network doesn't have `origBoundary` attribute.

**Symptoms**:
- `origBoundary="0,0,0,0"` or missing
- GPS-to-SUMO conversion returns `None`
- Map tiles don't load or are misaligned

**Solution**:
- Use OSMWebWizard (not manual netconvert)
- Ensure SUMO version is recent (1.0.0+)
- Try smaller area if generation fails
- Check browser console for errors

### Mistake 3: Coordinate Order Confusion

**Problem**: Mixing up latitude/longitude order.

**Different tools use different orders**:
- **OSMWebWizard search**: `lat,lon` (e.g., `41.15,-8.61`)
- **origBoundary in .net.xml**: `lon,lat,lon,lat` (e.g., `-8.7,41.1,-8.5,41.2`)
- **Bounding box (OSM export)**: `min_lat, min_lon, max_lat, max_lon`

**Solution**:
- Always check the format expected by each tool
- Document your bounds clearly with format: `lon_min, lat_min, lon_max, lat_max`
- Use the guide's format reference

### Mistake 4: Using Wrong Network File

**Problem**: Replacing network but `origBoundary` doesn't match your data.

**Symptoms**:
- GPS trajectories don't align with network
- Map tiles are offset
- Routes calculated incorrectly

**Solution**:
- Always verify `origBoundary` before using a network
- Keep backups of working networks
- Document which `origBoundary` each dataset requires

## Best Practices

1. **Always verify `origBoundary`** after generating a network
2. **Document GPS bounds** used for each network (save in a text file)
3. **Test GPS conversion** before using in production
4. **Keep backups** of working networks
5. **Use consistent area selection** - if regenerating, use the same bounds
6. **Start small** - test with a small area first, then scale up

## Quick Reference: Coordinate Formats

| Tool/Context | Format | Example |
|-------------|--------|---------|
| OSMWebWizard search | `lat,lon` | `41.15,-8.61` |
| origBoundary (.net.xml) | `lon_min,lat_min,lon_max,lat_max` | `-8.7,41.1,-8.5,41.2` |
| OSM Export bbox | `min_lat,min_lon,max_lat,max_lon` | `41.1,-8.7,41.2,-8.5` |
| Python dict (this codebase) | `{'lon_min': ..., 'lat_min': ..., 'lon_max': ..., 'lat_max': ...}` | See NetworkParser |

## Example: Downloading Porto Network

Assuming you want to regenerate the Porto network:

1. **Get current bounds** (if replacing existing):
   ```python
   from src.utils.network_parser import NetworkParser
   parser = NetworkParser("Porto/config/porto.net.xml")
   print(parser.orig_boundary)
   # Output: {'lon_min': -8.7, 'lat_min': 41.1, 'lon_max': -8.5, 'lat_max': 41.2}
   ```

2. **Launch OSMWebWizard**:
   ```bash
   python /usr/share/sumo/tools/osmWebWizard.py
   ```

3. **Navigate to Porto**:
   - Search: `Porto, Portugal` or `41.15,-8.61`
   - Zoom to show entire city area

4. **Select area**:
   - Enable "Select Area"
   - Adjust rectangle to cover Porto (approximately `-8.7, 41.1, -8.5, 41.2`)
   - Add small padding (5-10%)

5. **Configure**:
   - Add Polygon: ✅
   - Car-only: ❌
   - Left-hand: ❌

6. **Generate** and wait

7. **Verify**:
   ```bash
   grep origBoundary ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.net.xml
   # Should show: origBoundary="-8.7,41.1,-8.5,41.2" (or very close)
   ```

8. **Replace old network**:
   ```bash
   cp ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.net.xml Porto/config/porto.net.xml
   ```

9. **Test alignment**:
   - Load network in application
   - Enable OSM tiles
   - Verify roads align with map

## Summary

The alignment between map tiles and SUMO networks depends on:

1. ✅ **`origBoundary`** in the network file (GPS bounds of original area)
2. ✅ **`convBoundary`** in the network file (SUMO coordinate bounds)
3. ✅ **Linear interpolation** between GPS and SUMO coordinates
4. ✅ **Y-flip** for screen display (SUMO Y-up → screen Y-down)

**To download a new network correctly**:
1. Know your required GPS bounds
2. Select matching area in OSMWebWizard
3. Verify `origBoundary` exists after generation
4. Test GPS-to-SUMO conversion
5. Verify map tile alignment

**Key takeaway**: The `origBoundary` attribute is the critical link. Without it, nothing aligns. Always verify it exists and matches your requirements.




