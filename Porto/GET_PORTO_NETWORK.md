# How to Get Porto Map and Network for SUMO

This guide explains how to obtain and convert the Porto, Portugal map to SUMO network format.

## Method 1: Using OSM Web Wizard (Easiest)

### Step 1: Run OSM Web Wizard
```bash
# Make sure SUMO_HOME is set
export SUMO_HOME=/usr/share/sumo  # or your SUMO installation path

# Run the web wizard
python $SUMO_HOME/tools/osmWebWizard.py
```

### Step 2: Navigate to Porto
1. When the browser opens, use the search box to find "Porto, Portugal"
2. Zoom to the area you want to simulate (start with a smaller area first)
3. Check "Select Area" and drag to select the region

### Step 3: Configure Options
- **Add Polygon**: Check if you want buildings
- **Car-only Network**: Recommended for traffic simulation
- **Left-hand Traffic**: Unchecked (Portugal uses right-hand traffic)

### Step 4: Generate Network
1. Click "Generate Scenario"
2. Wait for processing (can take 5-30 minutes depending on area size)
3. Files will be saved in a timestamped directory

### Step 5: Copy Files to Porto/config
```bash
# Find the generated directory (usually in ~/SUMO/ or tools/)
# Copy the network file
cp ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.net.xml Porto/config/porto.net.xml

# Copy the sumocfg file
cp ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.sumocfg Porto/config/porto.sumocfg

# Edit porto.sumocfg to use relative paths
```

## Method 2: Manual Download and Conversion

### Step 1: Download Porto OSM Data

**Option A: Using Overpass API (Recommended for specific areas)**
```bash
# Create a query file for Porto city center
cat > porto_query.overpassql << EOF
[out:xml][timeout:25];
(
  relation["name"="Porto"]["admin_level"="8"];
);
out body;
>;
out skel qt;
EOF

# Download the data
wget --post-file=porto_query.overpassql -O Porto/config/porto.osm "https://overpass-api.de/api/interpreter"
```

**Option B: Using Geofabrik (Full region)**
```bash
# Download Portugal OSM data (includes Porto)
cd Porto/config
wget https://download.geofabrik.de/europe/portugal-latest.osm.pbf

# Extract Porto area using osmium or osmconvert
# You'll need to specify bounding box coordinates for Porto
```

**Option C: Using BBBike (City extracts)**
```bash
# Visit https://extract.bbbike.org/
# Select Porto, Portugal
# Download as OSM XML format
# Save to Porto/config/porto.osm
```

### Step 2: Convert OSM to SUMO Network

```bash
# Make sure SUMO_HOME is set
export SUMO_HOME=/usr/share/sumo

# Convert OSM to SUMO network
netconvert \
  --osm-files Porto/config/porto.osm \
  --output-file Porto/config/porto.net.xml \
  --geometry.remove \
  --roundabouts.guess \
  --ramps.guess \
  --junctions.join \
  --tls.guess-signals \
  --tls.discard-simple \
  --tls.join \
  --no-turnarounds \
  --no-internal-links \
  --remove-edges.by-vclass passenger
```

### Step 3: Create SUMO Configuration File

Create `Porto/config/porto.sumocfg`:

```xml
<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="porto.net.xml"/>
        <route-files value="porto.rou.xml"/>
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
```

### Step 4: Create Empty Route File (for now)

```bash
# Create an empty route file
cat > Porto/config/porto.rou.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
</routes>
EOF
```

## Method 3: Using Python Script (Automated)

See `Porto/scripts/download_porto_network.py` for an automated script.

## Porto Coordinates

If you need bounding box coordinates for Porto:
- **Latitude**: 41.1400° N to 41.2000° N
- **Longitude**: -8.6500° W to -8.5800° W

Or use these coordinates in decimal degrees:
- **North**: 41.2000
- **South**: 41.1400
- **East**: -8.5800
- **West**: -8.6500

## Verification

After conversion, verify the network:

```bash
# Check network statistics
netedit Porto/config/porto.net.xml

# Or use sumo-gui to visualize
sumo-gui -c Porto/config/porto.sumocfg
```

## Troubleshooting

### Network is too large
- Use a smaller bounding box
- Use `--remove-edges.by-vclass` to filter road types
- Use `--edges.join` to simplify the network

### Conversion fails
- Check OSM file is valid XML
- Try with `--geometry.remove` to simplify
- Check SUMO version compatibility

### Missing roads
- Ensure OSM data includes the area
- Check OSM tags are correct
- Try different conversion options

## Next Steps

After getting the network:
1. The network file should be at `Porto/config/porto.net.xml`
2. The config file should be at `Porto/config/porto.sumocfg`
3. Enable "Use Porto Taxi Dataset" checkbox in the application
4. The system will automatically detect and load the network

