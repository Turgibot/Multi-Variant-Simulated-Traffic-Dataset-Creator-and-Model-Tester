# Improved Network Generation Guide

## Problem: Network Gaps

If you're experiencing network gaps (routes cannot be found between edges), the current network conversion might be too restrictive. This guide provides better alternatives.

## Best Option: OSM Web Wizard (Recommended)

**OSM Web Wizard** produces the most complete networks because it:
- Uses optimized conversion settings
- Handles complex road networks better
- Provides interactive area selection
- Generates more complete networks

### Steps:

1. **Run OSM Web Wizard:**
   ```bash
   export SUMO_HOME=/usr/share/sumo  # or your SUMO path
   python $SUMO_HOME/tools/osmWebWizard.py
   ```

2. **In the web interface:**
   - Search for "Porto, Portugal"
   - Zoom to cover the entire area where taxi trajectories exist
   - Select a bounding box that covers:
     - Latitude: 41.02° to 41.26°
     - Longitude: -8.67° to -8.45°
   - **Important options:**
     - ✅ **Car-only Network**: Checked (keeps all vehicle roads)
     - ❌ **Left-hand Traffic**: Unchecked (Portugal uses right-hand)
     - ✅ **Add Polygon**: Optional (for buildings)

3. **Generate the network:**
   - Click "Generate Scenario"
   - Wait for processing (may take 10-30 minutes)

4. **Copy the generated network:**
   ```bash
   # Find the generated directory (usually ~/SUMO/YYYY-MM-DD-HH-MM-SS/)
   cp ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.net.xml Porto/config/porto.net.xml
   cp ~/SUMO/YYYY-MM-DD-HH-MM-SS/*.sumocfg Porto/config/porto.sumocfg
   ```

## Alternative: Improved Python Script

If you prefer automated conversion, use the improved script:

```bash
cd Porto/scripts
python3 download_porto_network_improved.py
```

**Key improvements:**
- Only removes pedestrian/bicycle paths (keeps all vehicle roads)
- Explicitly keeps passenger vehicles and taxis
- Uses edge joining to improve connectivity
- Falls back to permissive mode if initial conversion fails

## Manual Conversion with Better Options

If you want to convert manually with optimal settings:

```bash
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
  --remove-edges.by-vclass pedestrian,bicycle \
  --keep-edges.by-vclass passenger,taxi \
  --edges.join \
  --keep-edges.min-speed 0.1
```

**Key differences from original:**
- Removed: `--remove-edges.by-vclass` with many vehicle types
- Added: `--keep-edges.by-vclass passenger,taxi` (explicitly keep these)
- Added: `--edges.join` (improves connectivity)
- Added: `--keep-edges.min-speed 0.1` (keeps all drivable roads)

## Verify Network Quality

After generating a new network, verify it:

1. **Check network statistics:**
   ```bash
   netedit Porto/config/porto.net.xml
   ```
   Look for:
   - Number of edges (should be > 200,000 for Porto)
   - Number of nodes
   - Network bounds (should cover your GPS trajectory area)

2. **Test route finding:**
   - Use the application's "Show SUMO route" feature
   - Check if routes are found for problematic segments
   - If routes still fail, the network may need further improvement

3. **Check for disconnected components:**
   ```bash
   # Use SUMO's netdump tool to analyze connectivity
   netdump Porto/config/porto.net.xml --print-statistics
   ```

## Troubleshooting

### Still Getting Gaps?

1. **Try OSM Web Wizard** - It's usually the most reliable
2. **Check OSM data coverage** - Ensure the OSM file covers the full area
3. **Use a larger bounding box** - Include more surrounding area
4. **Check edge filtering** - Make sure important roads aren't being filtered out

### Network Too Large?

If the network is too large and slow:
- Use a smaller bounding box
- Remove `--edges.join` (but this may create more gaps)
- Use `--remove-edges.by-vclass` more selectively

### Network Too Small?

If routes are still missing:
- Use OSM Web Wizard (best option)
- Try the improved script with permissive settings
- Check if OSM data itself has gaps (use JOSM or QGIS to inspect)

## Comparison of Methods

| Method | Completeness | Ease of Use | Recommended |
|--------|--------------|-------------|-------------|
| OSM Web Wizard | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **Best** |
| Improved Script | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Good |
| Original Script | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ May have gaps |
| Manual netconvert | ⭐⭐⭐⭐ | ⭐⭐ | ✅ Good if you know options |

## Next Steps

After generating an improved network:
1. Replace `Porto/config/porto.net.xml` with the new file
2. Restart the application
3. Test route finding with "Show SUMO route" checkbox
4. Check if gaps are resolved

