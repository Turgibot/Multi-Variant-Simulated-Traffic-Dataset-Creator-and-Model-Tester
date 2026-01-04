# Network Update - 2025-12-21

## Summary

All Porto conversion projects have been updated to use the new SUMO network from `2025-12-21-10-08-57`.

## Network Details

- **Source**: `2025-12-21-10-08-57/osm.net.xml.gz`
- **origBoundary**: `-8.715283,41.081167,-8.481458,41.219752` (lon_min,lat_min,lon_max,lat_max)
- **GPS Bounds**:
  - Longitude: -8.715283 to -8.481458
  - Latitude: 41.081167 to 41.219752
- **Network Size**: 80,980 edges
- **Status**: ✅ Has `origBoundary` - map tiles will align correctly

## Updated Locations

The network file `porto.net.xml` has been copied to:

1. **Porto Project (Porto)**: `/home/guy/Projects/Traffic/Develop/Projects/Porto/config/porto.net.xml`
2. **Porto Project (PortoForSumo)**: `/home/guy/Projects/Traffic/Develop/Projects/PortoForSumo/config/porto.net.xml`
3. **Repository Porto Folder**: `Porto/config/porto.net.xml` (fallback location)

## Map Tile Alignment

The network has the required `origBoundary` attribute, which means:

✅ **GPS-to-SUMO conversion will work** - GPS coordinates can be converted to SUMO network coordinates  
✅ **Map tiles will align correctly** - OSM tiles will be positioned correctly over the network  
✅ **Route calculation will work** - GPS trajectories can be mapped to network edges  

### How Map Tiles Align

1. **Tile Download**: OSM tiles are requested based on GPS coordinates (using `origBoundary`)
2. **GPS to SUMO Conversion**: Each tile's GPS corners are converted to SUMO coordinates using linear interpolation between `origBoundary` (GPS) and `convBoundary` (SUMO)
3. **Positioning**: Tiles are positioned at their converted SUMO coordinates
4. **Y-Flip**: Tiles are flipped vertically to match SUMO's Y-up coordinate system

### Verification

To verify map tile alignment:

1. Open any Porto conversion project
2. Enable "Show OSM Map" checkbox
3. Verify that map tiles align with the network roads
4. If misaligned, use the map offset controls to fine-tune

## Configuration Files

The `porto.sumocfg` files in each project already reference `porto.net.xml`, so no changes were needed:

```xml
<input>
    <net-file value="porto.net.xml"/>
    <route-files value="porto.rou.xml"/>
</input>
```

## Testing

To test the network:

```python
from src.utils.network_parser import NetworkParser

# Load network
parser = NetworkParser("Porto/config/porto.net.xml")

# Verify origBoundary exists
if parser.orig_boundary:
    print(f"✅ origBoundary: {parser.orig_boundary}")
    
    # Test GPS conversion
    lon, lat = -8.61, 41.15  # Porto center
    sumo_coords = parser.gps_to_sumo_coords(lon, lat)
    if sumo_coords:
        print(f"✅ GPS conversion works: ({lon}, {lat}) -> {sumo_coords}")
    else:
        print("❌ GPS conversion failed")
else:
    print("❌ origBoundary missing - network incomplete!")
```

## Notes

- The network file is ~200MB (uncompressed)
- All Porto projects will automatically use this network when opened
- The network was generated using OSMWebWizard, ensuring complete metadata
- Map tiles should align correctly without manual adjustment

## Related Documentation

- See `MAP_TILES_AND_NETWORK_ALIGNMENT.md` for detailed explanation of how alignment works
- See `OSMWEBWIZARD_COMPLETE_NETWORK_GUIDE.md` for how to regenerate networks




