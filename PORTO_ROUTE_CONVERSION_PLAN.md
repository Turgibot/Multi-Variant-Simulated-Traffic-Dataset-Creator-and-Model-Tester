# Porto Dataset Route Conversion - Implementation Plan

## Goal
Add Porto taxi trajectory dataset conversion functionality to the route generation page without affecting existing route generation features.

## Approach: Conditional Porto Mode UI

### Strategy
1. **Detect Porto Mode**: Check if Porto mode is enabled for the current project
2. **Conditional UI**: Show Porto-specific conversion UI only when Porto mode is enabled
3. **Isolated Functionality**: Keep Porto conversion code separate from existing route generation
4. **No Breaking Changes**: Existing route generation remains unchanged for non-Porto projects

## Implementation Steps

### Step 1: Add Porto Mode Detection
- In `RouteGenerationPage.__init__()`, check `config_manager.get_use_porto_dataset()`
- Store Porto mode state and paths as instance variables
- Only initialize Porto UI if Porto mode is enabled

### Step 2: Create Porto Conversion Section
- Add a new `QGroupBox` for "Porto Dataset Conversion" 
- Place it at the top of the configuration panel (before existing route config)
- Only show this section when Porto mode is enabled
- Include:
  - Porto dataset file path display
  - Conversion progress/status
  - Convert button
  - Output route file location

### Step 3: Porto Conversion Logic
- Create separate methods for Porto conversion:
  - `convert_porto_trajectories_to_routes()`
  - `parse_porto_csv()`
  - `generate_sumo_routes_from_trajectories()`
- Keep conversion logic isolated from existing route generation

### Step 4: Preserve Existing Functionality
- All existing route generation features remain unchanged
- Porto conversion runs independently
- Both can coexist (Porto conversion generates routes, manual generation still works)

## Code Structure

```python
class RouteGenerationPage(QWidget):
    def __init__(self, ...):
        # ... existing code ...
        
        # Check Porto mode
        self.porto_mode_enabled = self.config_manager.get_use_porto_dataset()
        if self.porto_mode_enabled:
            self.porto_dataset_path = self.config_manager.get_porto_dataset_path()
            # Calculate Porto paths
        
        self.init_ui()
        self.load_network()
    
    def init_ui(self):
        # ... existing UI code ...
        
        # Add Porto conversion section ONLY if Porto mode enabled
        if self.porto_mode_enabled:
            self.add_porto_conversion_section()
        
        # ... rest of existing UI ...
    
    def add_porto_conversion_section(self):
        """Add Porto dataset conversion UI (only called if Porto mode enabled)"""
        # Porto-specific UI here
        pass
    
    def convert_porto_trajectories(self):
        """Convert Porto CSV trajectories to SUMO routes"""
        # Porto conversion logic here
        pass
```

## Benefits

1. **No Breaking Changes**: Existing route generation completely unaffected
2. **Clean Separation**: Porto code is isolated and conditional
3. **Easy to Test**: Can test Porto conversion independently
4. **Maintainable**: Clear separation of concerns

## Files to Modify

1. `src/gui/route_generation_page.py` - Add Porto conversion UI and logic
2. `src/utils/porto_converter.py` (NEW) - Porto trajectory to route conversion utilities

## Testing Strategy

1. Test with Porto mode disabled - ensure existing functionality works
2. Test with Porto mode enabled - ensure Porto UI appears and works
3. Test conversion process - ensure trajectories convert correctly
4. Test that both manual and Porto routes can coexist

