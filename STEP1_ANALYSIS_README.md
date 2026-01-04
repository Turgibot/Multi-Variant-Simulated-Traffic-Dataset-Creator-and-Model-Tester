# Step 1 Route Calculation Analysis

## Overview

This script analyzes Step 1 route calculation on 1000 trajectories from the Porto taxi dataset. It conducts a comprehensive statistical analysis including similarity scores, costs, distances, and generates detailed plots and diagrams.

## Requirements

- Python 3.8+
- SUMO installed with sumolib available
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - tqdm

## Usage

```bash
python3 analyze_step1_routes.py
```

## What It Does

1. **Loads Network**: Loads the Porto SUMO network and builds spatial index
2. **Processes Trajectories**: Loads 1000 trajectories from `Porto/dataset/train.csv`
3. **Calculates Routes**: For each trajectory:
   - Finds candidate edges for start and destination
   - Calculates up to 5 k-shortest paths
   - Calculates similarity scores for each route
   - Selects best route based on similarity score
4. **Collects Statistics**: Gathers comprehensive statistics on:
   - Similarity scores (overall, by rank, percentiles)
   - Coverage scores
   - Distance scores
   - Route costs
   - Number of edges
   - GPS point counts
5. **Generates Analysis**:
   - Detailed CSV with all statistics
   - Multiple plots and diagrams
   - Statistical summaries

## Output

All results are saved to `step1_analysis_results/` directory:

### Files Generated

1. **detailed_statistics.csv**: Complete dataset with all route calculations
   - Columns: trip_num, route_rank, cost, similarity_score, coverage_score, distance_score, num_edges, num_gps_points

2. **similarity_analysis.png**: 
   - Similarity score distributions by rank
   - Box plots
   - Coverage vs Distance scatter
   - Cost vs Similarity scatter

3. **route_statistics.png**:
   - Number of edges distribution
   - Cost distributions
   - Mean similarity scores by rank
   - GPS points vs Route edges

4. **score_components.png**:
   - Coverage score distribution
   - Distance score distribution
   - Similarity score distribution

5. **rank_comparison.png**:
   - Bar chart comparing mean scores across all ranks

### Statistics Reported

- Overall statistics (total trajectories, routes, averages)
- Statistics by route rank (1-5)
- Percentiles for best routes (10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Min, max, mean, std for all metrics

## Configuration

You can modify these constants in the script:

```python
NUM_TRAJECTORIES = 1000  # Number of trajectories to analyze
CSV_PATH = Path("Porto/dataset/train.csv")
NETWORK_PATH = Path("Porto/config/porto.net.xml")
OUTPUT_DIR = Path("step1_analysis_results")
```

## Expected Runtime

- Loading network: ~10-30 seconds
- Building spatial index: ~1-2 seconds
- Processing 1000 trajectories: ~10-30 minutes (depends on trajectory complexity and system performance)

## Notes

- The script processes trajectories sequentially
- Failed trajectories (no valid routes found) are skipped
- Progress is shown with a progress bar
- All statistics are printed to console and saved to files

