#!/usr/bin/env python3
"""
Script to analyze Step 1 route calculation on 1000 trajectories.
Conducts statistical analysis including similarity scores, costs, distances, etc.
"""

import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Setup SUMO path before importing sumolib
sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
# Try multiple common SUMO paths
sumo_paths = [sumo_home, '/usr/share/sumo', '/usr/shared/sumo']
for path in sumo_paths:
    if os.path.exists(path):
        tools_path = os.path.join(path, 'tools')
        if os.path.exists(tools_path) and tools_path not in sys.path:
            sys.path.insert(0, tools_path)
            break

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.network_parser import NetworkParser
from src.gui.dataset_conversion_page import EdgeSpatialIndex

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configuration
NUM_TRAJECTORIES = 1000
CSV_PATH = Path("Porto/dataset/train.csv")
NETWORK_PATH = Path("Porto/config/porto.net.xml")
OUTPUT_DIR = Path("step1_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)


class Step1Analyzer:
    """Analyzer for Step 1 route calculation."""
    
    def __init__(self, network_path: str, csv_path: str):
        """Initialize analyzer with network and CSV path."""
        self.network_path = network_path
        self.csv_path = csv_path
        
        # Load network
        print("Loading network...")
        self.network_parser = NetworkParser(str(network_path))
        
        # Load sumolib network for routing
        try:
            import sumolib
            print("Loading SUMO network for routing...")
            self.sumo_net = sumolib.net.readNet(str(network_path))
            print(f"✓ SUMO network loaded: {len(self.sumo_net.getEdges())} edges")
        except ImportError as e:
            print(f"ERROR: sumolib not available. Please install SUMO and set SUMO_HOME.")
            print(f"Import error: {e}")
            print(f"Tried SUMO_HOME: {os.environ.get('SUMO_HOME', 'not set')}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load SUMO network: {e}")
            sys.exit(1)
        
        # Build spatial index
        print("Building spatial index...")
        edges_dict = self.network_parser.get_edges()
        self.edge_spatial_index = EdgeSpatialIndex(edges_dict, cell_size=500.0)
        print(f"✓ Spatial index built")
        
        # Statistics storage
        self.results = []
    
    def load_trajectory(self, trip_num: int) -> Optional[List[List[float]]]:
        """Load a trajectory from CSV by trip number."""
        import csv
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for i, row in enumerate(reader, 1):
                if i == trip_num:
                    # Find POLYLINE column (last column)
                    if len(row) < 9:
                        return None
                    
                    polyline_str = row[-1]
                    try:
                        # Remove quotes if present
                        if polyline_str.startswith('"') and polyline_str.endswith('"'):
                            polyline_str = polyline_str[1:-1]
                        polyline = ast.literal_eval(polyline_str)
                        if isinstance(polyline, list) and len(polyline) >= 2:
                            return polyline
                    except Exception as e:
                        print(f"Error parsing polyline for trip {trip_num}: {e}")
                        return None
        
        return None
    
    def find_candidate_edges(self, lon: float, lat: float, max_candidates: int = 3) -> List[Tuple[str, float]]:
        """Find candidate edges for a GPS point."""
        # Convert GPS to SUMO coordinates
        sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
        if not sumo_coords:
            return []
        
        x, y = sumo_coords
        
        # Use spatial index
        search_radius = 500.0
        candidate_edge_ids = self.edge_spatial_index.get_candidates_in_radius(x, y, radius=search_radius)
        
        if not candidate_edge_ids:
            return []
        
        # Calculate precise distance
        edges_dict = self.network_parser.get_edges()
        edge_distances = []
        
        for edge_id in candidate_edge_ids:
            if edge_id not in edges_dict:
                continue
            
            edge_data = edges_dict[edge_id]
            lanes = edge_data.get('lanes', [])
            if not lanes:
                continue
            
            shape = lanes[0].get('shape', [])
            if len(shape) < 2:
                continue
            
            # Calculate minimum distance
            min_dist = float('inf')
            for i in range(len(shape) - 1):
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                dx = x2 - x1
                dy = y2 - y1
                len_sq = dx * dx + dy * dy
                if len_sq == 0:
                    dist = math.sqrt((x - x1)**2 + (y - y1)**2)
                else:
                    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / len_sq))
                    closest_x = x1 + t * dx
                    closest_y = y1 + t * dy
                    dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                edge_distances.append((edge_id, min_dist))
        
        # Deduplicate by base edge ID
        base_edge_map = {}
        for edge_id, distance in edge_distances:
            base_id = edge_id.split('#')[0] if '#' in edge_id else edge_id
            if base_id not in base_edge_map or distance < base_edge_map[base_id][1]:
                base_edge_map[base_id] = (edge_id, distance)
        
        # Sort and return top candidates
        candidates = sorted(base_edge_map.values(), key=lambda x: x[1])
        return candidates[:max_candidates]
    
    def calculate_route_similarity_score(
        self,
        route_edges: List[str],
        gps_points: List[List[float]]
    ) -> Tuple[float, float, float]:
        """
        Calculate similarity score for a route.
        Returns: (similarity_score, coverage_score, distance_score)
        """
        if not route_edges or not gps_points:
            return (0.0, 0.0, 0.0)
        
        route_base_ids = {edge_id.split('#')[0] for edge_id in route_edges}
        edges_dict = self.network_parser.get_edges()
        
        matched_points = 0
        total_distance = 0.0
        valid_points = 0
        
        for lon, lat in gps_points:
            candidates = self.find_candidate_edges(lon, lat, max_candidates=3)
            if not candidates:
                continue
            
            valid_points += 1
            
            # Coverage check
            candidate_base_ids = {edge_id.split('#')[0] for edge_id, _ in candidates}
            if route_base_ids.intersection(candidate_base_ids):
                matched_points += 1
            
            # Distance calculation
            sumo_coords = self.network_parser.gps_to_sumo_coords(lon, lat)
            if not sumo_coords:
                continue
            
            x, y = sumo_coords
            min_dist_to_route = float('inf')
            
            for edge_id in route_edges:
                if edge_id not in edges_dict:
                    continue
                
                edge_data = edges_dict[edge_id]
                lanes = edge_data.get('lanes', [])
                if not lanes:
                    continue
                
                shape = lanes[0].get('shape', [])
                if len(shape) < 2:
                    continue
                
                for i in range(len(shape) - 1):
                    x1, y1 = shape[i]
                    x2, y2 = shape[i + 1]
                    dx = x2 - x1
                    dy = y2 - y1
                    len_sq = dx * dx + dy * dy
                    
                    if len_sq == 0:
                        dist = math.sqrt((x - x1)**2 + (y - y1)**2)
                    else:
                        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / len_sq))
                        closest_x = x1 + t * dx
                        closest_y = y1 + t * dy
                        dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                    
                    min_dist_to_route = min(min_dist_to_route, dist)
            
            if min_dist_to_route < float('inf'):
                total_distance += min_dist_to_route
        
        if valid_points == 0:
            return (0.0, 0.0, 0.0)
        
        coverage_score = matched_points / valid_points
        avg_distance = total_distance / valid_points
        scale_factor = 100.0
        distance_score = math.exp(-avg_distance / scale_factor) if avg_distance < float('inf') else 0.0
        
        similarity_score = 0.6 * coverage_score + 0.4 * distance_score
        
        return (similarity_score, coverage_score, distance_score)
    
    def calculate_step1_route(self, segment: List[List[float]], seg_idx: int) -> List[Dict]:
        """Calculate Step 1 routes and return detailed results."""
        if len(segment) < 2:
            return []
        
        # Step 1.1: Find candidate edges
        start_lon, start_lat = segment[0]
        dest_lon, dest_lat = segment[-1]
        
        start_candidates = self.find_candidate_edges(start_lon, start_lat, max_candidates=3)
        dest_candidates = self.find_candidate_edges(dest_lon, dest_lat, max_candidates=3)
        
        if not start_candidates or not dest_candidates:
            return []
        
        # Step 1.2: Calculate k-shortest paths
        all_routes = []
        
        for start_edge_id, start_dist in start_candidates:
            for dest_edge_id, dest_dist in dest_candidates:
                try:
                    start_base_id = start_edge_id.split('#')[0]
                    dest_base_id = dest_edge_id.split('#')[0]
                    
                    if not self.sumo_net.hasEdge(start_base_id) or not self.sumo_net.hasEdge(dest_base_id):
                        continue
                    
                    start_edge = self.sumo_net.getEdge(start_base_id)
                    dest_edge = self.sumo_net.getEdge(dest_base_id)
                    
                    # Calculate k-shortest paths
                    if hasattr(self.sumo_net, 'getKShortestPaths'):
                        try:
                            k_routes = self.sumo_net.getKShortestPaths(start_edge, dest_edge, 5)
                            if k_routes:
                                for route_result in k_routes:
                                    if route_result and len(route_result) >= 2:
                                        route_edges, cost = route_result
                                        if route_edges:
                                            edge_ids = [edge.getID() for edge in route_edges]
                                            all_routes.append((edge_ids, cost))
                        except Exception:
                            # Fallback
                            route_result = self.sumo_net.getShortestPath(start_edge, dest_edge)
                            if route_result and len(route_result) >= 2:
                                route_edges, cost = route_result
                                if route_edges:
                                    edge_ids = [edge.getID() for edge in route_edges]
                                    all_routes.append((edge_ids, cost))
                    else:
                        route_result = self.sumo_net.getShortestPath(start_edge, dest_edge)
                        if route_result and len(route_result) >= 2:
                            route_edges, cost = route_result
                            if route_edges:
                                edge_ids = [edge.getID() for edge in route_edges]
                                all_routes.append((edge_ids, cost))
                except Exception:
                    continue
        
        if not all_routes:
            return []
        
        # Step 1.3: Get top 5 unique routes
        all_routes.sort(key=lambda x: x[1])
        seen_routes = set()
        top_routes = []
        for route_edges, cost in all_routes:
            route_tuple = tuple(route_edges)
            if route_tuple not in seen_routes:
                seen_routes.add(route_tuple)
                top_routes.append((route_edges, cost))
                if len(top_routes) >= 5:
                    break
        
        # Calculate similarity scores
        route_results = []
        for route_edges, cost in top_routes:
            similarity, coverage, distance_score = self.calculate_route_similarity_score(route_edges, segment)
            route_results.append({
                'route_edges': route_edges,
                'cost': cost,
                'similarity_score': similarity,
                'coverage_score': coverage,
                'distance_score': distance_score,
                'num_edges': len(route_edges),
                'num_gps_points': len(segment)
            })
        
        # Sort by similarity score
        route_results.sort(key=lambda x: (-x['similarity_score'], x['cost']))
        
        return route_results
    
    def analyze_trajectories(self, num_trajectories: int = 1000):
        """Analyze Step 1 on multiple trajectories."""
        print(f"\nAnalyzing Step 1 on {num_trajectories} trajectories...")
        
        successful = 0
        failed = 0
        
        for trip_num in tqdm(range(1, num_trajectories + 1), desc="Processing trajectories"):
            trajectory = self.load_trajectory(trip_num)
            if not trajectory or len(trajectory) < 2:
                failed += 1
                continue
            
            route_results = self.calculate_step1_route(trajectory, trip_num)
            if not route_results:
                failed += 1
                continue
            
            # Store results
            for route_idx, route_data in enumerate(route_results):
                self.results.append({
                    'trip_num': trip_num,
                    'route_rank': route_idx + 1,  # 1 = best, 2 = second best, etc.
                    'cost': route_data['cost'],
                    'similarity_score': route_data['similarity_score'],
                    'coverage_score': route_data['coverage_score'],
                    'distance_score': route_data['distance_score'],
                    'num_edges': route_data['num_edges'],
                    'num_gps_points': route_data['num_gps_points']
                })
            
            successful += 1
        
        print(f"\n✓ Completed: {successful} successful, {failed} failed")
        return successful, failed
    
    def generate_statistics(self):
        """Generate detailed statistics."""
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("STEP 1 ROUTE CALCULATION STATISTICAL ANALYSIS")
        print("="*80)
        
        # Overall statistics
        print("\n1. OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total trajectories analyzed: {df['trip_num'].nunique()}")
        print(f"Total routes calculated: {len(df)}")
        print(f"Average routes per trajectory: {len(df) / df['trip_num'].nunique():.2f}")
        
        # Statistics by route rank
        print("\n2. STATISTICS BY ROUTE RANK (1 = Best, 5 = Worst)")
        print("-" * 80)
        for rank in sorted(df['route_rank'].unique()):
            rank_df = df[df['route_rank'] == rank]
            print(f"\nRank {rank} (n={len(rank_df)}):")
            print(f"  Similarity Score: mean={rank_df['similarity_score'].mean():.4f}, "
                  f"std={rank_df['similarity_score'].std():.4f}, "
                  f"min={rank_df['similarity_score'].min():.4f}, "
                  f"max={rank_df['similarity_score'].max():.4f}")
            print(f"  Coverage Score: mean={rank_df['coverage_score'].mean():.4f}, "
                  f"std={rank_df['coverage_score'].std():.4f}")
            print(f"  Distance Score: mean={rank_df['distance_score'].mean():.4f}, "
                  f"std={rank_df['distance_score'].std():.4f}")
            print(f"  Cost: mean={rank_df['cost'].mean():.2f}, "
                  f"std={rank_df['cost'].std():.2f}, "
                  f"min={rank_df['cost'].min():.2f}, "
                  f"max={rank_df['cost'].max():.2f}")
            print(f"  Number of Edges: mean={rank_df['num_edges'].mean():.1f}, "
                  f"std={rank_df['num_edges'].std():.1f}, "
                  f"min={rank_df['num_edges'].min()}, "
                  f"max={rank_df['num_edges'].max()}")
        
        # Percentiles
        print("\n3. PERCENTILES FOR BEST ROUTE (Rank 1)")
        print("-" * 80)
        best_routes = df[df['route_rank'] == 1]
        for metric in ['similarity_score', 'coverage_score', 'distance_score', 'cost', 'num_edges']:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            print(f"\n{metric.replace('_', ' ').title()}:")
            for p in percentiles:
                val = best_routes[metric].quantile(p / 100)
                print(f"  {p:2d}th percentile: {val:.4f}")
        
        # Save detailed statistics to CSV
        stats_file = OUTPUT_DIR / "detailed_statistics.csv"
        df.to_csv(stats_file, index=False)
        print(f"\n✓ Detailed statistics saved to: {stats_file}")
        
        return df
    
    def generate_plots(self, df: pd.DataFrame):
        """Generate plots and diagrams."""
        print("\nGenerating plots...")
        
        # 1. Similarity Score Distribution by Rank
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Similarity scores by rank
        ax = axes[0, 0]
        for rank in sorted(df['route_rank'].unique()):
            rank_df = df[df['route_rank'] == rank]
            ax.hist(rank_df['similarity_score'], bins=50, alpha=0.6, label=f'Rank {rank}', density=True)
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Similarity Score Distribution by Route Rank')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot of similarity scores
        ax = axes[0, 1]
        df.boxplot(column='similarity_score', by='route_rank', ax=ax)
        ax.set_xlabel('Route Rank')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity Score by Route Rank (Box Plot)')
        plt.suptitle('')
        
        # Coverage vs Distance scores
        ax = axes[1, 0]
        best_routes = df[df['route_rank'] == 1]
        ax.scatter(best_routes['coverage_score'], best_routes['distance_score'], 
                  alpha=0.5, s=20)
        ax.set_xlabel('Coverage Score')
        ax.set_ylabel('Distance Score')
        ax.set_title('Coverage vs Distance Score (Best Routes)')
        ax.grid(True, alpha=0.3)
        
        # Cost vs Similarity Score
        ax = axes[1, 1]
        for rank in sorted(df['route_rank'].unique())[:3]:  # Top 3 ranks
            rank_df = df[df['route_rank'] == rank]
            ax.scatter(rank_df['cost'], rank_df['similarity_score'], 
                      alpha=0.5, s=20, label=f'Rank {rank}')
        ax.set_xlabel('Route Cost')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Cost vs Similarity Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "similarity_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: similarity_analysis.png")
        plt.close()
        
        # 2. Route Statistics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Number of edges distribution
        ax = axes[0, 0]
        best_routes = df[df['route_rank'] == 1]
        ax.hist(best_routes['num_edges'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Frequency')
        ax.set_title('Number of Edges Distribution (Best Routes)')
        ax.grid(True, alpha=0.3)
        
        # Cost distribution
        ax = axes[0, 1]
        for rank in sorted(df['route_rank'].unique()):
            rank_df = df[df['route_rank'] == rank]
            ax.hist(rank_df['cost'], bins=50, alpha=0.5, label=f'Rank {rank}', density=True)
        ax.set_xlabel('Route Cost')
        ax.set_ylabel('Density')
        ax.set_title('Route Cost Distribution by Rank')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Similarity score comparison across ranks
        ax = axes[1, 0]
        rank_means = df.groupby('route_rank')['similarity_score'].mean()
        rank_stds = df.groupby('route_rank')['similarity_score'].std()
        ax.errorbar(rank_means.index, rank_means.values, yerr=rank_stds.values, 
                   marker='o', capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel('Route Rank')
        ax.set_ylabel('Mean Similarity Score')
        ax.set_title('Mean Similarity Score by Route Rank (with std dev)')
        ax.grid(True, alpha=0.3)
        
        # GPS points vs Route edges
        ax = axes[1, 1]
        best_routes = df[df['route_rank'] == 1]
        ax.scatter(best_routes['num_gps_points'], best_routes['num_edges'], 
                  alpha=0.5, s=20)
        ax.set_xlabel('Number of GPS Points')
        ax.set_ylabel('Number of Route Edges')
        ax.set_title('GPS Points vs Route Edges (Best Routes)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "route_statistics.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: route_statistics.png")
        plt.close()
        
        # 3. Score Components Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        best_routes = df[df['route_rank'] == 1]
        
        # Coverage score distribution
        ax = axes[0]
        ax.hist(best_routes['coverage_score'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        ax.set_xlabel('Coverage Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Coverage Score Distribution (Best Routes)')
        ax.axvline(best_routes['coverage_score'].mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Mean: {best_routes["coverage_score"].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distance score distribution
        ax = axes[1]
        ax.hist(best_routes['distance_score'], bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
        ax.set_xlabel('Distance Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance Score Distribution (Best Routes)')
        ax.axvline(best_routes['distance_score'].mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Mean: {best_routes["distance_score"].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Similarity score distribution
        ax = axes[2]
        ax.hist(best_routes['similarity_score'], bins=50, alpha=0.7, edgecolor='black', color='salmon')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Similarity Score Distribution (Best Routes)')
        ax.axvline(best_routes['similarity_score'].mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Mean: {best_routes["similarity_score"].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "score_components.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: score_components.png")
        plt.close()
        
        # 4. Comparison across all ranks
        fig, ax = plt.subplots(figsize=(14, 8))
        
        rank_data = []
        for rank in sorted(df['route_rank'].unique()):
            rank_df = df[df['route_rank'] == rank]
            rank_data.append({
                'rank': rank,
                'similarity_mean': rank_df['similarity_score'].mean(),
                'similarity_std': rank_df['similarity_score'].std(),
                'coverage_mean': rank_df['coverage_score'].mean(),
                'distance_mean': rank_df['distance_score'].mean(),
                'cost_mean': rank_df['cost'].mean()
            })
        
        rank_df_plot = pd.DataFrame(rank_data)
        x = np.arange(len(rank_df_plot))
        width = 0.25
        
        ax.bar(x - width, rank_df_plot['similarity_mean'], width, 
              yerr=rank_df_plot['similarity_std'], label='Similarity', alpha=0.8)
        ax.bar(x, rank_df_plot['coverage_mean'], width, label='Coverage', alpha=0.8)
        ax.bar(x + width, rank_df_plot['distance_mean'], width, label='Distance', alpha=0.8)
        
        ax.set_xlabel('Route Rank')
        ax.set_ylabel('Score')
        ax.set_title('Mean Scores by Route Rank')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rank {r}' for r in rank_df_plot['rank']])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rank_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: rank_comparison.png")
        plt.close()
        
        print(f"\n✓ All plots saved to: {OUTPUT_DIR}")


def main():
    """Main function."""
    print("="*80)
    print("STEP 1 ROUTE CALCULATION ANALYSIS")
    print("="*80)
    
    # Check files exist
    if not CSV_PATH.exists():
        print(f"ERROR: CSV file not found: {CSV_PATH}")
        sys.exit(1)
    
    if not NETWORK_PATH.exists():
        print(f"ERROR: Network file not found: {NETWORK_PATH}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = Step1Analyzer(str(NETWORK_PATH), str(CSV_PATH))
    
    # Analyze trajectories
    successful, failed = analyzer.analyze_trajectories(NUM_TRAJECTORIES)
    
    if successful == 0:
        print("ERROR: No successful analyses!")
        sys.exit(1)
    
    # Generate statistics
    df = analyzer.generate_statistics()
    
    # Generate plots
    analyzer.generate_plots(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

