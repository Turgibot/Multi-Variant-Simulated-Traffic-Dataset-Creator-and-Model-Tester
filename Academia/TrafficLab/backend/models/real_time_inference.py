"""
Real-Time Inference for Dynamic Intent GNN

This module provides real-time ETA prediction for new vehicles in a SUMO simulation
environment. It integrates with existing simulation data and model checkpoints to
predict travel times for dynamically added vehicles.
"""

import os
import json
import math
import random
import torch
import numpy as np
import pandas as pd
import ast
import re
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data, Batch

# Import existing model components
from models.model_temporal_moe import TemporalMoEETA
from models.utils_targets import invert_to_seconds

def extract_step_number(filename):
    match = re.search(r"step_(\d+)\.pt", filename)
    return int(match.group(1)) if match else -1

def extract_numeric_suffix(s):
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else float('inf')

class RealTimeInference:
    """
    Main inference orchestrator for real-time ETA prediction.
    
    This class coordinates all components needed to predict ETA for a new vehicle
    in a SUMO simulation environment while maintaining temporal consistency with
    the training data.
    """
    
    def __init__(self, checkpoint_path: str = "./logs/one_day/temporal_route_aware/gru/moe_best.manifest.yaml", config_path: str = "./config.yaml", seed: int = 42):
        """
        Initialize the real-time inference system.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (default: best model from logs)
            config_path: Path to configuration file (default: ./config.yaml)
            seed: Random seed for deterministic inference (default: 42)
        """
        self.config_path = config_path
        
        # Set random seed for deterministic inference
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Load configuration first to get data_path and construct default checkpoint path
        self.config = self._load_config()
        self.data_path = self.config["data"]["playback_path"]  # Use playback_path for simulation data
        self.checkpoint_path = checkpoint_path
        
        # Playback parameters
        self.playback_num_files = self.config["data"]["playback_num_files"]
        self.playback_start_idx = self.config["data"]["playback_start_idx"]
        self.window_size = self.config["data"]["window_size"]
        
        # Initialize components
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize statistics data
        self.entities_data = {}
        
        # Load statistics from CSV files
        self._load_statistics()
        
        # Load model and configuration
        self.load_model_and_config()
    
    def set_seed(self, seed: int = 42):
        """Set random seed for deterministic inference."""
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Additional deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🔒 Set deterministic seed to {seed}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_statistics(self):
        """Load statistics from EDA exports CSV files."""
        # Define paths to CSV files
        eda_folder = "models/eda_exports"
        self.vehicle_features_file = os.path.join(eda_folder, "vehicle_feature_summary.csv")
        self.label_features_file = os.path.join(eda_folder, "labels_feature_summary.csv")
        self.edge_features_file = os.path.join(eda_folder, "edge_feature_summary.csv")
        
        # Load statistics for each entity type
        entities = ['vehicle', 'label', 'edge']
        statistics_files = [self.vehicle_features_file, self.label_features_file, self.edge_features_file]
        
        for i, entity in enumerate(entities):
            if entity not in self.entities_data:
                self.entities_data[entity] = {'stats': {}}
            
            if os.path.exists(statistics_files[i]):
                df = pd.read_csv(statistics_files[i])
                for _, row in df.iterrows():
                    feature_name = row['feature']
                    entry = {}
                    
                    # Handle numeric features
                    if row['type'] == 'numeric':
                        entry['mean'] = float(row.get('mean', 0.0))
                        entry['std'] = float(row.get('std', 1.0))
                        entry['min'] = float(row.get('min', 0.0))
                        entry['max'] = float(row.get('max', 1.0))
                        
                        # Add percentiles if available
                        for p in [97, 98, 99]:
                            percentile_name = f'{p}%'
                            if percentile_name in row:
                                entry[percentile_name] = float(row[percentile_name])
                        
                        # Special handling for ETA features
                        if feature_name == 'eta':
                            entry['log_mean'] = float(row.get('log_mean', 0.0))
                            entry['log_std'] = float(row.get('log_std', 1.0))
                    
                    # Handle categorical features
                    elif row['type'] == 'categorical':
                        try:
                            value_counts = ast.literal_eval(row.get('value_counts', '{}')) if pd.notna(row.get('value_counts', '')) else {}
                        except Exception:
                            value_counts = {}
                        entry['keys'] = sorted(value_counts.keys()) if value_counts else []
                    
                    self.entities_data[entity]['stats'][feature_name] = entry
                
                print(f"✅ Loaded {entity} statistics from {statistics_files[i]}")
            else:
                print(f"⚠️  Warning: {statistics_files[i]} not found, using fallback values")
                # Set fallback values for critical features
                if entity == 'vehicle':
                    self.entities_data[entity]['stats'] = {
                        'route_length': {'min': 476.6, 'max': 23133.41},
                        'current_x': {'min': -4.8, 'max': 18004.8},
                        'current_y': {'min': -6269.76, 'max': 5004.8},
                        'destination_x': {'min': 11.416129830593093, 'max': 17988.49958178945},
                        'destination_y': {'min': -6253.4779247844235, 'max': 4988.503014571509},
                        'current_zone': {'keys': ['A', 'B', 'C', 'H']},
                        'vehicle_type': {'keys': ['bus', 'passenger', 'truck']}
                    }
                elif entity == 'label':
                    self.entities_data[entity]['stats'] = {
                        'eta': {
                            'mean': 504.32398466382836,
                            'std': 416.1437177500672,
                            'min': 0.0,
                            'max': 4963.0,
                            'log_mean': 5.5,  # Approximate
                            'log_std': 1.0    # Approximate
                        }
                    }
    
    def load_model_and_config(self):
        """Load trained model and configuration."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        
        # Extract model configuration from checkpoint or use config defaults
        model_config = self.config.get("model", {})
        
        # Create model with configuration from config.yaml
        self.model = TemporalMoEETA(
            node_in_dim=28,  # Fixed node feature dimension
            d_hidden=model_config.get("hidden", 128),
            fusion_out=model_config.get("fusion_out", 192),
            n_experts=model_config.get("experts", 6),
            top_k=model_config.get("top_k", 2),
            dropout=model_config.get("dropout", 0.1),
            predict_on="last",
            edge_vocab_size=1294,  # Fixed edge vocabulary size
            route_emb_dim=64,
            edge_dim=7,
            temporal_kind=model_config.get("temporal_kind", "gru"),
            ablation_variant=model_config.get("ablation_variant", "temporal_route_aware"),
        ).to(self.device)
        
        # Load model state dict (use strict=False to handle architecture differences)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Ensure deterministic behavior
        self.set_seed(self.seed)
        
        print(f"✅ Model loaded from: {self.checkpoint_path}")
        print(f"✅ Model architecture: {model_config.get('ablation_variant', 'temporal_route_aware')}")
        print(f"✅ Temporal kind: {model_config.get('temporal_kind', 'gru')}")
        print(f"✅ Device: {self.device}")
        print(f"✅ Model set to eval mode with deterministic seed")
    
    def add_vehicle_to_last_snapshot(self, current_pt_file : Data, veh_id: str, start_step: int, route_edges: List[str], route_length: float, zone: str, 
                           current_x: float, current_y: float, destination_x: float, 
                           destination_y: float, current_edge_num_lanes: int, 
                           current_edge_id: str) -> Data:
        """
        Add a new vehicle to the last snapshot and return the updated pt file.
        
        Args:
            veh_id: Vehicle ID
            start_step: Current simulation step from SUMO
            route_edges: List of edges on the vehicle route
            route_length: Total route length in meters
            zone: Current zone ('A', 'B', 'C', 'H')
            current_x: Current X coordinate
            current_y: Current Y coordinate
            destination_x: Destination X coordinate
            destination_y: Destination Y coordinate
            current_edge_num_lanes: Number of lanes on current edge (1, 2, or 3)
            current_edge_id: Current edge ID
            
        Returns:
            Data: Updated pt file with the new vehicle added
        """
        # Load temporal window of pt files
        # get the number of vehicles in the 'x' by filtering for node_type == 1
        num_vehicles = current_pt_file.x[current_pt_file.x[:, 0] == 1].shape[0]
        print(f"Number of vehicles in the current pt file before adding new vehicle: {num_vehicles}")
        # Extract current edge demand and occupancy from current pt file
        current_edge_demand, current_edge_occupancy, current_edge_speed = self._get_current_edge_features(
            current_pt_file, current_edge_id
        )
        '''
        {
            "id": "AE0AE1",
            "from": "AE0",
            "to": "AE1",
            "speed": 13.89,
            "length": 482.4,
            "num_lanes": 1,
            "zone": "A",
            "density": 0.0020729684908789387,
            "avg_speed": 12.410898532432654,
            "vehicles_on_road": [
                "veh_131220"
            ]
            },
        '''
        print(f"Current edge demand: {current_edge_demand}, current edge occupancy: {current_edge_occupancy}, current edge speed: {current_edge_speed}")
        # denormalize current edge demand and occupancy
        current_edge_demand = self._denormalize_edge_demand(current_edge_demand)
        current_edge_occupancy = self._denormalize_edge_occupancy(current_edge_occupancy)
        current_edge_speed = self._denormalize_edge_speed(current_edge_speed)
        print(f"Before edge demand denormalized: {current_edge_demand}, current edge occupancy denormalized: {current_edge_occupancy}, current edge speed denormalized: {current_edge_speed}")
        # add the new vehicle effect to the edge attr
        current_edge_demand = current_edge_demand + 1
        current_edge_occupancy = current_edge_occupancy + 1
        current_edge_speed = current_edge_speed
        print(f"After edge demand denormalized: {current_edge_demand}, current edge occupancy denormalized: {current_edge_occupancy}, current edge speed denormalized: {current_edge_speed}")
        #normalize current edge demand, speed and occupancy
        current_edge_demand = self._normalize_edge_demand(current_edge_demand)
        current_edge_occupancy = self._normalize_edge_occupancy(current_edge_occupancy)
        current_edge_speed = self._normalize_edge_speed(current_edge_speed)
        print(f"After edge demand normalized: {current_edge_demand}, current edge occupancy normalized: {current_edge_occupancy}, current_edge_speed normalized: {current_edge_speed}")
        
        # Store original values for comparison BEFORE any updates
        original_edge_demands = {}
        for edge_id in route_edges:
            if edge_id in current_pt_file.edge_ids:
                edge_idx = current_pt_file.edge_ids.index(edge_id)
                # Make a copy of the value to avoid reference issues
                original_value = float(current_pt_file.edge_attr[edge_idx][5])
                original_edge_demands[edge_id] = original_value
        
        # Update the current edge (this might be one of the edges in the route)
        edge_attr = current_pt_file.edge_attr
        edge_idx = current_pt_file.edge_ids.index(current_edge_id)
        edge_attr[edge_idx][0] = current_edge_speed
        edge_attr[edge_idx][5] = current_edge_demand
        edge_attr[edge_idx][6] = current_edge_occupancy
        current_pt_file.edge_attr = edge_attr
        
        # If the current edge is in the route, we need to account for the fact that it was already updated
        if current_edge_id in route_edges:
            # The original value for the current edge should be the value before the current edge update
            # We need to reverse the current edge update to get the true original value
            raw_before_update = self._denormalize_edge_demand(current_edge_demand) - 1
            original_edge_demands[current_edge_id] = self._normalize_edge_demand(raw_before_update)

        # Update edge demand for all edges in the new vehicle's remaining route
        # This follows the same logic as dataset_creator.py: each vehicle contributes +1 
        # to the demand for each edge in its remaining route
        
        for i, edge_id in enumerate(route_edges):
            if edge_id in current_pt_file.edge_ids:
                edge_idx = current_pt_file.edge_ids.index(edge_id)
                
                # Get original normalized edge demand (before any updates)
                original_edge_demand_normalized = original_edge_demands[edge_id]
                
                # Denormalize to get raw count
                current_edge_demand_raw = self._denormalize_edge_demand(original_edge_demand_normalized)
                
                # Add 1 for the new vehicle's demand on this edge
                new_edge_demand_raw = current_edge_demand_raw + 1
                
                # Normalize back
                new_edge_demand_normalized = self._normalize_edge_demand(new_edge_demand_raw)
                
                # Update the edge attribute
                edge_attr[edge_idx][5] = new_edge_demand_normalized
                
                # Calculate change
                change = new_edge_demand_normalized - original_edge_demand_normalized
                
                print(f"Updated edge {edge_id} demand: {current_edge_demand_raw} → {new_edge_demand_raw} (normalized change: {change:.3f})")
            else:
                print(f"Warning: Edge {edge_id} not found in current pt file")
            
            # Update current_edge_demand for vehicles on this edge
            if hasattr(current_pt_file, 'vehicle_ids') and hasattr(current_pt_file, 'current_vehicle_current_edges'):
                for vehicle_idx, vehicle_id in enumerate(current_pt_file.vehicle_ids):
                    if current_pt_file.current_vehicle_current_edges[vehicle_idx] == edge_idx:
                        # Update the current_edge_demand in the vehicle's feature vector
                        if hasattr(current_pt_file, 'x') and current_pt_file.x.shape[0] > vehicle_idx:
                            current_pt_file.x[vehicle_idx][23] = new_edge_demand_normalized  # current_edge_demand is at index 23
            else:
                print(f"Warning: vehicle_ids or current_edge not found in current pt file")
        
        # Update the edge attributes in the pt file
        current_pt_file.edge_attr = edge_attr
        
        # Add the new vehicle to the pt file data
        new_vehicle_idx = len(current_pt_file.vehicle_ids)
        
        # Add vehicle ID
        current_pt_file.vehicle_ids.append(veh_id)
        
        # Add current edge (index of the first edge in route)
        if hasattr(current_pt_file, 'current_vehicle_current_edges'):
            if route_edges and route_edges[0] in current_pt_file.edge_ids:
                current_edge_idx = current_pt_file.edge_ids.index(route_edges[0])
                current_pt_file.current_vehicle_current_edges = torch.cat([current_pt_file.current_vehicle_current_edges, torch.tensor([current_edge_idx])])
            else:
                print(f"Warning: route_edges[0] not found in current pt file")
                current_pt_file.current_vehicle_current_edges = torch.cat([current_pt_file.current_vehicle_current_edges, torch.tensor([0])])  # Default to first edge
        else:
            print(f"Warning: current_vehicle_current_edges not found in current pt file")
            # Create current_vehicle_current_edges attribute if it doesn't exist
            if route_edges and route_edges[0] in current_pt_file.edge_ids:
                current_edge_idx = current_pt_file.edge_ids.index(route_edges[0])
                current_pt_file.current_vehicle_current_edges = torch.tensor([current_edge_idx])
            else:
                current_pt_file.current_vehicle_current_edges = torch.tensor([0])
        
        # Add position on edge (0.0 for new vehicle)
        if hasattr(current_pt_file, 'current_vehicle_position_on_edges'):
            current_pt_file.current_vehicle_position_on_edges = torch.cat([
                current_pt_file.current_vehicle_position_on_edges, 
                torch.tensor([0.0])
            ])
        else:
            print(f"Warning: current_vehicle_position_on_edges not found in current pt file")
            current_pt_file.current_vehicle_position_on_edges = torch.zeros(len(current_pt_file.vehicle_ids))
        
        # Add route information
        if hasattr(current_pt_file, 'vehicle_route_left'):
            # Add the new vehicle's route
            route_tensor = torch.tensor([current_pt_file.edge_ids.index(edge) if edge in current_pt_file.edge_ids else 0 for edge in route_edges])
            current_pt_file.vehicle_route_left = torch.cat([current_pt_file.vehicle_route_left, route_tensor])
            
            # Update route splits
            if hasattr(current_pt_file, 'vehicle_route_left_splits'):
                current_pt_file.vehicle_route_left_splits = torch.cat([
                    current_pt_file.vehicle_route_left_splits,
                    torch.tensor([len(route_edges)])
                ])
            else:
                print(f"Warning: vehicle_route_left_splits not found in current pt file")
                current_pt_file.vehicle_route_left_splits = torch.tensor([len(route_edges)])
        else:
            print(f"Warning: vehicle_route_left not found in current pt file")
            
        
        # Construct dynamic edges following dataset_creator.py pattern
        try:
            print(f"Constructing dynamic edges for new vehicle {new_vehicle_idx}")
            # Check if dynamic edges already exist in current_pt_file
            if hasattr(current_pt_file, 'edge_type'):
                edge_type_tensor = current_pt_file.edge_type
                j_to_v = (edge_type_tensor == 1).sum().item()
                v_to_j = (edge_type_tensor == 2).sum().item()
                v_to_v = (edge_type_tensor == 3).sum().item()
                print(f"   - Existing Junction→Vehicle: {j_to_v}")
                print(f"   - Existing Vehicle→Junction: {v_to_j}")
                print(f"   - Existing Vehicle→Vehicle: {v_to_v}")
            else:
                print(f"   - No existing dynamic edges in current_pt_file")
            
            dynamic_edge_index, dynamic_edge_type, dynamic_edge_attr = self._construct_dynamic_edges(current_pt_file, new_vehicle_idx)
            print(f"✅ Constructed {len(dynamic_edge_index[0])} dynamic edges")
            print(f"   - Junction→Vehicle: {dynamic_edge_type.count(1)}")
            print(f"   - Vehicle→Junction: {dynamic_edge_type.count(2)}")
            print(f"   - Vehicle→Vehicle: {dynamic_edge_type.count(3)}")
            
            # Update current_pt_file with dynamic edges
            if len(dynamic_edge_index[0]) > 0:
                # Convert to tensors
                dynamic_edge_index_tensor = torch.tensor(dynamic_edge_index, dtype=torch.long)
                dynamic_edge_type_tensor = torch.tensor(dynamic_edge_type, dtype=torch.long)
                dynamic_edge_attr_tensor = torch.tensor(dynamic_edge_attr, dtype=torch.float32)
                
                # Append dynamic edges to existing edges (don't override!)
                if hasattr(current_pt_file, 'edge_index'):
                    # Print BEFORE state
                    print(f"📊 BEFORE appending dynamic edges:")
                    print(f"   - Existing edge_index shape: {current_pt_file.edge_index.shape}")
                    print(f"   - Existing edge_type shape: {current_pt_file.edge_type.shape}")
                    print(f"   - Existing edge_attr shape: {current_pt_file.edge_attr.shape}")
                    
                    # Concatenate with existing edges
                    current_pt_file.edge_index = torch.cat([current_pt_file.edge_index, dynamic_edge_index_tensor], dim=1)
                    current_pt_file.edge_type = torch.cat([current_pt_file.edge_type, dynamic_edge_type_tensor], dim=0)
                    current_pt_file.edge_attr = torch.cat([current_pt_file.edge_attr, dynamic_edge_attr_tensor], dim=0)
                    
                    print(f"📝 AFTER appending {len(dynamic_edge_index[0])} dynamic edges:")
                    print(f"   - Total edge_index shape: {current_pt_file.edge_index.shape}")
                    print(f"   - Total edge_type shape: {current_pt_file.edge_type.shape}")
                    print(f"   - Total edge_attr shape: {current_pt_file.edge_attr.shape}")
                else:
                    print("⚠️  No existing edges to append to in current_pt_file")
            else:
                print("⚠️  No dynamic edges to add to current_pt_file")
                
        except Exception as e:
            print(f"⚠️  Dynamic edge construction failed: {e}")
            dynamic_edge_index, dynamic_edge_type, dynamic_edge_attr = [[], []], [], []
        
        # Build 28-feature vector
        feature_vector = []
        '''
        x=x_tensor, needs to be built from scratch
        edge_index=edge_index_tensor, not modified
        edge_type=edge_type_tensor, not modified
        edge_attr=full_edge_attr_tensor, needs to take another vehicle on route
        vehicle_ids=current_vehicle_ids,
        junction_ids=list(static_junction_ids_to_index.keys()), not modified
        edge_ids=list(static_edge_ids_to_index.keys()), not modified

        vehicle_route_left=vehicle_routes_flat_tensor, needs to be added with the new vehicle
        vehicle_route_left_splits=vehicle_route_splits_tensor, needs to be added with the new vehicle
        current_vehicle_current_edges=current_vehicle_current_edges_tensor, needs to be added with the new vehicle
        current_vehicle_position_on_edges=current_vehicle_position_on_edges_tensor, needs to be added with the new vehicle position 0.0

        x_base_dim=torch.tensor(BASE_FEATURES_COUNT, dtype=torch.long),   # 26
        route_feat_idx=torch.tensor([25, 27], dtype=torch.long),          # [start,end) = 25..27

        # targets are not relevant for inference
        y=y_dict["raw"],
        y_minmax=y_dict["minmax"],
        y_z=y_dict["z"],
        y_log=y_dict["log"],
        y_log_z=y_dict["log_z"],

        # categoricals/binary (from RAW seconds)
        y_equal_thirds=y_cat_tensors['equal_thirds'],
        y_quartile=y_cat_tensors['quartile'],
        y_mean_pm_0_5_std=y_cat_tensors['mean_pm_0_5_std'],
        y_median_pm_0_5_iqr=y_cat_tensors['median_pm_0_5_iqr'],
        y_binary_eta=y_binary_tensor,

        # normalization metadata (needed to invert during eval)
        eta_p98=torch.tensor(float(self.entities_data['label']['stats']['eta']['98%'])),
        eta_mean=torch.tensor(float(self.entities_data['label']['stats']['eta']['mean'])),
        eta_std=torch.tensor(max(1e-8, float(self.entities_data['label']['stats']['eta']['std']))),
        eta_log_mean=torch.tensor(float(self.entities_data['label']['stats']['eta']['log_mean'])),
        eta_log_std=torch.tensor(max(1e-8, float(self.entities_data['label']['stats']['eta']['log_std']))),
        
        x - Nodes (Junctions, Vehicle) main feature layout:

        | Index | Feature Name        | Notes                                                   |
        | ----- | ------------------- | ------------------------------------------------------- |
        | 0     | `node_type`         | 0 = junction    1 = vehicle                             |
        | 1-3   | `veh_type_oh`       | ['bus', 'passenger', 'truck']`[0, 0, 0]` for junctions  |
        | 4     | `speed`             | min-max normalized if normalize, else raw               |
        | 5     | `acceleration`      | min-max normalized if normalize, else raw               |
        | 6     | `sin_hour`          | represent time in a unit circle                         |  
        | 7     | `cos_hour`          | represent time in a unit circle                         |  
        | 8     | `sin_day`           | represent day in a unit circle                          |  
        | 9     | `cos_day`           | represent day in a unit circle                          |  
        | 10    | `route_length`      | min-max normalized if normalize, else raw               |
        | 11    | `progress`          | trip progress: 1 - (route_length_left / route_length)   |
        | 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
        | 16    | `current_x`         | min-max normalized if normalize, else raw               |
        | 17    | `current_y`         | min-max normalized if normalize, else raw               |
        | 18    | `destination_x`     | Normalized or raw; for vehicles only                     |
        | 19    | `destination_y`     | Normalized or raw; for vehicles only                     |
        | 20-22 | `current_edge_num_lanes_oh` | One-hot: [1,2,3] lanes; [0,0,0] for junctions         |
        | 23    | `current_edge_demand`     | Demand value for the current edge (from updated edge features) |
        | 24    | `current_edge_occupancy`  | Occupancy value for the current edge (from updated edge features) |
        | 25    | `route_left_demand_len_disc`        | Demand value for the route left (from updated edge features) |
        | 26    | `route_left_occupancy_len_disc`     | Occupancy value for the route left (from updated edge features) |
        | 27    | `j_type`            | Junction type (priority/traffic_light); 0 for vehicles  |

        '''
        
        # 1. node_type (hardcoded to 1 for vehicles)
        feature_vector.append(1)  # [0]
        
        # 2. vehicle_type one-hot (hardcoded to passenger)
        feature_vector.extend([0, 1, 0])  # [1-3] passenger = [0,1,0]
        
        # 3. speed and acceleration (hardcoded to 0 for starting vehicle)
        feature_vector.append(0.0)  # [4] speed
        feature_vector.append(0.0)  # [5] acceleration
        
        # 4. Temporal features calculated from start_step
        sin_hour, cos_hour, sin_day, cos_day = self._calculate_temporal_features(start_step)
        feature_vector.append(sin_hour)  # [6]
        feature_vector.append(cos_hour)  # [7]
        feature_vector.append(sin_day)   # [8]
        feature_vector.append(cos_day)   # [9]
        
        # 5. Route information
        route_length_norm = self._normalize_route_length(route_length)
        feature_vector.append(route_length_norm)  # [10]
        feature_vector.append(0.0)  # [11] progress (hardcoded to 0 for starting vehicle)
        
        # 6. Zone one-hot encoding
        zone_oh = self._encode_zone(zone)
        feature_vector.extend(zone_oh)  # [12-15]
        
        # 7. Current position (normalized)
        current_x_norm = self._normalize_coordinate(current_x, 'current_x')
        current_y_norm = self._normalize_coordinate(current_y, 'current_y')
        feature_vector.append(current_x_norm)  # [16]
        feature_vector.append(current_y_norm)  # [17]
        
        # 8. Destination position (normalized)
        dest_x_norm = self._normalize_coordinate(destination_x, 'destination_x')
        dest_y_norm = self._normalize_coordinate(destination_y, 'destination_y')
        feature_vector.append(dest_x_norm)  # [18]
        feature_vector.append(dest_y_norm)  # [19]
        
        # 9. Current edge number of lanes one-hot
        lanes_oh = self._encode_num_lanes(current_edge_num_lanes)
        feature_vector.extend(lanes_oh)  # [20-22]
        
        # 10. Current edge demand and occupancy (copied from current pt file)
        feature_vector.append(current_edge_demand)  # [23]
        feature_vector.append(current_edge_occupancy)  # [24]
        
        # 11. Route-aware features (hardcoded to 0 for starting vehicle)
        feature_vector.append(0.0)  # [25] route_left_demand_len_disc
        feature_vector.append(0.0)  # [26] route_left_occupancy_len_disc
        
        # 12. Junction type (hardcoded to 0 for vehicles)
        feature_vector.append(0)  # [27]
        
        # Add the new vehicle's feature vector to the pt file
        new_vehicle_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        current_pt_file.x = torch.cat([current_pt_file.x, new_vehicle_tensor.unsqueeze(0)], dim=0)
        
        return current_pt_file
    
    def _load_temporal_window(self, current_step: int) -> List[Data]:
        """
        Load temporal window of pt files with cyclic wrapping.
        
        Args:
            current_step: Current simulation step
            
        Returns:
            List of Data objects for temporal window
        """
        import glob
        import torch
        
        # Get all pt files in data directory
        pt_files = sorted(glob.glob(os.path.join(self.data_path, "step_*.pt")), key=extract_step_number)
        
        if not pt_files:
            raise FileNotFoundError(f"No pt files found in {self.data_path}")
        
        # Calculate file indices with cyclic wrapping
        total_files = len(pt_files)
        
        # Find the file that matches or is closest to current_step
        step_numbers = [extract_step_number(f) for f in pt_files]
        current_file_idx = 0
        min_diff = float('inf')
        
        current_step = current_step % (24 * 60 * 60)

        for i, step_num in enumerate(step_numbers):
            diff = abs(step_num - current_step)
            if diff < min_diff:
                min_diff = diff
                current_file_idx = i
        print(f"Current file idx: {current_file_idx} for step {current_step} and file {pt_files[current_file_idx]}")
        
        # Create window indices with cyclic wrapping
        window_indices = []
        for i in range(self.window_size):
            idx = (current_file_idx - (self.window_size - 1 - i)) % total_files
            window_indices.append(idx)
        
        # Load pt files
        temporal_window = []
        for idx in window_indices:
            if idx < len(pt_files):
                data = torch.load(pt_files[idx], map_location='cpu', weights_only=False)
                temporal_window.append(data)
            else:
                # This shouldn't happen with proper cyclic wrapping, but just in case
                raise IndexError(f"File index {idx} out of range for {len(pt_files)} files")
        print(f"Last pt file loaded: {pt_files[window_indices[-1]]}")
        return temporal_window
    
    def _get_current_edge_features(self, current_pt_file: Data, current_edge_id: str) -> Tuple[float, float, float]:
        """
        Extract current edge demand and occupancy from current pt file.
        
        Args:
            current_pt_file: Current Data object
            current_edge_id: Current edge ID
            
        Returns:
            Tuple of (demand, occupancy, speed)
        """
        # Find edge index for current_edge_id
        if hasattr(current_pt_file, 'edge_ids'):
            try:
                edge_idx = current_pt_file.edge_ids.index(current_edge_id)
                demand = current_pt_file.edge_attr[edge_idx][5].item()  # edge_demand
                occupancy = current_pt_file.edge_attr[edge_idx][6].item()  # edge_occupancy
                speed = current_pt_file.edge_attr[edge_idx][0].item()  # edge_speed
                return demand, occupancy, speed
            except ValueError:
                # If edge not found, return zeros (for testing purposes)
                print(f"⚠️  Edge {current_edge_id} not found in current pt file, using zeros")
                return 0.0, 0.0, 0.0
        else:
            # Fallback: return zeros if edge_ids not available
            return 0.0, 0.0, 0.0
    
    def _calculate_temporal_features(self, timestamp_seconds: int) -> Tuple[float, float, float, float]:
        """
        Calculate temporal features from start_step.
        
        Args:
            start_step: Simulation step (assuming 30-second intervals)
            
        Returns:
            Tuple of (sin_hour, cos_hour, sin_day, cos_day)
        """
        # Convert to minutes, hours, days
        minutes = timestamp_seconds // 60
        hours = minutes // 60
        days = hours // 24
        
        # Get day of week (0-6) and hour of day (0-23)
        day = days % 7
        hour = hours % 24
        minutes_in_hour = minutes % 60
        
        # Convert to fractional hour
        hour_frac = (hour + minutes_in_hour / 60) % 24
        
        # Calculate temporal features
        sin_hour = math.sin(2 * math.pi * hour_frac / 24)
        cos_hour = math.cos(2 * math.pi * hour_frac / 24)
        sin_day = math.sin(2 * math.pi * day / 7)
        cos_day = math.cos(2 * math.pi * day / 7)
        
        return sin_hour, cos_hour, sin_day, cos_day
    
    def _convert_step_to_time(self, step: int) -> tuple:
        """Convert step number back to normal time (hours, minutes, seconds)."""
        # Assuming 30-second intervals
        total_seconds = step * 30
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return hours, minutes, seconds
    
    def _convert_step_to_datetime(self, step: int, start_date: str = "2024-01-01") -> str:
        """Convert step number to datetime string."""
        hours, minutes, seconds = self._convert_step_to_time(step)
        
        # Create datetime string
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        current_dt = start_dt + timedelta(hours=hours, minutes=minutes, seconds=seconds)
        
        return current_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _convert_time_to_step(self, hours: int, minutes: int, seconds: int) -> int:
        """Convert normal time back to step number."""
        total_seconds = hours * 3600 + minutes * 60 + seconds
        step = total_seconds // 30  # Assuming 30-second intervals
        return step
    
    def _convert_seconds_to_step(self, seconds: int) -> int:
        """Convert seconds to step number."""
        return seconds // 30  # Assuming 30-second intervals
    
    def _convert_step_to_seconds(self, step: int) -> int:
        """Convert step number to seconds."""
        return step * 30  # Assuming 30-second intervals
    
    def _format_time_info(self, current_time: int) -> str:
        """Format current time information for display."""
        step = self._convert_seconds_to_step(current_time)
        hours, minutes, seconds = self._convert_step_to_time(step)
        datetime_str = self._convert_step_to_datetime(step)
        
        return f"Time: {current_time}s → Step: {step} → {hours:02d}h {minutes:02d}m {seconds:02d}s → {datetime_str}"
    
    def _convert_temporal_features_to_time(self, sin_hour: float, cos_hour: float, sin_day: float, cos_day: float) -> str:
        """Convert temporal features back to time in HH:MM:SS format."""
        import math
        
        # Convert sin/cos back to hour
        hour_frac = math.atan2(sin_hour, cos_hour) * 24 / (2 * math.pi)
        if hour_frac < 0:
            hour_frac += 24
        
        # Convert sin/cos back to day
        day = math.atan2(sin_day, cos_day) * 7 / (2 * math.pi)
        if day < 0:
            day += 7
        
        # Extract hour, minute, second components
        hour = int(hour_frac)
        minute_frac = (hour_frac - hour) * 60
        minute = int(minute_frac)
        second_frac = (minute_frac - minute) * 60
        second = int(round(second_frac))
        
        # Handle overflow
        if second >= 60:
            second = 0
            minute += 1
        if minute >= 60:
            minute = 0
            hour += 1
        if hour >= 24:
            hour = 0
        
        # Calculate total seconds
        total_seconds = hour * 3600 + minute * 60 + second
        
        return f"{hour:02d}:{minute:02d}:{second:02d} (Total: {total_seconds}s, Day: {day:.1f})"
    
    def _construct_dynamic_edges(self, current_pt_file, new_vehicle_idx):
        """
        Add dynamic edges for a single new vehicle on its current edge.
        
        This is much simpler than reconstructing the entire dynamic graph.
        We only need to handle the new vehicle's current edge.
        
        Args:
            current_pt_file: The current pt file data
            new_vehicle_idx: Index of the new vehicle in the vehicle list
            
        Returns:
            dynamic_edge_index: [2, N] list of source-target node indices
            dynamic_edge_type: [N] list of edge types (1=J→V, 2=V→J, 3=V→V)
            dynamic_edge_attr: [N, F] edge feature vectors
        """
        try:
            # Get the new vehicle's current edge
            current_vehicle_current_edges = getattr(current_pt_file, 'current_vehicle_current_edges', torch.zeros(len(current_pt_file.vehicle_ids), dtype=torch.long))
            new_vehicle_edge_idx = current_vehicle_current_edges[new_vehicle_idx].item()
            
            # Get edge ID from index
            edge_ids = current_pt_file.edge_ids
            junction_ids = current_pt_file.junction_ids
            
            if new_vehicle_edge_idx >= len(edge_ids):
                print(f"Warning: New vehicle edge index {new_vehicle_edge_idx} out of range")
                return [[], []], [], []
                
            new_vehicle_edge_id = edge_ids[new_vehicle_edge_idx]
            print(f"🔗 Adding dynamic edges for new vehicle on edge: {new_vehicle_edge_id}")
            
            # Parse edge ID to get from/to junctions
            import re
            matches = re.findall(r'[A-Z]+\d+', new_vehicle_edge_id)
            if len(matches) >= 2:
                # Format: AA0AB0, AA1AA0, etc. (from_junction_to_junction)
                from_junction = matches[0]
                to_junction = matches[1]
            elif len(matches) == 1:
                # Format: E6 (single junction reference)
                junction = matches[0]
                from_junction = junction
                to_junction = junction
            else:
                print(f"Warning: Cannot parse edge ID {new_vehicle_edge_id}")
                return [[], []], [], []
            
            # Get junction indices
            junction_id_to_index = {jid: i for i, jid in enumerate(junction_ids)}
            from_junction_idx = junction_id_to_index.get(from_junction)
            to_junction_idx = junction_id_to_index.get(to_junction)
            
            if from_junction_idx is None or to_junction_idx is None:
                print(f"Warning: Junction not found - from: {from_junction}, to: {to_junction}")
                return [[], []], [], []
            
            # Get global vehicle index (offset by number of junctions)
            new_vehicle_global_idx = len(junction_ids) + new_vehicle_idx
            
            # Initialize dynamic edges
            dynamic_edge_index = [[], []]
            dynamic_edge_type = []
            dynamic_edge_attr = []
            
            # Get edge feature dimension
            edge_feature_dim = current_pt_file.edge_attr.shape[1] if hasattr(current_pt_file, 'edge_attr') else 7
            
            # Check if there are any existing vehicles on this edge
            existing_vehicles_on_edge = []
            for i, edge_idx in enumerate(current_vehicle_current_edges):
                if i != new_vehicle_idx and edge_idx.item() == new_vehicle_edge_idx:
                    existing_vehicles_on_edge.append(i)
            
            if not existing_vehicles_on_edge:
                # Case 1: No existing vehicles on this edge
                # Create 2 edges: J→V and V→J
                print(f"  📍 No existing vehicles on edge {new_vehicle_edge_id}")
                print(f"  ➕ Creating J→V edge: {from_junction} → new_vehicle")
                print(f"  ➕ Creating V→J edge: new_vehicle → {to_junction}")
                
                # J→V edge
                dynamic_edge_index[0].append(from_junction_idx)
                dynamic_edge_index[1].append(new_vehicle_global_idx)
                dynamic_edge_type.append(1)  # JUNCTION → VEHICLE
                dynamic_edge_attr.append([0.0] * edge_feature_dim)
                
                # V→J edge
                dynamic_edge_index[0].append(new_vehicle_global_idx)
                dynamic_edge_index[1].append(to_junction_idx)
                dynamic_edge_type.append(2)  # VEHICLE → JUNCTION
                dynamic_edge_attr.append([0.0] * edge_feature_dim)
                
            else:
                # Case 2: Existing vehicles on this edge
                # Find the first vehicle (closest to start junction)
                current_vehicle_position_on_edges = getattr(current_pt_file, 'current_vehicle_position_on_edges', torch.zeros(len(current_pt_file.vehicle_ids)))
                
                # Sort existing vehicles by position
                existing_vehicles_sorted = sorted(
                    existing_vehicles_on_edge,
                    key=lambda i: current_vehicle_position_on_edges[i].item()
                )
                
                first_vehicle_idx = existing_vehicles_sorted[0]
                first_vehicle_global_idx = len(junction_ids) + first_vehicle_idx
                
                print(f"  📍 Found {len(existing_vehicles_on_edge)} existing vehicles on edge {new_vehicle_edge_id}")
                print(f"  🔄 Disconnecting first vehicle from start junction")
                print(f"  ➕ Creating J→V edge: {from_junction} → new_vehicle")
                print(f"  ➕ Creating V→V edge: new_vehicle → first_vehicle")
                
                # Find and remove the existing J→V edge from start junction to first vehicle
                existing_edge_index = current_pt_file.edge_index
                existing_edge_type = current_pt_file.edge_type
                existing_edge_attr = current_pt_file.edge_attr
                
                edges_to_remove = []
                for i in range(existing_edge_index.shape[1]):
                    if (existing_edge_index[0, i] == from_junction_idx and 
                        existing_edge_index[1, i] == first_vehicle_global_idx and
                        existing_edge_type[i] == 1):  # JUNCTION → VEHICLE
                        edges_to_remove.append(i)
                
                print(f"  🗑️  Removing {len(edges_to_remove)} existing J→V edges")
                
                # Remove the existing J→V edge from current_pt_file
                if edges_to_remove:
                    keep_mask = torch.ones(existing_edge_index.shape[1], dtype=torch.bool)
                    keep_mask[edges_to_remove] = False
                    
                    # Update current_pt_file by removing the old edge
                    current_pt_file.edge_index = existing_edge_index[:, keep_mask]
                    current_pt_file.edge_type = existing_edge_type[keep_mask]
                    current_pt_file.edge_attr = existing_edge_attr[keep_mask]
                
                # Add new edges
                # J→V edge (start junction to new vehicle)
                dynamic_edge_index[0].append(from_junction_idx)
                dynamic_edge_index[1].append(new_vehicle_global_idx)
                dynamic_edge_type.append(1)  # JUNCTION → VEHICLE
                dynamic_edge_attr.append([0.0] * edge_feature_dim)
                
                # V→V edge (new vehicle to first existing vehicle)
                dynamic_edge_index[0].append(new_vehicle_global_idx)
                dynamic_edge_index[1].append(first_vehicle_global_idx)
                dynamic_edge_type.append(3)  # VEHICLE → VEHICLE
                dynamic_edge_attr.append([0.0] * edge_feature_dim)
            
            print(f"  ✅ Created {len(dynamic_edge_index[0])} dynamic edges for new vehicle")
            return dynamic_edge_index, dynamic_edge_type, dynamic_edge_attr
            
        except Exception as e:
            print(f"Dynamic edge construction error: {e}")
            import traceback
            traceback.print_exc()
            return [[], []], [], []
    
    def _normalize_route_length(self, route_length: float) -> float:
        """Normalize route length using min-max normalization."""
        if 'vehicle' in self.entities_data and 'route_length' in self.entities_data['vehicle']['stats']:
            stats = self.entities_data['vehicle']['stats']['route_length']
            min_val = stats['min']
            max_val = stats['max']
        else:
            # Fallback values
            min_val = 476.6
            max_val = 23133.41
        return (route_length - min_val) / max(1e-8, (max_val - min_val))
    
    def _encode_zone(self, zone: str) -> List[float]:
        """Encode zone as one-hot vector."""
        zone_mapping = {'A': 0, 'B': 1, 'C': 2, 'H': 3}
        zone_oh = [0.0] * 4
        if zone in zone_mapping:
            zone_oh[zone_mapping[zone]] = 1.0
        return zone_oh
    
    def _normalize_coordinate(self, coord: float, coord_type: str) -> float:
        """Normalize coordinate using min-max normalization."""
        if 'vehicle' in self.entities_data and coord_type in self.entities_data['vehicle']['stats']:
            stats = self.entities_data['vehicle']['stats'][coord_type]
            min_val = stats['min']
            max_val = stats['max']
        else:
            # Fallback values
            if coord_type == 'current_x':
                min_val = -4.8
                max_val = 18004.8
            elif coord_type == 'current_y':
                min_val = -6269.76
                max_val = 5004.8
            elif coord_type == 'destination_x':
                min_val = 11.416129830593093
                max_val = 17988.49958178945
            elif coord_type == 'destination_y':
                min_val = -6253.4779247844235
                max_val = 4988.503014571509
            else:
                return coord  # Return raw value if type not recognized
        
        # Min-max normalization: (x - min) / (max - min)
        return (coord - min_val) / max(1e-8, (max_val - min_val))
    
    def _encode_num_lanes(self, num_lanes: int) -> List[float]:
        """Encode number of lanes as one-hot vector."""
        lanes_oh = [0.0] * 3
        if 1 <= num_lanes <= 3:
            lanes_oh[num_lanes - 1] = 1.0
        return lanes_oh
    
    def _denormalize_edge_demand(self, normalized_demand: float) -> int:
        """Convert log-normalized edge demand back to raw count."""
        if 'edge' in self.entities_data and 'edge_route_count_log' in self.entities_data['edge']['stats']:
            stats = self.entities_data['edge']['stats']['edge_route_count_log']
            log_mean = stats['mean']
            log_std = stats['std']
        else:
            # Fallback values
            log_mean = 0.522151
            log_std = 0.836544
        raw_count = math.expm1(normalized_demand * log_std + log_mean)
        return max(0, int(round(raw_count)))  # Ensure non-negative integer
    
    def _denormalize_edge_occupancy(self, normalized_occupancy: float) -> int:
        """Convert log-normalized edge occupancy back to raw count."""
        if 'edge' in self.entities_data and 'vehicles_on_road_count_log' in self.entities_data['edge']['stats']:
            stats = self.entities_data['edge']['stats']['vehicles_on_road_count_log']
            log_mean = stats['mean']
            log_std = stats['std']
        else:
            # Fallback values
            log_mean = 0.093278
            log_std = 0.325330
        raw_count = math.expm1(normalized_occupancy * log_std + log_mean)
        return max(0, int(round(raw_count)))  # Ensure non-negative integer
    
    def _denormalize_edge_speed(self, normalized_speed: float) -> float:
        """Convert normalized edge speed back to raw speed."""
        # Statistics from edge_feature_summary.csv
        if 'edge' in self.entities_data and 'avg_speed' in self.entities_data['edge']['stats']:
            stats = self.entities_data['edge']['stats']['avg_speed']
            min_val = stats['min']
            max_val = stats['max']
            # Reverse min-max normalization: raw = normalized * (max - min) + min
            return normalized_speed * (max_val - min_val) + min_val
        else:
            # Fallback values
            min_val = 0.0
            max_val = 33.33
            return normalized_speed * (max_val - min_val) + min_val
    
    def _normalize_edge_demand(self, raw_demand: float) -> float:
        """Convert raw edge demand count to log-normalized value."""
        if 'edge' in self.entities_data and 'edge_route_count_log' in self.entities_data['edge']['stats']:
            stats = self.entities_data['edge']['stats']['edge_route_count_log']
            log_mean = stats['mean']
            log_std = stats['std']
        else:
            # Fallback values
            log_mean = 0.522151
            log_std = 0.836544
        return (math.log1p(raw_demand) - log_mean) / log_std
    
    def _normalize_edge_occupancy(self, raw_occupancy: float) -> float:
        """Convert raw edge occupancy count to log-normalized value."""
        if 'edge' in self.entities_data and 'vehicles_on_road_count_log' in self.entities_data['edge']['stats']:
            stats = self.entities_data['edge']['stats']['vehicles_on_road_count_log']
            log_mean = stats['mean']
            log_std = stats['std']
        else:
            # Fallback values
            log_mean = 0.093278
            log_std = 0.325330
        return (math.log1p(raw_occupancy) - log_mean) / log_std
    
    def _normalize_edge_speed(self, raw_speed: float) -> float:
        """Convert raw edge speed to normalized value."""
        # Statistics from edge_feature_summary.csv
        if 'edge' in self.entities_data and 'avg_speed' in self.entities_data['edge']['stats']:
            stats = self.entities_data['edge']['stats']['avg_speed']
            min_val = stats['min']
            max_val = stats['max']
            # Min-max normalization: (raw - min) / (max - min)
            return (raw_speed - min_val) / max(1e-8, (max_val - min_val))
        else:
            # Fallback values
            min_val = 0.0
            max_val = 33.33
            return (raw_speed - min_val) / max(1e-8, (max_val - min_val))
    
    def predict_eta(self, vehicle_info: Dict, route_info: Dict, current_time: int) -> float:
        """
        Predict ETA for a new vehicle.
        
        Args:
            vehicle_info: Dictionary containing vehicle properties
            route_info: Dictionary containing route information  
            current_time: Current simulation time in seconds
            
        Returns:
            Predicted ETA in seconds
        """
        # Set deterministic seed for consistent results
        self.set_seed(self.seed)
        #print param 
        print(f"Vehicle info: {vehicle_info}")
        print(f"Route info: {route_info}")
        print(f"Current time: {current_time}")
        
        with torch.no_grad():
            # current_time is already in seconds, no conversion needed
            # The step number in file names corresponds to the second when captured
            
            # Load temporal window 
            temporal_window = self._load_temporal_window(current_time)
            current_pt_file = temporal_window[-1]
            
            # Add new vehicle to the last snapshot and get updated pt file
            updated_pt_file = self.add_vehicle_to_last_snapshot(
                current_pt_file=current_pt_file,
                veh_id=vehicle_info.get("veh_id", f"new_vehicle_{current_time}"),
                start_step=current_time,
                route_edges=route_info.get("route_edges", []),
                route_length=vehicle_info.get("route_length", 5000.0),
                zone=vehicle_info.get("zone", "A"),
                current_x=vehicle_info.get("current_x", 1000.0),
                current_y=vehicle_info.get("current_y", 2000.0),
                destination_x=vehicle_info.get("destination_x", 5000.0),
                destination_y=vehicle_info.get("destination_y", 6000.0),
                current_edge_num_lanes=vehicle_info.get("current_edge_num_lanes", 2),
                current_edge_id=vehicle_info.get("current_edge_id", "edge_123")
            )
            
            # Move all timesteps to the correct device
            temporal_window = [timestep.to(self.device) for timestep in temporal_window]
            
            # Replace the last timestep with our updated pt file
            temporal_window[-1] = updated_pt_file.to(self.device)
            
            # Create temporal batch for model input (following evaluate_moe.py pattern)
            time_batches = temporal_window
            
            # Run model inference (following evaluate_moe.py pattern)
            y_hat, aux, veh_mask = self.model(time_batches, train=False)
            
            # Get the last timestep batch (following evaluate_moe.py pattern)
            bt = time_batches[-1]
            
            # Get target key from config
            target_key = self.config.get("train", {}).get("target_key", "y_log_z")
            
            # For real-time inference, we need to create a proper batch structure
            # Since we're only processing one graph, we need to create batch indices
            if bt.batch is None:
                # Create batch indices for all nodes (junctions + vehicles)
                num_nodes = bt.x.size(0)
                bt.batch = torch.zeros(num_nodes, dtype=torch.long, device=bt.x.device)
            
            # Get batch vehicle information (following evaluate_moe.py pattern)
            batch_veh = bt.batch[veh_mask]  # [Nv] - batch indices for vehicles
            
            # Convert predictions to seconds using the same method as evaluation
            from utils_targets import invert_to_seconds
            yhat_sec = invert_to_seconds(y_hat, bt, target_key, batch_veh)  # [Nv]
            
            # The new vehicle is the last vehicle added, so it's the last prediction
            new_vehicle_eta_seconds = yhat_sec[-1].item()
            
            return new_vehicle_eta_seconds
    
    def _add_vehicle_to_timestep(self, timestep_data: Data, vehicle_tensor: torch.Tensor, 
                                vehicle_info: Dict, route_info: Dict) -> Data:
        """
        Add new vehicle to a timestep data.
        
        Args:
            timestep_data: Original timestep data
            vehicle_tensor: New vehicle feature tensor
            vehicle_info: Vehicle information
            route_info: Route information
            
        Returns:
            Modified timestep data with new vehicle added
        """
        # Create a copy of the timestep data
        new_data = timestep_data.clone()
        
        # Add new vehicle to node features
        new_vehicle_features = vehicle_tensor.unsqueeze(0).to(new_data.x.device)  # Add batch dimension and ensure same device
        new_data.x = torch.cat([new_data.x, new_vehicle_features], dim=0)
        
        # Update vehicle count
        if hasattr(new_data, 'num_vehicles'):
            new_data.num_vehicles += 1
        
        # Handle route information for route-aware models
        # The model expects 'vehicle_route_left' and 'vehicle_route_left_splits' attributes
        if hasattr(new_data, 'vehicle_route_left') and hasattr(new_data, 'vehicle_route_left_splits'):
            # Convert route to edge indices
            route_edges = route_info.get("route_edges", [])
            route_indices = []
            
            # Map route edges to edge indices
            if hasattr(new_data, 'edge_ids'):
                for edge_id in route_edges:
                    if edge_id in new_data.edge_ids:
                        edge_idx = new_data.edge_ids.index(edge_id)
                        route_indices.append(edge_idx)
                    else:
                        # If edge not found, use a dummy edge (index 0)
                        route_indices.append(0)
            else:
                # If no edge_ids available, create dummy route
                route_indices = [0] * len(route_edges) if route_edges else [0]
            
            # Ensure we have at least one edge in the route
            if not route_indices:
                route_indices = [0]
            
            # Add route to existing routes
            new_route_tensor = torch.tensor(route_indices, dtype=torch.long, device=new_data.vehicle_route_left.device)
            new_data.vehicle_route_left = torch.cat([new_data.vehicle_route_left, new_route_tensor])
            
            # Add route split for the new vehicle
            new_split_tensor = torch.tensor([len(route_indices)], dtype=torch.long, device=new_data.vehicle_route_left_splits.device)
            new_data.vehicle_route_left_splits = torch.cat([new_data.vehicle_route_left_splits, new_split_tensor])
            
            # Verify route alignment: number of vehicles should match number of route splits
            vehicle_count = (new_data.x[:, 0] == 1).sum().item()  # Count vehicles (node_type == 1)
            route_splits_count = new_data.vehicle_route_left_splits.shape[0]
            
            if vehicle_count != route_splits_count:
                print(f"Warning: Vehicle count ({vehicle_count}) != Route splits count ({route_splits_count})")
                # This should not happen with proper implementation, but let's handle it gracefully
                if vehicle_count > route_splits_count:
                    # Add dummy routes for missing vehicles
                    missing_routes = vehicle_count - route_splits_count
                    dummy_routes = torch.tensor([0], dtype=torch.long, device=new_data.vehicle_route_left.device)
                    dummy_splits = torch.tensor([1], dtype=torch.long, device=new_data.vehicle_route_left_splits.device)
                    
                    for _ in range(missing_routes):
                        new_data.vehicle_route_left = torch.cat([new_data.vehicle_route_left, dummy_routes])
                        new_data.vehicle_route_left_splits = torch.cat([new_data.vehicle_route_left_splits, dummy_splits])
        else:
            # If route attributes don't exist, create them
            route_edges = route_info.get("route_edges", [])
            route_indices = []
            
            # Map route edges to edge indices
            if hasattr(new_data, 'edge_ids'):
                for edge_id in route_edges:
                    if edge_id in new_data.edge_ids:
                        edge_idx = new_data.edge_ids.index(edge_id)
                        route_indices.append(edge_idx)
                    else:
                        # If edge not found, use a dummy edge (index 0)
                        route_indices.append(0)
            else:
                # If no edge_ids available, create dummy route
                route_indices = [0] * len(route_edges) if route_edges else [0]
            
            # Ensure we have at least one edge in the route
            if not route_indices:
                route_indices = [0]
            
            # Create route attributes for all vehicles (existing + new)
            vehicle_count = (new_data.x[:, 0] == 1).sum().item()  # Count vehicles (node_type == 1)
            
            # Create dummy routes for existing vehicles
            existing_vehicle_count = vehicle_count - 1  # Subtract 1 for the new vehicle we just added
            dummy_routes = torch.tensor([0], dtype=torch.long, device=new_data.x.device)
            dummy_splits = torch.tensor([1], dtype=torch.long, device=new_data.x.device)
            
            # Create route data for all vehicles
            all_routes = []
            all_splits = []
            
            # Add dummy routes for existing vehicles
            for _ in range(existing_vehicle_count):
                all_routes.append(dummy_routes)
                all_splits.append(dummy_splits)
            
            # Add route for the new vehicle
            new_route_tensor = torch.tensor(route_indices, dtype=torch.long, device=new_data.x.device)
            all_routes.append(new_route_tensor)
            all_splits.append(torch.tensor([len(route_indices)], dtype=torch.long, device=new_data.x.device))
            
            # Concatenate all routes
            new_data.vehicle_route_left = torch.cat(all_routes)
            new_data.vehicle_route_left_splits = torch.cat(all_splits)
        
        return new_data
    


def main():
    """
    Main function to run real-time ETA prediction for a new vehicle.
    
    Vehicle data from step 184080:
    - Vehicle ID: my_vehicle
    - Current position: (8853.59, -2766.56) in zone B
    - Current edge: AR7AS7
    - Route length: 12932.86 meters
    - Destination: (501.6, 1900.82)
    """

    '''
    labels_184080.json
      {
    "vehicle_id": "my_vehicle",
    "origin_time_sec": 184081,
    "destination_time_sec": 184980,
    "total_travel_time_seconds": 899,
    "eta": 900
    },
    '''
    print("🚀 Real-Time ETA Prediction for New Vehicle")
    print("=" * 60)
    
    # Initialize inference system
    print("1. Initializing RealTimeInference...")
    inference = RealTimeInference(
        checkpoint_path="./logs/inference_testing/temporal_route_aware/gru/moe_best.pt",
        config_path="./config.test.yaml",
        seed=42
    )
    print("✅ Initialization complete!")
    print()
    
    # Vehicle information from the provided data
    vehicle_info = {
        "veh_id": "my_vehicle",
        "current_x": 8853.589184611308,
        "current_y": -2766.56,
        "destination_x": 501.6,
        "destination_y": 1900.8151167961119,
        "current_edge_num_lanes": 2,  # Assuming 2 lanes for AR7AS7
        "zone": "B",
        "route_length": 12932.86,
        "current_edge_id": "AR7AS7"  # Use the actual edge ID from the vehicle data
    }
    
    # Route information
    route_info = {
        "route_edges": [
            "AR7AS7", "AS7AS8", "AS8AS9", "AS9AS10", "AS10AR10",
            "AR10AQ10", "AQ10AP10", "AP10AO10", "AO10AN10", "-E1",
            "AK0AJ0", "AJ0AI0", "AI0AH0", "AH0AG0", "AG0AF0",
            "AF0AE0", "AE0AD0", "AD0AC0", "AC0AB0", "AB0AB1",
            "AB1AB2", "AB2AB3", "AB3AB4"
        ],
        "route_length": 12932.86
    }
    
    # Current simulation time (step 184080 in seconds)
    current_time = 184080
    
    print("2. Vehicle Information:")
    print(f"   - Vehicle ID: {vehicle_info['veh_id']}")
    print(f"   - Current Position: ({vehicle_info['current_x']:.2f}, {vehicle_info['current_y']:.2f})")
    print(f"   - Destination: ({vehicle_info['destination_x']:.2f}, {vehicle_info['destination_y']:.2f})")
    print(f"   - Zone: {vehicle_info['zone']}")
    print(f"   - Route Length: {vehicle_info['route_length']:.2f} meters")
    print(f"   - Number of route edges: {len(route_info['route_edges'])}")
    print()
    
    print("3. Running ETA Prediction...")
    try:
        # Run prediction
        eta_seconds = inference.predict_eta(
            vehicle_info=vehicle_info,
            route_info=route_info,
            current_time=current_time
        )
        
        # Convert to minutes and seconds
        eta_minutes = int(eta_seconds // 60)
        eta_remaining_seconds = int(eta_seconds % 60)
        
        print("✅ Prediction completed!")
        print()
        print("🎯 ETA Prediction Results:")
        print(f"   - Predicted ETA: {eta_seconds:.2f} seconds")
        print(f"   - Predicted ETA: {eta_minutes} minutes {eta_remaining_seconds} seconds")
        print(f"   - Predicted ETA: {eta_minutes + eta_remaining_seconds/60:.2f} minutes")
        print()
        
        # Show time conversion
        hours, minutes, seconds = inference._convert_step_to_time(184080)
        print(f"📅 Simulation Time: {hours:02d}h {minutes:02d}m {seconds:02d}s (Step 184080)")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("🏁 Real-time inference complete!")


if __name__ == "__main__":
    main()