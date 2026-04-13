#!/usr/bin/env python3
"""
ETA_Predictor.py - Proven inference approach
"""

import torch
import numpy as np
import random
import yaml
from models.temporal_dataset import TemporalGraphDataset, temporal_collate
from torch.utils.data import DataLoader
from models.model_temporal_moe import TemporalMoEETA
from models.utils_targets import invert_to_seconds, get_target_tensor

def set_deterministic_seed(seed=42):
    """Set all random seeds for deterministic behavior."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Inference:
    
    def __init__(self, checkpoint_path, config_path, seed=42):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.seed = seed
        
        # Set deterministic seed
        set_deterministic_seed(seed)
        
        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = TemporalMoEETA(
            node_in_dim=28,
            d_hidden=128,
            fusion_out=192,
            n_experts=6,
            top_k=2,
            dropout=0.1,
            predict_on="last",
            edge_vocab_size=1294,
            route_emb_dim=64,
            edge_dim=7,
            temporal_kind=self.cfg["model"].get("temporal_kind", "gru"),
            ablation_variant=self.cfg["model"].get("ablation_variant", "temporal_route_aware"),
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()
        
        print(f"✅ Inference initialized with seed {seed}")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.cfg['model']['ablation_variant']}")
    
    def get_baseline_predictions(self, step):
        """Get baseline predictions for the original simulation at given step."""
        print(f"\n1. BASELINE: Original Simulation at Step {step}")
        print("=" * 50)
        # Map simulation step to 24-hour cycle data
        # 24 * 60 * 60 = 86400 seconds in a day
        # Map current step to corresponding step in available data
        step_in_24h_cycle = step % (24 * 60 * 60)
        # Update config to use the mapped step
        self.cfg["data"]["playback_start_idx"] = max(29, step_in_24h_cycle // 30 - 29)  # 30 files ending at step
        self.cfg["data"]["playback_num_files"] = 30

        # Load original data
        playback_ds = TemporalGraphDataset(
            root=self.cfg["data"]["playback_path"],
            window_size=self.cfg["data"]["window_size"],
            stride_size=self.cfg["data"]["stride_size"],
            num_files=self.cfg["data"].get("playback_num_files"),
            start_idx=self.cfg["data"].get("playback_start_idx", 0),
            allow_incomplete_tail=self.cfg["data"].get("allow_incomplete_tail", False),
            shuffle_windows=False,
        )
        
        collate = temporal_collate
        playback_loader = DataLoader(
            playback_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=collate
        )
        
        # Get original batch
        batch = next(iter(playback_loader))
        time_batches_original = [tb.to(self.device) for tb in batch["time_batches"]]
        
        # Run original prediction
        with torch.no_grad():
            y_hat_original, aux_original, veh_mask_original = self.model(time_batches_original, train=False)
            bt_original = time_batches_original[-1]
            target_key = self.cfg["train"]["target_key"]
            batch_veh_original = bt_original.batch[veh_mask_original]
            y_tgt_original = get_target_tensor(bt_original, target_key).float()
            y_sec_original = invert_to_seconds(y_tgt_original, bt_original, target_key, batch_veh_original)
            yhat_sec_original = invert_to_seconds(y_hat_original, bt_original, target_key, batch_veh_original)
        
        print(f"   Original simulation has {veh_mask_original.sum().item()} vehicles")
        print(f"   Step: {step} ({step//3600}h {(step%3600)//60}m {step%60}s)")
        
        # Show first few vehicles
        print(f"   First 5 vehicle predictions:")
        for i in range(min(5, len(yhat_sec_original))):
            target = y_sec_original[i].item()
            prediction = yhat_sec_original[i].item()
            print(f"   Vehicle {i}: Target={target:.0f}s, Pred={prediction:.2f}s")
        
        return {
            'time_batches_original': time_batches_original,
            'y_sec_original': y_sec_original,
            'yhat_sec_original': yhat_sec_original,
            'veh_mask_original': veh_mask_original,
            'bt_original': bt_original,
            'target_key': target_key
        }
    
    def add_vehicle_and_predict(self, baseline_data, vehicle_info, route_info, step):
        """Add a new vehicle and get updated predictions."""
        print(f"\n2. ADDING NEW VEHICLE '{vehicle_info['veh_id']}'")
        print("=" * 50)
        
        # Import real-time inference
        from models.real_time_inference import RealTimeInference
        
        # Initialize real-time inference
        inference = RealTimeInference(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            seed=self.seed
        )
        
        # Replace the model with our shared model
        inference.model = self.model
        
        # Load temporal window for the step
        temporal_window = inference._load_temporal_window(step)
        current_pt_file = temporal_window[-1]
        
        print(f"   Current simulation has {current_pt_file.x[current_pt_file.x[:, 0] == 1].shape[0]} vehicles")
        print(f"   Adding vehicle '{vehicle_info['veh_id']}' with route: {len(route_info['route_edges'])} edges")
        print(f"   Route length: {route_info['route_length']:.2f} meters")
        print(f"   Current edge: {vehicle_info['current_edge_id']}")
        print(f"   Zone: {vehicle_info['zone']}")
        #print all the params
        print(f"   Current step: {step}")
        print(f"   Vehicle info: {vehicle_info}")
        print(f"   Route info: {route_info}")
     
        
        # Add the new vehicle to the simulation
        updated_pt_file = inference.add_vehicle_to_last_snapshot(
            current_pt_file=current_pt_file,
            veh_id=vehicle_info["veh_id"],
            start_step=step,
            route_edges=route_info["route_edges"],
            route_length=vehicle_info["route_length"],
            zone=vehicle_info["zone"],
            current_x=vehicle_info["current_x"],
            current_y=vehicle_info["current_y"],
            destination_x=vehicle_info["destination_x"],
            destination_y=vehicle_info["destination_y"],
            current_edge_num_lanes=vehicle_info["current_edge_num_lanes"],
            current_edge_id=vehicle_info["current_edge_id"]
        )
        
        print(f"   After adding new vehicle: {updated_pt_file.x[updated_pt_file.x[:, 0] == 1].shape[0]} vehicles")
        
        print("\n3. RUNNING PREDICTION WITH NEW VEHICLE")
        print("=" * 50)
        
        # Create updated temporal window
        temporal_window_updated = temporal_window[:-1] + [updated_pt_file]
        time_batches_updated = [timestep.to(self.device) for timestep in temporal_window_updated]
        
        # Ensure batch structure exists
        for timestep in time_batches_updated:
            if timestep.batch is None:
                num_nodes = timestep.x.size(0)
                timestep.batch = torch.zeros(num_nodes, dtype=torch.long, device=timestep.x.device)
        
        # Run prediction with new vehicle
        with torch.no_grad():
            y_hat_updated, aux_updated, veh_mask_updated = self.model(time_batches_updated, train=False)
            bt_updated = time_batches_updated[-1]
            batch_veh_updated = bt_updated.batch[veh_mask_updated]
            yhat_sec_updated = invert_to_seconds(y_hat_updated, bt_updated, baseline_data['target_key'], batch_veh_updated)
        
        print(f"   Updated simulation has {veh_mask_updated.sum().item()} vehicles")
        
        return {
            'yhat_sec_updated': yhat_sec_updated,
            'veh_mask_updated': veh_mask_updated
        }
    
    def compare_results(self, baseline_data, updated_data, vehicle_info):
        """Compare baseline vs updated results."""
        print(f"\n4. COMPARISON: Before vs After Adding '{vehicle_info['veh_id']}'")
        print("=" * 70)
        print(f"{'Idx':<4} {'Target':<8} {'Original':<10} {'Updated':<10} {'Change':<8} {'Impact':<8}")
        print("-" * 70)
        
        y_sec_original = baseline_data['y_sec_original']
        yhat_sec_original = baseline_data['yhat_sec_original']
        yhat_sec_updated = updated_data['yhat_sec_updated']
        
        for i in range(min(10, len(yhat_sec_original))):  # Show first 10 vehicles
            target = y_sec_original[i].item()
            original_pred = yhat_sec_original[i].item()
            updated_pred = yhat_sec_updated[i].item()
            change = updated_pred - original_pred
            change_pct = (change / max(original_pred, 1e-8)) * 100.0
            
            # Determine impact level
            if abs(change) < 1.0:
                impact = "None"
            elif abs(change) < 10.0:
                impact = "Low"
            elif abs(change) < 50.0:
                impact = "Med"
            else:
                impact = "High"
            
            print(f"{i:<4} {target:<8.0f} {original_pred:<10.2f} {updated_pred:<10.2f} {change:+.2f}s{'':<3} {impact:<8}")
        
        # Show new vehicle
        print("-" * 70)
        print(f"{'NEW':<4} {'N/A':<8} {'N/A':<10} {yhat_sec_updated[-1].item():<10.2f} {'NEW':<8} {'NEW':<8}")
        print("=" * 70)
        
        # Calculate impact statistics
        changes = []
        for i in range(len(yhat_sec_original)):
            original_pred = yhat_sec_original[i].item()
            updated_pred = yhat_sec_updated[i].item()
            change = updated_pred - original_pred
            changes.append(change)
        
        changes = np.array(changes)
        affected_vehicles = (abs(changes) > 1.0).sum()
        
        print(f"\n5. IMPACT SUMMARY")
        print("=" * 50)
        print(f"Original vehicles: {len(yhat_sec_original)}")
        print(f"Updated vehicles: {len(yhat_sec_updated)}")
        print(f"New vehicle '{vehicle_info['veh_id']}' ETA: {yhat_sec_updated[-1].item():.2f} seconds ({yhat_sec_updated[-1].item()/60:.1f} minutes)")
        print(f"Vehicles affected: {affected_vehicles}/{len(yhat_sec_original)} ({affected_vehicles/len(yhat_sec_original)*100:.1f}%)")
        print(f"Average change: {changes.mean():.2f}s")
        print(f"Max improvement: {changes.min():.2f}s")
        print(f"Max delay: {changes.max():.2f}s")
        print(f"Standard deviation: {changes.std():.2f}s")
        
        return {
            'new_vehicle_eta': yhat_sec_updated[-1].item(),
            'affected_vehicles': affected_vehicles,
            'total_vehicles': len(yhat_sec_original),
            'avg_change': changes.mean(),
            'max_improvement': changes.min(),
            'max_delay': changes.max()
        }
    
    def predict_eta(self, vehicle_info, route_info, step):
        """
        Predict ETA for a new vehicle and return the prediction and average change.
        
        Args:
            vehicle_info: Dictionary containing vehicle information
                Required keys: veh_id, current_x, current_y, destination_x, destination_y,
                              current_edge_num_lanes, zone, route_length, current_edge_id
            route_info: Dictionary containing route information
                Required keys: route_edges (list of edge IDs), route_length
            step: Current simulation step (e.g., 64080, 184080)
        
        Returns:
            tuple: (predicted_eta_seconds, average_change_seconds)
        """
        # Get baseline predictions
        print(f"Getting baseline predictions for step {step}")
        baseline_data = self.get_baseline_predictions(step)
        
        # Add vehicle and get updated predictions
        updated_data = self.add_vehicle_and_predict(baseline_data, vehicle_info, route_info, step)
        
        # Calculate changes
        yhat_sec_original = baseline_data['yhat_sec_original']
        yhat_sec_updated = updated_data['yhat_sec_updated']
        
        changes = []
        for i in range(len(yhat_sec_original)):
            original_pred = yhat_sec_original[i].item()
            updated_pred = yhat_sec_updated[i].item()
            change = updated_pred - original_pred
            changes.append(change)
        
        changes = np.array(changes)
        avg_change = changes.mean()
        
        # Get new vehicle ETA (last vehicle in updated predictions)
        predicted_eta = yhat_sec_updated[-1].item()
        
        return predicted_eta, avg_change

