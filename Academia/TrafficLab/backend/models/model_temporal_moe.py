# model_temporal_moe.py
import math
from typing import List, Literal, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
from models.moe_head import MoEHead, load_balancing_loss


# Indices consistent with your dataset
NODE_TYPE_IDX = 0  # 0=junction, 1=vehicle

# -----------------------------
# Graph encoder (unchanged)
# -----------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2,
                 p_drop: float = 0.1, heads: int = 4, edge_dim: int = 7):
        super().__init__()
        self.convs, self.norms, self.skips = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        d_in = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(d_in, hidden // heads, heads=heads,
                                        edge_dim=edge_dim, add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden))
            self.skips.append(nn.Identity() if d_in == hidden else nn.Linear(d_in, hidden))
            d_in = hidden
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, ln, skip in zip(self.convs, self.norms, self.skips):
            h_new = conv(h, edge_index, edge_attr)
            h = ln(torch.nn.functional.gelu(h_new) + skip(h))
            h = self.drop(h)
        return h
# -----------------------------
# Temporal aggregator
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard transformer-style sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) or (B, T, N*D). Adds PE to the last dim.
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)  # (1, T, D) + (B, T, D)


class TemporalAggregator(nn.Module):
    """
    Aggregates temporal context from static edge features across time.
    
    For the "temporal" ablation variants:
    - Processes static edge features from first T-1 snapshots
    - Aggregates temporal patterns in edge-level representations
    - Returns temporal context for graph-level prediction enhancement
    
    Expected input: edge features from static edges across time
    Output: aggregated temporal context
    """
    def __init__(
        self,
        d: int,
        kind: Literal["gru", "transformer", "none"] = "none",
        *,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kind = kind
        self.d = d

        if kind == "gru":
            # GRU for temporal aggregation
            self.rnn = nn.GRU(
                input_size=d,
                hidden_size=d,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
        elif kind == "transformer":
            # Transformer encoder for temporal aggregation
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, batch_first=True,
                dim_feedforward=2 * d, dropout=dropout, activation="gelu"
            )
            self.rnn = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.posenc = SinusoidalPositionalEncoding(d)
            self.layer_norm = nn.LayerNorm(d)
        elif kind == "none":
            # No temporal aggregation; just return zeros for context
            self.rnn = nn.Identity()
        else:
            raise ValueError(f"Unknown TemporalAggregator kind: {kind}")

    @staticmethod
    def _all_same_shape(ts: List[torch.Tensor]) -> bool:
        if not ts:
            return True
        s0 = ts[0].shape
        return all(t.shape == s0 for t in ts)

    def forward(self, edge_features_seq: List[torch.Tensor]) -> torch.Tensor:
        """
        edge_features_seq: List of (E_static, edge_dim) tensors from first T-1 snapshots
        Returns: (E_static, edge_dim) aggregated temporal edge features
        """
        if self.kind == "none" or not edge_features_seq:
            # No temporal processing - return zeros
            if edge_features_seq:
                return torch.zeros_like(edge_features_seq[0])
            else:
                return torch.zeros(1, 7)  # edge_dim = 7_graph
            
        # Check all edge feature tensors have same shape (same static edges)
        if not self._all_same_shape(edge_features_seq):
            print("Warning: TemporalAggregator edge features differ across time; returning zeros.")
            return torch.zeros_like(edge_features_seq[0])

        # Stack edge features across time: (E_static, T, edge_dim)
        x = torch.stack(edge_features_seq, dim=1)  # (E_static, T, edge_dim)
        E, T, edge_dim = x.shape
        
        # Reshape for temporal processing: (E_static, T, edge_dim) -> (E_static, T, d_hidden)
        if edge_dim != self.d:
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(edge_dim, self.d).to(x.device)
            x_proj = self.input_proj(x)  # (E_static, T, d_hidden)
        else:
            x_proj = x

        if self.kind == "gru":
            # GRU over time for each edge independently
            out, h_n = self.rnn(x_proj)      # out: (E_static, T, d_hidden)
            agg = out[:, -1, :]              # (E_static, d_hidden) - last output
        else:  # transformer
            # Transformer over time; add positional encodings
            x_pe = self.posenc(x_proj)       # (E_static, T, d_hidden)
            out = self.rnn(x_pe)             # (E_static, T, d_hidden)
            last = out[:, -1, :]             # (E_static, d_hidden)
            agg = self.layer_norm(last)      # (E_static, d_hidden)

        # Project back to edge_dim
        if not hasattr(self, 'output_proj'):
            self.output_proj = nn.Linear(self.d, edge_dim).to(x.device)
        
        return self.output_proj(agg)  # (E_static, edge_dim)



# -----------------------------
# Simple route sequence encoder
# -----------------------------
class RouteEncoder(nn.Module):
    """
    Encodes a ragged list of edge IDs per-vehicle using an edge embedding + mean pooling.

    Inputs:
      vehicle_route_left         : LongTensor [sum(L_i)]
      vehicle_route_left_splits  : LongTensor [Nv], lengths per vehicle (sum = above)

    Output:
      route_z : FloatTensor [Nv, d_edge]
    """
    def __init__(self, edge_vocab_size: int, d_edge: int = 64):
        super().__init__()
        self.edge_emb = nn.Embedding(edge_vocab_size, d_edge)

    def forward(self,
                route_flat: torch.Tensor,
                route_splits: torch.Tensor) -> torch.Tensor:
        device = route_flat.device
        d_edge = self.edge_emb.embedding_dim

        if route_splits.numel() == 0:
            return torch.zeros(0, d_edge, device=device)

        emb = self.edge_emb(route_flat)  # [sumL, d_edge]
        # Split into per-veh sequences
        lengths = route_splits.tolist()
        seqs = torch.split(emb, lengths, dim=0)

        pooled = []
        for s in seqs:
            if s.size(0) == 0:
                pooled.append(torch.zeros(d_edge, device=device))
            else:
                pooled.append(s.mean(dim=0))
        return torch.stack(pooled, dim=0)  # [Nv, d_edge]


# -----------------------------
# Fusion MLP
# -----------------------------
class Fusion(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 256, d_out: int = 192, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(p),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, z):
        return self.net(z)

# -----------------------------
# Full model
# -----------------------------
class TemporalMoEETA(nn.Module):
    """
    End-to-end model:
      - Encode graph at each t
      - Build temporal context from the first T-1 slices (per-graph pooled)
      - Extract vehicle embeddings at prediction t* = last
      - Concat [veh_z_last, temporal_ctx_graph, (optional) route_z]
      - Fuse → MoE head predicts y for vehicles in the last snapshot
    """
    def __init__(
        self,
        node_in_dim: int,
        d_hidden: int = 128,
        fusion_out: int = 192,
        n_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.1,
        predict_on: Literal["last","all"]="last",
        edge_vocab_size: Optional[int] = None,       # REQUIRED when ablation enables route awareness
        route_emb_dim: int = 64,                     # embedding size for edge IDs
        edge_dim: int = 7,                           # edge_attr feature size
        zero_agg_route_cols_in_full: bool = True,    # zero x[:, 25:27] for veh nodes when using route features
        temporal_kind: Literal["gru","transformer", "none"] = "none",
        ablation_variant: Literal["base_graph","dynamic_graph","route_graph","temporal_base","temporal_dynamic","temporal_route_aware"]="base_graph",
    ):
        super().__init__()
        self.predict_on = predict_on
        self.ablation_variant = ablation_variant
        
        # Apply ablation logic - determine what features to use based on variant
        if ablation_variant == "base_graph":
            # Base road graph only: no temporal, no route features, road topology only
            self.route_awareness = "none"
            self.use_temporal_context = False
            self.use_dynamic_edges = False
            temporal_kind = "none"  # Override for temporal aggregator
        elif ablation_variant == "dynamic_graph":
            # Add dynamic graph: no temporal, no route features, but WITH vehicle interaction edges
            self.route_awareness = "none"
            self.use_temporal_context = False
            self.use_dynamic_edges = True
            temporal_kind = "none"  # Override for temporal aggregator
        elif ablation_variant == "route_graph":
            # Add route awareness: no temporal, WITH route features and dynamic edges
            self.route_awareness = "full"
            self.use_temporal_context = False
            self.use_dynamic_edges = True
            temporal_kind = "none"  # Override for temporal aggregator
        elif ablation_variant == "temporal_base":
            # Add temporal to base_graph: static roads only, no route features, WITH temporal context
            self.route_awareness = "none"
            self.use_temporal_context = temporal_kind != "none"
            self.use_dynamic_edges = False  # Static roads only
            self.use_static_temporal = True  # Use static edges for temporal context
        elif ablation_variant == "temporal_dynamic":
            # Temporal + dynamic: temporal context from dynamic graph, no route features
            self.route_awareness = "none"
            self.use_temporal_context = temporal_kind != "none"
            self.use_dynamic_edges = True
            self.use_static_temporal = False  # Use full graph for temporal context
        elif ablation_variant == "temporal_route_aware":
            # Full model: temporal context from dynamic graph WITH route features
            self.route_awareness = "full"
            self.use_temporal_context = temporal_kind != "none"
            self.use_dynamic_edges = True
            self.use_static_temporal = False  # Use full graph for temporal context
        else:
            raise ValueError(f"Unknown ablation_variant: {ablation_variant}. Valid options: base_graph, dynamic_graph, route_graph, temporal_base, temporal_dynamic, temporal_route_aware")
            
        self.zero_agg_route_cols_in_full = zero_agg_route_cols_in_full

        # Graph encoder (unchanged)
        self.encoder = GraphEncoder(node_in_dim, hidden=d_hidden, layers=2,
                                    p_drop=dropout, edge_dim=edge_dim)

        # Temporal aggregator for static edge features over t = 0..T-2 (if enabled)
        self.temporal = TemporalAggregator(d_hidden, kind=temporal_kind)  # processes edge features only
        # Project aggregated edge features to context size
        self.edge_to_context = nn.Sequential(
            nn.Linear(edge_dim, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout)
        )

        # Route encoder (only when route awareness is enabled)
        if self.route_awareness == "full":
            if edge_vocab_size is None:
                raise ValueError("edge_vocab_size must be provided when route_awareness='full'.")
            self.route_encoder = RouteEncoder(edge_vocab_size, d_edge=route_emb_dim)
            fusion_in = d_hidden + d_hidden + route_emb_dim  # veh_z + ctx + route_z
        else:
            self.route_encoder = None
            fusion_in = d_hidden + d_hidden  # veh_z + ctx

        # Fusion + MoE head
        self.fusion = Fusion(fusion_in, d_hidden*2, fusion_out, p=dropout)
        self.head = MoEHead(fusion_out, n_experts=n_experts, k=top_k,
                            d_hidden=fusion_out, p_drop=dropout, out_dim=1)

    @torch.no_grad()
    def _maybe_zero_route_features(self, bt: Batch, x: torch.Tensor) -> torch.Tensor:
        """Zero out route-related features based on ablation variant and route awareness."""
        x = x.clone()
        veh_mask = (bt.x[:, NODE_TYPE_IDX] > 0.5)
        
        # For ablation variants that disable route features
        if self.ablation_variant in ["base_graph", "dynamic_graph", "temporal_base", "temporal_dynamic"]:
            # Zero out route-related vehicle features for static/dynamic variants:
            # [10] route_length, [11] progress (trip progress)
            # [20-22] current_edge_num_lanes_oh (one-hot lanes of current edge)
            # [23] current_edge_demand, [24] current_edge_occupancy, 
            # [25] route_left_demand_len_disc, [26] route_left_occupancy_len_disc
            if x.size(1) > 26:  # Ensure we have enough features
                x[veh_mask, 10:12] = 0.0   # route_length, progress
                x[veh_mask, 20:27] = 0.0   # current_edge features (lanes + demand/occupancy + route_left)
        elif self.ablation_variant in ["route_graph", "temporal_route_aware"]:
            # For route_aware variants: zero only current edge features [20-26], keep route_length/progress
            # [20-22] current_edge_num_lanes_oh, [23-26] current edge demand/occupancy + route_left
            if x.size(1) > 26:  # Ensure we have enough features
                x[veh_mask, 20:27] = 0.0   # current_edge features only
        elif self.ablation_variant in ["no_route", "no_temporal_no_route"]:
            # Zero out all route-related vehicle features for other no_route variants:
            # [10] route_length, [11] progress (trip progress)
            # [20-22] current_edge_num_lanes_oh (one-hot lanes of current edge)
            # [23] current_edge_demand, [24] current_edge_occupancy, 
            # [25] route_left_demand_len_disc, [26] route_left_occupancy_len_disc
            if x.size(1) > 26:  # Ensure we have enough features
                x[veh_mask, 10:12] = 0.0   # route_length, progress
                x[veh_mask, 20:27] = 0.0   # current_edge features (lanes + demand/occupancy + route_left)
        
        # Original logic for route aggregation features (25-27) when using full route awareness
        elif self.route_awareness == "full" and self.zero_agg_route_cols_in_full:
            if hasattr(bt, "route_feat_idx"):
                idx = bt.route_feat_idx
                if torch.is_tensor(idx):
                    rs, re = int(idx[0].item()), int(idx[1].item())
                else:
                    rs, re = int(idx[0]), int(idx[1])
            else:
                rs, re = 25, 27
            if rs < re and re <= x.size(1):
                x[veh_mask, rs:re] = 0.0
                
        return x
    
    @torch.no_grad()
    def _maybe_zero_edge_route_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Zero out route-related edge features based on ablation variant."""
        if self.ablation_variant in ["base_graph", "dynamic_graph", "temporal_base", "temporal_dynamic"]:
            # Zero out route-related edge features:
            # [5] edge_demand (future demand from remaining routes)
            # [6] edge_occupancy (current vehicles on road)
            edge_attr = edge_attr.clone()
            if edge_attr.size(1) > 6:  # Ensure we have enough edge features
                edge_attr[:, 5:7] = 0.0
        # For route_aware: keep all edge features (no zeroing)
        return edge_attr
    
    @torch.no_grad()
    def _maybe_filter_dynamic_edges(self, bt: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Filter out dynamic edges for static-only ablation variants."""
        if not self.use_dynamic_edges:
            # Keep only static edges (edge_type == 0)
            if hasattr(bt, 'edge_type'):
                static_mask = (bt.edge_type == 0)
                edge_index = bt.edge_index[:, static_mask]
                edge_attr = bt.edge_attr[static_mask] if hasattr(bt, 'edge_attr') and bt.edge_attr is not None else None
            else:
                # Fallback if no edge_type available
                edge_index = bt.edge_index
                edge_attr = getattr(bt, "edge_attr", None)
        else:
            # Use all edges (static + dynamic)
            edge_index = bt.edge_index
            edge_attr = getattr(bt, "edge_attr", None)
            
        return edge_index, edge_attr

    def _graph_means(self, h: torch.Tensor, bt: Batch) -> torch.Tensor:
        """
        Compute per-graph means for vehicles and junctions separately, then concat.
        Returns: (B, 2*d_hidden) where [:d] is veh mean, [d:] is junc mean.
        """
        d = h.size(-1)
        B = int(bt.batch.max().item()) + 1 if hasattr(bt, "batch") else 1

        veh_mask = (bt.x[:, NODE_TYPE_IDX] > 0.5)
        jnc_mask = ~veh_mask

        # Helper: safe mean over selected rows of each graph
        def group_mean(mask):
            out = []
            for b in range(B):
                sel = (bt.batch == b) & mask if hasattr(bt, "batch") else mask
                if sel.any():
                    out.append(h[sel].mean(dim=0))
                else:
                    out.append(torch.zeros(d, device=h.device, dtype=h.dtype))
            return torch.stack(out, dim=0)  # (B, d)

        veh_mean = group_mean(veh_mask)     # (B, d)
        jnc_mean = group_mean(jnc_mask)     # (B, d)
        return torch.cat([veh_mean, jnc_mean], dim=-1)  # (B, 2d)

    def forward(self, time_batches: List[Batch], train: bool = True) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        time_batches: list len T (T >= 1). We form context from t = 0..T-2,
        and predict for vehicles at t* = T-1 (last).
        Returns:
            y_hat: [Nv] predictions for vehicle nodes at t*
            aux: router stats
            veh_mask: boolean mask over nodes at t* selecting vehicle rows
        """
        T = len(time_batches)
        static_edge_features_temporal: List[torch.Tensor] = []  # For edge-based temporal processing

        # Encode all snapshots through the graph encoder
        for t in range(T):
            bt = time_batches[t]
            x = self._maybe_zero_route_features(bt, bt.x)
            
            # Normal graph processing for encoding
            edge_index, edge_attr = self._maybe_filter_dynamic_edges(bt)
            if edge_attr is not None:
                edge_attr = self._maybe_zero_edge_route_features(edge_attr)
                
            h = self.encoder(x, edge_index, edge_attr)  # [N, d]
            
            # Collect static edge features for temporal processing (first T-1 only)
            if self.use_temporal_context and t < T - 1:
                # 1. Static edge features
                if hasattr(bt, 'edge_type') and hasattr(bt, 'edge_attr') and bt.edge_attr is not None:
                    static_mask = (bt.edge_type == 0)  # Only static edges (type 0)
                    static_edge_features = bt.edge_attr[static_mask]  # (E_static, edge_dim)
                    
                    # Apply feature masking for temporal processing
                    if self.ablation_variant in ["temporal_base", "temporal_dynamic"]:
                        # Zero out demand/occupancy for temporal_base and temporal_dynamic
                        static_edge_features = static_edge_features.clone()
                        if static_edge_features.size(1) > 6:
                            static_edge_features[:, 5:7] = 0.0  # Zero route-related features [5-6]
                    # For temporal_route_aware: keep all edge features
                    
                    static_edge_features_temporal.append(static_edge_features)

        # Use the last snapshot for prediction
        bt = time_batches[-1]
        h_star = h  # [N*, d] - from the last snapshot
        
        # Extract vehicle embeddings from last snapshot
        veh_mask = (bt.x[:, NODE_TYPE_IDX] > 0.5)
        z_veh = h_star[veh_mask]                         # [Nv, d_hidden]

        # Temporal context: aggregate static edge features over time
        if self.use_temporal_context and len(static_edge_features_temporal) > 0:
            # Aggregate temporal edge features
            temporal_edge_features = self.temporal(static_edge_features_temporal)  # (E_static, edge_dim)
            
            # Project to context and pool to graph-level representation
            edge_context = self.edge_to_context(temporal_edge_features)  # (E_static, d_hidden)
            # Pool edge context to graph-level (mean over static edges)
            ctx_global = edge_context.mean(dim=0, keepdim=True)  # (1, d_hidden)
            
            # Broadcast to each vehicle
            ctx_veh = ctx_global.expand(z_veh.size(0), -1)  # [Nv, d_hidden]
        else:
            # No temporal context available → zeros
            ctx_veh = torch.zeros(z_veh.size(0), z_veh.size(1),
                                  device=z_veh.device, dtype=z_veh.dtype)

        # Optional route sequence encoding (aligned with vehicle order in bt)
        if self.route_awareness == "full":
            if not (hasattr(bt, "vehicle_route_left") and hasattr(bt, "vehicle_route_left_splits")):
                raise RuntimeError("Batch is missing vehicle_route_left / vehicle_route_left_splits required for 'full' route awareness.")
            route_flat   = bt.vehicle_route_left.to(z_veh.device)           # [sumL]
            route_splits = bt.vehicle_route_left_splits.to(z_veh.device)    # [Nv]
            route_z = self.route_encoder(route_flat, route_splits)          # [Nv, d_route]
            if route_z.size(0) != z_veh.size(0):
                raise RuntimeError(f"Route encoder output mismatches vehicles: {route_z.size(0)} vs {z_veh.size(0)}")
            z = torch.cat([z_veh, ctx_veh, route_z], dim=-1)                # [Nv, d + d + d_route]
        else:
            z = torch.cat([z_veh, ctx_veh], dim=-1)                         # [Nv, d + d]

        # Fuse + predict
        zf = self.fusion(z)                             # [Nv, fusion_out]
        y_hat, aux = self.head(zf, train=train)         # [Nv, 1]
        return y_hat.squeeze(-1), aux, veh_mask
