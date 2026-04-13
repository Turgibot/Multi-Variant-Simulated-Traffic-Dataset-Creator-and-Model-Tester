import math
import torch
import torch.nn.functional as F

def denormalize_route_length(normalized_route_length: torch.Tensor) -> torch.Tensor:
    """
    Denormalize route_length from [0,1] back to meters using min-max normalization.
    Based on vehicle_feature_summary.csv: min=476.6, max=23133.41 meters
    """
    min_val, max_val = 476.6, 23133.41
    return normalized_route_length * (max_val - min_val) + min_val

def _as_per_token(stat_tensor: torch.Tensor, batch_veh: torch.Tensor, device) -> torch.Tensor:
    """
    Expand a per-graph tensor (shape [B] or scalar []) to per-token shape [Nv]
    using the per-vehicle graph ids in batch_veh.
    """
    if stat_tensor.dim() == 0:
        # scalar → broadcast
        return stat_tensor.to(device).expand(batch_veh.numel())
    elif stat_tensor.dim() == 1:
        # [B] → index with batch ids (batch_veh is [Nv] with values in [0..B-1])
        return stat_tensor.to(device)[batch_veh]
    else:
        raise ValueError(f"Unexpected stat tensor shape {tuple(stat_tensor.shape)}")

def get_target_tensor(bt_last, target_key: str) -> torch.Tensor:
    if not hasattr(bt_last, target_key):
        raise KeyError(f"Batch is missing target '{target_key}'.")
    return getattr(bt_last, target_key).float()

@torch.no_grad()
def invert_to_seconds(y_like: torch.Tensor, bt_last, target_key: str, batch_veh: torch.Tensor) -> torch.Tensor:
    """
    Map predictions/labels from target space back to **seconds**.
    batch_veh: [Nv] per-vehicle graph ids (bt_last.batch[veh_mask]).
    """
    device = y_like.device

    if target_key == "y":  # raw seconds
        return y_like.clamp_min(0.0)

    elif target_key == "y_minmax":  # eta / p98
        p98 = _as_per_token(bt_last.eta_p98, batch_veh, device).clamp_min(1.0)
        return (y_like * p98).clamp_min(0.0)

    elif target_key == "y_z":  # (eta - mean) / std
        mean = _as_per_token(bt_last.eta_mean, batch_veh, device)
        std  = _as_per_token(bt_last.eta_std,  batch_veh, device).clamp_min(1e-8)
        return (y_like * std + mean).clamp_min(0.0)

    elif target_key == "y_log":  # log1p(eta)
        return torch.expm1(y_like).clamp_min(0.0)

    elif target_key == "y_log_z":  # (log1p(eta) - log_mean) / log_std
        log_mean = _as_per_token(bt_last.eta_log_mean, batch_veh, device)
        log_std  = _as_per_token(bt_last.eta_log_std,  batch_veh, device).clamp_min(1e-8)
        logv = y_like * log_std + log_mean
        return torch.expm1(logv).clamp_min(0.0)

    else:
        raise ValueError(f"Unknown target_key '{target_key}'")

def huber_beta_for_target(bt_last, target_key: str, batch_veh: torch.Tensor, beta_seconds: float = 30.0) -> torch.Tensor:
    """
    Convert a kink in **seconds** to target space. Returns [Nv] tensor.
    """
    device = batch_veh.device
    beta_seconds = float(beta_seconds)
    Nv = batch_veh.numel()

    if target_key == "y":              # raw seconds
        return torch.full((Nv,), beta_seconds, device=device)

    elif target_key == "y_minmax":     # eta / p98
        p98 = _as_per_token(bt_last.eta_p98, batch_veh, device).clamp_min(1.0)
        return torch.as_tensor(beta_seconds, device=device) / p98

    elif target_key == "y_z":          # (eta - mean) / std
        std = _as_per_token(bt_last.eta_std, batch_veh, device).clamp_min(1e-8)
        return torch.as_tensor(beta_seconds, device=device) / std

    elif target_key == "y_log":        # log1p(eta)
        return torch.full((Nv,), math.log1p(beta_seconds), device=device)

    elif target_key == "y_log_z":      # (log1p(eta) - log_mean)/log_std
        log_std = _as_per_token(bt_last.eta_log_std, batch_veh, device).clamp_min(1e-8)
        return torch.as_tensor(math.log1p(beta_seconds), device=device) / log_std

    else:
        raise ValueError(f"Unknown target_key '{target_key}'")
