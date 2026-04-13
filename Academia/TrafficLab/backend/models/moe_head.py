# moe_head.py
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        h = self.drop(h)
        # project for residual if dims mismatch
        if h.shape[-1] != x.shape[-1]:
            x = F.linear(x, torch.empty(h.shape[-1], x.shape[-1], device=x.device).normal_(0, 1e-6))
        return self.norm(h + x)

class Router(nn.Module):
    """Linear router with optional Gaussian noise and temperature; returns softmax probs."""
    def __init__(self, d_in: int, n_experts: int, temperature: float = 1.0, noise_std: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, n_experts)
        self.tau = temperature
        self.noise_std = noise_std

    def forward(self, z, train: bool = True):
        logits = self.proj(z)
        if train and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        return F.softmax(logits / max(1e-6, self.tau), dim=-1)

def topk_mask(probs: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (indices, mask) for Top-k per row."""
    topk = torch.topk(probs, k=k, dim=-1)
    idx = topk.indices  # [B, k]
    mask = torch.zeros_like(probs, dtype=probs.dtype)
    mask.scatter_(1, idx, 1.0)
    return idx, mask

class MoEHead(nn.Module):
    """
    Sparse Top-k Mixture-of-Experts regression head.
    Each expert is a small residual MLP producing: (residual,) optional extras.
    """
    def __init__(
        self,
        d_in: int,
        n_experts: int = 6,
        k: int = 2,
        d_hidden: int = 192,
        p_drop: float = 0.1,
        out_dim: int = 1,    # regression scalar by default
        temperature: float = 1.2,
        noise_std: float = 0.15,
    ):
        super().__init__()
        self.k = k
        self.router = Router(d_in, n_experts, temperature, noise_std)
        self.experts = nn.ModuleList([
            nn.Sequential(
                ResidualMLP(d_in, d_hidden, d_hidden, p_drop),
                nn.Linear(d_hidden, out_dim)
            )
            for _ in range(n_experts)
        ])

    def forward(self, z: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        z: [N, d_in]  (N = #vehicle nodes at prediction time)
        returns:
            y_hat: [N, out_dim]
            aux: { 'probs': [N,E], 'topk_idx':[N,k], 'importance':..., 'load':... }
        """
        probs = self.router(z, train=train)           # [N, E]
        topk_idx, topk_mask_ = topk_mask(probs, self.k)
        # renormalize probs over selected experts
        topk_probs = torch.gather(probs, 1, topk_idx) # [N, k]
        denom = topk_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        w = topk_probs / denom                        # [N, k]

        # expert outputs
        outs = []
        for j in range(self.k):
            e_idx = topk_idx[:, j]
            # gather expert modules per token; run in groups for efficiency
            # (simple loop; fine to optimize later)
            yj = torch.empty(z.size(0), self.experts[0][-1].out_features, device=z.device)
            # run each expert on its assigned tokens
            for e_id in range(len(self.experts)):
                sel = (e_idx == e_id)
                if sel.any():
                    yj[sel] = self.experts[e_id](z[sel])
            outs.append(yj)
        outs = torch.stack(outs, dim=1)               # [N, k, out_dim]
        y_hat = (outs * w.unsqueeze(-1)).sum(dim=1)   # [N, out_dim]

        # router stats for load-balancing (Switch-style)
        importance = probs.mean(dim=0)                # expected mass per expert
        load = (topk_mask_ > 0).float().mean(dim=0)   # fraction of tokens actually routed
        aux = {
            "probs": probs, "topk_idx": topk_idx,
            "importance": importance, "load": load
        }
        return y_hat, aux

def load_balancing_loss(importance: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
    """
    Encourage uniform router usage (Switch Transformer).
    Both vectors are length E and sum to ~1.
    """
    return (importance * load).sum() * importance.size(0)  # small positive when balanced; minimize -> encourage spread
