from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


@dataclass
class SeqMoELogitsConfig:
    in_dim: int
    num_labels: int
    num_experts: int
    top_k: int
    expert_hidden: Optional[int] = None
    dropout_p: float = 0.0
    router_bias: bool = True
    router_jitter: float = 0.0


class _ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_labels: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeqMoELogits(nn.Module):
    """Sequence-level MoE that outputs delta logits.

    Input:  x [B, D]
    Output: delta_logits [B, C]

    Caches:
      last_router_logits: [B, E]
      last_topk_idx: [B, K]
    """

    def __init__(self, cfg: SeqMoELogitsConfig):
        super().__init__()
        self.cfg = cfg
        self.num_experts = int(cfg.num_experts)
        self.moe_top_k = int(cfg.top_k)

        hidden = int(cfg.expert_hidden or max(64, cfg.in_dim))
        self.router = nn.Linear(cfg.in_dim, self.num_experts, bias=bool(cfg.router_bias))
        self.dropout = nn.Dropout(float(cfg.dropout_p))
        self.experts = nn.ModuleList(
            [_ExpertMLP(cfg.in_dim, hidden, cfg.num_labels, float(cfg.dropout_p)) for _ in range(self.num_experts)]
        )

        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None

    def set_top_k(self, k: int) -> None:
        k = int(k)
        if k < 1:
            raise ValueError("top_k must be >= 1")
        if k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.moe_top_k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"SeqMoELogits expects [B, D], got {tuple(x.shape)}")

        # Router logits
        router_logits = self.router(x)  # [B, E]

        if self.training and self.cfg.router_jitter and self.cfg.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * float(self.cfg.router_jitter)

        # TopK
        k = int(self.moe_top_k)
        topk_vals, topk_idx = torch.topk(router_logits, k=k, dim=-1)  # [B, K]
        topk_w = torch.softmax(topk_vals, dim=-1)  # [B, K]

        # Cache for debug and aux loss
        self.last_router_logits = router_logits
        self.last_topk_idx = topk_idx

        # Weighted sum of expert outputs
        # Compute outputs only for selected experts (small E and K, simple loop)
        B = x.size(0)
        C = self.cfg.num_labels
        out = x.new_zeros((B, C))

        for j in range(k):
            idx_j = topk_idx[:, j]  # [B]
            w_j = topk_w[:, j].unsqueeze(-1)  # [B, 1]

            # Group samples by expert id to avoid per-sample expert calls
            for e in torch.unique(idx_j):
                e_int = int(e.item())
                mask = (idx_j == e)
                if not mask.any():
                    continue
                x_e = x[mask]
                y_e = self.experts[e_int](x_e)  # [n_e, C]
                out[mask] = out[mask] + y_e * w_j[mask]

        out = self.dropout(out)
        return out

    @torch.no_grad()
    def debug_stats(self) -> Optional[Dict[str, Any]]:
        if self.last_router_logits is None or self.last_topk_idx is None:
            return None
        logits = self.last_router_logits
        topk_idx = self.last_topk_idx
        E = int(self.num_experts)

        probs = torch.softmax(logits, dim=-1)
        usage_soft = probs.mean(dim=0)  # [E]

        flat = topk_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=E).float()
        frac = counts / (counts.sum().clamp_min(1.0))

        return {
            "usage_soft": usage_soft.detach().cpu(),
            "topk_frac": frac.detach().cpu(),
            "moe_top_k": int(self.moe_top_k),
        }
