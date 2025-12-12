from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 1
    router_bias: bool = True
    router_jitter: float = 0.0
    capacity_factor: Optional[float] = None
    route_mask_pad_tokens: bool = False  # flag bạn yêu cầu


def moe_load_balance_loss(
    router_logits: torch.Tensor,  # [N, E]
    topk_idx: torch.Tensor,       # [N, K]
    num_experts: int,
) -> torch.Tensor:
    probs = torch.softmax(router_logits, dim=-1)  # [N, E]
    importance = probs.mean(dim=0)                # [E]

    # load from hard top-1 routing
    one_hot = torch.zeros((topk_idx.size(0), num_experts), device=topk_idx.device, dtype=probs.dtype)
    one_hot.scatter_(1, topk_idx[:, :1], 1.0)
    load = one_hot.mean(dim=0)                    # [E]

    return num_experts * torch.sum(importance * load)


class MoEFFN(nn.Module):
    """
    Replaces (Intermediate + Output) FFN with MoE FFN.
    Optionally masks pad tokens from routing via attention_mask.
    Stores last router info for aux loss.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_p: float,
        layer_norm_eps: float,
        act_fn,
        base_dense1: nn.Linear,
        base_dense2: nn.Linear,
        base_layernorm: nn.LayerNorm,
        moe_cfg: MoEConfig,
    ):
        super().__init__()
        assert 1 <= moe_cfg.top_k <= moe_cfg.num_experts

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = nn.Dropout(dropout_p)
        self.act_fn = act_fn
        self.moe_cfg = moe_cfg

        self.router = nn.Linear(hidden_size, moe_cfg.num_experts, bias=moe_cfg.router_bias)

        self.expert_dense1 = nn.ModuleList(
            [nn.Linear(hidden_size, intermediate_size) for _ in range(moe_cfg.num_experts)]
        )
        self.expert_dense2 = nn.ModuleList(
            [nn.Linear(intermediate_size, hidden_size) for _ in range(moe_cfg.num_experts)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # init experts from original FFN weights
        for e in range(moe_cfg.num_experts):
            self.expert_dense1[e].weight.data.copy_(base_dense1.weight.data)
            self.expert_dense1[e].bias.data.copy_(base_dense1.bias.data)
            self.expert_dense2[e].weight.data.copy_(base_dense2.weight.data)
            self.expert_dense2[e].bias.data.copy_(base_dense2.bias.data)

        self.layer_norm.weight.data.copy_(base_layernorm.weight.data)
        self.layer_norm.bias.data.copy_(base_layernorm.bias.data)

        # init router near-uniform
        nn.init.zeros_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

        # cache for aux loss
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _apply_capacity(self, token_idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if token_idx.numel() <= max_tokens:
            return token_idx
        return token_idx[:max_tokens]

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, h = hidden_states.shape
        x = hidden_states
        flat = x.reshape(-1, h)  # [N, H]
        n_tokens = flat.size(0)

        logits = self.router(flat)  # [N, E]
        if self.training and self.moe_cfg.router_jitter and self.moe_cfg.router_jitter > 0:
            logits = logits + torch.randn_like(logits) * self.moe_cfg.router_jitter

        # route mask for pad tokens
        active_idx = None
        if self.moe_cfg.route_mask_pad_tokens and attention_mask is not None:
            mask_flat = attention_mask.reshape(-1).to(dtype=torch.bool)  # [N]
            active_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)  # [Na]
            logits_active = logits.index_select(0, active_idx)
        else:
            logits_active = logits

        topk_vals, topk_idx = torch.topk(logits_active, k=self.moe_cfg.top_k, dim=-1)  # [Na, K] or [N, K]
        topk_w = F.softmax(topk_vals, dim=-1)

        # cache for aux loss (only active tokens)
        self.last_router_logits = logits_active
        self.last_topk_idx = topk_idx

        out_active = flat.new_zeros((logits_active.size(0), h))  # [Na, H] or [N, H]

        max_tokens_per_expert = None
        if self.moe_cfg.capacity_factor is not None:
            nt = logits_active.size(0)
            max_tokens_per_expert = int(((nt / self.moe_cfg.num_experts) * self.moe_cfg.capacity_factor) + 0.999)

        for e in range(self.moe_cfg.num_experts):
            mask = topk_idx.eq(e)
            if not mask.any():
                continue

            tok_pos, k_pos = torch.where(mask)
            if max_tokens_per_expert is not None:
                tok_pos = self._apply_capacity(tok_pos, max_tokens_per_expert)
                k_pos = k_pos[: tok_pos.numel()]

            x_e = (flat.index_select(0, active_idx).index_select(0, tok_pos)) if active_idx is not None else flat.index_select(0, tok_pos)
            w_e = topk_w[tok_pos, k_pos].unsqueeze(-1)

            y = self.expert_dense1[e](x_e)
            y = self.act_fn(y)
            y = self.expert_dense2[e](y)
            y = self.dropout(y)

            out_active.index_add_(0, tok_pos, y * w_e)

        # scatter back if masked routing
        out = flat.new_zeros((n_tokens, h))
        if active_idx is not None:
            out.index_copy_(0, active_idx, out_active)
        else:
            out = out_active

        out = out.view(bsz, seqlen, h)
        out = self.layer_norm(out + x)
        return out
