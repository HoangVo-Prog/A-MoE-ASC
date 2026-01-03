import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel


class MoE(nn.Module):
    """
    A lightweight token-level Mixture-of-Experts FFN block.

    - Routes active tokens (optionally mask pad tokens) to top-k experts.
    - Applies per-expert 2-layer FFN: H -> I -> H.
    - Returns residual + LayerNorm: LN(moe(x) + x)
    - Caches router logits + hard topk indices for aux loss / debug.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dropout_p: float,
        act_fn: nn.Module,
        router_bias: bool,
        router_jitter: float,
        capacity_factor: Optional[float],
        route_mask_pad_tokens: bool,
        layer_norm: Optional[nn.Module],
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.moe_top_k = int(top_k)

        self.dropout = nn.Dropout(float(dropout_p))
        self.act_fn = act_fn

        self.router_jitter = float(router_jitter)
        self.capacity_factor = capacity_factor
        self.route_mask_pad_tokens = bool(route_mask_pad_tokens)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=bool(router_bias))

        self.experts_dense1 = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.intermediate_size) for _ in range(self.num_experts)]
        )
        self.experts_dense2 = nn.ModuleList(
            [nn.Linear(self.intermediate_size, self.hidden_size) for _ in range(self.num_experts)]
        )

        self.ln = layer_norm if layer_norm is not None else nn.LayerNorm(self.hidden_size)

        # cache for aux loss / debug
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None

        # init router near-uniform but break symmetry (important for deterministic torch.topk ties)
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(self, hidden_states: torch.Tensor, *, token_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, hdim = hidden_states.shape
        x = hidden_states.reshape(-1, hdim)  # [N, H]

        active_idx = None
        x_active = x

        if self.route_mask_pad_tokens and token_mask is not None:
            m = token_mask.reshape(-1).bool()
            if not torch.any(m):
                self.last_router_logits = None
                self.last_topk_idx = None
                return self.ln(hidden_states)
            active_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            x_active = x.index_select(0, active_idx)

        if self.router_jitter > 0.0:
            noise = (torch.rand_like(x_active) - 0.5) * 2.0 * self.router_jitter
            x_route = x_active + noise
        else:
            x_route = x_active

        router_logits = self.router(x_route)  # [N_active, E]
        topk_vals, topk_idx = torch.topk(router_logits, k=self.moe_top_k, dim=-1)  # [N_active, K]
        topk_w = torch.softmax(topk_vals, dim=-1)  # [N_active, K]

        self.last_router_logits = router_logits
        self.last_topk_idx = topk_idx

        cap = None
        if self.capacity_factor is not None:
            cap = int(math.ceil((x_active.shape[0] / self.num_experts) * float(self.capacity_factor)))
            cap = max(cap, 1)

        out_active = torch.zeros_like(x_active)

        flat_idx = topk_idx.reshape(-1)  # [N_active*K]
        flat_tok = torch.arange(x_active.shape[0], device=x_active.device).repeat_interleave(self.moe_top_k)
        flat_kpos = torch.arange(self.moe_top_k, device=x_active.device).repeat(x_active.shape[0])

        for e in range(self.num_experts):
            sel = flat_idx == e
            if not torch.any(sel):
                continue

            tok_pos = flat_tok[sel]
            k_pos = flat_kpos[sel]

            if cap is not None and tok_pos.numel() > cap:
                # keep higher routing weights first to reduce arbitrary drops
                w_sel = topk_w.index_select(0, tok_pos).gather(1, k_pos.unsqueeze(1)).squeeze(1)  # [M]
                keep = torch.topk(w_sel, k=cap, largest=True, sorted=False).indices
                tok_pos = tok_pos.index_select(0, keep)
                k_pos = k_pos.index_select(0, keep)

            xe = x_active.index_select(0, tok_pos)
            y = self.experts_dense1[e](xe)
            y = self.act_fn(y)
            y = self.experts_dense2[e](y)
            y = self.dropout(y)

            w = topk_w.index_select(0, tok_pos).gather(1, k_pos.unsqueeze(1)).squeeze(1)  # [M]
            out_active.index_add_(0, tok_pos, y * w.unsqueeze(1))

        if active_idx is not None:
            out = x.clone()
            out.index_copy_(0, active_idx, out_active)
        else:
            out = out_active

        out = out.reshape(bsz, seqlen, hdim)
        return self.ln(out + hidden_states)

    def set_top_k(self, k: int) -> None:
        k = int(k)
        if k < 1:
            k = 1
        if k > self.num_experts:
            k = self.num_experts
        self.moe_top_k = k


def _extract_token_mask_from_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    token_mask is expected 0/1 with 1 = valid token.
    Supports common HF mask formats:
      - 2D {0,1}
      - 4D extended/additive (0 keep, negative mask)
    """
    if attention_mask is None:
        return None

    if attention_mask.dim() == 4:
        m = attention_mask[:, 0, 0, :]
        if m.dtype == torch.bool:
            return m.long()
        m_f = m.float()
        if torch.min(m_f) < 0.0 and torch.max(m_f) <= 0.0:
            return (m_f == 0.0).long()
        return (m_f > 0.0).long()

    return attention_mask


class EncoderWithMoEHead(nn.Module):
    """
    Wraps a base encoder and applies a single MoE FFN block to the last hidden states.
    """

    def __init__(self, *, base_encoder: nn.Module, moe_ffn: MoE) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffn = moe_ffn

    def forward(self, *args, **kwargs):
        outputs = self.base_encoder(*args, **kwargs)

        token_mask = _extract_token_mask_from_attention_mask(kwargs.get("attention_mask", None))

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            new_hidden = self.moe_ffn(last_hidden, token_mask=token_mask)
            return (new_hidden,) + tuple(outputs[1:])

        if hasattr(outputs, "last_hidden_state"):
            new_hidden = self.moe_ffn(outputs.last_hidden_state, token_mask=token_mask)
            outputs.last_hidden_state = new_hidden
            return outputs

        return self.moe_ffn(outputs, token_mask=token_mask)


class MoEHead(BaseModel):
    """
    MoE Head (single MoE layer applied on encoder output).
    Keeps BaseModel's fusion/classifier behavior; only augments encoder output with MoE.
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str,
        class_weights,
        focal_gamma: float,
        aux_loss_weight: float,
        num_experts: int,
        moe_top_k: int,
        moe_topk_schedule: bool,
        moe_topk_start: int,
        moe_topk_end: int,
        moe_topk_switch_epoch: int,
        router_bias: bool,
        router_jitter: float,
        capacity_factor,
        route_mask_pad_tokens: bool,
        router_entropy_weight: float = 0.0,
        aux_warmup_steps: int = 0,
        jitter_warmup_steps: int = 0,
        jitter_end: float = 0.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            loss_type=loss_type,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

        self.aux_loss_weight = float(aux_loss_weight)

        self.router_entropy_weight = float(router_entropy_weight)
        self.aux_warmup_steps = int(aux_warmup_steps)
        self.jitter_warmup_steps = int(jitter_warmup_steps)
        self.jitter_end = float(jitter_end)

        self._global_step = 0

        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn: nn.Module = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))

        self._topk_schedule_enabled = bool(moe_topk_schedule)
        self._topk_start = int(moe_topk_start)
        self._topk_end = int(moe_topk_end)
        self._topk_switch_epoch = int(moe_topk_switch_epoch)

        moe_head = MoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=moe_top_k,
            dropout_p=dropout_p,
            act_fn=act_fn,
            router_bias=router_bias,
            router_jitter=router_jitter,
            capacity_factor=capacity_factor,
            route_mask_pad_tokens=route_mask_pad_tokens,
            layer_norm=nn.LayerNorm(hidden_size),
        )

        self._jitter_start = float(router_jitter)

        base_encoder = self.encoder
        self.encoder = EncoderWithMoEHead(base_encoder=base_encoder, moe_ffn=moe_head)

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        if self.training:
            self._global_step += 1
            moe = getattr(self.encoder, "moe_ffn", None)
            if moe is not None:
                moe.router_jitter = float(self._jitter_now())

        return super().forward(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )

    def _collect_aux_loss(self) -> torch.Tensor:
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        logits = getattr(moe, "last_router_logits", None)
        topk_idx = getattr(moe, "last_topk_idx", None)
        if logits is None or topk_idx is None:
            return torch.zeros((), device=next(self.parameters()).device)

        if logits.ndim != 2 or topk_idx.ndim != 2 or logits.shape[0] == 0:
            return torch.zeros((), device=logits.device)

        n_tokens, n_experts = logits.shape
        k = topk_idx.shape[1]

        probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)  # [N, E]
        importance = probs.sum(dim=0) / float(n_tokens)  # [E]

        oh = F.one_hot(topk_idx, num_classes=n_experts).to(dtype=probs.dtype)  # [N, K, E]
        load = oh.sum(dim=(0, 1)) / float(n_tokens * k)  # [E]

        aux = n_experts * torch.sum(importance * load)
        return torch.clamp(aux, min=0.0, max=10.0)

    def _aux_weight_now(self) -> float:
        w = float(self.aux_loss_weight)
        if not self.training:
            return w
        if self.aux_warmup_steps and self.aux_warmup_steps > 0:
            t = min(1.0, float(self._global_step) / float(self.aux_warmup_steps))
            return w * t
        return w

    def _jitter_now(self) -> float:
        if not self.training:
            return float(self._jitter_start)
        if self.jitter_warmup_steps and self.jitter_warmup_steps > 0:
            t = min(1.0, float(self._global_step) / float(self.jitter_warmup_steps))
            return float(self._jitter_start) * (1.0 - t) + float(self.jitter_end) * t
        return float(self._jitter_start)

    def _collect_router_entropy(self) -> torch.Tensor:
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        logits = getattr(moe, "last_router_logits", None)
        if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
            return torch.zeros((), device=next(self.parameters()).device)

        n_experts = logits.shape[-1]
        probs = torch.softmax(logits.float(), dim=-1)  # [N, E]
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean()
        ent = ent / float(math.log(n_experts + 1e-9))
        return ent.to(dtype=logits.dtype, device=logits.device)

    def _compute_loss(self, logits, labels) -> Dict[str, Any]:
        if labels is None:
            return {
                "loss": None,
                "logits": logits,
                "aux_loss": None,
                "loss_main": None,
                "loss_lambda": None,
                "loss_total": None,
            }

        if self.loss_type == "ce":
            loss_main = F.cross_entropy(logits, labels)
        elif self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_main = F.cross_entropy(logits, labels, weight=w)
        elif self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            loss_main = loss_fn(logits, labels)
        else:
            raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")

        aux = self._collect_aux_loss()
        aux_w = self._aux_weight_now()
        loss_lambda = aux_w * aux

        entropy = self._collect_router_entropy()
        loss_entropy = float(self.router_entropy_weight) * entropy

        loss_total = loss_main + loss_lambda + loss_entropy

        return {
            "loss": loss_total,
            "logits": logits,
            "aux_loss": aux,
            "router_entropy": entropy,
            "loss_main": loss_main,
            "loss_lambda": loss_lambda,
            "loss_entropy": loss_entropy,
            "loss_total": loss_total,
        }

    def configure_topk_schedule(self, *, enabled: bool, start_k: int, end_k: int, switch_epoch: int) -> None:
        self._topk_schedule_enabled = bool(enabled)
        self._topk_start = int(start_k)
        self._topk_end = int(end_k)
        self._topk_switch_epoch = int(switch_epoch)

        if self._topk_schedule_enabled:
            self.encoder.moe_ffn.set_top_k(self._topk_start)
        else:
            self.encoder.moe_ffn.set_top_k(self._topk_end)

    def set_epoch(self, epoch_idx_0based: int) -> None:
        if not getattr(self, "_topk_schedule_enabled", False):
            return
        k = self._topk_start if epoch_idx_0based < self._topk_switch_epoch else self._topk_end
        self.encoder.moe_ffn.set_top_k(k)
