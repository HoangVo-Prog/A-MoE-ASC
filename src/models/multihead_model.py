import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel

from moehead_model import MoE, _extract_token_mask_from_attention_mask


class EncoderWithMultiMoEHead(nn.Module):
    """
    Wraps a base encoder and applies multiple MoE FFN blocks sequentially
    on the last hidden states.

    Example: 2-layer head -> MoE1 then MoE2.
    """

    def __init__(self, *, base_encoder: nn.Module, moe_ffns: Sequence[MoE]) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffns = nn.ModuleList(list(moe_ffns))

    def forward(self, *args, **kwargs):
        outputs = self.base_encoder(*args, **kwargs)
        token_mask = _extract_token_mask_from_attention_mask(kwargs.get("attention_mask", None))

        def _apply_stack(x: torch.Tensor) -> torch.Tensor:
            for moe in self.moe_ffns:
                x = moe(x, token_mask=token_mask)
            return x

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            new_hidden = _apply_stack(last_hidden)
            return (new_hidden,) + tuple(outputs[1:])

        if hasattr(outputs, "last_hidden_state"):
            outputs.last_hidden_state = _apply_stack(outputs.last_hidden_state)
            return outputs

        return _apply_stack(outputs)


class MultiMoEHead(BaseModel):
    """
    Multi-MoE Head (stacked MoE layers applied on encoder output).
    This is the 2-layer variant requested, but supports any n_layers >= 1.

    Notes:
    - Aux loss is summed across all MoE layers.
    - Router entropy regularization is averaged across layers (then weighted).
    - Jitter + top-k schedules are applied to every layer.
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
        moe_n_layers: int = 2,
        moe_topk_schedule: bool = False,
        moe_topk_start: int = 1,
        moe_topk_end: int = 1,
        moe_topk_switch_epoch: int = 0,
        router_bias: bool = True,
        router_jitter: float = 0.0,
        capacity_factor=None,
        route_mask_pad_tokens: bool = True,
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

        n_layers = int(moe_n_layers)
        if n_layers < 1:
            n_layers = 1

        moe_layers = []
        for _ in range(n_layers):
            moe_layers.append(
                MoE(
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
            )

        self._jitter_start = float(router_jitter)

        base_encoder = self.encoder
        self.encoder = EncoderWithMultiMoEHead(base_encoder=base_encoder, moe_ffns=moe_layers)

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
            enc = getattr(self, "encoder", None)
            moe_ffns = getattr(enc, "moe_ffns", None)
            if moe_ffns is not None:
                for moe in moe_ffns:
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
        enc = getattr(self, "encoder", None)
        moe_ffns = getattr(enc, "moe_ffns", None)
        if moe_ffns is None or len(moe_ffns) == 0:
            return torch.zeros((), device=next(self.parameters()).device)

        aux_sum = torch.zeros((), device=next(self.parameters()).device)
        for moe in moe_ffns:
            logits = getattr(moe, "last_router_logits", None)
            topk_idx = getattr(moe, "last_topk_idx", None)
            if logits is None or topk_idx is None:
                continue
            if logits.ndim != 2 or topk_idx.ndim != 2 or logits.shape[0] == 0:
                continue

            n_tokens, n_experts = logits.shape
            k = topk_idx.shape[1]

            probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)  # [N, E]
            importance = probs.sum(dim=0) / float(n_tokens)  # [E]

            oh = F.one_hot(topk_idx, num_classes=n_experts).to(dtype=probs.dtype)  # [N, K, E]
            load = oh.sum(dim=(0, 1)) / float(n_tokens * k)  # [E]

            aux = n_experts * torch.sum(importance * load)
            aux_sum = aux_sum + torch.clamp(aux, min=0.0, max=10.0)

        return aux_sum

    def _collect_router_entropy(self) -> torch.Tensor:
        enc = getattr(self, "encoder", None)
        moe_ffns = getattr(enc, "moe_ffns", None)
        if moe_ffns is None or len(moe_ffns) == 0:
            return torch.zeros((), device=next(self.parameters()).device)

        ents = []
        for moe in moe_ffns:
            logits = getattr(moe, "last_router_logits", None)
            if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
                continue
            n_experts = logits.shape[-1]
            probs = torch.softmax(logits.float(), dim=-1)  # [N, E]
            ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean()
            ent = ent / float(math.log(n_experts + 1e-9))
            ents.append(ent.to(dtype=logits.dtype, device=logits.device))

        if not ents:
            return torch.zeros((), device=next(self.parameters()).device)

        return torch.stack(ents, dim=0).mean()

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

        enc = getattr(self, "encoder", None)
        moe_ffns = getattr(enc, "moe_ffns", None)
        if moe_ffns is None:
            return

        k0 = self._topk_start if self._topk_schedule_enabled else self._topk_end
        for moe in moe_ffns:
            moe.set_top_k(k0)

    def set_epoch(self, epoch_idx_0based: int) -> None:
        if not getattr(self, "_topk_schedule_enabled", False):
            return

        k = self._topk_start if epoch_idx_0based < self._topk_switch_epoch else self._topk_end

        enc = getattr(self, "encoder", None)
        moe_ffns = getattr(enc, "moe_ffns", None)
        if moe_ffns is None:
            return

        for moe in moe_ffns:
            moe.set_top_k(k)
