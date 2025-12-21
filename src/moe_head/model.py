from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from moe_shared import MoEBertConcatClassifier, moe_load_balance_loss
from moe_head.moe import MoEHead
from moe_head.config import MultiMoEConfig as MoEConfig

from shared import (
    DEVICE,
)

class EncoderWithMoEHead(nn.Module):
    """Wrap a base encoder and apply MoEHead on last_hidden_state.

    The MoE module is exposed as attribute name `moe_ffn` so that utilities
    that look for "moe_ffn" in parameter names continue to work.
    """

    def __init__(self, *, base_encoder: nn.Module, moe_ffn: MoEHead) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffn = moe_ffn

    def forward(self, *args: Any, **kwargs: Any):
        outputs = self.base_encoder(*args, **kwargs)

        attn_mask = kwargs.get("attention_mask", None)
        token_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                token_mask = (attn_mask[:, 0, 0, :] == 0).long()
            else:
                token_mask = attn_mask

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            new_hidden = self.moe_ffn(last_hidden, token_mask=token_mask)
            return (new_hidden,) + tuple(outputs[1:])

        if hasattr(outputs, "last_hidden_state"):
            new_hidden = self.moe_ffn(outputs.last_hidden_state, token_mask=token_mask)
            outputs.last_hidden_state = new_hidden
            return outputs

        return self.moe_ffn(outputs, token_mask=token_mask)


class HeadBertConcatClassifier(MoEBertConcatClassifier):
    """moe_head model: encoder output, then MoE head, then classifier.

    Forward and main loss pipeline are inherited.
    This class overrides aux loss collection and debug to use head MoE.
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights: Optional[Any] = None,
        focal_gamma: float = 2.0,
        moe_cfg: MoEConfig,
        aux_loss_weight: float,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            loss_type=loss_type,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            moe_cfg=moe_cfg,
            aux_loss_weight=aux_loss_weight,
        )

        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn: nn.Module = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))
        
        self._topk_schedule_enabled = False
        self._topk_start = moe_cfg.moe_topk_start
        self._topk_end = moe_cfg.moe_topk_end
        self._topk_switch_epoch = moe_cfg.moe_topk_switch_epoch

        moe_head = MoEHead(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=moe_cfg.num_experts,
            top_k=moe_cfg.moe_top_k,
            dropout_p=dropout_p,
            act_fn=act_fn,
            router_bias=bool(getattr(moe_cfg, "router_bias", True)),
            router_jitter=float(getattr(moe_cfg, "router_jitter", 0.05)),
            capacity_factor=getattr(moe_cfg, "capacity_factor", None),
            route_mask_pad_tokens=bool(getattr(moe_cfg, "route_mask_pad_tokens", False)),
            layer_norm=nn.LayerNorm(hidden_size),
        )

        base_encoder = self.encoder
        self.encoder = EncoderWithMoEHead(base_encoder=base_encoder, moe_ffn=moe_head)


    def _collect_aux_loss(self):
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None or moe.last_router_logits is None or moe.last_topk_idx is None:
            return torch.tensor(0.0, device=self.device)
        return moe_load_balance_loss(
            moe.last_router_logits,
            moe.last_topk_idx,
            moe.num_experts,
        )

    @torch.no_grad()
    def _moe_debug_stats_head(self):
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None or moe.last_router_logits is None or moe.last_topk_idx is None:
            return None

        logits = moe.last_router_logits  # [N_active, E]
        topk_idx = moe.last_topk_idx     # [N_active, K]
        E = int(moe.num_experts)

        # A) Softmax usage over all experts (your current metric)
        probs = torch.softmax(logits, dim=-1)
        usage_soft = probs.mean(dim=0)  # [E]

        eps = 1e-9
        ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
        ent_norm = float(ent / math.log(E))

        # B) Router "is it learning" signals
        logits_f = logits.float()
        logits_std = float(logits_f.std().item())
        logits_absmean = float(logits_f.abs().mean().item())

        w_norm = float(moe.router.weight.detach().float().norm().item())
        b_norm = 0.0
        if moe.router.bias is not None:
            b_norm = float(moe.router.bias.detach().float().norm().item())

        # C) Actual selected expert histogram from topk indices
        flat = topk_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=E).float()  # [E]
        frac = counts / (counts.sum() + eps)                # [E]

        topk_ent = -(frac * (frac + eps).log()).sum()
        topk_ent_norm = float(topk_ent / math.log(E))

        return {
            # softmax based stats
            "entropy_norm": ent_norm,
            "max_load": float(usage_soft.max().item()),
            "min_load": float(usage_soft.min().item()),
            "usage": usage_soft.detach().cpu(),

            # learning signals
            "logits_std": logits_std,
            "logits_absmean": logits_absmean,
            "router_w_norm": w_norm,
            "router_b_norm": b_norm,

            # topk selection stats
            "topk_frac": frac.detach().cpu(),
            "topk_entropy_norm": topk_ent_norm,
            "topk_max": float(frac.max().item()),
            "topk_min": float(frac.min().item()),
        }


    def print_moe_debug(self, topn: int = 3):
        s = self._moe_debug_stats_head()
        if not s:
            print("[MoE] No stats yet (maybe first batch not run or missing caches).")
            return

        usage = s["usage"]
        topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
        top_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])

        topk_frac = s["topk_frac"]
        topkv, topki = torch.topk(topk_frac, k=min(topn, topk_frac.numel()))
        topk_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topkv, topki)])

        print()
        print(
            f"[MoE][head] softmax entropy_norm={s['entropy_norm']:.3f} "
            f"max={s['max_load']:.3f} min={s['min_load']:.3f} | top: {top_pairs}"
        )
        print(
            f"[MoE][head] logits_std={s['logits_std']:.4f} logits_absmean={s['logits_absmean']:.4f} "
            f"router_w_norm={s['router_w_norm']:.4f} router_b_norm={s['router_b_norm']:.4f}"
        )
        print(
            f"[MoE][head] topk entropy_norm={s['topk_entropy_norm']:.3f} "
            f"max={s['topk_max']:.3f} min={s['topk_min']:.3f} | topk: {topk_pairs}"
        )
        
        moe = getattr(self.encoder, "moe_ffn", None)
        cur_k = getattr(moe, "top_k", None)
        print(f"[MoE][head] top_k={cur_k} ...")


    def configure_topk_schedule(
        self,
        *,
        enabled: bool,
        start_k: int,
        end_k: int,
        switch_epoch: int,
    ) -> None:
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

        if epoch_idx_0based < self._topk_switch_epoch:
            k = self._topk_start
        else:
            k = self._topk_end

        self.encoder.moe_ffn.set_top_k(k)



def build_model(cfg, moe_cfg: Optional[MoEConfig], num_labels: int) -> nn.Module:
    assert moe_cfg is not None, "moe_head requires moe_cfg, but got None"
    model = HeadBertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        head_type=cfg.head_type,
        loss_type=cfg.loss_type,
        class_weights=cfg.class_weights,
        focal_gamma=cfg.focal_gamma,
        moe_cfg=moe_cfg,
        aux_loss_weight=cfg.aux_loss_weight,
    )
    return model.to(device=DEVICE)
