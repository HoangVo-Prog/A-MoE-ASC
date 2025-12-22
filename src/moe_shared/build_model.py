import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Sequence, Union
import torch.nn.functional as F


from .config import MoEConfig
from shared import BaseBertConcatClassifier, FocalLoss


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

class MoEBertConcatClassifier(BaseBertConcatClassifier):
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
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
            focal_gamma=focal_gamma
        )

        self.aux_loss_weight = aux_loss_weight

    def _collect_aux_loss(self):
        total, count = 0.0, 0
        for layer in self.encoder.encoder.layer:
            moe = getattr(layer, "moe_ffn", None)
            if moe is None or moe.last_router_logits is None:
                continue
            total += moe_load_balance_loss(
                moe.last_router_logits, moe.last_topk_idx, moe.moe_cfg.num_experts
            )
            count += 1
        return total / count if count > 0 else torch.tensor(0.0, device=self.device)

    def _compute_loss(self, logits, labels):
        if labels is None:
            return {"loss": None, "logits": logits, "aux_loss": None}

        ce = nn.CrossEntropyLoss()(logits, labels)
        aux = self._collect_aux_loss()
        return {
            "loss": ce + self.aux_loss_weight * aux,
            "logits": logits,
            "aux_loss": aux,
        }
        
    def _compute_loss(self, logits, labels):
        if labels is None:
            return {"loss": None, "logits": logits}

        if self.loss_type == "ce":
            loss = F.cross_entropy(logits, labels)

        elif self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, labels, weight=w)

        elif self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            loss = loss_fn(logits, labels)

        else:
            raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")
        
        aux = self._collect_aux_loss()
        return {
            "loss": loss + self.aux_loss_weight * aux,
            "logits": logits,
            "aux_loss": aux,
        }

        
    @torch.no_grad()
    def _moe_debug_stats_per_layer(self):

        stats = []
        E = None

        for li, layer in enumerate(self.encoder.encoder.layer):
            moe = getattr(layer, "moe_ffn", None)
            if moe is None:
                continue
            if moe.last_router_logits is None or moe.last_topk_idx is None:
                continue

            logits = moe.last_router_logits
            topk_idx = moe.last_topk_idx
            E = moe.moe_cfg.num_experts

            if logits.dim() == 3:
                logits2 = logits.reshape(-1, logits.size(-1))
            else:
                logits2 = logits

            if topk_idx.dim() == 3:
                topk2 = topk_idx.reshape(-1, topk_idx.size(-1))
            else:
                topk2 = topk_idx

            probs = torch.softmax(logits2, dim=-1)

            eps = 1e-9
            ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
            ent_norm = ent / math.log(E)

            counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk2.reshape(-1),
                torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
            )
            usage = counts / counts.sum().clamp_min(1.0)

            stats.append(
                {
                    "layer": li,
                    "entropy_norm": float(ent_norm.item()),
                    "max_load": float(usage.max().item()),
                    "min_load": float(usage.min().item()),
                    "usage": usage.detach().cpu(),
                }
            )

        return stats

    def print_moe_debug(self, topn: int = 3):
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            print("[MoE] No stats yet (maybe first batch not run or last_router_logits missing).")
            return
        print()
        for s in stats:
            usage = s["usage"]
            topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
            top_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])
            print(
                f"[MoE][layer {s['layer']}] entropy_norm={s['entropy_norm']:.3f} "
                f"max={s['max_load']:.3f} min={s['min_load']:.3f} | top: {top_pairs}"
            )
        moe = getattr(self.encoder, "moe_ffn", None)
        print(moe)
        cur_k = getattr(moe, "moe_top_k", None)
        print(f"[MoE] top_k={cur_k} ...")

