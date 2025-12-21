from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from moe_ffn.config import MoEConfig
from moe_ffn.moe import MoEClassifierHead, moe_load_balance_loss
from shared import DEVICE, BaseBertConcatClassifier


class BertConcatClassifier(BaseBertConcatClassifier):
    """BERT/RoBERTa + (optional) MoE classifier heads.

    This version does NOT modify the encoder (no encoder-FFN MoE).
    MoE is applied only at the classification head level.
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        moe_cfg: Optional[MoEConfig],
        aux_loss_weight: float,
        freeze_moe: bool,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
        )

        self.aux_loss_weight = float(aux_loss_weight)
        self.moe_cfg = moe_cfg

        hidden_size = int(self.encoder.config.hidden_size)

        self.moe_head_single: Optional[MoEClassifierHead] = None
        self.moe_head_concat: Optional[MoEClassifierHead] = None

        if moe_cfg is not None:
            self.moe_head_single = MoEClassifierHead(
                in_dim=hidden_size,
                num_labels=num_labels,
                moe_cfg=moe_cfg,
                dropout=dropout,
            )
            self.moe_head_concat = MoEClassifierHead(
                in_dim=2 * hidden_size,
                num_labels=num_labels,
                moe_cfg=moe_cfg,
                dropout=dropout,
            )

            if freeze_moe:
                for p in self.moe_head_single.parameters():
                    p.requires_grad = False
                for p in self.moe_head_concat.parameters():
                    p.requires_grad = False

    def _collect_aux_loss(self) -> torch.Tensor:
        if self.moe_cfg is None:
            return torch.tensor(0.0, device=DEVICE)

        total = torch.tensor(0.0, device=DEVICE)
        count = 0

        for head in (self.moe_head_single, self.moe_head_concat):
            if head is None:
                continue
            if head.last_router_logits is None or head.last_topk_idx is None:
                continue
            total = total + moe_load_balance_loss(
                head.last_router_logits,
                head.last_topk_idx,
                head.moe_cfg.num_experts,
            )
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=DEVICE)
        return total / float(count)

    def _compute_loss(self, logits: torch.Tensor, labels: Optional[torch.Tensor]):
        if labels is None:
            return {"loss": None, "logits": logits, "aux_loss": None}

        ce = nn.CrossEntropyLoss()(logits, labels)
        aux = self._collect_aux_loss()
        return {
            "loss": ce + self.aux_loss_weight * aux,
            "logits": logits,
            "aux_loss": aux,
        }

    @torch.no_grad()
    def _moe_debug_stats(self):
        if self.moe_cfg is None:
            return []

        stats = []
        E = int(self.moe_cfg.num_experts)

        for name, head in (("single", self.moe_head_single), ("concat", self.moe_head_concat)):
            if head is None:
                continue
            if head.last_router_logits is None or head.last_topk_idx is None:
                continue

            logits = head.last_router_logits  # [B,E]
            topk_idx = head.last_topk_idx     # [B,K]

            probs = torch.softmax(logits, dim=-1)

            eps = 1e-9
            ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
            ent_norm = ent / math.log(E)

            counts = torch.zeros(E, device=logits.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk_idx.reshape(-1),
                torch.ones_like(topk_idx.reshape(-1), dtype=torch.float32),
            )
            usage = counts / counts.sum().clamp_min(1.0)

            stats.append(
                {
                    "head": name,
                    "entropy_norm": float(ent_norm.item()),
                    "max_load": float(usage.max().item()),
                    "min_load": float(usage.min().item()),
                    "usage": usage.detach().cpu(),
                }
            )

        return stats

    def print_moe_debug(self, topn: int = 3):
        if self.moe_cfg is None:
            print("[MoE] Disabled (moe_cfg is None).")
            return

        stats = self._moe_debug_stats()
        if not stats:
            print("[MoE] No stats yet (maybe first batch not run or router cache missing).")
            return

        print()
        for s in stats:
            usage = s["usage"]
            topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
            top_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])
            print(
                f"[MoE][head {s['head']}] entropy_norm={s['entropy_norm']:.3f} "
                f"max={s['max_load']:.3f} min={s['min_load']:.3f} | top: {top_pairs}"
            )

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ):
        # Copies the base fusion logic, but swaps heads to MoE heads when enabled.

        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)

        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]

        fusion_method = fusion_method.lower().strip()

        if fusion_method == "sent":
            x = self.dropout(cls_sent)
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        elif fusion_method == "term":
            x = self.dropout(cls_term)
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        elif fusion_method == "concat":
            x = self.dropout(torch.cat([cls_sent, cls_term], dim=-1))
            logits = self.moe_head_concat(x) if self.moe_head_concat is not None else self.head_concat(x)

        elif fusion_method == "add":
            x = self.dropout(cls_sent + cls_term)
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        elif fusion_method == "mul":
            x = self.dropout(cls_sent * cls_term)
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        elif fusion_method == "cross":
            q = out_term.last_hidden_state[:, 0:1, :]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(
                q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
            )
            x = self.dropout(attn_out.squeeze(1))
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        elif fusion_method == "coattn":
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term2sent, _ = self.coattn_term_to_sent(
                out_term.last_hidden_state,
                out_sent.last_hidden_state,
                out_sent.last_hidden_state,
                key_padding_mask=kpm_sent,
            )
            sent2term, _ = self.coattn_sent_to_term(
                out_sent.last_hidden_state,
                out_term.last_hidden_state,
                out_term.last_hidden_state,
                key_padding_mask=kpm_term,
            )

            cls_a = term2sent[:, 0, :]
            cls_b = sent2term[:, 0, :]

            x = self.dropout(torch.cat([cls_a, cls_b], dim=-1))
            logits = self.moe_head_concat(x) if self.moe_head_concat is not None else self.head_concat(x)

        elif fusion_method == "gate":
            g = torch.sigmoid(self.gate(torch.cat([cls_sent, cls_term], dim=-1)))
            x = self.dropout(g * cls_sent + (1.0 - g) * cls_term)
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        elif fusion_method == "bilinear":
            a = self.bilinear_proj_sent(cls_sent)
            b = self.bilinear_proj_term(cls_term)
            x = self.dropout(self.bilinear_out(a * b))
            logits = self.moe_head_single(x) if self.moe_head_single is not None else self.head_single(x)

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        return self._compute_loss(logits, labels)


def build_model(*, cfg, moe_cfg, num_labels: int):
    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        head_type=cfg.head_type,
        moe_cfg=moe_cfg,
        aux_loss_weight=float(getattr(cfg, "aux_loss_weight", 0.01)),
        freeze_moe=bool(getattr(cfg, "freeze_moe", False)),
    ).to(DEVICE)
