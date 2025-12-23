from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from shared import DEVICE

from .moe import SeqMoELogits, SeqMoELogitsConfig
from moe_shared import moe_load_balance_loss



@dataclass
class SkBetaSchedule:
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 0  # 0 means constant beta_end


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce).clamp_min(1e-8)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_head(head_type: str, in_dim: int, num_labels: int, dropout: float) -> nn.Module:
    head_type = head_type.lower().strip()
    if head_type in {"linear", "lin"}:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_labels),
        )
    if head_type in {"mlp", "2layer", "two_layer"}:
        hidden = in_dim
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
    raise ValueError(f"Unsupported head_type: {head_type}. Use 'linear' or 'mlp'.")


class EncoderWithSkMoE(nn.Module):
    """Wrap encoder and expose MoE modules under names containing moe_ffn."""

    def __init__(self, *, base_encoder: nn.Module, moe_ffn_h: SeqMoELogits, moe_ffn_2h: SeqMoELogits):
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffn_h = moe_ffn_h
        self.moe_ffn_2h = moe_ffn_2h

    def forward(self, *args: Any, **kwargs: Any):
        return self.base_encoder(*args, **kwargs)


class SkBertConcatClassifier(nn.Module):
    """Baseline 10-method classifier with sequence-level MoE logits residual.

    MoE router input is aligned with current fusion_method.
    Two MoE modules are used:
      - moe_ffn_h: expects [B, H]
      - moe_ffn_2h: expects [B, 2H] (for concat)
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights: Optional[Union[torch.Tensor, Sequence[float], str]] = None,
        focal_gamma: float = 2.0,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_router_bias: bool = True,
        moe_router_jitter: float = 0.0,
        beta_schedule: Optional[SkBetaSchedule] = None,
        aux_loss_weight: float = 0.0,  # lambda cho MoE loss
    ) -> None:
        super().__init__()
        base = AutoModel.from_pretrained(model_name)
        hidden_size = int(base.config.hidden_size)

        _candidates = [8, 4, 2, 1]
        num_heads = next((x for x in _candidates if hidden_size % x == 0), 1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.coattn_term_to_sent = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.coattn_sent_to_term = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Linear(2 * hidden_size, hidden_size)

        bilinear_rank = max(32, min(256, hidden_size // 4))
        self.bilinear_proj_sent = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_proj_term = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_out = nn.Linear(bilinear_rank, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.head_single = build_head(head_type, hidden_size, num_labels, dropout)
        self.head_concat = build_head(head_type, 2 * hidden_size, num_labels, dropout)

        moe_h = SeqMoELogits(
            SeqMoELogitsConfig(
                in_dim=hidden_size,
                num_labels=num_labels,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
                dropout_p=float(dropout),
                router_bias=bool(moe_router_bias),
                router_jitter=float(moe_router_jitter),
            )
        )
        moe_2h = SeqMoELogits(
            SeqMoELogitsConfig(
                in_dim=2 * hidden_size,
                num_labels=num_labels,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
                dropout_p=float(dropout),
                router_bias=bool(moe_router_bias),
                router_jitter=float(moe_router_jitter),
            )
        )

        self.encoder = EncoderWithSkMoE(
            base_encoder=base,
            moe_ffn_h=moe_h,
            moe_ffn_2h=moe_2h,
        )

        self.loss_type = loss_type.lower().strip()
        cw: Optional[torch.Tensor]
        if class_weights is None:
            cw = None
        elif isinstance(class_weights, torch.Tensor):
            cw = class_weights.detach().float()
        elif isinstance(class_weights, str):
            s = class_weights.strip()
            if not s:
                cw = None
            else:
                cw = torch.tensor([float(x.strip()) for x in s.split(",") if x.strip()], dtype=torch.float)
        else:
            cw = torch.tensor([float(x) for x in class_weights], dtype=torch.float)

        self.register_buffer("class_weights", cw if cw is not None else None)
        self.focal_gamma = float(focal_gamma)

        if self.loss_type not in {"ce", "weighted_ce", "focal"}:
            raise ValueError("loss_type must be one of: ce, weighted_ce, focal")
        if self.loss_type in {"weighted_ce", "focal"} and self.class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce or focal")

        self.beta_schedule = beta_schedule or SkBetaSchedule()
        self._beta_current: float = float(self.beta_schedule.beta_end)

        self.aux_loss_weight: float = float(aux_loss_weight)

        self.device = DEVICE

        self._last_loss_main = None
        self._last_loss_moe = None
        self._last_loss_moe_weighted = None
        self._last_loss_ratio = None

    def set_epoch(self, epoch_idx_0based: int) -> None:
        sch = self.beta_schedule
        warm = int(sch.beta_warmup_epochs)
        if warm <= 0:
            self._beta_current = float(sch.beta_end)
            return

        e = int(epoch_idx_0based)
        if e <= 0:
            self._beta_current = float(sch.beta_start)
            return
        if e >= warm:
            self._beta_current = float(sch.beta_end)
            return

        t = e / float(warm)
        self._beta_current = float(sch.beta_start + (sch.beta_end - sch.beta_start) * t)

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ) -> Dict[str, torch.Tensor]:

        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)

        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]

        fusion_method = fusion_method.lower().strip()

        if fusion_method == "sent":
            rep = cls_sent
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "term":
            rep = cls_term
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "concat":
            rep = torch.cat([cls_sent, cls_term], dim=-1)
            logits_base = self.head_concat(self.dropout(rep))

        elif fusion_method == "add":
            rep = cls_sent + cls_term
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "mul":
            rep = cls_sent * cls_term
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "cross":
            q = out_term.last_hidden_state[:, 0:1, :]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(
                q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
            )
            rep = attn_out.squeeze(1)
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "gated_concat":
            g = torch.sigmoid(self.gate(torch.cat([cls_sent, cls_term], dim=-1)))
            rep = g * cls_sent + (1 - g) * cls_term
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "bilinear":
            rep = self.bilinear_out(self.bilinear_proj_sent(cls_sent) * self.bilinear_proj_term(cls_term))
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "coattn":
            q_term = out_term.last_hidden_state[:, 0:1, :]
            q_sent = out_sent.last_hidden_state[:, 0:1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term_ctx, _ = self.coattn_term_to_sent(
                q_term, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_sent
            )
            sent_ctx, _ = self.coattn_sent_to_term(
                q_sent, out_term.last_hidden_state, out_term.last_hidden_state, key_padding_mask=kpm_term
            )

            rep = term_ctx.squeeze(1) + sent_ctx.squeeze(1)
            logits_base = self.head_single(self.dropout(rep))

        elif fusion_method == "late_interaction":
            sent_tok = out_sent.last_hidden_state
            term_tok = out_term.last_hidden_state

            sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
            term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))

            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)
                sim = sim.masked_fill(mask, -1e9)

            max_sim = sim.max(dim=-1).values

            if attention_mask_term is not None:
                term_valid = attention_mask_term.float()
                denom = term_valid.sum(dim=1).clamp_min(1.0)
                pooled = (max_sim * term_valid).sum(dim=1) / denom
            else:
                pooled = max_sim.mean(dim=1)

            cond = self.gate(torch.cat([cls_sent, cls_term], dim=-1))
            rep = cond * pooled.unsqueeze(-1)
            logits_base = self.head_single(self.dropout(rep))

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        beta = float(self._beta_current)

        moe_mod = self.encoder.moe_ffn_2h if fusion_method == "concat" else self.encoder.moe_ffn_h

        if beta == 0.0:
            delta = None
            logits = logits_base
        else:
            delta = moe_mod(rep)
            logits = logits_base + (beta * delta)

        return self._compute_loss(logits, labels)

    def _compute_loss(self, logits, labels):
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
        loss_lambda = self.aux_loss_weight * aux

        loss_total = loss_main + loss_lambda

        return {
            "loss": loss_total,          
            "logits": logits,            
            "aux_loss": aux,            
            "loss_main": loss_main,      
            "loss_lambda": loss_lambda,  
            "loss_total": loss_total,    
        }

    def print_moe_debug(self, topn: int = 3) -> None:
        s_h = self.encoder.moe_ffn_h.debug_stats()
        s_2h = self.encoder.moe_ffn_2h.debug_stats()

        if not s_h and not s_2h:
            print("[MoE][sk] No stats yet.")
            return

        def _fmt_top(vec: torch.Tensor, n: int) -> str:
            topv, topi = torch.topk(vec, k=min(n, vec.numel()))
            return ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])

        if s_h:
            usage = s_h["usage_soft"]
            frac = s_h["topk_frac"]
            print(f"[MoE][sk][H] k={s_h['moe_top_k']} soft: {_fmt_top(usage, topn)} topk: {_fmt_top(frac, topn)}")

        if s_2h:
            usage = s_2h["usage_soft"]
            frac = s_2h["topk_frac"]
            print(f"[MoE][sk][2H] k={s_2h['moe_top_k']} soft: {_fmt_top(usage, topn)} topk: {_fmt_top(frac, topn)}")

        print(f"[MoE][sk] beta={self._beta_current:.4f} lambda={self.aux_loss_weight:.6f}")


def build_model(cfg: Any, moe_cfg: Optional[Any], num_labels: int) -> nn.Module:
    # Scaffold uses cfg fields that exist in moe_head and shared configs.
    beta = SkBetaSchedule(
        beta_start=float(getattr(cfg, "sk_beta_start", 0.0)),
        beta_end=float(getattr(cfg, "sk_beta_end", 1.0)),
        beta_warmup_epochs=int(getattr(cfg, "sk_beta_warmup_epochs", 0)),
    )

    model = SkBertConcatClassifier(
        model_name=str(getattr(cfg, "model_name")),
        num_labels=int(num_labels),
        dropout=float(getattr(cfg, "dropout")),
        head_type=str(getattr(cfg, "head_type")),
        loss_type=str(getattr(cfg, "loss_type", "ce")),
        class_weights=getattr(cfg, "class_weights", None),
        focal_gamma=float(getattr(cfg, "focal_gamma", 2.0)),
        moe_num_experts=int(getattr(moe_cfg, "num_experts", 4)) if moe_cfg is not None else 4,
        moe_top_k=int(getattr(moe_cfg, "moe_top_k", 2)) if moe_cfg is not None else 2,
        moe_router_bias=bool(getattr(moe_cfg, "router_bias", True)) if moe_cfg is not None else True,
        moe_router_jitter=float(getattr(moe_cfg, "router_jitter", 0.0)) if moe_cfg is not None else 0.0,
        beta_schedule=beta,
    )
    return model.to(device=DEVICE)
