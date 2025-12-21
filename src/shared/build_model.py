from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from typing import Optional, Dict, Sequence, Union


def build_head(head_type: str, in_dim: int, num_labels: int, dropout: float) -> nn.Module:
    head_type = head_type.lower().strip()
    if head_type in {"linear", "lin"}:
        return LinearHead(in_dim, num_labels, dropout)
    if head_type in {"mlp", "2layer", "two_layer"}:
        return MLPHead(in_dim, num_labels, dropout)
    raise ValueError(f"Unsupported head_type: {head_type}. Use 'linear' or 'mlp'.")


class LinearHead(nn.Module):
    """
    Linear head with LayerNorm + Dropout for stability.
    """
    def __init__(self, in_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, dropout: float):
        super().__init__()
        hidden = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Args:
        gamma: focusing parameter
        alpha: optional class weights (same semantics as CrossEntropy weight)
        reduction: "mean" | "sum" | "none"
    """
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per sample
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )  # [B]

        # pt = P(correct class)
        pt = torch.exp(-ce).clamp_min(1e-8)  # [B]
        loss = ((1.0 - pt) ** self.gamma) * ce  # [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BaseBertConcatClassifier(nn.Module):
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
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

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

        # Loss config
        self.loss_type = loss_type.lower().strip()

        cw: Optional[torch.Tensor]
        if class_weights is None:
            cw = None
        elif isinstance(class_weights, torch.Tensor):
            cw = class_weights.detach().float()
        else:
            cw = torch.tensor(list(class_weights), dtype=torch.float)

        self.register_buffer("class_weights", cw if cw is not None else None)

        self.focal_gamma = float(focal_gamma)

        if self.loss_type not in {"ce", "weighted_ce", "focal"}:
            raise ValueError("loss_type must be one of: ce, weighted_ce, focal")

        if self.loss_type in {"weighted_ce", "focal"} and self.class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce or focal")

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
            logits = self.head_single(self.dropout(cls_sent))

        elif fusion_method == "term":
            logits = self.head_single(self.dropout(cls_term))

        elif fusion_method == "concat":
            logits = self.head_concat(self.dropout(torch.cat([cls_sent, cls_term], dim=-1)))

        elif fusion_method == "add":
            logits = self.head_single(self.dropout(cls_sent + cls_term))

        elif fusion_method == "mul":
            logits = self.head_single(self.dropout(cls_sent * cls_term))

        elif fusion_method == "cross":
            q = out_term.last_hidden_state[:, 0:1, :]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm)
            logits = self.head_single(self.dropout(attn_out.squeeze(1)))

        elif fusion_method == "gated_concat":
            g = torch.sigmoid(self.gate(torch.cat([cls_sent, cls_term], dim=-1)))
            fused = g * cls_sent + (1 - g) * cls_term
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "bilinear":
            fused = self.bilinear_out(
                self.bilinear_proj_sent(cls_sent) * self.bilinear_proj_term(cls_term)
            )
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "coattn":
            q_term = out_term.last_hidden_state[:, 0:1, :]
            q_sent = out_sent.last_hidden_state[:, 0:1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term_ctx, _ = self.coattn_term_to_sent(q_term, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_sent)
            sent_ctx, _ = self.coattn_sent_to_term(q_sent, out_term.last_hidden_state, out_term.last_hidden_state, key_padding_mask=kpm_term)

            logits = self.head_single(self.dropout(term_ctx.squeeze(1) + sent_ctx.squeeze(1)))

        elif fusion_method == "late_interaction":
            sent_tok = out_sent.last_hidden_state  # [B, Ls, H]
            term_tok = out_term.last_hidden_state  # [B, Lt, H]

            sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
            term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))  # [B, Lt, Ls]

            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)  # [B, 1, Ls]
                sim = sim.masked_fill(mask, -1e9)

            max_sim = sim.max(dim=-1).values  # [B, Lt]

            if attention_mask_term is not None:
                term_valid = attention_mask_term.float()
                denom = term_valid.sum(dim=1).clamp_min(1.0)
                pooled = (max_sim * term_valid).sum(dim=1) / denom  # [B]
            else:
                pooled = max_sim.mean(dim=1)

            cond = self.gate(torch.cat([cls_sent, cls_term], dim=-1))  # [B, H]
            fused = cond * pooled.unsqueeze(-1)
            logits = self.head_single(self.dropout(fused))

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        return self._compute_loss(logits, labels)

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

        return {"loss": loss, "logits": logits}
