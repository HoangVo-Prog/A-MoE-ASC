from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import BaseBertConcatClassifier


class MoFRouter(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        num_experts: int,
        hidden_dim: int,
        dropout: float,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature) if float(temperature) > 0 else 1.0
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_experts)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x) / self.temperature
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        return weights, logits


def _mha_forward(
    attn: nn.MultiheadAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    batch_first = bool(getattr(attn, "batch_first", False))
    if batch_first:
        out, _ = attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return out
    q2 = q.transpose(0, 1)
    k2 = k.transpose(0, 1)
    v2 = v.transpose(0, 1)
    out2, _ = attn(q2, k2, v2, key_padding_mask=key_padding_mask, need_weights=False)
    return out2.transpose(0, 1)


class _MoFBase:
    mof_experts: List[str]
    mof_router: MoFRouter

    def _init_mof(
        self,
        *,
        dropout: float,
        include_sent_term: bool,
        experts: Optional[List[str]] = None,
    ) -> None:
        encoder = getattr(self, "encoder", None)
        if encoder is None or getattr(encoder, "config", None) is None:
            raise RuntimeError("MoF requires self.encoder with config.hidden_size")
        h = int(encoder.config.hidden_size)

        if experts is not None:
            self.mof_experts = list(experts)
        else:
            base = [
                "concat",
                "add",
                "mul",
                "gated_concat",
                "bilinear",
                "cross",
                "coattn",
                "late_interaction",
            ]
            if bool(include_sent_term):
                # add as extra experts, can be toggled via flag
                base = ["sent", "term"] + base
            self.mof_experts = base

        self.mof_router = MoFRouter(
            in_dim=4 * h,
            num_experts=len(self.mof_experts),
            hidden_dim=h,
            dropout=float(dropout),
            temperature=1.0,
        )

    def _mof_logits(
        self,
        *,
        out_sent,
        out_term,
        cls_sent: torch.Tensor,
        cls_term: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        attention_mask_term: torch.Tensor,
    ) -> List[torch.Tensor]:
        dropout = getattr(self, "dropout", nn.Identity())
        head_single = getattr(self, "head_single", None)
        head_concat = getattr(self, "head_concat", None)
        if head_single is None or head_concat is None:
            raise RuntimeError("MoF expects head_single and head_concat from BaseBertConcatClassifier")

        sent_tokens = out_sent.last_hidden_state
        term_tokens = out_term.last_hidden_state

        expert_logits: List[torch.Tensor] = []

        for name in self.mof_experts:
            if name == "sent":
                expert_logits.append(head_single(dropout(cls_sent)))
                continue

            if name == "term":
                expert_logits.append(head_single(dropout(cls_term)))
                continue

            if name == "concat":
                fused = torch.cat([cls_sent, cls_term], dim=-1)
                expert_logits.append(head_concat(dropout(fused)))
                continue

            if name == "add":
                fused = cls_sent + cls_term
                expert_logits.append(head_single(dropout(fused)))
                continue

            if name == "mul":
                fused = cls_sent * cls_term
                expert_logits.append(head_single(dropout(fused)))
                continue

            if name == "gated_concat":
                gate = getattr(self, "gate", None)
                if gate is None:
                    raise RuntimeError("gated_concat expert requires self.gate")
                g = torch.sigmoid(gate(torch.cat([cls_sent, cls_term], dim=-1)))
                fused = g * cls_sent + (1.0 - g) * cls_term
                expert_logits.append(head_single(dropout(fused)))
                continue

            if name == "bilinear":
                ps = getattr(self, "bilinear_proj_sent", None)
                pt = getattr(self, "bilinear_proj_term", None)
                bo = getattr(self, "bilinear_out", None)
                if ps is None or pt is None or bo is None:
                    raise RuntimeError("bilinear expert requires bilinear_proj_sent, bilinear_proj_term, bilinear_out")
                fused = bo(ps(cls_sent) * pt(cls_term))
                expert_logits.append(head_single(dropout(fused)))
                continue

            if name == "cross":
                cross_attn = getattr(self, "cross_attn", None)
                if cross_attn is None:
                    raise RuntimeError("cross expert requires self.cross_attn")

                q = cls_term.unsqueeze(1)
                k = sent_tokens
                v = sent_tokens
                kpm = (attention_mask_sent == 0) if attention_mask_sent is not None else None

                out = _mha_forward(cross_attn, q, k, v, kpm)
                fused = out.squeeze(1)
                expert_logits.append(head_single(dropout(fused)))
                continue

            if name == "coattn":
                attn_t2s = getattr(self, "coattn_term_to_sent", None)
                attn_s2t = getattr(self, "coattn_sent_to_term", None)
                if attn_t2s is None or attn_s2t is None:
                    raise RuntimeError("coattn expert requires coattn_term_to_sent and coattn_sent_to_term")

                q_term = cls_term.unsqueeze(1)
                q_sent = cls_sent.unsqueeze(1)

                kpm_sent = (attention_mask_sent == 0) if attention_mask_sent is not None else None
                kpm_term = (attention_mask_term == 0) if attention_mask_term is not None else None

                ctx_t = _mha_forward(attn_t2s, q_term, sent_tokens, sent_tokens, kpm_sent).squeeze(1)
                ctx_s = _mha_forward(attn_s2t, q_sent, term_tokens, term_tokens, kpm_term).squeeze(1)
                fused = ctx_t + ctx_s
                expert_logits.append(head_single(dropout(fused)))
                continue

            if name == "late_interaction":
                gate = getattr(self, "gate", None)
                if gate is None:
                    raise RuntimeError("late_interaction expert requires self.gate")

                sent_norm = F.normalize(sent_tokens, p=2, dim=-1)
                term_norm = F.normalize(term_tokens, p=2, dim=-1)

                sim = torch.matmul(term_norm, sent_norm.transpose(1, 2))
                if attention_mask_sent is not None:
                    m_sent = attention_mask_sent.to(dtype=torch.bool).unsqueeze(1)
                    sim = sim.masked_fill(~m_sent, -1e9)

                max_sim = sim.max(dim=-1).values
                if attention_mask_term is not None:
                    m_term = attention_mask_term.to(dtype=max_sim.dtype)
                    pooled = (max_sim * m_term).sum(dim=-1) / m_term.sum(dim=-1).clamp_min(1.0)
                else:
                    pooled = max_sim.mean(dim=-1)

                cond = torch.sigmoid(gate(torch.cat([cls_sent, cls_term], dim=-1)))
                fused = cond * pooled.unsqueeze(-1)
                expert_logits.append(head_single(dropout(fused)))
                continue

            raise RuntimeError(f"Unknown MoF expert: {name}")

        return expert_logits

    def _forward_mof(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor],
    ):
        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)

        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]

        expert_logits = self._mof_logits(
            out_sent=out_sent,
            out_term=out_term,
            cls_sent=cls_sent,
            cls_term=cls_term,
            attention_mask_sent=attention_mask_sent,
            attention_mask_term=attention_mask_term,
        )

        router_x = torch.cat(
            [cls_sent, cls_term, cls_sent * cls_term, torch.abs(cls_sent - cls_term)],
            dim=-1,
        )
        weights, _ = self.mof_router(router_x)

        logits_stack = torch.stack(expert_logits, dim=1)  # [B, E, C]
        logits = torch.sum(logits_stack * weights.unsqueeze(-1), dim=1)  # [B, C]

        return self._compute_loss(logits, labels)


class MoFBertConcatClassifier(_MoFBase, BaseBertConcatClassifier):
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights=None,
        focal_gamma: float = 2.0,
        mof_include_sent_term: bool = False,
        mof_experts: Optional[List[str]] = None,
    ) -> None:
        head_type_in = str(head_type).lower()
        base_head_type = "linear" if head_type_in == "mof" else str(head_type)

        super().__init__(
            model_name=model_name,
            num_labels=int(num_labels),
            dropout=float(dropout),
            head_type=base_head_type,
            loss_type=str(loss_type),
            class_weights=class_weights,
            focal_gamma=float(focal_gamma),
        )

        self._use_mof = True
        self._init_mof(
            dropout=float(dropout),
            include_sent_term=bool(mof_include_sent_term),
            experts=mof_experts,
        )

    def forward(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ):
        if bool(getattr(self, "_use_mof", False)) or str(fusion_method).lower() == "mof":
            return self._forward_mof(
                input_ids_sent=input_ids_sent,
                attention_mask_sent=attention_mask_sent,
                input_ids_term=input_ids_term,
                attention_mask_term=attention_mask_term,
                labels=labels,
            )

        return super().forward(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )
