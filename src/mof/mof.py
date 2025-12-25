# mof.py
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
        # softmax in fp32 for stability under AMP
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


def _safe_get_loss(out: object) -> Optional[torch.Tensor]:
    if isinstance(out, dict):
        v = out.get("loss", None)
        return v if isinstance(v, torch.Tensor) else None
    return None


class _MoFBase:
    mof_experts: List[str]
    mof_router: MoFRouter

    def _init_mof(
        self,
        *,
        dropout: float,
        experts: Optional[List[str]] = None,
        router_temperature: float = 1.0,
        mix_level: str = "repr",  # repr or logit
        lb_coef: float = 0.0,
        disable_expert_scaling: bool = False,
        expert_norm_clamp: float = 0.0,
        logit_clamp: float = 0.0,
    ) -> None:
        encoder = getattr(self, "encoder", None)
        if encoder is None or getattr(encoder, "config", None) is None:
            raise RuntimeError("MoF requires self.encoder with config.hidden_size")
        h = int(encoder.config.hidden_size)

        if experts is not None:
            self.mof_experts = list(experts)
        else:
            self.mof_experts = [
                "concat",
                "add",
                "mul",
                "gated_concat",
                "bilinear",
                "cross",
                "coattn",
                "late_interaction",
            ]

        self.mof_router = MoFRouter(
            in_dim=4 * h,
            num_experts=len(self.mof_experts),
            hidden_dim=h,
            dropout=float(dropout),
            temperature=float(router_temperature),
        )

        # Project 2H concat into H so we can mix representations.
        self.mof_proj_concat = nn.Linear(2 * h, h)

        # Optional per expert scaling. Use log scale so scale is always positive.
        self._mof_disable_expert_scaling = bool(disable_expert_scaling)
        self.mof_expert_log_scale = nn.Parameter(torch.zeros(len(self.mof_experts)))

        self._mof_mix_level = str(mix_level).lower()
        if self._mof_mix_level not in ("repr", "logit"):
            self._mof_mix_level = "repr"

        self._mof_lb_coef = float(lb_coef)
        self._mof_expert_norm_clamp = float(expert_norm_clamp)
        self._mof_logit_clamp = float(logit_clamp)

    def _mof_debug_print(
        self,
        *,
        weights: torch.Tensor,                       # [B, E]
        expert_logits: Optional[List[torch.Tensor]],  # list of [B, C] or None
        final_logits: torch.Tensor,                   # [B, C]
    ) -> None:
        if not bool(getattr(self, "_mof_debug", False)):
            return

        every = int(getattr(self, "_mof_debug_every", 1))
        if every <= 0:
            every = 1

        self._mof_debug_calls = int(getattr(self, "_mof_debug_calls", 0)) + 1
        if (self._mof_debug_calls % every) != 0:
            return

        max_b = int(getattr(self, "_mof_debug_max_batch", 1))
        if max_b <= 0:
            max_b = 1

        max_e = int(getattr(self, "_mof_debug_max_experts", 0))
        if max_e <= 0:
            max_e = len(self.mof_experts)

        B = int(final_logits.size(0))
        C = int(final_logits.size(-1))
        bshow = min(B, max_b)
        eshow = min(len(self.mof_experts), max_e)

        print("")
        print(f"[MoF][debug] call={self._mof_debug_calls} B={B} C={C} E={len(self.mof_experts)}")
        print(f"[MoF][debug] experts={self.mof_experts[:eshow]}{' ...' if eshow < len(self.mof_experts) else ''}")

        w0 = weights[:bshow, :eshow].detach().float().cpu()
        for bi in range(bshow):
            wline = ", ".join([f"{self.mof_experts[i]}={float(w0[bi, i]):.4f}" for i in range(eshow)])
            print(f"[MoF][debug] sample{bi} weights: {wline}")

        if expert_logits is not None:
            for i in range(eshow):
                name = self.mof_experts[i]
                li = expert_logits[i][:bshow].detach().float().cpu()
                for bi in range(bshow):
                    vec = ", ".join([f"{float(li[bi, c]):.6f}" for c in range(C)])
                    print(f"[MoF][debug] sample{bi} expert={name} logits: [{vec}]")

        fout = final_logits[:bshow].detach().float().cpu()
        for bi in range(bshow):
            vec = ", ".join([f"{float(fout[bi, c]):.6f}" for c in range(C)])
            print(f"[MoF][debug] sample{bi} final logits: [{vec}]")
        print("", flush=True)

    def _mof_reprs(
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

        sent_tokens = out_sent.last_hidden_state
        term_tokens = out_term.last_hidden_state

        reprs: List[torch.Tensor] = []

        for name in self.mof_experts:
            if name == "sent":
                reprs.append(dropout(cls_sent))
                continue

            if name == "term":
                reprs.append(dropout(cls_term))
                continue

            if name == "concat":
                fused2h = torch.cat([cls_sent, cls_term], dim=-1)
                reprs.append(dropout(self.mof_proj_concat(fused2h)))
                continue

            if name == "add":
                reprs.append(dropout(cls_sent + cls_term))
                continue

            if name == "mul":
                reprs.append(dropout(cls_sent * cls_term))
                continue

            if name == "gated_concat":
                gate = getattr(self, "gate", None)
                if gate is None:
                    raise RuntimeError("gated_concat expert requires self.gate")
                g = torch.sigmoid(gate(torch.cat([cls_sent, cls_term], dim=-1)))
                reprs.append(dropout(g * cls_sent + (1.0 - g) * cls_term))
                continue

            if name == "bilinear":
                ps = getattr(self, "bilinear_proj_sent", None)
                pt = getattr(self, "bilinear_proj_term", None)
                bo = getattr(self, "bilinear_out", None)
                if ps is None or pt is None or bo is None:
                    raise RuntimeError("bilinear expert requires bilinear_proj_sent, bilinear_proj_term, bilinear_out")
                reprs.append(dropout(bo(ps(cls_sent) * pt(cls_term))))
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
                reprs.append(dropout(out.squeeze(1)))
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
                reprs.append(dropout(ctx_t + ctx_s))
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
                reprs.append(dropout(cond * pooled.unsqueeze(-1)))
                continue

            raise RuntimeError(f"Unknown MoF expert: {name}")

        return reprs

    def _apply_expert_scaling_and_clamp(self, repr_stack: torch.Tensor) -> torch.Tensor:
        # repr_stack: [B, E, H]
        if not bool(getattr(self, "_mof_disable_expert_scaling", False)):
            scale = torch.exp(self.mof_expert_log_scale).to(dtype=repr_stack.dtype)  # [E]
            repr_stack = repr_stack * scale.view(1, -1, 1)

        max_norm = float(getattr(self, "_mof_expert_norm_clamp", 0.0))
        if max_norm > 0:
            norms = torch.linalg.vector_norm(repr_stack.float(), ord=2, dim=-1, keepdim=True).clamp_min(1e-6)
            ratio = (max_norm / norms).clamp_max(1.0).to(dtype=repr_stack.dtype)
            repr_stack = repr_stack * ratio

        return repr_stack

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

        expert_reprs = self._mof_reprs(
            out_sent=out_sent,
            out_term=out_term,
            cls_sent=cls_sent,
            cls_term=cls_term,
            attention_mask_sent=attention_mask_sent,
            attention_mask_term=attention_mask_term,
        )

        repr_stack = torch.stack(expert_reprs, dim=1)  # [B, E, H]
        repr_stack = self._apply_expert_scaling_and_clamp(repr_stack)

        router_x = torch.cat(
            [cls_sent, cls_term, cls_sent * cls_term, torch.abs(cls_sent - cls_term)],
            dim=-1,
        )
        weights, _ = self.mof_router(router_x)  # [B, E]

        dropout = getattr(self, "dropout", nn.Identity())
        head_single = getattr(self, "head_single", None)
        if head_single is None:
            raise RuntimeError("MoF expects head_single from BaseBertConcatClassifier")

        mix_level = str(getattr(self, "_mof_mix_level", "repr")).lower()
        expert_logits_for_debug: Optional[List[torch.Tensor]] = None

        if mix_level == "logit":
            # Logit-level mixing (ablation or backward comparison).
            expert_logits = [head_single(r) for r in repr_stack.unbind(dim=1)]
            logits_stack = torch.stack(expert_logits, dim=1)  # [B, E, C]
            logits = torch.sum(logits_stack * weights.unsqueeze(-1), dim=1)

            clamp_v = float(getattr(self, "_mof_logit_clamp", 0.0))
            if clamp_v > 0:
                logits = logits.clamp(min=-clamp_v, max=clamp_v)

            if bool(getattr(self, "_mof_debug", False)):
                expert_logits_for_debug = expert_logits
        else:
            # Representation-level mixing then shared head.
            mixed_repr = torch.sum(repr_stack * weights.unsqueeze(-1), dim=1)  # [B, H]
            logits = head_single(dropout(mixed_repr))

            clamp_v = float(getattr(self, "_mof_logit_clamp", 0.0))
            if clamp_v > 0:
                logits = logits.clamp(min=-clamp_v, max=clamp_v)

            if bool(getattr(self, "_mof_debug", False)):
                expert_logits_for_debug = [head_single(r) for r in repr_stack.unbind(dim=1)]

        self._mof_debug_print(weights=weights, expert_logits=expert_logits_for_debug, final_logits=logits)

        out = self._compute_loss(logits, labels)

        # Load balancing regularizer on router weights.
        lb_coef = float(getattr(self, "_mof_lb_coef", 0.0))
        if lb_coef > 0 and labels is not None:
            p = weights.float().mean(dim=0)  # [E]
            # Encourage uniform usage: E * sum(p^2) has minimum 1 when p is uniform.
            lb_loss = p.pow(2).sum() * float(p.numel())
            base_loss = _safe_get_loss(out)
            if base_loss is not None:
                out["loss"] = base_loss + (lb_coef * lb_loss.to(dtype=base_loss.dtype))

        return out


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
        mof_experts: Optional[List[str]] = None,
        mof_debug: bool = False,
        mof_debug_every: int = 200,
        mof_debug_max_batch: int = 1,
        mof_debug_max_experts: int = 0,
        # new knobs
        mof_mix_level: str = "repr",
        mof_lb_coef: float = 0.0,
        mof_disable_expert_scaling: bool = False,
        mof_expert_norm_clamp: float = 0.0,
        mof_logit_clamp: float = 0.0,
        mof_router_temperature: float = 1.0,
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
            experts=mof_experts,
            router_temperature=float(mof_router_temperature),
            mix_level=str(mof_mix_level),
            lb_coef=float(mof_lb_coef),
            disable_expert_scaling=bool(mof_disable_expert_scaling),
            expert_norm_clamp=float(mof_expert_norm_clamp),
            logit_clamp=float(mof_logit_clamp),
        )

        self._mof_debug = bool(mof_debug)
        self._mof_debug_every = int(mof_debug_every)
        self._mof_debug_max_batch = int(mof_debug_max_batch)
        self._mof_debug_max_experts = int(mof_debug_max_experts)
        self._mof_debug_calls = 0

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
