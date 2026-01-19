from __future__ import annotations

import ast
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import build_head


class FFNExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float, act_fn: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = act_fn
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class HAGMoE(nn.Module):
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
        num_experts: int = 8,
        router_temperature: float = 1.0,
        hag_num_groups: int = 3,
        hag_experts_per_group: int = 8,
        hag_router_temperature: float = 1.0,
        hag_merge: str = "residual",
        hag_fusion_method: str = "concat",
        hag_use_group_loss: bool = False,
        hag_use_balance_loss: bool = False,
        hag_use_diversity_loss: bool = False,
        hag_lambda_group: float = 0.5,
        hag_lambda_balance: float = 0.01,
        hag_lambda_diversity: float = 0.1,
        hag_verbose_loss: bool = False,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)

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

        self.opinion_q = nn.Linear(hidden_size, hidden_size, bias=False)

        self.fusion_concat = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, hidden_size),
        )

        self.gate = nn.Linear(3 * hidden_size, hidden_size)

        bilinear_rank = max(32, min(256, hidden_size // 4))
        self.bilinear_proj_sent = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_proj_term = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_out = nn.Linear(bilinear_rank, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = build_head(head_type, hidden_size, num_labels, dropout)

        self.num_groups = int(hag_num_groups) if hag_num_groups else 3
        self.num_experts = int(hag_experts_per_group) if hag_experts_per_group else int(num_experts)
        self.router_temperature = float(hag_router_temperature if hag_router_temperature else router_temperature)

        self.hag_merge = str(hag_merge).lower().strip() if hag_merge is not None else "residual"
        self.hag_fusion_method = str(hag_fusion_method).strip() if hag_fusion_method else ""
        self.hag_use_group_loss = bool(hag_use_group_loss)
        self.hag_use_balance_loss = bool(hag_use_balance_loss)
        self.hag_use_diversity_loss = bool(hag_use_diversity_loss)
        self.hag_lambda_group = float(hag_lambda_group)
        self.hag_lambda_balance = float(hag_lambda_balance)
        self.hag_lambda_diversity = float(hag_lambda_diversity)
        self.hag_verbose_loss = bool(hag_verbose_loss)

        cfg = getattr(self.encoder, "config", None)
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))

        self.group_router = nn.Linear(hidden_size, self.num_groups)
        self.cond_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.expert_routers = nn.ModuleList(
            [nn.Linear(hidden_size, self.num_experts) for _ in range(self.num_groups)]
        )
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FFNExpert(hidden_size, intermediate_size, dropout_p, act_fn)
                        for _ in range(self.num_experts)
                    ]
                )
                for _ in range(self.num_groups)
            ]
        )

        self.loss_type = loss_type.lower().strip()

        cw = None
        if class_weights is None:
            cw = None
        elif isinstance(class_weights, torch.Tensor):
            cw = class_weights.detach().float()
        else:
            s = str(class_weights).strip()
            if not s:
                cw = None
            elif s.startswith("[") and s.endswith("]"):
                vals = ast.literal_eval(s)
                cw = torch.tensor([float(v) for v in vals], dtype=torch.float)
            else:
                cw = torch.tensor(
                    [float(x.strip()) for x in s.split(",") if x.strip()],
                    dtype=torch.float,
                )

        self.register_buffer("class_weights", cw)

        if self.loss_type not in {"ce", "weighted_ce", "focal"}:
            raise ValueError("loss_type must be one of: ce, weighted_ce, focal")
        if self.loss_type in {"weighted_ce", "focal"} and self.class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce or focal")

        self.focal_gamma = float(focal_gamma)

        labelid_to_groupid = None
        if id2label is None and label2id is not None:
            id2label = {int(v): k for k, v in label2id.items()}
        if id2label is not None:
            labelid_to_groupid = self._build_label_group_mapping(
                id2label=id2label,
                num_labels=num_labels,
                num_groups=self.num_groups,
            )
        self.register_buffer("labelid_to_groupid", labelid_to_groupid)

    @staticmethod
    def _neg_inf(dtype: torch.dtype) -> float:
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4
        return -1e9

    @staticmethod
    def _build_label_group_mapping(
        *,
        id2label: dict,
        num_labels: int,
        num_groups: int,
    ) -> torch.Tensor:
        if num_groups < 3:
            raise ValueError("HAGMoE requires num_groups >= 3 for polarity-aware group loss")

        mapping = torch.full((int(num_labels),), -1, dtype=torch.long)
        for idx in range(int(num_labels)):
            name = str(id2label.get(idx, "")).lower().strip()
            # Group index order is fixed: 0=positive, 1=negative, 2=neutral.
            if name in {"positive", "pos", "posi"}:
                group_id = 0
            elif name in {"negative", "neg"}:
                group_id = 1
            elif name in {"neutral", "neu", "neut"}:
                group_id = 2
            else:
                group_id = -1
            if group_id >= 0:
                mapping[idx] = group_id

        if torch.any(mapping < 0):
            missing = [i for i in range(int(num_labels)) if int(mapping[i].item()) < 0]
            raise ValueError(
                "HAGMoE group loss requires id2label with positive/negative/neutral labels. "
                f"Unmapped label ids: {missing}"
            )
        return mapping

    @staticmethod
    def _gather_aspect_tokens(
        h_sent_tokens: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if aspect_mask_sent is None:
            return h_sent_tokens, None

        mask = aspect_mask_sent.to(dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        max_len = max(1, max_len)

        bsz, _, hidden = h_sent_tokens.shape
        term_tok = h_sent_tokens.new_zeros((bsz, max_len, hidden))
        term_attn_mask = mask.new_zeros((bsz, max_len), dtype=mask.dtype)

        for i in range(bsz):
            idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            take = h_sent_tokens[i, idx, :]
            take_len = min(max_len, int(take.size(0)))
            term_tok[i, :take_len, :] = take[:take_len]
            term_attn_mask[i, :take_len] = 1

        return term_tok, term_attn_mask

    def compute_opinion(
        self,
        h_sent_tokens: torch.Tensor,
        attention_mask_sent: torch.Tensor | None,
        h_aspect: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.opinion_q(h_sent_tokens)
        scores = (q * h_aspect.unsqueeze(1)).sum(dim=-1) / math.sqrt(q.size(-1))

        if attention_mask_sent is not None:
            pad_mask = attention_mask_sent.eq(0)
            scores = scores.masked_fill(pad_mask, self._neg_inf(scores.dtype))

        if aspect_mask_sent is not None:
            aspect_mask = aspect_mask_sent.to(dtype=torch.bool)
            scores = scores.masked_fill(aspect_mask, self._neg_inf(scores.dtype))

        attn = torch.softmax(scores, dim=-1)
        h_opinion = torch.bmm(attn.unsqueeze(1), h_sent_tokens).squeeze(1)
        return h_opinion

    def build_fusion(
        self,
        *,
        h_sent: torch.Tensor,
        h_aspect: torch.Tensor,
        h_opinion: torch.Tensor,
        out_sent,
        attention_mask_sent: torch.Tensor,
        aspect_mask_sent: torch.Tensor | None,
        fusion_method: str,
    ) -> torch.Tensor:
        fusion_method = fusion_method.lower().strip()
        rep_sent = h_sent

        if fusion_method == "sent":
            raise ValueError(
                "HAGMoE does not support fusion_method 'sent' or 'term'. "
                "Use one of: concat, add, mul, cross, gated_concat, bilinear, coattn, late_interaction."
            )
        elif fusion_method == "term":
            raise ValueError(
                "HAGMoE does not support fusion_method 'sent' or 'term'. "
                "Use one of: concat, add, mul, cross, gated_concat, bilinear, coattn, late_interaction."
            )
        elif fusion_method == "concat":
            fusion_in = torch.cat([rep_sent, h_aspect, h_opinion], dim=-1)
            if not hasattr(self, "_fusion_debug_printed"):
                self._fusion_debug_printed = True
                print(f"[HAGMoE] fusion=concat input={tuple(fusion_in.shape)}")
            h_fused = self.fusion_concat(fusion_in)
        elif fusion_method == "add":
            h_fused = rep_sent + h_aspect + h_opinion
        elif fusion_method == "mul":
            h_fused = rep_sent * h_aspect * h_opinion
        elif fusion_method == "cross":
            q = h_aspect.unsqueeze(1)
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(
                q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
            )
            q_o = h_opinion.unsqueeze(1)
            attn_out_o, _ = self.cross_attn(
                q_o, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
            )
            h_fused = attn_out.squeeze(1) + attn_out_o.squeeze(1)
        elif fusion_method == "gated_concat":
            g = torch.sigmoid(self.gate(torch.cat([rep_sent, h_aspect, h_opinion], dim=-1)))
            h_fused = g * rep_sent + (1 - g) * (0.5 * (h_aspect + h_opinion))
        elif fusion_method == "bilinear":
            h_fused = self.bilinear_out(
                self.bilinear_proj_sent(rep_sent)
                * self.bilinear_proj_term(h_aspect)
                * self.bilinear_proj_term(h_opinion)
            )
        elif fusion_method == "coattn":
            sent_tok = out_sent.last_hidden_state
            term_tok, term_attn_mask = self._gather_aspect_tokens(sent_tok, aspect_mask_sent)

            q_sent = sent_tok[:, 0:1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = term_attn_mask.eq(0) if term_attn_mask is not None else None

            term_ctx, _ = self.coattn_term_to_sent(
                term_tok, sent_tok, sent_tok, key_padding_mask=kpm_sent
            )
            if term_attn_mask is not None:
                term_valid = term_attn_mask.float()
                denom = term_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                term_ctx = (term_ctx * term_valid.unsqueeze(-1)).sum(dim=1) / denom
            else:
                term_ctx = term_ctx.mean(dim=1)

            sent_ctx, _ = self.coattn_sent_to_term(
                q_sent, term_tok, term_tok, key_padding_mask=kpm_term
            )
            h_fused = term_ctx + sent_ctx.squeeze(1)
        elif fusion_method == "late_interaction":
            sent_tok = out_sent.last_hidden_state
            term_tok, term_attn_mask = self._gather_aspect_tokens(sent_tok, aspect_mask_sent)

            sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
            term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))
            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)
                sim = sim.masked_fill(mask.bool(), self._neg_inf(sim.dtype))
            if term_attn_mask is not None:
                term_mask = term_attn_mask.unsqueeze(-1).eq(0)
                sim = sim.masked_fill(term_mask.bool(), self._neg_inf(sim.dtype))

            max_sim = sim.max(dim=-1).values
            if term_attn_mask is not None:
                term_valid = term_attn_mask.float()
                denom = term_valid.sum(dim=1).clamp_min(1.0)
                pooled = (max_sim * term_valid).sum(dim=1) / denom
                empty = term_valid.sum(dim=1).eq(0)
                if torch.any(empty):
                    pooled = torch.where(empty, torch.ones_like(pooled), pooled)
            else:
                pooled = max_sim.mean(dim=1)

            # TODO: consider a better calibrated fusion for late_interaction beyond scalar gating.
            cond = self.gate(torch.cat([rep_sent, h_aspect, h_opinion], dim=-1))
            h_fused = cond * pooled.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        return h_fused

    def route_group(self, h_fused: torch.Tensor) -> torch.Tensor:
        return self.group_router(h_fused)

    def _pool_aspect(
        self,
        hidden_states: torch.Tensor,
        aspect_mask: torch.Tensor | None,
        h_sent: torch.Tensor,
    ) -> torch.Tensor:
        if aspect_mask is None:
            return h_sent

        mask = aspect_mask.to(dtype=hidden_states.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.size(1) != hidden_states.size(1):
            mask = mask[:, : hidden_states.size(1)]

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denom

        has_span = mask.sum(dim=1) > 0
        if torch.any(~has_span):
            pooled = torch.where(has_span.unsqueeze(-1), pooled, h_sent)

        return pooled

    def _build_aspect_mask(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        aspect_start: torch.Tensor | None,
        aspect_end: torch.Tensor | None,
        aspect_mask_sent: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if aspect_mask_sent is not None:
            return aspect_mask_sent

        if aspect_start is not None and aspect_end is not None:
            start = aspect_start.to(device=input_ids_sent.device)
            end = aspect_end.to(device=input_ids_sent.device)
            if start.dim() == 0:
                start = start.unsqueeze(0)
            if end.dim() == 0:
                end = end.unsqueeze(0)
            L = input_ids_sent.size(1)
            positions = torch.arange(L, device=input_ids_sent.device).unsqueeze(0)
            mask = (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))
            return mask.to(dtype=torch.long)

        if input_ids_term is None:
            return None

        cls_id = getattr(self.encoder.config, "cls_token_id", None)
        sep_id = getattr(self.encoder.config, "sep_token_id", None)
        pad_id = getattr(self.encoder.config, "pad_token_id", None)
        special_ids = {x for x in (cls_id, sep_id, pad_id) if x is not None}

        bsz, L = input_ids_sent.shape
        mask_out = torch.zeros((bsz, L), device=input_ids_sent.device, dtype=torch.long)

        for i in range(bsz):
            sent_ids = input_ids_sent[i].tolist()
            sent_mask = attention_mask_sent[i].tolist()
            valid_len = int(sum(sent_mask))
            if valid_len <= 0:
                continue
            content_start = 1
            content_end = valid_len
            if (
                content_end > content_start
                and sep_id is not None
                and sent_ids[content_end - 1] == sep_id
            ):
                content_end -= 1
            if content_end < content_start:
                content_end = content_start
            content_ids = sent_ids[content_start:content_end]

            term_ids_full = input_ids_term[i].tolist()
            term_ids = [tid for tid in term_ids_full if tid not in special_ids]
            if not term_ids:
                continue

            match_idx = -1
            for j in range(len(content_ids) - len(term_ids) + 1):
                if content_ids[j : j + len(term_ids)] == term_ids:
                    match_idx = j
                    break
            if match_idx < 0:
                continue

            start = content_start + match_idx
            end = start + len(term_ids)
            if start >= L:
                continue
            end = min(end, L)
            mask_out[i, start:end] = 1

        return mask_out

    def apply_grouped_experts(
        self, h_fused: torch.Tensor, h_aspect: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        h_cond = self.cond_proj(torch.cat([h_fused, h_aspect], dim=-1))
        group_logits = self.route_group(h_fused)
        p_group = torch.softmax(group_logits, dim=-1)

        h_moe = torch.zeros_like(h_fused)
        p_expert_list: list[torch.Tensor] = []
        expert_outs_list: list[torch.Tensor] = []
        for g in range(self.num_groups):
            logits = self.expert_routers[g](h_cond) / self.router_temperature
            p_expert = torch.softmax(logits, dim=-1)
            p_expert_list.append(p_expert)

            expert_outs = []
            for expert in self.experts[g]:
                expert_outs.append(expert(h_fused))
            expert_stack = torch.stack(expert_outs, dim=1)
            expert_outs_list.append(expert_stack)

            h_g = (p_expert.unsqueeze(-1) * expert_stack).sum(dim=1)
            h_moe = h_moe + p_group[:, g].unsqueeze(-1) * h_g

        return h_moe, group_logits, p_expert_list, expert_outs_list

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        aspect_start: torch.Tensor | None = None,
        aspect_end: torch.Tensor | None = None,
        aspect_mask_sent: torch.Tensor | None = None,
        labels=None,
        fusion_method: str = "concat",
    ):
        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)

        h_sent = out_sent.last_hidden_state[:, 0, :]
        aspect_mask = self._build_aspect_mask(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            aspect_start=aspect_start,
            aspect_end=aspect_end,
            aspect_mask_sent=aspect_mask_sent,
        )
        h_aspect = self._pool_aspect(out_sent.last_hidden_state, aspect_mask, h_sent)

        h_opinion = self.compute_opinion(
            out_sent.last_hidden_state, attention_mask_sent, h_aspect, aspect_mask
        )

        fusion_arg = str(fusion_method).strip() if fusion_method is not None else ""
        effective_fusion = fusion_arg if fusion_arg else str(self.hag_fusion_method or "").strip()
        if not effective_fusion:
            effective_fusion = "concat"
        self._last_fusion_method = effective_fusion

        h_fused = self.build_fusion(
            h_sent=h_sent,
            h_aspect=h_aspect,
            h_opinion=h_opinion,
            out_sent=out_sent,
            attention_mask_sent=attention_mask_sent,
            aspect_mask_sent=aspect_mask,
            fusion_method=effective_fusion,
        )

        h_moe, group_logits, p_expert_list, expert_outs_list = self.apply_grouped_experts(
            h_fused, h_aspect
        )
        self._last_group_probs = torch.softmax(group_logits, dim=-1).detach()
        self._last_expert_probs = [p.detach() for p in p_expert_list]
        if self.hag_merge == "moe_only":
            h_final = h_moe
        else:
            h_final = h_fused + h_moe
        logits = self.classifier(self.dropout(h_final))

        out = self._compute_loss(
            logits,
            labels,
            group_logits=group_logits,
            p_expert_list=p_expert_list,
            expert_outs_list=expert_outs_list,
        )

        if self.training and self.hag_verbose_loss and labels is not None:
            print(
                "[HAGMoE] "
                f"main={out['loss_main'].item():.4f} "
                f"group={(out['loss_group'].item() if out['loss_group'] is not None else 0.0):.4f} "
                f"balance={(out['loss_balance'].item() if out['loss_balance'] is not None else 0.0):.4f} "
                f"diversity={(out['loss_diversity'].item() if out['loss_diversity'] is not None else 0.0):.4f}"
            )

        return out

    def _compute_loss(
        self,
        logits,
        labels,
        *,
        group_logits: torch.Tensor | None = None,
        p_expert_list: list[torch.Tensor] | None = None,
        expert_outs_list: list[torch.Tensor] | None = None,
    ):
        if labels is None:
            return {
                "loss": None,
                "logits": logits,
                "loss_main": None,
                "aux_loss": None,
                "loss_group": None,
                "loss_balance": None,
                "loss_diversity": None,
                "loss_lambda": None,
                "lambda_group": float(self.hag_lambda_group),
                "lambda_balance": float(self.hag_lambda_balance),
                "lambda_diversity": float(self.hag_lambda_diversity),
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

        loss_group = None
        if group_logits is not None:
            if self.labelid_to_groupid is None:
                raise RuntimeError(
                    "hag_use_group_loss=True but labelid_to_groupid is not set. "
                    "Provide id2label/label2id in cfg."
                )
            group_target = self.labelid_to_groupid[labels]
            loss_group = F.cross_entropy(group_logits, group_target)

        loss_balance = None
        if p_expert_list is not None and len(p_expert_list) > 0:
            # Switch-style load balance: N * sum_i f_i * P_i
            losses = []
            for p_expert in p_expert_list:
                if p_expert.numel() == 0:
                    continue
                p_mean = p_expert.mean(dim=0)
                assign = torch.argmax(p_expert, dim=-1)
                f = F.one_hot(assign, num_classes=p_expert.size(1)).float().mean(dim=0)
                n_experts = p_expert.size(1)
                losses.append(n_experts * torch.sum(f * p_mean))
            if losses:
                loss_balance = sum(losses) / len(losses)

        loss_diversity = None
        # Expert diversity via weight orthogonality within each group.
        losses = []
        eps = 1e-8
        for group in self.experts:
            if group is None or len(group) == 0:
                continue
            weights = []
            for expert in group:
                w = expert.fc1.weight
                weights.append(w.reshape(-1))
            if len(weights) < 2:
                continue
            W = torch.stack(weights, dim=0)
            W = W / (W.norm(dim=1, keepdim=True) + eps)
            G = W @ W.t()
            eye = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
            losses.append(((G - eye) ** 2).sum())
        if losses:
            loss_diversity = sum(losses) / len(losses)

        lambda_group = float(self.hag_lambda_group) if self.hag_use_group_loss else 0.0
        lambda_balance = float(self.hag_lambda_balance) if self.hag_use_balance_loss else 0.0
        lambda_diversity = float(self.hag_lambda_diversity) if self.hag_use_diversity_loss else 0.0

        aux_loss = torch.zeros((), device=logits.device)
        if loss_group is not None:
            aux_loss = aux_loss + lambda_group * loss_group
        if loss_balance is not None:
            aux_loss = aux_loss + lambda_balance * loss_balance
        if loss_diversity is not None:
            aux_loss = aux_loss + lambda_diversity * loss_diversity

        loss = loss_main + aux_loss

        return {
            "loss": loss,
            "loss_total": loss.detach(),
            "logits": logits,
            "loss_main": loss_main.detach(),
            "aux_loss": aux_loss.detach(),
            "loss_group": loss_group.detach() if loss_group is not None else None,
            "loss_balance": loss_balance.detach() if loss_balance is not None else None,
            "loss_diversity": loss_diversity.detach() if loss_diversity is not None else None,
            "loss_lambda": aux_loss.detach(),
            "lambda_group": lambda_group,
            "lambda_balance": lambda_balance,
            "lambda_diversity": lambda_diversity,
        }

    def _collect_aux_loss(self):
        return torch.zeros((), device=next(self.parameters()).device)

    @torch.no_grad()
    def print_moe_debug(self, topn: int = 3, eps_dead: float = 1e-6):
        if not hasattr(self, "_last_group_probs") or not hasattr(self, "_last_expert_probs"):
            print("[HAGMoE] No routing stats yet.")
            return

        group_probs = getattr(self, "_last_group_probs", None)
        expert_probs = getattr(self, "_last_expert_probs", None)
        if group_probs is None or expert_probs is None:
            print("[HAGMoE] No routing stats yet.")
            return

        group_mean = group_probs.mean(dim=0).detach().cpu()
        group_pairs = " ".join([f"g{gi}={float(p):.6f}" for gi, p in enumerate(group_mean)])
        print("\n[HAGMoE Debug - Grouped Router]")
        print(f"  group_mean: {group_pairs}")

        for g, p_expert in enumerate(expert_probs):
            if p_expert is None or p_expert.numel() == 0:
                print(f"  group{g}: no expert stats")
                continue

            usage = p_expert.mean(dim=0).detach().cpu()
            dead = int((usage < eps_dead).sum().item())
            topk = min(topn, usage.numel())
            botk = min(topn, usage.numel())
            topv, topi = torch.topk(usage, k=topk, largest=True)
            botv, boti = torch.topk(usage, k=botk, largest=False)
            top_pairs = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv, topi)])
            bot_pairs = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv, boti)])

            print(
                f"  group{g}: min={float(usage.min()):.6f} max={float(usage.max()):.6f} dead(<{eps_dead:g})={dead}"
            )
            print(f"    top: {top_pairs}")
            print(f"    bot: {bot_pairs}")
