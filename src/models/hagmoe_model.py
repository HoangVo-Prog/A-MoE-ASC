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
            nn.LayerNorm(2 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
        )

        self.gate = nn.Linear(2 * hidden_size, hidden_size)

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
            scores = scores.masked_fill(pad_mask, torch.finfo(scores.dtype).min)

        if aspect_mask_sent is not None:
            aspect_mask = aspect_mask_sent.to(dtype=torch.bool)
            scores = scores.masked_fill(aspect_mask, torch.finfo(scores.dtype).min)

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
        rep_sent = h_opinion

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
            h_fused = self.fusion_concat(torch.cat([rep_sent, h_aspect], dim=-1))
        elif fusion_method == "add":
            h_fused = rep_sent + h_aspect
        elif fusion_method == "mul":
            h_fused = rep_sent * h_aspect
        elif fusion_method == "cross":
            q = h_aspect.unsqueeze(1)
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(
                q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
            )
            h_fused = attn_out.squeeze(1)
        elif fusion_method == "gated_concat":
            g = torch.sigmoid(self.gate(torch.cat([rep_sent, h_aspect], dim=-1)))
            h_fused = g * rep_sent + (1 - g) * h_aspect
        elif fusion_method == "bilinear":
            h_fused = self.bilinear_out(
                self.bilinear_proj_sent(rep_sent) * self.bilinear_proj_term(h_aspect)
            )
        elif fusion_method == "coattn":
            q_term = h_aspect.unsqueeze(1)
            q_sent = out_sent.last_hidden_state[:, 0:1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = aspect_mask_sent.eq(0) if aspect_mask_sent is not None else None

            term_ctx, _ = self.coattn_term_to_sent(
                q_term, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_sent
            )
            sent_ctx, _ = self.coattn_sent_to_term(
                q_sent, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_term
            )
            h_fused = term_ctx.squeeze(1) + sent_ctx.squeeze(1)
        elif fusion_method == "late_interaction":
            sent_tok = out_sent.last_hidden_state
            term_tok = out_sent.last_hidden_state

            sent_tok = torch.nn.functional.normalize(sent_tok, p=2, dim=-1)
            term_tok = torch.nn.functional.normalize(term_tok, p=2, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))
            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)
                sim = sim.masked_fill(mask.bool(), torch.finfo(sim.dtype).min)

            max_sim = sim.max(dim=-1).values
            if aspect_mask_sent is not None:
                term_valid = aspect_mask_sent.float()
                denom = term_valid.sum(dim=1).clamp_min(1.0)
                pooled = (max_sim * term_valid).sum(dim=1) / denom
                empty = term_valid.sum(dim=1).eq(0)
                if torch.any(empty):
                    pooled = torch.where(empty, torch.ones_like(pooled), pooled)
            else:
                pooled = max_sim.mean(dim=1)

            cond = self.gate(torch.cat([rep_sent, h_aspect], dim=-1))
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

        h_fused = self.build_fusion(
            h_sent=h_sent,
            h_aspect=h_aspect,
            h_opinion=h_opinion,
            out_sent=out_sent,
            attention_mask_sent=attention_mask_sent,
            aspect_mask_sent=aspect_mask,
            fusion_method=self.hag_fusion_method or fusion_method,
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
        if self.hag_use_group_loss and group_logits is not None:
            loss_group = F.cross_entropy(group_logits, labels)

        loss_balance = None
        if self.hag_use_balance_loss and p_expert_list is not None and len(p_expert_list) > 0:
            losses = []
            for p_expert in p_expert_list:
                p_mean = p_expert.mean(dim=0)
                uniform = torch.full_like(p_mean, 1.0 / max(1, p_mean.numel()))
                losses.append(F.mse_loss(p_mean, uniform))
            loss_balance = sum(losses) / max(1, len(losses))

        loss_diversity = None
        if (
            self.hag_use_diversity_loss
            and expert_outs_list is not None
            and len(expert_outs_list) > 0
        ):
            losses = []
            for expert_stack in expert_outs_list:
                if expert_stack.numel() == 0:
                    continue
                means = expert_stack.mean(dim=0)
                means = F.normalize(means, p=2, dim=-1)
                sim = torch.matmul(means, means.transpose(0, 1))
                eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
                off_diag = sim * (1.0 - eye)
                losses.append((off_diag ** 2).mean())
            if losses:
                loss_diversity = sum(losses) / len(losses)

        aux_loss = torch.zeros((), device=logits.device)
        if loss_group is not None:
            aux_loss = aux_loss + self.hag_lambda_group * loss_group
        if loss_balance is not None:
            aux_loss = aux_loss + self.hag_lambda_balance * loss_balance
        if loss_diversity is not None:
            aux_loss = aux_loss + self.hag_lambda_diversity * loss_diversity

        loss = loss_main + aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "loss_main": loss_main.detach(),
            "aux_loss": aux_loss.detach(),
            "loss_group": loss_group.detach() if loss_group is not None else None,
            "loss_balance": loss_balance.detach() if loss_balance is not None else None,
            "loss_diversity": loss_diversity.detach() if loss_diversity is not None else None,
            "loss_lambda": aux_loss.detach(),
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
