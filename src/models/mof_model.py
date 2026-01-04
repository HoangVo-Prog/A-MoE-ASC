import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel


class MoFRouter(nn.Module):
    """Router network for Mixture of Fusion"""
    def __init__(
        self,
        *,
        in_dim: int,
        num_experts: int,
        hidden_dim: int,
        dropout: float,
        temperature: float = 1.0,
        router_bias: bool = True,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature) if float(temperature) > 0 else 1.0
        self.router_jitter = float(router_jitter)
        
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_experts), bias=bool(router_bias)),
        )
        
        # Initialize router near-uniform
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.last_router_logits = None
        self.last_weights = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.router_jitter > 0.0:
            noise = (torch.rand_like(x) - 0.5) * 2.0 * self.router_jitter
            x = x + noise
        
        logits = self.net(x) / self.temperature
        # Softmax in fp32 for stability
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        
        self.last_router_logits = logits
        self.last_weights = weights
        
        return weights, logits


class MoFModel(BaseModel):
    """Mixture of Fusion model - router selects among different fusion strategies"""
    
    # Allowed fusion experts
    _ALLOWED_EXPERTS = {
        "sent", "term", "concat", "add", "mul", 
        "cross", "gated_concat", "bilinear", "coattn", "late_interaction"
    }
    
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
        # MoF specific parameters
        mof_experts: Optional[List[str]] = None,
        mof_mix_level: str = "repr",  # "repr" or "logit"
        mof_router_temperature: float = 1.0,
        mof_router_bias: bool = True,
        mof_router_jitter: float = 0.0,
        mof_lb_coef: float = 0.001,
        mof_lb_mode: str = "switch",  # "switch" or "l2"
        mof_entropy_coef: float = 0.0,
        mof_disable_expert_scaling: bool = False,
        mof_expert_norm_clamp: float = 0.0,
        mof_mixed_repr_norm: str = "none",  # "none", "layernorm", "clamp"
        mof_mixed_repr_norm_clamp: float = 0.0,
        mof_logit_clamp: float = 0.0,
        # Warmup schedules
        lb_warmup_steps: int = 0,
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
        
        # Setup fusion experts
        if mof_experts is not None:
            self.mof_experts = list(mof_experts)
        else:
            self.mof_experts = [
                "concat", "add", "mul", "gated_concat", 
                "bilinear", "cross", "coattn", "late_interaction"
            ]
        
        # Validate experts
        unknown = [e for e in self.mof_experts if e not in self._ALLOWED_EXPERTS]
        if unknown:
            raise ValueError(f"Unknown mof_experts: {unknown}. Allowed: {sorted(self._ALLOWED_EXPERTS)}")
        
        hidden_size = self.encoder.config.hidden_size
        num_experts = len(self.mof_experts)
        
        # Router takes 4H features: [cls_sent, cls_term, cls_sent*cls_term, |cls_sent-cls_term|]
        self.mof_router = MoFRouter(
            in_dim=4 * hidden_size,
            num_experts=num_experts,
            hidden_dim=hidden_size,
            dropout=dropout,
            temperature=mof_router_temperature,
            router_bias=mof_router_bias,
            router_jitter=mof_router_jitter,
        )
        
        # Projection for concat expert (2H -> H)
        self.mof_proj_concat = nn.Linear(2 * hidden_size, hidden_size)
        
        # Per-expert scaling (log scale for stability)
        self.mof_disable_expert_scaling = bool(mof_disable_expert_scaling)
        self.mof_expert_log_scale = nn.Parameter(torch.zeros(num_experts))
        
        # MoF configuration
        self.mof_mix_level = str(mof_mix_level).lower()
        if self.mof_mix_level not in ("repr", "logit"):
            self.mof_mix_level = "repr"
        
        self.mof_lb_coef = float(mof_lb_coef)
        self.mof_lb_mode = str(mof_lb_mode).lower()
        if self.mof_lb_mode not in ("l2", "switch"):
            self.mof_lb_mode = "l2"
        
        self.mof_entropy_coef = float(mof_entropy_coef)
        self.mof_expert_norm_clamp = float(mof_expert_norm_clamp)
        self.mof_logit_clamp = float(mof_logit_clamp)
        
        # Mixed representation normalization
        self.mof_mixed_repr_norm = str(mof_mixed_repr_norm).lower()
        if self.mof_mixed_repr_norm not in ("none", "layernorm", "clamp"):
            self.mof_mixed_repr_norm = "none"
        
        self.mof_mixed_repr_norm_clamp = float(mof_mixed_repr_norm_clamp)
        if self.mof_mixed_repr_norm == "layernorm":
            self.mof_mixed_repr_ln = nn.LayerNorm(hidden_size)
        else:
            self.mof_mixed_repr_ln = None
        
        # Warmup schedules
        self.lb_warmup_steps = int(lb_warmup_steps)
        self.jitter_warmup_steps = int(jitter_warmup_steps)
        self.jitter_end = float(jitter_end)
        self._jitter_start = float(mof_router_jitter)
        self._global_step = 0

    def _get_expert_representations(
        self,
        *,
        out_sent,
        out_term,
        cls_sent: torch.Tensor,
        cls_term: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        attention_mask_term: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute representation for each fusion expert"""
        
        sent_tokens = out_sent.last_hidden_state
        term_tokens = out_term.last_hidden_state
        dropout = self.dropout
        
        reprs: List[torch.Tensor] = []
        
        for name in self.mof_experts:
            if name == "sent":
                reprs.append(dropout(cls_sent))
            
            elif name == "term":
                reprs.append(dropout(cls_term))
            
            elif name == "concat":
                fused_2h = torch.cat([cls_sent, cls_term], dim=-1)
                reprs.append(dropout(self.mof_proj_concat(fused_2h)))
            
            elif name == "add":
                reprs.append(dropout(cls_sent + cls_term))
            
            elif name == "mul":
                reprs.append(dropout(cls_sent * cls_term))
            
            elif name == "gated_concat":
                g = torch.sigmoid(self.gate(torch.cat([cls_sent, cls_term], dim=-1)))
                reprs.append(dropout(g * cls_sent + (1.0 - g) * cls_term))
            
            elif name == "bilinear":
                fused = self.bilinear_out(
                    self.bilinear_proj_sent(cls_sent) * self.bilinear_proj_term(cls_term)
                )
                reprs.append(dropout(fused))
            
            elif name == "cross":
                q = term_tokens[:, 0:1, :]
                kpm = attention_mask_sent.eq(0)
                attn_out, _ = self.cross_attn(
                    q, sent_tokens, sent_tokens, key_padding_mask=kpm
                )
                reprs.append(dropout(attn_out.squeeze(1)))
            
            elif name == "coattn":
                q_term = term_tokens[:, 0:1, :]
                q_sent = sent_tokens[:, 0:1, :]
                kpm_sent = attention_mask_sent.eq(0)
                kpm_term = attention_mask_term.eq(0)
                
                term_ctx, _ = self.coattn_term_to_sent(
                    q_term, sent_tokens, sent_tokens, key_padding_mask=kpm_sent
                )
                sent_ctx, _ = self.coattn_sent_to_term(
                    q_sent, term_tokens, term_tokens, key_padding_mask=kpm_term
                )
                reprs.append(dropout(term_ctx.squeeze(1) + sent_ctx.squeeze(1)))
            
            elif name == "late_interaction":
                sent_norm = F.normalize(sent_tokens, p=2, dim=-1)
                term_norm = F.normalize(term_tokens, p=2, dim=-1)
                
                sim = torch.matmul(term_norm, sent_norm.transpose(1, 2))
                if attention_mask_sent is not None:
                    m_sent = attention_mask_sent.unsqueeze(1).eq(0)
                    sim = sim.masked_fill(m_sent, -1e9)
                
                max_sim = sim.max(dim=-1).values
                if attention_mask_term is not None:
                    m_term = attention_mask_term.float()
                    pooled = (max_sim * m_term).sum(dim=-1) / m_term.sum(dim=-1).clamp_min(1.0)
                else:
                    pooled = max_sim.mean(dim=-1)
                
                cond = self.gate(torch.cat([cls_sent, cls_term], dim=-1))
                reprs.append(dropout(cond * pooled.unsqueeze(-1)))
            
            else:
                raise RuntimeError(f"Unknown MoF expert: {name}")
        
        return reprs

    def _apply_expert_scaling_and_clamp(self, repr_stack: torch.Tensor) -> torch.Tensor:
        """Apply per-expert scaling and norm clamping"""
        # repr_stack: [B, E, H]
        
        if not self.mof_disable_expert_scaling:
            scale = torch.exp(self.mof_expert_log_scale).to(dtype=repr_stack.dtype)  # [E]
            repr_stack = repr_stack * scale.view(1, -1, 1)
        
        if self.mof_expert_norm_clamp > 0:
            norms = torch.linalg.vector_norm(
                repr_stack.float(), ord=2, dim=-1, keepdim=True
            ).clamp_min(1e-6)
            ratio = (self.mof_expert_norm_clamp / norms).clamp_max(1.0).to(dtype=repr_stack.dtype)
            repr_stack = repr_stack * ratio
        
        return repr_stack

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "mof",
    ):
        if self.training:
            self._global_step += 1
            # Update jitter schedule
            self.mof_router.router_jitter = self._jitter()
        
        # Encode sentence and term
        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)
        
        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]
        
        # Get expert representations
        expert_reprs = self._get_expert_representations(
            out_sent=out_sent,
            out_term=out_term,
            cls_sent=cls_sent,
            cls_term=cls_term,
            attention_mask_sent=attention_mask_sent,
            attention_mask_term=attention_mask_term,
        )
        
        # Stack and scale expert representations
        repr_stack = torch.stack(expert_reprs, dim=1)  # [B, E, H]
        repr_stack = self._apply_expert_scaling_and_clamp(repr_stack)
        
        # Router input: concatenate multiple features
        router_input = torch.cat([
            cls_sent, 
            cls_term, 
            cls_sent * cls_term, 
            torch.abs(cls_sent - cls_term)
        ], dim=-1)
        
        # Get routing weights
        weights, router_logits = self.mof_router(router_input)  # [B, E]
        
        # Mix representations or logits based on mix_level
        if self.mof_mix_level == "logit":
            # Logit-level mixing
            expert_logits = [self.head_single(r) for r in repr_stack.unbind(dim=1)]
            logits_stack = torch.stack(expert_logits, dim=1)  # [B, E, C]
            logits = torch.sum(logits_stack * weights.unsqueeze(-1), dim=1)  # [B, C]
        else:
            # Representation-level mixing (default)
            mixed_repr = torch.sum(repr_stack * weights.unsqueeze(-1), dim=1)  # [B, H]
            
            # Apply normalization to mixed representation
            if self.mof_mixed_repr_norm == "layernorm" and self.mof_mixed_repr_ln is not None:
                mixed_repr = self.mof_mixed_repr_ln(mixed_repr)
            elif self.mof_mixed_repr_norm == "clamp" and self.mof_mixed_repr_norm_clamp > 0:
                norms = torch.linalg.vector_norm(
                    mixed_repr.float(), ord=2, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                ratio = (self.mof_mixed_repr_norm_clamp / norms).clamp_max(1.0).to(dtype=mixed_repr.dtype)
                mixed_repr = mixed_repr * ratio
            
            # Pass through classification head
            logits = self.head_single(self.dropout(mixed_repr))
        
        # Clamp logits if specified
        if self.mof_logit_clamp > 0:
            logits = logits.clamp(min=-self.mof_logit_clamp, max=self.mof_logit_clamp)
        
        return self._compute_loss(logits, labels)

    def _compute_loss(self, logits, labels):
        if labels is None:
            return {
                "loss": None,
                "logits": logits,
                "aux_loss": None,
                "router_entropy": None,
                "loss_main": None,
                "loss_lb": None,
                "loss_entropy": None,
                "loss_total": None,
            }
        
        # Main classification loss
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
        
        # Load balancing loss
        aux_loss = self._collect_load_balancing_loss()
        lb_weight = self._lb_weight()
        loss_lb = lb_weight * aux_loss
        
        # Router entropy regularization
        router_entropy = self._collect_router_entropy()
        loss_entropy = self.mof_entropy_coef * router_entropy
        
        # Total loss
        loss_total = loss_main + loss_lb + loss_entropy
        
        return {
            "loss": loss_total,
            "logits": logits,
            "aux_loss": aux_loss,
            "router_entropy": router_entropy,
            "loss_main": loss_main,
            "loss_lb": loss_lb,
            "loss_entropy": loss_entropy,
            "loss_total": loss_total,
        }

    def _collect_load_balancing_loss(self) -> torch.Tensor:
        """Compute load balancing loss on router weights"""
        weights = getattr(self.mof_router, "last_weights", None)
        
        if weights is None or weights.shape[0] == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        
        num_experts = weights.shape[-1]
        
        if self.mof_lb_mode == "switch":
            # Switch-style: importance * load
            importance = weights.float().mean(dim=0)  # [E]
            
            # Load: fraction of tokens routed to each expert (based on argmax)
            top1 = torch.argmax(weights.float(), dim=-1)
            load = F.one_hot(top1, num_classes=num_experts).float().mean(dim=0)  # [E]
            
            aux = num_experts * torch.sum(importance * load)
        else:
            # L2: encourage uniform distribution
            p = weights.float().mean(dim=0)  # [E]
            aux = num_experts * p.pow(2).sum()
        
        aux = torch.clamp(aux, min=0.0, max=10.0)
        return aux.to(dtype=weights.dtype)

    def _collect_router_entropy(self) -> torch.Tensor:
        """Compute mean normalized entropy of routing distribution"""
        logits = getattr(self.mof_router, "last_router_logits", None)
        
        if logits is None or logits.shape[0] == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        
        num_experts = logits.shape[-1]
        probs = torch.softmax(logits.float(), dim=-1)
        
        # Per-token entropy
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        
        # Normalize by max entropy and take mean
        normalized_entropy = entropy.mean() / math.log(num_experts + 1e-9)
        
        return normalized_entropy.to(dtype=logits.dtype)

    def _lb_weight(self) -> float:
        """Get current load balancing weight with warmup"""
        w = self.mof_lb_coef
        if not self.training or self.lb_warmup_steps <= 0:
            return w
        
        t = min(1.0, float(self._global_step) / float(self.lb_warmup_steps))
        return w * t

    def _jitter(self) -> float:
        """Get current router jitter with warmup/cooldown schedule"""
        if not self.training or self.jitter_warmup_steps <= 0:
            return self._jitter_start
        
        t = min(1.0, float(self._global_step) / float(self.jitter_warmup_steps))
        return self._jitter_start * (1.0 - t) + self.jitter_end * t

    @torch.no_grad()
    def _mof_debug_stats(self):
        """Collect debug statistics for MoF routing"""
        
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            return -(p * (p + eps).log()).sum()
        
        def _cv(p: torch.Tensor) -> float:
            """Coefficient of variation"""
            mu = p.mean().clamp_min(1e-12)
            return float((p.std(unbiased=False) / mu).item())
        
        weights = getattr(self.mof_router, "last_weights", None)
        logits = getattr(self.mof_router, "last_router_logits", None)
        
        if weights is None or logits is None:
            return None
        
        E = len(self.mof_experts)
        if E <= 0:
            return None
        
        # Ensure 2D tensors
        if weights.dim() > 2:
            weights = weights.reshape(-1, E)
        if logits.dim() > 2:
            logits = logits.reshape(-1, E)
        
        # Hard assignment (argmax)
        top1 = torch.argmax(weights, dim=-1)
        counts = torch.zeros(E, device=weights.device, dtype=torch.float32)
        counts.scatter_add_(
            0,
            top1,
            torch.ones_like(top1, dtype=torch.float32),
        )
        usage_hard = counts / counts.sum().clamp_min(1.0)
        
        # Soft assignment (average routing weights)
        usage_soft = weights.mean(dim=0)
        usage_soft = usage_soft / usage_soft.sum().clamp_min(1e-12)
        
        # Entropies
        probs = torch.softmax(logits, dim=-1)
        ent_full = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
        ent_full_norm = ent_full / math.log(E)
        
        ent_hard = _entropy_from_dist(usage_hard)
        ent_hard_norm = ent_hard / math.log(E)
        
        ent_soft = _entropy_from_dist(usage_soft)
        ent_soft_norm = ent_soft / math.log(E)
        
        # Logit statistics
        logits_std = float(logits.std(unbiased=False).item())
        logits_maxabs = float(logits.abs().max().item())
        
        # Gap between top-1 and top-2
        top2 = torch.topk(logits, k=min(2, E), dim=-1).values
        if top2.size(-1) >= 2:
            gap = float((top2[:, 0] - top2[:, 1]).mean().item())
        else:
            gap = 0.0
        
        # Usage statistics
        max_hard = float(usage_hard.max().item())
        min_hard = float(usage_hard.min().item())
        max_soft = float(usage_soft.max().item())
        min_soft = float(usage_soft.min().item())
        
        return {
            "H_full_norm": float(ent_full_norm.item()),
            "H_hard_norm": float(ent_hard_norm.item()),
            "H_soft_norm": float(ent_soft_norm.item()),
            "logits_std": logits_std,
            "logits_maxabs": logits_maxabs,
            "gap_top1_top2": gap,
            "usage_hard": usage_hard.detach().cpu(),
            "usage_soft": usage_soft.detach().cpu(),
            "max_hard": max_hard,
            "min_hard": min_hard,
            "max_soft": max_soft,
            "min_soft": min_soft,
            "cv_hard": _cv(usage_hard),
            "cv_soft": _cv(usage_soft),
        }

    def print_mof_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        """Print debug information about MoF routing"""
        s = self._mof_debug_stats()
        if s is None:
            print("[MoF] No routing stats available yet.")
            return
        
        print("\n" + "="*80)
        print("[MoF Debug] Mixture of Fusion Routing Statistics")
        print("="*80)
        
        uh = s["usage_hard"].float()
        us = s["usage_soft"].float()
        
        # Count dead experts
        dead_h0 = int((uh == 0).sum().item())
        dead_h = int((uh < eps_dead).sum().item())
        dead_s0 = int((us == 0).sum().item())
        dead_s = int((us < eps_dead).sum().item())
        
        # Top-k and bottom-k experts
        topk = min(topn, uh.numel())
        botk = min(bottomn, uh.numel())
        
        topv_h, topi_h = torch.topk(uh, k=topk, largest=True)
        botv_h, boti_h = torch.topk(uh, k=botk, largest=False)
        topv_s, topi_s = torch.topk(us, k=topk, largest=True)
        botv_s, boti_s = torch.topk(us, k=botk, largest=False)
        
        # Format expert names with usage
        top_pairs_h = " ".join([
            f"{self.mof_experts[int(i)]}={float(v):.6f}" 
            for v, i in zip(topv_h, topi_h)
        ])
        bot_pairs_h = " ".join([
            f"{self.mof_experts[int(i)]}={float(v):.6f}" 
            for v, i in zip(botv_h, boti_h)
        ])
        top_pairs_s = " ".join([
            f"{self.mof_experts[int(i)]}={float(v):.6f}" 
            for v, i in zip(topv_s, topi_s)
        ])
        bot_pairs_s = " ".join([
            f"{self.mof_experts[int(i)]}={float(v):.6f}" 
            for v, i in zip(botv_s, boti_s)
        ])
        
        # Imbalance ratio
        imb_h = float(s["max_hard"] / (s["min_hard"] + 1e-12))
        imb_s = float(s["max_soft"] / (s["min_soft"] + 1e-12))
        
        # Print summary - following moehead_model format
        print(f"\n[Experts] {len(self.mof_experts)} fusion strategies: {', '.join(self.mof_experts)}")
        print(f"[Mix Level] {self.mof_mix_level}")
        
        print(
            f"\nMoF Router | "
            f"H_full={s['H_full_norm']:.6f} H_soft={s['H_soft_norm']:.6f} H_hard={s['H_hard_norm']:.6f} | "
            f"logits_std={s['logits_std']:.6f} maxabs={s['logits_maxabs']:.6f} gap12={s['gap_top1_top2']:.6f}"
        )
        
        print(
            f"  HARD: min={s['min_hard']:.6f} max={s['max_hard']:.6f} cv={s['cv_hard']:.3f} imb={imb_h:.2f} "
            f"dead(==0)={dead_h0} dead(<{eps_dead:g})={dead_h}"
        )
        print(f"    top: {top_pairs_h}")
        print(f"    bot: {bot_pairs_h}")
        
        print(
            f"  SOFT: min={s['min_soft']:.6f} max={s['max_soft']:.6f} cv={s['cv_soft']:.3f} imb={imb_s:.2f} "
            f"dead(==0)={dead_s0} dead(<{eps_dead:g})={dead_s}"
        )
        print(f"    top: {top_pairs_s}")
        print(f"    bot: {bot_pairs_s}")
        print()