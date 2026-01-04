import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.moehead_model import MoE, MoEHead


class StackedMoELayers(nn.Module):
    """Multiple MoE layers stacked together with decreasing top-k"""
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_layers: int,
        top_k_schedule: list,  # [k1, k2, k3, ...] for each layer
        dropout_p: float,
        act_fn: nn.Module,
        router_bias: bool,
        router_jitter: float,
        capacity_factor: float,
        route_mask_pad_tokens: bool,
        router_temperature: float,
    ) -> None:
        super().__init__()
        
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        
        # Validate top_k_schedule
        if len(top_k_schedule) != num_layers:
            raise ValueError(f"top_k_schedule length ({len(top_k_schedule)}) must match num_layers ({num_layers})")
        
        # Create multiple MoE layers
        self.moe_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            top_k = top_k_schedule[layer_idx]
            moe_layer = MoE(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                dropout_p=dropout_p,
                act_fn=act_fn,
                router_bias=router_bias,
                router_jitter=router_jitter,
                capacity_factor=capacity_factor,
                route_mask_pad_tokens=route_mask_pad_tokens,
                router_temperature=router_temperature,
                layer_norm=nn.LayerNorm(hidden_size),
            )
            self.moe_layers.append(moe_layer)
    
    def forward(self, hidden_states: torch.Tensor, *, token_mask) -> torch.Tensor:
        x = hidden_states
        for moe_layer in self.moe_layers:
            x = moe_layer(x, token_mask=token_mask)
        return x
    
    def set_top_k(self, k_schedule: list) -> None:
        """Update top-k for all layers"""
        if len(k_schedule) != self.num_layers:
            raise ValueError(f"k_schedule length must be {self.num_layers}")
        for i, moe_layer in enumerate(self.moe_layers):
            moe_layer.set_top_k(k_schedule[i])


class EncoderWithMultiMoEHead(nn.Module):
    """Wrapper that adds stacked MoE layers on top of base encoder output"""
    def __init__(self, *, base_encoder: nn.Module, stacked_moe: StackedMoELayers) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.stacked_moe = stacked_moe

    def forward(self, *args, **kwargs):
        outputs = self.base_encoder(*args, **kwargs)

        # Extract token mask from attention mask
        attn_mask = kwargs.get("attention_mask", None)
        token_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                m = attn_mask[:, 0, 0, :]
                if m.dtype == torch.bool:
                    token_mask = m.long()
                else:
                    m_f = m.float()
                    if torch.min(m_f) < 0.0 and torch.max(m_f) <= 0.0:
                        token_mask = (m_f == 0.0).long()
                    else:
                        token_mask = (m_f > 0.0).long()
            else:
                token_mask = attn_mask

        # Pass through stacked MoE layers
        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            new_hidden = self.stacked_moe(last_hidden, token_mask=token_mask)
            return (new_hidden,) + tuple(outputs[1:])

        if hasattr(outputs, "last_hidden_state"):
            new_hidden = self.stacked_moe(outputs.last_hidden_state, token_mask=token_mask)
            outputs.last_hidden_state = new_hidden
            return outputs

        return self.stacked_moe(outputs, token_mask=token_mask)


class MultiMoEHead(MoEHead):
    """Model with MoE layers embedded in multiple MoE heads"""
    
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str,
        class_weights,
        focal_gamma: float,
        aux_loss_weight: float,
        num_experts: int,
        multi_moe_top_k: int,  # Starting top_k for first layer
        num_moe_layers: int = 3,  # Number of stacked MoE layers
        top_k_decay: str = "linear",  # "linear", "exponential", or "custom"
        top_k_schedule: list = None,  # Custom schedule [k1, k2, k3, ...]
        router_bias: bool = True,
        router_jitter: float = 0.0,
        capacity_factor: float = None,
        route_mask_pad_tokens: bool = True,
        router_temperature: float = 1.0,
        router_entropy_weight: float = 0.0,
        aux_warmup_steps: int = 0,
        jitter_warmup_steps: int = 0,
        jitter_end: float = 0.0,
    ) -> None:
        # We need to override parent's __init__ to avoid creating single MoE
        # Call grandparent's __init__ directly
        from src.models.base_model import BaseModel
        BaseModel.__init__(
            self,
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            loss_type=loss_type,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
        
        self.aux_loss_weight = aux_loss_weight
        self.router_entropy_weight = float(router_entropy_weight)
        self.aux_warmup_steps = int(aux_warmup_steps)
        self.jitter_warmup_steps = int(jitter_warmup_steps)
        self.jitter_end = float(jitter_end)
        self._global_step = 0
        self.num_moe_layers = int(num_moe_layers)

        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))
        self._jitter_start = float(router_jitter)

        # Generate top-k schedule
        if top_k_schedule is not None:
            if len(top_k_schedule) != num_moe_layers:
                raise ValueError(f"Custom top_k_schedule must have length {num_moe_layers}")
            self.top_k_schedule = top_k_schedule
        else:
            self.top_k_schedule = self._generate_top_k_schedule(
                start_k=multi_moe_top_k,
                num_layers=num_moe_layers,
                num_experts=num_experts,
                decay_type=top_k_decay
            )
        
        print(f"[MultiMoEHead] Using top-k schedule: {self.top_k_schedule}")

        # Create stacked MoE layers
        stacked_moe = StackedMoELayers(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_layers=num_moe_layers,
            top_k_schedule=self.top_k_schedule,
            dropout_p=dropout_p,
            act_fn=act_fn,
            router_bias=router_bias,
            router_jitter=router_jitter,
            capacity_factor=capacity_factor,
            route_mask_pad_tokens=route_mask_pad_tokens,
            router_temperature=router_temperature,
        )

        # Replace encoder with multi-MoE version
        base_encoder = self.encoder
        self.encoder = EncoderWithMultiMoEHead(base_encoder=base_encoder, stacked_moe=stacked_moe)

    def _generate_top_k_schedule(self, start_k: int, num_layers: int, num_experts: int, decay_type: str) -> list:
        """Generate decreasing top-k schedule for multiple layers"""
        schedule = []
        
        if decay_type == "linear":
            # Linear decay from start_k to 1
            for i in range(num_layers):
                k = max(1, start_k - int((start_k - 1) * i / max(1, num_layers - 1)))
                schedule.append(min(k, num_experts))
        
        elif decay_type == "exponential":
            # Exponential decay
            for i in range(num_layers):
                ratio = i / max(1, num_layers - 1)
                k = max(1, int(start_k * (0.5 ** ratio)))
                schedule.append(min(k, num_experts))
        
        else:  # constant or unknown
            schedule = [min(start_k, num_experts)] * num_layers
        
        return schedule

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        if self.training:
            self._global_step += 1
            stacked_moe = getattr(self.encoder, "stacked_moe", None)
            if stacked_moe is not None:
                jitter_val = float(self._jitter())
                for moe_layer in stacked_moe.moe_layers:
                    moe_layer.router_jitter = jitter_val

        # Call grandparent's forward (BaseModel.forward)
        from src.models.base_model import BaseModel
        return BaseModel.forward(
            self,
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )

    def _collect_aux_loss(self):
        """Load-balancing loss for all MoE layers"""
        stacked_moe = getattr(self.encoder, "stacked_moe", None)
        if stacked_moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        total_aux = torch.zeros((), device=next(self.parameters()).device)
        
        for moe_layer in stacked_moe.moe_layers:
            logits = getattr(moe_layer, "last_router_logits", None)
            topk_idx = getattr(moe_layer, "last_topk_idx", None)
            
            if logits is None or topk_idx is None:
                continue
            
            if logits.ndim != 2 or topk_idx.ndim != 2 or logits.shape[0] == 0:
                continue

            n_tokens, n_experts = logits.shape
            k = topk_idx.shape[1]

            probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
            importance = probs.sum(dim=0) / float(n_tokens)

            oh = F.one_hot(topk_idx, num_classes=n_experts).to(dtype=probs.dtype)
            load = oh.sum(dim=(0, 1)) / float(n_tokens * k)

            aux = n_experts * torch.sum(importance * load)
            aux = torch.clamp(aux, min=0.0, max=10.0)
            total_aux = total_aux + aux

        return total_aux / float(self.num_moe_layers)

    def _collect_router_entropy(self) -> torch.Tensor:
        """Mean per-token routing entropy for all MoE layers"""
        stacked_moe = getattr(self.encoder, "stacked_moe", None)
        if stacked_moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        total_ent = torch.zeros((), device=next(self.parameters()).device)
        count = 0

        for moe_layer in stacked_moe.moe_layers:
            logits = getattr(moe_layer, "last_router_logits", None)
            if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
                continue

            n_experts = logits.shape[-1]
            probs = torch.softmax(logits.float(), dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
            ent = ent.mean()
            ent = ent / float(math.log(n_experts + 1e-9))
            total_ent = total_ent + ent.to(dtype=logits.dtype, device=logits.device)
            count += 1

        if count > 0:
            return total_ent / float(count)
        return total_ent

    @torch.no_grad()
    def _moe_debug_stats(self):
        """Debug statistics for all MoE layers"""
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            return -(p * (p + eps).log()).sum()

        def _cv(p: torch.Tensor) -> float:
            mu = p.mean().clamp_min(1e-12)
            return float((p.std(unbiased=False) / mu).item())

        stacked_moe = getattr(self.encoder, "stacked_moe", None)
        if stacked_moe is None:
            return None

        all_stats = []

        for layer_idx, moe_layer in enumerate(stacked_moe.moe_layers):
            logits = getattr(moe_layer, "last_router_logits", None)
            topk_idx = getattr(moe_layer, "last_topk_idx", None)
            
            if logits is None or topk_idx is None:
                continue

            E = int(getattr(moe_layer, "num_experts", 0) or 0)
            if E <= 0:
                continue

            logits2 = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
            topk2 = topk_idx.reshape(-1, topk_idx.size(-1)) if topk_idx.dim() == 3 else topk_idx

            probs = torch.softmax(logits2, dim=-1)

            counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk2.reshape(-1),
                torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
            )
            usage_hard = counts / counts.sum().clamp_min(1.0)

            usage_soft = probs.mean(dim=0)
            usage_soft = usage_soft / usage_soft.sum().clamp_min(1e-12)

            ent_full = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
            ent_full_norm = ent_full / math.log(E)

            ent_hard = _entropy_from_dist(usage_hard)
            ent_hard_norm = ent_hard / math.log(E)

            ent_soft = _entropy_from_dist(usage_soft)
            ent_soft_norm = ent_soft / math.log(E)

            logits_std = float(logits2.std(unbiased=False).item())
            logits_maxabs = float(logits2.abs().max().item())

            top2 = torch.topk(logits2, k=min(2, E), dim=-1).values
            if top2.size(-1) >= 2:
                gap = float((top2[:, 0] - top2[:, 1]).mean().item())
            else:
                gap = 0.0

            max_hard = float(usage_hard.max().item())
            min_hard = float(usage_hard.min().item())
            max_soft = float(usage_soft.max().item())
            min_soft = float(usage_soft.min().item())

            stats = {
                "layer": layer_idx,
                "top_k": int(getattr(moe_layer, "multi_moe_top_k", 0)),
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
            all_stats.append(stats)

        return all_stats if all_stats else None

    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        all_stats = self._moe_debug_stats()
        if all_stats is None or len(all_stats) == 0:
            print("[MoE] No stats yet.")
            return

        print(f"\n[MoE Debug - Multi-Head with {len(all_stats)} layers]")
        
        for s in all_stats:
            uh = s["usage_hard"].float()
            us = s["usage_soft"].float()

            dead_h0 = int((uh == 0).sum().item())
            dead_h = int((uh < eps_dead).sum().item())
            dead_s0 = int((us == 0).sum().item())
            dead_s = int((us < eps_dead).sum().item())

            topk = min(topn, uh.numel())
            botk = min(bottomn, uh.numel())

            topv_h, topi_h = torch.topk(uh, k=topk, largest=True)
            botv_h, boti_h = torch.topk(uh, k=botk, largest=False)
            topv_s, topi_s = torch.topk(us, k=topk, largest=True)
            botv_s, boti_s = torch.topk(us, k=botk, largest=False)

            top_pairs_h = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv_h, topi_h)])
            bot_pairs_h = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv_h, boti_h)])
            top_pairs_s = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv_s, topi_s)])
            bot_pairs_s = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv_s, boti_s)])

            imb_h = float(s["max_hard"] / (s["min_hard"] + 1e-12))
            imb_s = float(s["max_soft"] / (s["min_soft"] + 1e-12))

            print(
                f"Layer {s['layer']} (top_k={s['top_k']}) | "
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

    def configure_topk_schedule(
        self,
        *,
        enabled: bool,
        start_k_schedule: list,
        end_k_schedule: list,
        switch_epoch: int,
    ) -> None:
        """Configure top-k schedule for all layers during training"""
        self._topk_schedule_enabled = bool(enabled)
        
        if len(start_k_schedule) != self.num_moe_layers:
            raise ValueError(f"start_k_schedule must have length {self.num_moe_layers}")
        if len(end_k_schedule) != self.num_moe_layers:
            raise ValueError(f"end_k_schedule must have length {self.num_moe_layers}")
        
        self._topk_start_schedule = start_k_schedule
        self._topk_end_schedule = end_k_schedule
        self._topk_switch_epoch = int(switch_epoch)

        if self._topk_schedule_enabled:
            self.encoder.stacked_moe.set_top_k(self._topk_start_schedule)
        else:
            self.encoder.stacked_moe.set_top_k(self._topk_end_schedule)

    def set_epoch(self, epoch_idx_0based: int) -> None:
        """Update top-k schedule based on current epoch"""
        if not getattr(self, "_topk_schedule_enabled", False):
            return

        if epoch_idx_0based < self._topk_switch_epoch:
            k_schedule = self._topk_start_schedule
        else:
            k_schedule = self._topk_end_schedule

        self.encoder.stacked_moe.set_top_k(k_schedule)