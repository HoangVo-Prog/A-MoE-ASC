import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.moehead_model import MoE, MoEHead


class TransformerLayerWithMoE(nn.Module):
    """Wrapper that adds MoE FFN to a transformer layer"""
    def __init__(self, *, base_layer: nn.Module, moe_ffn: MoE) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.moe_ffn = moe_ffn
        
    def forward(self, *args, **kwargs):
        # Forward through original layer
        outputs = self.base_layer(*args, **kwargs)
        
        # Extract attention mask for routing
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
        
        # Apply MoE FFN to hidden states
        if isinstance(outputs, (tuple, list)):
            hidden = outputs[0]
            new_hidden = self.moe_ffn(hidden, token_mask=token_mask)
            return (new_hidden,) + tuple(outputs[1:])
        
        if hasattr(outputs, "last_hidden_state"):
            new_hidden = self.moe_ffn(outputs.last_hidden_state, token_mask=token_mask)
            outputs.last_hidden_state = new_hidden
            return outputs
        
        return self.moe_ffn(outputs, token_mask=token_mask)


class EncoderWithMultiMoE(nn.Module):
    """Wrapper that embeds MoE layers into multiple transformer layers"""
    def __init__(self, *, base_encoder: nn.Module, moe_layer_indices: list, moe_configs: dict) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_layer_indices = sorted(moe_layer_indices)
        
        # Find the layers list in base encoder
        self.layers = self._find_layers(base_encoder)
        if self.layers is None:
            raise ValueError("Cannot find transformer layers in base encoder")
        
        # Wrap specified layers with MoE
        for idx in self.moe_layer_indices:
            if idx >= len(self.layers):
                raise ValueError(f"Layer index {idx} out of range (encoder has {len(self.layers)} layers)")
            
            # Create MoE module for this layer
            moe = MoE(**moe_configs)
            
            # Wrap the layer
            original_layer = self.layers[idx]
            self.layers[idx] = TransformerLayerWithMoE(
                base_layer=original_layer,
                moe_ffn=moe
            )
    
    def _find_layers(self, encoder):
        """Find the layers list in various encoder architectures"""
        # Try common patterns
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            return encoder.encoder.layer
        elif hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
            return encoder.encoder.layers
        elif hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
            return encoder.transformer.layer
        elif hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layers"):
            return encoder.transformer.layers
        elif hasattr(encoder, "layers"):
            return encoder.layers
        return None
    
    def forward(self, *args, **kwargs):
        return self.base_encoder(*args, **kwargs)


class MultiMoEHead(MoEHead):
    """Model with MoE layers embedded in multiple transformer layers"""
    
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
        moe_top_k: int,
        moe_topk_schedule: bool,
        moe_topk_start: int,
        moe_topk_end: int,
        moe_topk_switch_epoch: int,
        router_bias: bool,
        router_jitter: float,
        capacity_factor: float,
        route_mask_pad_tokens: bool,
        router_entropy_weight: float = 0.0,
        aux_warmup_steps: int = 0,
        jitter_warmup_steps: int = 0,
        jitter_end: float = 0.0,
        moe_layer_indices: list = None,  # Which layers to add MoE to
    ) -> None:
        # Call grandparent init to avoid MoEHead's single-layer setup
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
        
        # MoE-specific configs
        self.aux_loss_weight = aux_loss_weight
        self.router_entropy_weight = float(router_entropy_weight)
        self.aux_warmup_steps = int(aux_warmup_steps)
        self.jitter_warmup_steps = int(jitter_warmup_steps)
        self.jitter_end = float(jitter_end)
        self._jitter_start = float(router_jitter)
        self._global_step = 0
        
        # Top-k schedule
        self._topk_schedule_enabled = moe_topk_schedule
        self._topk_start = moe_topk_start
        self._topk_end = moe_topk_end
        self._topk_switch_epoch = moe_topk_switch_epoch
        
        # Default to last 3 layers if not specified
        if moe_layer_indices is None:
            cfg = getattr(self.encoder, "config", None)
            num_layers = getattr(cfg, "num_hidden_layers", 12)
            moe_layer_indices = list(range(max(0, num_layers - 3), num_layers))
        
        self.moe_layer_indices = moe_layer_indices
        
        # Get encoder config
        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))
        
        # Prepare MoE config
        moe_configs = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": moe_top_k if not moe_topk_schedule else moe_topk_start,
            "dropout_p": dropout_p,
            "act_fn": act_fn,
            "router_bias": router_bias,
            "router_jitter": router_jitter,
            "capacity_factor": capacity_factor,
            "route_mask_pad_tokens": route_mask_pad_tokens,
            "layer_norm": nn.LayerNorm(hidden_size),
        }
        
        # Wrap encoder with multi-layer MoE
        base_encoder = self.encoder
        self.encoder = EncoderWithMultiMoE(
            base_encoder=base_encoder,
            moe_layer_indices=moe_layer_indices,
            moe_configs=moe_configs
        )
    
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
            # Update jitter for all MoE layers
            new_jitter = float(self._jitter())
            for moe in self._get_all_moe_modules():
                moe.router_jitter = new_jitter
        
        return MoEHead.forward(
            self,
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )
    
    def _get_all_moe_modules(self):
        """Get all MoE modules from wrapped layers"""
        moe_modules = []
        layers = self.encoder.layers
        for idx in self.moe_layer_indices:
            if idx < len(layers):
                layer = layers[idx]
                if isinstance(layer, TransformerLayerWithMoE):
                    moe_modules.append(layer.moe_ffn)
        return moe_modules
    
    def _get_moe_by_layer_id(self, layer_id: int):
        """Helper to retrieve MoE module by layer ID"""
        if layer_id < 0 or layer_id >= len(self.encoder.layers):
            return None
        
        layer = self.encoder.layers[layer_id]
        if isinstance(layer, TransformerLayerWithMoE):
            return layer.moe_ffn
        return None
    
    @torch.no_grad()
    def _moe_debug_stats_per_layer(self):
        """Collect debug statistics from all MoE layers in the model"""
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            eps = 1e-12
            return -(p * (p + eps).log()).sum()

        def _cv(p: torch.Tensor) -> float:
            mu = p.mean().clamp_min(1e-12)
            return float((p.std(unbiased=False) / mu).item())

        def _stats_from_moe(moe, layer_id):
            if moe is None:
                return None
            logits = getattr(moe, "last_router_logits", None)
            topk_idx = getattr(moe, "last_topk_idx", None)
            if logits is None or topk_idx is None:
                return None

            E = int(getattr(moe, "num_experts", 0) or 0)
            if E <= 0:
                return None

            logits2 = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
            topk2 = topk_idx.reshape(-1, topk_idx.size(-1)) if topk_idx.dim() == 3 else topk_idx

            probs = torch.softmax(logits2, dim=-1)

            # Hard usage from topk
            counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk2.reshape(-1),
                torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
            )
            usage_hard = counts / counts.sum().clamp_min(1.0)

            # Soft expected usage from probs
            usage_soft = probs.mean(dim=0)
            usage_soft = usage_soft / usage_soft.sum().clamp_min(1e-12)

            # Entropy
            ent_full = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
            ent_full_norm = ent_full / math.log(E)

            ent_hard = _entropy_from_dist(usage_hard)
            ent_hard_norm = ent_hard / math.log(E)

            ent_soft = _entropy_from_dist(usage_soft)
            ent_soft_norm = ent_soft / math.log(E)

            # Logits stats
            logits_std = float(logits2.std(unbiased=False).item())
            logits_maxabs = float(logits2.abs().max().item())

            # Margin top1-top2
            top2 = torch.topk(logits2, k=min(2, E), dim=-1).values
            if top2.size(-1) >= 2:
                gap = float((top2[:, 0] - top2[:, 1]).mean().item())
            else:
                gap = 0.0

            # Loads
            max_hard = float(usage_hard.max().item())
            min_hard = float(usage_hard.min().item())
            max_soft = float(usage_soft.max().item())
            min_soft = float(usage_soft.min().item())

            return {
                "layer": int(layer_id),
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

        stats = []
        
        # Collect stats from each MoE layer
        for layer_id in self.moe_layer_indices:
            moe = self._get_moe_by_layer_id(layer_id)
            s = _stats_from_moe(moe, layer_id=layer_id)
            if s is not None:
                stats.append(s)

        return stats

    def _collect_aux_loss(self):
        """Collect auxiliary loss from all MoE layers"""
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            return torch.zeros((), device=next(self.parameters()).device)
        
        # Aggregate aux loss from all layers
        total_aux = torch.zeros((), device=next(self.parameters()).device)
        
        for s in stats:
            layer_moe = self._get_moe_by_layer_id(s["layer"])
            if layer_moe is None:
                continue
            
            logits = getattr(layer_moe, "last_router_logits", None)
            topk_idx = getattr(layer_moe, "last_topk_idx", None)
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
        
        # Average across layers
        if len(stats) > 0:
            total_aux = total_aux / float(len(stats))
        
        return total_aux

    def _collect_router_entropy(self) -> torch.Tensor:
        """Collect mean routing entropy across all MoE layers"""
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            return torch.zeros((), device=next(self.parameters()).device)
        
        total_entropy = torch.zeros((), device=next(self.parameters()).device)
        
        for s in stats:
            layer_moe = self._get_moe_by_layer_id(s["layer"])
            if layer_moe is None:
                continue
            
            logits = getattr(layer_moe, "last_router_logits", None)
            if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
                continue

            n_experts = logits.shape[-1]
            probs = torch.softmax(logits.float(), dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
            ent = ent.mean()
            ent = ent / float(math.log(n_experts + 1e-9))
            total_entropy = total_entropy + ent.to(dtype=logits.dtype, device=logits.device)
        
        # Average across layers
        if len(stats) > 0:
            total_entropy = total_entropy / float(len(stats))
        
        return total_entropy

    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        """Print debug statistics for all MoE layers"""
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            print("[MoE] No stats yet.")
            return

        print("\n[MoE Debug - Multi-Layer]")
        for s in stats:
            layer_id = int(s["layer"])
            layer_txt = f"{layer_id:02d}"

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
                f"Layer {layer_txt} | "
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
        start_k: int,
        end_k: int,
        switch_epoch: int,
    ) -> None:
        """Configure top-k scheduling for all MoE layers"""
        self._topk_schedule_enabled = bool(enabled)
        self._topk_start = int(start_k)
        self._topk_end = int(end_k)
        self._topk_switch_epoch = int(switch_epoch)

        # Apply to all MoE layers
        k = self._topk_start if self._topk_schedule_enabled else self._topk_end
        for moe in self._get_all_moe_modules():
            moe.set_top_k(k)

    def set_epoch(self, epoch_idx_0based: int) -> None:
        """Update top-k based on epoch for all MoE layers"""
        if not getattr(self, "_topk_schedule_enabled", False):
            return

        k = self._topk_start if epoch_idx_0based < self._topk_switch_epoch else self._topk_end
        
        for moe in self._get_all_moe_modules():
            moe.set_top_k(k)