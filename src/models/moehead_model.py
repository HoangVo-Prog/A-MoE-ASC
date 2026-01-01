import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel
from .moeffn_model import moe_load_balance_loss


class MoE(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dropout_p: float,
        act_fn: nn.Module,
        router_bias: bool,
        router_jitter: float ,
        capacity_factor: float,
        route_mask_pad_tokens,
        layer_norm,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.moe_top_k = int(top_k)
        self.dropout = nn.Dropout(float(dropout_p))
        self.act_fn = act_fn
        self.router_jitter = float(router_jitter)
        self.capacity_factor = capacity_factor
        self.route_mask_pad_tokens = bool(route_mask_pad_tokens)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=bool(router_bias))

        self.experts_dense1 = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.intermediate_size) for _ in range(self.num_experts)]
        )
        self.experts_dense2 = nn.ModuleList(
            [nn.Linear(self.intermediate_size, self.hidden_size) for _ in range(self.num_experts)]
        )

        self.ln = layer_norm if layer_norm is not None else nn.LayerNorm(self.hidden_size)

        self.last_router_logit = None
        self.last_topk_idx = None

        # init router near uniform
        nn.init.zeros_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(self, hidden_states: torch.Tensor, *, token_mask) -> torch.Tensor:
        bsz, seqlen, hdim = hidden_states.shape
        x = hidden_states.reshape(-1, hdim)  # [N, H]

        active_idx = None
        x_active = x

        if self.route_mask_pad_tokens and token_mask is not None:
            m = token_mask.reshape(-1).bool()
            if not torch.any(m):
                self.last_router_logits = None
                self.last_topk_idx = None
                return self.ln(hidden_states)
            active_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            x_active = x.index_select(0, active_idx)

        if self.router_jitter > 0.0:
            noise = (torch.rand_like(x_active) - 0.5) * 2.0 * self.router_jitter
            x_route = x_active + noise
        else:
            x_route = x_active

        router_logits = self.router(x_route)  # [N_active, E]
        topk_vals, topk_idx = torch.topk(router_logits, k=self.moe_top_k, dim=-1)  # [N_active, K]
        topk_w = torch.softmax(topk_vals, dim=-1)  # [N_active, K]

        # cache for aux loss and debug
        self.last_router_logits = router_logits
        self.last_topk_idx = topk_idx 

        cap = None
        if self.capacity_factor is not None:
            cap = int(math.ceil((x_active.shape[0] / self.num_experts) * float(self.capacity_factor)))
            cap = max(cap, 1)

        out_active = torch.zeros_like(x_active)

        flat_idx = topk_idx.reshape(-1)  # [N_active*K]
        flat_tok = torch.arange(x_active.shape[0], device=x_active.device).repeat_interleave(self.moe_top_k)
        flat_kpos = torch.arange(self.moe_top_k, device=x_active.device).repeat(x_active.shape[0])

        for e in range(self.num_experts):
            sel = (flat_idx == e)
            if not torch.any(sel):
                continue

            tok_pos = flat_tok[sel]
            k_pos = flat_kpos[sel]

            if cap is not None and tok_pos.numel() > cap:
                tok_pos = tok_pos[:cap]
                k_pos = k_pos[:cap]

            xe = x_active.index_select(0, tok_pos)
            y = self.experts_dense1[e](xe)
            y = self.act_fn(y)
            y = self.experts_dense2[e](y)
            y = self.dropout(y)

            w = topk_w.index_select(0, tok_pos).gather(1, k_pos.unsqueeze(1)).squeeze(1)  # [M]
            out_active.index_add_(0, tok_pos, y * w.unsqueeze(1))

        if active_idx is not None:
            out = x.clone()
            out.index_copy_(0, active_idx, out_active)
        else:
            out = out_active

        out = out.reshape(bsz, seqlen, hdim)
        return self.ln(out + hidden_states)

    def set_top_k(self, k: int) -> None:
        k = int(k)
        if k < 1:
            k = 1
        if k > self.num_experts:
            k = self.num_experts
        self.moe_top_k = k

class EncoderWithMoEHead(nn.Module):
    def __init__(self, *, base_encoder: nn.Module, moe_ffn: MoE) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffn = moe_ffn

    def forward(self, *args, **kwargs):
        outputs = self.base_encoder(*args, **kwargs)

        attn_mask = kwargs.get("attention_mask", None)
        token_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                token_mask = (attn_mask[:, 0, 0, :] == 0).long()
            else:
                token_mask = attn_mask

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            new_hidden = self.moe_ffn(last_hidden, token_mask=token_mask)
            return (new_hidden,) + tuple(outputs[1:])

        if hasattr(outputs, "last_hidden_state"):
            new_hidden = self.moe_ffn(outputs.last_hidden_state, token_mask=token_mask)
            outputs.last_hidden_state = new_hidden
            return outputs

        return self.moe_ffn(outputs, token_mask=token_mask)


class MoEHead(BaseModel):
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
        capacity_factor,
        route_mask_pad_tokens: bool,
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
        
        self.aux_loss_weight=aux_loss_weight

        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn: nn.Module = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))
                
        self._topk_schedule_enabled = moe_topk_schedule
        self._topk_start = moe_topk_start
        self._topk_end = moe_topk_end
        self._topk_switch_epoch = moe_topk_switch_epoch

        moe_head = MoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=moe_top_k,
            dropout_p=dropout_p,
            act_fn=act_fn,
            router_bias=router_bias,
            router_jitter=router_jitter,
            capacity_factor=capacity_factor,
            route_mask_pad_tokens=route_mask_pad_tokens,
            layer_norm=nn.LayerNorm(hidden_size),
        )

        base_encoder = self.encoder
        self.encoder = EncoderWithMoEHead(base_encoder=base_encoder, moe_ffn=moe_head)

    def _switch_aux_loss_topk(
        router_logits: torch.Tensor,
        topk_idx: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Switch-style load balancing loss for top-k routing.

        router_logits: [B,T,E] or [N,E]
        topk_idx:      [B,T,K] or [N,K]
        token_mask:    [B,T] or [N] (optional). True/1 = keep token.
        Returns scalar aux loss.
        """
        if router_logits is None or topk_idx is None:
            return torch.tensor(0.0, device=router_logits.device if router_logits is not None else "cpu")

        # Flatten to [N, E] and [N, K]
        if router_logits.dim() == 3:
            B, T, E = router_logits.shape
            logits2 = router_logits.reshape(B * T, E)
            topk2 = topk_idx.reshape(B * T, -1)
            mask2 = token_mask.reshape(B * T) if token_mask is not None else None
        else:
            E = router_logits.size(-1)
            logits2 = router_logits
            topk2 = topk_idx
            mask2 = token_mask if token_mask is not None else None

        # Optional token mask
        if mask2 is not None:
            if mask2.dtype != torch.bool:
                mask2 = mask2 != 0
            if mask2.any():
                logits2 = logits2[mask2]
                topk2 = topk2[mask2]
            else:
                # no valid tokens
                return torch.tensor(0.0, device=router_logits.device)

        N = logits2.size(0)
        if N == 0:
            return torch.tensor(0.0, device=router_logits.device)

        # Soft importance
        probs = F.softmax(logits2, dim=-1)  # [N, E]
        importance = probs.sum(dim=0)  # [E]
        importance = importance / importance.sum().clamp_min(eps)

        # Hard load from top-k assignment
        counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
        ones = torch.ones_like(topk2.reshape(-1), dtype=torch.float32, device=logits2.device)
        counts.scatter_add_(0, topk2.reshape(-1), ones)
        load = counts / counts.sum().clamp_min(eps)

        # Switch aux loss
        aux = float(E) * torch.sum(importance * load)
        return aux

    def _collect_aux_loss(self):
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None or moe.last_topk_idx is None:
            return torch.tensor(0.0, device=self.device)

        topk_idx = moe.last_topk_idx  # [N_active, K]
        E = int(getattr(moe, "num_experts", 0) or 0)
        if E <= 0:
            return torch.tensor(0.0, device=self.device)

        device = topk_idx.device
        eps = 1e-12

        # hard counts over top-k selections
        flat = topk_idx.reshape(-1)
        counts = torch.zeros(E, device=device, dtype=torch.float32)
        ones = torch.ones_like(flat, dtype=torch.float32, device=device)
        counts.scatter_add_(0, flat, ones)

        total = counts.sum().clamp_min(1.0)
        load = counts / total  # sum = 1

        # hard load penalty: uniform -> 1, imbalanced -> >1
        aux = float(E) * torch.sum(load * load)

        # safety
        if not torch.isfinite(aux):
            aux = torch.zeros((), device=device, dtype=torch.float32)

        return aux

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

    @torch.no_grad()
    def _moe_debug_stats_per_layer(self):
        def _entropy_from_dist(p: torch.Tensor) -> torch.Tensor:
            # p: [E], sum=1
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

            # logits: [B, T, E] or [N, E]
            logits2 = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
            # topk_idx: [B, T, K] or [N, K]
            topk2 = topk_idx.reshape(-1, topk_idx.size(-1)) if topk_idx.dim() == 3 else topk_idx

            # Soft distribution per token
            probs = torch.softmax(logits2, dim=-1)  # [N, E]

            # 1) Hard usage from topk
            counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk2.reshape(-1),
                torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
            )
            usage_hard = counts / counts.sum().clamp_min(1.0)

            # 2) Soft expected usage from probs
            usage_soft = probs.mean(dim=0)  # [E]
            usage_soft = usage_soft / usage_soft.sum().clamp_min(1e-12)

            # Entropy
            ent_full = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()  # scalar
            ent_full_norm = ent_full / math.log(E)

            ent_hard = _entropy_from_dist(usage_hard)
            ent_hard_norm = ent_hard / math.log(E)

            ent_soft = _entropy_from_dist(usage_soft)
            ent_soft_norm = ent_soft / math.log(E)

            # Logits stats
            logits_std = float(logits2.std(unbiased=False).item())
            logits_maxabs = float(logits2.abs().max().item())

            # Margin top1-top2 (how sharp routing is)
            top2 = torch.topk(logits2, k=min(2, E), dim=-1).values  # [N,2]
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

                # entropy
                "H_full_norm": float(ent_full_norm.item()),
                "H_hard_norm": float(ent_hard_norm.item()),
                "H_soft_norm": float(ent_soft_norm.item()),

                # logits
                "logits_std": logits_std,
                "logits_maxabs": logits_maxabs,
                "gap_top1_top2": gap,

                # usage
                "usage_hard": usage_hard.detach().cpu(),
                "usage_soft": usage_soft.detach().cpu(),

                # min max
                "max_hard": max_hard,
                "min_hard": min_hard,
                "max_soft": max_soft,
                "min_soft": min_soft,

                # dispersion
                "cv_hard": _cv(usage_hard),
                "cv_soft": _cv(usage_soft),
            }

        stats = []

        # Case 1: MoEHead style
        enc = getattr(self, "encoder", None)
        head_moe = getattr(enc, "moe_ffn", None) if enc is not None else None
        s_head = _stats_from_moe(head_moe, layer_id=-1)
        if s_head is not None:
            stats.append(s_head)
            return stats

        # Case 2: per-layer MoE FFN style
        base = enc
        if base is None:
            return stats

        base = getattr(base, "base_encoder", base)

        layers = None
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            layers = base.encoder.layer
        elif hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
            layers = base.encoder.layers
        elif hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
            layers = base.transformer.layer
        elif hasattr(base, "transformer") and hasattr(base.transformer, "layers"):
            layers = base.transformer.layers
        elif hasattr(base, "layers"):
            layers = base.layers

        if layers is None:
            return stats

        for li, layer in enumerate(layers):
            moe = getattr(layer, "moe_ffn", None)
            s = _stats_from_moe(moe, layer_id=li)
            if s is not None:
                stats.append(s)

        return stats


    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            print("[MoE] No stats yet.")
            return

        print("\n[MoE Debug]")
        for s in stats:
            layer_id = int(s["layer"])
            layer_txt = "Head" if layer_id == -1 else f"{layer_id:02d}"

            uh = s["usage_hard"].float()
            us = s["usage_soft"].float()

            # dead counts
            dead_h0 = int((uh == 0).sum().item())
            dead_h = int((uh < eps_dead).sum().item())
            dead_s0 = int((us == 0).sum().item())
            dead_s = int((us < eps_dead).sum().item())

            # top/bottom
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

            # imbalance
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
        self._topk_schedule_enabled = bool(enabled)
        self._topk_start = int(start_k)
        self._topk_end = int(end_k)
        self._topk_switch_epoch = int(switch_epoch)

        if self._topk_schedule_enabled:
            self.encoder.moe_ffn.set_top_k(self._topk_start)
        else:
            self.encoder.moe_ffn.set_top_k(self._topk_end)

    def set_epoch(self, epoch_idx_0based: int) -> None:
        if not getattr(self, "_topk_schedule_enabled", False):
            return

        if epoch_idx_0based < self._topk_switch_epoch:
            k = self._topk_start
        else:
            k = self._topk_end

        self.encoder.moe_ffn.set_top_k(k)
