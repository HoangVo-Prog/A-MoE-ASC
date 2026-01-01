import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.core.loss.focal_loss import FocalLoss
from src.models.base_model import BaseModel
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
        router_jitter: float,
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

        self.last_router_logits = None
        self.last_topk_idx = None

        # init router near-uniform, but break symmetry (important for top-k routing)
        # If all router logits are exactly equal (e.g., all-zero weights), torch.topk will pick a fixed subset
        # of experts, causing dead experts and huge imbalance metrics. Use tiny random init instead.
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
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
                # Prefer keeping tokens with higher routing weight to reduce arbitrary drops
                w_sel = topk_w.index_select(0, tok_pos).gather(1, k_pos.unsqueeze(1)).squeeze(1)  # [M]
                keep = torch.topk(w_sel, k=cap, largest=True, sorted=False).indices
                tok_pos = tok_pos.index_select(0, keep)
                k_pos = k_pos.index_select(0, keep)

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

    def _collect_aux_loss(self):
        """Switch-style load-balancing loss adapted to top-k routing.

        Uses:
          - router logits (for differentiable importance via softmax probs)
          - hard top-k indices (for load via assignment counts)

        This tends to reduce expert collapse without requiring gradients through hard routing.
        """
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None:
            return torch.zeros((), device=next(self.parameters()).device)

        logits = getattr(moe, "last_router_logits", None)
        topk_idx = getattr(moe, "last_topk_idx", None)
        if logits is None or topk_idx is None:
            return torch.zeros((), device=next(self.parameters()).device)

        # logits: [N_active, E], topk_idx: [N_active, K]
        if logits.ndim != 2 or topk_idx.ndim != 2 or logits.shape[0] == 0:
            return torch.zeros((), device=logits.device)

        n_tokens, n_experts = logits.shape
        k = topk_idx.shape[1]

        # importance: differentiable
        probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)  # [N, E]
        importance = probs.sum(dim=0) / float(n_tokens)  # [E]

        # load: hard top-k counts (non-differentiable)
        oh = F.one_hot(topk_idx, num_classes=n_experts).to(dtype=probs.dtype)  # [N, K, E]
        load = oh.sum(dim=(0, 1)) / float(n_tokens * k)  # [E]

        aux = n_experts * torch.sum(importance * load)

        # optional safety clamp to avoid exploding logs if routing collapses early
        aux = torch.clamp(aux, min=0.0, max=10.0)
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

        stats = []
        E = None

        for li, layer in enumerate(self.encoder.encoder.layer):
            moe = getattr(layer, "moe_ffn", None)
            if moe is None:
                continue
            if moe.last_router_logits is None or moe.last_topk_idx is None:
                continue

            logits = moe.last_router_logits
            topk_idx = moe.last_topk_idx
            E = moe.num_experts

            if logits.dim() == 3:
                logits2 = logits.reshape(-1, logits.size(-1))
            else:
                logits2 = logits

            if topk_idx.dim() == 3:
                topk2 = topk_idx.reshape(-1, topk_idx.size(-1))
            else:
                topk2 = topk_idx

            probs = torch.softmax(logits2, dim=-1)

            eps = 1e-9
            ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
            ent_norm = ent / math.log(E)

            counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
            counts.scatter_add_(
                0,
                topk2.reshape(-1),
                torch.ones_like(topk2.reshape(-1), dtype=torch.float32),
            )
            usage = counts / counts.sum().clamp_min(1.0)

            stats.append(
                {
                    "layer": li,
                    "entropy_norm": float(ent_norm.item()),
                    "max_load": float(usage.max().item()),
                    "min_load": float(usage.min().item()),
                    "usage": usage.detach().cpu(),
                }
            )

        return stats

    def print_moe_debug(self, topn: int = 3):
        stats = self._moe_debug_stats_per_layer()
        if not stats:
            print("[MoE] No stats yet.")
            return

        print("\n[MoE Debug]")
        for s in stats:
            usage = s["usage"]
            topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
            top_pairs = " ".join(
                [f"e{int(i)}={float(v):.2f}" for v, i in zip(topv, topi)]
            )

            imbalance = s["max_load"] / (s["min_load"] + 1e-9)

            print(
                f"Layer {s['layer']:02d} | "
                f"H={s['entropy_norm']:.3f} | "
                f"max={s['max_load']:.2f} min={s['min_load']:.2f} "
                f"(imb={imbalance:.1f}) | "
                f"top: {top_pairs}"
            )
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
