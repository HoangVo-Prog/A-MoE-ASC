import math
from typing import Optional

import torch
import torch.nn as nn


class MoEHead(nn.Module):
    """Head level MoE block.

    Input: hidden_states [B, T, H]
    Output: hidden_states [B, T, H]

    Caches for aux loss and debug:
      - last_router_logits: [N_active, E]
      - last_topk_idx: [N_active, K]
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dropout_p: float,
        act_fn: nn.Module,
        router_bias: bool = True,
        router_jitter: float = 0.05,
        capacity_factor: Optional[float] = None,
        route_mask_pad_tokens: bool = False,
        layer_norm: Optional[nn.Module] = None,
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

        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None

        # init router near uniform
        nn.init.zeros_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(self, hidden_states: torch.Tensor, *, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    """Wrap a base encoder and apply MoEHead on last_hidden_state.

    The MoE module is exposed as attribute name `moe_ffn` so that utilities
    that look for "moe_ffn" in parameter names continue to work.
    """

    def __init__(self, *, base_encoder: nn.Module, moe_ffn: MoEHead) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.moe_ffn = moe_ffn

    def forward(self, *args: Any, **kwargs: Any):
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


class HeadBertConcatClassifier(MoEBertConcatClassifier):
    """moe_head model: encoder output, then MoE head, then classifier.

    Forward and main loss pipeline are inherited.
    This class overrides aux loss collection and debug to use head MoE.
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights: Optional[Any] = None,
        focal_gamma: float = 2.0,
        moe_cfg: MoEConfig,
        aux_loss_weight: float,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            loss_type=loss_type,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            moe_cfg=moe_cfg,
            aux_loss_weight=aux_loss_weight,
        )

        cfg = getattr(self.encoder, "config", None)
        hidden_size = int(getattr(cfg, "hidden_size"))
        intermediate_size = int(getattr(cfg, "intermediate_size", hidden_size * 4))
        hidden_act = str(getattr(cfg, "hidden_act", "gelu")).lower()
        act_fn: nn.Module = nn.GELU() if hidden_act == "gelu" else nn.ReLU()
        dropout_p = float(getattr(cfg, "hidden_dropout_prob", dropout))
                
        self._topk_schedule_enabled = moe_cfg.moe_topk_schedule
        self._topk_start = moe_cfg.moe_topk_start
        self._topk_end = moe_cfg.moe_topk_end
        self._topk_switch_epoch = moe_cfg.moe_topk_switch_epoch

        moe_head = MoEHead(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=moe_cfg.num_experts,
            top_k=moe_cfg.moe_top_k,
            dropout_p=dropout_p,
            act_fn=act_fn,
            router_bias=bool(getattr(moe_cfg, "router_bias", True)),
            router_jitter=float(getattr(moe_cfg, "router_jitter", 0.05)),
            capacity_factor=getattr(moe_cfg, "capacity_factor", None),
            route_mask_pad_tokens=bool(getattr(moe_cfg, "route_mask_pad_tokens", False)),
            layer_norm=nn.LayerNorm(hidden_size),
        )

        base_encoder = self.encoder
        self.encoder = EncoderWithMoEHead(base_encoder=base_encoder, moe_ffn=moe_head)


    def _collect_aux_loss(self):
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None or moe.last_router_logits is None or moe.last_topk_idx is None:
            return torch.tensor(0.0, device=self.device)
        return moe_load_balance_loss(
            moe.last_router_logits,
            moe.last_topk_idx,
            moe.num_experts,
        )

    @torch.no_grad()
    def _moe_debug_stats_head(self):
        moe = getattr(self.encoder, "moe_ffn", None)
        if moe is None or moe.last_router_logits is None or moe.last_topk_idx is None:
            return None

        logits = moe.last_router_logits  # [N_active, E]
        topk_idx = moe.last_topk_idx     # [N_active, K]
        E = int(moe.num_experts)

        # A) Softmax usage over all experts (your current metric)
        probs = torch.softmax(logits, dim=-1)
        usage_soft = probs.mean(dim=0)  # [E]

        eps = 1e-9
        ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
        ent_norm = float(ent / math.log(E))

        # B) Router "is it learning" signals
        logits_f = logits.float()
        logits_std = float(logits_f.std().item())
        logits_absmean = float(logits_f.abs().mean().item())

        w_norm = float(moe.router.weight.detach().float().norm().item())
        b_norm = 0.0
        if moe.router.bias is not None:
            b_norm = float(moe.router.bias.detach().float().norm().item())

        # C) Actual selected expert histogram from topk indices
        flat = topk_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=E).float()  # [E]
        frac = counts / (counts.sum() + eps)                # [E]

        topk_ent = -(frac * (frac + eps).log()).sum()
        topk_ent_norm = float(topk_ent / math.log(E))

        return {
            # softmax based stats
            "entropy_norm": ent_norm,
            "max_load": float(usage_soft.max().item()),
            "min_load": float(usage_soft.min().item()),
            "usage": usage_soft.detach().cpu(),

            # learning signals
            "logits_std": logits_std,
            "logits_absmean": logits_absmean,
            "router_w_norm": w_norm,
            "router_b_norm": b_norm,

            # topk selection stats
            "topk_frac": frac.detach().cpu(),
            "topk_entropy_norm": topk_ent_norm,
            "topk_max": float(frac.max().item()),
            "topk_min": float(frac.min().item()),
        }


    def print_moe_debug(self, topn: int = 3):
        s = self._moe_debug_stats_head()
        if not s:
            print("[MoE] No stats yet (maybe first batch not run or missing caches).")
            return

        usage = s["usage"]
        topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
        top_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])

        topk_frac = s["topk_frac"]
        topkv, topki = torch.topk(topk_frac, k=min(topn, topk_frac.numel()))
        topk_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topkv, topki)])

        print()
        print(
            f"[MoE][head] softmax entropy_norm={s['entropy_norm']:.3f} "
            f"max={s['max_load']:.3f} min={s['min_load']:.3f} | top: {top_pairs}"
        )
        print(
            f"[MoE][head] logits_std={s['logits_std']:.4f} logits_absmean={s['logits_absmean']:.4f} "
            f"router_w_norm={s['router_w_norm']:.4f} router_b_norm={s['router_b_norm']:.4f}"
        )
        print(
            f"[MoE][head] topk entropy_norm={s['topk_entropy_norm']:.3f} "
            f"max={s['topk_max']:.3f} min={s['topk_min']:.3f} | topk: {topk_pairs}"
        )
        
        moe = getattr(self.encoder, "moe_ffn", None)
        cur_k = getattr(moe, "moe_top_k", None)
        print(f"[MoE][head] moe_top_k={cur_k} ...")


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


