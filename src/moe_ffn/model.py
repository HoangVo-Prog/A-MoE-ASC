from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from moe_ffn.moe import MoEConfig, MoEFFN, moe_load_balance_loss
from shared import DEVICE, BaseBertConcatClassifier


def _get_act_fn_from_intermediate(intermediate_module: nn.Module):
    if hasattr(intermediate_module, "intermediate_act_fn"):
        return intermediate_module.intermediate_act_fn
    if hasattr(intermediate_module, "activation"):
        return intermediate_module.activation
    raise ValueError("Cannot find activation function on intermediate module")


def replace_encoder_ffn_with_moe(encoder: nn.Module, moe_cfg: MoEConfig) -> None:
    if not hasattr(encoder, "encoder") or not hasattr(encoder.encoder, "layer"):
        raise ValueError("Encoder does not look like BERT/RoBERTa model with encoder.layer")

    for layer in encoder.encoder.layer:
        intermediate = layer.intermediate
        output = layer.output

        base_dense1 = intermediate.dense
        base_dense2 = output.dense
        base_ln = output.LayerNorm
        dropout_p = output.dropout.p if hasattr(output.dropout, "p") else encoder.config.hidden_dropout_prob
        act_fn = _get_act_fn_from_intermediate(intermediate)

        moe_ffn = MoEFFN(
            hidden_size=encoder.config.hidden_size,
            intermediate_size=encoder.config.intermediate_size,
            dropout_p=dropout_p,
            layer_norm_eps=encoder.config.layer_norm_eps,
            act_fn=act_fn,
            base_dense1=base_dense1,
            base_dense2=base_dense2,
            base_layernorm=base_ln,
            moe_cfg=moe_cfg,
        )

        layer.intermediate = nn.Identity()
        layer.output = nn.Identity()
        layer.moe_ffn = moe_ffn

        def new_forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            **kwargs,
        ):
            # transformers can pass past_key_value or past_key_values depending on version
            past = kwargs.get("past_key_value", None)
            if past is None:
                past = kwargs.get("past_key_values", None)

            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=past,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]

            if encoder_hidden_states is not None:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:]

            # token_mask is 2D for route_mask_pad_tokens, do not use extended 4D mask
            token_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    token_mask = (attention_mask[:, 0, 0, :] == 0).to(dtype=torch.long)
                elif attention_mask.dim() == 2:
                    token_mask = attention_mask.to(dtype=torch.long)

            layer_output = self.moe_ffn(attention_output, token_mask=token_mask)
            return (layer_output,) + outputs

        layer.forward = new_forward.__get__(layer, layer.__class__)


class BertConcatClassifier(BaseBertConcatClassifier):
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        moe_cfg: MoEConfig,
        aux_loss_weight: float,
        freeze_moe: bool,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
        )

        self.aux_loss_weight = aux_loss_weight
        replace_encoder_ffn_with_moe(self.encoder, moe_cfg)

    def _collect_aux_loss(self):
        total, count = 0.0, 0
        for layer in self.encoder.encoder.layer:
            moe = getattr(layer, "moe_ffn", None)
            if moe is None or moe.last_router_logits is None:
                continue
            total += moe_load_balance_loss(
                moe.last_router_logits, moe.last_topk_idx, moe.moe_cfg.num_experts
            )
            count += 1
        return total / count if count > 0 else torch.tensor(0.0, device=self.device)

    def _compute_loss(self, logits, labels):
        if labels is None:
            return {"loss": None, "logits": logits, "aux_loss": None}

        ce = nn.CrossEntropyLoss()(logits, labels)
        aux = self._collect_aux_loss()
        return {
            "loss": ce + self.aux_loss_weight * aux,
            "logits": logits,
            "aux_loss": aux,
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
            E = moe.moe_cfg.num_experts

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
            print("[MoE] No stats yet (maybe first batch not run or last_router_logits missing).")
            return
        print()
        for s in stats:
            usage = s["usage"]
            topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
            top_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])
            print(
                f"[MoE][layer {s['layer']}] entropy_norm={s['entropy_norm']:.3f} "
                f"max={s['max_load']:.3f} min={s['min_load']:.3f} | top: {top_pairs}"
            )


def build_model(*, cfg, moe_cfg, num_labels: int):
    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        head_type=cfg.head_type,
        moe_cfg=moe_cfg,
        aux_loss_weight=float(cfg.aux_loss_weight),
        freeze_moe=bool(getattr(cfg, "freeze_moe", False)),
    ).to(DEVICE)