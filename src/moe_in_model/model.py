from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel
import math 

from moe import MoEConfig, MoEFFN, moe_load_balance_loss


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
            # transformers có thể truyền past_key_value hoặc past_key_values tùy version
            past = kwargs.get("past_key_value", None)
            if past is None:
                past = kwargs.get("past_key_values", None)

            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=past,  # attention module dùng key này
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

            # token_mask 2D cho route mask pad, tránh dùng extended mask 4D
            token_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    token_mask = (attention_mask[:, 0, 0, :] == 0).to(dtype=torch.long)
                elif attention_mask.dim() == 2:
                    token_mask = attention_mask.to(dtype=torch.long)

            layer_output = self.moe_ffn(attention_output, token_mask=token_mask)
            return (layer_output,) + outputs

        layer.forward = new_forward.__get__(layer, layer.__class__)


def freeze_all_but_moe_and_heads(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    # classifier heads should train
    for name, p in model.named_parameters():
        if name.startswith("classifier_"):
            p.requires_grad = True

    # moe params should train
    if hasattr(model, "encoder") and hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
        for layer in model.encoder.encoder.layer:
            moe = getattr(layer, "moe_ffn", None)
            if moe is not None:
                for p in moe.parameters():
                    p.requires_grad = True


def build_head(head_type: str, in_dim: int, num_labels: int, dropout: float) -> nn.Module:
    head_type = head_type.lower().strip()
    if head_type in {"linear", "lin"}:
        return LinearHead(in_dim, num_labels, dropout)
    if head_type in {"mlp", "2layer", "two_layer"}:
        return MLPHead(in_dim, num_labels, dropout)
    raise ValueError(f"Unsupported head_type: {head_type}. Use 'linear' or 'mlp'.")


class LinearHead(nn.Module):
    """
    Linear head with LayerNorm + Dropout for stability.
    """
    def __init__(self, in_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, dropout: float):
        super().__init__()
        hidden = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class BertConcatClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        moe_cfg: Optional["MoEConfig"] = None,
        freeze_base: bool = False,
        aux_loss_weight: float = 0.01,
        head_type: str = "linear",  # "linear" or "mlp"
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        _candidates = [8, 4, 2, 1]
        num_heads = next((x for x in _candidates if hidden_size % x == 0), 1)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.head_type = head_type
        self.head_concat = build_head(head_type, 2 * hidden_size, num_labels, dropout)
        self.head_single = build_head(head_type, hidden_size, num_labels, dropout)

        self.use_moe = use_moe
        self.aux_loss_weight = aux_loss_weight

        if self.use_moe:
            if moe_cfg is None:
                raise ValueError("use_moe=True but moe_cfg is None")
            replace_encoder_ffn_with_moe(self.encoder, moe_cfg)

        if freeze_base:
            freeze_all_but_moe_and_heads(self)

    def _collect_aux_loss(self) -> torch.Tensor:
        if not self.use_moe:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total = 0.0
        count = 0
        for layer in self.encoder.encoder.layer:
            moe = getattr(layer, "moe_ffn", None)
            if moe is None:
                continue
            if moe.last_router_logits is None or moe.last_topk_idx is None:
                continue
            total = total + moe_load_balance_loss(
                moe.last_router_logits, moe.last_topk_idx, moe.moe_cfg.num_experts
            )
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return total / count

    @torch.no_grad()
    def _moe_debug_stats_per_layer(self):
        if not self.use_moe:
            return []

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

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ) -> Dict[str, torch.Tensor]:

        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        cls_sent = out_sent.last_hidden_state[:, 0, :]

        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)
        cls_term = out_term.last_hidden_state[:, 0, :]

        if fusion_method == "concat":
            fused = torch.cat([cls_sent, cls_term], dim=-1)
            logits = self.head_concat(self.dropout(fused))
        elif fusion_method == "add":
            fused = cls_sent + cls_term
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "mul":
            fused = cls_sent * cls_term
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "cross":
            query = out_term.last_hidden_state[:, 0:1, :]
            key = out_sent.last_hidden_state
            value = out_sent.last_hidden_state

            key_padding_mask = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
            fused = attn_out.squeeze(1)
            logits = self.head_single(self.dropout(fused))
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        loss = None
        aux_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce = loss_fct(logits, labels)
            aux = self._collect_aux_loss()
            loss = ce + (self.aux_loss_weight * aux)
            aux_loss = aux

        return {"loss": loss, "logits": logits, "aux_loss": aux_loss}
