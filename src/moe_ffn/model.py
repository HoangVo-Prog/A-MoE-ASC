from __future__ import annotations

import math
import torch
import torch.nn as nn

from moe_ffn.moe import MoEFFN

from shared import DEVICE

from moe_shared import MoEBertConcatClassifier, MoEConfig


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
            past = kwargs.get("past_key_values", None)
            if past is None:
                past = kwargs.get("past_key_value", None)

            try:
                self_attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    past_key_values=past,
                )
            except TypeError:
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


class BertConcatClassifier(MoEBertConcatClassifier):
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        moe_cfg: MoEConfig,
        aux_loss_weight: float,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            moe_cfg=moe_cfg,
            aux_loss_weight=aux_loss_weight,
        )

        # Attach MoE FFN into the encoder (this class is the "MoE-enabled" variant).
        replace_encoder_ffn_with_moe(self.encoder, moe_cfg)


def build_model(*, cfg, moe_cfg, num_labels: int):
    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        head_type=cfg.head_type,
        moe_cfg=moe_cfg,
        aux_loss_weight=float(cfg.aux_loss_weight),
    ).to(DEVICE)
