import torch.nn as nn


def set_encoder_trainable(
    model: nn.Module,
    trainable: bool,
    *,
    keep_moe_trainable: bool = True,
) -> None:
    """
    Set requires_grad for encoder params.

    If keep_moe_trainable=True:
      - encoder base params follow `trainable`
      - MoE FFN params are ALWAYS trainable
    """
    for name, p in model.encoder.named_parameters():
        if keep_moe_trainable and "moe_ffn" in name:
            p.requires_grad = True
        else:
            p.requires_grad = trainable




def maybe_freeze_encoder(
    model: nn.Module,
    epoch_idx_0based: int,
    freeze_epochs: int,
    freeze_moe: bool = False
) -> None:
    """
    Freeze base encoder for first `freeze_epochs` epochs,
    but keep MoE FFN trainable.
    """
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        # Phase 1: freeze encoder base, train MoE + heads
        set_encoder_trainable(
            model,
            trainable=False,
            keep_moe_trainable=not freeze_moe,
        )
    else:
        # Phase 2: unfreeze everything
        set_encoder_trainable(
            model,
            trainable=True,
            keep_moe_trainable=False,
        )

