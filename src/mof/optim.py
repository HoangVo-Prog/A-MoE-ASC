from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(
    *,
    model,
    lr: float,
    warmup_ratio: float,
    total_steps: int,
    params=None,
    encoder_lr_scale: float = 1.0,
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
):
    """
    Phase aware AdamW builder.

    params:
      If provided, optimizer is built only for these parameters.
      This is used for level A, build head only while encoder is frozen.

    adamw_foreach:
      Default False for level B to reduce temporary buffer allocations at optimizer.step.

    adamw_fused:
      Optional. Some PyTorch builds support fused AdamW. Guarded with try except.
    """
    if params is None:
        # Full optimizer build (typically after encoder unfreezes).
        # If an encoder exists, allow scaling its LR via encoder_lr_scale.
        encoder = getattr(model, 'encoder', None)
        scale = float(encoder_lr_scale) if encoder_lr_scale is not None else 1.0
        if encoder is not None and scale != 1.0:
            enc_params = [p for p in encoder.parameters() if p.requires_grad]
            other_params = [p for n, p in model.named_parameters() if (not n.startswith('encoder.')) and p.requires_grad]
            params = [
                {'params': other_params, 'lr': lr},
                {'params': enc_params, 'lr': lr * scale},
            ]
        else:
            params = [p for p in model.parameters() if p.requires_grad]

    try:
        optimizer = AdamW(
            params,
            lr=lr,
            foreach=adamw_foreach,
            fused=adamw_fused,
        )
    except TypeError:
        optimizer = AdamW(
            params,
            lr=lr,
            foreach=adamw_foreach,
        )

    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler
