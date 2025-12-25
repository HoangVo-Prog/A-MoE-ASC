from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def _split_param_groups(model, lr: float, encoder_lr_scale: float):
    """
    Split params into 2 groups:
      - encoder group: lr * encoder_lr_scale
      - non-encoder group (head/MoF/etc): lr
    Assumes encoder params are under attribute names like:
      model.encoder, model.bert, model.backbone, model.transformer
    We fall back to name-based matching.
    """
    enc_ids = set()

    # Try common attributes first
    for attr in ["encoder", "bert", "backbone", "transformer", "base_model"]:
        m = getattr(model, attr, None)
        if m is not None:
            for p in m.parameters():
                enc_ids.add(id(p))

    enc_params = []
    other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in enc_ids or n.startswith(("encoder.", "bert.", "backbone.", "transformer.", "base_model.")):
            enc_params.append(p)
        else:
            other_params.append(p)

    if len(enc_params) == 0:
        # Fallback: no encoder found, treat everything the same
        return [{"params": other_params, "lr": lr}]

    return [
        {"params": other_params, "lr": lr},
        {"params": enc_params, "lr": lr * encoder_lr_scale},
    ]


def build_optimizer_and_scheduler(
    *,
    model,
    lr: float,
    warmup_ratio: float,
    total_steps: int,
    params=None,
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
    encoder_lr_scale: float = 0.1,
):
    """
    Phase aware AdamW builder.

    params:
      If provided, optimizer is built only for these parameters.
      This is used for head-only training while encoder is frozen.

    encoder_lr_scale:
      Only used when params is None, to create 2 param groups.
      encoder lr = lr * encoder_lr_scale
      head/MoF lr = lr
    """
    if params is None:
        # Build param groups so encoder can have smaller lr after unfreeze
        params = _split_param_groups(model, lr=lr, encoder_lr_scale=encoder_lr_scale)

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
