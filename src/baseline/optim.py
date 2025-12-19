from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(
    *,
    model,
    lr: float,
    warmup_ratio: float,
    total_steps: int,
    params=None,
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
):
    """
    Build AdamW + linear warmup scheduler.

    - params: iterable of parameters to optimize (phase-aware)
    - adamw_foreach: disable by default to avoid VRAM spike
    - adamw_fused: optional fused AdamW if supported
    """
    if params is None:
        params = model.parameters()

    optimizer = AdamW(
        params,
        lr=lr,
        foreach=adamw_foreach,
        fused=adamw_fused,
    )

    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler
