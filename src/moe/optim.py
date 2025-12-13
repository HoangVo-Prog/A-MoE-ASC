from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(*, model, lr: float, warmup_ratio: float, total_steps: int):
    optimizer = AdamW(model.parameters(), lr=lr)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler
