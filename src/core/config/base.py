from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class BaseTrainConfig:
    model_name: str
    fusion_method: str

    epochs: int
    train_batch_size: int
    eval_batch_size: int
    lr: float
    warmup_ratio: float
    dropout: float

    freeze_epochs: int
    rolling_k: int
    early_stop_patience: int

    max_len_sent: int
    max_len_term: int

    output_dir: str
    output_name: str
    verbose_report: bool

    train_full_only: bool = False
    head_type: str = "linear"  # "linear" or "mlp"

    do_ensemble_logits: bool = True
    
    loss_type: str = "ce"  # "ce" | "weighted_ce" | "focal"
    class_weights: Optional[Sequence[float]] = None
    focal_gamma: float = 2.0