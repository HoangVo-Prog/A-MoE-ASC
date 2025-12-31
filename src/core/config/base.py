from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class BaseTrainConfig:
    model_name: str = "bert-base-uncased"
    fusion_method: str = "concat"
    benchmark_fusion: bool = False

    epochs: int = 10
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    dropout: float = 0.1

    freeze_epochs: int = 3
    rolling_k: int = 3
    early_stop_patience: int = 5

    max_len_sent: int = 24
    max_len_term: int = 4

    output_dir: str = "results"
    output_name: str =  ""
    verbose_report: bool = False

    train_full_only: bool = False
    head_type: str = "linear"  # "linear" or "mlp"

    do_ensemble_logits: bool = True
    
    loss_type: str = "ce"  # "ce" | "weighted_ce" | "focal"
    class_weights: Optional[Sequence[float]] = None
    focal_gamma: float = 2.0
    
    mode: str = "BaseModel"  # "BaseModel","MoEFFN", "MoEHead", "MultiMoEHead", "MoESkconnection", "MoF" 