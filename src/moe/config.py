from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
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

    k_folds: int
    seed: int

    max_len_sent: int
    max_len_term: int

    output_dir: str
    output_name: str
    verbose_report: bool
    
    use_moe: bool = False
    freeze_base: bool = False 
    aux_loss_weight: float = 0.01


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 1
    router_bias: bool = True
    router_jitter: float = 0.0
    capacity_factor: Optional[float] = None
    route_mask_pad_tokens: bool = False 
