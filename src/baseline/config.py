from dataclasses import dataclass

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
    
    head_type: str = "linear" # "linear" or "mlp"
