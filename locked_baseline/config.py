from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


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

    train_full_only: bool = False
    head_type: str = "linear"  # "linear" or "mlp"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def locked_baseline_config(
    *,
    fusion_method: str,
    output_dir: str = "outputs",
    output_name: str = "baseline_locked",
) -> TrainConfig:
    """Return a fixed experimental baseline config.

    Intended for fair, reproducible fusion comparisons where the only
    independent variable is `fusion_method`.
    """
    return TrainConfig(
        model_name="bert-base-uncased",
        fusion_method=fusion_method,

        epochs=20,
        train_batch_size=32,
        eval_batch_size=64,
        lr=2e-5,
        warmup_ratio=0.1,
        dropout=0.1,

        freeze_epochs=0,
        rolling_k=3,
        early_stop_patience=3,

        k_folds=0,
        seed=42,

        max_len_sent=24,
        max_len_term=4,

        output_dir=output_dir,
        output_name=output_name,
        verbose_report=False,

        train_full_only=False,
        head_type="linear",
    )
