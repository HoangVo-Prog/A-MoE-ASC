from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
from shared import BaseTrainConfig, _filter_config_kwargs


@dataclass
class TrainConfig(BaseTrainConfig):
    mof_include_sent_term: bool = False
    mof_debug: bool = False
    mof_debug_every: int = 200
    mof_debug_max_batch: int = 1
    mof_debug_max_experts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def build_train_config(args) -> TrainConfig:
    args_dict = _filter_config_kwargs(vars(args), TrainConfig)

    if getattr(args, "locked_baseline", False):
        return locked_baseline_config(args_dict)

    return TrainConfig(**args_dict)


def locked_baseline_config(cfg_dict) -> TrainConfig:
    """
    Return a fixed experimental baseline config.

    Intended for fair, reproducible fusion comparisons where the only
    independent variable is fusion_method.
    """
    config = TrainConfig(**cfg_dict)
     
    config.train_batch_size=16,
    config.eval_batch_size=32,
    config.lr=2e-5,
    config.warmup_ratio=0.1,
    config.dropout=0.1,
    config.rolling_k=3,
    config.early_stop_patience=5,

    config.k_folds=5,
        
    config.max_len_sent=24,
    config.max_len_term=4,

    return config
