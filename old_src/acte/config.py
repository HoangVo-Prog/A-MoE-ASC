# src/acte/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
from shared import BaseTrainConfig, _filter_config_kwargs

@dataclass
class ACTETrainConfig(BaseTrainConfig):
    acte_num_experts: int = 4
    acte_top_k: int = 2
    acte_top_m: int = 8
    acte_expert_hidden: int = 256
    acte_router_dropout: float = 0.0
    acte_expert_dropout: float = 0.1
    acte_score_temperature: float = 1.0
    acte_combine_with_base: str = "add"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def build_train_config(args) -> ACTETrainConfig:
    args_dict = _filter_config_kwargs(vars(args), ACTETrainConfig)
    return ACTETrainConfig(**args_dict)
