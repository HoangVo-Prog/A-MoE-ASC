from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
from shared import BaseTrainConfig, _filter_config_kwargs


@dataclass
class TrainConfig(BaseTrainConfig):
    use_moe: bool = False
    freeze_moe: bool = False
    aux_loss_weight: float = 0.01

    step_print_moe: float = 100

    train_full_only: bool = False
    head_type: str = "linear"  # "linear" or "mlp"

    do_ensemble_logits: bool = True
    
    use_amp: bool = True
    amp_dtype: str = "fp16"
    adamw_foreach: bool = False
    adamw_fused: bool = False


@dataclass
class MoEConfig:
    num_experts: int = 8
    moe_top_k: int = 1
    router_bias: bool = True
    router_jitter: float = 0.05
    capacity_factor: Optional[float] = None
    route_mask_pad_tokens: bool = False


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    args_dict = _filter_config_kwargs(vars(args), TrainConfig)
    return TrainConfig(**args_dict)


def build_moe_config(args: argparse.Namespace) -> Optional[MoEConfig]:
    if not bool(getattr(args, "use_moe", False)):
        return None
    args_dict = _filter_config_kwargs(vars(args), MoEConfig)
    return MoEConfig(**args_dict)


def locked_baseline_config(
    args: argparse.Namespace,
    fusion_method: str,
    seed: int,
    use_moe: Optional[bool] = None,
    build_train_config_fn = build_train_config,
    build_moe_config_fn = build_moe_config,
) -> Tuple[TrainConfig, Optional[MoEConfig]]:
    """Build a benchmark configuration that is locked across runs for fair comparison.

    Only a small set of fields is overridden per run (fusion_method, seed, optionally use_moe).
    All other hyperparameters are taken from args at the time the benchmark is launched.
    """
    cfg = build_train_config_fn(args)
    cfg.fusion_method = fusion_method
    cfg.seed = seed
    if use_moe is not None:
        cfg.use_moe = bool(use_moe)

    moe_cfg = build_moe_config_fn(args)
    if use_moe is not None and not bool(use_moe):
        moe_cfg = None
        
    cfg.model_name="roberta-base"
    cfg.train_batch_size=16
    cfg.eval_batch_size=32
    cfg.lr=2e-5
    cfg.warmup_ratio=0.1
    cfg.dropout=0.1
    cfg.freeze_epochs=3
    cfg.rolling_k=3
    cfg.early_stop_patience=5

    cfg.k_folds=5

    cfg.max_len_sent=24
    cfg.max_len_term=4


    cfg.verbose_report=False

    cfg.train_full_only=False
    cfg.head_type="linear"
    
    return cfg, moe_cfg
