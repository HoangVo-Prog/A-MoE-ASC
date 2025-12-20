from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
from utils import BaseTrainConfig


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
    top_k: int = 1
    router_bias: bool = True
    router_jitter: float = 0.0
    capacity_factor: Optional[float] = None
    route_mask_pad_tokens: bool = False


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model_name=args.model_name,
        fusion_method=args.fusion_method,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        freeze_epochs=args.freeze_epochs,
        rolling_k=args.rolling_k,
        early_stop_patience=args.early_stop_patience,
        k_folds=args.k_folds,
        seed=args.seed,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        output_dir=args.output_dir,
        output_name=args.output_name,
        verbose_report=args.verbose_report,
        use_moe=bool(getattr(args, "use_moe", False)),
        freeze_moe=bool(getattr(args, "freeze_moe", False)),
        aux_loss_weight=float(getattr(args, "aux_loss_weight", 0.01)),
        step_print_moe=float(getattr(args, "step_print_moe", 100)),
        train_full_only=bool(getattr(args, "train_full_only", False)),
        head_type=str(getattr(args, "head_type", "linear")),
        do_ensemble_logits=bool(getattr(args, "do_ensemble_logits", True)),
        use_amp=(not getattr(args, "no_amp", False)),
        amp_dtype=str(getattr(args, "amp_dtype", "fp16")),
        adamw_foreach=bool(getattr(args, "adamw_foreach", False)),
        adamw_fused=bool(getattr(args, "adamw_fused", False)),
    )


def build_moe_config(args: argparse.Namespace) -> Optional[MoEConfig]:
    if not bool(getattr(args, "use_moe", False)):
        return None
    return MoEConfig(
        num_experts=int(getattr(args, "moe_num_experts", 8)),
        top_k=int(getattr(args, "moe_top_k", 1)),
        route_mask_pad_tokens=bool(getattr(args, "route_mask_pad_tokens", False)),
    )


def locked_baseline_config(
    args: argparse.Namespace,
    fusion_method: str,
    seed: int,
    use_moe: Optional[bool] = None,
) -> Tuple[TrainConfig, Optional[MoEConfig]]:
    """Build a benchmark configuration that is locked across runs for fair comparison.

    Only a small set of fields is overridden per run (fusion_method, seed, optionally use_moe).
    All other hyperparameters are taken from args at the time the benchmark is launched.
    """
    cfg = build_train_config(args)
    cfg.fusion_method = fusion_method
    cfg.seed = seed
    if use_moe is not None:
        cfg.use_moe = bool(use_moe)

    moe_cfg = build_moe_config(args)
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
