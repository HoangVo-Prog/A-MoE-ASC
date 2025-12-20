from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
from shared import BaseTrainConfig


@dataclass
class TrainConfig(BaseTrainConfig):
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_train_config(args) -> TrainConfig:
    if getattr(args, "locked_baseline", False):
        return locked_baseline_config(
            fusion_method=args.fusion_method,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
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
        train_full_only=args.train_full_only,
        head_type=args.head_type,
    )


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
        model_name="roberta-base",
        fusion_method=fusion_method,

        epochs=10,
        train_batch_size=16,
        eval_batch_size=32,
        lr=2e-5,
        warmup_ratio=0.1,
        dropout=0.1,

        freeze_epochs=3,
        rolling_k=3,
        early_stop_patience=5,

        k_folds=5,
        seed=42,

        max_len_sent=24,
        max_len_term=4,

        output_dir=output_dir,
        output_name=output_name,
        verbose_report=False,

        train_full_only=False,
        head_type="linear",
    )
