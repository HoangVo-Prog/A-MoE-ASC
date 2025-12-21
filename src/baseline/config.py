from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
from shared import BaseTrainConfig, _filter_config_kwargs


@dataclass
class TrainConfig(BaseTrainConfig):
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def build_train_config(args) -> TrainConfig:
    args_dict = _filter_config_kwargs(vars(args), TrainConfig)

    if getattr(args, "locked_baseline", False):
        return locked_baseline_config(
            fusion_method=args.fusion_method,
            output_dir=args.output_dir,
            output_name=args.output_name,
            loss_type=args.loss_type,
            class_weights=args.class_weights,
            focal_gamma=args.focal_gamma,
        )

    return TrainConfig(**args_dict)


def locked_baseline_config(
    *,
    fusion_method: str,
    output_dir: str = "outputs",
    output_name: str = "baseline_locked",
    loss_type: str = "ce",
    class_weights: str = "",
    focal_gamma: float = 2.0,
) -> TrainConfig:
    """
    Return a fixed experimental baseline config.

    Intended for fair, reproducible fusion comparisons where the only
    independent variable is fusion_method.
    """
    return TrainConfig(
        model_name="roberta-base",
        fusion_method=fusion_method,

        # training setup
        train_batch_size=16,
        eval_batch_size=32,
        lr=2e-5,
        warmup_ratio=0.1,
        dropout=0.1,

        # stabilization
        freeze_epochs=3,
        rolling_k=3,
        early_stop_patience=5,

        # reproducibility
        k_folds=5,
        seed=42,

        # input constraints
        max_len_sent=24,
        max_len_term=4,

        # loss configuration
        loss_type=loss_type,
        class_weights=class_weights,
        focal_gamma=focal_gamma,

        # output
        output_dir=output_dir,
        output_name=output_name,
        verbose_report=False,

        # experiment control
        train_full_only=False,
        head_type="linear",
    )
