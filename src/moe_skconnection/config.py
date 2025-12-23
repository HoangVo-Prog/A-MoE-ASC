import argparse
from typing import Optional

from moe_head.config import MultiMoEConfig  # reuse same dataclass
from moe_head.config import build_multi_moe_config as _build_moe_cfg_base

from moe_shared import TrainConfig, build_train_config as _build_train_cfg_base


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    cfg: TrainConfig = _build_train_cfg_base(args)

    # attach sk fields onto cfg
    setattr(cfg, "sk_mix_mode", str(getattr(args, "sk_mix_mode", "logits")))
    setattr(cfg, "sk_beta_start", float(getattr(args, "sk_beta_start", 0.0)))
    setattr(cfg, "sk_beta_end", float(getattr(args, "sk_beta_end", 1.0)))
    setattr(cfg, "sk_beta_warmup_epochs", int(getattr(args, "sk_beta_warmup_epochs", 3)))

    return cfg


def build_multi_moe_config(args: argparse.Namespace) -> Optional[MultiMoEConfig]:
    return _build_moe_cfg_base(args)
