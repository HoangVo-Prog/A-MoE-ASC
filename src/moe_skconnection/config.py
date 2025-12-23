from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SkConnectionConfig:
    sk_mix_mode: str = "logits"
    sk_beta_start: float = 0.0
    sk_beta_end: float = 1.0
    sk_beta_warmup_epochs: int = 0


def apply_skconnection_config(cfg: Any, args: Any) -> Any:
    """Attach skconnection fields onto an existing config object.

    This is designed to work with configs created by existing runners.
    """
    setattr(cfg, "sk_mix_mode", str(getattr(args, "sk_mix_mode", "logits")))
    setattr(cfg, "sk_beta_start", float(getattr(args, "sk_beta_start", 0.0)))
    setattr(cfg, "sk_beta_end", float(getattr(args, "sk_beta_end", 1.0)))
    setattr(cfg, "sk_beta_warmup_epochs", int(getattr(args, "sk_beta_warmup_epochs", 0)))
    return cfg
