from dataclasses import dataclass, field

from src.core.config.base import BaseTrainConfig as Base
from src.core.config.moe import MoeBaseTrainConfig as MoEBase, MoEConfig


@dataclass
class KFoldConfig:
    k: int = 5
    seed: int = 42
    shuffle: bool = True

@dataclass
class Config:
    kfold: "KFoldConfig" = field(default_factory=KFoldConfig)
    base: "MoEBase" = field(default_factory=MoEBase)
    moe: "MoEConfig" = field(default_factory=MoEConfig)


