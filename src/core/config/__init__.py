from dataclasses import dataclass

from src.core.config.base import BaseTrainConfig as Base
from src.core.config.moe import MoeBaseTrainConfig as MoEBase, MoEConfig


@dataclass
class KFoldConfig:
    k: int = 5
    seed: int = 42
    shuffle: bool = True

@dataclass
class Config:
    kfold: KFoldConfig = KFoldConfig()
    base: MoEBase = MoEBase()
    moe: MoEConfig = MoEConfig()
    


