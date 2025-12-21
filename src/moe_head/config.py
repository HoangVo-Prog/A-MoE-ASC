from moe_shared import MoEConfig
from shared import _filter_config_kwargs
import argparse
from typing import Optional
from dataclasses import dataclass


@dataclass
class MultiMoEConfig(MoEConfig):
    moe_topk_schedule: bool = False
    moe_topk_start: int = 4
    moe_topk_end: int = 2
    moe_topk_switch_epoch: int = 3
    
    
def build_multi_moe_config(args: argparse.Namespace) -> Optional[MultiMoEConfig]:
    if not bool(getattr(args, "use_moe", False)):
        return None
    args_dict = _filter_config_kwargs(vars(args), MultiMoEConfig)
    print("ARGS")
    print(args_dict)
    print()
    return MultiMoEConfig(**args_dict)