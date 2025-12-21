from moe_shared import MoEConfig
from shared import _filter_config_kwargs
import argparse
from typing import Optional


class MultiMoEConfig(MoEConfig):
    moe_topk_schudule: bool = False
    moe_topk_start: int 
    moe_topk_end: int 
    moe_topk_switch_epoch: int
    
    
def build_multi_moe_config(args: argparse.Namespace) -> Optional[MoEConfig]:
    if not bool(getattr(args, "use_moe", False)):
        return None
    args_dict = _filter_config_kwargs(vars(args), MultiMoEConfig)
    return MultiMoEConfig(**args_dict)