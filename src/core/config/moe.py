from __future__ import annotations
from dataclasses import dataclass
from src.core.config.base import BaseTrainConfig


@dataclass
class MoeBaseTrainConfig(BaseTrainConfig):
    use_moe: bool = False
    freeze_moe: bool = False
    aux_loss_weight: float = 0.01

    step_print_moe: float = 100

    train_full_only: bool = False
    head_type: str = "linear" 

    do_ensemble_logits: bool = True
    
    use_amp: bool = True
    amp_dtype: str = "fp16"
    adamw_foreach: bool = False
    adamw_fused: bool = False
    


@dataclass
class MoEConfig:
    mode: str = "moe_head"  # "moe_ffn", "moe_head", "multi_moe_head", "moe_skconnection", "mof"    
    num_experts: int = 8
    moe_top_k: int = 2
    router_bias: bool = True
    router_jitter: float = 0.001 
    jitter_warmup_steps: int = 0
    router_entropy_weight: float = 0.01
    capacity_factor = None 
    route_mask_pad_tokens: bool = False 
    
    router_temperature: float = 1.0
    
    # Multi Moe Head 
    moe_topk_schedule: bool = False
    moe_topk_start: int = 4
    moe_topk_end: int = 2
    moe_topk_switch_epoch: int = 3
    
    # MoE Skip Connection
    sk_mix_mode: str = "rep"  # "logits" or "rep"
    sk_beta_start: float = 0.0
    sk_beta_end: float = 1.0
    sk_beta_warmup_epochs: int = 0  # 0 disables warmup

    # MoF
    mof_experts: str = "sent,term,concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction"
    mof_mix_level: str = "rep"  # "repr" or "logit" #TODO: fix logic to rep 
    mof_lb_coef: float = 0.001
    mof_lb_coef: str = "switch"  # "l2" or "switch"
    mof_entropy_coef: float = 0.001
    mof_mixed_repr_norm: str = "layernorm"  # "layernorm" or "none" or "clamp"
    
    mof_mixed_repr_norm_clamp: float = 0.0
    mof_residual_alpha_init: float = 0.1
    mof_residual_alpha_learnable: int = 1  # 1 = learnable, 0 = fixed
    mof_router_temperature: float = 1.0

    mof_disable_expert_scaling: bool = False
    mof_expert_norm_clamp: float = 0.0
    mof_logit_clamp: float = 0.0

    mof_debug: bool = False
    mof_debug_every: int = 100
    mof_debug_max_batch: int = 1
    mof_debug_max_experts: int = 0

    encoder_lr_scale: float = 0.1