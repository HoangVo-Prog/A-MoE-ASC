from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, MISSING
from typing import Optional, Sequence, List, get_type_hints, get_origin, get_args, Union


@dataclass
class Config:
    # ====== Core / data ======
    model_name: str = "bert-base-uncased"
    train_path: str = "dataset/atsa/laptop14/train.json"
    test_path: str = "dataset/atsa/laptop14/test.json"
    max_len_sent: int = 64
    max_len_term: int = 4
    num_labels: int = 3
    num_workers: int = 4
    id2label: Optional[dict] = None
    label2id: Optional[dict] = None


    # ====== Training ======
    epochs: int = 15
    train_batch_size: int = 16
    eval_batch_size: int = 32
    test_batch_size: int = 32

    lr: float = 2e-5
    lr_head: float = 1e-4 
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    max_grad_norm: float = 1.0

    freeze_epochs: int = 1
    rolling_k: int = 3
    early_stop_patience: int = 5

    # ====== Run mode ======
    mode: str = "BaseModel"  # "BaseModel","MoEFFN","MoEHead","MultiMoe","MoESkconnection","MoFModel","SDModel","SDMoEDirModel","HAGMoE"
    fusion_method: str = "concat"
    benchmark_fusions: bool = False
    benchmark_methods: str = "sent,term,concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction"
    train_full_only: bool = False

    # ====== Seeds / kfold ======
    seed: int = 42
    k_folds: int = 5
    num_seeds: int = 3
    seeds: str = ""  # optional "42,43,44"
    shuffle: bool = True

    # ====== Output ======
    output_dir: str = "results"
    output_name: str = "results.json"
    verbose_report: bool = False

    # ====== Loss ======
    loss_type: str = "ce"  # "ce" | "weighted_ce" | "focal"
    class_weights: Optional[Sequence[float]] = None
    focal_gamma: float = 2.0

    # ====== AMP / AdamW ======
    use_amp: bool = True
    amp_dtype: str = "fp16"
    adamw_foreach: bool = False
    adamw_fused: bool = False

    # ====== Debug ======
    debug_aspect_span: bool = False

    # ====== MoE toggles ======
    freeze_moe: bool = False
    aux_loss_weight: float = 0.01
    aux_warmup_steps: int = 0
    step_print_moe: float = 100
    do_ensemble_logits: bool = True
    head_type: str = "linear"

    num_experts: int = 8
    moe_top_k: int = 2
    router_bias: bool = True
    router_jitter: float = 0.001
    jitter_warmup_steps: int = 0
    jitter_end: float = 0
    router_entropy_weight: float = 0.01
    router_entropy_target: float = 0.45

    route_mask_pad_tokens: bool = True
    router_temperature: float = 1.0
    capacity_factor: Optional[float] = None

    moe_topk_schedule: bool = False
    moe_topk_start: int = 4
    moe_topk_end: int = 2
    moe_topk_switch_epoch: int = 3

    # ====== Multi MoE ======
    multi_moe_top_k: int = 6
    num_moe_layers: int = 3
    top_k_decay: str = "custom"  # "linear", "exponential", "custom"
    top_k_schedule: Optional[list[int]] = field(default_factory=lambda: [6, 4, 2])

    # Skip connection
    expert_hidden: Optional[int] = None
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_steps: int = 0

    # ====== HAG-MoE ======
    hag_num_groups: int = 3
    hag_experts_per_group: int = 8
    hag_router_temperature: float = 1.0
    hag_group_temperature: float = 1.0
    hag_group_temperature_anneal: Optional[str] = 2.0,0.9
    hag_merge: str = "residual"  # "residual" | "moe_only"
    hag_fusion_method: str = "concat"
    hag_use_group_loss: bool = True
    hag_use_balance_loss: bool = True
    hag_use_diversity_loss: bool = True
    hag_lambda_group: float = 0.5
    hag_lambda_balance: float = 0.01
    hag_lambda_diversity: float = 0.2
    hag_verbose_loss: bool = False

    # Mixture of Fusion
    mof_experts: Optional[list[str]] = field(
        default_factory=lambda: 
            [
                "sent",
                "term",
                "concat",
                "add",
                "mul",
                "cross",
                "gated_concat",
                "bilinear",
                "coattn",
                "late_interaction",
            ]
    )

    # Semantic Deformation
    
    # ====== Semantic Deformation (SD) ======
    sd_rank: int = 8
    sd_alpha: float = 4.0         
    sd_lambda_bal: float = 0.05   
    sd_lambda_div: float = 0.0  
    router_hidden_mult: float = 1.0    

    # ----------------- factory -----------------
    @staticmethod
    def _is_bool_type(t) -> bool:
        if t is bool:
            return True
        origin = get_origin(t)
        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]
            return len(args) == 1 and args[0] is bool
        return False

    @staticmethod
    def _unwrap_optional(t):
        origin = get_origin(t)
        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]
            return args[0] if len(args) == 1 else t
        return t

    @classmethod
    def from_cli(cls, argv=None):
        parser = argparse.ArgumentParser("ATSC Trainer")
        type_hints = get_type_hints(cls)

        for f in fields(cls):
            name = f.name
            hinted_type = type_hints.get(name, f.type)
            hinted_type = cls._unwrap_optional(hinted_type)

            # Resolve default (support default_factory)
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                default = f.default_factory()        # type: ignore[misc]
            else:
                default = None

            # BOOL: add BOTH --x and --no_x
            if cls._is_bool_type(hinted_type):
                group = parser.add_mutually_exclusive_group(required=False)
                group.add_argument(f"--{name}", dest=name, action="store_true")
                group.add_argument(f"--no_{name}", dest=name, action="store_false")
                parser.set_defaults(**{name: bool(default)})
                continue

            # list[int] (hoặc Optional[list[int]]): parse as repeated ints
            origin = get_origin(hinted_type)
            if hinted_type == list[int] or origin is list:
                # Nếu bạn muốn cho phép rỗng: dùng nargs="*"
                # Nếu muốn bắt buộc có ít nhất 1 số khi truyền flag: dùng nargs="+"
                parser.add_argument(f"--{name}", nargs="*", type=int, default=default)
                continue

            # NON-BOOL
            if default is None:
                parser.add_argument(f"--{name}", default=None)
            else:
                parser.add_argument(f"--{name}", type=type(default), default=default)

        ns = parser.parse_args(argv)
        return cls(**vars(ns))

    # ----------------- helpers -----------------
    @property
    def is_benchmark(self) -> bool:
        return bool(self.benchmark_fusions)

    @property
    def seed_list(self) -> List[int]:
        if self.seeds.strip():
            return [int(x.strip()) for x in self.seeds.split(",") if x.strip()]
        return [self.seed + i for i in range(self.num_seeds)]

    def finalize(self) -> "Config":
        # class_weights: "1.0,2.5,1.0" -> [1.0,2.5,1.0]
        if isinstance(self.class_weights, str):
            s = self.class_weights.strip()
            self.class_weights = None if not s else [float(x.strip()) for x in s.split(",")]

        self.benchmark_methods = (self.benchmark_methods or "").strip()
        return self

    def validate(self) -> "Config":
        if self.loss_type == "focal" and (self.focal_gamma is None):
            raise ValueError("loss_type=focal requires focal_gamma")
        if self.k_folds < 0:
            raise ValueError("k_folds must be >= 0")
        return self
