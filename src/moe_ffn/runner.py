from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from moe_ffn.model import build_model
from moe_ffn.engine import maybe_freeze_encoder

from shared import (
    set_all_seeds,
    set_determinism,
    _parse_int_list,
    _parse_str_list,
    eval_model,
    run_benchmark_kfold_plus_full,
    train_full_multi_seed_then_test_generic
)

from moe_shared import (
    FUSION_METHOD_CHOICES,
    parse_args,
    TrainConfig,
    build_moe_config,
    build_train_config,
    locked_baseline_config,    
)

from moe_shared import run_training_loop as run_training_loop_fn


def _resolve_seeds_from_args(cfg: TrainConfig, args) -> list[int]:
    seeds = _parse_int_list(getattr(args, "seeds", ""))
    if seeds:
        return [int(s) for s in seeds]

    n = int(getattr(args, "num_seeds", 1))
    return [int(cfg.seed) + i for i in range(n)]


def _moe_trainloop_kwargs(cfg: TrainConfig, extra: dict) -> dict:
    return {
        "freeze_moe": bool(getattr(cfg, "freeze_moe", False)),
        "step_print_moe": float(getattr(cfg, "step_print_moe", 100)),
        "use_amp": bool(getattr(cfg, "use_amp", True)),
        "amp_dtype": str(getattr(cfg, "amp_dtype", "fp16")),
        "adamw_foreach": bool(getattr(cfg, "adamw_foreach", False)),
        "adamw_fused": bool(getattr(cfg, "adamw_fused", False)),
    }


def main(args: argparse.Namespace) -> None:
    cfg: TrainConfig = build_train_config(args)
    moe_cfg = build_moe_config(args)

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    set_all_seeds(int(cfg.seed))
    set_determinism(int(cfg.seed))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Optional: locked baseline mode, override cfg + moe_cfg (keep consistent with your current runner)
    if bool(getattr(args, "locked_baseline", False)):
        cfg_locked, moe_locked = locked_baseline_config(
            args,
            fusion_method=str(getattr(cfg, "fusion_method", "sent")),
            seed=int(getattr(cfg, "seed", 0)),
        )
        cfg = cfg_locked
        moe_cfg = moe_locked

    # Benchmark mode: K-fold CV + FULL multi-seed per fusion method
    if bool(getattr(args, "benchmark_fusions", False)):
        methods = _parse_str_list(getattr(args, "benchmark_methods", ""))
        if not methods:
            methods = list(FUSION_METHOD_CHOICES)

        seeds = _resolve_seeds_from_args(cfg, args)
        out_path = os.path.join(cfg.output_dir, "phase1_benchmark_all.json")

        run_benchmark_kfold_plus_full(
            base_cfg=cfg,
            train_path=train_path,
            test_path=test_path,
            tokenizer=tokenizer,
            methods=methods,
            seeds=seeds,
            output_path=out_path,
            model_factory=lambda cfg_, num_labels, extra: build_model(
                cfg=cfg_, moe_cfg=extra["moe_cfg"], num_labels=num_labels
            ),
            run_training_loop_fn=run_training_loop_fn,
            maybe_freeze_encoder_fn=maybe_freeze_encoder,
            eval_model_fn=eval_model,
            train_full_multi_seed_then_test_fn=train_full_multi_seed_then_test_generic,
            trainloop_kwargs_factory=_moe_trainloop_kwargs,
            extra={"moe_cfg": moe_cfg},
        )

        print(f"Benchmark complete. Results written to {out_path}")
        return

if __name__ == "__main__":
    args = parse_args()
    main(args)
