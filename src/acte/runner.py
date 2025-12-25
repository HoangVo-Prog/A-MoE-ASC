# src/acte/runner.py
from __future__ import annotations

import os
from transformers import AutoTokenizer

from acte.cli import parse_args
from acte.config import ACTETrainConfig, build_train_config
from acte.model import build_model

from shared import (
    FUSION_METHOD_CHOICES,
    set_all_seeds,
    set_determinism,
    _parse_str_list,
    _parse_int_list,
    maybe_freeze_encoder,
    eval_model,
    run_training_loop,
    run_benchmark_kfold_plus_full,
    train_full_multi_seed_then_test_generic,
)

def _resolve_seeds_from_args(cfg: ACTETrainConfig, args) -> list[int]:
    seeds = _parse_int_list(getattr(args, "seeds", ""))
    if seeds:
        return [int(s) for s in seeds]
    n = int(getattr(args, "num_seeds", 3))
    return [int(cfg.seed) + i for i in range(n)]

def main(args) -> None:
    cfg: ACTETrainConfig = build_train_config(args)

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    set_all_seeds(int(cfg.seed))
    set_determinism(int(cfg.seed))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if bool(getattr(args, "benchmark_fusions", False)):
        methods = _parse_str_list(getattr(args, "benchmark_methods", ""))
        if not methods:
            methods = FUSION_METHOD_CHOICES
        seeds = _resolve_seeds_from_args(cfg, args)

        out_path = os.path.join(cfg.output_dir, "phase1_benchmark_all.json")
        os.makedirs(cfg.output_dir, exist_ok=True)

        run_benchmark_kfold_plus_full(
            base_cfg=cfg,
            train_path=train_path,
            test_path=test_path,
            tokenizer=tokenizer,
            methods=methods,
            seeds=seeds,
            output_path=out_path,
            model_factory=lambda cfg_, num_labels, extra: build_model(cfg=cfg_, num_labels=num_labels),
            run_training_loop_fn=run_training_loop,
            maybe_freeze_encoder_fn=maybe_freeze_encoder,
            eval_model_fn=eval_model,
            train_full_multi_seed_then_test_fn=train_full_multi_seed_then_test_generic,
            trainloop_kwargs_factory=lambda cfg_, extra: {},
            extra={},
        )
        print(f"Benchmark complete. Results written to {out_path}")
        return


if __name__ == "__main__":
    args = parse_args()
    main(args)
