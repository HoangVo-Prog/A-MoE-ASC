from __future__ import annotations

import hashlib
import json
import os

from transformers import AutoTokenizer

from .cli import parse_args
from .config import TrainConfig, build_train_config
from .model import build_model

from shared import (
    FUSION_METHOD_CHOICES,
    set_all_seeds,
    set_determinism,
    _parse_str_list,
    _parse_int_list,
    eval_model,
    run_training_loop,
    maybe_freeze_encoder,
    run_benchmark_kfold_plus_full,
    train_full_multi_seed_then_test_generic
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_locked_baseline_metadata(
    *,
    cfg: TrainConfig,
    locked_baseline: bool,
    train_path: str,
    val_path: str,
    test_path: str,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    payload = {
        "locked_baseline": locked_baseline,
        "fusion_method": cfg.fusion_method,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__,
        "dataset_paths": {"train": train_path, "val": val_path, "test": test_path},
        "dataset_sha256": {
            "train": _sha256_file(train_path) if train_path and os.path.exists(train_path) else None,
            "val": _sha256_file(val_path) if val_path and os.path.exists(val_path) else None,
            "test": _sha256_file(test_path) if test_path and os.path.exists(test_path) else None,
        },
    }
    out_path = os.path.join(cfg.output_dir, f"{cfg.output_name}.baseline_lock.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_seeds_from_args(cfg: TrainConfig, args) -> list[int]:
    seeds = _parse_int_list(getattr(args, "seeds", ""))
    if seeds:
        return [int(s) for s in seeds]

    n = int(getattr(args, "num_seeds", 3))
    return [int(cfg.seed) + i for i in range(n)]


def main(args) -> None:
    cfg: TrainConfig = build_train_config(args)

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    set_all_seeds(int(cfg.seed))
    set_determinism(int(cfg.seed))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Optional: write metadata for locked baseline runs (if you use this flag)
    locked_baseline = bool(getattr(args, "locked_baseline", False))
    if locked_baseline:
        write_locked_baseline_metadata(
            cfg=cfg,
            locked_baseline=True,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
        )

    # Phase 1 benchmark mode: for each fusion method, run K-fold CV + FULL multi-seed test.
    if bool(getattr(args, "benchmark_fusions", False)):
        methods = _parse_str_list(getattr(args, "benchmark_methods", ""))
        if not methods:
            methods = FUSION_METHOD_CHOICES
            
        if cfg.head_type=="mof":
            methods = ["mof"]

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
            model_factory=lambda cfg_, num_labels, extra: build_model(cfg=cfg_, num_labels=num_labels),
            run_training_loop_fn=run_training_loop,
            eval_model_fn=eval_model,
            train_full_multi_seed_then_test_fn=train_full_multi_seed_then_test_generic,
            maybe_freeze_encoder_fn=maybe_freeze_encoder,
            trainloop_kwargs_factory=lambda cfg_, extra: {},
            extra={},  
        )

        print(f"Benchmark complete. Results written to {out_path}")
        return


if __name__ == "__main__":
    args = parse_args()
    main(args)
