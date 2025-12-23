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

from shared import (
    DEVICE,
    AspectSentimentDataset,
    AspectSentimentDatasetFromSamples,
    set_all_seeds,
    set_determinism,
    make_train_loader_with_seed,
    cleanup_cuda,
    _parse_int_list,
    _parse_str_list,
    eval_model,
    run_benchmark_kfold_plus_full,
    train_full_multi_seed_then_test_generic,
)

from moe_shared import (
    FUSION_METHOD_CHOICES,
    TrainConfig,
    build_train_config,
    locked_baseline_config,
)

# Key change: use moe_skconnection build_moe_config + parse_args + build_model
from moe_skconnection.config import build_multi_moe_config as build_moe_config
from moe_skconnection.cli import parse_args
from moe_skconnection.model import build_model

# Keep the same training engine behavior as moe_head engine
from moe_head.engine import run_training_loop as run_training_loop_fn


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

    # Optional: locked baseline mode, preserve your current runner logic
    if bool(getattr(args, "locked_baseline", False)):
        cfg_locked, moe_locked = locked_baseline_config(
            args,
            fusion_method=str(getattr(cfg, "fusion_method", "sent")),
            seed=int(getattr(cfg, "seed", 0)),
            build_moe_config_fn=build_moe_config,
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
            eval_model_fn=eval_model,
            train_full_multi_seed_then_test_fn=train_full_multi_seed_then_test_generic,
            trainloop_kwargs_factory=_moe_trainloop_kwargs,
            extra={"moe_cfg": moe_cfg},
        )

        print(f"Benchmark complete. Results written to {out_path}")
        return

    # ===== Normal training mode =====
    train_dataset_full = AspectSentimentDataset(
        json_path=train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=None,
    )
    label2id: Dict[str, int] = train_dataset_full.label2id
    id2label: Dict[int, str] = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)

    test_dataset = AspectSentimentDataset(
        json_path=test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    # Case 1: single split train/val/test
    if int(cfg.k_folds) <= 1:
        print("Running single split training")

        val_dataset = AspectSentimentDataset(
            json_path=val_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            label2id=label2id,
        )

        train_loader = make_train_loader_with_seed(
            train_dataset_full, cfg.train_batch_size, int(cfg.seed)
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

        model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))

        out = run_training_loop_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            epochs=cfg.epochs,
            fusion_method=cfg.fusion_method,
            freeze_epochs=cfg.freeze_epochs,
            freeze_moe=bool(getattr(cfg, "freeze_moe", False)),
            rolling_k=cfg.rolling_k,
            early_stop_patience=cfg.early_stop_patience,
            id2label=id2label,
            tag="",
            **_moe_trainloop_kwargs(cfg, {"moe_cfg": moe_cfg}),
        )

        best_state = out.get("best_state_dict", None)
        if best_state is not None:
            model.load_state_dict(best_state)

        best_test = eval_model(
            model=model,
            dataloader=test_loader,
            id2label=id2label,
            print_confusion_matrix=True,
            verbose_report=bool(getattr(cfg, "verbose_report", False)),
            fusion_method=cfg.fusion_method,
            f1_average="macro",
        )
        print(f"Test: loss {best_test['loss']:.4f} f1 {best_test['f1']:.4f} acc {best_test['acc']:.4f}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, cfg.output_name)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

        del model
        cleanup_cuda()

    else:
        print(f"Running {cfg.k_folds}-fold CV training")
        skf = StratifiedKFold(n_splits=int(cfg.k_folds), shuffle=True, random_state=int(cfg.seed))

        y = [s["label"] for s in train_dataset_full.samples]
        fold_val_f1 = []
        fold_test_f1 = []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            print(f"\n===== Fold {fold_idx}/{cfg.k_folds} =====")

            train_samples = [train_dataset_full.samples[i] for i in tr_idx]
            val_samples = [train_dataset_full.samples[i] for i in va_idx]

            train_dataset = AspectSentimentDatasetFromSamples(
                samples=train_samples,
                tokenizer=tokenizer,
                max_len_sent=cfg.max_len_sent,
                max_len_term=cfg.max_len_term,
                label2id=label2id,
            )
            val_dataset = AspectSentimentDatasetFromSamples(
                samples=val_samples,
                tokenizer=tokenizer,
                max_len_sent=cfg.max_len_sent,
                max_len_term=cfg.max_len_term,
                label2id=label2id,
            )

            train_loader = make_train_loader_with_seed(train_dataset, cfg.train_batch_size, int(cfg.seed))
            val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

            model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))

            out = run_training_loop_fn(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=cfg.lr,
                warmup_ratio=cfg.warmup_ratio,
                epochs=cfg.epochs,
                fusion_method=cfg.fusion_method,
                freeze_epochs=cfg.freeze_epochs,
                freeze_moe=bool(getattr(cfg, "freeze_moe", False)),
                rolling_k=cfg.rolling_k,
                early_stop_patience=cfg.early_stop_patience,
                id2label=id2label,
                tag=f"[fold{fold_idx}] ",
                **_moe_trainloop_kwargs(cfg, {"moe_cfg": moe_cfg}),
            )

            best_state = out.get("best_state_dict", None)
            if best_state is not None:
                model.load_state_dict(best_state)

            best_val = eval_model(
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                print_confusion_matrix=False,
                verbose_report=False,
                fusion_method=cfg.fusion_method,
                f1_average="macro",
            )
            best_test = eval_model(
                model=model,
                dataloader=test_loader,
                id2label=id2label,
                print_confusion_matrix=True,
                verbose_report=False,
                fusion_method=cfg.fusion_method,
                f1_average="macro",
            )

            fold_val_f1.append(float(best_val["f1"]))
            fold_test_f1.append(float(best_test["f1"]))

            print(
                f"Fold {fold_idx} | Best rolling Val F1 {out.get('best_val_f1_rolling', 0.0):.4f} | "
                f"Val F1 {best_val['f1']:.4f} | Test F1 {best_test['f1']:.4f}"
            )

            os.makedirs(cfg.output_dir, exist_ok=True)
            save_path = os.path.join(cfg.output_dir, f"fold{fold_idx}_{cfg.output_name}")
            torch.save(model.state_dict(), save_path)
            print(f"Saved fold model to {save_path}")

            del model
            cleanup_cuda()

        print("\n===== CV Summary =====")
        print(f"Val macro-F1 mean {np.mean(fold_val_f1):.4f} std {np.std(fold_val_f1, ddof=1):.4f}")
        print(f"Test macro-F1 mean {np.mean(fold_test_f1):.4f} std {np.std(fold_test_f1, ddof=1):.4f}")

    # Full multi-seed training then test (generic)
    if bool(getattr(cfg, "train_full_only", False)):
        print("Skipping full training because train_full_only is enabled")
        return

    full_seeds = _resolve_seeds_from_args(cfg, args)

    out = train_full_multi_seed_then_test_generic(
        cfg=cfg,
        train_dataset_full=train_dataset_full,
        test_loader=test_loader,
        label2id=label2id,
        id2label=id2label,
        seeds=full_seeds,
        print_confusion_matrix=True,
        do_ensemble_logits=bool(getattr(cfg, "do_ensemble_logits", True)),
        verbose_ensemble_report=False,
        extra={"moe_cfg": moe_cfg},
        model_factory=lambda cfg_, num_labels, extra: build_model(
            cfg=cfg_, moe_cfg=extra["moe_cfg"], num_labels=num_labels
        ),
        run_training_loop_fn=run_training_loop_fn,
        trainloop_kwargs_factory=_moe_trainloop_kwargs,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_json = os.path.join(cfg.output_dir, "full_multi_seed_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Full multi-seed summary written to {out_json}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
