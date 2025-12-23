from __future__ import annotations

import hashlib
import json
import os
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from baseline.cli import parse_args
from baseline.config import TrainConfig, build_train_config
from baseline.model import build_model

from shared import (
    DEVICE,
    FUSION_METHOD_CHOICES,
    AspectSentimentDataset,
    AspectSentimentDatasetFromSamples,
    set_all_seeds,
    set_determinism,
    make_train_loader_with_seed,
    cleanup_cuda,
    _parse_str_list,
    _parse_int_list,
    eval_model,
    run_training_loop,
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
            trainloop_kwargs_factory=lambda cfg_, extra: {},
            extra={},  # baseline has no extra context
        )

        print(f"Benchmark complete. Results written to {out_path}")
        return

    # Build datasets and loaders
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

    # Case 1: single split train/val/test (no k-fold)
    if int(cfg.k_folds) <= 1:
        print("Running single split training")

        val_dataset = AspectSentimentDataset(
            json_path=val_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            label2id=label2id,
        )

        train_loader = make_train_loader_with_seed(train_dataset_full, cfg.train_batch_size, int(cfg.seed))
        val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

        model = build_model(cfg=cfg, num_labels=len(label2id))

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            epochs=cfg.epochs,
            fusion_method=cfg.fusion_method,
            freeze_epochs=cfg.freeze_epochs,
            rolling_k=cfg.rolling_k,
            early_stop_patience=cfg.early_stop_patience,
            id2label=id2label,
            tag="",
        )

        if out.get("best_state_dict") is not None:
            model.load_state_dict(out["best_state_dict"])
            model.to(DEVICE)

        final_test = eval_model(
            model=model,
            dataloader=test_loader,
            id2label=id2label,
            verbose_report=bool(cfg.verbose_report),
            print_confusion_matrix=True,
            fusion_method=cfg.fusion_method,
            f1_average="macro",
            return_confusion=False,
        )

        print(f"Final Test loss {final_test['loss']:.4f} F1 {final_test['f1']:.4f} acc {final_test['acc']:.4f}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, cfg.output_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        return

    # Case 2: k-fold CV, then optional full multi-seed
    if not bool(cfg.train_full_only):
        print(f"Running StratifiedKFold with k={cfg.k_folds}")

        samples = train_dataset_full.samples
        y = [label2id[s["sentiment"]] for s in samples]

        skf = StratifiedKFold(n_splits=int(cfg.k_folds), shuffle=True, random_state=int(cfg.seed))

        fold_val_f1: list[float] = []
        fold_test_f1: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, y), start=1):
            print(f"\n===== Fold {fold_idx}/{cfg.k_folds} =====")

            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            train_ds = AspectSentimentDatasetFromSamples(
                samples=train_samples,
                tokenizer=tokenizer,
                max_len_sent=cfg.max_len_sent,
                max_len_term=cfg.max_len_term,
                label2id=label2id,
            )
            val_ds = AspectSentimentDatasetFromSamples(
                samples=val_samples,
                tokenizer=tokenizer,
                max_len_sent=cfg.max_len_sent,
                max_len_term=cfg.max_len_term,
                label2id=label2id,
            )

            fold_seed = int(cfg.seed) + int(fold_idx)
            train_loader = make_train_loader_with_seed(train_ds, cfg.train_batch_size, fold_seed)
            val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False)

            model = build_model(cfg=cfg, num_labels=len(label2id))

            out = run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=cfg.lr,
                warmup_ratio=cfg.warmup_ratio,
                epochs=cfg.epochs,
                fusion_method=cfg.fusion_method,
                freeze_epochs=cfg.freeze_epochs,
                rolling_k=cfg.rolling_k,
                early_stop_patience=cfg.early_stop_patience,
                id2label=id2label,
                tag=f"[Fold {fold_idx}] ",
            )

            if out.get("best_state_dict") is not None:
                model.load_state_dict(out["best_state_dict"])
                model.to(DEVICE)

            best_val = eval_model(
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                verbose_report=False,
                print_confusion_matrix=False,
                fusion_method=cfg.fusion_method,
                f1_average="macro",
                return_confusion=False,
            )
            best_test = eval_model(
                model=model,
                dataloader=test_loader,
                id2label=id2label,
                verbose_report=False,
                print_confusion_matrix=False,
                fusion_method=cfg.fusion_method,
                f1_average="macro",
                return_confusion=False,
            )

            fold_val_f1.append(float(best_val["f1"]))
            fold_test_f1.append(float(best_test["f1"]))

            print(
                f"Fold {fold_idx} | Best rolling Val F1 {out['best_val_f1_rolling']:.4f} | "
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
    full_seeds = _resolve_seeds_from_args(cfg, args)

    train_full_multi_seed_then_test_generic(
        cfg=cfg,
        train_dataset_full=train_dataset_full,
        test_loader=test_loader,
        label2id=label2id,
        id2label=id2label,
        seeds=full_seeds,
        print_confusion_matrix=True,
        do_ensemble_logits=bool(getattr(cfg, "do_ensemble_logits", True)),
        verbose_ensemble_report=False,
        extra={},  # baseline
        model_factory=lambda cfg_, num_labels, extra: build_model(cfg=cfg_, num_labels=num_labels),
        run_training_loop_fn=run_training_loop,
        trainloop_kwargs_factory=lambda cfg_, extra: {},
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
