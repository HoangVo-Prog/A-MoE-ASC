from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cli import FUSION_METHOD_CHOICES, parse_args
from config import TrainConfig, build_moe_config, build_train_config, locked_baseline_config
from constants import DEVICE
from datasets import AspectSentimentDataset, AspectSentimentDatasetFromSamples
from engine import _print_confusion_matrix, eval_model, run_training_loop, logits_to_metrics, collect_test_logits
from model import BertConcatClassifier
from optim import build_optimizer_and_scheduler


def clear_model(model, optimizer, scheduler):
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_determinism(seed: int) -> None:
    # Deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_all_seeds(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _parse_int_list(csv: str) -> list[int]:
    s = (csv or "").strip()
    if not s:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _parse_str_list(csv: str) -> list[str]:
    s = (csv or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def make_train_loader_with_seed(dataset, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=_seed_worker,
    )


def build_model(*, cfg: TrainConfig, moe_cfg, num_labels: int):
    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        use_moe=bool(cfg.use_moe),
        moe_cfg=moe_cfg,
        freeze_base=bool(cfg.freeze_base),
        aux_loss_weight=float(cfg.aux_loss_weight),
        head_type=cfg.head_type,
    ).to(DEVICE)


def train_full_then_test(
    *,
    cfg: TrainConfig,
    moe_cfg,
    train_dataset_full,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    step_print_moe: float,
    print_confusion_matrix: bool,
):
    print("\n===== Train FULL then Test =====")

    train_loader = make_train_loader_with_seed(train_dataset_full, cfg.train_batch_size, cfg.seed)

    model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))
    total_steps = len(train_loader) * cfg.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model, lr=cfg.lr, warmup_ratio=cfg.warmup_ratio, total_steps=total_steps
    )

    out = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.epochs,
        fusion_method=cfg.fusion_method,
        freeze_epochs=cfg.freeze_epochs,
        rolling_k=cfg.rolling_k,
        early_stop_patience=cfg.early_stop_patience,
        id2label=id2label,
        tag="[FULL] ",
        step_print_moe=step_print_moe,
    )

    if out.get("best_state_dict", None) is not None:
        model.load_state_dict(out["best_state_dict"])
        model.to(DEVICE)
        if out.get("best_epoch", None) is not None:
            print(f"Loaded best FULL model at epoch {out['best_epoch']}")

    print("\n===== Final TEST evaluation =====")
    logits, labels = collect_test_logits(model=model, dataloader=test_loader, fusion_method=cfg.fusion_method)
    preds = logits.argmax(axis=-1).tolist()
    labels_list = labels.tolist()

    if print_confusion_matrix:
        _print_confusion_matrix(
            labels_list,
            preds,
            id2label=id2label,
            normalize=True,
        )

    print("\nClassification report (TEST):")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(labels_list, preds, target_names=target_names, digits=4))

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"final_{cfg.output_name}")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    clear_model(model, optimizer, scheduler)


def train_full_multi_seed_then_test(
    *,
    cfg: TrainConfig,
    moe_cfg,
    train_dataset_full,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    seeds: list[int],
    print_confusion_matrix: bool = True,
    do_ensemble_logits: bool = True,
    verbose_ensemble_report: bool = False,
) -> dict:
    per_seed_metrics: list[dict] = []
    all_seed_logits: list[np.ndarray] = []

    for seed in seeds:
        print(f"\n===== FULL seed={seed} fusion={cfg.fusion_method} =====")
        set_determinism(int(seed))

        cfg_s = TrainConfig(**{**cfg.__dict__, "seed": int(seed)})
        train_loader = make_train_loader_with_seed(train_dataset_full, cfg_s.train_batch_size, int(seed))

        model = build_model(cfg=cfg_s, moe_cfg=moe_cfg, num_labels=len(label2id))
        total_steps = len(train_loader) * cfg_s.epochs
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model, lr=cfg_s.lr, warmup_ratio=cfg_s.warmup_ratio, total_steps=total_steps
        )

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg_s.epochs,
            fusion_method=cfg_s.fusion_method,
            freeze_epochs=cfg_s.freeze_epochs,
            rolling_k=cfg_s.rolling_k,
            early_stop_patience=cfg_s.early_stop_patience,
            id2label=id2label,
            tag=f"[FULL seed={seed}] ",
            step_print_moe=float(cfg_s.step_print_moe),
        )

        if out.get("best_state_dict", None) is not None:
            model.load_state_dict(out["best_state_dict"])
            model.to(DEVICE)

        logits, labels = collect_test_logits(model=model, dataloader=test_loader, fusion_method=cfg_s.fusion_method)
        m = logits_to_metrics(logits, labels)
        per_seed_metrics.append({"seed": int(seed), **m})
        all_seed_logits.append(logits)

        clear_model(model, optimizer, scheduler)

    accs = [r["acc"] for r in per_seed_metrics]
    f1s = [r["f1"] for r in per_seed_metrics]
    acc_mean, acc_std = _mean_std(accs)
    f1_mean, f1_std = _mean_std(f1s)

    ensemble_block = None
    if do_ensemble_logits and len(all_seed_logits) >= 2:
        ens_logits = np.mean(np.stack(all_seed_logits, axis=0), axis=0)
        ens_metrics = logits_to_metrics(ens_logits, labels)
        ensemble_block = {"metrics": ens_metrics}

        if verbose_ensemble_report:
            preds = ens_logits.argmax(axis=-1).tolist()
            labels_list = labels.tolist()
            print("\n===== Ensemble classification report (TEST) =====")
            target_names = [id2label[i] for i in range(len(id2label))]
            print(classification_report(labels_list, preds, target_names=target_names, digits=4))

        if print_confusion_matrix:
            preds = ens_logits.argmax(axis=-1).tolist()
            labels_list = labels.tolist()
            _print_confusion_matrix(labels_list, preds, id2label=id2label, normalize=True)

    out = {
        "per_seed": per_seed_metrics,
        "mean": {"acc": acc_mean, "acc_std": acc_std, "f1": f1_mean, "f1_std": f1_std},
        "ensemble": ensemble_block,
    }
    return out


def run_phase1_benchmark_kfold_plus_full(
    *,
    base_cfg: TrainConfig,
    moe_cfg,
    train_path: str,
    test_path: str,
    tokenizer,
    methods: list[str],
    seeds: list[int],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    train_dataset_full = AspectSentimentDataset(
        json_path=train_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=None,
    )
    label2id = train_dataset_full.label2id
    id2label = {v: k for k, v in label2id.items()}
    samples = train_dataset_full.samples
    y = [label2id[s["sentiment"]] for s in samples]

    test_dataset = AspectSentimentDataset(
        json_path=test_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=base_cfg.eval_batch_size, shuffle=False)

    all_results: dict = {}
    k = int(getattr(base_cfg, "k_folds", 0) or 0)

    for method in methods:
        print(f"\n==================== Benchmark fusion={method} ====================")
        cfg_method = TrainConfig(**{**base_cfg.__dict__, "fusion_method": method, "k_folds": k})

        per_seed_records: list[dict] = []

        for seed in seeds:
            cfg = TrainConfig(**{**cfg_method.__dict__, "seed": int(seed)})
            set_determinism(int(seed))

            seed_record: dict = {"seed": int(seed)}

            # K-fold CV if requested
            if cfg.k_folds and cfg.k_folds > 1:
                skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=int(seed))
                fold_val_f1: list[float] = []
                fold_test_f1: list[float] = []

                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, y), start=1):
                    train_samples = [samples[i] for i in train_idx]
                    val_samples = [samples[i] for i in val_idx]

                    train_ds = AspectSentimentDatasetFromSamples(
                        train_samples, tokenizer, cfg.max_len_sent, cfg.max_len_term, label2id
                    )
                    val_ds = AspectSentimentDatasetFromSamples(
                        val_samples, tokenizer, cfg.max_len_sent, cfg.max_len_term, label2id
                    )

                    train_loader = make_train_loader_with_seed(
                        train_ds, cfg.train_batch_size, int(seed) + 1000 * int(fold_idx)
                    )
                    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False)

                    model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))
                    total_steps = len(train_loader) * cfg.epochs
                    optimizer, scheduler = build_optimizer_and_scheduler(
                        model=model, lr=cfg.lr, warmup_ratio=cfg.warmup_ratio, total_steps=total_steps
                    )

                    out = run_training_loop(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epochs=cfg.epochs,
                        fusion_method=cfg.fusion_method,
                        freeze_epochs=cfg.freeze_epochs,
                        rolling_k=cfg.rolling_k,
                        early_stop_patience=cfg.early_stop_patience,
                        id2label=id2label,
                        tag=f"[Fold {fold_idx}] ",
                        step_print_moe=float(cfg.step_print_moe),
                    )

                    if out.get("best_state_dict", None) is not None:
                        model.load_state_dict(out["best_state_dict"])
                        model.to(DEVICE)

                    best_val = eval_model(
                        model=model,
                        dataloader=val_loader,
                        id2label=id2label,
                        verbose_report=False,
                        fusion_method=cfg.fusion_method,
                        f1_average="macro",
                    )
                    best_test = eval_model(
                        model=model,
                        dataloader=test_loader,
                        id2label=id2label,
                        verbose_report=False,
                        fusion_method=cfg.fusion_method,
                        f1_average="macro",
                    )

                    fold_val_f1.append(best_val["f1"])
                    fold_test_f1.append(best_test["f1"])
                    clear_model(model, optimizer, scheduler)

                seed_record["cv_val_f1_mean"] = float(np.mean(fold_val_f1))
                seed_record["cv_val_f1_std"] = float(np.std(fold_val_f1))
                seed_record["cv_test_f1_mean"] = float(np.mean(fold_test_f1))
                seed_record["cv_test_f1_std"] = float(np.std(fold_test_f1))

            # FULL train then TEST
            full_metrics = train_full_multi_seed_then_test(
                cfg=cfg,
                moe_cfg=moe_cfg,
                train_dataset_full=train_dataset_full,
                test_loader=test_loader,
                label2id=label2id,
                id2label=id2label,
                seeds=[int(seed)],
                print_confusion_matrix=False,
                do_ensemble_logits=False,
            )
            seed_record["full_test_acc"] = full_metrics["per_seed"][0]["acc"]
            seed_record["full_test_f1"] = full_metrics["per_seed"][0]["f1"]

            per_seed_records.append(seed_record)

        all_results[method] = {
            "per_seed": per_seed_records,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark aggregate written to: {output_path}")


def main(args: argparse.Namespace) -> None:
    # Build configs from config.py (Phase 2 style)
    cfg = build_train_config(args)
    moe_cfg = build_moe_config(args)

    
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    set_determinism(int(cfg.seed))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ===== Phase 1 benchmark mode =====
    if getattr(args, "benchmark_fusions", False):
        methods = _parse_str_list(getattr(args, "benchmark_methods", ""))
        if not methods:
            methods = list(FUSION_METHOD_CHOICES)

        seeds = _parse_int_list(getattr(args, "seeds", ""))
        if not seeds:
            n = int(getattr(args, "num_seeds", 1))
            seeds = [int(cfg.seed) + i for i in range(n)]

        out_path = os.path.join(cfg.output_dir, "phase1_benchmark_all.json")

        # Locked baseline: same args, only override (fusion, seed, use_moe)
        if getattr(args, "locked_baseline", False):
            base_cfg, base_moe = locked_baseline_config(args, fusion_method=cfg.fusion_method, seed=int(cfg.seed))
            cfg = base_cfg
            moe_cfg = base_moe

        run_phase1_benchmark_kfold_plus_full(
            base_cfg=cfg,
            moe_cfg=moe_cfg,
            train_path=train_path,
            test_path=test_path,
            tokenizer=tokenizer,
            methods=methods,
            seeds=seeds,
            output_path=out_path,
        )
        return

    # ===== Normal training mode =====
    train_dataset_full = AspectSentimentDataset(
        json_path=train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=None,
    )
    label2id = train_dataset_full.label2id
    id2label = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)

    test_dataset = AspectSentimentDataset(
        json_path=test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    # Case 1: single split
    if cfg.k_folds <= 1:
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

        model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))
        total_steps = len(train_loader) * cfg.epochs
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model, lr=cfg.lr, warmup_ratio=cfg.warmup_ratio, total_steps=total_steps
        )

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg.epochs,
            fusion_method=cfg.fusion_method,
            freeze_epochs=cfg.freeze_epochs,
            rolling_k=cfg.rolling_k,
            early_stop_patience=cfg.early_stop_patience,
            id2label=id2label,
            tag="",
            step_print_moe=float(cfg.step_print_moe),
        )

        if out.get("best_state_dict", None) is not None:
            model.load_state_dict(out["best_state_dict"])
            model.to(DEVICE)

        final_test = eval_model(
            model=model,
            dataloader=test_loader,
            id2label=id2label,
            verbose_report=cfg.verbose_report,
            fusion_method=cfg.fusion_method,
            f1_average="macro",
        )

        print(f"Final Test loss {final_test['loss']:.4f} F1 {final_test['f1']:.4f} acc {final_test['acc']:.4f}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, cfg.output_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        clear_model(model, optimizer, scheduler)
        return

    # Case 2: k fold CV
    if not cfg.train_full_only:
        print(f"Running StratifiedKFold with k={cfg.k_folds}")

        samples = train_dataset_full.samples
        y = [label2id[s["sentiment"]] for s in samples]

        skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=int(cfg.seed))

        fold_val_f1, fold_test_f1 = [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, y), start=1):
            print(f"\n===== Fold {fold_idx}/{cfg.k_folds} =====")

            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            train_ds = AspectSentimentDatasetFromSamples(
                train_samples, tokenizer, cfg.max_len_sent, cfg.max_len_term, label2id
            )
            val_ds = AspectSentimentDatasetFromSamples(
                val_samples, tokenizer, cfg.max_len_sent, cfg.max_len_term, label2id
            )

            train_loader = make_train_loader_with_seed(train_ds, cfg.train_batch_size, int(cfg.seed) + 1000 * fold_idx)
            val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False)

            model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))
            total_steps = len(train_loader) * cfg.epochs
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model, lr=cfg.lr, warmup_ratio=cfg.warmup_ratio, total_steps=total_steps
            )

            out = run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=cfg.epochs,
                fusion_method=cfg.fusion_method,
                freeze_epochs=cfg.freeze_epochs,
                rolling_k=cfg.rolling_k,
                early_stop_patience=cfg.early_stop_patience,
                id2label=id2label,
                tag=f"[Fold {fold_idx}] ",
                step_print_moe=float(cfg.step_print_moe),
            )

            if out.get("best_state_dict", None) is not None:
                model.load_state_dict(out["best_state_dict"])
                model.to(DEVICE)

            best_val = eval_model(
                model=model,
                dataloader=val_loader,
                id2label=label2id,
                verbose_report=False,
                fusion_method=cfg.fusion_method,
                f1_average="macro",
            )
            best_test = eval_model(
                model=model,
                dataloader=test_loader,
                id2label=label2id,
                verbose_report=False,
                fusion_method=cfg.fusion_method,
                f1_average="macro",
            )

            fold_val_f1.append(best_val["f1"])
            fold_test_f1.append(best_test["f1"])

            print(
                f"Fold {fold_idx} | Best rolling Val F1 {out.get('best_val_f1_rolling', 0.0):.4f} | "
                f"Val F1 {best_val['f1']:.4f} | Test F1 {best_test['f1']:.4f}"
            )

            os.makedirs(cfg.output_dir, exist_ok=True)
            save_path = os.path.join(cfg.output_dir, f"fold{fold_idx}_{cfg.output_name}")
            torch.save(model.state_dict(), save_path)
            print(f"Saved fold model to {save_path}")

            clear_model(model, optimizer, scheduler)

        print("\n===== CV Summary =====")
        print(f"Val macro-F1 mean {np.mean(fold_val_f1):.4f} std {np.std(fold_val_f1):.4f}")
        print(f"Test macro-F1 mean {np.mean(fold_test_f1):.4f} std {np.std(fold_test_f1):.4f}")

    if cfg.train_full_only:
        print("Skipping full training (train_full_only enabled)")
        return

    # ===== Full multi-seed then test (with optional ensemble logits) =====
    seeds = _parse_int_list(getattr(args, "seeds", ""))
    if not seeds:
        n = int(getattr(args, "num_seeds", 1))
        seeds = [int(cfg.seed) + i for i in range(n)]

    out = train_full_multi_seed_then_test(
        cfg=cfg,
        moe_cfg=moe_cfg,
        train_dataset_full=train_dataset_full,
        test_loader=test_loader,
        label2id=label2id,
        id2label=id2label,
        seeds=seeds,
        print_confusion_matrix=True,
        do_ensemble_logits=bool(getattr(args, "ensemble_logits", False)),
        verbose_ensemble_report=bool(getattr(args, "verbose_report", False)),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "full_multi_seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Full multi-seed summary written to {os.path.join(cfg.output_dir, 'full_multi_seed_summary.json')}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
