from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
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
    """Best-effort determinism for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


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


def _aggregate_confusions(cms: list[np.ndarray]) -> dict:
    # cms: list of [C, C] raw count matrices
    if len(cms) == 0:
        return {}

    arr = np.stack([np.asarray(cm, dtype=np.float64) for cm in cms], axis=0)  # [N, C, C]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    # Normalize AFTER aggregating raw counts
    denom = mean.sum(axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    mean_norm = mean / denom
    std_norm = std / denom

    return {
        "cm_mean": mean.tolist(),
        "cm_std": std.tolist(),
        "cm_mean_normalized": mean_norm.tolist(),
        "cm_std_normalized": std_norm.tolist(),
    }


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

    print("\n===== Train FULL (multi-seed) then Test =====")
    print(f"Seeds: {seeds}")
    per_seed_metrics: list[dict] = []
    all_seed_logits: list[np.ndarray] = []
    all_seed_cms: list[np.ndarray] = []
    num_classes = len(label2id)

    for i, seed in enumerate(seeds, start=1):
        print(f"\n----- FULL run {i}/{len(seeds)} | seed={seed} -----")
        set_all_seeds(seed)
        set_determinism(seed)

        train_loader = make_train_loader_with_seed(
            train_dataset_full, 
            cfg.train_batch_size, 
            seed
        )

        model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=num_classes)

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=None,
            scheduler=None,
            lr=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            epochs=cfg.epochs,
            fusion_method=cfg.fusion_method,
            freeze_epochs=cfg.freeze_epochs,
            rolling_k=cfg.rolling_k,
            early_stop_patience=cfg.early_stop_patience,
            id2label=id2label,
            tag=f"[FULL seed={seed}] ",
            step_print_moe=float(cfg.step_print_moe),
        )

        best_sd = out.get("best_state_dict", None)
        if best_sd is not None:
            model.load_state_dict(best_sd)
            model.to(DEVICE)
            be = out.get("best_epoch", None)
            if be is not None:
                print(f"Loaded best FULL model at epoch {be} (seed={seed})")

        logits, labels = collect_test_logits(
            model=model, dataloader=test_loader, fusion_method=cfg.fusion_method
        )
        
        m = logits_to_metrics(logits, labels)
        preds = logits.argmax(axis=-1)
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))  # raw counts
        all_seed_cms.append(cm)

        per_seed_metrics.append({"seed": int(seed), **m})
        all_seed_logits.append(logits)

        clear_model(model, None, None)

    accs = [r["acc"] for r in per_seed_metrics]
    f1s = [r["f1"] for r in per_seed_metrics]
    acc_mean, acc_std = _mean_std(accs)
    f1_mean, f1_std = _mean_std(f1s)

    # Aggregate FULL confusion across seeds
    full_confusion_block = _aggregate_confusions(all_seed_cms)

    ensemble_block = None
    if do_ensemble_logits and len(all_seed_logits) >= 2:
        ens_logits = np.mean(np.stack(all_seed_logits, axis=0), axis=0)
        ens_metrics = logits_to_metrics(ens_logits, labels)

        ens_preds = ens_logits.argmax(axis=-1)
        ens_cm = confusion_matrix(labels, ens_preds, labels=list(range(num_classes)))

        ensemble_block = {
            "metrics": ens_metrics,
            "confusion": {
                "cm": ens_cm.tolist(),
                "cm_normalized": (ens_cm / np.clip(ens_cm.sum(axis=1, keepdims=True), 1e-12, None)).tolist(),
            },
        }

        if verbose_ensemble_report:
            preds_list = ens_preds.tolist()
            labels_list = labels.tolist()
            print("\n===== Ensemble classification report (TEST) =====")
            target_names = [id2label[i] for i in range(len(id2label))]
            print(classification_report(labels_list, preds_list, target_names=target_names, digits=4))

        if print_confusion_matrix:
            preds_list = ens_preds.tolist()
            labels_list = labels.tolist()
            _print_confusion_matrix(labels_list, preds_list, id2label=id2label, normalize=True)

    out = {
        "per_seed": per_seed_metrics,
        "mean": {"acc": acc_mean, "acc_std": acc_std, "f1": f1_mean, "f1_std": f1_std},
        "confusion": full_confusion_block, 
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

    k = int(base_cfg.k_folds)

    all_results: dict = {
        "benchmark_type": "kfold_plus_full_multiseed",
        "methods": methods,
        "seeds": seeds,
        "k_folds": k,
        "config": base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else base_cfg.__dict__,
        "runs": {},
        "summary": {},
    }

    per_method_seed_records: dict[str, list[dict]] = {m: [] for m in methods}
    num_classes = len(label2id)

    for method in methods:
        cfg_method = TrainConfig(**{**base_cfg.__dict__, "fusion_method": method, "k_folds": k})

        for seed in seeds:
            cfg = TrainConfig(**{**cfg_method.__dict__, "seed": int(seed)})
            set_all_seeds(int(seed))
            set_determinism(int(seed))

            # K-fold CV if requested
            if cfg.k_folds and cfg.k_folds > 1:
                skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=int(seed))

                fold_val_f1: list[float] = []
                fold_test_f1: list[float] = []

                fold_val_cms: list[np.ndarray] = []
                fold_test_cms: list[np.ndarray] = []

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

                    model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=num_classes)

                    out = run_training_loop(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=None,
                        scheduler=None,
                        lr=cfg.lr,
                        warmup_ratio=cfg.warmup_ratio,
                        epochs=cfg.epochs,
                        fusion_method=cfg.fusion_method,
                        freeze_epochs=cfg.freeze_epochs,
                        rolling_k=cfg.rolling_k,
                        early_stop_patience=cfg.early_stop_patience,
                        id2label=id2label,
                        tag=f"[CV {method} seed={seed} fold={fold_idx}] ",
                        step_print_moe=float(cfg.step_print_moe),
                    )

                    if out.get("best_state_dict", None) is not None:
                        model.load_state_dict(out["best_state_dict"])
                        model.to(DEVICE)

                    val_m = eval_model(
                        model=model,
                        dataloader=val_loader,
                        id2label=id2label,
                        verbose_report=False,
                        print_confusion_matrix=False,
                        fusion_method=cfg.fusion_method,
                        return_confusion=True,
                        f1_average="macro",
                    )
                    test_m = eval_model(
                        model=model,
                        dataloader=test_loader,
                        id2label=id2label,
                        verbose_report=False,
                        print_confusion_matrix=False,
                        fusion_method=cfg.fusion_method,
                        return_confusion=True,
                        f1_average="macro",
                    )

                    fold_val_f1.append(val_m["f1"])
                    fold_test_f1.append(test_m["f1"])

                    fold_val_cms.append(np.asarray(val_m["confusion"], dtype=np.float64))
                    fold_test_cms.append(np.asarray(test_m["confusion"], dtype=np.float64))

                    clear_model(model, None, None)

                cv_val_mean, cv_val_std = _mean_std(fold_val_f1)
                cv_test_mean, cv_test_std = _mean_std(fold_test_f1)

                # Aggregate confusion across folds for this seed
                cv_val_conf = _aggregate_confusions(fold_val_cms)
                cv_test_conf = _aggregate_confusions(fold_test_cms)

                record = {
                    "fusion_method": method,
                    "seed": int(seed),
                    "cv_val_f1_folds": fold_val_f1,
                    "cv_test_f1_folds": fold_test_f1,
                    "cv_val_f1_mean": float(cv_val_mean),
                    "cv_val_f1_std": float(cv_val_std),
                    "cv_test_f1_mean": float(cv_test_mean),
                    "cv_test_f1_std": float(cv_test_std),
                    "cv_val_confusion": cv_val_conf,
                    "cv_test_confusion": cv_test_conf,
                }
                per_method_seed_records[method].append(record)

        # ===== FULL multi-seed then test (and ensemble) =====
        # IMPORTANT: run FULL across ALL seeds, not a single last seed
        cfg_full = TrainConfig(**{**cfg_method.__dict__})
        full_out = train_full_multi_seed_then_test(
            cfg=cfg_full,
            moe_cfg=moe_cfg,
            train_dataset_full=train_dataset_full,
            test_loader=test_loader,
            label2id=label2id,
            id2label=id2label,
            seeds=[int(s) for s in seeds],
            print_confusion_matrix=False,
            do_ensemble_logits=getattr(base_cfg, "do_ensemble_logits", True),
            verbose_ensemble_report=False,
        )

        # Merge FULL per-seed metrics back into each record (same seed order)
        full_by_seed = {r["seed"]: r for r in full_out["per_seed"]}
        for rec in per_method_seed_records[method]:
            s = int(rec["seed"])
            if s in full_by_seed:
                rec["full_test_acc"] = float(full_by_seed[s]["acc"])
                rec["full_test_f1"] = float(full_by_seed[s]["f1"])

        # Attach method-level FULL confusion aggregation across seeds
        # This is the requested cm_mean/cm_std for FULL run
        all_results.setdefault("full_confusion", {})
        all_results["full_confusion"][method] = full_out.get("confusion", {})

        # Attach method-level ensemble metrics (and confusion if present)
        ens = full_out.get("ensemble", None)
        if ens is not None:
            all_results.setdefault("ensemble", {})
            all_results["ensemble"][method] = {
                "full_ens_test_acc": float(ens["metrics"]["acc"]),
                "full_ens_test_f1": float(ens["metrics"]["f1"]),
                "confusion": ens.get("confusion", None),
            }

    all_results["runs"] = per_method_seed_records

    # ===== Aggregate summary across seeds per method =====
    summary: dict[str, dict] = {}
    for method in methods:
        recs = per_method_seed_records[method]
        cv_val_means = [float(r["cv_val_f1_mean"]) for r in recs]
        cv_test_means = [float(r["cv_test_f1_mean"]) for r in recs]
        full_f1s = [float(r.get("full_test_f1", 0.0)) for r in recs]
        full_accs = [float(r.get("full_test_acc", 0.0)) for r in recs]

        m1, s1 = _mean_std(cv_val_means)
        m2, s2 = _mean_std(cv_test_means)
        m3, s3 = _mean_std(full_f1s)
        m4, s4 = _mean_std(full_accs)

        method_sum = {
            "cv_val_f1_mean_over_seeds": float(m1),
            "cv_val_f1_std_over_seeds": float(s1),
            "cv_test_f1_mean_over_seeds": float(m2),
            "cv_test_f1_std_over_seeds": float(s2),
            "full_test_f1_mean_over_seeds": float(m3),
            "full_test_f1_std_over_seeds": float(s3),
            "full_test_acc_mean_over_seeds": float(m4),
            "full_test_acc_std_over_seeds": float(s4),
        }

        # NEW: aggregate k-fold confusion across seeds (using per-seed fold-mean matrices)
        if len(recs) > 0 and "cv_test_confusion" in recs[0]:
            cv_val_seed_means = [np.asarray(r["cv_val_confusion"]["cm_mean"], dtype=np.float64) for r in recs]
            cv_test_seed_means = [np.asarray(r["cv_test_confusion"]["cm_mean"], dtype=np.float64) for r in recs]
            method_sum["cv_val_confusion_over_seeds"] = _aggregate_confusions(cv_val_seed_means)
            method_sum["cv_test_confusion_over_seeds"] = _aggregate_confusions(cv_test_seed_means)

        # FULL confusion across seeds already computed by train_full_multi_seed_then_test
        if "full_confusion" in all_results and method in all_results["full_confusion"]:
            method_sum["full_confusion_over_seeds"] = all_results["full_confusion"][method]

        if "ensemble" in all_results and method in all_results["ensemble"]:
            method_sum["full_ens_test_acc"] = float(all_results["ensemble"][method]["full_ens_test_acc"])
            method_sum["full_ens_test_f1"] = float(all_results["ensemble"][method]["full_ens_test_f1"])
            if all_results["ensemble"][method].get("confusion", None) is not None:
                method_sum["full_ens_confusion"] = all_results["ensemble"][method]["confusion"]

        summary[method] = method_sum

    # Deltas vs sent baseline (if present)
    if "sent" in summary:
        base = summary["sent"]["full_test_f1_mean_over_seeds"]
        for method in methods:
            summary[method]["delta_full_test_f1_vs_sent"] = float(
                summary[method]["full_test_f1_mean_over_seeds"] - base
            )
            if "full_ens_test_f1" in summary[method] and "full_ens_test_f1" in summary["sent"]:
                summary[method]["delta_full_ens_test_f1_vs_sent"] = float(
                    summary[method]["full_ens_test_f1"] - summary["sent"]["full_ens_test_f1"]
                )

    all_results["summary"] = summary

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


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
