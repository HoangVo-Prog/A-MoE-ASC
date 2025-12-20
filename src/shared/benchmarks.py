from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .datasets import AspectSentimentDataset, AspectSentimentDatasetFromSamples
from .seed import set_all_seeds, set_determinism, make_train_loader_with_seed
from .utils import _mean_std, _aggregate_confusions, _cfg_to_dict, cleanup_cuda


ModelFactory = Callable[[Any, int, Dict[str, Any]], torch.nn.Module]
TrainLoopFn = Callable[..., Dict[str, Any]]
EvalFn = Callable[..., Dict[str, Any]]
TrainLoopKwargsFactory = Callable[[Any, Dict[str, Any]], Dict[str, Any]]
FullTrainFn = Callable[..., Dict[str, Any]]


def run_benchmark_kfold_plus_full(
    *,
    base_cfg: Any,
    train_path: str,
    test_path: str,
    tokenizer,
    methods: list[str],
    seeds: list[int],
    output_path: str,
    model_factory: ModelFactory,
    run_training_loop_fn: TrainLoopFn,
    eval_model_fn: EvalFn,
    train_full_multi_seed_then_test_fn: FullTrainFn,
    trainloop_kwargs_factory: Optional[TrainLoopKwargsFactory] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    extra = extra or {}
    if trainloop_kwargs_factory is None:
        trainloop_kwargs_factory = lambda cfg, extra: {}

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
    num_classes = len(label2id)

    test_dataset = AspectSentimentDataset(
        json_path=test_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=base_cfg.eval_batch_size, shuffle=False)

    k_folds = int(getattr(base_cfg, "k_folds", 0) or 0)

    all_results: dict = {
        "benchmark_type": "kfold_plus_full_multiseed",
        "methods": methods,
        "seeds": seeds,
        "k_folds": k_folds,
        "config": _cfg_to_dict(base_cfg),
        "runs": {},
        "summary": {},
        "full_confusion": {},
        "ensemble": {},
    }

    per_method_seed_records: dict[str, list[dict]] = {m: [] for m in methods}

    for method in methods:
        cfg_method = type(base_cfg)(**{**base_cfg.__dict__, "fusion_method": method, "k_folds": k_folds})
        per_method_seed_records[method] = []

        for seed in seeds:
            set_all_seeds(int(seed))
            set_determinism(int(seed))

            cfg = type(cfg_method)(**{**cfg_method.__dict__, "seed": int(seed)})

            if k_folds and k_folds > 1:
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=int(seed))

                fold_val_f1: list[float] = []
                fold_test_f1: list[float] = []
                fold_val_cms: list[np.ndarray] = []
                fold_test_cms: list[np.ndarray] = []

                for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
                    train_samples = [samples[i] for i in tr_idx]
                    val_samples = [samples[i] for i in va_idx]

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

                    fold_seed = int(seed) + 1000 * int(fold_idx)
                    train_loader = make_train_loader_with_seed(train_ds, cfg.train_batch_size, fold_seed)
                    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False)

                    model = model_factory(cfg, num_classes, extra)

                    out = run_training_loop_fn(
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
                        tag=f"[CV {method} seed={seed} fold={fold_idx}] ",
                        **trainloop_kwargs_factory(cfg, extra),
                    )

                    best_sd = out.get("best_state_dict", None)
                    if best_sd is not None:
                        model.load_state_dict(best_sd)

                    val_m = eval_model_fn(
                        model=model,
                        dataloader=val_loader,
                        id2label=id2label,
                        verbose_report=False,
                        print_confusion_matrix=False,
                        fusion_method=cfg.fusion_method,
                        f1_average="macro",
                        return_confusion=True,
                    )
                    test_m = eval_model_fn(
                        model=model,
                        dataloader=test_loader,
                        id2label=id2label,
                        verbose_report=False,
                        print_confusion_matrix=False,
                        fusion_method=cfg.fusion_method,
                        f1_average="macro",
                        return_confusion=True,
                    )

                    fold_val_f1.append(float(val_m["f1"]))
                    fold_test_f1.append(float(test_m["f1"]))
                    fold_val_cms.append(np.asarray(val_m["confusion"], dtype=np.float64))
                    fold_test_cms.append(np.asarray(test_m["confusion"], dtype=np.float64))

                    del model
                    cleanup_cuda()

                cv_val_mean, cv_val_std = _mean_std(fold_val_f1)
                cv_test_mean, cv_test_std = _mean_std(fold_test_f1)

                record = {
                    "fusion_method": method,
                    "seed": int(seed),
                    "cv_val_f1_folds": fold_val_f1,
                    "cv_test_f1_folds": fold_test_f1,
                    "cv_val_f1_mean": float(cv_val_mean),
                    "cv_val_f1_std": float(cv_val_std),
                    "cv_test_f1_mean": float(cv_test_mean),
                    "cv_test_f1_std": float(cv_test_std),
                    "cv_val_confusion": _aggregate_confusions(fold_val_cms),
                    "cv_test_confusion": _aggregate_confusions(fold_test_cms),
                }
                per_method_seed_records[method].append(record)

        cfg_full = type(cfg_method)(**{**cfg_method.__dict__})
        full_out = train_full_multi_seed_then_test_fn(
            cfg=cfg_full,
            train_dataset_full=train_dataset_full,
            test_loader=test_loader,
            label2id=label2id,
            id2label=id2label,
            seeds=[int(s) for s in seeds],
            print_confusion_matrix=False,
            do_ensemble_logits=bool(getattr(base_cfg, "do_ensemble_logits", True)),
            verbose_ensemble_report=False,
            extra=extra,
            model_factory=model_factory,
            run_training_loop_fn=run_training_loop_fn,
            trainloop_kwargs_factory=trainloop_kwargs_factory,
        )

        full_by_seed = {int(r["seed"]): r for r in full_out.get("per_seed", [])}
        for rec in per_method_seed_records[method]:
            s = int(rec["seed"])
            if s in full_by_seed:
                rec["full_test_acc"] = float(full_by_seed[s]["acc"])
                rec["full_test_f1"] = float(full_by_seed[s]["f1"])

        all_results["full_confusion"][method] = full_out.get("confusion", {})
        ens = full_out.get("ensemble", None)
        if ens is not None:
            all_results["ensemble"][method] = {
                "full_ens_test_acc": float(ens["metrics"]["acc"]),
                "full_ens_test_f1": float(ens["metrics"]["f1"]),
                "confusion": ens.get("confusion", None),
            }

    all_results["runs"] = per_method_seed_records

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

        method_sum: dict[str, Any] = {
            "cv_val_f1_mean_over_seeds": float(m1),
            "cv_val_f1_std_over_seeds": float(s1),
            "cv_test_f1_mean_over_seeds": float(m2),
            "cv_test_f1_std_over_seeds": float(s2),
            "full_test_f1_mean_over_seeds": float(m3),
            "full_test_f1_std_over_seeds": float(s3),
            "full_test_acc_mean_over_seeds": float(m4),
            "full_test_acc_std_over_seeds": float(s4),
        }

        if len(recs) > 0 and "cv_test_confusion" in recs[0]:
            cv_val_seed_means = [np.asarray(r["cv_val_confusion"]["cm_mean"], dtype=np.float64) for r in recs]
            cv_test_seed_means = [np.asarray(r["cv_test_confusion"]["cm_mean"], dtype=np.float64) for r in recs]
            method_sum["cv_val_confusion_over_seeds"] = _aggregate_confusions(cv_val_seed_means)
            method_sum["cv_test_confusion_over_seeds"] = _aggregate_confusions(cv_test_seed_means)

        if method in all_results.get("full_confusion", {}):
            method_sum["full_confusion_over_seeds"] = all_results["full_confusion"][method]

        if method in all_results.get("ensemble", {}):
            method_sum["full_ens_test_acc"] = float(all_results["ensemble"][method]["full_ens_test_acc"])
            method_sum["full_ens_test_f1"] = float(all_results["ensemble"][method]["full_ens_test_f1"])
            if all_results["ensemble"][method].get("confusion", None) is not None:
                method_sum["full_ens_confusion"] = all_results["ensemble"][method]["confusion"]

        summary[method] = method_sum

    if "sent" in summary:
        base = float(summary["sent"]["full_test_f1_mean_over_seeds"])
        for method in methods:
            summary[method]["delta_full_test_f1_vs_sent"] = float(
                float(summary[method]["full_test_f1_mean_over_seeds"]) - base
            )
            if "full_ens_test_f1" in summary[method] and "full_ens_test_f1" in summary["sent"]:
                summary[method]["delta_full_ens_test_f1_vs_sent"] = float(
                    float(summary[method]["full_ens_test_f1"]) - float(summary["sent"]["full_ens_test_f1"])
                )

    all_results["summary"] = summary

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results
