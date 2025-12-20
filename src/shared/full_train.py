from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .seed import set_all_seeds, set_determinism, make_train_loader_with_seed
from .logit import collect_test_logits, logits_to_metrics
from .utils import _mean_std, cleanup_cuda
from .plotting import _print_confusion_matrix, _aggregate_confusions


ModelFactory = Callable[[Any, int, Dict[str, Any]], torch.nn.Module]
TrainLoopFn = Callable[..., Dict[str, Any]]
TrainLoopKwargsFactory = Callable[[Any, Dict[str, Any]], Dict[str, Any]]


def train_full_multi_seed_then_test_generic(
    *,
    cfg: Any,
    train_dataset_full,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    seeds: list[int],
    print_confusion_matrix: bool = True,
    do_ensemble_logits: bool = True,
    verbose_ensemble_report: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    model_factory: Optional[ModelFactory] = None,
    run_training_loop_fn: Optional[TrainLoopFn] = None,
    trainloop_kwargs_factory: Optional[TrainLoopKwargsFactory] = None,
) -> dict:
    if model_factory is None:
        raise ValueError("model_factory is required")
    if run_training_loop_fn is None:
        raise ValueError("run_training_loop_fn is required")
    if trainloop_kwargs_factory is None:
        trainloop_kwargs_factory = lambda cfg, extra: {}
    extra = extra or {}

    print("\n===== Train FULL (multi-seed) then Test =====")
    print(f"Seeds: {seeds}")

    num_classes = len(label2id)

    per_seed_metrics: list[dict] = []
    all_seed_logits: list[np.ndarray] = []
    all_seed_cms: list[np.ndarray] = []

    labels_last = None

    for seed in seeds:
        print(f"\n===== FULL seed={seed} fusion={cfg.fusion_method} =====")
        set_all_seeds(int(seed))
        set_determinism(int(seed))

        train_loader = make_train_loader_with_seed(train_dataset_full, cfg.train_batch_size, int(seed))

        model = model_factory(cfg, num_classes, extra)

        out = run_training_loop_fn(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            lr=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            epochs=cfg.epochs,
            fusion_method=cfg.fusion_method,
            freeze_epochs=cfg.freeze_epochs,
            rolling_k=cfg.rolling_k,
            early_stop_patience=cfg.early_stop_patience,
            id2label=id2label,
            tag=f"[FULL seed={seed}] ",
            **trainloop_kwargs_factory(cfg, extra),
        )

        if out.get("best_state_dict") is not None:
            model.load_state_dict(out["best_state_dict"])
            if out.get("best_epoch") is not None:
                print(f"Loaded best FULL model from epoch {out.get('best_epoch')}")

        logits, labels = collect_test_logits(
            model=model,
            dataloader=test_loader,
            fusion_method=cfg.fusion_method,
        )
        labels_last = labels

        m = logits_to_metrics(logits, labels)
        preds = logits.argmax(axis=-1)
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        all_seed_cms.append(cm)

        per_seed_metrics.append({"seed": int(seed), **m})
        all_seed_logits.append(logits)

        del model
        cleanup_cuda()

    accs = [float(r["acc"]) for r in per_seed_metrics]
    f1s = [float(r["f1"]) for r in per_seed_metrics]
    acc_mean, acc_std = _mean_std(accs)
    f1_mean, f1_std = _mean_std(f1s)

    full_confusion_block = _aggregate_confusions(all_seed_cms)

    ensemble_block = None
    if do_ensemble_logits and len(all_seed_logits) >= 2:
        ens_logits = np.mean(np.stack(all_seed_logits, axis=0), axis=0)
        if labels_last is None:
            raise RuntimeError("labels not collected for ensemble")

        ens_metrics = logits_to_metrics(ens_logits, labels_last)

        ens_preds = ens_logits.argmax(axis=-1)
        ens_cm = confusion_matrix(labels_last, ens_preds, labels=list(range(num_classes)))

        ensemble_block = {
            "metrics": ens_metrics,
            "confusion": {
                "cm": ens_cm.tolist(),
                "cm_normalized": (
                    ens_cm / np.clip(ens_cm.sum(axis=1, keepdims=True), 1e-12, None)
                ).tolist(),
            },
        }

        if verbose_ensemble_report:
            print("verbose_ensemble_report is enabled but generic full train does not print report by default")

        if print_confusion_matrix:
            _print_confusion_matrix(
                labels_last.tolist(),
                ens_preds.tolist(),
                id2label=id2label,
                normalize=True,
            )

    return {
        "per_seed": per_seed_metrics,
        "mean": {"acc": float(acc_mean), "acc_std": float(acc_std), "f1": float(f1_mean), "f1_std": float(f1_std)},
        "confusion": full_confusion_block,
        "ensemble": ensemble_block,
    }
