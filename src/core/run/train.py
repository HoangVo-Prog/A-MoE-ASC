import numpy as np
from sklearn.metrics import confusion_matrix

from src.core.utils.helper import set_seed, get_model
from .engine import run_training_loop
from src.core.utils.general import (
    cleanup_cuda,
    collect_test_logits,
    logits_to_metrics,
    mean_std,
    aggregate_confusions,
)
from src.core.utils.plotting import print_confusion_matrix

def train_multi_seed(
    config,
    method,
    full_train_dataloader,
    test_loader,
    label2id,
    id2label,
    seeds,
    print_cf_matrix=False,
    do_ensemble_logits=True,
    verbose_ensemble_report=False,
):
    print("\n===== Train FULL (multi-seed) then Test =====")
    print(f"Seeds: {seeds}")

    num_classes = len(label2id)

    per_seed_metrics = []
    all_seed_logits = []
    all_seed_cms = []
    
    labels_last = None 
    
    for seed in seeds:
        print(f"\n===== FULL seed={seed} fusion={method} =====")
        set_seed(int(seed))
        
        model = get_model(config)
        out = run_training_loop(
            cfg=config,
            model=model,
            method=method,
            train_loader=full_train_dataloader,
            val_loader=None,
            test_loader=test_loader,
            id2label=id2label,
            tag=f"[FULL seed={seed}] ",
        )
        
        if out.get("best_state_dict") is not None:
            model.load_state_dict(out["best_state_dict"])
            if out.get("best_epoch") is not None:
                print(f"Loaded best FULL model from epoch {out.get('best_epoch')}")

        logits, labels = collect_test_logits(
            model=model,
            test_loader=test_loader,
            fusion_method=method,
        )
        labels_last = labels

        m = logits_to_metrics(logits, labels)
        preds = logits.argmax(axis=-1)
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        all_seed_cms.append(cm)

        extra = out.get("last_test_metrics") or {}
        per_seed_metrics.append(
            {
                "seed": int(seed),
                **m,
                "calibration": extra.get("calibration"),
                "moe_metrics": extra.get("moe_metrics"),
                "f1_per_class": (
                    extra.get("f1_per_class").tolist()
                    if hasattr(extra.get("f1_per_class"), "tolist")
                    else extra.get("f1_per_class")
                ),
            }
        )
        all_seed_logits.append(logits)

        del model
        cleanup_cuda()

    accs = [float(r["acc"]) for r in per_seed_metrics]
    f1s = [float(r["f1"]) for r in per_seed_metrics]
    acc_mean, acc_std = mean_std(accs)
    f1_mean, f1_std = mean_std(f1s)

    full_confusion_block = aggregate_confusions(all_seed_cms)

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

        if print_cf_matrix:
            print_confusion_matrix(
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
        
