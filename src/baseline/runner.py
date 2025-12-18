import os
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import TrainConfig, locked_baseline_config
from constants import DEVICE
from datasets import AspectSentimentDataset, AspectSentimentDatasetFromSamples
from engine import eval_model, run_training_loop, _print_confusion_matrix, logits_to_metrics, collect_test_logits
from model import BertConcatClassifier
from optim import build_optimizer_and_scheduler
from cli import parse_args
from plotting import plot_history
import random
import json
import hashlib
import os


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
def make_train_loader_with_seed(train_dataset_full, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        train_dataset_full,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )


def clear_model(model, optimizer, scheduler):
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    

def build_train_config(args) -> TrainConfig:
    if getattr(args, "locked_baseline", False):
        return locked_baseline_config(
            fusion_method=args.fusion_method,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
    return TrainConfig(
        model_name=args.model_name,
        fusion_method=args.fusion_method,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        freeze_epochs=args.freeze_epochs,
        rolling_k=args.rolling_k,
        early_stop_patience=args.early_stop_patience,
        k_folds=args.k_folds,
        seed=args.seed,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        output_dir=args.output_dir,
        output_name=args.output_name,
        verbose_report=args.verbose_report,
        train_full_only=args.train_full_only,
        head_type=args.head_type,
    )

def build_model(*, cfg: TrainConfig, num_labels: int):
    return BertConcatClassifier(
        cfg.model_name, 
        num_labels=num_labels, 
        dropout=cfg.dropout,
        head_type=cfg.head_type,
    ).to(DEVICE)


def _parse_int_list(csv: str) -> list[int]:
    s = (csv or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    return out


def _parse_str_list(csv: str) -> list[str]:
    s = (csv or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _run_single_method_single_seed(
    *,
    base_cfg: TrainConfig,
    fusion_method: str,
    seed: int,
    train_dataset,
    val_dataset,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    run_dir: str,
) -> Dict[str, float]:
    """Train on train, select best on rolling val F1, then evaluate on test."""
    os.makedirs(run_dir, exist_ok=True)

    # Ensure the same initialization and data order for this (method, seed) run.
    set_all_seeds(seed)
    set_determinism(seed)

    cfg = TrainConfig(**{**base_cfg.__dict__, "fusion_method": fusion_method, "seed": seed})

    train_loader = make_train_loader_with_seed(train_dataset, cfg.train_batch_size, seed)
    val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    model = build_model(cfg=cfg, num_labels=len(label2id))
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
        fusion_method=fusion_method,
        freeze_epochs=cfg.freeze_epochs,
        rolling_k=cfg.rolling_k,
        early_stop_patience=cfg.early_stop_patience,
        id2label=id2label,
        tag=f"[PHASE1 seed={seed} method={fusion_method}] ",
    )

    history = out.get("history")
    if history is not None:
        plot_history(history, save_dir=run_dir, prefix="")

    if out.get("best_state_dict") is not None:
        model.load_state_dict(out["best_state_dict"])
        model.to(DEVICE)

    test_metrics = eval_model(
        model=model,
        dataloader=test_loader,
        id2label=id2label,
        print_confusion_matrix=False,
        verbose_report=False,
        fusion_method=fusion_method,
        f1_average="macro",
    )

    # Save model checkpoint for this run.
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    # Save a compact JSON record.
    record = {
        "seed": seed,
        "fusion_method": fusion_method,
        "test_macro_f1": float(test_metrics["f1"]),
        "test_acc": float(test_metrics["acc"]),
        "best_epoch": out.get("best_epoch"),
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    clear_model(model, optimizer, scheduler)
    return record


def run_phase1_benchmark(
    *,
    base_cfg: TrainConfig,
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer,
    methods: list[str],
    seeds: list[int],
) -> None:
    """Benchmark multiple fusion methods across multiple seeds."""
    os.makedirs(base_cfg.output_dir, exist_ok=True)

    train_dataset = AspectSentimentDataset(
        json_path=train_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=None,
    )
    label2id = train_dataset.label2id
    id2label = {v: k for k, v in label2id.items()}

    val_dataset = AspectSentimentDataset(
        json_path=val_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=label2id,
    )
    test_dataset = AspectSentimentDataset(
        json_path=test_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=base_cfg.eval_batch_size, shuffle=False)

    # Results structure: method -> list of f1 by seed
    rows = []
    for seed in seeds:
        for method in methods:
            run_dir = os.path.join(base_cfg.output_dir, "phase1", method, f"seed_{seed}")
            rec = _run_single_method_single_seed(
                base_cfg=base_cfg,
                fusion_method=method,
                seed=seed,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_loader=test_loader,
                label2id=label2id,
                id2label=id2label,
                run_dir=run_dir,
            )
            rows.append(rec)

    # Summaries per method.
    by_method: dict[str, list[float]] = {m: [] for m in methods}
    for r in rows:
        by_method[r["fusion_method"]].append(float(r["test_macro_f1"]))

    summary = []
    for m in methods:
        arr = np.array(by_method[m], dtype=float)
        mean = float(arr.mean()) if arr.size else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary.append({"fusion_method": m, "mean_macro_f1": mean, "std_macro_f1": std})

    # Baseline comparison: sent only.
    sent_mean = next((x["mean_macro_f1"] for x in summary if x["fusion_method"] == "sent"), None)
    if sent_mean is not None:
        for x in summary:
            x["delta_vs_sent"] = float(x["mean_macro_f1"] - sent_mean)

    out_dir = os.path.join(base_cfg.output_dir, "phase1")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results_per_seed.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write CSVs for quick copy.
    import csv

    with open(os.path.join(out_dir, "results_per_seed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fusion_method", "seed", "test_macro_f1", "test_acc", "best_epoch"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "fusion_method": r["fusion_method"],
                    "seed": r["seed"],
                    "test_macro_f1": r["test_macro_f1"],
                    "test_acc": r["test_acc"],
                    "best_epoch": r["best_epoch"],
                }
            )

    # Sort summary by mean descending.
    summary_sorted = sorted(summary, key=lambda x: x["mean_macro_f1"], reverse=True)
    summary_fields = ["fusion_method", "mean_macro_f1", "std_macro_f1"]
    if sent_mean is not None:
        summary_fields.append("delta_vs_sent")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_sorted:
            w.writerow({k: r.get(k) for k in summary_fields})


def train_full_then_test(
    *,
    cfg: TrainConfig,
    train_dataset_full,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    print_confusion_matrix: bool
):
    print("\n===== Train FULL then Test =====")

    train_loader = make_train_loader_with_seed(train_dataset_full, cfg.train_batch_size, cfg.seed)

    model = build_model(cfg=cfg, num_labels=len(label2id))
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
    )

    if out["best_state_dict"] is not None:
        model.load_state_dict(out["best_state_dict"])
        model.to(DEVICE)
        print(f"Loaded best FULL model at epoch {out['best_epoch']}")

    print("\n===== Final TEST evaluation =====")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=None,
                fusion_method=cfg.fusion_method,
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
             
    if print_confusion_matrix:
        _print_confusion_matrix(
            all_labels,
            all_preds,
            id2label=id2label,
            normalize=True,
        )

    print("\nClassification report (TEST):")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"final_{cfg.output_name}")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    clear_model(model, optimizer, scheduler)


def train_full_multi_seed_then_test(
    *,
    cfg: TrainConfig,
    train_dataset_full,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    seeds: list[int],
    print_confusion_matrix: bool = True,
    do_ensemble_logits: bool = True,
):
    print("\n===== Train FULL (multi-seed) then Test =====")
    print(f"Seeds: {seeds}")

    per_seed_metrics = []
    sum_logits = None
    fixed_labels = None

    for i, seed in enumerate(seeds, start=1):
        print(f"\n----- FULL run {i}/{len(seeds)} | seed={seed} -----")
        set_all_seeds(seed)

        train_loader = make_train_loader_with_seed(
            train_dataset_full, batch_size=cfg.train_batch_size, seed=seed
        )

        model = build_model(cfg=cfg, num_labels=len(label2id))
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
            tag=f"[FULL seed={seed}] ",
        )


        logits, labels = collect_test_logits(
            model=model, test_loader=test_loader, fusion_method=cfg.fusion_method
        )

        if fixed_labels is None:
            fixed_labels = labels
        else:
            # sanity check: labels phải giống nhau giữa các seed
            if not np.array_equal(fixed_labels, labels):
                raise RuntimeError("Test labels differ across runs. Check test_loader shuffle.")

        m = logits_to_metrics(logits, labels)
        per_seed_metrics.append(m)
        print(f"Seed {seed} | Test acc {m['acc']:.4f} | Test macro-F1 {m['f1']:.4f}")

        if do_ensemble_logits:
            if sum_logits is None:
                sum_logits = logits.astype(np.float64)
            else:
                sum_logits += logits.astype(np.float64)

        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, f"full_seed{seed}_{cfg.output_name}")
        torch.save(model.state_dict(), save_path)
        print(f"Saved full model to {save_path}")

        clear_model(model, optimizer, scheduler)

    # 1) Report mean ± std trên test
    f1s = np.array([m["f1"] for m in per_seed_metrics], dtype=np.float64)
    accs = np.array([m["acc"] for m in per_seed_metrics], dtype=np.float64)

    print("\n===== FULL multi-seed TEST summary =====")
    print(f"Test macro-F1 mean {f1s.mean():.4f} std {f1s.std(ddof=0):.4f}")
    print(f"Test acc      mean {accs.mean():.4f} std {accs.std(ddof=0):.4f}")

    # 2) Ensemble logits
    if do_ensemble_logits and sum_logits is not None:
        ens_logits = (sum_logits / len(seeds)).astype(np.float32)
        ens_preds = ens_logits.argmax(axis=-1)

        if print_confusion_matrix:
            _print_confusion_matrix(
                fixed_labels.tolist(),
                ens_preds.tolist(),
                id2label=id2label,
                normalize=True,
            )

        print("\nClassification report (TEST, Ensemble logits):")
        target_names = [id2label[i] for i in range(len(id2label))]
        print(classification_report(fixed_labels, ens_preds, target_names=target_names, digits=4))


def main(args) -> None:
    cfg = build_train_config(args)

    # Resolve dataset paths
    if getattr(args, "locked_baseline", False):
        train_path = "dataset/atsa/laptop14/train.json"
        val_path = "dataset/atsa/laptop14/val.json"
        test_path = "dataset/atsa/laptop14/test.json"
    else:
        train_path = args.train_path
        val_path = args.val_path
        test_path = args.test_path

    set_all_seeds(cfg.seed)
    set_determinism(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if getattr(args, "benchmark_fusions", False):
        methods = _parse_str_list(getattr(args, "benchmark_methods", ""))
        seed_list = _parse_int_list(getattr(args, "seeds", ""))
        if not seed_list:
            n = int(getattr(args, "num_seeds", 3))
            seed_list = [cfg.seed + i for i in range(n)]

        run_phase1_benchmark(
            base_cfg=cfg,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            tokenizer=tokenizer,
            methods=methods,
            seeds=seed_list,
        )
        print(f"Phase 1 benchmark done. Results saved under: {os.path.join(cfg.output_dir, 'phase1')}")
        return

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

        train_loader = make_train_loader_with_seed(train_dataset_full, cfg.train_batch_size, cfg.seed)
        val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

        model = build_model(cfg=cfg, num_labels=len(label2id))
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
        )

        if out["best_state_dict"] is not None:
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
        return

    if not cfg.train_full_only:
        # Case 2: k fold CV
        print(f"Running StratifiedKFold with k={cfg.k_folds}")

        samples = train_dataset_full.samples
        y = [label2id[s["sentiment"]] for s in samples]

        skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)

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

            train_loader = make_train_loader_with_seed(train_ds, cfg.train_batch_size, cfg.seed + fold_idx)
            val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False)

            model = build_model(cfg=cfg, num_labels=len(label2id))
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
            )

            if out["best_state_dict"] is not None:
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

            print(
                f"Fold {fold_idx} | Best rolling Val F1 {out['best_val_f1_rolling']:.4f} | "
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

    seeds = [cfg.seed + i for i in range(cfg.k_folds)]  
    train_full_multi_seed_then_test(
        cfg=cfg,
        train_dataset_full=train_dataset_full,
        test_loader=test_loader,
        label2id=label2id,
        id2label=id2label,
        seeds=seeds,
        print_confusion_matrix=True,
        do_ensemble_logits=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)