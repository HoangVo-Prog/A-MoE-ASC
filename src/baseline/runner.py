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
import random
import json
import hashlib
import os


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """Sample mean and sample std (ddof=1)."""
    arr = np.array(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


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
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _parse_str_list(csv: str) -> list[str]:
    s = (csv or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def run_phase1_benchmark_kfold_plus_full(
    *,
    base_cfg: TrainConfig,
    train_path: str,
    test_path: str,
    tokenizer,
    methods: list[str],
    seeds: list[int],
    output_path: str,
    do_ensemble_logits: bool = True,
) -> None:
    """Benchmark that always runs: K-fold CV + FULL multi-seed train then test (with optional ensemble).

    Saving policy (per user requirement):
    - DO NOT save model checkpoints
    - DO NOT save per-run result files
    - DO NOT call plt.show()
    - MUST output a single aggregate file containing all results
    """

    # Build datasets once.
    train_dataset_full = AspectSentimentDataset(
        json_path=train_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=None,
    )
    label2id = train_dataset_full.label2id
    id2label = {v: k for k, v in label2id.items()}

    test_dataset = AspectSentimentDataset(
        json_path=test_path,
        tokenizer=tokenizer,
        max_len_sent=base_cfg.max_len_sent,
        max_len_term=base_cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=base_cfg.eval_batch_size, shuffle=False)

    samples = train_dataset_full.samples
    y = [label2id[s["sentiment"]] for s in samples]

    k = int(base_cfg.k_folds)
    if k <= 1:
        k = 5

    all_results: dict = {
        "phase": 2,  # phase-agnostic benchmark; using 2 here since you now run Phase 2 methods too
        "benchmark_type": "kfold_plus_full_multiseed",
        "methods": methods,
        "seeds": seeds,
        "k_folds": k,
        "do_ensemble_logits": bool(do_ensemble_logits),
        "config": base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else base_cfg.__dict__,
        "runs": {},
        "summary": {},
    }

    per_method_seed_records: dict[str, list[dict]] = {m: [] for m in methods}

    for method in methods:
        # Config template for this method
        cfg_method = TrainConfig(**{**base_cfg.__dict__, "fusion_method": method, "k_folds": k})

        # ===== K-fold CV per seed =====
        for seed in seeds:
            cfg = TrainConfig(**{**cfg_method.__dict__, "seed": int(seed)})

            set_all_seeds(int(seed))
            set_determinism(int(seed))

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
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

                train_loader = make_train_loader_with_seed(train_ds, cfg.train_batch_size, int(seed) + fold_idx)
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
                    tag=f"[CV {method} seed={seed} fold={fold_idx}] ",
                )

                best_sd = out.get("best_state_dict", None)
                if best_sd is not None:
                    model.load_state_dict(best_sd)
                    model.to(DEVICE)

                val_m = eval_model(
                    model=model,
                    dataloader=val_loader,
                    id2label=id2label,
                    verbose_report=False,
                    print_confusion_matrix=False,
                    fusion_method=cfg.fusion_method,
                    f1_average="macro",
                )
                test_m = eval_model(
                    model=model,
                    dataloader=test_loader,
                    id2label=id2label,
                    verbose_report=False,
                    print_confusion_matrix=False,
                    fusion_method=cfg.fusion_method,
                    f1_average="macro",
                )

                fold_val_f1.append(float(val_m["f1"]))
                fold_test_f1.append(float(test_m["f1"]))

                clear_model(model, optimizer, scheduler)

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
            }
            per_method_seed_records[method].append(record)

        # ===== FULL multi-seed then test (and ensemble) =====
        full_out = train_full_multi_seed_then_test(
            cfg=cfg_method,
            train_dataset_full=train_dataset_full,
            test_loader=test_loader,
            label2id=label2id,
            id2label=id2label,
            seeds=[int(s) for s in seeds],
            print_confusion_matrix=False,
            do_ensemble_logits=bool(do_ensemble_logits),
            verbose_ensemble_report=False,
        )

        # Merge FULL per-seed metrics back into each record (same seed order)
        full_by_seed = {r["seed"]: r for r in full_out["per_seed"]}
        for rec in per_method_seed_records[method]:
            s = int(rec["seed"])
            rec["full_test_acc"] = float(full_by_seed[s]["acc"])
            rec["full_test_f1"] = float(full_by_seed[s]["f1"])

        # Attach method-level ensemble metrics
        ens = full_out.get("ensemble", None)
        if ens is not None:
            all_results.setdefault("ensemble", {})
            all_results["ensemble"][method] = {"full_ens_test_acc": float(ens["acc"]), "full_ens_test_f1": float(ens["f1"])}

    all_results["runs"] = per_method_seed_records

    # ===== Aggregate summary across seeds per method =====
    summary: dict[str, dict] = {}
    for method in methods:
        recs = per_method_seed_records[method]
        cv_val_means = [float(r["cv_val_f1_mean"]) for r in recs]
        cv_test_means = [float(r["cv_test_f1_mean"]) for r in recs]
        full_f1s = [float(r["full_test_f1"]) for r in recs]
        full_accs = [float(r["full_test_acc"]) for r in recs]

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

        if "ensemble" in all_results and method in all_results["ensemble"]:
            method_sum.update(all_results["ensemble"][method])

        summary[method] = method_sum

    # Deltas vs sent baseline (if present)
    if "sent" in summary:
        base = summary["sent"]["full_test_f1_mean_over_seeds"]
        for method in methods:
            summary[method]["delta_full_test_f1_vs_sent"] = float(
                summary[method]["full_test_f1_mean_over_seeds"] - base
            )
            if "full_ens_test_f1" in summary[method] and "full_ens_test_f1" in summary["sent"]:
                summary[method]["delta_full_ens_test_f1_vs_sent"] = float(summary[method]["full_ens_test_f1"] - summary["sent"]["full_ens_test_f1"])

    all_results["summary"] = summary

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

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
    verbose_ensemble_report: bool = True,
) -> dict:
    """
    Train FULL multiple times with different seeds, evaluate each run on test,
    and optionally compute an ensemble by averaging logits.

    Returns a dict:
      {
        "per_seed": [{"seed": int, "acc": float, "f1": float}, ...],
        "mean": {"acc": float, "acc_std": float, "f1": float, "f1_std": float},
        "ensemble": {"acc": float, "f1": float} or None
      }

    Notes:
    - No checkpoints are saved here.
    - If run_training_loop provides best_state_dict (even in no-val mode), we load it
      before collecting test logits to avoid using a suboptimal final epoch.
    """
    print("\n===== Train FULL (multi-seed) then Test =====")
    print(f"Seeds: {seeds}")

    per_seed_metrics: list[dict] = []
    sum_logits = None
    fixed_labels = None

    for i, seed in enumerate(seeds, start=1):
        print(f"\n----- FULL run {i}/{len(seeds)} | seed={seed} -----")
        set_all_seeds(seed)
        set_determinism(seed)

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

        # IMPORTANT: ensure we evaluate the "best" state if training loop provides it.
        best_sd = out.get("best_state_dict", None)
        if best_sd is not None:
            model.load_state_dict(best_sd)
            model.to(DEVICE)
            be = out.get("best_epoch", None)
            if be is not None:
                print(f"Loaded best FULL model at epoch {be} (seed={seed})")

        logits, labels = collect_test_logits(
            model=model, test_loader=test_loader, fusion_method=cfg.fusion_method
        )

        if fixed_labels is None:
            fixed_labels = labels
        else:
            if not np.array_equal(fixed_labels, labels):
                raise RuntimeError("Test labels differ across runs. Check test_loader shuffle.")

        m = logits_to_metrics(logits, labels)
        per_seed_metrics.append({"seed": int(seed), "acc": float(m["acc"]), "f1": float(m["f1"])})
        print(f"Seed {seed} | Test acc {m['acc']:.4f} | Test macro-F1 {m['f1']:.4f}")

        if do_ensemble_logits:
            if sum_logits is None:
                sum_logits = logits.astype(np.float64)
            else:
                sum_logits += logits.astype(np.float64)

        clear_model(model, optimizer, scheduler)

    # Mean ± std across seeds (sample std, ddof=1)
    f1s = np.array([m["f1"] for m in per_seed_metrics], dtype=np.float64)
    accs = np.array([m["acc"] for m in per_seed_metrics], dtype=np.float64)
    f1_mean = float(f1s.mean()) if f1s.size else float("nan")
    acc_mean = float(accs.mean()) if accs.size else float("nan")
    f1_std = float(f1s.std(ddof=1)) if f1s.size > 1 else 0.0
    acc_std = float(accs.std(ddof=1)) if accs.size > 1 else 0.0

    print("\n===== FULL multi-seed TEST summary =====")
    print(f"Test macro-F1 mean {f1_mean:.4f} std {f1_std:.4f}")
    print(f"Test acc      mean {acc_mean:.4f} std {acc_std:.4f}")

    ensemble_block = None
    if do_ensemble_logits and sum_logits is not None:
        ens_logits = (sum_logits / len(seeds)).astype(np.float32)
        ens_m = logits_to_metrics(ens_logits, fixed_labels)
        ensemble_block = {"acc": float(ens_m["acc"]), "f1": float(ens_m["f1"])}

        ens_preds = ens_logits.argmax(axis=-1)

        if print_confusion_matrix:
            _print_confusion_matrix(
                fixed_labels.tolist(),
                ens_preds.tolist(),
                id2label=id2label,
                normalize=True,
            )

        if verbose_ensemble_report:
            print("\nClassification report (TEST, Ensemble logits):")
            target_names = [id2label[i] for i in range(len(id2label))]
            print(classification_report(fixed_labels, ens_preds, target_names=target_names, digits=4))

    return {
        "per_seed": per_seed_metrics,
        "mean": {"acc": acc_mean, "acc_std": acc_std, "f1": f1_mean, "f1_std": f1_std},
        "ensemble": ensemble_block,
    }

def main(args) -> None:
    cfg = build_train_config(args)

    
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    set_all_seeds(cfg.seed)
    set_determinism(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Phase 1 benchmark mode: for each fusion method, run K-fold CV + FULL run on test.
    # Saving policy in benchmark:
    # - no checkpoints
    # - no per-run result files
    # - a single aggregate JSON file containing all results
    if getattr(args, "benchmark_fusions", False):
        methods = _parse_str_list(getattr(args, "benchmark_methods", ""))
        if not methods:
            methods = ["sent", "term", "concat", "add", "mul", "cross", "gated_concat", "bilinear", "coattn", "late_interaction", "moe"]

        seeds = _parse_int_list(getattr(args, "seeds", ""))
        if not seeds:
            n = int(getattr(args, "num_seeds", 3))
            seeds = [int(cfg.seed) + i for i in range(n)]

        out_path = os.path.join(cfg.output_dir, "phase1_benchmark_all.json")
        run_phase1_benchmark_kfold_plus_full(
            base_cfg=cfg,
            train_path=train_path,
            test_path=test_path,
            tokenizer=tokenizer,
            methods=methods,
            seeds=seeds,
            output_path=out_path,
        )
        print(f"Benchmark complete. Aggregate results written to: {out_path}")
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