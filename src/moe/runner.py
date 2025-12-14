import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from datasets import AspectSentimentDataset, AspectSentimentDatasetFromSamples
from model import BertConcatClassifier
from constants import DEVICE
from cli import parse_args
from config import TrainConfig, MoEConfig
from engine import eval_model, run_training_loop
from optim import build_optimizer_and_scheduler


def build_config(args):
    train_cfg = TrainConfig(
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
        verbose_report=bool(args.verbose_report),
        use_moe=bool(args.use_moe),
        freeze_base=bool(args.freeze_base),
        aux_loss_weight=float(args.aux_loss_weight),
        step_print_moe=float(args.step_print_moe),
        train_full_only=bool(args.train_full_only)
    )

    moe_cfg = None
    if args.use_moe:
        moe_cfg = MoEConfig(
            num_experts=args.moe_num_experts,
            top_k=args.moe_top_k,
            route_mask_pad_tokens=bool(args.route_mask_pad_tokens),
        )

    return train_cfg, moe_cfg


def build_model(*, cfg: TrainConfig, moe_cfg: MoEConfig, num_labels: int):
    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        use_moe=bool(cfg.use_moe),
        moe_cfg=moe_cfg,
        freeze_base=bool(cfg.freeze_base),
        aux_loss_weight=float(cfg.aux_loss_weight),
    ).to(DEVICE)


def train_full_then_test(
    *,
    cfg: TrainConfig,
    moe_cfg: MoEConfig,
    train_dataset_full,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    step_print_moe: float
):
    print("\n===== Train FULL then Test =====")

    train_loader = DataLoader(train_dataset_full, batch_size=cfg.train_batch_size, shuffle=True)

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

    print("\nClassification report (TEST):")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"final_{cfg.output_name}")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")


def main(args: argparse.Namespace) -> None:
    
    cfg, moe_cfg = build_config(args)
    print(cfg)
    print(moe_cfg)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset_full = AspectSentimentDataset(
        json_path=args.train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=None,
    )
    label2id = train_dataset_full.label2id
    id2label = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)

    test_dataset = AspectSentimentDataset(
        json_path=args.test_path,
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
            json_path=args.val_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            label2id=label2id,
        )

        train_loader = DataLoader(train_dataset_full, batch_size=cfg.train_batch_size, shuffle=True)
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
            step_print_moe=cfg.step_print_moe
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

            train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True)
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
                step_print_moe=cfg.step_print_moe
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

        print("\n===== CV Summary =====")
        print(f"Val macro-F1 mean {np.mean(fold_val_f1):.4f} std {np.std(fold_val_f1):.4f}")
        print(f"Test macro-F1 mean {np.mean(fold_test_f1):.4f} std {np.std(fold_test_f1):.4f}")

    train_full_then_test(
        cfg=cfg,
        moe_cfg=moe_cfg,
        train_dataset_full=train_dataset_full,
        test_loader=test_loader,
        label2id=label2id,
        id2label=id2label,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)