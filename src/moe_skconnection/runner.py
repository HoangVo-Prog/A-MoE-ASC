from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from shared import (
    AspectSentimentDataset,
    set_all_seeds,
    set_determinism,
    make_train_loader_with_seed,
    cleanup_cuda,
)

from moe_shared import (
    TrainConfig,
    build_train_config,
    locked_baseline_config,
)

from moe_head.config import build_multi_moe_config as build_moe_config
from moe_head.engine import run_training_loop as run_training_loop_fn
from moe_head.engine import eval_model as eval_model_fn

from moe_skconnection.cli import parse_args
from moe_skconnection.config import apply_skconnection_config
from moe_skconnection.model import build_model


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def main(args) -> None:
    cfg: TrainConfig = build_train_config(args)
    cfg = apply_skconnection_config(cfg, args)

    if bool(getattr(args, "locked_baseline", False)):
        cfg = locked_baseline_config(cfg)

    seed = int(getattr(cfg, "seed", 42))
    set_all_seeds(seed)
    set_determinism(bool(getattr(cfg, "deterministic", True)))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset_full = AspectSentimentDataset(
        json_path=cfg.train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=None,
    )
    label2id: Dict[str, int] = train_dataset_full.label2id
    id2label: Dict[int, str] = {v: k for k, v in label2id.items()}

    val_loader: Optional[DataLoader] = None
    if getattr(cfg, "val_path", None):
        val_dataset = AspectSentimentDataset(
            json_path=cfg.val_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            label2id=label2id,
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    test_dataset = AspectSentimentDataset(
        json_path=cfg.test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    moe_cfg = build_moe_config(args)

    model = build_model(cfg=cfg, moe_cfg=moe_cfg, num_labels=len(label2id))

    train_loader = make_train_loader_with_seed(train_dataset_full, cfg.train_batch_size, seed)

    out = run_training_loop_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        epochs=cfg.epochs,
        fusion_method=cfg.fusion_method,
        freeze_epochs=cfg.freeze_epochs,
        freeze_moe=cfg.freeze_moe,
        rolling_k=cfg.rolling_k,
        early_stop_patience=cfg.early_stop_patience,
        id2label=id2label,
        tag="",
        step_print_moe=float(getattr(cfg, "step_print_moe", 100)),
        use_amp=bool(getattr(cfg, "use_amp", True)),
        amp_dtype=str(getattr(cfg, "amp_dtype", "fp16")),
        adamw_foreach=bool(getattr(cfg, "adamw_foreach", False)),
        adamw_fused=bool(getattr(cfg, "adamw_fused", False)),
        max_grad_norm=getattr(cfg, "max_grad_norm", None),
    )

    best_state = out.get("best_state_dict", None)
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test_metrics = eval_model_fn(
        model=model,
        dataloader=test_loader,
        id2label=id2label,
        verbose_report=bool(getattr(cfg, "verbose_report", False)),
        print_confusion_matrix=True,
        fusion_method=cfg.fusion_method,
        f1_average="macro",
        return_confusion=True,
    )
    print("Test metrics:", {k: v for k, v in test_metrics.items() if k != "confusion"})

    _ensure_dir(cfg.output_dir)
    if cfg.output_dir:
        model_path = os.path.join(cfg.output_dir, "best_model.pt")
        torch.save(model.state_dict(), model_path)

        summary = {
            "best_epoch": out.get("best_epoch", None),
            "best_val_f1_rolling": out.get("best_val_f1_rolling", None),
            "history": out.get("history", None),
            "test": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in test_metrics.items()},
        }
        out_json = os.path.join(cfg.output_dir, "run_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {out_json}")
        print(f"Saved model to {model_path}")

    cleanup_cuda()


if __name__ == "__main__":
    args = parse_args()
    main(args)
