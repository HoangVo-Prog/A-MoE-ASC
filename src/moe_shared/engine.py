from collections import deque
from typing import Dict, Optional, Any
from torch.amp import autocast, GradScaler

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import numpy as np

from shared import (
    DEVICE,
    build_optimizer_and_scheduler,
    cleanup_cuda,
)


def set_encoder_trainable(
    model: nn.Module,
    trainable: bool,
    *,
    keep_moe_trainable: bool = True,
) -> None:
    """
    Set requires_grad for encoder params.

    If keep_moe_trainable=True:
      - encoder base params follow trainable
      - MoE params whose name contains 'moe_ffn' are always trainable
    """
    for name, p in model.encoder.named_parameters():
        if keep_moe_trainable and "moe_ffn" in name:
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def maybe_freeze_encoder(
    model: nn.Module,
    epoch_idx_0based: int,
    freeze_epochs: int,
    freeze_moe: bool = False,
) -> None:
    """
    Freeze base encoder for first freeze_epochs epochs, but keep MoE trainable unless freeze_moe=True.
    """
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        set_encoder_trainable(model, trainable=False, keep_moe_trainable=not freeze_moe)
    else:
        set_encoder_trainable(model, trainable=True, keep_moe_trainable=False)


def _safe_float(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float("nan")


def train_one_epoch_train_only(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    step_print_moe: float = 100,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train()

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    lambda_loss_sum = 0.0
    n_steps = 0

    all_preds = []
    all_labels = []

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            "cuda",
            enabled=use_amp,
            dtype=torch.float16 if amp_dtype == "fp16" else torch.bfloat16,
        ):
            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=batch["label"],
                fusion_method=fusion_method,
            )

            loss_total = outputs.get("loss", None)
            logits = outputs["logits"]

            # Optional keys from model
            loss_main = outputs.get("loss_main", None)
            loss_lambda = outputs.get("loss_lambda", None)

        if loss_total is None:
            raise RuntimeError("Model returned loss=None during training.")

        if use_amp and scaler is not None:
            scaler.scale(loss_total).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss_sum += _safe_float(loss_total)
        main_loss_sum += _safe_float(loss_main)
        lambda_loss_sum += _safe_float(loss_lambda)
        n_steps += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())

        if step_print_moe is not None and int(step_print_moe) > 0:
            if step > 0 and step % int(step_print_moe) == 0:
                # Keep existing MoE stats print
                if hasattr(model, "print_moe_debug"):
                    model.print_moe_debug(topn=3)

                # Step-level loss print (train only)
                lm = _safe_float(loss_main)
                ll = _safe_float(loss_lambda)
                lt = _safe_float(loss_total)
                print(
                    f"[Train step {step}] main_loss={lm:.6f} "
                    f"lambda_loss={ll:.6f} total_loss={lt:.6f}"
                )

    denom = max(1, n_steps)
    avg_total = total_loss_sum / denom
    avg_main = main_loss_sum / denom
    avg_lambda = lambda_loss_sum / denom

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    return {
        "loss_total": avg_total,
        "loss_main": avg_main,
        "loss_lambda": avg_lambda,
        "acc": float(acc),
        "f1": float(f1),
    }


def run_training_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    warmup_ratio: float,
    epochs: int,
    fusion_method: str,
    freeze_epochs: int,
    freeze_moe: bool,
    tag: str = "",
    step_print_moe: float = 100,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, Any]:
    history = {
        "train_total_loss": [],
        "train_main_loss": [],
        "train_lambda_loss": [],
        "train_f1": [],
        "train_acc": [],
    }

    print("=======================================================================")
    print("Fusion Method:", fusion_method)
    print("Train only: printing main_loss, lambda_loss, total_loss")
    print("=======================================================================")

    scaler = GradScaler() if (use_amp and amp_dtype == "fp16") else None
    steps_per_epoch = max(1, len(train_loader))

    def trainable_params():
        return [p for p in model.parameters() if p.requires_grad]

    # ===== PHASE INIT (epoch 0) =====
    maybe_freeze_encoder(model, epoch_idx_0based=0, freeze_epochs=freeze_epochs, freeze_moe=freeze_moe)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        warmup_ratio=warmup_ratio,
        total_steps=steps_per_epoch * epochs,
        params=trainable_params(),
        adamw_foreach=adamw_foreach,
        adamw_fused=adamw_fused,
    )

    for epoch in range(epochs):
        print(f"{tag}Epoch {epoch + 1}/{epochs}")

        prev_trainable = any(p.requires_grad for p in model.encoder.parameters())

        maybe_freeze_encoder(model, epoch_idx_0based=epoch, freeze_epochs=freeze_epochs, freeze_moe=freeze_moe)

        now_trainable = any(p.requires_grad for p in model.encoder.parameters())

        if freeze_epochs > 0 and epoch < freeze_epochs:
            print(f"Encoder base frozen (epoch {epoch + 1}/{freeze_epochs})")
        elif freeze_epochs > 0 and epoch == freeze_epochs:
            print("Encoder base unfrozen")

        if (not prev_trainable) and now_trainable:
            print("Rebuilding optimizer for unfrozen encoder")

            try:
                del optimizer
                del scheduler
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            remaining_steps = steps_per_epoch * max(1, epochs - epoch)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=lr,
                warmup_ratio=warmup_ratio,
                total_steps=remaining_steps,
                params=trainable_params(),
                adamw_foreach=adamw_foreach,
                adamw_fused=adamw_fused,
            )

        train_metrics = train_one_epoch_train_only(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            fusion_method=fusion_method,
            f1_average="macro",
            step_print_moe=step_print_moe,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
        )

        history["train_total_loss"].append(train_metrics["loss_total"])
        history["train_main_loss"].append(train_metrics["loss_main"])
        history["train_lambda_loss"].append(train_metrics["loss_lambda"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_acc"].append(train_metrics["acc"])

        print(
            f"Train main_loss {train_metrics['loss_main']:.6f} "
            f"lambda_loss {train_metrics['loss_lambda']:.6f} "
            f"total_loss {train_metrics['loss_total']:.6f} "
            f"F1 {train_metrics['f1']:.4f} acc {train_metrics['acc']:.4f}"
        )

    del optimizer, scheduler, scaler
    cleanup_cuda()

    return {"history": history}
