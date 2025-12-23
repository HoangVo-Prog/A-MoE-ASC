from collections import deque
from typing import Dict, Optional, Any
from torch.amp import autocast, GradScaler

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
import numpy as np

from shared import (
    DEVICE, 
    build_optimizer_and_scheduler, 
    cleanup_cuda,
    _print_confusion_matrix
)

from moe_shared import eval_model, train_one_epoch


def set_encoder_trainable(
    model: nn.Module,
    trainable: bool,
    *,
    keep_moe_trainable: bool = True,
) -> None:
    """
    moe_head behavior:
      - During freeze phase (trainable=False), only embeddings are frozen.
      - If keep_moe_trainable=True, params whose name contains "moe_ffn" stay trainable.
    """
    for name, p in model.encoder.named_parameters():
        if keep_moe_trainable and "moe_ffn" in name:
            p.requires_grad = True
            continue

        if not trainable:
            if "embeddings" in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        else:
            p.requires_grad = True


def maybe_freeze_encoder(
    model: nn.Module,
    epoch_idx_0based: int,
    freeze_epochs: int,
    freeze_moe: bool = False,
) -> None:
    """
    For first `freeze_epochs` epochs, freeze only the embedding part of encoder.
    MoE head and classifier remain trainable.
    """
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        set_encoder_trainable(model, trainable=False, keep_moe_trainable=not freeze_moe)
    else:
        set_encoder_trainable(model, trainable=True, keep_moe_trainable=False)

    
def run_training_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    lr: float,
    warmup_ratio: float,
    epochs: int,
    fusion_method: str,
    freeze_epochs: int,
    freeze_moe: bool,
    rolling_k: int,
    early_stop_patience: int,
    id2label: Dict[int, str],
    tag: str = "",
    step_print_moe: float = 100,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
    max_grad_norm: Optional[float] = None,
    maybe_freeze_encoder_fn = None,
):
    history = {
        "train_total_loss": [],
        "train_main_loss": [],
        "train_lambda_loss": [],
        "train_f1": [],
        "train_acc": [],
        "val_loss": [],
        "val_f1": [],
    }

    val_f1_window = deque(maxlen=rolling_k)
    best_val_f1_rolling = -1.0
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0

    print("=======================================================================")
    print("Fusion Method:", fusion_method)
    print("=======================================================================")

    scaler = GradScaler() if (use_amp and amp_dtype == "fp16") else None
    steps_per_epoch = max(1, len(train_loader))

    def trainable_params():
        return [p for p in model.parameters() if p.requires_grad]

    # ===== PHASE INIT (epoch 0) =====
    
    model.set_epoch(0)
    
    maybe_freeze_encoder_fn(model, epoch_idx_0based=0, freeze_epochs=freeze_epochs, freeze_moe=freeze_moe)

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
        
        model.set_epoch(epoch)

        prev_trainable = any(p.requires_grad for p in model.encoder.parameters())

        # update freeze / unfreeze state
        maybe_freeze_encoder_fn(model, epoch_idx_0based=epoch, freeze_epochs=freeze_epochs, freeze_moe=freeze_moe)

        now_trainable = any(p.requires_grad for p in model.encoder.parameters())

        if freeze_epochs > 0 and epoch < freeze_epochs:
            print(f"Encoder base frozen (epoch {epoch + 1}/{freeze_epochs})")
        elif freeze_epochs > 0 and epoch == freeze_epochs:
            print("Encoder base unfrozen")

        # ===== rebuild optimizer EXACTLY ON TRANSITION =====
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

        # ===== TRAIN =====
        train_metrics = train_one_epoch(
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

        log = (
            f"Train main_loss {train_metrics['loss_main']:.6f} "
            f"aux_loss {train_metrics['aux_loss']:.6f} "
            f"lambda_loss {train_metrics['loss_lambda']:.6f} "
            f"total_loss {train_metrics['loss_total']:.6f} "
            f"\nF1 {train_metrics['f1']:.4f} acc {train_metrics['acc']:.4f}"
        )
        log += ("\n")

        # ===== VALIDATION =====
        if val_loader is not None:
            val_metrics = eval_model(
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                print_confusion_matrix=True,
                verbose_report=False,
                fusion_method=fusion_method,
                f1_average="macro",
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])

            val_f1_window.append(val_metrics["f1"])
            val_f1_rolling = float(np.mean(val_f1_window))

            log += (
                f"Val loss {val_metrics['loss']:.4f} "
                f"F1 {val_metrics['f1']:.4f} "
                f"acc {val_metrics['acc']:.4f} "
                f"| Val F1 rolling({rolling_k}) {val_f1_rolling:.4f}"
            )

            if val_f1_rolling > best_val_f1_rolling:
                best_val_f1_rolling = val_f1_rolling
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_no_improve = 0
                print("New best model on rolling val F1")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(
                        f"Early stopping triggered after "
                        f"{early_stop_patience} epochs without improvement"
                    )
                    print(log)
                    break

        print(log)

    del optimizer, scheduler, scaler
    cleanup_cuda()

    return {
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_f1_rolling": best_val_f1_rolling,
        "history": history,
    }
