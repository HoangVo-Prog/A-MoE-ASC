from collections import deque
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from torch.amp import autocast, GradScaler


from shared import DEVICE, cleanup_cuda, _print_confusion_matrix
from .optim import build_optimizer_and_scheduler


def set_encoder_trainable(
    model: nn.Module,
    trainable: bool,
    *,
    keep_moe_trainable: bool = False,
) -> None:
    """
    Unified behavior:

    - If keep_moe_trainable=False:
        Set all encoder params requires_grad = trainable (baseline behavior).
    - If keep_moe_trainable=True:
        Encoder base params follow `trainable`, but any param name containing "moe_ffn"
        stays trainable (MoE behavior).
    """
    if not hasattr(model, "encoder"):
        return

    for name, p in model.encoder.named_parameters():
        if keep_moe_trainable and ("moe_ffn" in name):
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
    Freeze encoder for first `freeze_epochs` epochs.

    - Baseline path:
        freeze_moe=False and model has no MoE, same as baseline behavior.
    - MoE path:
        If freeze_moe=False: freeze base encoder but keep moe_ffn trainable.
        If freeze_moe=True: freeze everything in encoder (including moe_ffn).
    """
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        set_encoder_trainable(
            model,
            trainable=False,
            keep_moe_trainable=not freeze_moe,
        )
    else:
        set_encoder_trainable(
            model,
            trainable=True,
            keep_moe_trainable=False,
        )


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    step_print_moe: float = 100,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    # AMP dtype mapping
    amp_dtype_torch = torch.float16 if (amp_dtype or "").lower().strip() == "fp16" else torch.bfloat16
    step_print_i = int(step_print_moe) if step_print_moe is not None else 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast("cuda", enabled=True, dtype=amp_dtype_torch):
                outputs = model(
                    input_ids_sent=batch["input_ids_sent"],
                    attention_mask_sent=batch["attention_mask_sent"],
                    input_ids_term=batch["input_ids_term"],
                    attention_mask_term=batch["attention_mask_term"],
                    labels=batch["label"],
                    fusion_method=fusion_method,
                )
                loss = outputs["loss"]
                logits = outputs["logits"]

            if scaler is None:
                raise RuntimeError("use_amp=True but scaler is None")

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=batch["label"],
                fusion_method=fusion_method,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())

        if step_print_i and (step > 0) and (step % step_print_i == 0):
            if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
                try:
                    model.print_moe_debug(topn=3)
                except Exception:
                    pass

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


def eval_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    id2label: Optional[Dict[int, str]] = None,
    verbose_report: bool = False,
    print_confusion_matrix: bool = True,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    return_confusion: bool = False,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=batch["label"],
                fusion_method=fusion_method,
            )

            loss = outputs.get("loss", None)
            logits = outputs["logits"]

            if loss is not None:
                total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    if verbose_report and id2label is not None:
        target_names = [id2label[i] for i in range(len(id2label))]
        print("Classification report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    num_labels = len(id2label) if id2label is not None else None
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_labels)) if num_labels is not None else None,
    )

    if print_confusion_matrix:
        _print_confusion_matrix(all_labels, all_preds, id2label=id2label, normalize=True)

    f1_per_class = f1_score(all_labels, all_preds, average=None)
    out: Dict[str, Any] = {"loss": avg_loss, "acc": acc, "f1": f1, "f1_per_class": f1_per_class}
    if return_confusion:
        out["confusion"] = cm  # raw counts [C, C]
    return out


def run_training_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader] = None,
    lr: float,
    warmup_ratio: float,
    epochs: int,
    fusion_method: str,
    freeze_epochs: int,
    rolling_k: int,
    early_stop_patience: int,
    id2label: Dict[int, str],
    tag: str = "",
    # MoE or AMP knobs
    freeze_moe: bool = False,
    step_print_moe: float = 100,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    adamw_foreach: bool = False,
    adamw_fused: bool = False,
    max_grad_norm: Optional[float] = 1.0,
    encoder_lr_scale=None,
    train_one_epoch_fn=train_one_epoch,
    eval_model_fn=eval_model,
    maybe_freeze_encoder_fn=maybe_freeze_encoder,
) -> Dict[str, Any]:
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    best_macro_f1 = -1.0
    best_f1_neutral = -1.0
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0

    print("=======================================================================")
    print("Fusion Method:", fusion_method)
    print("=======================================================================")

    steps_per_epoch = max(1, len(train_loader))
    
    neutral_idx = None
    for k, v in id2label.items():
        if str(v).lower().strip() == "neutral":
            neutral_idx = int(k)
            break
    if neutral_idx is None:
        raise RuntimeError("Cannot find 'neutral' in id2label")

    def trainable_params():
        return [p for p in model.parameters() if p.requires_grad]

    # Ensure correct freeze state before building phase-1 optimizer
    maybe_freeze_encoder_fn(model, epoch_idx_0based=0, freeze_epochs=freeze_epochs, freeze_moe=freeze_moe)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        warmup_ratio=warmup_ratio,
        total_steps=steps_per_epoch * max(1, epochs),
        params=trainable_params(),
        encoder_lr_scale=float(encoder_lr_scale),
        adamw_foreach=adamw_foreach,
        adamw_fused=adamw_fused,
    )

    scaler = GradScaler() if use_amp else None

    for epoch in range(int(epochs)):
        print(f"{tag}Epoch {epoch + 1}/{epochs}")

        prev_trainable = True
        if hasattr(model, "encoder"):
            prev_trainable = any(p.requires_grad for p in model.encoder.parameters())

        maybe_freeze_encoder_fn(model, epoch_idx_0based=epoch, freeze_epochs=freeze_epochs, freeze_moe=freeze_moe)

        now_trainable = True
        if hasattr(model, "encoder"):
            now_trainable = any(p.requires_grad for p in model.encoder.parameters())

        if freeze_epochs > 0 and epoch < freeze_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{freeze_epochs})")
        elif freeze_epochs > 0 and epoch == freeze_epochs:
            print("Encoder unfrozen")

        # Rebuild optimizer exactly when encoder becomes trainable
        if (not prev_trainable) and now_trainable:
            print("Rebuilding optimizer for unfrozen encoder params")
            try:
                del optimizer
                del scheduler
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            remaining_steps = steps_per_epoch * max(1, int(epochs) - int(epoch))
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=lr,
                warmup_ratio=warmup_ratio,
                total_steps=remaining_steps,
                params=None,
                encoder_lr_scale=float(encoder_lr_scale),
                adamw_foreach=adamw_foreach,
                adamw_fused=adamw_fused,
            )

        train_metrics = train_one_epoch_fn(
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
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_f1"].append(float(train_metrics["f1"]))

        log = (
            f"Train loss {train_metrics['loss']:.4f} "
            f"F1 {train_metrics['f1']:.4f} "
            f"acc {train_metrics['acc']:.4f}"
        )

        if val_loader is not None:
            print("Validation Confusion Matrix")
            val_metrics = eval_model_fn(
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                print_confusion_matrix=True,
                verbose_report=False,
                fusion_method=fusion_method,
                f1_average="macro",
            )
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_f1"].append(float(val_metrics["f1"]))

            macro_f1 = float(val_metrics["f1"])
            neutral_f1 = float(val_metrics["f1_per_class"][neutral_idx])

            log += (
                f" | Val loss {val_metrics['loss']:.4f} "
                f"F1 {val_metrics['f1']:.4f} "
                f"acc {val_metrics['acc']:.4f} "
                f"| Val neutral f1 {neutral_f1:.4f}"

            )

            should_save = (macro_f1 > best_macro_f1) and (neutral_f1 >= best_f1_neutral)
            if should_save:
                best_macro_f1 = macro_f1
                best_f1_neutral = neutral_f1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
                print("[MODEL] New best model on macro_f1 with neutral_f1 constraint")
            else:
                epochs_no_improve += 1
                if early_stop_patience > 0 and epochs_no_improve >= int(early_stop_patience):
                    print(
                        f"Early stopping triggered after {early_stop_patience} epochs without improvement"
                    )
                    print(log)
                    break
                
        if test_loader is not None:
            print("Test Confusion Matrix")
            test_metrics =  eval_model_fn(
                model=model,
                dataloader=test_loader,
                id2label=id2label,
                print_confusion_matrix=True,
                verbose_report=False,
                fusion_method=fusion_method,
                f1_average="macro",
            )
            log += (
                f"\nTest loss {test_metrics['loss']:.4f} "
                f"F1 {test_metrics['f1']:.4f} "
                f"acc {test_metrics['acc']:.4f} "
            )

        print(log)

    try:
        del optimizer
        del scheduler
        del scaler
    except Exception:
        pass

    cleanup_cuda()

    return {
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "history": history,
    }
