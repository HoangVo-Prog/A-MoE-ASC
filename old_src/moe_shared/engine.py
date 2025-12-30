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
    _print_confusion_matrix,
    _safe_float
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
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    
    total_loss_sum = 0.0
    main_loss_sum = 0.0
    aux_loss_sum = 0.0
    lambda_loss_sum = 0.0
    n_steps = 0
    
    total_loss = 0.0
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
            loss_total = outputs["loss"]
            logits = outputs["logits"]
            
            loss_main = outputs.get("loss_main", None)
            loss_lambda = outputs.get("loss_lambda", None)
            loss_aux = outputs.get("aux_loss", None)
            
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
        aux_loss_sum += _safe_float(loss_aux)
        n_steps += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())

        if step_print_moe is not None and int(step_print_moe) > 0:
            if step > 0 and step % int(step_print_moe) == 0:
                # Keep existing MoE stats print
                if hasattr(model, "print_moe_debug"):
                    model.print_moe_debug(topn=3)

    denom = max(1, n_steps)
    avg_total = total_loss_sum / denom
    avg_main = main_loss_sum / denom
    avg_lambda = lambda_loss_sum / denom
    avg_aux = aux_loss_sum / denom

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    return {
        "loss_total": avg_total,
        "loss_main": avg_main,
        "loss_lambda": avg_lambda,
        "aux_loss": avg_aux,
        "acc": float(acc),
        "f1": float(f1),
    }


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
    all_preds = []
    all_labels = []

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

            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += float(loss.item()) if loss is not None else 0.0
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    if verbose_report and id2label is not None:
        target_names = [id2label[i] for i in range(len(id2label))]
        print("Classification report:")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=target_names,
                digits=4,
            )
        )
        
    num_labels = len(id2label) if id2label is not None else None
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_labels)) if num_labels is not None else None,
    )

    if print_confusion_matrix:
        _print_confusion_matrix(
            all_labels,
            all_preds,
            id2label=id2label,
            normalize=True,
        )

    out = {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
    }

    if return_confusion:
        out["confusion"] = cm  # shape [C, C], raw counts

    return out
    
    
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
