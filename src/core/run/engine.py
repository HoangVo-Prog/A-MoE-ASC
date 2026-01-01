from collections import deque
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from torch.amp import autocast, GradScaler

from src.core.utils.const import DEVICE
from src.core.utils.general import cleanup_cuda, safe_float
from src.core.utils.optim import build_optimizer_and_scheduler
from src.core.utils.plotting import print_confusion_matrix


def _set_encoder_requires_grad(
    cfg,
    model: nn.Module,
    *,
    trainable: bool,
    keep_moe_trainable: bool,
) -> None:
    if not hasattr(model, "encoder"):
        return

    for name, p in model.encoder.named_parameters():
        if keep_moe_trainable and ("moe_ffn" in name):
            p.requires_grad = True
        else:
            p.requires_grad = bool(trainable)


def _set_encoder_train_eval(model: nn.Module, *, frozen: bool) -> None:
    if not hasattr(model, "encoder"):
        return
    model.encoder.eval() if frozen else model.encoder.train()


def maybe_freeze_encoder(cfg, model: nn.Module, *, epoch_idx_0based: int) -> bool:
    """
    Apply freeze policy for current epoch.
    Returns: True if encoder is in frozen phase (including partial-freeze), else False.
    """
    fe = int(getattr(cfg.base, "freeze_epochs", 0) or 0)
    if fe <= 0:
        _set_encoder_requires_grad(cfg, model, trainable=True, keep_moe_trainable=False)
        _set_encoder_train_eval(model, frozen=False)
        return False

    in_freeze = epoch_idx_0based < fe
    mode = str(getattr(cfg.base, "mode", "")).strip()

    if in_freeze:
        if mode == "MoEFFN":
            print("MoEFFN mode: freezing base encoder, keeping MoE FFN trainable")
            keep_moe = not bool(getattr(cfg.base, "freeze_moe", False))
            _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=keep_moe)
            _set_encoder_train_eval(model, frozen=True)
            return True

        if mode == "MoEHead":
            print("MoEHead mode: freezing embeddings only, keeping MoE and classifier trainable")

            # We only touch model.encoder params here.
            # Classifier is outside encoder, so it remains trainable.
            if not hasattr(model, "encoder"):
                return False

            # 1) Start from "everything trainable" inside encoder
            for name, p in model.encoder.named_parameters():
                p.requires_grad = True

            # 2) Keep MoE trainable (in case some naming overlaps or future changes)
            #    and freeze embeddings only
            for name, p in model.encoder.named_parameters():
                # Keep MoE params trainable
                if "moe_ffn" in name:
                    p.requires_grad = True
                    continue

                # Freeze embeddings only (HF style: "...embeddings...")
                if "embeddings" in name:
                    p.requires_grad = False

            # Encoder should stay in train mode because most of it is still trainable
            _set_encoder_train_eval(model, frozen=False)
            return True

        print(f"{mode} mode: freezing entire encoder")
        _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=False)
        _set_encoder_train_eval(model, frozen=True)
        return True

    # unfreeze
    _set_encoder_requires_grad(cfg, model, trainable=True, keep_moe_trainable=False)
    _set_encoder_train_eval(model, frozen=False)
    return False


    # unfreeze
    
    _set_encoder_requires_grad(cfg, model, trainable=True, keep_moe_trainable=False)
    _set_encoder_train_eval(model, frozen=False)
    return False


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    step_print_moe: Optional[float] = 100,
    use_amp: bool = True,
    amp_dtype: str = "fp16",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train()

    moe = bool(getattr(model, "_collect_aux_loss", False))

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    aux_loss_sum = 0.0
    lambda_loss_sum = 0.0
    n_steps = 0

    all_preds: list[int] = []
    all_labels: list[int] = []

    amp_dtype_torch = (
        torch.float16 if (amp_dtype or "").lower().strip() == "fp16" else torch.bfloat16
    )
    step_print_i = int(step_print_moe) if step_print_moe is not None else 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            "cuda",
            enabled=bool(use_amp),
            dtype=amp_dtype_torch,
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

            # only meaningful for ver2-return path; safe even if missing
            loss_main = outputs.get("loss_main", None)
            loss_lambda = outputs.get("loss_lambda", None)
            loss_aux = outputs.get("aux_loss", None)

        # backward + step
        if use_amp:
            if scaler is None:
                raise RuntimeError("use_amp=True but scaler is None")
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

        # stats
        total_loss_sum += safe_float(loss_total)
        if moe:
            main_loss_sum += safe_float(loss_main)
            lambda_loss_sum += safe_float(loss_lambda)
            aux_loss_sum += safe_float(loss_aux)
        n_steps += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())

        if step_print_i and (step > 0) and (step % step_print_i == 0):
            if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
                try:
                    model.print_moe_debug(topn=3)
                except Exception:
                    pass

    denom = max(1, n_steps)
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average=f1_average))

    if moe:
        return {
            "loss_total": total_loss_sum / denom,
            "loss_main": main_loss_sum / denom,
            "loss_lambda": lambda_loss_sum / denom,
            "aux_loss": aux_loss_sum / denom,
            "acc": acc,
            "f1": f1,
        }

    # ver1 compatible output
    return {
        "loss": total_loss_sum / denom,
        "acc": acc,
        "f1": f1,
    }


def eval_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    id2label: Optional[Dict[int, str]] = None,
    verbose_report: bool = False,
    print_cf_matrix: bool = True,
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

    if print_cf_matrix:
        print_confusion_matrix(all_labels, all_preds, id2label=id2label, normalize=True)

    f1_per_class = f1_score(all_labels, all_preds, average=None)
    out: Dict[str, Any] = {"loss": avg_loss, "acc": acc, "f1": f1, "f1_per_class": f1_per_class}
    if return_confusion:
        out["confusion"] = cm  # raw counts [C, C]
    return out


def run_training_loop(
    cfg,
    model,
    method,
    train_loader,
    val_loader,
    test_loader,
    id2label,
    tag,
):     
    moe = bool(getattr(model, "_collect_aux_loss", False))
    
    if moe:
        history = {
            "train_total_loss": [],
            "train_main_loss": [],
            "train_lambda_loss": [],
            "train_f1": [],
            "train_acc": [],
            "val_loss": [],
            "val_f1": [],
        }
    else:
        history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}   
        
    best_macro_f1 = -1.0
    best_f1_neutral = -1.0
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0

    print("=======================================================================")
    print("Fusion Method:", method)
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

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=cfg.base.lr,
        warmup_ratio=cfg.base.warmup_ratio,
        total_steps=steps_per_epoch * max(1, int(cfg.base.epochs)),
        params=trainable_params(),
        adamw_foreach=cfg.base.adamw_foreach,
        adamw_fused=cfg.base.adamw_fused,
    )

    scaler = GradScaler() if cfg.base.use_amp else None

    for epoch in range(int(cfg.base.epochs)):
        print(f"{tag}Epoch {epoch + 1}/{cfg.base.epochs}")

        # --- Apply freeze policy for this epoch ---
        freeze = maybe_freeze_encoder(cfg, model, epoch_idx_0based=epoch)

        if cfg.base.freeze_epochs > 0 and epoch < cfg.base.freeze_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{cfg.base.freeze_epochs})")

        # --- Rebuild optimizer exactly at unfreeze boundary ---
        if cfg.base.freeze_epochs > 0 and epoch == cfg.base.freeze_epochs:
            print("Encoder unfrozen, rebuilding optimizer to include newly-trainable params")
            try:
                del optimizer
                del scheduler
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            remaining_steps = steps_per_epoch * max(1, int(cfg.base.epochs) - int(epoch))
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=cfg.base.lr,
                warmup_ratio=cfg.base.warmup_ratio,
                total_steps=remaining_steps,
                params=trainable_params(),
                adamw_foreach=cfg.base.adamw_foreach,
                adamw_fused=cfg.base.adamw_fused,
            )

        # --- training ---
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            fusion_method=method,
            f1_average="macro",
            step_print_moe=cfg.base.step_print_moe,
            use_amp=cfg.base.use_amp,
            amp_dtype=cfg.base.amp_dtype,
            scaler=scaler,
            max_grad_norm=cfg.base.max_grad_norm,
        )
        
        if moe:
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

        else: 
            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_f1"].append(float(train_metrics["f1"]))
            
            log = (
                f"Train loss {train_metrics['loss']:.4f} "
                f"F1 {train_metrics['f1']:.4f} "
                f"acc {train_metrics['acc']:.4f}"
            )
            log += ("\n")
            
        if val_loader is not None:
            print("Validation Confusion Matrix")
            val_metrics = eval_model(
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                print_cf_matrix=True,
                verbose_report=False,
                fusion_method=method,
                f1_average="macro",
            )
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_f1"].append(float(val_metrics["f1"]))

            macro_f1 = float(val_metrics["f1"])
            neutral_f1 = float(val_metrics["f1_per_class"][neutral_idx])

            log += (
                f"Val loss {val_metrics['loss']:.4f} "
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
                if cfg.base.early_stop_patience > 0 and epochs_no_improve >= int(cfg.base.early_stop_patience):
                    print(
                        f"Early stopping triggered after {cfg.base.early_stop_patience} epochs without improvement"
                    )
                    print(log)
                    break
                
        if test_loader is not None:
            print("Test Confusion Matrix")
            test_metrics =  eval_model(
                model=model,
                dataloader=test_loader,
                id2label=id2label,
                print_cf_matrix=True,
                verbose_report=False,
                fusion_method=method,
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
