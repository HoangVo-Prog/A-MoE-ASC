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
    Schedule A (stable warmup):
    - For the first `cfg.freeze_epochs` epochs: freeze the entire encoder (BERT backbone),
      so only modules outside `model.encoder` are trained.
    - After that: unfreeze the encoder and train everything.

    Exception:
    - If `cfg.mode == "MoEFFN"`, we keep the existing specialized logic
      (MoE FFN lives inside the encoder and is handled there).
    """
    fe = cfg.freeze_epochs
    if fe <= 0:
        _set_encoder_requires_grad(cfg, model, trainable=True, keep_moe_trainable=False)
        _set_encoder_train_eval(model, frozen=False)
        return False

    in_freeze = epoch_idx_0based < fe
    mode = cfg.mode

    if in_freeze:
        # Keep the user's existing MoEFFN logic untouched.
        if mode == "MoEFFN":
            print("MoEFFN mode: freezing base encoder, keeping MoE FFN trainable")
            keep_moe = cfg.freeze_moe
            _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=keep_moe)
            _set_encoder_train_eval(model, frozen=True)
            return True
        
        if mode == "MoESkConnectionModel":
            enc = getattr(model, "encoder", None)
            if enc is None:
                return False

            base = getattr(enc, "base_encoder", None)
            if base is None:
                # fallback: behave like general case
                print(f"{mode} mode: encoder has no base_encoder, freezing entire encoder")
                _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=True)
                _set_encoder_train_eval(model, frozen=True)
                return True

            print(f"{mode} mode: freezing base_encoder only, keeping MoE-skip trainable")

            # 1) Freeze backbone params
            for p in base.parameters():
                p.requires_grad = False

            # 2) Keep everything outside base_encoder trainable (moe_sk_h/moe_sk_2h, heads, router, experts)
            for name, p in enc.named_parameters():
                if name.startswith("base_encoder."):
                    continue
                p.requires_grad = True

        # General case: freeze the entire encoder (head-only warmup).
        print(f"{mode} mode: freezing entire encoder")
        _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=True)
        _set_encoder_train_eval(model, frozen=True)
        return True

    # Unfreeze
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
    use_amp: bool = False,  # bạn đang tắt AMP
    amp_dtype: str = "fp16",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = 1.0,
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

    step_print_i = int(step_print_moe) if step_print_moe is not None else 0

    def _get_router(model_: nn.Module):
        enc = getattr(model_, "encoder", None)
        moe_ffn = getattr(enc, "moe_ffn", None) if enc is not None else None
        router = getattr(moe_ffn, "router", None) if moe_ffn is not None else None
        return router

    # precompute router param ids for checking which optimizer group contains router
    router_param_ids = {id(p) for n, p in model.named_parameters() if "moe_ffn.router" in n}

    for step, batch in enumerate(dataloader):
        
        if step == 0:
            print("[DEBUG] loss_total.requires_grad:", loss_total.requires_grad)
            print("[DEBUG] logits.requires_grad:", logits.requires_grad)
            print("[DEBUG] loss_main.requires_grad:", (loss_main.requires_grad if loss_main is not None else None))
            n_trainable = sum(p.requires_grad for p in model.parameters())
            print("[DEBUG] num_trainable_params:", n_trainable)


        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

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

        # backward
        loss_total.backward()

        # clip (optional)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

        # optimizer step
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

        if step_print_i > step or (step_print_i and (step > 0) and (step % step_print_i == 0)):
            if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
                try:
                    model.print_moe_debug(topn=3)
                except Exception as e:
                    print("[ERROR] Can't print MoE debug logs:", e)
            
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

    # Phase fix: apply freeze for epoch 0 BEFORE building optimizer/scheduler
    freeze0 = maybe_freeze_encoder(cfg, model, epoch_idx_0based=0)
    warmup_ratio_phase1 = 0.0 if freeze0 else float(cfg.warmup_ratio)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=cfg.lr,
        lr_head=cfg.lr_head,
        warmup_ratio=warmup_ratio_phase1,
        total_steps=steps_per_epoch * max(1, int(cfg.epochs)),
        params=trainable_params(),
        adamw_foreach=cfg.adamw_foreach,
        adamw_fused=cfg.adamw_fused,
    )

    scaler = GradScaler() if cfg.use_amp else None

    for epoch in range(int(cfg.epochs)):
        print("=======================================================================")
        print(f"{tag}Epoch {epoch + 1}/{cfg.epochs}")

        # Apply freeze policy for this epoch
        if epoch == 0:
            freeze = freeze0
        else:
            freeze = maybe_freeze_encoder(cfg, model, epoch_idx_0based=epoch)

        if cfg.freeze_epochs > 0 and epoch < cfg.freeze_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{cfg.freeze_epochs})")

        # Rebuild optimizer exactly at unfreeze boundary
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs:
            print("Encoder unfrozen, rebuilding optimizer to include newly-trainable params")
            try:
                del optimizer
                del scheduler
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            remaining_steps = steps_per_epoch * max(1, int(cfg.epochs) - int(epoch))

            # Phase 2 uses the real warmup_ratio (or set to 0.0 if you want to avoid reset completely)
            warmup_ratio_phase2 = 0.0

            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=cfg.lr,
                lr_head=cfg.lr_head,
                warmup_ratio=warmup_ratio_phase2,
                total_steps=remaining_steps,
                params=trainable_params(),
                adamw_foreach=cfg.adamw_foreach,
                adamw_fused=cfg.adamw_fused,
            )

        # Training
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            fusion_method=method,
            f1_average="macro",
            step_print_moe=cfg.step_print_moe,
            use_amp=cfg.use_amp,
            amp_dtype=cfg.amp_dtype,
            scaler=scaler,
            max_grad_norm=cfg.max_grad_norm,
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
                f"\nTrain F1 {train_metrics['f1']:.4f} acc {train_metrics['acc']:.4f}"
            )
            log += "\n"
        else:
            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_f1"].append(float(train_metrics["f1"]))

            log = (
                f"Train loss {train_metrics['loss']:.4f} "
                f"F1 {train_metrics['f1']:.4f} "
                f"acc {train_metrics['acc']:.4f}"
            )
            log += "\n"

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
                print()
                print("*************************************************************")
                print("[MODEL] New best model on macro_f1 with neutral_f1 constraint")
                print()
            else:
                epochs_no_improve += 1
                if cfg.early_stop_patience > 0 and epochs_no_improve >= int(cfg.early_stop_patience):
                    print(f"Early stopping triggered after {cfg.early_stop_patience} epochs without improvement")
                    print(log)
                    break

        if test_loader is not None:
            print("Test Confusion Matrix")
            test_metrics = eval_model(
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
