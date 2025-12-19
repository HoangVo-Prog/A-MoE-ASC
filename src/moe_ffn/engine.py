from collections import deque
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from constants import DEVICE
import numpy as np

from optim import build_optimizer_and_scheduler

def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def maybe_freeze_encoder(model: nn.Module, epoch_idx_0based: int, freeze_epochs: int) -> None:
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        set_encoder_trainable(model, False)
    else:
        set_encoder_trainable(model, True)


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    step_print_moe: float = 100,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for step, batch in enumerate(dataloader):
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

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())
        
        if step > 0 and step % step_print_moe == 0:
            model.print_moe_debug(topn=3)

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


def _print_confusion_matrix(
    y_true,
    y_pred,
    *,
    id2label: Optional[Dict[int, str]] = None,
    normalize: bool = True,
    digits: int = 3,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)

    if id2label is not None:
        labels = [id2label[i] for i in range(len(id2label))]
    else:
        labels = [str(i) for i in range(cm.shape[0])]

    max_label_len = max(len(l) for l in labels)

    header = " " * (max_label_len + 2)
    for lbl in labels:
        header += f"{lbl:>{max_label_len+2}}"
    print(header)

    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>{max_label_len}} |"
        for val in row:
            if normalize:
                row_str += f"{val:>{max_label_len+2}.{digits}f}"
            else:
                row_str += f"{int(val):>{max_label_len+2}d}"
        print(row_str)

    print()


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
    optimizer,
    scheduler,
    lr: float = None,
    warmup_ratio: float = None,
    epochs: int,
    fusion_method: str,
    freeze_epochs: int,
    rolling_k: int,
    early_stop_patience: int,
    id2label: Dict[int, str],
    tag: str = "",
    step_print_moe: float = 100,
):
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}


    def _trainable_params(m: nn.Module):
        return [p for p in m.parameters() if p.requires_grad]

    # Build phase-specific optimizer and scheduler to avoid allocating AdamW
    # state for frozen parameters. When the encoder becomes trainable, the
    # optimizer is rebuilt with the newly trainable parameters.
    steps_per_epoch = max(1, len(train_loader))

    # Ensure encoder trainable flags are correct before building the initial optimizer.
    maybe_freeze_encoder(model, epoch_idx_0based=0, freeze_epochs=freeze_epochs)

    if optimizer is None or scheduler is None:
        if lr is None or warmup_ratio is None:
            raise ValueError("lr and warmup_ratio must be provided when optimizer/scheduler are None.")

        if freeze_epochs > 0:
            phase1_epochs = min(freeze_epochs, epochs)
            phase1_steps = steps_per_epoch * max(1, phase1_epochs)
        else:
            phase1_steps = steps_per_epoch * max(1, epochs)

        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            lr=lr,
            warmup_ratio=warmup_ratio,
            total_steps=phase1_steps,
            params=_trainable_params(model),
        )

    val_f1_window = deque(maxlen=rolling_k)
    best_val_f1_rolling = -1.0
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0
    
    print("=======================================================================")
    print("Fusion Method:", fusion_method)
    print("=======================================================================")
    
    for epoch in range(epochs):
        print(f"{tag}Epoch {epoch + 1}/{epochs}")

        maybe_freeze_encoder(model, epoch_idx_0based=epoch, freeze_epochs=freeze_epochs)

        if freeze_epochs > 0 and epoch < freeze_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{freeze_epochs})")

        elif freeze_epochs > 0 and epoch == freeze_epochs:
            print("Encoder unfrozen")

            # Rebuild optimizer and scheduler for the remaining epochs now that
            # the encoder parameters are trainable.
            remaining_steps = steps_per_epoch * max(1, epochs - epoch)
            try:
                del optimizer
                del scheduler
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                lr=lr,
                warmup_ratio=warmup_ratio,
                total_steps=remaining_steps,
                params=_trainable_params(model),
            )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            fusion_method=fusion_method,
            f1_average="macro",
            step_print_moe=step_print_moe,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])

        log = (
            f"Train loss {train_metrics['loss']:.4f} "
            f"F1 {train_metrics['f1']:.4f} "
            f"acc {train_metrics['acc']:.4f}"
        )

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
            val_f1_rolling = sum(val_f1_window) / len(val_f1_window)

            log += (
                f" | Val loss {val_metrics['loss']:.4f} "
                f"F1 {val_metrics['f1']:.4f} "
                f"acc {val_metrics['acc']:.4f} "
                f"| Val F1 rolling({rolling_k}) {val_f1_rolling:.4f}"
            )

            if val_f1_rolling > best_val_f1_rolling:
                best_val_f1_rolling = val_f1_rolling
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_no_improve = 0
                print("New best model on rolling val F1")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping triggered after {early_stop_patience} epochs without improvement")
                    break

        print(log)

    return {
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_f1_rolling": best_val_f1_rolling,
        "history": history,
    }


def logits_to_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"acc": float(acc), "f1": float(f1)}


@torch.no_grad()
def collect_test_logits(
    *,
    model: torch.nn.Module,
    test_loader: DataLoader,
    fusion_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_chunks = []
    labels_chunks = []

    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(
            input_ids_sent=batch["input_ids_sent"],
            attention_mask_sent=batch["attention_mask_sent"],
            input_ids_term=batch["input_ids_term"],
            attention_mask_term=batch["attention_mask_term"],
            labels=None,
            fusion_method=fusion_method,
        )
        logits = outputs["logits"].detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()

        logits_chunks.append(logits)
        labels_chunks.append(labels)

    logits_all = np.concatenate(logits_chunks, axis=0)
    labels_all = np.concatenate(labels_chunks, axis=0)
    return logits_all, labels_all
    