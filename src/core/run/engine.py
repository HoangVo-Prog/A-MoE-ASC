from collections import deque
import re
import string
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


def _normalize_aspect_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("-", " ")
    s = s.strip(string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    
    if cfg.mode == "SDModel":
        _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=False)
        _set_encoder_train_eval(model, frozen=True)
        return True  

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
                print(f"{mode} mode: encoder has no base_encoder, freezing entire encoder")
                _set_encoder_requires_grad(cfg, model, trainable=False, keep_moe_trainable=True)
                _set_encoder_train_eval(model, frozen=True)
                return True

            print(f"{mode} mode: freezing base_encoder only, keeping MoE-skip + fusion modules trainable")

            # 1) Freeze backbone params
            for p in base.parameters():
                p.requires_grad = False
            base.eval()

            # 2) Keep MoE modules trainable
            for name, p in enc.named_parameters():
                if name.startswith("base_encoder."):
                    continue
                p.requires_grad = True
            
            # 3) Keep fusion modules trainable (nằm ngoài encoder)
            # Các modules này đã tự động trainable vì không thuộc base_encoder
            # Nhưng cần đảm bảo chúng ở train mode
            for name, module in model.named_modules():
                if name.startswith("encoder.base_encoder"):
                    continue
                if hasattr(module, 'train') and callable(module.train):
                    if name and not name.startswith("encoder.base_encoder"):
                        module.train()
            
            return True
              
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
    cfg=None,
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
    epoch_idx: Optional[int] = None,
) -> Dict[str, float]:
    model.train()

    moe = bool(getattr(model, "_collect_aux_loss", False))
    hag_mode = cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE"
    # Debug in main process to avoid num_workers dataset isolation.
    debug_aspect_span = bool(getattr(cfg, "debug_aspect_span", False))

    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "reset_match_stats"):
        dataset.reset_match_stats()

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

    hag_log_sums = {
        "loss": 0.0,
        "loss_main": 0.0,
        "aux_loss": 0.0,
        "loss_group": 0.0,
        "loss_balance": 0.0,
        "loss_diversity": 0.0,
    }
    hag_log_counts = {k: 0 for k in hag_log_sums}

    match_total = 0
    match_matched = 0
    match_mask_sum = 0.0
    match_zero = 0
    token_mismatch_count = 0
    truncated_count = 0
    not_found_raw_count = 0
    unknown_count = 0

    def _move_batch_to_device(batch_dict):
        out = {}
        for k, v in batch_dict.items():
            if torch.is_tensor(v):
                out[k] = v.to(DEVICE)
            else:
                out[k] = v
        return out

    for step, batch in enumerate(dataloader):
        batch = _move_batch_to_device(batch)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            "cuda",
            enabled=bool(use_amp),
            dtype=amp_dtype_torch,
        ):
            if cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE":
                outputs = model(
                    input_ids_sent=batch["input_ids_sent"],
                    attention_mask_sent=batch["attention_mask_sent"],
                    input_ids_term=batch["input_ids_term"],
                    attention_mask_term=batch["attention_mask_term"],
                    aspect_start=batch.get("aspect_start"),
                    aspect_end=batch.get("aspect_end"),
                    aspect_mask_sent=batch.get("aspect_mask_sent"),
                    labels=batch["label"],
                    fusion_method=fusion_method,
                )
            else:
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

        if hag_mode:
            if step == 0:
                print(f"[HAGMoE] output keys: {sorted(list(outputs.keys()))}")
                effective = getattr(model, "_last_fusion_method", None)
                model_cfg_fusion = str(getattr(cfg, "hag_fusion_method", "")).strip()
                model_attr_fusion = str(getattr(model, "hag_fusion_method", "")).strip()
                print(
                    "[HAGMoE] "
                    f"benchmark_method={fusion_method} "
                    f"cfg.hag_fusion_method={model_cfg_fusion or '""'} "
                    f"model.hag_fusion_method={model_attr_fusion or '""'} "
                    f"effective_fusion={effective}"
                )
            for key in hag_log_sums:
                if key in outputs and outputs.get(key) is not None:
                    hag_log_sums[key] += safe_float(outputs.get(key))
                    hag_log_counts[key] += 1

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

        if debug_aspect_span and "aspect_mask_sent" in batch:
            mask_sum = batch["aspect_mask_sent"].detach().sum(dim=1)
            matched = mask_sum > 0
            match_total += int(mask_sum.numel())
            match_matched += int(matched.sum().item())
            match_mask_sum += float(mask_sum[matched].sum().item())
            match_zero += int((~matched).sum().item())

            if "sentence_raw" in batch and "aspect_raw" in batch and "valid_len" in batch and "sep_idx" in batch:
                sentence_raw = batch["sentence_raw"]
                aspect_raw = batch["aspect_raw"]
                valid_len = batch["valid_len"].detach().cpu().tolist()
                sep_idx = batch["sep_idx"].detach().cpu().tolist()
                max_len_sent = int(getattr(cfg, "max_len_sent", 0) or 0)
                for i in range(len(mask_sum)):
                    if matched[i]:
                        continue
                    try:
                        sent_norm = _normalize_aspect_text(sentence_raw[i])
                        asp_norm = _normalize_aspect_text(aspect_raw[i])
                    except Exception:
                        unknown_count += 1
                        continue
                    raw_found = asp_norm != "" and asp_norm in sent_norm
                    truncated = (valid_len[i] >= max_len_sent) or (sep_idx[i] >= max_len_sent - 1)
                    if raw_found and truncated:
                        truncated_count += 1
                    elif raw_found:
                        token_mismatch_count += 1
                    else:
                        not_found_raw_count += 1
            else:
                unknown_count += int((~matched).sum().item())

        if (not hag_mode) and step_print_i and (step > 0) and (step % step_print_i == 0):
            if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
                try:
                    model.print_moe_debug(topn=3)
                except Exception as e:
                    print("Cannot print_moe_debug:", e)

    denom = max(1, n_steps)
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average=f1_average))

    if hag_mode:
        parts = ["epoch_summary"]
        for key, total in hag_log_sums.items():
            cnt = max(1, hag_log_counts[key])
            if hag_log_counts[key] > 0:
                parts.append(f"{key}={total / cnt:.6f}")
        print("[HAGMoE] " + " ".join(parts))
        if hasattr(model, "print_moe_debug") and callable(getattr(model, "print_moe_debug")):
            try:
                model.print_moe_debug(topn=3)
            except Exception as e:
                print("Cannot print_moe_debug:", e)

    if debug_aspect_span:
        print(
            f"[AspectSpanDiag] split=train total={match_total} matched={match_matched} "
            f"match_rate={(match_matched / max(1, match_total)) * 100:.2f}% "
            f"token_mismatch={token_mismatch_count} truncated={truncated_count} "
            f"not_found_raw={not_found_raw_count} unknown={unknown_count} "
            f"avg_mask_sum={(match_mask_sum / max(1, match_matched)):.2f}"
        )

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
    cfg=None,
    model: nn.Module,
    dataloader: DataLoader,
    id2label: Optional[Dict[int, str]] = None,
    verbose_report: bool = False,
    print_cf_matrix: bool = True,
    fusion_method: str = "concat",
    f1_average: str = "macro",
    return_confusion: bool = False,
    epoch_idx: Optional[int] = None,
    split: str = "eval",
    debug_aspect_span: bool = False,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "reset_match_stats"):
        dataset.reset_match_stats()

    match_total = 0
    match_matched = 0
    match_mask_sum = 0.0

    debug_aspect_span = bool(debug_aspect_span)

    match_total = 0
    match_matched = 0
    match_mask_sum = 0.0
    match_zero = 0
    token_mismatch_count = 0
    truncated_count = 0
    not_found_raw_count = 0
    unknown_count = 0

    with torch.no_grad():
        def _move_batch_to_device(batch_dict):
            out = {}
            for k, v in batch_dict.items():
                if torch.is_tensor(v):
                    out[k] = v.to(DEVICE)
                else:
                    out[k] = v
            return out

        for batch_idx, batch in enumerate(dataloader):
            batch = _move_batch_to_device(batch)

            hag_mode = cfg is not None and str(getattr(cfg, "mode", "")).strip() == "HAGMoE"
            if hag_mode:
                outputs = model(
                    input_ids_sent=batch["input_ids_sent"],
                    attention_mask_sent=batch["attention_mask_sent"],
                    input_ids_term=batch["input_ids_term"],
                    attention_mask_term=batch["attention_mask_term"],
                    aspect_start=batch.get("aspect_start"),
                    aspect_end=batch.get("aspect_end"),
                    aspect_mask_sent=batch.get("aspect_mask_sent"),
                    labels=batch["label"],
                    fusion_method=fusion_method,
                )
            else:
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

            if hag_mode and batch_idx == 0:
                effective = getattr(model, "_last_fusion_method", None)
                model_cfg_fusion = str(getattr(cfg, "hag_fusion_method", "")).strip()
                model_attr_fusion = str(getattr(model, "hag_fusion_method", "")).strip()
                print(
                    "[HAGMoE] "
                    f"benchmark_method={fusion_method} "
                    f"cfg.hag_fusion_method={model_cfg_fusion or '""'} "
                    f"model.hag_fusion_method={model_attr_fusion or '""'} "
                    f"effective_fusion={effective}"
                )

            if loss is not None:
                total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

            if "aspect_mask_sent" in batch:
                mask_sum = batch["aspect_mask_sent"].detach().sum(dim=1)
                matched = mask_sum > 0
                match_total += int(mask_sum.numel())
                match_matched += int(matched.sum().item())
                match_mask_sum += float(mask_sum[matched].sum().item())
                match_zero += int((~matched).sum().item())

                if "sentence_raw" in batch and "aspect_raw" in batch and "valid_len" in batch and "sep_idx" in batch:
                    sentence_raw = batch["sentence_raw"]
                    aspect_raw = batch["aspect_raw"]
                    valid_len = batch["valid_len"].detach().cpu().tolist()
                    sep_idx = batch["sep_idx"].detach().cpu().tolist()
                    max_len_sent = int(
                        batch["max_len_sent"][0].item()
                        if "max_len_sent" in batch and torch.is_tensor(batch["max_len_sent"])
                        else 0
                    )
                    for i in range(len(mask_sum)):
                        if matched[i]:
                            continue
                        try:
                            sent_norm = _normalize_aspect_text(sentence_raw[i])
                            asp_norm = _normalize_aspect_text(aspect_raw[i])
                        except Exception:
                            unknown_count += 1
                            continue
                        raw_found = asp_norm != "" and asp_norm in sent_norm
                        truncated = (valid_len[i] >= max_len_sent) or (sep_idx[i] >= max_len_sent - 1)
                        if raw_found and truncated:
                            truncated_count += 1
                        elif raw_found:
                            token_mismatch_count += 1
                        else:
                            not_found_raw_count += 1
                else:
                    unknown_count += int((~matched).sum().item())

                if (
                    debug_aspect_span
                    and epoch_idx == 0
                    and split in {"val", "test"}
                    and batch_idx == 0
                ):
                    sentence_raw = batch.get("sentence_raw", [])
                    aspect_raw = batch.get("aspect_raw", [])
                    valid_len = (
                        batch["valid_len"].detach().cpu().tolist()
                        if "valid_len" in batch
                        else [0] * len(mask_sum)
                    )
                    sep_idx = (
                        batch["sep_idx"].detach().cpu().tolist()
                        if "sep_idx" in batch
                        else [-1] * len(mask_sum)
                    )
                    max_len_sent = int(
                        batch["max_len_sent"][0].item()
                        if "max_len_sent" in batch and torch.is_tensor(batch["max_len_sent"])
                        else 0
                    )

                    for i in range(min(10, len(mask_sum))):
                        if mask_sum[i].item() > 0:
                            continue
                        try:
                            sent_norm = _normalize_aspect_text(sentence_raw[i])
                            asp_norm = _normalize_aspect_text(aspect_raw[i])
                        except Exception:
                            sent_norm = ""
                            asp_norm = ""
                        raw_idx = sent_norm.find(asp_norm) if asp_norm else -1
                        raw_found = raw_idx >= 0
                        truncated = (valid_len[i] >= max_len_sent) or (sep_idx[i] >= max_len_sent - 1)
                        if raw_found and truncated:
                            reason = "TRUNCATED"
                        elif raw_found:
                            reason = "TOKEN_MISMATCH"
                        else:
                            reason = "NOT_FOUND_RAW"

                        block = [
                            f"[AspectSpanDebug] epoch={epoch_idx} split={split} batch={batch_idx} sample={i}",
                            f"  max_len_sent={max_len_sent} valid_len={valid_len[i]} sep_idx={sep_idx[i]}",
                            f"  sentence_raw: {sentence_raw[i] if i < len(sentence_raw) else ''}",
                            f"  aspect_raw: {aspect_raw[i] if i < len(aspect_raw) else ''}",
                            f"  sentence_norm: {sent_norm}",
                            f"  aspect_norm: {asp_norm}",
                            f"  aspect_mask_sum: {int(mask_sum[i].item())}",
                            f"  raw_found_substring: {raw_found} idx={raw_idx}",
                            f"  truncated: {truncated}",
                            f"  fail_reason: {reason}",
                        ]
                        print("\n".join(block))

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

    if debug_aspect_span:
        print(
            f"[AspectSpanDiag] split={split} total={match_total} matched={match_matched} "
            f"match_rate={(match_matched / max(1, match_total)) * 100:.2f}% "
            f"token_mismatch={token_mismatch_count} truncated={truncated_count} "
            f"not_found_raw={not_found_raw_count} unknown={unknown_count} "
            f"avg_mask_sum={(match_mask_sum / max(1, match_matched)):.2f}"
        )
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

    # SDModel: build optimizer once and keep it for the whole training
    if cfg.mode == "SDModel":
        warmup_ratio_phase1 = float(cfg.warmup_ratio)
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
    else:
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
        if (cfg.mode != "SDModel") and (cfg.freeze_epochs > 0) and (epoch == cfg.freeze_epochs):
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
            cfg=cfg,
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
            epoch_idx=epoch,
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
                cfg=cfg,
                model=model,
                dataloader=val_loader,
                id2label=id2label,
                print_cf_matrix=True,
                verbose_report=False,
                fusion_method=method,
                f1_average="macro",
                epoch_idx=epoch,
                split="val",
                debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
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
                print("*"*100)
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
                cfg=cfg,
                model=model,
                dataloader=test_loader,
                id2label=id2label,
                print_cf_matrix=True,
                verbose_report=False,
                fusion_method=method,
                f1_average="macro",
                epoch_idx=epoch,
                split="test",
                debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
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
