from __future__ import annotations
import torch
import gc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, Iterable, Optional, Mapping
import argparse
import inspect

from .const import DEVICE


def cleanup_cuda(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _parse_int_list(csv: str) -> list[int]:
    s = (csv or "").strip()
    if not s:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def parse_str_list(csv: str) -> list[str]:
    s = (csv or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_confusions(cms: list[np.ndarray]) -> dict:
    if len(cms) == 0:
        return {}

    arr = np.stack([np.asarray(cm, dtype=np.float64) for cm in cms], axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    denom = mean.sum(axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    mean_norm = mean / denom
    std_norm = std / denom

    return {
        "cm_mean": mean.tolist(),
        "cm_std": std.tolist(),
        "cm_mean_normalized": mean_norm.tolist(),
        "cm_std_normalized": std_norm.tolist(),
    }


def _cfg_to_dict(cfg) -> dict:
    if hasattr(cfg, "to_dict") and callable(getattr(cfg, "to_dict")):
        return cfg.to_dict()
    try:
        return asdict(cfg)
    except Exception:
        return dict(getattr(cfg, "__dict__", {}))


def cfg_to_flat_dict(cfg: Any) -> Dict[str, Any]:
    # Support dataclass Config hoặc object có __dict__
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(getattr(cfg, "__dict__", {}))


def build_kwargs_from_signature(cfg_dict: Dict[str, Any], model_cls: type) -> Dict[str, Any]:
    sig = inspect.signature(model_cls.__init__)

    # Nếu model __init__ có **kwargs thì pass hết cho nhanh
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # nhưng vẫn bỏ self
        return dict(cfg_dict)

    kwargs: Dict[str, Any] = {}
    missing_required = []

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        if name in cfg_dict:
            kwargs[name] = cfg_dict[name]
            continue

        # param bắt buộc mà cfg không có
        if p.default is inspect._empty and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            f"{model_cls.__name__} is missing required args from cfg: {missing_required}. "
            f"Either add them to Config or give defaults in {model_cls.__name__}.__init__."
        )

    return kwargs


def safe_float(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float("nan")


def logits_to_metrics(logits: np.ndarray, labels: np.ndarray):
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"acc": float(acc), "f1": float(f1)}


@torch.no_grad()
def collect_test_logits(
    *,
    model: torch.nn.Module,
    test_loader,
    fusion_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_chunks = []
    labels_chunks = []

    for batch in test_loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
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
    
