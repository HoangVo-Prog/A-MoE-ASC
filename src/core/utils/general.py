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

def _dataclass_to_dict_shallow(obj: Any) -> Dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _to_dict(d: Any) -> dict:
    if is_dataclass(d) and not isinstance(d, type):
        return _dataclass_to_dict_shallow(d)
    if isinstance(d, argparse.Namespace):
        return vars(d)
    if isinstance(d, Mapping):
        return dict(d)
    raise TypeError(f"Unsupported input type: {type(d)}")


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if obj is None:
        raise KeyError(key)
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    raise KeyError(key)


def filter_config_kwargs(
    d: Any,
    config_or_model: Any,
    *,
    fallback_sources: Optional[Iterable[Any]] = None,
) -> dict:
    raw = _to_dict(d)

    # Case A: dataclass schema
    cls = config_or_model if isinstance(config_or_model, type) else type(config_or_model)
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in raw.items() if k in allowed}

    # Case B: class/callable signature
    target = config_or_model.__init__ if inspect.isclass(config_or_model) else config_or_model
    sig = inspect.signature(target)

    # nếu có **kwargs thì không cần lọc
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return raw

    allowed = {k for k in sig.parameters.keys() if k != "self"}

    out = {}
    # 1) ưu tiên lấy trực tiếp từ raw
    for k in allowed:
        if k in raw:
            out[k] = raw[k]

    # 2) nếu thiếu, thử lấy từ fallback_sources (ví dụ cfg.base/cfg.moe/cfg.kfold)
    if fallback_sources:
        for k in allowed:
            if k in out:
                continue
            for src in fallback_sources:
                try:
                    out[k] = _get_attr_or_key(src, k)
                    break
                except KeyError:
                    continue

    return out
def _safe_float(x) -> float:
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
    