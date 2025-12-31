from __future__ import annotations
import torch
import gc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Mapping
import argparse

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


def _parse_str_list(csv: str) -> list[str]:
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

def _to_dict(d: Any) -> dict:
    # dataclass instance
    if is_dataclass(d) and not isinstance(d, type):
        return asdict(d)
    # argparse Namespace
    if isinstance(d, argparse.Namespace):
        return vars(d)
    # plain dict or mapping
    if isinstance(d, Mapping):
        return dict(d)
    raise TypeError(f"Unsupported input type for config kwargs: {type(d)}")

def _unflatten(d: Mapping[str, Any], sep: str = ".") -> dict:
    """
    Convert {'base.lr':1e-5, 'kfold.k':5} -> {'base':{'lr':...}, 'kfold':{'k':...}}
    If no sep in keys, returns shallow copy.
    """
    out: dict = {}
    for k, v in d.items():
        if not isinstance(k, str) or sep not in k:
            out[k] = v
            continue
        cur = out
        parts = k.split(sep)
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = v
    return out

def filter_config_kwargs(d: Any, config_cls: Any, *, allow_dot_keys: bool = True, sep: str = ".") -> dict:
    """
    Filter input (cfg dataclass / dict / argparse Namespace) by dataclass schema (config_cls).
    Keeps only fields defined in config_cls, and recursively filters nested dataclasses.
    Supports dot-keys like 'base.lr' if allow_dot_keys=True.
    """
    cls = config_cls if isinstance(config_cls, type) else type(config_cls)
    if not is_dataclass(cls):
        raise TypeError(f"config_cls must be a dataclass type/instance, got {cls}")

    raw = _to_dict(d)
    if allow_dot_keys:
        raw = _unflatten(raw, sep=sep)

    out: dict = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        v = raw[f.name]

        # nested dataclass field + input value is dict
        ftype = f.type
        if is_dataclass(ftype) and isinstance(v, Mapping):
            out[f.name] = filter_config_kwargs(v, ftype, allow_dot_keys=allow_dot_keys, sep=sep)
        else:
            out[f.name] = v

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
    