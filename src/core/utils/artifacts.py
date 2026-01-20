from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_artifacts(
    *,
    output_dir: str,
    mode: str,
    method: str,
    seed: int | str,
    fold: int | str,
    split: str,
    metrics: Dict[str, Any],
) -> None:
    base = os.path.join(
        output_dir,
        str(mode),
        str(method),
        f"seed_{seed}",
        f"fold_{fold}",
        str(split),
    )
    os.makedirs(base, exist_ok=True)

    _write_json(os.path.join(base, "metrics.json"), _to_jsonable(metrics))

    confusion = metrics.get("confusion")
    if confusion is not None:
        _plot_confusion(confusion, os.path.join(base, "confusion.png"))

    calibration = metrics.get("calibration")
    if calibration is not None:
        _plot_reliability(calibration, os.path.join(base, "reliability.png"))
        _plot_confidence_hist(calibration, os.path.join(base, "confidence_hist.png"))

    moe_metrics = metrics.get("moe_metrics")
    if moe_metrics is not None:
        _plot_entropy_hist(moe_metrics, os.path.join(base, "router_entropy_hist.png"))
        _plot_expert_usage(moe_metrics, os.path.join(base, "expert_usage.png"))
        _plot_top1_hist(moe_metrics, os.path.join(base, "top1_hist.png"))


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _plot_confusion(confusion: Dict[str, Any], path: str) -> None:
    cm = np.asarray(confusion.get("cm", []), dtype=np.float32)
    if cm.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_reliability(calibration: Dict[str, Any], path: str) -> None:
    bins = calibration.get("bins", [])
    if not bins:
        return
    acc = [b["acc"] if b["acc"] is not None else np.nan for b in bins]
    conf = [b["conf"] if b["conf"] is not None else np.nan for b in bins]
    x = np.arange(len(acc))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, len(acc) - 1], [0, 1], linestyle="--", color="gray")
    ax.plot(x, acc, marker="o", label="Accuracy")
    ax.plot(x, conf, marker="o", label="Confidence")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_confidence_hist(calibration: Dict[str, Any], path: str) -> None:
    hist = calibration.get("conf_hist", {})
    edges = hist.get("edges")
    counts = hist.get("counts")
    if not edges or not counts:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(counts)), counts, width=0.8)
    ax.set_title("Confidence Histogram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_entropy_hist(moe_metrics: Dict[str, Any], path: str) -> None:
    hist = moe_metrics.get("entropy_hist")
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(len(hist)), hist, width=0.8)
    ax.set_title("Router Entropy Histogram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_expert_usage(moe_metrics: Dict[str, Any], path: str) -> None:
    mean_prob = moe_metrics.get("mean_prob")
    if not mean_prob:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(mean_prob)), mean_prob, width=0.8)
    ax.set_title("Expert Usage (Mean Prob)")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Mean Prob")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_top1_hist(moe_metrics: Dict[str, Any], path: str) -> None:
    top1_hist = moe_metrics.get("top1_hist")
    if not top1_hist:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(top1_hist)), top1_hist, width=0.8)
    ax.set_title("Top1 Histogram")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
