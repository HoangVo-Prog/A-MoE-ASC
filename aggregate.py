import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.utils.artifacts import aggregate_metrics
from src.core.utils.general import aggregate_confusions, mean_std


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _is_dir(path: str) -> bool:
    return os.path.isdir(path)


def _list_dirs(path: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(path) if _is_dir(os.path.join(path, d))])
    except FileNotFoundError:
        return []


def _collect_seed_metrics(loss_dir: str) -> tuple[List[Dict[str, Any]], List[int], List[np.ndarray]]:
    per_seed_records: List[Dict[str, Any]] = []
    seeds: List[int] = []
    cms: List[np.ndarray] = []

    for seed_dir in _list_dirs(loss_dir):
        m = re.match(r"^seed_(\d+)$", seed_dir)
        if not m:
            continue
        seed = int(m.group(1))
        metrics_path = os.path.join(loss_dir, seed_dir, "fold_full", "test", "metrics.json")
        metrics = _load_json(metrics_path)
        if not metrics:
            continue
        seeds.append(seed)
        per_seed_records.append(
            {
                "seed": seed,
                "loss": metrics.get("loss"),
                "acc": float(metrics.get("acc", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "f1_per_class": metrics.get("f1_per_class"),
                "confusion": metrics.get("confusion"),
                "calibration": metrics.get("calibration"),
                "moe_metrics": metrics.get("moe_metrics"),
            }
        )
        confusion = metrics.get("confusion") or {}
        cm = confusion.get("cm")
        if cm is not None:
            try:
                cms.append(np.asarray(cm, dtype=np.float64))
            except Exception:
                pass

    return per_seed_records, seeds, cms


def _collect_ensemble(loss_dir: str) -> Optional[Dict[str, Any]]:
    metrics_path = os.path.join(loss_dir, "seed_ens", "fold_full", "test", "metrics.json")
    metrics = _load_json(metrics_path)
    if not metrics:
        return None
    return {"metrics": metrics}


def _build_summary(mode: str, method: str, loss_type: str, loss_dir: str) -> Dict[str, Any]:
    per_seed_records, seeds, cms = _collect_seed_metrics(loss_dir)

    accs = [float(r.get("acc", 0.0)) for r in per_seed_records]
    f1s = [float(r.get("f1", 0.0)) for r in per_seed_records]
    acc_mean, acc_std = mean_std(accs)
    f1_mean, f1_std = mean_std(f1s)
    acc_min = float(np.min(accs)) if accs else float("nan")
    acc_max = float(np.max(accs)) if accs else float("nan")
    f1_min = float(np.min(f1s)) if f1s else float("nan")
    f1_max = float(np.max(f1s)) if f1s else float("nan")

    agg_confusions = aggregate_confusions(cms)

    metrics_list = []
    for record in per_seed_records:
        metrics_list.append(
            {
                "loss": record.get("loss"),
                "acc": float(record.get("acc", 0.0)),
                "f1": float(record.get("f1", 0.0)),
                "f1_per_class": record.get("f1_per_class"),
                "confusion": record.get("confusion"),
                "moe_metrics": record.get("moe_metrics"),
                "calibration": record.get("calibration"),
            }
        )
    agg = aggregate_metrics(metrics_list)

    ensemble_block = _collect_ensemble(loss_dir)

    return {
        "seeds": seeds,
        "runs": per_seed_records,
        "summary": {
            "acc_mean": float(acc_mean),
            "acc_std": float(acc_std),
            "acc_min": acc_min,
            "acc_max": acc_max,
            "f1_mean": float(f1_mean),
            "f1_std": float(f1_std),
            "f1_min": f1_min,
            "f1_max": f1_max,
        },
        "confusion": agg_confusions,
        "aggregate": agg,
        "ensemble": ensemble_block,
    }


def aggregate_results(input_dir: str, mode: Optional[str], output_path: str) -> Dict[str, Any]:
    mode_name = mode or os.path.basename(os.path.normpath(input_dir))
    methods = _list_dirs(input_dir)
    method_summaries: Dict[str, Any] = {}
    all_loss_types: List[str] = []

    for method in methods:
        method_dir = os.path.join(input_dir, method)
        loss_types = _list_dirs(method_dir)
        if not loss_types:
            continue
        for loss_type in loss_types:
            if loss_type not in all_loss_types:
                all_loss_types.append(loss_type)
        if len(loss_types) == 1:
            loss_type = loss_types[0]
            loss_dir = os.path.join(method_dir, loss_type)
            method_summaries[method] = {
                "loss_type": loss_type,
                **_build_summary(mode_name, method, loss_type, loss_dir),
                "loss_type_order": [loss_type],
            }
        else:
            per_loss: Dict[str, Any] = {}
            for loss_type in loss_types:
                loss_dir = os.path.join(method_dir, loss_type)
                per_loss[loss_type] = _build_summary(mode_name, method, loss_type, loss_dir)
            method_summaries[method] = {
                "loss_types": per_loss,
                "loss_type_order": loss_types,
            }

    combined = {
        "mode": mode_name,
        "methods": method_summaries,
        "method_order": methods,
        "loss_type_order": all_loss_types,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=True, indent=2)

    return combined


def _is_glob(path: str) -> bool:
    return any(ch in path for ch in ("*", "?", "["))


def _resolve_input_dirs(input_path: str, batch: bool) -> List[str]:
    if _is_glob(input_path):
        candidates = glob.glob(input_path)
        return sorted([p for p in candidates if _is_dir(p)])
    if batch:
        return [os.path.join(input_path, d) for d in _list_dirs(input_path)]
    return [input_path]


def _output_path_for_input(input_dir: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    model_name = os.path.basename(os.path.normpath(input_dir))
    dataset_type = os.path.basename(os.path.normpath(os.path.dirname(input_dir)))
    filename = f"{model_name}_{dataset_type}.json"
    return os.path.join(input_dir, filename)


def aggregate_many(input_path: str, mode: Optional[str], output_path: Optional[str], batch: bool) -> None:
    input_dirs = _resolve_input_dirs(input_path, batch)
    if not input_dirs:
        raise FileNotFoundError(f"No input directories matched: {input_path}")

    for input_dir in input_dirs:
        resolved_output = _output_path_for_input(input_dir, output_path if len(input_dirs) == 1 else None)
        aggregate_results(input_dir, mode, resolved_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-method results into a combined JSON.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder or glob (e.g., results/laptop14/HAGMoE or results/laptop14/*)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: {model_name}_{dataset_type}.json in the input folder)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Override mode name (default: basename of input)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all immediate subfolders of --input",
    )
    args = parser.parse_args()

    aggregate_many(args.input, args.mode, args.output, args.batch)


if __name__ == "__main__":
    main()
