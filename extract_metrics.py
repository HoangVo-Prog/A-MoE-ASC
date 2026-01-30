import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _is_dir(path: str) -> bool:
    return os.path.isdir(path)


def _list_dirs(path: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(path) if _is_dir(os.path.join(path, d))])
    except FileNotFoundError:
        return []


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _find_aggregate_jsons(dataset_dir: str) -> List[str]:
    json_paths: List[str] = []
    for model_name in _list_dirs(dataset_dir):
        model_dir = os.path.join(dataset_dir, model_name)
        for name in os.listdir(model_dir):
            if name.endswith(".json") and name.startswith(f"{model_name}_"):
                json_paths.append(os.path.join(model_dir, name))
    return sorted(json_paths)


def _format_mean_std(mean_val: Any, std_val: Any) -> str:
    try:
        mean_f = float(mean_val)
        std_f = float(std_val)
    except (TypeError, ValueError):
        return "NA"
    return f"{mean_f:.4f}±{std_f:.4f}"


def _format_float(val: Any) -> str:
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "NA"


def _iter_rows(model_name: str, data: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    methods = data.get("methods", {})
    for method_name, method_data in methods.items():
        if "loss_types" in method_data:
            loss_types = method_data.get("loss_types", {})
            for loss_name, loss_data in loss_types.items():
                yield _row(model_name, method_name, loss_name, loss_data)
        else:
            loss_name = method_data.get("loss_type") or "single"
            yield _row(model_name, method_name, loss_name, method_data)


def _row(model: str, method: str, loss: str, loss_data: Dict[str, Any]) -> Dict[str, str]:
    summary = loss_data.get("summary", {})
    ensemble = loss_data.get("ensemble", {}) or {}
    f1_mean = summary.get("f1_mean")
    f1_std = summary.get("f1_std")
    f1_max = summary.get("f1_max")

    return {
        "Model": model,
        "Loss": loss,
        "Method": method,
        "Acc (mean±std)": _format_mean_std(summary.get("acc_mean"), summary.get("acc_std")),
        "Acc min": _format_float(summary.get("acc_min")),
        "Acc max": _format_float(summary.get("acc_max")),
        "F1 (mean±std)": _format_mean_std(summary.get("f1_mean"), summary.get("f1_std")),
        "F1 min": _format_float(summary.get("f1_min")),
        "F1 max": _format_float(summary.get("f1_max")),
        "Ensemble acc": _format_float(ensemble.get("metrics", {}).get("acc")),
        "Ensemble f1": _format_float(ensemble.get("metrics", {}).get("f1")),
        "_f1_mean": f1_mean,
        "_f1_std": f1_std,
        "_f1_max": f1_max,
    }


def _write_markdown(rows: List[Dict[str, str]], output_path: str) -> None:
    headers = [
        "Model",
        "Loss",
        "Method",
        "Acc (mean±std)",
        "Acc min",
        "Acc max",
        "F1 (mean±std)",
        "F1 min",
        "F1 max",
        "Ensemble acc",
        "Ensemble f1",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "NA") for h in headers) + " |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def extract_metrics(input_root: str, dataset: Optional[str], sort_f1: str) -> None:
    datasets = [dataset] if dataset else _list_dirs(input_root)
    for dataset_name in datasets:
        dataset_dir = os.path.join(input_root, dataset_name)
        json_paths = _find_aggregate_jsons(dataset_dir)
        rows: List[Dict[str, str]] = []
        for path in json_paths:
            data = _load_json(path)
            if not data:
                continue
            model_name = data.get("mode") or os.path.basename(os.path.dirname(path))
            rows.extend(_iter_rows(model_name, data))
        if not rows:
            continue
        sort_key = "_f1_mean" if sort_f1 == "mean" else "_f1_max"
        rows.sort(
            key=lambda r: (
                r.get("Model", ""),
                -(float(r[sort_key]) if r.get(sort_key) is not None else float("-inf")),
                float(r["_f1_std"]) if r.get("_f1_std") is not None else float("inf"),
                r.get("Loss", ""),
                r.get("Method", ""),
            )
        )
        output_path = os.path.join(dataset_dir, f"{dataset_name}_summary.md")
        
        if os.path.isfile(output_path):
            os.remove(output_path)
        
        _write_markdown(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract summary and ensemble metrics into markdown tables.")
    parser.add_argument(
        "--input",
        default="results",
        help="Root results folder (default: results)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset folder name (e.g., laptop14)",
    )
    parser.add_argument(
        "--sort-f1",
        choices=["mean", "max"],
        default="mean",
        help="Sort rows within each Model by F1 mean or F1 max (default: mean).",
    )
    args = parser.parse_args()

    extract_metrics(args.input, args.dataset, args.sort_f1)


if __name__ == "__main__":
    main()
