"""Generate LaTeX ablation tables from a markdown summary file."""
from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RowData:
    method: str
    per_dataset: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)


@dataclass
class ModelData:
    order: List[str] = field(default_factory=list)
    rows: Dict[str, RowData] = field(default_factory=dict)


def normalize_header(text: str) -> str:
    text = text.strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_dataset_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    if text.startswith("rest") and not text.startswith("restaurant"):
        text = "restaurant" + text[4:]
    if text.startswith("lap") and not text.startswith("laptop"):
        text = "laptop" + text[3:]
    return text


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def update_max(existing: Optional[float], new: Optional[float]) -> Optional[float]:
    if new is None:
        return existing
    if existing is None or new > existing:
        return new
    return existing


def find_dataset_label(prior_lines: Sequence[str]) -> Optional[str]:
    for line in reversed(prior_lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^#+\s*(.+)$", stripped)
        if m:
            return m.group(1).strip()
        m = re.match(r"(?i)^dataset\s*[:\-]\s*(.+)$", stripped)
        if m:
            return m.group(1).strip()
        break
    return None


def sanitize_name(text: str) -> str:
    text = text.replace(" ", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "model"


def parse_markdown_tables(lines: Sequence[str]) -> List[Tuple[Optional[str], List[Dict[str, str]]]]:
    tables: List[Tuple[Optional[str], List[Dict[str, str]]]] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        if "|" not in line:
            i += 1
            continue
        sep = lines[i + 1]
        if not re.search(r"\|\s*[-:]+", sep):
            i += 1
            continue
        header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header_cells:
            i += 1
            continue
        dataset_label = find_dataset_label(lines[:i])
        rows: List[Dict[str, str]] = []
        i += 2
        while i < len(lines) and "|" in lines[i]:
            row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if len(row_cells) < len(header_cells):
                row_cells += [""] * (len(header_cells) - len(row_cells))
            row = dict(zip(header_cells, row_cells))
            rows.append(row)
            i += 1
        tables.append((dataset_label, rows))
    return tables


def infer_dataset_key_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    return stem or "dataset"


def discover_input_files(input_path: str) -> List[Tuple[Optional[str], str]]:
    if os.path.isfile(input_path):
        return [(None, input_path)]
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path not found: {input_path}")

    files: List[Tuple[Optional[str], str]] = []
    for entry in sorted(os.listdir(input_path)):
        full = os.path.join(input_path, entry)
        if os.path.isdir(full):
            candidates = sorted(
                name for name in os.listdir(full) if name.endswith("_summary.md")
            )
            for name in candidates:
                files.append((entry, os.path.join(full, name)))
        elif os.path.isfile(full) and full.endswith("_summary.md"):
            dataset_key = infer_dataset_key_from_filename(full)
            files.append((dataset_key, full))

    if not files:
        raise ValueError("No summary markdown files found in input directory.")
    return files


def extract_dataset_label(
    tables: List[Tuple[Optional[str], List[Dict[str, str]]]], fallback: str
) -> str:
    for label, _ in tables:
        if label:
            return label
    return fallback


def order_dataset_names(dataset_names: List[str], dataset_order: Optional[str]) -> List[str]:
    if not dataset_order:
        return dataset_names
    requested = [name.strip() for name in dataset_order.split(",") if name.strip()]
    norm_to_name: Dict[str, List[str]] = {}
    for name in dataset_names:
        norm = normalize_dataset_name(name)
        norm_to_name.setdefault(norm, []).append(name)
    result: List[str] = []
    used = set()
    for name in requested:
        norm = normalize_dataset_name(name)
        for candidate in norm_to_name.get(norm, []):
            if candidate not in used:
                result.append(candidate)
                used.add(candidate)
                break
    for name in dataset_names:
        if name not in used:
            result.append(name)
    return result


def collect_model_data(
    files: List[Tuple[Optional[str], str]],
    dataset_order: Optional[str],
) -> Tuple[List[str], Dict[str, ModelData]]:
    dataset_names: List[str] = []
    models: Dict[str, ModelData] = {}

    for dataset_key, path in files:
        lines = read_input(path)
        tables = parse_markdown_tables(lines)
        if not tables:
            raise ValueError(f"No markdown table found in {path}")
        fallback = (dataset_key or infer_dataset_key_from_filename(path)).strip() or "dataset"
        dataset_label = extract_dataset_label(tables, fallback)
        if dataset_label not in dataset_names:
            dataset_names.append(dataset_label)

        for _, rows in tables:
            for row in rows:
                headers = {normalize_header(k): k for k in row.keys()}
                required = ["model", "loss", "method", "acc max", "f1 max"]
                if not all(req in headers for req in required):
                    continue
                model = row[headers["model"]].strip()
                method = row[headers["method"]].strip()
                acc = parse_float(row[headers["acc max"]])
                f1 = parse_float(row[headers["f1 max"]])
                if not model or not method:
                    continue

                model_data = models.setdefault(model, ModelData())
                if method not in model_data.rows:
                    model_data.order.append(method)
                    model_data.rows[method] = RowData(method=method)
                row_data = model_data.rows[method]
                row_data.per_dataset.setdefault(dataset_label, {"acc": None, "f1": None})
                row_data.per_dataset[dataset_label]["acc"] = update_max(
                    row_data.per_dataset[dataset_label]["acc"], acc
                )
                row_data.per_dataset[dataset_label]["f1"] = update_max(
                    row_data.per_dataset[dataset_label]["f1"], f1
                )

    dataset_names = order_dataset_names(dataset_names, dataset_order)
    return dataset_names, models


def compute_max(values: List[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else None


def is_max(value: Optional[float], max_value: Optional[float]) -> bool:
    if value is None or max_value is None:
        return False
    return abs(value - max_value) <= 1e-12


def format_value(value: Optional[float], bold: bool) -> str:
    if value is None:
        return ""
    formatted = f"{value * 100:.2f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def fusion_label(row: RowData) -> str:
    return row.method.strip()


def build_table(model: str, data: ModelData, dataset_names: List[str]) -> str:
    # If no dataset names, treat as single implicit dataset.
    datasets = dataset_names or [""]

    per_dataset_max: Dict[str, Dict[str, Optional[float]]] = {}
    for dataset in datasets:
        accs = []
        f1s = []
        for key in data.order:
            row = data.rows[key]
            accs.append(row.per_dataset.get(dataset, {}).get("acc"))
            f1s.append(row.per_dataset.get(dataset, {}).get("f1"))
        per_dataset_max[dataset] = {
            "acc": compute_max(accs),
            "f1": compute_max(f1s),
        }

    avg_accs: List[Optional[float]] = []
    avg_f1s: List[Optional[float]] = []
    row_avgs: Dict[str, Dict[str, Optional[float]]] = {}
    for key in data.order:
        row = data.rows[key]
        acc_values = [row.per_dataset.get(d, {}).get("acc") for d in datasets]
        f1_values = [row.per_dataset.get(d, {}).get("f1") for d in datasets]
        acc_filtered = [v for v in acc_values if v is not None]
        f1_filtered = [v for v in f1_values if v is not None]
        avg_acc = statistics.fmean(acc_filtered) if acc_filtered else None
        avg_f1 = statistics.fmean(f1_filtered) if f1_filtered else None
        row_avgs[key] = {"acc": avg_acc, "f1": avg_f1}
        avg_accs.append(avg_acc)
        avg_f1s.append(avg_f1)

    max_avg_acc = compute_max(avg_accs)
    max_avg_f1 = compute_max(avg_f1s)

    columns = ["l", "l"] + ["c", "c"] * len(datasets) + ["c", "c"]
    lines = [
        f"\\begin{{tabular}}{{{' '.join(columns)}}}",
        "\\toprule",
    ]

    if dataset_names:
        header = ["\\multirow{2}{*}{Method}", "\\multirow{2}{*}{Fusion}"]
        for dataset in datasets:
            header.append(f"\\multicolumn{{2}}{{c}}{{{dataset}}}")
        header.append("\\multicolumn{2}{c}{Average}")
        lines.append(" ".join([part + " &" for part in header[:-1]]) + f" {header[-1]} \\\\")
        cmidrules = []
        col = 3
        for _ in range(len(datasets) + 1):
            cmidrules.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
            col += 2
        lines.append("".join(cmidrules))
        subheader = ["", ""] + ["Acc", "F1"] * (len(datasets) + 1)
        lines.append(" ".join([part + " &" for part in subheader[:-1]]) + f" {subheader[-1]} \\\\")
    else:
        header_parts = ["Method", "Fusion", "Acc", "F1", "Avg Acc", "Avg F1"]
        lines.append(" ".join([part + " &" for part in header_parts[:-1]]) + f" {header_parts[-1]} \\\\")
    lines.append("\\midrule")

    row_count = len(data.order)
    for idx, key in enumerate(data.order):
        row = data.rows[key]
        model_cell = f"\\multirow{{{row_count}}}{{*}}{{{model}}}" if idx == 0 else ""
        parts = [model_cell, fusion_label(row)]
        for dataset in datasets:
            acc = row.per_dataset.get(dataset, {}).get("acc")
            f1 = row.per_dataset.get(dataset, {}).get("f1")
            parts.append(format_value(acc, is_max(acc, per_dataset_max[dataset]["acc"])))
            parts.append(format_value(f1, is_max(f1, per_dataset_max[dataset]["f1"])))
        avg_acc = row_avgs[key]["acc"]
        avg_f1 = row_avgs[key]["f1"]
        parts.append(format_value(avg_acc, is_max(avg_acc, max_avg_acc)))
        parts.append(format_value(avg_f1, is_max(avg_f1, max_avg_f1)))
        lines.append(" ".join([part + " &" for part in parts[:-1]]) + f" {parts[-1]} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def read_input(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_tables(
    input_path: str,
    output_dir: str,
    caption: Optional[str],
    label_prefix: Optional[str],
    dataset_order: Optional[str],
    dry_run: bool,
) -> None:
    files = discover_input_files(input_path)
    dataset_names, models = collect_model_data(files, dataset_order)
    if not models:
        raise ValueError("No valid rows found with required columns.")

    os.makedirs(output_dir, exist_ok=True)
    model_names = sorted(models.keys())
    output_paths: List[str] = []

    for model in model_names:
        sanitized = sanitize_name(model)
        filename = f"{sanitized}.tex"
        path = os.path.join(output_dir, filename)
        output_paths.append(path)
        if dry_run:
            continue
        lines_out: List[str] = []
        if caption:
            lines_out.append(f"% caption: {caption}")
        if label_prefix:
            lines_out.append(f"% label: {label_prefix}:{sanitized}")
        lines_out.append(build_table(model, models[model], dataset_names))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out))

    index_path = os.path.join(output_dir, "all_tables.tex")
    if not dry_run:
        with open(index_path, "w", encoding="utf-8") as f:
            for model in model_names:
                sanitized = sanitize_name(model)
                f.write(f"\\input{{{sanitized}.tex}}\n")

    if dry_run:
        print(f"models: {len(model_names)}")
        print("datasets: " + (", ".join(dataset_names) if dataset_names else "none"))
        for model in model_names:
            rows = len(models[model].order)
            print(f"- {model}: {rows} rows")
        for path in output_paths:
            print(f"would write: {path}")
        print(f"would write: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX ablation tables from markdown.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a markdown summary file or a results directory.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for .tex files.")
    parser.add_argument("--caption", default=None, help="Optional caption comment.")
    parser.add_argument("--label_prefix", default=None, help="Optional label prefix comment.")
    parser.add_argument(
        "--dataset_order",
        default=None,
        help="Optional comma-separated dataset display order.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print summary without writing files.")
    args = parser.parse_args()

    write_tables(
        input_path=args.input,
        output_dir=args.output_dir,
        caption=args.caption,
        label_prefix=args.label_prefix,
        dataset_order=args.dataset_order,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
