import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

FIELDS = [
    "cv_val_f1_mean_over_seeds",
    "cv_val_f1_std_over_seeds",
    "cv_test_f1_mean_over_seeds",
    "cv_test_f1_std_over_seeds",
    "full_test_f1_mean_over_seeds",
    "full_test_f1_std_over_seeds",
    "full_test_acc_mean_over_seeds",
    "full_test_acc_std_over_seeds",
    "full_ens_test_acc",
    "full_ens_test_f1",
    "delta_full_test_f1_vs_sent",
    "delta_full_ens_test_f1_vs_sent",
    "cv_val_seed_ens_acc",
    "cv_val_seed_ens_f1",
    "cv_test_seed_ens_acc",
    "cv_test_seed_ens_f1",
]

METHOD_ORDER = [
    "sent",
    "term",
    "concat",
    "add",
    "mul",
    "cross",
    "gated_concat",
    "bilinear",
    "coattn",
    "late_interaction",
]


def sort_methods(methods: list[str], method_order: Sequence[str] = METHOD_ORDER) -> list[str]:
    priority = {m: i for i, m in enumerate(method_order)}
    return sorted(methods, key=lambda m: (priority.get(m, len(priority)), m))


def json_to_csv(json_path: Path, output_csv: Path, fields: Sequence[str] = FIELDS) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {}) or {}
    ensemble = data.get("ensemble", {}) or {}

    all_methods = sort_methods(list(set(summary) | set(ensemble)))
    rows = []

    for method in all_methods:
        row = {"method": method}

        sum_m = summary.get(method, {}) or {}
        ens_m = ensemble.get(method, {}) or {}

        for k in fields:
            if k in sum_m:
                row[k] = sum_m[k]
            if k in ens_m:
                row[k] = ens_m[k]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")

    for k in fields:
        if k not in df.columns:
            df[k] = pd.NA
    df = df[list(fields)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, float_format="%.6f")
    return df


def batch_convert_json_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    glob_pattern: str = "*.json",
    recursive: bool = True,
    overwrite: bool = True,
    quiet: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        json_files = sorted(input_dir.rglob(glob_pattern))
    else:
        json_files = sorted(input_dir.glob(glob_pattern))

    if not json_files:
        if not quiet:
            print(f"[WARN] No JSON files found in {input_dir} with pattern {glob_pattern}")
        return

    ok = 0
    fail = 0
    skipped = 0

    for jp in json_files:
        out_csv = output_dir / f"{jp.stem}.csv"

        if out_csv.exists() and not overwrite:
            skipped += 1
            continue

        try:
            json_to_csv(jp, out_csv)
            ok += 1
        except Exception as e:
            fail += 1
            if not quiet:
                print(f"[FAIL] {jp.name}: {e}")

    if not quiet:
        print(f"[DONE] Converted {ok} files, failed {fail}, skipped {skipped}, output at {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert ATSC result JSON files (summary/ensemble) into CSV tables."
    )
    p.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("../../results/baseline/laptop14/json"),
        help="Input directory containing JSON files.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("../../results/baseline/laptop14/csv"),
        help="Output directory for generated CSV files.",
    )
    p.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.json",
        help='Glob pattern for JSON files. Example: "*.json" or "*_focal.json".',
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search (use glob instead of rglob).",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip CSV files that already exist instead of overwriting.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    batch_convert_json_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        glob_pattern=args.pattern,
        recursive=not args.no_recursive,
        overwrite=not args.no_overwrite,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
