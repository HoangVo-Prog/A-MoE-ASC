import json
from pathlib import Path
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

def json_to_csv(json_path: Path, output_csv: Path, fields=FIELDS) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {}) or {}
    ensemble = data.get("ensemble", {}) or {}

    rows = []
    methods = set(summary.keys()) | set(ensemble.keys())

    for method in sorted(methods):
        row = {"method": method}

        sum_m = summary.get(method, {}) or {}
        ens_m = ensemble.get(method, {}) or {}

        # ưu tiên ensemble nếu field nằm ở cả hai (vì thường ensemble là kết quả cuối)
        for k in fields:
            if k in sum_m:
                row[k] = sum_m[k]
            if k in ens_m:
                row[k] = ens_m[k]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")

    # giữ đúng thứ tự cột, thiếu field thì vẫn tạo cột (NaN)
    for k in fields:
        if k not in df.columns:
            df[k] = pd.NA
    df = df[fields]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, float_format="%.6f")
    return df

def batch_convert_json_dir(input_dir: str | Path, output_dir: str | Path, glob_pattern: str = "*.json"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob(glob_pattern))
    if not json_files:
        print(f"[WARN] No JSON files found in: {input_dir} (pattern={glob_pattern})")
        return

    ok, fail = 0, 0
    for jp in json_files:
        out_csv = output_dir / f"{jp.stem}.csv"
        try:
            json_to_csv(jp, out_csv)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {jp} -> {out_csv}: {type(e).__name__}: {e}")

    print(f"[DONE] Converted: {ok} | Failed: {fail} | Output dir: {output_dir}")

if __name__ == "__main__":
    # ====== CONFIG ======
    INPUT_DIR = Path("../../results/baseline/rest14/json")         
    OUTPUT_DIR = Path("../../results/baseline/rest14/csv")    
    PATTERN = "*.json"                                

    batch_convert_json_dir(INPUT_DIR, OUTPUT_DIR, glob_pattern=PATTERN)
