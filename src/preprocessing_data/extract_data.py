import json
from pathlib import Path
import pandas as pd

# ====== CONFIG ======
json_path = Path("../../results/baseline/laptop14_focal.json")         
output_csv = Path("../../results/baseline/laptop14_focal.csv")

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

# ====== LOAD JSON ======
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", {})
ensemble = data.get("ensemble", {})

# ====== COLLECT ======
rows = []

for method, sum_m in summary.items():
    row = {"method": method}

    # from summary
    for k in FIELDS:
        if k in sum_m:
            row[k] = sum_m[k]

    # from ensemble block
    ens_m = ensemble.get(method, {})
    for k in FIELDS:
        if k in ens_m:
            row[k] = ens_m[k]

    rows.append(row)

# ====== SAVE CSV ======
df = pd.DataFrame(rows)
df = df.set_index("method")
df = df[FIELDS]  # giữ đúng thứ tự cột

df.to_csv(output_csv, float_format="%.6f")
