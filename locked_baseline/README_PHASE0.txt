PHASE 0 - Experimental Baseline Locking

Run with the files in this folder.

Locked baseline (only fusion_method changes)
python runner.py --locked_baseline --fusion_method concat

Repeat to check reproducibility
python runner.py --locked_baseline --fusion_method concat

Compare fusion methods (only change fusion_method)
python runner.py --locked_baseline --fusion_method add
python runner.py --locked_baseline --fusion_method mul
python runner.py --locked_baseline --fusion_method cross

Outputs
- outputs/<output_name>.pt
- outputs/<output_name>.baseline_lock.json
