PHASE 1 - Core Fusion Methods Benchmark

What this adds
- Two additional fusion methods: sent and term.
- A benchmark mode that runs multiple fusion methods across multiple seeds.
- Per run artifacts: model checkpoint, result JSON, and learning curves.
- Aggregate artifacts: per seed results and summary tables.

Supported fusion methods
- sent   : sentence CLS only
- term   : aspect term CLS only
- concat : [CLS_sent ; CLS_term]
- add    : CLS_sent + CLS_term
- mul    : CLS_sent * CLS_term
- cross  : MultiheadAttention with term CLS as query, sentence tokens as key and value

Commands

1) Locked baseline benchmark (recommended)

python runner.py \
  --locked_baseline \
  --benchmark_fusions \
  --num_seeds 3 \
  --benchmark_methods sent,term,concat,add,mul,cross \
  --output_dir saved_model \
  --output_name phase1_locked_baseline

2) Override explicit seeds

python runner.py \
  --locked_baseline \
  --benchmark_fusions \
  --seeds 42,43,44 \
  --output_dir saved_model \
  --output_name phase1_locked_baseline

3) Non locked baseline benchmark (uses CLI dataset paths and hyperparams)

python runner.py \
  --benchmark_fusions \
  --num_seeds 3 \
  --model_name roberta-base \
  --train_path dataset/atsa/laptop14/train.json \
  --val_path dataset/atsa/laptop14/val.json \
  --test_path dataset/atsa/laptop14/test.json \
  --epochs 20 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --lr 2e-5 \
  --warmup_ratio 0.1 \
  --rolling_k 3 \
  --early_stop_patience 3 \
  --freeze_epochs 5 \
  --output_dir saved_model \
  --output_name phase1_custom

Outputs
- saved_model/phase1/summary.csv
- saved_model/phase1/summary.json
- saved_model/phase1/results_per_seed.csv
- saved_model/phase1/results_per_seed.json

Per run directory layout
- saved_model/phase1/<method>/seed_<seed>/
  - model.pt
  - result.json
  - loss_curve.png
  - f1_curve.png
