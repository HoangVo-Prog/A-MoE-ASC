# src/acte/cli.py
import argparse

def parse_args():
    p = argparse.ArgumentParser("ACTE Token Evidence MoE")

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, default="")
    p.add_argument("--test_path", type=str, required=True)

    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--fusion_method", type=str, default="concat")
    p.add_argument("--head_type", type=str, default="linear")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--freeze_epochs", type=int, default=3)
    p.add_argument("--rolling_k", type=int, default=3)
    p.add_argument("--early_stop_patience", type=int, default=5)

    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_full_only", action="store_true")

    p.add_argument("--max_len_sent", type=int, default=24)
    p.add_argument("--max_len_term", type=int, default=4)

    p.add_argument("--loss_type", type=str, default="ce", choices=["ce", "weighted_ce", "focal"])
    p.add_argument("--class_weights", type=str, default="")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument("--output_dir", type=str, default="outputs_acte")
    p.add_argument("--output_name", type=str, default="acte_temoe")

    p.add_argument("--benchmark_fusions", action="store_true")
    p.add_argument("--benchmark_methods", type=str, default="")
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--num_seeds", type=int, default=3)

    # ACTE specific
    p.add_argument("--acte_num_experts", type=int, default=4)
    p.add_argument("--acte_top_k", type=int, default=2)
    p.add_argument("--acte_top_m", type=int, default=8)
    p.add_argument("--acte_expert_hidden", type=int, default=256)
    p.add_argument("--acte_router_dropout", type=float, default=0.0)
    p.add_argument("--acte_expert_dropout", type=float, default=0.1)
    p.add_argument("--acte_score_temperature", type=float, default=1.0)
    p.add_argument("--acte_combine_with_base", type=str, default="add", choices=["add", "concat"])

    return p.parse_args()
