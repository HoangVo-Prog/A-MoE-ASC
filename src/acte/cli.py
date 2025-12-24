# src/acte/cli.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser("ACTE Token Evidence MoE")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--fusion_method", type=str, default="concat")
    parser.add_argument("--head_type", type=str, default="linear")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--freeze_epochs", type=int, default=3)
    parser.add_argument("--rolling_k", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_full_only", action="store_true")

    parser.add_argument("--max_len_sent", type=int, default=24)
    parser.add_argument("--max_len_term", type=int, default=4)

    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--class_weights", type=str, default="")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--output_dir", type=str, default="outputs_acte")
    parser.add_argument("--output_name", type=str, default="acte_temoe")
    
    parser.add_argument("--verbose_report", action="store_true")


    parser.add_argument("--benchmark_fusions", action="store_true")
    parser.add_argument("--benchmark_methods", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--num_seeds", type=int, default=3)

    # ACTE specific
    parser.add_argument("--acte_num_experts", type=int, default=4)
    parser.add_argument("--acte_top_k", type=int, default=2)
    parser.add_argument("--acte_top_m", type=int, default=8)
    parser.add_argument("--acte_expert_hidden", type=int, default=256)
    parser.add_argument("--acte_router_dropout", type=float, default=0.0)
    parser.add_argument("--acte_expert_dropout", type=float, default=0.1)
    parser.add_argument("--acte_score_temperature", type=float, default=1.0)
    parser.add_argument("--acte_combine_with_base", type=str, default="add", choices=["add", "concat"])

    return parser.parse_args()
