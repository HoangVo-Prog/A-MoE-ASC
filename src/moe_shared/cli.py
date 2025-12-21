import argparse


from shared import FUSION_METHOD_CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SentenceTransformer classifier + MoE for ATSC with sentence term fusion"
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--train_path", type=str, default="dataset/atsa/laptop14/train.json")
    parser.add_argument("--val_path", type=str, default="dataset/atsa/laptop14/val.json")
    parser.add_argument("--test_path", type=str, default="dataset/atsa/laptop14/test.json")

    parser.add_argument("--max_len_sent", type=int, default=24)
    parser.add_argument("--max_len_term", type=int, default=4)

    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--fusion_method", type=str, default="concat", choices=FUSION_METHOD_CHOICES)

    parser.add_argument("--output_dir", type=str, default="saved_model")
    parser.add_argument("--output_name", type=str, default="bert_concat_asc.pt")

    parser.add_argument("--verbose_report", action="store_true")

    parser.add_argument("--k_folds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rolling_k", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--freeze_epochs", type=int, default=0)

    # Benchmark options
    parser.add_argument("--benchmark_fusions", action="store_true")
    parser.add_argument(
        "--benchmark_methods",
        type=str,
        default="",
        help="Comma-separated list of fusion methods to benchmark. Empty means all supported methods.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of integer seeds. Empty means use --seed and --num_seeds.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="Number of seeds to run when --seeds is empty. Seeds are derived from --seed.",
    )
    parser.add_argument("--locked_baseline", action="store_true")
    parser.add_argument("--ensemble_logits", action="store_true")

    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "weighted_ce", "focal"],
        help="Loss type. Use weighted_ce or focal with --class_weights.",
    )
    parser.add_argument(
        "--class_weights",
        type=str,
        default="",
        help="Optional comma-separated class weights, example: 1.0,2.5,1.0",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss (only used when --loss_type focal).",
    )

    # MoE options
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--moe_num_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", dest="top_k", type=int, default=1)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)

    parser.add_argument("--freeze_moe", action="store_true")
    parser.add_argument("--route_mask_pad_tokens", action="store_true")
    parser.add_argument("--step_print_moe", type=float, default=200)

    parser.add_argument("--train_full_only", action="store_true")
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument(
        "--do-ensemble-logits",
        type=bool,
        default=True,
    )
    
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--adamw_foreach", action="store_true")
    parser.add_argument("--adamw_fused", action="store_true")
    
    return parser.parse_args()
