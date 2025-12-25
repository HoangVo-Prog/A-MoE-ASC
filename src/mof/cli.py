# cli.py
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SentenceTransformer classifier for ATSC with sentence term fusion"
    )

    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--train_path", type=str, default="dataset/atsa/laptop14/train.json")
    parser.add_argument("--val_path", type=str, default="dataset/atsa/laptop14/val.json")
    parser.add_argument("--test_path", type=str, default="dataset/atsa/laptop14/test.json")

    parser.add_argument("--max_len_sent", type=int, default=24)
    parser.add_argument("--max_len_term", type=int, default=4)

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument(
        "--fusion_method",
        type=str,
        default="concat",
        choices=[
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
            "mof",
        ],
    )
    parser.add_argument(
        "--locked_baseline",
        action="store_true",
        help="Lock the experimental baseline so only fusion_method changes across runs.",
    )

    parser.add_argument("--output_dir", type=str, default="saved_model")
    parser.add_argument("--output_name", type=str, default="bert_concat_asc.pt")

    parser.add_argument("--verbose_report", action="store_true")

    parser.add_argument("--k_folds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rolling_k", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--freeze_epochs", type=int, default=3)

    parser.add_argument("--train_full_only", action="store_true")

    parser.add_argument(
        "--benchmark_fusions",
        action="store_true",
        help="Run fusion benchmark across multiple methods and multiple seeds.",
    )
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument(
        "--benchmark_methods",
        type=str,
        default="sent,term,concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction",
    )
    parser.add_argument("--do-ensemble-logits", type=bool, default=True)

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
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument(
        "--head_type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "mof"],
    )

    # MoF knobs
    parser.add_argument(
        "--mof_experts",
        type=str,
        default="sent,term,concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction",
        help="Comma-separated MoF experts. Empty means use default list in mof.py.",
    )

    parser.add_argument("--mof_mix_level", type=str, default="repr", choices=["repr", "logit"])
    parser.add_argument("--mof_lb_coef", type=float, default=0.001)
    parser.add_argument("--mof_lb_mode", type=str, default="switch", choices=["l2", "switch"])
    parser.add_argument("--mof_entropy_coef", type=float, default=0.001)
    parser.add_argument(
        "--mof_mixed_repr_norm",
        type=str,
        default="layernorm",
        choices=["none", "layernorm", "clamp"],
    )
    parser.add_argument("--mof_mixed_repr_norm_clamp", type=float, default=0.0)
    parser.add_argument("--mof_residual_alpha_init", type=float, default=0.1)
    parser.add_argument(
        "--mof_residual_alpha_learnable",
        type=int,
        default=1,
        help="1 = learnable residual alpha, 0 = fixed.",
    )
    parser.add_argument("--mof_router_temperature", type=float, default=1.0)

    parser.add_argument("--mof_disable_expert_scaling", action="store_true")
    parser.add_argument("--mof_expert_norm_clamp", type=float, default=0.0)
    parser.add_argument("--mof_logit_clamp", type=float, default=0.0)

    parser.add_argument("--mof_debug", action="store_true")
    parser.add_argument("--mof_debug_every", type=int, default=100)
    parser.add_argument("--mof_debug_max_batch", type=int, default=1)
    parser.add_argument("--mof_debug_max_experts", type=int, default=0)
    parser.add_argument("--encoder_lr_scale", type=str, default=0.1)

    return parser.parse_args()
