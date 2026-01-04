import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FUSION_METHOD_CHOICES = [
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