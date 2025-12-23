from typing import Optional
from shared import DEVICE, BaseBertConcatClassifier

from .mof import MoFBertConcatClassifier


class BertConcatClassifier(BaseBertConcatClassifier):
    pass


def build_model(*, cfg, num_labels: int):
    head_type = str(getattr(cfg, "head_type", "linear")).lower()

    if head_type == "mof":
        return MoFBertConcatClassifier(
            model_name=cfg.model_name,
            num_labels=num_labels,
            dropout=cfg.dropout,
            head_type=cfg.head_type,
            loss_type=cfg.loss_type,
            class_weights=cfg.class_weights,
            focal_gamma=cfg.focal_gamma,
            mof_include_sent_term=bool(getattr(cfg, "mof_include_sent_term", False)),
            mof_debug=bool(getattr(cfg, "mof_debug", False)),
            mof_debug_every=int(getattr(cfg, "mof_debug_every", 200)),
            mof_debug_max_batch=int(getattr(cfg, "mof_debug_max_batch", 1)),
            mof_debug_max_experts=int(getattr(cfg, "mof_debug_max_experts", 0)),
        ).to(DEVICE)

    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        head_type=cfg.head_type,
        loss_type=cfg.loss_type,
        class_weights=cfg.class_weights,
        focal_gamma=cfg.focal_gamma,
    ).to(DEVICE)
