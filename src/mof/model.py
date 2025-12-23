import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
from shared import DEVICE, BaseBertConcatClassifier

from baseline.mof import MoFBertConcatClassifier


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
