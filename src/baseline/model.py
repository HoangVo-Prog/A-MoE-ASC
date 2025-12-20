import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional
from shared import DEVICE, BaseBertConcatClassifier


class BertConcatClassifier(BaseBertConcatClassifier):
    pass

def build_model(*, cfg, num_labels: int):
    return BertConcatClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
        head_type=cfg.head_type,
    ).to(DEVICE)
    