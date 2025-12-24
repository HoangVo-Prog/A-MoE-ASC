# src/acte/model.py
from __future__ import annotations
from shared import DEVICE
from .ac_temoe import ACTokenEvidenceMoEClassifier, ACTEConfig

def build_model(*, cfg, num_labels: int):
    acte_cfg = ACTEConfig(
        num_experts=int(getattr(cfg, "acte_num_experts", 4)),
        top_k=int(getattr(cfg, "acte_top_k", 2)),
        top_m_tokens=int(getattr(cfg, "acte_top_m", 8)),
        expert_hidden=int(getattr(cfg, "acte_expert_hidden", 256)),
        router_dropout=float(getattr(cfg, "acte_router_dropout", 0.0)),
        expert_dropout=float(getattr(cfg, "acte_expert_dropout", 0.1)),
        score_temperature=float(getattr(cfg, "acte_score_temperature", 1.0)),
        combine_with_base=str(getattr(cfg, "acte_combine_with_base", "add")),
    )

    return ACTokenEvidenceMoEClassifier(
        model_name=str(cfg.model_name),
        num_labels=int(num_labels),
        dropout=float(cfg.dropout),
        head_type=str(cfg.head_type),
        loss_type=str(getattr(cfg, "loss_type", "ce")),
        class_weights=getattr(cfg, "class_weights", None),
        focal_gamma=float(getattr(cfg, "focal_gamma", 2.0)),
        acte_cfg=acte_cfg,
    ).to(DEVICE)
