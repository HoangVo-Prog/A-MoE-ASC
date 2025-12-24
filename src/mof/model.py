# model.py
from typing import List, Optional

from shared import DEVICE, BaseBertConcatClassifier
from .mof import MoFBertConcatClassifier


class BertConcatClassifier(BaseBertConcatClassifier):
    pass


_MOF_EXPERTS_ALLOWED = {
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
}


def _parse_mof_experts(value) -> Optional[List[str]]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        experts = [str(x).strip() for x in value if str(x).strip()]
    else:
        s = str(value).strip()
        if not s:
            return None
        experts = [p.strip() for p in s.split(",") if p.strip()]

    if not experts:
        return None

    unknown = [e for e in experts if e not in _MOF_EXPERTS_ALLOWED]
    if unknown:
        raise ValueError(
            f"Unknown mof_experts: {unknown}. Allowed: {sorted(_MOF_EXPERTS_ALLOWED)}"
        )

    return experts


def build_model(*, cfg, num_labels: int):
    head_type = str(getattr(cfg, "head_type", "linear")).lower()

    if head_type == "mof":
        mof_experts = _parse_mof_experts(getattr(cfg, "mof_experts", ""))

        return MoFBertConcatClassifier(
            model_name=str(getattr(cfg, "model_name")),
            num_labels=int(num_labels),
            dropout=float(getattr(cfg, "dropout", 0.1)),
            head_type=str(getattr(cfg, "head_type", "mof")),
            loss_type=str(getattr(cfg, "loss_type", "ce")),
            class_weights=getattr(cfg, "class_weights", None),
            focal_gamma=float(getattr(cfg, "focal_gamma", 2.0)),
            mof_experts=mof_experts,
            mof_debug=bool(getattr(cfg, "mof_debug", False)),
            mof_debug_every=int(getattr(cfg, "mof_debug_every", 200)),
            mof_debug_max_batch=int(getattr(cfg, "mof_debug_max_batch", 1)),
            mof_debug_max_experts=int(getattr(cfg, "mof_debug_max_experts", 0)),
            mof_mix_level=str(getattr(cfg, "mof_mix_level", "repr")),
            mof_lb_coef=float(getattr(cfg, "mof_lb_coef", 0.001)),
            mof_router_temperature=float(getattr(cfg, "mof_router_temperature", 1.0)),
            mof_disable_expert_scaling=bool(getattr(cfg, "mof_disable_expert_scaling", False)),
            mof_expert_norm_clamp=float(getattr(cfg, "mof_expert_norm_clamp", 0.0)),
            mof_logit_clamp=float(getattr(cfg, "mof_logit_clamp", 0.0)),
        ).to(DEVICE)

    return BertConcatClassifier(
        model_name=str(getattr(cfg, "model_name")),
        num_labels=int(num_labels),
        dropout=float(getattr(cfg, "dropout", 0.1)),
        head_type=str(getattr(cfg, "head_type", "linear")),
        loss_type=str(getattr(cfg, "loss_type", "ce")),
        class_weights=getattr(cfg, "class_weights", None),
        focal_gamma=float(getattr(cfg, "focal_gamma", 2.0)),
    ).to(DEVICE)
