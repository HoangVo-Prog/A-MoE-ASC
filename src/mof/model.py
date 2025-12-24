from typing import Optional, List

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

    # Accept list/tuple or comma-separated string
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
            model_name=cfg.model_name,
            num_labels=num_labels,
            dropout=cfg.dropout,
            head_type=cfg.head_type,
            loss_type=cfg.loss_type,
            class_weights=cfg.class_weights,
            focal_gamma=cfg.focal_gamma,
            mof_experts=mof_experts,
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
