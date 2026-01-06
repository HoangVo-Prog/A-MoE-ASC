from __future__ import annotations

import torch

from src.models.base_model import BaseModel


class SDModel(BaseModel):
    """
    Patch 1 (skeleton):
    - Mục tiêu: plug-in được vào pipeline qua cfg.mode="SDModel"
    - Chưa implement MoE Semantic Deformation (Patch 3)
    - Chưa freeze encoder cứng (Patch 2)
    - Forward hiện tại gọi y hệt BaseModel để benchmark không bị phá
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights=None,
        focal_gamma: float = 2.0,
        # --- SD placeholders (không dùng ở Patch 1) ---
        num_experts: int = 8,
        sd_rank: int = 8,
        sd_alpha: float = 16.0,
        sd_lambda_bal: float = 0.01,
        sd_lambda_div: float = 0.001,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_type=head_type,
            loss_type=loss_type,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

        # Lưu lại hyper SD để Patch 3 dùng (hiện chưa ảnh hưởng forward)
        self.num_experts = int(num_experts)
        self.sd_rank = int(sd_rank)
        self.sd_alpha = float(sd_alpha)
        self.sd_lambda_bal = float(sd_lambda_bal)
        self.sd_lambda_div = float(sd_lambda_div)
        self.router_temperature = float(router_temperature)

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        # Patch 1: chạy y hệt BaseModel
        return super().forward(
            input_ids_sent=input_ids_sent,
            attention_mask_sent=attention_mask_sent,
            input_ids_term=input_ids_term,
            attention_mask_term=attention_mask_term,
            labels=labels,
            fusion_method=fusion_method,
        )
