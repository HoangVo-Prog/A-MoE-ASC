from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.core.loss.focal_loss import FocalLoss


class SDMoEDirModel(BaseModel):
    """SDMoEDirModel: MoE Semantic Deformation as mixture of directions on fusion vector.

    Ý tưởng chính (so với SDModel):
    - Gate chỉ phụ thuộc aspect vector t (CLS của term), soft routing, không top-k.
    - Expert apply trực tiếp lên z0 = concat(t, s) ở level vector fusion.
    - Mỗi expert là một hướng low rank: A[k] in R^{2d x r}, B[k] in R^{r x 2d}.
    - Init B = 0 để delta = 0 lúc bắt đầu, model khởi đầu giống baseline concat.

    Output dict tương thích engine hiện tại.
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
        # SDMoEDir params
        num_experts: int = 4,
        sd_rank: int = 8,
        sd_alpha: float = 4.0,
        sd_lambda_bal: float = 0.02,
        sd_lambda_div: float = 1e-4,
        router_temperature: float = 1.0,
        router_hidden_mult: float = 1.0,
        router_bias: bool = True,
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

        self.num_experts = int(num_experts)
        self.sd_rank = int(sd_rank)
        self.sd_alpha = float(sd_alpha)
        self.sd_lambda_bal = float(sd_lambda_bal)
        self.sd_lambda_div = float(sd_lambda_div)
        self.router_temperature = float(router_temperature)
        self.router_bias = bool(router_bias)

        # Engine bật nhánh moe nếu attribute này truthy
        self._collect_aux_loss = True

        # Hard-freeze encoder để giữ tinh thần PEFT kiểu deformation
        self._freeze_encoder_hard()

        d = int(self.encoder.config.hidden_size)
        router_hidden = max(1, int(d * float(router_hidden_mult)))

        # GateNet(t) -> [B, K]
        self.gate_net = nn.Sequential(
            nn.Linear(d, router_hidden, bias=self.router_bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, self.num_experts, bias=self.router_bias),
        )

        # Expert directions on fusion space 2d
        # A: [K, 2d, r] random init (nhỏ)
        a = torch.empty(self.num_experts, 2 * d, self.sd_rank, dtype=torch.float32)
        nn.init.normal_(a, mean=0.0, std=0.02)
        self.A = nn.Parameter(a)

        # B: [K, r, 2d] zero init để delta=0
        self.B = nn.Parameter(torch.zeros(self.num_experts, self.sd_rank, 2 * d, dtype=torch.float32))

        # cache cho debug
        self.last_gate = None
        self.last_router_logits = None

    def _freeze_encoder_hard(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()
        return self

    def _compute_gate(self, *, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.gate_net(t)  # [B, K]
        tau = float(self.router_temperature) if self.router_temperature is not None else 1.0
        g = torch.softmax(u / tau, dim=-1)
        return u, g

    def _apply_directions(self, *, z0: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """z = z0 + sum_k g_k * delta_k(z0), delta_k = (z0 A_k) B_k."""
        # z0: [B, 2d]
        # zA: [B, K, r]
        zA = torch.einsum("bd,kdr->bkr", z0, self.A)
        # zAB: [B, K, 2d]
        zAB = torch.einsum("bkr,krd->bkd", zA, self.B)
        # mixture: [B, 2d]
        delta = torch.einsum("bk,bkd->bd", g, zAB)
        scale = float(self.sd_alpha) / float(self.sd_rank)
        return z0 + (scale * delta)

    def _loss_balancing(self, g: torch.Tensor) -> torch.Tensor:
        # g: [B, K]
        p = g.mean(dim=0)  # [K]
        target = 1.0 / float(self.num_experts)
        return ((p - target) ** 2).mean()

    def _loss_diversity(self) -> torch.Tensor:
        """Cosine off-diagonal penalty giữa các hướng B.

        B được flatten: [K, r*2d], normalize theo L2, rồi penalize cosine similarity off-diagonal.
        """
        K = self.num_experts
        Bf = self.B.reshape(K, -1)
        Bn = F.normalize(Bf, p=2, dim=-1, eps=1e-12)
        C = Bn @ Bn.t()  # [K, K]
        C_off = C - torch.diag(torch.diag(C))
        return (C_off ** 2).mean()

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        # 1) Encode (frozen)
        self.encoder.eval()
        with torch.no_grad():
            out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent, return_dict=True)
            out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term, return_dict=True)

        Hs = out_sent.last_hidden_state  # [B, Ls, d]
        Ht = out_term.last_hidden_state  # [B, Lt, d]

        s = Hs[:, 0, :]  # [B, d]
        t = Ht[:, 0, :]  # [B, d]

        # 2) Base representation
        z0 = torch.cat([t, s], dim=-1)  # [B, 2d]

        # 3) Gate (aspect only)
        u, g = self._compute_gate(t=t)
        self.last_router_logits = u.detach()
        self.last_gate = g.detach()

        # 4) Mixture of directions on z0
        z = self._apply_directions(z0=z0, g=g)  # [B, 2d]

        # 5) Classification head (giữ đúng head concat để fair)
        logits = self.head_concat(self.dropout(z))  # [B, C]

        if labels is None:
            return {"loss": None, "logits": logits}

        # 6) Main loss
        if self.loss_type == "ce":
            loss_main = F.cross_entropy(logits, labels)

        elif self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_main = F.cross_entropy(logits, labels, weight=w)

        elif self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            loss_main = loss_fn(logits, labels)

        else:
            raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")

        # 7) Aux loss
        loss_bal = self._loss_balancing(g)
        loss_div = self._loss_diversity()
        aux_loss = self.sd_lambda_bal * loss_bal + self.sd_lambda_div * loss_div

        loss_total = loss_main + aux_loss

        # 8) Debug stats (soft usage)
        eps = 1e-12
        gate_entropy_mean = (-(g.clamp_min(eps) * g.clamp_min(eps).log()).sum(dim=-1)).mean()
        p_k = g.mean(dim=0).detach()

        return {
            "loss": loss_total,
            "logits": logits,
            "loss_main": loss_main,
            "aux_loss": aux_loss,
            "loss_lambda": aux_loss,
            "loss_bal": loss_bal.detach(),
            "loss_div": loss_div.detach(),
            "gate_entropy_mean": gate_entropy_mean.detach(),
            "p_k": p_k,
            "gate": g.detach(),
        }
