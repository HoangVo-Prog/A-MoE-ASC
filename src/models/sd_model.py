from __future__ import annotations

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from src.core.loss.focal_loss import FocalLoss
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
        
        # ===== Patch 2: freeze cứng encoder =====
        self._freeze_encoder_hard()
        
        # ===== Patch 3: Router + Expert bank (A,B) =====
        hidden_size = self.encoder.config.hidden_size
        router_hidden = hidden_size  # đơn giản, ổn định benchmark

        self.router = nn.Sequential(
            nn.Linear(2 * hidden_size, router_hidden, bias=bool(getattr(self, "router_bias", True))),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, self.num_experts, bias=bool(getattr(self, "router_bias", True))),
        )

        # A: [K, d, r] zeros init
        self.A = nn.Parameter(torch.zeros(self.num_experts, hidden_size, self.sd_rank))

        # B: [K, r, d] small random init
        b = torch.empty(self.num_experts, self.sd_rank, hidden_size)
        nn.init.normal_(b, mean=0.0, std=0.02)
        self.B = nn.Parameter(b)


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

    def _freeze_encoder_hard(self) -> None:
        # 1) tắt grad
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2) luôn eval để tắt dropout của BERT
        self.encoder.eval()
        
    def train(self, mode: bool = True):
        # gọi train cho toàn model để head, router, fusion vẫn train đúng
        super().train(mode)

        # nhưng encoder luôn eval
        if hasattr(self, "encoder"):
            self.encoder.eval()
        return self

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels=None,
        fusion_method: str = "concat",
    ):
        # ===== 1) Encode (frozen) =====
        self.encoder.eval()
        with torch.no_grad():
            out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
            out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)

        H_sent = out_sent.last_hidden_state  # [B, Ls, d]
        H_term = out_term.last_hidden_state  # [B, Lt, d]

        s = H_sent[:, 0, :]  # [B, d]
        t = H_term[:, 0, :]  # [B, d]

        # ===== 2) Router gate =====
        g = self._compute_gate(t=t, s=s)  # [B, K]

        # ===== 3) Deform H_sent and t =====
        H_sent_tilde = self._deform_sentence_tokens(H_sent=H_sent, g=g)  # [B, Ls, d]
        t_tilde = self._deform_aspect_vec(t=t, g=g)  # [B, d]

        cls_sent_tilde = H_sent_tilde[:, 0, :]  # [B, d]
        cls_term_tilde = t_tilde                # [B, d]

        fusion_method = fusion_method.lower().strip()

        # ===== 4) Fusion baseline mapping (copy from BaseModel, replace tensors) =====
        if fusion_method == "sent":
            logits = self.head_single(self.dropout(cls_sent_tilde))

        elif fusion_method == "term":
            logits = self.head_single(self.dropout(cls_term_tilde))

        elif fusion_method == "concat":
            logits = self.head_concat(self.dropout(torch.cat([cls_sent_tilde, cls_term_tilde], dim=-1)))

        elif fusion_method == "add":
            logits = self.head_single(self.dropout(cls_sent_tilde + cls_term_tilde))

        elif fusion_method == "mul":
            logits = self.head_single(self.dropout(cls_sent_tilde * cls_term_tilde))

        elif fusion_method == "cross":
            q = cls_term_tilde.unsqueeze(1)  # [B, 1, d]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(q, H_sent_tilde, H_sent_tilde, key_padding_mask=kpm)
            logits = self.head_single(self.dropout(attn_out.squeeze(1)))

        elif fusion_method == "gated_concat":
            gate_inp = torch.cat([cls_sent_tilde, cls_term_tilde], dim=-1)
            g_sig = torch.sigmoid(self.gate(gate_inp))
            fused = g_sig * cls_sent_tilde + (1 - g_sig) * cls_term_tilde
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "bilinear":
            fused = torch.tanh(
                self.bilinear_proj_sent(cls_sent_tilde) * self.bilinear_proj_term(cls_term_tilde)
            )
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "coattn":
            # Chốt theo bạn: sentence tokens deform, term tokens giữ nguyên, CLS term thay bằng t_tilde
            q_term = cls_term_tilde.unsqueeze(1)  # [B, 1, d]
            q_sent = cls_sent_tilde.unsqueeze(1)  # [B, 1, d]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term_ctx, _ = self.coattn_term_to_sent(q_term, H_sent_tilde, H_sent_tilde, key_padding_mask=kpm_sent)
            sent_ctx, _ = self.coattn_sent_to_term(q_sent, H_term, H_term, key_padding_mask=kpm_term)

            logits = self.head_single(self.dropout(term_ctx.squeeze(1) + sent_ctx.squeeze(1)))

        elif fusion_method == "late_interaction":
            # sentence tokens deform, term tokens giữ nguyên
            sent_tok = H_sent_tilde
            term_tok = H_term

            sent_tok = F.normalize(sent_tok, p=2, dim=-1)
            term_tok = F.normalize(term_tok, p=2, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))  # [B, Lt, Ls]

            if attention_mask_sent is not None:
                mask = attention_mask_sent.unsqueeze(1).eq(0)  # [B, 1, Ls]
                sim = sim.masked_fill(mask, -1e9)

            max_sim = sim.max(dim=-1).values  # [B, Lt]

            if attention_mask_term is not None:
                term_valid = attention_mask_term.float()
                denom = term_valid.sum(dim=1).clamp_min(1.0)
                pooled = (max_sim * term_valid).sum(dim=1) / denom  # [B]
            else:
                pooled = max_sim.mean(dim=1)

            cond = self.gate(torch.cat([cls_sent_tilde, cls_term_tilde], dim=-1))  # [B, d]
            fused = cond * pooled.unsqueeze(-1)
            logits = self.head_single(self.dropout(fused))

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        # ===== 5) Loss (y hệt BaseModel) =====
        if labels is None:
            return {"loss": None, "logits": logits}

        if self.loss_type == "ce":
            loss = F.cross_entropy(logits, labels)

        elif self.loss_type == "weighted_ce":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, labels, weight=w)

        elif self.loss_type == "focal":
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=w, reduction="mean")
            loss = loss_fn(logits, labels)

        else:
            raise RuntimeError(f"Unexpected loss_type: {self.loss_type}")

        # Patch 3 chưa thêm aux loss và dict MoE, sẽ làm ở Patch 4
                # ===== 5) Loss main (giữ y hệt BaseModel) =====
        if labels is None:
            return {"loss": None, "logits": logits}

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

        # ===== 6) Aux losses (SD) =====
        loss_bal = self._loss_balancing(g)
        loss_div = self._loss_diversity()

        aux_loss = (self.sd_lambda_bal * loss_bal) + (self.sd_lambda_div * loss_div)
        loss_total = loss_main + aux_loss

        # ===== 7) Optional debug stats =====
        eps = 1e-12
        gate_entropy_mean = (-(g.clamp_min(eps) * g.clamp_min(eps).log()).sum(dim=-1)).mean()
        p_k = g.mean(dim=0).detach()

        # ===== 8) Output dict compatible with engine MoE logging =====
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
        }


        
    def _compute_gate(self, *, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # t, s: [B, d]
        x = torch.cat([t, s], dim=-1)  # [B, 2d]
        u = self.router(x)  # [B, K]
        tau = float(self.router_temperature) if self.router_temperature is not None else 1.0
        g = torch.softmax(u / tau, dim=-1)  # [B, K]
        return g

    def _deform_sentence_tokens(
        self,
        *,
        H_sent: torch.Tensor,  # [B, Ls, d]
        g: torch.Tensor,       # [B, K]
    ) -> torch.Tensor:
        # HB = einsum("bld,krd->blkr", H_sent, B)  -> [B, Ls, K, r]
        HB = torch.einsum("bld,krd->blkr", H_sent, self.B)

        # delta = einsum("blkr,kdr->blkd", HB, A)  -> [B, Ls, K, d]
        delta = torch.einsum("blkr,kdr->blkd", HB, self.A)

        # weight gate and sum over K
        g_exp = g[:, None, :, None]  # [B, 1, K, 1]
        delta_w = (g_exp * delta).sum(dim=2)  # [B, Ls, d]

        scale = float(self.sd_alpha) / float(self.sd_rank)
        return H_sent + (scale * delta_w)

    def _deform_aspect_vec(
        self,
        *,
        t: torch.Tensor,  # [B, d]
        g: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        # tB = einsum("bd,krd->bkr") -> [B, K, r]
        tB = torch.einsum("bd,krd->bkr", t, self.B)

        # tAB = einsum("bkr,kdr->bkd") -> [B, K, d]
        tAB = torch.einsum("bkr,kdr->bkd", tB, self.A)

        g_term = g[:, :, None]  # [B, K, 1]
        t_delta = (g_term * tAB).sum(dim=1)  # [B, d]

        scale = float(self.sd_alpha) / float(self.sd_rank)
        return t + (scale * t_delta)
    
    def _collect_aux_loss(self) -> bool:
        # Engine dùng hasattr(model, "_collect_aux_loss") để bật nhánh MoE logging
        return True

    def _loss_balancing(self, g: torch.Tensor) -> torch.Tensor:
        # g: [B, K]
        p = g.mean(dim=0)  # [K]
        target = 1.0 / float(self.num_experts)
        return ((p - target) ** 2).sum()

    def _loss_diversity(self) -> torch.Tensor:
        # A: [K, d, r]
        K = self.num_experts
        A_flat = self.A.reshape(K, -1)  # [K, d*r]
        M = A_flat @ A_flat.t()         # [K, K]
        M_off = M - torch.diag(torch.diag(M))
        return (M_off ** 2).sum()


