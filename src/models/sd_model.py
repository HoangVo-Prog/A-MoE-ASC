from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.core.loss.focal_loss import FocalLoss


class SDModel(BaseModel):
    """
    SDModel: MoE Semantic Deformation (dual-encode, CLS pooling)

    Invariant:
    - Freeze toàn bộ BERT encoder cứng ở model-level
    - Gate per-instance: g in R^{B x K}
    - Expert bank low-rank: A[K,d,r], B[K,r,d]
    - Deform sentence tokens H_sent và aspect vector t (CLS term)
    - Fusion baseline giữ nguyên, chỉ thay tensor đầu vào đúng mapping
    - Output dict tương thích MoEHead để engine log nhánh moe không gãy
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
        # SD params
        num_experts: int = 4,
        sd_rank: int = 8,
        sd_alpha: float = 16.0,
        sd_lambda_bal: float = 0.01,
        sd_lambda_div: float = 0.001,
        router_temperature: float = 1.0,
        router_bias: bool = True,
        router_hidden_mult: float = 1.0,
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

        # Store SD hyperparams
        self.num_experts = int(num_experts)
        self.sd_rank = int(sd_rank)
        self.sd_alpha = float(sd_alpha)
        self.sd_lambda_bal = float(sd_lambda_bal)
        self.sd_lambda_div = float(sd_lambda_div)
        self.router_temperature = float(router_temperature)
        self.router_bias = bool(router_bias)

        # Hard-freeze BERT encoder
        self._freeze_encoder_hard()

        # Router and expert bank
        d = int(self.encoder.config.hidden_size)
        router_hidden = max(1, int(d * float(router_hidden_mult)))

        self.router = nn.Sequential(
            nn.Linear(2 * d, router_hidden, bias=self.router_bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, self.num_experts, bias=self.router_bias),
        )

        # A: [K, d, r] zeros init
        self.A = nn.Parameter(torch.zeros(self.num_experts, d, self.sd_rank))

        # B: [K, r, d] small random init
        b = torch.empty(self.num_experts, self.sd_rank, d, dtype=torch.float32)
        nn.init.normal_(b, mean=0.0, std=0.02)
        self.B = nn.Parameter(b.to(dtype=self.A.dtype))

    # Engine checks hasattr(model, "_collect_aux_loss") to enable MoE logging
    def _collect_aux_loss(self) -> bool:
        return True

    def _freeze_encoder_hard(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()
        return self

    def _compute_gate(self, *, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, s], dim=-1)  # [B, 2d]
        u = self.router(x)             # [B, K]
        tau = float(self.router_temperature) if self.router_temperature is not None else 1.0
        g = torch.softmax(u / tau, dim=-1)  # [B, K]
        return g

    def _deform_sentence_tokens(self, *, H_sent: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # H_sent: [B, Ls, d], g: [B, K]
        # HB = einsum("bld,krd->blkr", H_sent, B) -> [B, Ls, K, r]
        HB = torch.einsum("bld,krd->blkr", H_sent, self.B)
        # delta = einsum("blkr,kdr->blkd", HB, A) -> [B, Ls, K, d]
        delta = torch.einsum("blkr,kdr->blkd", HB, self.A)
        # weight gate and sum over experts
        g_exp = g[:, None, :, None]  # [B, 1, K, 1]
        delta_w = (g_exp * delta).sum(dim=2)  # [B, Ls, d]
        scale = float(self.sd_alpha) / float(self.sd_rank)
        return H_sent + (scale * delta_w)

    def _deform_aspect_vec(self, *, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # t: [B, d], g: [B, K]
        # tB = einsum("bd,krd->bkr") -> [B, K, r]
        tB = torch.einsum("bd,krd->bkr", t, self.B)
        # tAB = einsum("bkr,kdr->bkd") -> [B, K, d]
        tAB = torch.einsum("bkr,kdr->bkd", tB, self.A)
        # weight gate and sum
        g_term = g[:, :, None]  # [B, K, 1]
        t_delta = (g_term * tAB).sum(dim=1)  # [B, d]
        scale = float(self.sd_alpha) / float(self.sd_rank)
        return t + (scale * t_delta)

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

        H_sent = out_sent.last_hidden_state  # [B, Ls, d]
        H_term = out_term.last_hidden_state  # [B, Lt, d]

        s = H_sent[:, 0, :]  # [B, d]
        t = H_term[:, 0, :]  # [B, d]

        # 2) Gate
        g = self._compute_gate(t=t, s=s)  # [B, K]

        # 3) Deform
        H_sent_tilde = self._deform_sentence_tokens(H_sent=H_sent, g=g)  # [B, Ls, d]
        t_tilde = self._deform_aspect_vec(t=t, g=g)                      # [B, d]

        cls_sent_tilde = H_sent_tilde[:, 0, :]  # [B, d]
        cls_term_tilde = t_tilde                # [B, d]

        fm = fusion_method.lower().strip()

        # 4) Fusion baseline mapping (match BaseModel, replace tensors)
        if fm == "sent":
            logits = self.head_single(self.dropout(cls_sent_tilde))

        elif fm == "term":
            logits = self.head_single(self.dropout(cls_term_tilde))

        elif fm == "concat":
            logits = self.head_concat(self.dropout(torch.cat([cls_sent_tilde, cls_term_tilde], dim=-1)))

        elif fm == "add":
            logits = self.head_single(self.dropout(cls_sent_tilde + cls_term_tilde))

        elif fm == "mul":
            logits = self.head_single(self.dropout(cls_sent_tilde * cls_term_tilde))

        elif fm == "cross":
            q = cls_term_tilde.unsqueeze(1)  # [B, 1, d]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(q, H_sent_tilde, H_sent_tilde, key_padding_mask=kpm)
            logits = self.head_single(self.dropout(attn_out.squeeze(1)))

        elif fm == "gated_concat":
            gate_val = torch.sigmoid(self.gate(torch.cat([cls_sent_tilde, cls_term_tilde], dim=-1)))
            fused = gate_val * cls_sent_tilde + (1 - gate_val) * cls_term_tilde
            logits = self.head_single(self.dropout(fused))

        elif fm == "bilinear":
            fused = self.bilinear_out(
                self.bilinear_proj_sent(cls_sent_tilde) * self.bilinear_proj_term(cls_term_tilde)
            )
            logits = self.head_single(self.dropout(fused))

        elif fm == "coattn":
            # Sentence tokens deform, term tokens keep original H_term, CLS term replaced by t_tilde
            q_term = cls_term_tilde.unsqueeze(1)  # [B, 1, d]
            q_sent = cls_sent_tilde.unsqueeze(1)  # [B, 1, d]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term_ctx, _ = self.coattn_term_to_sent(q_term, H_sent_tilde, H_sent_tilde, key_padding_mask=kpm_sent)
            sent_ctx, _ = self.coattn_sent_to_term(q_sent, H_term, H_term, key_padding_mask=kpm_term)

            logits = self.head_single(self.dropout(term_ctx.squeeze(1) + sent_ctx.squeeze(1)))

        elif fm == "late_interaction":
            # Sentence tokens deform, term tokens keep original
            sent_tok = F.normalize(H_sent_tilde, p=2, dim=-1)
            term_tok = F.normalize(H_term, p=2, dim=-1)

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

        # 5) If no labels, inference only
        if labels is None:
            return {"loss": None, "logits": logits}

        # 6) Main loss (match BaseModel)
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

        # 7) Aux loss (SD)
        loss_bal = self._loss_balancing(g)
        loss_div = self._loss_diversity()
        aux_loss = (self.sd_lambda_bal * loss_bal) + (self.sd_lambda_div * loss_div)

        loss_total = loss_main + aux_loss

        # 8) Debug stats
        eps = 1e-12
        gate_entropy_mean = (-(g.clamp_min(eps) * g.clamp_min(eps).log()).sum(dim=-1)).mean()
        p_k = g.mean(dim=0).detach()

        # 9) Output dict compatible with engine MoE logging
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
            
    def print_moe_debug(self, topn: int = 3, bottomn: int = 3, eps_dead: float = 1e-6):
        stats = self._moe_debug_stats()
        if stats is None:
            print("[MoE] No stats yet.")
            return
        
        print("\n[MoE Debug - Single Head]")

        uh = stats["usage_hard"].float()
        us = stats["usage_soft"].float()

        dead_h0 = int((uh == 0).sum().item())
        dead_h = int((uh < eps_dead).sum().item())
        dead_s0 = int((us == 0).sum().item())
        dead_s = int((us < eps_dead).sum().item())

        topk = min(topn, uh.numel())
        botk = min(bottomn, uh.numel())

        topv_h, topi_h = torch.topk(uh, k=topk, largest=True)
        botv_h, boti_h = torch.topk(uh, k=botk, largest=False)
        topv_s, topi_s = torch.topk(us, k=topk, largest=True)
        botv_s, boti_s = torch.topk(us, k=botk, largest=False)

        top_pairs_h = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv_h, topi_h)])
        bot_pairs_h = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv_h, boti_h)])
        top_pairs_s = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(topv_s, topi_s)])
        bot_pairs_s = " ".join([f"e{int(i)}={float(v):.6f}" for v, i in zip(botv_s, boti_s)])

        imb_h = float(stats["max_hard"] / (stats["min_hard"] + 1e-12))
        imb_s = float(stats["max_soft"] / (stats["min_soft"] + 1e-12))

        print(
            f"MoE Head | "
            f"H_full={stats['H_full_norm']:.6f} H_soft={stats['H_soft_norm']:.6f} H_hard={stats['H_hard_norm']:.6f} | "
            f"logits_std={stats['logits_std']:.6f} maxabs={stats['logits_maxabs']:.6f} gap12={stats['gap_top1_top2']:.6f}"
        )
        print(
            f"  HARD: min={stats['min_hard']:.6f} max={stats['max_hard']:.6f} cv={stats['cv_hard']:.3f} imb={imb_h:.2f} "
            f"dead(==0)={dead_h0} dead(<{eps_dead:g})={dead_h}"
        )
        print(f"    top: {top_pairs_h}")
        print(f"    bot: {bot_pairs_h}")
        print(
            f"  SOFT: min={stats['min_soft']:.6f} max={stats['max_soft']:.6f} cv={stats['cv_soft']:.3f} imb={imb_s:.2f} "
            f"dead(==0)={dead_s0} dead(<{eps_dead:g})={dead_s}"
        )
        print(f"    top: {top_pairs_s}")
        print(f"    bot: {bot_pairs_s}")
        print()

