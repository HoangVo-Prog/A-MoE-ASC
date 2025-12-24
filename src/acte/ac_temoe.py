# baseline/ac_temoe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from shared import DEVICE


@dataclass
class ACTEConfig:
    num_experts: int = 4
    top_k: int = 2
    top_m_tokens: int = 8
    router_dropout: float = 0.0
    expert_hidden: int = 256
    expert_dropout: float = 0.1
    score_temperature: float = 1.0
    combine_with_base: str = "add"  # add or concat


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        tgt = targets.long()
        logp_t = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
        p_t = p.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
        focal = ((1.0 - p_t) ** self.gamma) * (-logp_t)
        if self.weight is not None:
            w = self.weight.gather(dim=0, index=tgt)
            focal = focal * w
        return focal.mean()


def _parse_class_weights(class_weights: Any, num_labels: int) -> Optional[torch.Tensor]:
    if class_weights is None:
        return None
    if isinstance(class_weights, torch.Tensor):
        return class_weights.float()
    if isinstance(class_weights, (list, tuple)):
        if len(class_weights) != num_labels:
            return None
        return torch.tensor(class_weights, dtype=torch.float)
    if isinstance(class_weights, str):
        s = class_weights.strip()
        if not s:
            return None
        parts = [p.strip() for p in s.split(",")]
        vals = [float(p) for p in parts if p]
        if len(vals) != num_labels:
            return None
        return torch.tensor(vals, dtype=torch.float)
    return None


class TokenExpert(nn.Module):
    def __init__(self, h: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenEvidenceMoE(nn.Module):
    def __init__(self, h: int, cfg: ACTEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_experts = int(cfg.num_experts)
        self.top_k = int(cfg.top_k)

        router_in = h + h + 1  # token, aspect, score
        self.router = nn.Sequential(
            nn.Linear(router_in, h),
            nn.Tanh(),
            nn.Dropout(float(cfg.router_dropout)),
            nn.Linear(h, self.num_experts),
        )

        self.experts = nn.ModuleList(
            [TokenExpert(h=h, hidden=int(cfg.expert_hidden), dropout=float(cfg.expert_dropout)) for _ in range(self.num_experts)]
        )

        self._last_gate: Optional[torch.Tensor] = None  # [B, M, E]
        self._last_topk_idx: Optional[torch.Tensor] = None  # [B, M, K]

    def forward(
        self,
        token_x: torch.Tensor,          # [B, M, H]
        aspect_q: torch.Tensor,         # [B, H]
        token_score: torch.Tensor,      # [B, M]
    ) -> torch.Tensor:
        b, m, h = token_x.shape
        q = aspect_q.unsqueeze(1).expand(b, m, h)
        s = token_score.unsqueeze(-1)
        router_inp = torch.cat([token_x, q, s], dim=-1)

        gate_logits = self.router(router_inp)  # [B, M, E]
        gate = F.softmax(gate_logits, dim=-1)

        k = max(1, min(self.top_k, self.num_experts))
        topk_val, topk_idx = torch.topk(gate, k=k, dim=-1)  # [B, M, K]

        self._last_gate = gate.detach()
        self._last_topk_idx = topk_idx.detach()

        out = torch.zeros_like(token_x)
        for e_i, expert in enumerate(self.experts):
            y_e = expert(token_x)  # [B, M, H]
            mask = (topk_idx == e_i).float()  # [B, M, K]
            w = (mask * topk_val).sum(dim=-1, keepdim=True)  # [B, M, 1]
            out = out + w * y_e
        return out


class ACTokenEvidenceMoEClassifier(nn.Module):
    """
    Aspect Conditioned Token Evidence MoE for ATSC.

    Forward contract must match engine:
      inputs: input_ids_sent, attention_mask_sent, input_ids_term, attention_mask_term, labels, fusion_method
      outputs: {"logits": [B,C], "loss": scalar}
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights: Any = None,
        focal_gamma: float = 2.0,
        acte_cfg: Optional[ACTEConfig] = None,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = int(getattr(self.encoder.config, "hidden_size"))
        self.h = h
        self.num_labels = int(num_labels)

        self.dropout = nn.Dropout(float(dropout))
        self.acte_cfg = acte_cfg or ACTEConfig()

        self.term_pool = "mean"
        self.sent_pool = "cls"

        self.score_proj = nn.Linear(h, h, bias=False)

        self.moe = TokenEvidenceMoE(h=h, cfg=self.acte_cfg)

        base_dim = h
        if self.acte_cfg.combine_with_base == "concat":
            clf_in = h + h
        else:
            clf_in = h

        if str(head_type).lower() == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, clf_in),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(clf_in, self.num_labels),
            )
        else:
            self.classifier = nn.Linear(clf_in, self.num_labels)

        cw = _parse_class_weights(class_weights, self.num_labels)
        if cw is not None:
            cw = cw.to(torch.float)
        self.register_buffer("class_weight", cw if cw is not None else None)

        self.loss_type = str(loss_type).lower().strip()
        self.focal_gamma = float(focal_gamma)
        if self.loss_type == "focal":
            self.focal = FocalLoss(gamma=self.focal_gamma, weight=self.class_weight)
        else:
            self.focal = None

    def _pool_term(self, term_hidden: torch.Tensor, term_mask: torch.Tensor) -> torch.Tensor:
        # term_hidden [B, Lt, H], term_mask [B, Lt]
        m = term_mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (term_hidden * m.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def _pool_sent(self, sent_hidden: torch.Tensor) -> torch.Tensor:
        return sent_hidden[:, 0]  # CLS

    def _topm_select(
        self,
        sent_hidden: torch.Tensor,     # [B, L, H]
        sent_mask: torch.Tensor,       # [B, L]
        score: torch.Tensor,           # [B, L]
        top_m: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, h = sent_hidden.shape
        m = max(1, min(int(top_m), l))
        score_masked = score.masked_fill(sent_mask == 0, float("-inf"))
        topv, topi = torch.topk(score_masked, k=m, dim=-1)  # [B, M]
        idx = topi.unsqueeze(-1).expand(b, m, h)
        tok = torch.gather(sent_hidden, dim=1, index=idx)  # [B, M, H]
        sel_mask = torch.isfinite(topv).float()
        topv = topv.masked_fill(sel_mask == 0, 0.0)
        return tok, topv, topi

    def _evidence_pool(self, tok_after: torch.Tensor, tok_score: torch.Tensor) -> torch.Tensor:
        # tok_after [B,M,H], tok_score [B,M]
        t = float(self.acte_cfg.score_temperature)
        w = F.softmax(tok_score / max(1e-6, t), dim=-1)  # [B,M]
        ev = torch.sum(tok_after * w.unsqueeze(-1), dim=1)  # [B,H]
        return ev

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None:
            return torch.tensor(0.0, device=logits.device)
        y = labels.long()
        if self.loss_type == "weighted_ce":
            return F.cross_entropy(logits, y, weight=self.class_weight)
        if self.loss_type == "focal":
            if self.focal is None:
                return F.cross_entropy(logits, y)
            return self.focal(logits, y)
        return F.cross_entropy(logits, y)

    def forward(
        self,
        *,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ) -> Dict[str, torch.Tensor]:
        sent_out = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        term_out = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)

        sent_hidden = sent_out.last_hidden_state  # [B,L,H]
        term_hidden = term_out.last_hidden_state  # [B,Lt,H]

        sent_vec = self._pool_sent(sent_hidden)  # [B,H]
        term_vec = self._pool_term(term_hidden, attention_mask_term)  # [B,H]
        sent_vec = self.dropout(sent_vec)
        term_vec = self.dropout(term_vec)

        # Aspect conditioned token evidence score
        q = term_vec  # [B,H]
        score = torch.sum(self.score_proj(sent_hidden) * q.unsqueeze(1), dim=-1)  # [B,L]

        tok_x, tok_score, _ = self._topm_select(
            sent_hidden=sent_hidden,
            sent_mask=attention_mask_sent,
            score=score,
            top_m=int(self.acte_cfg.top_m_tokens),
        )

        tok_delta = self.moe(token_x=tok_x, aspect_q=q, token_score=tok_score)
        tok_after = tok_x + tok_delta
        evidence_vec = self._evidence_pool(tok_after, tok_score)
        evidence_vec = self.dropout(evidence_vec)

        # Base fusion vector
        fm = str(fusion_method).lower().strip()
        if fm == "sent":
            base = sent_vec
        elif fm == "term":
            base = term_vec
        elif fm == "add":
            base = sent_vec + term_vec
        elif fm == "mul":
            base = sent_vec * term_vec
        elif fm == "concat":
            base = torch.cat([sent_vec, term_vec], dim=-1)
        else:
            # Fallback: treat other fusion methods as concat at vector level
            base = torch.cat([sent_vec, term_vec], dim=-1)

        # Combine base with evidence
        if base.shape[-1] != self.h:
            base_proj = base
            if base.shape[-1] == 2 * self.h:
                base_proj = base[:, : self.h]
            base = base_proj

        if self.acte_cfg.combine_with_base == "concat":
            feat = torch.cat([base, evidence_vec], dim=-1)
        else:
            feat = base + evidence_vec

        logits = self.classifier(feat)
        loss = self._compute_loss(logits, labels) if labels is not None else None

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if loss is not None:
            out["loss"] = loss
        return out

    def print_moe_debug(self, topn: int = 3) -> None:
        gate = self.moe._last_gate
        idx = self.moe._last_topk_idx
        if gate is None or idx is None:
            print("MoE debug: no cached gate yet")
            return
        # gate [B,M,E]
        g_mean = gate.mean(dim=(0, 1))  # [E]
        vals, ids = torch.topk(g_mean, k=min(int(topn), g_mean.numel()))
        msg = "MoE debug: mean_gate_top = " + ", ".join([f"e{int(i)}:{float(v):.4f}" for v, i in zip(vals, ids)])
        print(msg)
