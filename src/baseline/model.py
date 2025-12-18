import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional


def build_head(head_type: str, in_dim: int, num_labels: int, dropout: float) -> nn.Module:
    head_type = head_type.lower().strip()
    if head_type in {"linear", "lin"}:
        return LinearHead(in_dim, num_labels, dropout)
    if head_type in {"mlp", "2layer", "two_layer"}:
        return MLPHead(in_dim, num_labels, dropout)
    raise ValueError(f"Unsupported head_type: {head_type}. Use 'linear' or 'mlp'.")


class LinearHead(nn.Module):
    """
    Linear head with LayerNorm + Dropout for stability.
    """
    def __init__(self, in_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, dropout: float):
        super().__init__()
        hidden = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class BertConcatClassifier(nn.Module):
    def __init__(
        self, model_name: str, 
        num_labels: int, 
        dropout: float = 0.1,
        head_type: str = "linear",  # "linear" or "mlp"
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = self.encoder.config.hidden_size

        # Cross-domain attention: term (query) attends over sentence (key/value)
        # Choose a num_heads that divides hidden size to avoid runtime error
        _candidates = [8, 4, 2, 1]
        num_heads = next((x for x in _candidates if hidden_size % x == 0), 1)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        
        self.head_type = head_type
        self.head_concat = build_head(head_type, 2 * hidden_size, num_labels, dropout)
        self.head_single = build_head(head_type, hidden_size, num_labels, dropout)

        # ===== Phase 2: Expressive Fusion Extensions =====
        # 1) Gated concat: learn a per-dimension gate between sentence CLS and term CLS
        self.gated_concat_gate = nn.Linear(2 * hidden_size, hidden_size)

        # 2) Bilinear fusion (low-rank factorized)
        # Keep rank modest to control parameter growth
        bilinear_rank = max(32, min(256, hidden_size // 4))
        self.bilinear_rank = bilinear_rank
        self.bilinear_proj_sent = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_proj_term = nn.Linear(hidden_size, bilinear_rank)
        self.bilinear_out = nn.Linear(bilinear_rank, hidden_size)

        # 3) Co-attention (CLS-to-tokens both directions)
        self.coattn_term_to_sent = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.coattn_sent_to_term = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 4) Mixture of fusion experts (lightweight router over several fixed experts)
        # Experts are computed from existing representations, keeping head unchanged.
        self.moe_num_experts = 5  # [sent, term, add, cross, bilinear]
        self.moe_router = nn.Linear(2 * hidden_size, self.moe_num_experts)

    
    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ) -> Dict[str, torch.Tensor]:
        out_sent = self.encoder(input_ids=input_ids_sent, attention_mask=attention_mask_sent)
        cls_sent = out_sent.last_hidden_state[:, 0, :]

        out_term = self.encoder(input_ids=input_ids_term, attention_mask=attention_mask_term)
        cls_term = out_term.last_hidden_state[:, 0, :]

        fusion_method = (fusion_method or "").lower().strip()

        if fusion_method == "sent":
            fused = cls_sent
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "term":
            fused = cls_term
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "concat":
            fused = torch.cat([cls_sent, cls_term], dim=-1)
            logits = self.head_concat(self.dropout(fused))
        elif fusion_method == "add":
            fused = cls_sent + cls_term
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "mul":
            fused = cls_sent * cls_term
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "cross":
            # term embedding as query, sentence token embeddings as key/value
            # query: [B, 1, H], key/value: [B, Ls, H]
            query = out_term.last_hidden_state[:, 0:1, :]
            key = out_sent.last_hidden_state
            value = out_sent.last_hidden_state

            # key_padding_mask expects True for positions that should be ignored (pads)
            key_padding_mask = attention_mask_sent.eq(0)
            attn_out, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
            fused = attn_out.squeeze(1)  # [B, H]
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "gated_concat":
            # Learn a gate g in (0,1) per hidden dimension from [cls_sent; cls_term]
            gate = torch.sigmoid(self.gated_concat_gate(torch.cat([cls_sent, cls_term], dim=-1)))  # [B, H]
            fused = torch.cat([gate * cls_sent, (1.0 - gate) * cls_term], dim=-1)  # [B, 2H]
            logits = self.head_concat(self.dropout(fused))

        elif fusion_method == "bilinear":
            # Low-rank bilinear: (Us*s) ⊙ (Ut*t) -> project back to H
            ps = self.bilinear_proj_sent(cls_sent)  # [B, r]
            pt = self.bilinear_proj_term(cls_term)  # [B, r]
            fused = self.bilinear_out(ps * pt)       # [B, H]
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "coattn":
            # Co-attention with CLS queries both ways (term->sentence and sentence->term)
            sent_tokens = out_sent.last_hidden_state  # [B, Ls, H]
            term_tokens = out_term.last_hidden_state  # [B, Lt, H]

            # term CLS attends sentence tokens
            q_term = term_tokens[:, 0:1, :]  # [B, 1, H]
            kpm_sent = attention_mask_sent.eq(0)
            term_ctx, _ = self.coattn_term_to_sent(q_term, sent_tokens, sent_tokens, key_padding_mask=kpm_sent)
            term_ctx = term_ctx.squeeze(1)  # [B, H]

            # sentence CLS attends term tokens
            q_sent = sent_tokens[:, 0:1, :]  # [B, 1, H]
            kpm_term = attention_mask_term.eq(0)
            sent_ctx, _ = self.coattn_sent_to_term(q_sent, term_tokens, term_tokens, key_padding_mask=kpm_term)
            sent_ctx = sent_ctx.squeeze(1)  # [B, H]

            fused = torch.cat([sent_ctx, term_ctx], dim=-1)  # [B, 2H]
            logits = self.head_concat(self.dropout(fused))

        elif fusion_method == "late_interaction":
            # Token-level late interaction without adding trainable parameters.
            # Build an interaction-aware sentence vector using max-sim pooling over term tokens.
            sent_tokens = out_sent.last_hidden_state  # [B, Ls, H]
            term_tokens = out_term.last_hidden_state  # [B, Lt, H]

            # Masks
            sent_mask = attention_mask_sent.bool()  # [B, Ls]
            term_mask = attention_mask_term.bool()  # [B, Lt]

            # Normalize for stable dot products
            sent_norm = torch.nn.functional.normalize(sent_tokens, p=2, dim=-1)
            term_norm = torch.nn.functional.normalize(term_tokens, p=2, dim=-1)

            # Similarity [B, Lt, Ls]
            sim = torch.matmul(term_norm, sent_norm.transpose(1, 2))

            # Mask out padded positions
            sim = sim.masked_fill(~sent_mask[:, None, :], float("-inf"))
            sim = sim.masked_fill(~term_mask[:, :, None], float("-inf"))

            # For each sentence token, take max over term tokens -> [B, Ls]
            s_score = sim.max(dim=1).values
            s_score = s_score.masked_fill(~sent_mask, float("-inf"))

            # Attention weights over sentence tokens
            attn_w = torch.softmax(s_score, dim=-1)  # [B, Ls]
            attn_w = attn_w.masked_fill(~sent_mask, 0.0)

            fused = torch.sum(sent_tokens * attn_w.unsqueeze(-1), dim=1)  # [B, H]
            logits = self.head_single(self.dropout(fused))

        elif fusion_method == "moe":
            # Mixture of simple experts with a learned router, keeping output dimension H.
            # Experts: sent CLS, term CLS, add, cross-attn, bilinear
            # Router input: [cls_sent; cls_term]
            router_in = torch.cat([cls_sent, cls_term], dim=-1)  # [B, 2H]
            alpha = torch.softmax(self.moe_router(router_in), dim=-1)  # [B, K]

            # Expert 0: sent
            e0 = cls_sent
            # Expert 1: term
            e1 = cls_term
            # Expert 2: add
            e2 = cls_sent + cls_term

            # Expert 3: cross-attn (term CLS attends sentence tokens)
            query = out_term.last_hidden_state[:, 0:1, :]
            key = out_sent.last_hidden_state
            value = out_sent.last_hidden_state
            key_padding_mask = attention_mask_sent.eq(0)
            e3, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
            e3 = e3.squeeze(1)  # [B, H]

            # Expert 4: bilinear
            ps = self.bilinear_proj_sent(cls_sent)
            pt = self.bilinear_proj_term(cls_term)
            e4 = self.bilinear_out(ps * pt)

            experts = torch.stack([e0, e1, e2, e3, e4], dim=1)  # [B, K, H]
            fused = torch.sum(experts * alpha.unsqueeze(-1), dim=1)  # [B, H]
            logits = self.head_single(self.dropout(fused))

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
