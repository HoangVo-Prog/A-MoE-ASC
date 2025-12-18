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
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
