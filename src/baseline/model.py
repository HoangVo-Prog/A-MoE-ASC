import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional

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
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        h = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.head_concat = MLPHead(2 * h, num_labels, dropout)
        self.head_single = MLPHead(h, num_labels, dropout)

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

        if fusion_method == "concat":
            fused = torch.cat([cls_sent, cls_term], dim=-1)
            logits = self.head_concat(self.dropout(fused))
        elif fusion_method == "add":
            fused = cls_sent + cls_term
            logits = self.head_single(self.dropout(fused))
        elif fusion_method == "mul":
            fused = cls_sent * cls_term
            logits = self.head_single(self.dropout(fused))
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
