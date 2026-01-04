from __future__ import annotations


import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import hdbscan

from src.models import BaseModel
from src.core.utils.helper import get_dataloader, get_kfold_dataset, get_tokenizer

def test_representation_clusterability_hdbscan_cpu(
    dataloader: torch.utils.data.DataLoader,
    *,
    model_name: str,
    num_labels: int,
    num_experts: int = 8,
    dropout: float = 0.1,
    head_type: str = "linear",
    loss_type: str = "ce",
    class_weights: Optional[str] = None,
    focal_gamma: float = 2.0,
    fusion_method: str = "concat",
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    use_standardize: bool = True,
    hdb_min_cluster_size: Optional[int] = None,
    hdb_min_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Test hướng 1: kiểm tra clusterability của vector fused đúng theo fusion_method
    (chính là input mà model dùng cho head).

    Batch requirement: mỗi batch từ dataloader cần có các key:
      - input_ids_sent, attention_mask_sent
      - input_ids_term, attention_mask_term
      - labels (optional, nhưng khuyến nghị có)

    fusion_method options (bám BaseModel.forward):
      - sent, term, concat, add, mul
      - cross, gated_concat, bilinear, coattn, late_interaction
    """

    # ------------------------
    # Utils
    # ------------------------
    def _set_seed(s: int) -> None:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _pick_device(d: Optional[str]) -> str:
        return d if d is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    def _safe_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    def _cluster_size_stats(labels: np.ndarray) -> Dict[str, Any]:
        uniq, counts = np.unique(labels, return_counts=True)
        return {
            "n_clusters_including_noise": int(len(uniq)),
            "sizes": {int(u): int(c) for u, c in zip(uniq, counts)},
            "min_size": int(counts.min()),
            "max_size": int(counts.max()),
            "mean_size": float(counts.mean()),
            "std_size": float(counts.std()),
        }

    def _silhouette_safe(Xv: np.ndarray, lv: np.ndarray) -> Optional[float]:
        uniq = np.unique(lv)
        if len(uniq) < 2:
            return None
        _, counts = np.unique(lv, return_counts=True)
        if np.any(counts < 2):
            return None
        return float(silhouette_score(Xv, lv))

    def _db_safe(Xv: np.ndarray, lv: np.ndarray) -> Optional[float]:
        if len(np.unique(lv)) < 2:
            return None
        return float(davies_bouldin_score(Xv, lv))

    def _ch_safe(Xv: np.ndarray, lv: np.ndarray) -> Optional[float]:
        if len(np.unique(lv)) < 2:
            return None
        return float(calinski_harabasz_score(Xv, lv))

    # ------------------------
    # Fusion mirror (pre-head)
    # ------------------------
    def _get_fused_repr_from_fusion_method(
        model,
        *,
        out_sent,
        out_term,
        attention_mask_sent: torch.Tensor,
        attention_mask_term: torch.Tensor,
        fusion_method: str,
    ) -> torch.Tensor:

        cls_sent = out_sent.last_hidden_state[:, 0, :]
        cls_term = out_term.last_hidden_state[:, 0, :]
        fm = fusion_method.lower().strip()

        if fm == "sent":
            return cls_sent
        if fm == "term":
            return cls_term
        if fm == "concat":
            return torch.cat([cls_sent, cls_term], dim=-1)
        if fm == "add":
            return cls_sent + cls_term
        if fm == "mul":
            return cls_sent * cls_term

        if fm == "cross":
            q = out_term.last_hidden_state[:, :1, :]
            kpm = attention_mask_sent.eq(0)
            attn_out, _ = model.cross_attn(
                q, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm
            )
            return attn_out.squeeze(1)

        if fm == "gated_concat":
            g = torch.sigmoid(model.gate(torch.cat([cls_sent, cls_term], dim=-1)))
            return g * cls_sent + (1 - g) * cls_term

        if fm == "bilinear":
            return model.bilinear_out(
                model.bilinear_proj_sent(cls_sent) * model.bilinear_proj_term(cls_term)
            )

        if fm == "coattn":
            q_term = out_term.last_hidden_state[:, :1, :]
            q_sent = out_sent.last_hidden_state[:, :1, :]
            kpm_sent = attention_mask_sent.eq(0)
            kpm_term = attention_mask_term.eq(0)

            term_ctx, _ = model.coattn_term_to_sent(
                q_term, out_sent.last_hidden_state, out_sent.last_hidden_state, key_padding_mask=kpm_sent
            )
            sent_ctx, _ = model.coattn_sent_to_term(
                q_sent, out_term.last_hidden_state, out_term.last_hidden_state, key_padding_mask=kpm_term
            )
            return term_ctx.squeeze(1) + sent_ctx.squeeze(1)

        if fm == "late_interaction":
            sent_tok = torch.nn.functional.normalize(out_sent.last_hidden_state, dim=-1)
            term_tok = torch.nn.functional.normalize(out_term.last_hidden_state, dim=-1)

            sim = torch.matmul(term_tok, sent_tok.transpose(1, 2))
            sim = sim.masked_fill(attention_mask_sent.unsqueeze(1).eq(0), -1e9)
            max_sim = sim.max(dim=-1).values

            term_valid = attention_mask_term.float()
            pooled = (max_sim * term_valid).sum(dim=1) / term_valid.sum(dim=1).clamp_min(1.0)

            cond = model.gate(torch.cat([cls_sent, cls_term], dim=-1))
            return cond * pooled.unsqueeze(-1)

        raise ValueError(f"Unsupported fusion_method: {fusion_method}")

    # ------------------------
    # Main
    # ------------------------
    _set_seed(seed)
    device = _pick_device(device)

    model = BaseModel(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        head_type=head_type,
        loss_type=loss_type,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
    ).to(device)
    model.eval()

    reps, ys = [], []

    with torch.no_grad():
        for bidx, batch in enumerate(dataloader):
            if max_batches is not None and bidx >= max_batches:
                break

            def _get(k): 
                return batch[k].to(device)

            out_sent = model.encoder(_get("input_ids_sent"), _get("attention_mask_sent"))
            out_term = model.encoder(_get("input_ids_term"), _get("attention_mask_term"))

            rep = _get_fused_repr_from_fusion_method(
                model,
                out_sent=out_sent,
                out_term=out_term,
                attention_mask_sent=_get("attention_mask_sent"),
                attention_mask_term=_get("attention_mask_term"),
                fusion_method=fusion_method,
            )

            reps.append(_to_numpy(rep))
            if "labels" in batch:
                ys.append(_to_numpy(batch["labels"]).astype(np.int64))

    X = np.concatenate(reps)
    y = np.concatenate(ys) if ys else None

    if max_samples is not None:
        X = X[:max_samples]
        if y is not None:
            y = y[:max_samples]

    if use_standardize:
        X = StandardScaler().fit_transform(X)

    # ------------------------
    # Clustering
    # ------------------------
    if hdb_min_cluster_size is None:
        hdb_min_cluster_size = max(10, X.shape[0] // (2 * num_experts))
    if hdb_min_samples is None:
        hdb_min_samples = max(1, hdb_min_cluster_size // 2)

    hdb = hdbscan.HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        min_samples=hdb_min_samples,
        cluster_selection_method="eom",
    )
    hdb_labels = hdb.fit_predict(X)

    km_labels = KMeans(n_clusters=num_experts, random_state=seed, n_init="auto").fit_predict(X)

    mask = hdb_labels != -1
    Xc, lc = X[mask], hdb_labels[mask]

    results = {
        "setup": {
            "model_name": model_name,
            "fusion_method": fusion_method,
            "num_experts": num_experts,
            "n_samples": X.shape[0],
            "dim": X.shape[1],
        },
        "hdbscan": {
            "noise_ratio": float((hdb_labels == -1).mean()),
            "silhouette_core": _silhouette_safe(Xc, lc) if Xc.size else None,
            "cluster_stats": _cluster_size_stats(hdb_labels),
        },
        "kmeans": {
            "silhouette": _silhouette_safe(X, km_labels),
            "cluster_stats": _cluster_size_stats(km_labels),
        },
    }

    print("===============================================================================")
    print(f"[SETUP] model={model_name} | fusion={fusion_method} | experts={num_experts}")
    print(f"[DATA ] n_samples={X.shape[0]} | dim={X.shape[1]} | device={device}")
    print(f"[HDBSCAN] noise_ratio={results['hdbscan']['noise_ratio']:.4f} "
          f"| silhouette_core={results['hdbscan']['silhouette_core']}")
    print(f"[KMEANS ] silhouette={results['kmeans']['silhouette']}")
    print("===============================================================================")

    return results


def main():
    from src.core.config import Config
    config = Config.from_cli().finalize().validate()
    tokenizer = get_tokenizer(config)
    kfold_train_set =  get_kfold_dataset(config, tokenizer)
    
    for fold in range(config.k_folds):
        print("=================================================================================")
        print("KFold:", fold + 1)

        train_ds, val_ds =  kfold_train_set.get_fold(fold)
        _, val_loader, _ = get_dataloader(cfg=config, train_set=train_ds, val_set=val_ds) 

        results = test_representation_clusterability_hdbscan_cpu(
            val_loader,
            model_name=config.model_name,
            num_labels=config.num_labels,
            num_experts=config.num_experts,
            fusion_method=config.fusion_method,
            max_batches=50,
        )
        print(results)
        
        



if __name__ == "__main__":
    main()
