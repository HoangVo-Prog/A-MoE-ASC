from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


class MoEMetricsAccumulator:
    def __init__(self, *, num_labels: Optional[int] = None, compute_mi: bool = False) -> None:
        self.num_labels = int(num_labels) if num_labels is not None else None
        self.compute_mi = bool(compute_mi)

        self.count = 0
        self.sum_ent = 0.0
        self.sum_ent2 = 0.0
        self.sum_kl = 0.0
        self.sum_margin = 0.0
        self.sum_effective = 0.0
        self.sum_probs: Optional[torch.Tensor] = None
        self.top1_counts: Optional[torch.Tensor] = None
        self.sum_uniform = 0.0
        self.ent_hist = torch.zeros(20, dtype=torch.int64)

        self.label_ent_sum: Dict[int, float] = {}
        self.label_count: Dict[int, int] = {}
        self.label_top1_counts: Dict[int, torch.Tensor] = {}

        self.joint_counts: Optional[torch.Tensor] = None
        self.label_counts_mi: Optional[torch.Tensor] = None
        self.top1_counts_mi: Optional[torch.Tensor] = None

    def update(self, moe_stats: Optional[Dict[str, Any]], labels: Optional[torch.Tensor]) -> None:
        if moe_stats is None:
            return
        probs = _extract_router_probs(moe_stats)
        if probs is None or probs.numel() == 0 or probs.dim() != 2:
            return

        probs = probs.detach()
        if probs.device.type != "cpu":
            probs = probs.to("cpu")
        labels_cpu = None
        if labels is not None and torch.is_tensor(labels):
            labels_cpu = labels.detach().to("cpu").long()

        bsz, k = probs.shape
        if bsz == 0 or k == 0:
            return

        if self.sum_probs is None:
            self.sum_probs = torch.zeros(k, dtype=torch.float64)
            self.top1_counts = torch.zeros(k, dtype=torch.int64)
            if self.compute_mi and self.num_labels is not None:
                self.joint_counts = torch.zeros(self.num_labels, k, dtype=torch.int64)
                self.label_counts_mi = torch.zeros(self.num_labels, dtype=torch.int64)
                self.top1_counts_mi = torch.zeros(k, dtype=torch.int64)

        eps = 1e-12
        ent = -(probs * (probs + eps).log()).sum(dim=-1)
        ent_norm = ent / float(math.log(k + eps))
        kl = float(math.log(k + eps)) - ent
        top2 = torch.topk(probs, k=min(2, k), dim=-1).values
        if top2.size(-1) >= 2:
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.zeros_like(ent_norm)

        eff = 1.0 / probs.pow(2).sum(dim=-1).clamp_min(eps)
        top1 = probs.argmax(dim=-1)

        self.count += int(bsz)
        self.sum_ent += float(ent_norm.sum().item())
        self.sum_ent2 += float((ent_norm * ent_norm).sum().item())
        self.sum_kl += float(kl.sum().item())
        self.sum_margin += float(margin.sum().item())
        self.sum_effective += float(eff.sum().item())

        self.sum_probs += probs.sum(dim=0).to(dtype=torch.float64)
        self.top1_counts += torch.bincount(top1, minlength=k).to(dtype=torch.int64)

        self.sum_uniform += float((ent_norm > 0.9).sum().item())

        hist_idx = torch.clamp((ent_norm * 19).long(), 0, 19)
        self.ent_hist += torch.bincount(hist_idx, minlength=20).to(dtype=torch.int64)

        if labels_cpu is not None:
            for lbl in labels_cpu.unique().tolist():
                mask = labels_cpu == int(lbl)
                if not torch.any(mask):
                    continue
                lbl_int = int(lbl)
                self.label_ent_sum[lbl_int] = self.label_ent_sum.get(lbl_int, 0.0) + float(
                    ent_norm[mask].sum().item()
                )
                self.label_count[lbl_int] = self.label_count.get(lbl_int, 0) + int(mask.sum().item())
                if lbl_int not in self.label_top1_counts:
                    self.label_top1_counts[lbl_int] = torch.zeros(k, dtype=torch.int64)
                self.label_top1_counts[lbl_int] += torch.bincount(
                    top1[mask], minlength=k
                ).to(dtype=torch.int64)

            if self.compute_mi and self.num_labels is not None:
                if self.joint_counts is not None:
                    for i in range(labels_cpu.numel()):
                        li = int(labels_cpu[i].item())
                        ti = int(top1[i].item())
                        if 0 <= li < self.num_labels and 0 <= ti < k:
                            self.joint_counts[li, ti] += 1
                    self.label_counts_mi += torch.bincount(
                        labels_cpu, minlength=self.num_labels
                    ).to(dtype=torch.int64)
                    self.top1_counts_mi += torch.bincount(top1, minlength=k).to(dtype=torch.int64)

    def finalize(self) -> Optional[Dict[str, Any]]:
        if self.count <= 0 or self.sum_probs is None or self.top1_counts is None:
            return None

        mean_ent = self.sum_ent / self.count
        var_ent = self.sum_ent2 / self.count - mean_ent**2
        if var_ent < 0.0:
            var_ent = 0.0
        std_ent = float(var_ent ** 0.5)
        mean_prob = (self.sum_probs / self.count).tolist()
        top1_hist = (self.top1_counts.to(dtype=torch.float64) / self.count).tolist()
        dead_count = int((torch.tensor(mean_prob) < 1e-3).sum().item())
        uniform_rate = float(self.sum_uniform / self.count)
        ent_hist = (self.ent_hist / self.ent_hist.sum().clamp_min(1)).tolist()

        per_label_entropy = {}
        per_label_top1 = {}
        for lbl, cnt in self.label_count.items():
            if cnt <= 0:
                continue
            per_label_entropy[str(lbl)] = float(self.label_ent_sum.get(lbl, 0.0) / cnt)
            counts = self.label_top1_counts.get(lbl)
            if counts is not None:
                per_label_top1[str(lbl)] = (counts.to(dtype=torch.float64) / cnt).tolist()

        mi = None
        if self.compute_mi and self.joint_counts is not None:
            total = float(self.joint_counts.sum().item())
            if total > 0:
                p_xy = self.joint_counts.to(dtype=torch.float64) / total
                p_x = p_xy.sum(dim=1, keepdim=True)
                p_y = p_xy.sum(dim=0, keepdim=True)
                denom = p_x @ p_y
                nz = p_xy > 0
                mi = float((p_xy[nz] * (p_xy[nz] / denom[nz]).log()).sum().item())

        return {
            "entropy_norm_mean": float(mean_ent),
            "entropy_norm_std": float(std_ent),
            "kl_to_uniform_mean": float(self.sum_kl / self.count),
            "margin_mean": float(self.sum_margin / self.count),
            "effective_num_experts": float(self.sum_effective / self.count),
            "mean_prob": mean_prob,
            "top1_hist": top1_hist,
            "dead_count": dead_count,
            "uniform_rate": uniform_rate,
            "entropy_hist": ent_hist,
            "per_label_entropy_mean": per_label_entropy,
            "per_label_top1_hist": per_label_top1,
            "mi_top1_label": mi,
        }


def _extract_router_probs(moe_stats: Dict[str, Any]) -> Optional[torch.Tensor]:
    probs = moe_stats.get("router_probs")
    if torch.is_tensor(probs):
        return probs
    logits = moe_stats.get("router_logits")
    if torch.is_tensor(logits):
        return torch.softmax(logits, dim=-1)
    group_probs = moe_stats.get("group_probs")
    if torch.is_tensor(group_probs):
        return group_probs
    logits_list = moe_stats.get("router_logits_list")
    if isinstance(logits_list, list):
        for item in reversed(logits_list):
            if torch.is_tensor(item):
                return torch.softmax(item, dim=-1)
    return None
