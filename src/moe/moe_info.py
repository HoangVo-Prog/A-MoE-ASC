import math
import torch


@torch.no_grad()
def _moe_debug_stats_per_layer(self):
    """
    Trả về list stats cho từng layer có MoE:
    - usage: phân phối load theo expert (mean over tokens)
    - entropy_norm: entropy chuẩn hóa [0..1]
    - max_load, min_load
    """
    if not self.use_moe:
        return []

    stats = []
    E = None

    for li, layer in enumerate(self.encoder.encoder.layer):
        moe = getattr(layer, "moe_ffn", None)
        if moe is None:
            continue
        if moe.last_router_logits is None or moe.last_topk_idx is None:
            continue

        logits = moe.last_router_logits          # thường shape [B*T, E] hoặc [B, T, E]
        topk_idx = moe.last_topk_idx            # thường shape [B*T, K] hoặc [B, T, K]
        E = moe.moe_cfg.num_experts

        # ép về [N, E] và [N, K]
        if logits.dim() == 3:
            logits2 = logits.reshape(-1, logits.size(-1))
        else:
            logits2 = logits

        if topk_idx.dim() == 3:
            topk2 = topk_idx.reshape(-1, topk_idx.size(-1))
        else:
            topk2 = topk_idx

        # probs full để tính entropy
        probs = torch.softmax(logits2, dim=-1)  # [N, E]

        # entropy chuẩn hoá
        eps = 1e-9
        ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()  # scalar
        ent_norm = ent / math.log(E)

        # usage dựa trên top-k selection: count frequency expert được chọn
        # mỗi token góp K lượt chọn
        counts = torch.zeros(E, device=logits2.device, dtype=torch.float32)
        counts.scatter_add_(0, topk2.reshape(-1), torch.ones_like(topk2.reshape(-1), dtype=torch.float32))
        usage = counts / counts.sum().clamp_min(1.0)

        stats.append({
            "layer": li,
            "entropy_norm": float(ent_norm.item()),
            "max_load": float(usage.max().item()),
            "min_load": float(usage.min().item()),
            "usage": usage.detach().cpu(),  # tensor [E]
        })

    return stats


def print_moe_debug(self, topn: int = 3):
    """
    In nhanh: entropy, max/min load, top expert.
    """
    stats = self._moe_debug_stats_per_layer()
    if not stats:
        print("[MoE] No stats yet (maybe first batch not run or last_router_logits missing).")
        return

    for s in stats:
        usage = s["usage"]
        topv, topi = torch.topk(usage, k=min(topn, usage.numel()))
        top_pairs = ", ".join([f"e{int(i)}={float(v):.3f}" for v, i in zip(topv, topi)])

        print(
            f"[MoE][layer {s['layer']}] entropy_norm={s['entropy_norm']:.3f} "
            f"max={s['max_load']:.3f} min={s['min_load']:.3f} | top: {top_pairs}"
        )
