# HAGMoE Model – Execution Flow Documentation

## 1. Tổng quan luồng xử lý

HAGMoE được thiết kế cho bài toán **Aspect-based Sentiment Analysis / Extraction**, với mục tiêu sử dụng **Mixture of Experts để điều biến semantic embedding**, thay vì đóng vai trò như một layer FFN thông thường.

Luồng tổng quát:

```text
Sentence + Aspect
      ↓
BERT Encoder
      ↓
Aspect / Opinion / Sentence representations
      ↓
Fusion Module
      ↓
Conditioned MoE (Grouped Routing)
      ↓
Merge (Residual hoặc MoE-only)
      ↓
Classifier
      ↓
Main Loss + Auxiliary MoE Losses
```

---

## 2. Input contract

Model nhận các input sau:

* `input_ids_sent`, `attention_mask_sent`
* `input_ids_term`, `attention_mask_term`
* (optional) `aspect_start`, `aspect_end`
* (optional) `aspect_mask_sent`
* (optional) `labels`
* `fusion_method` (runtime override)

Thiết kế cho phép **fallback logic** để xác định aspect span theo thứ tự ưu tiên:

1. `aspect_mask_sent` (nếu được cung cấp)
2. `(aspect_start, aspect_end)`
3. Matching `input_ids_term` trong sentence content

---

## 3. Encoder và biểu diễn cơ sở

### 3.1 Sentence Encoding

* Sử dụng `AutoModel` (BERT backbone)
* `h_sent` được lấy từ `[CLS]` token
* `last_hidden_state` được giữ nguyên cho các bước attention sau

### 3.2 Aspect Representation

* Aspect được pool trực tiếp từ **span trong sentence embedding**
* Không encode term bằng encoder riêng
* Nếu không xác định được span, fallback sang `h_sent`

### 3.3 Opinion Representation

* Opinion được tính bằng **aspect-conditioned attention**
* Query: sentence tokens
* Key: aspect embedding
* Mask toàn bộ aspect tokens để tránh self-attention trivial

---

## 4. Fusion Layer

Fusion kết hợp ba thành phần:

* Sentence representation (`h_sent`)
* Aspect representation (`h_aspect`)
* Opinion representation (`h_opinion`)

Các phương pháp fusion được hỗ trợ:

* `concat`
* `add`
* `mul`
* `cross`
* `gated_concat`
* `bilinear`
* `coattn`
* `late_interaction`

> Lưu ý:
> `sent` và `term` fusion **không được phép** trong HAGMoE.

Fusion method có thể được:

* Truyền trực tiếp qua `forward`
* Hoặc lấy từ `hag_fusion_method` trong config

Fusion output được ký hiệu là `h_fused`.

---

## 5. Conditioned Input cho MoE

Trước khi routing vào MoE, model tạo input có điều kiện:

```text
h_expert_in = Linear([h_fused ; h_aspect])
```

Mục tiêu:

* Cho phép expert **nhận biết context sentiment + aspect**
* Tránh expert chỉ học identity mapping từ sentence

---

## 6. Grouped Mixture of Experts (HAGMoE)

### 6.1 Group Router

* `group_router: h_fused → logits_group`
* Softmax với `group_temperature`
* Output: `p_group ∈ R^{B × G}`

Group mang ý nghĩa **polarity-aware** (positive, negative, neutral).

### 6.2 Expert Router (per group)

Với mỗi group `g`:

* `expert_router[g]: h_expert_in → logits_expert`
* Softmax với `router_temperature`
* Output: `p_expert[g] ∈ R^{B × E}`

### 6.3 Expert Computation

* Mỗi expert là một FFN độc lập
* Expert outputs được weighted sum theo `p_expert`
* Group output được weighted bởi `p_group`

Kết quả MoE cuối cùng:

```text
h_moe = Σ_g p_group[g] · (Σ_e p_expert[g,e] · expert_out[g,e])
```

---

## 7. Merge Strategy

Hai chế độ merge:

* `residual` (default):

  ```text
  h_final = h_fused + h_moe
  ```
* `moe_only`:

  ```text
  h_final = h_moe
  ```

---

## 8. Classification Head

* `h_final` → Dropout → Classification Head
* Head có thể là linear hoặc MLP (build qua `build_head`)
* Output: `logits`

---

## 9. Loss Composition

### 9.1 Main Loss

Tùy `loss_type`:

* Cross Entropy
* Weighted Cross Entropy
* Focal Loss

### 9.2 Auxiliary MoE Losses (có thể bật/tắt độc lập)

| Loss           | Mục tiêu                               |
| -------------- | -------------------------------------- |
| Group Loss     | Supervise group routing theo polarity  |
| Balance Loss   | Tránh expert collapse trong mỗi group  |
| Diversity Loss | Khuyến khích expert outputs khác nhau  |
| Entropy Loss   | Tránh routing quá sắc hoặc quá uniform |
| Collapse Loss  | Phạt khi một group bị bỏ rơi           |

Tổng loss:

```text
L_total = L_main + Σ λ_i · L_aux_i
```

Các hệ số `λ` được kiểm soát hoàn toàn bằng config.

---

## 10. Logging và Debug Hooks

Model tự động lưu:

* `group_probs`
* `expert_probs`
* Entropy statistics
* Group usage statistics
* Loss breakdown chi tiết

Có thể gọi:

```python
model.print_moe_debug()
```

để in routing statistics theo batch gần nhất.

---

## 11. Design Intent Summary

* MoE **không phải layer FFN thông thường**
* Routing dựa trên **semantic condition**, không phải token position
* Group mang ý nghĩa **polarity-aware**
* Loss được thiết kế để:

  * Tránh uniform routing
  * Tránh expert collapse
  * Khuyến khích specialization có kiểm soát

