Dưới đây là **tài liệu hóa pipeline MoE cho bài toán ATSE** được chuẩn hóa, viết lại rõ ràng theo hướng **research-ready**, đủ để dùng làm **README / Method section / code reference**. Mình giữ nguyên tinh thần kiến trúc của bạn nhưng chỉnh lại ký hiệu, luồng logic và diễn giải cho chặt chẽ.

---

# Polarity-aware Grouped Mixture of Experts for ATSE

## 1. Tổng quan Pipeline

Pipeline tổng quát cho **Aspect Term Sentiment Extraction (ATSE)** được thiết kế theo chuỗi sau:

```text
Sentence (S) + Aspect (A)
        ↓
     Fusion
        ↓
   Polarity-aware MoE
        ↓
      Merge
        ↓
   Sentiment Classifier
```

Mục tiêu cốt lõi:
Sử dụng **MoE theo nhóm polarity** để **điều biến semantic embedding** một cách có điều kiện, thay vì dùng MoE như một layer thuần túy.

---

## 2. Kiến trúc chi tiết theo từng Layer

---

## Layer 1: Semantic Extractor

### Input

* Sentence tokens: `S = {w1, ..., wL}`
* Aspect span: `A = {wi, ..., w(i+m)}`

Encoder nền tảng: BERT (hoặc backbone tương đương).

### Output representations

#### 1.1 Sentence representation

```text
h_sent = H_CLS ∈ R^d
```

* CLS token đại diện ngữ nghĩa toàn câu.

---

#### 1.2 Aspect representation

```text
h_aspect = (1/m) * Σ H_k , k = i → i+m
```

* Trung bình embedding các token thuộc aspect span.
* Đại diện ngữ nghĩa cục bộ của aspect.

---

#### 1.3 Opinion representation (latent)

Opinion không được gán nhãn trực tiếp, được suy luận **ẩn (latent)** thông qua attention có điều kiện theo aspect.

**Query generation**

```math
Q = MLP(H)  
```
(H=?)

**Aspect-conditioned attention**

```math
α_k = Q_k · h_{aspect}^T
```

Chuẩn hóa:

```math
α = softmax(α)
```

**Opinion embedding**

```text
h_opinion = Σ α_k H_k
```

> Trực giác: model tự học token nào mang sắc thái cảm xúc liên quan đến aspect.

---

### Kết quả Layer 1

```text
{ h_sent, h_aspect, h_opinion }
```

---

## Layer 2: Fusion Layer

Hợp nhất các representation:

```text
h_fused = Fusion(h_sent, h_aspect, h_opinion)
```

Fusion có thể là:

* Concatenation + Linear
* Gated fusion
* Bilinear interaction
...

Yêu cầu:

```text
h_fused ∈ R^d
```

---

## Layer 3: Polarity-aware Grouped MoE

### 3.1 Động cơ xác suất

Phân phối xác suất đầu ra được mô hình hóa như sau:

```math
P(y|x) = Σ_g P(g|x) Σ_{e∈g} P(e|g,x) P(y|e,x)
```

Trong đó:

* `g`: polarity group (Positive / Negative / Neutral)
* `e`: expert thuộc group `g`

---

### 3.2 Group Router (Polarity Router)

```math
g = softmax(W_g h_{fused} + b_g)
```

```text
g = [P_pos, P_neg, P_neu]
```

Mapping expert:

* Positive: E1, E2, E3
* Negative: E4, E5, E6
* Neutral:  E7, E8, E9

---

### 3.3 Expert Router (conditioned)

Tạo representation có điều kiện theo aspect:

```math
h_{cond} = W_{cond} [h_{fused} ; h_{aspect}] + b_{cond}
```

Với mỗi group `g`:

```math
P(e|g,x) = softmax(W_{expert}^g h_{cond} / τ)
```

* `τ`: temperature, điều khiển độ sắc routing
* Routing chỉ diễn ra **trong group được chọn**

---

### 3.4 Group-wise MoE aggregation

Output MoE:

```text
h_moe = Σ_g P(g|x) Σ_{e∈g} P(e|g,x) · Expert_e(h_cond)
```

> Đây là **mixture theo nhóm**, không phải MoE phẳng.

---

## Layer 4: Merge & Classification

Ghép lại representation cuối:

```text
h_final = Merge(h_fused, h_moe) # Or using only h_moe ...
```

Classifier:

```math
ŷ = softmax(W_{cls} h_{final} + b_{cls})
```

---

## 4. Loss Functions

Tổng loss gồm nhiều thành phần:

```math
L_{total} = L_{cls} + a·L_{group} + b·L_{balance} + c·L_{diversity}
```

---

### 4.1 Classification Loss

```math
L_{cls} = FocalLoss(ŷ, y)
```

---

### 4.2 Group Supervision Loss

Nếu có prior polarity signal hoặc pseudo-label:

```math
L_{group}
= - \log g_y
= - \log softmax(W_g h_{fused})_y  
```
Can be normalize to B when apply batchsize

---

### 4.3 Expert Load Balancing Loss

```math
L_{balance} = N · Σ_{i=1}^{N} f_i · P_i
```

Trong đó:

* `f_i`: tỉ lệ mẫu được route tới expert i trong batch
* `P_i`: xác suất trung bình expert i
* `N`: số expert

Mục tiêu: tránh expert chết.

---

### 4.4 Expert Diversity Loss (Intra-group)

```math
L_{diversity} = || W W^T − I ||_2
```

* `W`: ma trận trọng số expert trong cùng group
* Ép các expert học các subspace khác nhau

---

### 4.5 Trọng số loss

```text
a = 0.5
b = 0.01
c = 0.1
```

---

## 5. Điểm Novelty Cốt lõi

* MoE **không phải layer**, mà là **semantic deformation module**
* Routing có cấu trúc theo **polarity prior**
* Opinion representation là **latent**, không cần annotation
* Expert specialization diễn ra **trong từng polarity manifold**

