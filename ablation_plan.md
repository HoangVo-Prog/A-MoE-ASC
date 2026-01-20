# Ablation Plan for MoE / HAGMoE (ATSA / ASTE)

## Scope & Datasets

**Datasets**:

* `laptop14`
* `rest14`

**Loss variants (áp dụng cho mọi tier trừ khi ghi rõ)**:

* `ce`
* `weighted_ce`
* `focal`

**Seeds**:

* Mặc định: `num_seeds = 3`
* Tier 0–2: dùng default
* Tier 4: explicit seed stability

**Benchmark fusion**:

* Theo script hiện tại (`--benchmark_fusions`)
* HAGMoE **không dùng** `sent`, `term`

---

## TIER 0 – Sanity & Reference

### Mục tiêu

* Establish task ceiling
* Calibration reference
* Confusion matrix baseline
* Không đánh giá router

### Runs

#### T0.1 BaseModel – laptop14

```bash
bash scripts/run_base_model.sh ce laptop14
bash scripts/run_base_model.sh weighted_ce laptop14
bash scripts/run_base_model.sh focal laptop14
```

#### T0.2 BaseModel – rest14

```bash
bash scripts/run_base_model.sh ce rest14
bash scripts/run_base_model.sh weighted_ce rest14
bash scripts/run_base_model.sh focal rest14
```

### Artifact bắt buộc

* Task metrics (val, test)
* Confusion matrix (raw + normalized)
* Calibration metrics
* Reliability diagram
* Confidence histogram

### Mục tiêu khoa học

Thiết lập **đường chuẩn** để mọi kết luận về MoE đều có điểm tựa rõ ràng, tránh over-claim.

### Câu hỏi chính cần trả lời

1. **Task ceiling hiện tại là bao nhiêu?**

   * Macro F1, Neutral F1 tối đa mà không cần MoE
2. **Calibration baseline có tốt không?**

   * Base model có overconfident không?
3. **Fusion nào là mạnh nhất trong non-MoE?**

   * Có cần giữ tất cả fusion cho các tier sau không?

### Phân tích bắt buộc

* So sánh fusion methods theo:

  * macro F1
  * neutral F1
  * ECE
* Phân tích confusion matrix:

  * neutral bị nhầm sang pos/neg nhiều hay không
* Reliability diagram:

  * base model có bị confidence inflation không

### Kết luận cần rút ra

* Chọn **1–2 fusion tốt nhất** làm reference
* Xác định rõ: **MoE không được phép tệ hơn baseline này**

---

## TIER 1 – Architecture Sweep (Default Config)

### Mục tiêu

* So sánh **non-MoE vs flat MoE vs HAGMoE**
* Phát hiện router uniform ở mức kiến trúc

### Mode được chạy

* BaseModel
* MoEHead
* MoEFFN
* MoESkConnection
* MultiMoE
* MoF
* HAGMoE

### Runs cho mỗi dataset

#### T1.x – laptop14

```bash
bash scripts/run_base_model.sh ce laptop14
bash scripts/run_moe_head.sh ce laptop14
bash scripts/run_moe_ffn.sh ce laptop14
bash scripts/run_moe_skconnection.sh ce laptop14
bash scripts/run_multi_moe.sh ce laptop14
bash scripts/run_mof_model.sh ce laptop14
bash scripts/run_hagmoe_model.sh ce laptop14
```

#### T1.x – rest14

```bash
bash scripts/run_base_model.sh ce rest14
bash scripts/run_moe_head.sh ce rest14
bash scripts/run_moe_ffn.sh ce rest14
bash scripts/run_moe_skconnection.sh ce rest14
bash scripts/run_multi_moe.sh ce rest14
bash scripts/run_mof_model.sh ce rest14
bash scripts/run_hagmoe_model.sh ce rest14
```

### Artifact bổ sung (MoE modes)

* moe_metrics.json
* router entropy histogram
* expert usage bar chart
* top1 expert histogram
* MI(top1, label)

### Mục tiêu khoa học

Phân biệt **vấn đề đến từ kiến trúc** hay **đến từ router mechanics**.

### Câu hỏi chính cần trả lời

1. **Flat MoE có thực sự khác non-MoE không?**
2. **Router trong flat MoE có học gì hay chỉ là noise?**
3. **HAGMoE có thay đổi hành vi router ngay từ kiến trúc không?**

### Phân tích bắt buộc

#### Task-level

* Δ macro F1, Δ neutral F1 so với base
* Calibration shift:

  * MoE có làm ECE tăng không?

#### Router-level

* entropy_norm (mean, std)
* effective_num_experts
* MI(top1, label)

#### So sánh liên mode

* Base vs MoEHead vs MoEFFN vs HAGMoE:

  * Mode nào entropy cao nhất?
  * Mode nào MI ≈ 0?

### Kết luận cần rút ra

* Mode nào **uniform router rõ ràng**
* Mode nào có **dấu hiệu specialization sơ khai**
* Kiến trúc nào đáng để đi sâu Tier 2–3

---

## TIER 2 – Router Mechanics (Uniform vs Collapse)

Áp dụng cho:

* **HAGMoE**
* **1 flat MoE reference**: chọn `MoEHead`

### 2.1 Temperature Sweep

#### Values

* `router_temperature ∈ {0.5, 1.0, 2.0}`
* Với HAGMoE thêm `hag_group_temperature`

#### Runs (ví dụ laptop14)

```bash
# MoEHead
python -m main --mode MoEHead --router_temperature 0.5 ...
python -m main --mode MoEHead --router_temperature 1.0 ...
python -m main --mode MoEHead --router_temperature 2.0 ...

# HAGMoE
python -m main --mode HAGMoE --hag_group_temperature 0.5 ...
python -m main --mode HAGMoE --hag_group_temperature 1.0 ...
python -m main --mode HAGMoE --hag_group_temperature 2.0 ...
```

Chạy cho **cả laptop14 và rest14**.

### 2.2 Loss Component Sweep (One-factor-at-a-time)

Áp dụng cho HAGMoE:

| Config       | group | balance | diversity |
| ------------ | ----- | ------- | --------- |
| default      | on    | on      | on        |
| no_group     | off   | on      | on        |
| no_balance   | on    | off     | on        |
| no_diversity | on    | on      | off       |

Runs:

```bash
python -m main --mode HAGMoE --hag_use_group_loss false ...
python -m main --mode HAGMoE --hag_use_balance_loss false ...
python -m main --mode HAGMoE --hag_use_diversity_loss false ...
```

### 2.3 Capacity Sweep

#### HAGMoE

* `hag_experts_per_group ∈ {2, 4, 8}`
* `hag_num_groups ∈ {2, 3, 4}`

Tổng hợp theo grid hợp lý, không chạy full Cartesian nếu quá lớn.

### Mục tiêu khoa học

Chứng minh **uniform routing không phải do hyperparameter kém**, mà là do **thiếu signal hoặc kiến trúc không phù hợp**.

### Câu hỏi chính cần trả lời

1. **Uniform routing có “cứng” không?**

   * Temperature sweep có phá được không?
2. **Router có dễ collapse không?**
3. **Loss component nào thực sự có tác dụng?**

### Phân tích bắt buộc

#### Temperature sweep

* entropy_norm vs temperature
* margin vs temperature
* MI vs temperature
* Task F1 vs temperature

#### Loss ablation

* Bật/tắt từng loss:

  * entropy
  * balance
  * diversity
* Quan sát:

  * entropy ↓ hay ↑?
  * dead expert có xuất hiện không?
  * task có cải thiện không?

#### Capacity sweep

* effective_num_experts vs K
* dead_expert_count vs K
* Có saturation không?

### Kết luận cần rút ra

* Uniform router:

  * **do thiếu signal** hay **do thiếu regularization**
* Collapse:

  * xảy ra ở config nào
* Loss nào:

  * chỉ làm router đẹp
  * loss nào giúp task thật

---

## TIER 3 – Signal & Representation (Quan trọng nhất)

### Mục tiêu

* Router có học **signal từ input** hay không
* Phân biệt specialization hữu ích vs vô ích

Áp dụng cho:

* HAGMoE
* MoEHead (reference)

---

### 3.1 Conditioning Ablation

| Setting        | Description               |
| -------------- | ------------------------- |
| default        | aspect-conditioned        |
| sentence_only  | không conditioning aspect |
| polarity_aware | group theo polarity       |
| no_cond_proj   | tắt cond projection       |

Runs:

```bash
python -m main --mode HAGMoE --no_cond_proj ...
python -m main --mode HAGMoE --sentence_only ...
```

---

### 3.2 Merge Strategy (HAGMoE)

| hag_merge |
| --------- |
| residual  |
| moe_only  |

```bash
python -m main --mode HAGMoE --hag_merge residual ...
python -m main --mode HAGMoE --hag_merge moe_only ...
```

---

### 3.3 Masking & Pooling

| Setting               |
| --------------------- |
| opinion mask on       |
| opinion mask off      |
| aspect span pooling   |
| full sentence pooling |

Áp dụng cho HAGMoE và MoEHead.

### Mục tiêu khoa học

Trả lời câu hỏi cốt lõi của thesis:

> **Router có học được semantic signal từ input hay không?**

### Câu hỏi chính cần trả lời

1. **Aspect / polarity có ảnh hưởng đến routing không?**
2. **Router có phân biệt label hay chỉ phân biệt embedding noise?**
3. **Specialization có thực sự giúp task không?**

### Phân tích bắt buộc

#### Conditioning ablation

* entropy_norm_by_label
* top1_hist_by_label
* MI(top1, label)
* So sánh:

  * aspect-conditioned vs sentence-only

#### Merge strategy

* residual vs moe_only:

  * moe_only có làm collapse không?
  * residual có che mất specialization không?

#### Masking & pooling

* opinion mask on/off:

  * router entropy thay đổi không?
* aspect span vs full sentence:

  * router có nhạy với vị trí aspect không?

#### Liên hệ router ↔ task

* entropy ↓ + MI ↑ nhưng F1 không ↑?
* trường hợp specialization vô ích

### Kết luận cần rút ra

* Router **có học semantic signal hay không**
* Specialization:

  * hữu ích
  * hay chỉ overfit / shortcut

---

## TIER 4 – Training Dynamics & Robustness

### 4.1 Freeze Schedule

| freeze_epochs |
| ------------- |
| 0             |
| 1             |
| 2             |

Chạy cho **2–3 config tốt nhất** từ Tier 3.

---

### 4.2 Warmup Interaction

| warmup_ratio  |
| ------------- |
| 0             |
| default (0.1) |

---

### 4.3 Seed Stability

Chọn:

* 2 config tốt nhất HAGMoE
* 1 flat MoE

Chạy:

```bash
--seeds 42,43,44
```

Đánh giá:

* mean ± std macro F1
* entropy_norm variance
* MI variance

---

## Decision Rules (Áp dụng sau mỗi Tier)

### Uniform Router

* entropy_norm ≈ 1
* KL ≈ 0
* MI ≈ 0
* Sweep không đổi

### Collapse

* entropy_norm rất thấp
* effective_num_experts ≈ 1
* dead_expert_count cao

### Specialization vô ích

* entropy ↓, MI ↑
* task F1 không tăng
* calibration xấu hơn

### Specialization hữu ích

* entropy ↓ vừa phải
* MI ↑ (val)
* macro hoặc neutral F1 ↑
* ECE không tăng mạnh

### Mục tiêu khoa học

Đảm bảo kết luận **ổn định và tái lập**, không phải may mắn.

### Câu hỏi chính cần trả lời

1. **Specialization có ổn định theo seed không?**
2. **Có phụ thuộc freeze schedule không?**
3. **Warmup có che hoặc phá routing không?**

### Phân tích bắt buộc

* mean ± std của:

  * macro F1
  * entropy_norm
  * MI
* Quan sát variance:

  * router metrics có ổn định hơn task metrics không?

### Kết luận cần rút ra

* Cấu hình nào:

  * ổn định
  * đáng tin cho paper
* Loại bỏ config:

  * quá nhạy seed
  * routing không tái lập

---

## Final Deliverables

Sau toàn bộ ablation, báo cáo phải trả lời được:

1. Mode nào router uniform nhất
2. Router học signal ở đâu
3. Specialization có giúp task không
4. HAGMoE có lợi thế gì trong data nhỏ
5. Config nào **giữ lại cho paper / thesis**


## Tổng kết vai trò từng TIER (1 câu mỗi tier)

* **Tier 0**: đặt nền tảng, tránh over-claim
* **Tier 1**: kiến trúc có tạo khác biệt thật không
* **Tier 2**: router mechanics có vấn đề hay không
* **Tier 3**: router có học semantic hay không
* **Tier 4**: kết luận có đáng tin không