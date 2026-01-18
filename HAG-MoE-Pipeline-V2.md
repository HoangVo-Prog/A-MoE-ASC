# HAG-MoE-Pipeline-V2.md

## (Codex-Oriented Design Specification)

---

## 0. Purpose of This Document

This document defines the **exact architectural intent** of the HAG-MoE pipeline.

It is written for:

* Code agents (Codex)
* Contributors implementing new model variants

It is **not** a tutorial or a paper draft.

Any implementation MUST follow the constraints below.

---

## 1. Task Definition

**Task**: Aspect Term Sentiment Analysis (ATSA / ATSE)

**Input**:

* Sentence tokens `S = {w₁ … wₙ}`
* Aspect span `A = {wᵢ … wⱼ}`

**Output**:

* Sentiment label ∈ {Positive, Negative, Neutral}

---

## 2. High-level Pipeline (Invariant)

```text
Sentence + Aspect
        ↓
Semantic Extractor (BERT)
        ↓
Aspect / Opinion / Sentence representations
        ↓
Fusion Module
        ↓
Polarity-aware Grouped MoE
        ↓
Merge
        ↓
Sentiment Classifier
```

**Invariant**:

* MoE is a **semantic deformation module**
* MoE is **NOT** a generic stacked neural layer

---

## 3. Layer 1: Semantic Extractor

### Backbone

* Pretrained BERT (or equivalent)
* Encoder structure MUST NOT be modified

### Outputs

Let `H ∈ R^{n × d}` be token embeddings.

* Sentence representation
  `h_sent = H[CLS]`

* Aspect representation
  `h_aspect = mean(H[i:j])`

* Opinion representation (latent, no labels)

Opinion is inferred via **aspect-conditioned attention**:

```math
α_k = (Q_k · h_aspect) / sqrt(d)
```

* Aspect tokens MUST be masked
* Softmax over non-aspect tokens
* Opinion embedding:

```math
h_opinion = Σ α_k · H_k
```

**Output of Layer 1**:

```text
{ h_sent, h_aspect, h_opinion }
```

---

## 4. Layer 2: Fusion Module

### Input

```text
S = h_sent
A = h_aspect
O = h_opinion
```

### Output

```text
h_fused ∈ R^d
```

### Allowed fusion strategies

* Concatenation + Linear
* Additive
* Multiplicative
* Cross-attention
* Bilinear
* Gated fusion

### Optional: Fusion Selector

A lightweight selector MAY be used to weight multiple fusion strategies.

Constraint:

* Fusion output dimensionality MUST remain `d`
* Fusion module MUST be swappable

---

## 5. Layer 3: Polarity-aware Grouped MoE (Core Contribution)

### 5.1 Design Principle

* Experts are **grouped by polarity**
* Routing is **hierarchical**
* Expert specialization happens **inside polarity manifolds**

---

### 5.2 Polarity Group Router

Input:

```text
h_fused
```

Compute group probabilities:

```math
P(g | x) = softmax(W_g h_fused)
```

Groups:

* Positive
* Negative
* Neutral

Each group owns a **disjoint expert set**.

---

### 5.3 Aspect-conditioned Expert Router

Conditioned input:

```math
h_cond = W_cond [h_fused ; h_aspect]
```

For group `g`:

```math
P(e | g, x) = softmax(W_expert^g h_cond / τ)
```

Constraints:

* Routing is ONLY inside selected group
* Temperature `τ` controls sharpness

---

### 5.4 Expert Computation

Each expert produces a **semantic deformation**:

Allowed expert forms:

* FFN
* Low-rank parameter delta
* Additive embedding shift

Example FFN expert:

```math
Expert(x) = W₂ GELU(W₁ x)
```

Within-group aggregation:

```math
h_g = Σ P(e | g, x) · Expert_e(h_fused)
```

Cross-group aggregation:

```math
h_moe = Σ P(g | x) · h_g
```

---

## 6. Layer 4: Merge & Classification

Final representation:

```text
h_final = Merge(h_fused, h_moe)
```

Classifier:

```math
ŷ = softmax(W_cls h_final)
```

Merge strategies:

* Use `h_moe` only
* Residual: `h_fused + h_moe`

---

## 7. Loss Functions

Total loss:

```math
L = L_cls + λ₁ L_group + λ₂ L_balance + λ₃ L_diversity
```

### 7.1 Classification Loss

* Cross-entropy or Focal Loss

### 7.2 Group Supervision Loss (Optional)

* Align predicted polarity group with sentiment label

### 7.3 Expert Load Balancing Loss

* Prevent expert collapse
* Encourage uniform usage

### 7.4 Expert Diversity Loss

* Orthogonality constraint within group

---

## 8. Architectural Constraints (Hard Rules)

Agents MUST NOT:

* Treat MoE as a generic layer
* Add MoE blocks inside BERT
* Modify dataset formats
* Modify training loops unless explicitly instructed

Agents MUST:

* Keep forward interfaces consistent
* Add new variants as new model files
* Follow this document when implementing grouped MoE

---

## 9. What This Document Is For

This document is used to:

* Guide Codex when implementing new models
* Enforce architectural intent
* Prevent hallucinated MoE designs

If any instruction conflicts with this document,
**this document takes priority**.


## Implementation Mapping (Spec → Code)

This section maps each conceptual component of the HAG-MoE pipeline
to its expected implementation location in the codebase.

This mapping is **binding for agents**.

---

### Layer 1: Semantic Extractor

**Spec concepts**:

* Sentence representation `h_sent`
* Aspect representation `h_aspect`
* Latent opinion representation `h_opinion`

**Code location**:

* `HAGMoEModel.forward(...)`
* Helper method: `compute_opinion(H, aspect_mask, h_aspect)`

**Constraints**:

* Aspect tokens MUST be masked during opinion attention
* No supervision labels for opinion tokens

---

### Layer 2: Fusion Module

**Spec concepts**:

* Fusion of `{h_sent, h_aspect, h_opinion}`
* Output `h_fused ∈ R^d`

**Code location**:

* `HAGMoEModel.build_fusion(...)`
* or reuse fusion logic from `base_model.py` / `moehead_model.py`

**Constraints**:

* Fusion output dimensionality MUST remain `d`
* Fusion strategy MUST be swappable

---

### Layer 3.1: Polarity Group Router

**Spec concepts**:

* Polarity groups: Positive, Negative, Neutral
* Group probability `P(g | x)`

**Code location**:

* `HAGMoEModel.route_group(h_fused)`
* or class `PolarityGroupRouter`

**Constraints**:

* Exactly 3 groups
* Softmax over groups

---

### Layer 3.2: Aspect-conditioned Expert Router

**Spec concepts**:

* Conditioned input `h_cond = [h_fused ; h_aspect]`
* Expert probability `P(e | g, x)`

**Code location**:

* `HAGMoEModel.route_expert(h_cond, group_id)`
* or class `GroupExpertRouter`

**Constraints**:

* Routing is restricted to experts within the selected group
* Temperature `τ` must be configurable

---

### Layer 3.3: Grouped Expert Computation

**Spec concepts**:

* Within-group aggregation
* Cross-group aggregation
* Semantic deformation via experts

**Code location**:

* `HAGMoEModel.apply_experts(...)`
* or class `GroupedExperts`

**Constraints**:

* Experts MUST output representations in `R^d`
* Experts MUST act as semantic deformation, not generic layers

---

### Layer 4: Merge & Classification

**Spec concepts**:

* Merge `h_fused` and `h_moe`
* Sentiment classifier

**Code location**:

* `HAGMoEModel.merge(h_fused, h_moe)`
* `HAGMoEModel.classifier`

**Constraints**:

* Default merge SHOULD be residual
* Output logits MUST match existing engine expectations

---

### Loss Functions

**Spec concepts**:

* Classification loss
* Group supervision loss
* Expert balance loss
* Expert diversity loss

**Code location**:

* `HAGMoEModel.compute_aux_losses(...)`

**Constraints**:

* Auxiliary losses MUST NOT break existing training loops
* BaseModel and other modes MUST remain unaffected
* Loss integration MUST respect mode-specific branching

---

### Mode Compatibility Rule

If any shared code path is modified to support HAG-MoE:

* BaseModel MUST follow the original logic
* Differences MUST be introduced via:

  * explicit `mode` checks, or
  * model-specific overrides

Silent behavior changes are forbidden.

---

## Priority Rule

If there is a conflict between:

* conceptual description above, and
* any inferred implementation choice,

this **Implementation Mapping takes priority**.