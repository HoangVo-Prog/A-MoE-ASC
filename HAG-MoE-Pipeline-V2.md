
---

# Polarity-aware Grouped Mixture of Experts for ATSE

## 1. Pipeline Overview

The overall pipeline for **Aspect Term Sentiment Extraction (ATSE)** is designed as the following sequence:

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

Core objective:
Use a **polarity grouped Mixture of Experts** to **conditionally modulate semantic embeddings**, rather than treating MoE as a purely feedforward layer.

---

## 2. Layer-wise Architecture

---

## Layer 1: Semantic Extractor

### Input

* Sentence tokens: `S = {w1, ..., wL}`
* Aspect span: `A = {wi, ..., w(i+m)}`

Backbone encoder: BERT or an equivalent pretrained transformer.

### Output Representations

> **Note**
> $$\mathbf{H} = \text{Encoder}(\text{input_ids}, \text{attention_mask}) \in \mathbb{R}^{n \times d}$$

#### 1.1 Sentence Representation

```text
h_sent = H_[CLS] ∈ R^d
```

* The CLS token represents the global sentence semantics.

---

#### 1.2 Aspect Representation

```text
h_aspect = (1/m) * Σ H_k , k = i → i+m
```

* Mean pooling over embeddings of tokens within the aspect span.
* Represents the local semantics of the aspect.

---

#### 1.3 Latent Opinion Representation

Opinion terms are not explicitly labeled. They are inferred **latently** through aspect conditioned attention.

**Query generation**

> **Note**
> $$\mathbf{Q} = \text{MLP}(\mathbf{H}) = \mathbf{W}_q \mathbf{H} + \mathbf{b}_q \in \mathbb{R}^{n \times d}$$

```math
Q = MLP(H)
```

**Aspect conditioned attention**

> **Note**
> $$\alpha_k = \frac{\mathbf{Q}*k \cdot \mathbf{h}*{aspect}^T}{\sqrt{d}}, \quad \forall k \in {1, ..., n}$$

```math
α_k = Q_k · h_{aspect}^T
```

> **Note**
> A masking operation is applied to prevent aspect tokens from attending to themselves. This forces the model to search for opinion words outside the aspect span.

$$
\tilde{\alpha}_k =
\begin{cases}
-\infty & \text{if } k \in [i, j] \text{ (aspect tokens)} \
\alpha_k & \text{otherwise}
\end{cases}
$$

Normalization:

$$
\boldsymbol{\beta} = \text{softmax}(\tilde{\boldsymbol{\alpha}}) \in \mathbb{R}^n
$$

$$
\beta_k = \frac{\exp(\tilde{\alpha}*k)}{\sum*{m=1}^{n} \exp(\tilde{\alpha}_m)}
$$

This can be interpreted as:

$$
\beta_k = P(\text{token } k \text{ is an opinion} \mid \text{Aspect}, S)
$$

**Opinion embedding**

$$
\mathbf{h}*{opinion} = \sum*{k=1}^{n} \beta_k \cdot \mathbf{H}_k \in \mathbb{R}^d
$$

```text
h_opinion = Σ α_k H_k
```

Intuition: the model learns which tokens express sentiment toward the given aspect.

---

### Layer 1 Output

```text
{ h_sent, h_aspect, h_opinion }
```

---

## Layer 2: Fusion Layer

The representations are fused as:

```text
h_fused = Fusion(S = h_sent, A = h_aspect, O = h_opinion)
```

Possible fusion strategies include:

* **Concatenation**
  $$
  f_1 = \mathbf{W}_c[\mathbf{S}; \mathbf{A}; \mathbf{O}] + \mathbf{b}_c \in \mathbb{R}^d
  $$

* **Additive**
  $$
  f_2 = \mathbf{S} + \mathbf{A} + \mathbf{O}
  $$

* **Multiplicative**
  $$
  f_3 = \mathbf{S} \odot \mathbf{A} \odot \mathbf{O}
  $$

* **Cross Attention**
  $$
  f_4 = \text{CrossAttn}(\mathbf{A}, \mathbf{S}, \mathbf{S}) + \text{CrossAttn}(\mathbf{O}, \mathbf{S}, \mathbf{S})
  $$

  where
  $$
  \text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
  \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}
  $$

* **Bilinear interaction**
  $$
  f_5 = \mathbf{A}^T \mathbf{W}_b \mathbf{O}
  $$

* **Gated fusion**
  $$
  f_6 = g \cdot \mathbf{S} + (1 - g) \cdot \frac{\mathbf{A} + \mathbf{O}}{2}
  $$

  where
  $$
  g = \sigma(\mathbf{W}_g[\mathbf{A}; \mathbf{O}] + b_g)
  $$

Let $\mathbf{S} = \mathbf{h}*{sent}$, $\mathbf{A} = \mathbf{h}*{aspect}$, $\mathbf{O} = \mathbf{h}_{opinion}$.

| Strategy        | Formula | Parameters       |
| --------------- | ------- | ---------------- |
| Concatenation   | $f_1$   | $O(3d \times d)$ |
| Additive        | $f_2$   | $O(0)$           |
| Multiplicative  | $f_3$   | $O(0)$           |
| Cross Attention | $f_4$   | $O(d^2)$         |
| Bilinear        | $f_5$   | $O(d^2)$         |
| Gated           | $f_6$   | $O(2d)$          |

Constraint:

```text
h_fused ∈ R^d
```

### Extension: Fusion Selector

Different samples may benefit from different fusion strategies:

* Explicit sentiment: simple additive fusion
* Implicit sentiment: cross attention
* Comparative sentiment: bilinear interaction

Let:

$$
\mathbf{z} = [\mathbf{h}*{aspect}; \mathbf{h}*{opinion}] \in \mathbb{R}^{2d}
$$

Strategy selection:

$$
\boldsymbol{\pi} = \text{softmax}\left(\frac{\mathbf{W}_{sel}\mathbf{z}}{\tau}\right) \in \mathbb{R}^6
$$

where:

* $\mathbf{W}_{sel} \in \mathbb{R}^{6 \times 2d}$
* $\tau$ is a temperature parameter

Weighted aggregation:

$$
\mathbf{h}*{fused} = \sum*{i=1}^{6} \pi_i \cdot f_i
$$

---

## Layer 3: Polarity-aware Grouped MoE

### 3.1 Probabilistic Motivation

The output distribution is modeled as:

$$
P(y \mid x) = \sum_g P(g \mid x) \sum_{e \in g} P(e \mid g, x) P(y \mid e, x)
$$

where:

* $g$ is a polarity group (positive, negative, neutral)
* $e$ is an expert within group $g$

---

### 3.2 Polarity Group Router

$$
\mathbf{g} = \text{softmax}(\mathbf{W}*g \mathbf{h}*{fused} + \mathbf{b}_g)
$$

```text
g = [P_pos, P_neg, P_neu]
```

Expert assignment:

* Positive: E1, E2, E3
* Negative: E4, E5, E6
* Neutral: E7, E8, E9

---

### 3.3 Aspect-conditioned Expert Router

Conditioned representation:

$$
\mathbf{h}*{cond} =
\mathbf{W}*{cond}[\mathbf{h}*{fused}; \mathbf{h}*{aspect}] + \mathbf{b}_{cond}
$$

For each group $g$:

$$
\mathbf{e}*g =
\text{softmax}\left(\frac{\mathbf{W}*{expert}^{(g)} \mathbf{h}_{cond}}{\tau}\right)
\in \mathbb{R}^{E_g}
$$

where $\mathbf{W}_{expert}^{(g)}$ is group specific.

Routing occurs **only within the selected group**.

---

### 3.4 Group-wise MoE Aggregation

**Within group**

$$
\mathbf{h}*g = \sum*{i=1}^{E_g} e_{g,i} \cdot \text{Expert}*{g,i}(\mathbf{h}*{fused})
$$

Each expert is an FFN:

$$
\text{Expert}_i(\mathbf{x}) =
\mathbf{W}_2^{(i)} \cdot \text{GELU}(\mathbf{W}_1^{(i)} \mathbf{x} + \mathbf{b}_1^{(i)}) + \mathbf{b}_2^{(i)}
$$

**Across groups**

$$
\mathbf{h}*{moe} = \sum*{g=1}^{3} P(g \mid x) \cdot \mathbf{h}_g
$$

Final form:

```text
h_moe = Σ_g P(g|x) Σ_{e∈g} P(e|g,x) · Expert_e(h_cond)
```

This is a **grouped MoE**, not a flat MoE.

---

## Layer 4: Merge and Classification

**Classification**

$$
\mathbf{logits} = \mathbf{W}*{cls} \mathbf{h}*{moe} + \mathbf{b}_{cls}
$$

**Prediction**

$$
\hat{y} = \arg\max(\text{softmax}(\mathbf{logits}))
$$

**Probability**

$$
P(Y = c \mid x) =
\frac{\exp(\text{logits}*c)}{\sum*{c'} \exp(\text{logits}_{c'})}
$$

Final representation:

```text
h_final = Merge(h_fused, h_moe)
```

Classifier:

$$
\hat{y} = \text{softmax}(\mathbf{W}*{cls} \mathbf{h}*{final} + \mathbf{b}_{cls})
$$

---

## 4. Loss Functions

Overall loss:

$$
\mathcal{L}*{total}
= \mathcal{L}*{cls}

* \lambda_1 \mathcal{L}_{group}
* \lambda_2 \mathcal{L}_{balance}
* \lambda_3 \mathcal{L}_{diversity}
  $$

---

### 4.1 Classification Loss

$$
\mathcal{L}_{cls} = \text{FocalLoss}(\hat{y}, y)
$$

---

### 4.2 Group Supervision Loss

If polarity priors or pseudo labels are available:

$$
\mathcal{L}*{group}
= -\sum*{g=1}^{3} y_g \log P(g \mid x)
$$

where $y_g = 1$ if the true sentiment belongs to group $g$.

---

### 4.3 Expert Load Balancing Loss

$$
\mathcal{L}*{balance}
= N \sum*{i=1}^{N} f_i P_i
$$

where:

* $f_i$ is the routing frequency of expert $i$
* $P_i$ is the mean routing probability
* $N$ is the number of experts

Purpose: prevent expert collapse.

---

### 4.4 Expert Diversity Loss

$$
\mathcal{L}_{diversity}
= \left| \mathbf{W} \mathbf{W}^T - \mathbf{I} \right|_F^2
$$

Encourages experts within the same group to learn distinct subspaces.

---

### 4.5 Loss Weights

$$
\lambda_1 = 0.5,\quad
\lambda_2 = 0.01,\quad
\lambda_3 = 0.1
$$

```text
a = 0.5
b = 0.01
c = 0.1
```

---

## 5. Core Novelty

* MoE is treated as a **semantic deformation module**, not a standard layer
* Routing is structured using **polarity priors**
* Opinion representation is **latent** and does not require annotation
* Expert specialization occurs **within polarity specific manifolds**

---
