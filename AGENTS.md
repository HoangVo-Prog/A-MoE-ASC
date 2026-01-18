# AGENTS.md

## Project Overview

This repository implements research pipelines for
Aspect Term Sentiment Analysis (ATSA / ATSE),
including multiple Mixture-of-Experts (MoE) variants.

Core research principle:

Mixture-of-Experts is used as a **semantic deformation / parameter adaptation mechanism**,
NOT as a generic stacked neural layer.

This repository is **code-driven**.
README.md is NOT used as a source of truth for architecture or behavior.

Authoritative design specification:
- HAG-MoE-Pipeline-V2.md

If instructions conflict, this file and HAG-MoE-Pipeline-V2.md take priority.

---

## Entry Points & Execution

Primary entry point:
- main.py

Canonical execution patterns:
- bash scripts/run_base_model.sh
- bash scripts/run_moe_*.sh
- PYTHONPATH=src python -m main [args]

Training and evaluation flow:
- src/core/run/train.py (high-level orchestration)
- src/core/run/engine.py (step-level forward, loss, backward, metrics)

---

## Key Files Summary

### Entry & Control Flow

* **`main.py`**
  Central entry point.
  Parses CLI arguments, selects model via `--mode`, builds dataset, initializes model, and dispatches training/evaluation.

* **`src/core/run/train.py`**
  High-level training orchestration.
  Handles epoch loops, validation calls, and logging.
  Should remain stable across model variants.

* **`src/core/run/engine.py`**
  Step-level execution engine.
  Calls `model.forward`, computes loss, performs backward pass, and updates metrics.
  Any refactor affecting loss handling MUST preserve backward compatibility for all modes.

---

### Data Pipeline

* **`src/core/data/datasets.py`**
  Loads ATSA datasets from JSON.
  Responsible for tokenization, aspect span handling, masks, and batch dictionary structure.
  Output batch format is a **shared contract** across all models.

---

### Models

* **`src/models/base_model.py`**
  Baseline implementation.
  Defines the canonical forward interface and behavior reference.
  MUST remain functionally unchanged unless explicitly requested.

* **`src/models/moe*_model.py`**
  Existing MoE-based variants (MoEHead, MoEFFN, MOF, SD, MultiMoE, etc.).
  Implement different expert-routing or semantic deformation strategies while respecting the shared interface.

* **`src/models/hagmoe_model.py`** (to be added)
  Polarity-aware Grouped MoE implementation following `HAG-MoE-Pipeline-V2.md`.

* **`src/models/__init__.py`**
  Model registry.
  Maps `--mode` strings to concrete model classes.

---

### Scripts

* **`scripts/run_*.sh`**
  Canonical experiment launchers.
  Used for verification after changes.
  New model variants SHOULD add a corresponding script.

---

### Non-editable Artifacts

* **`dataset/`**, **`dataset/raw_atsa/`**
  Source datasets. Never modify unless explicitly instructed.

* **`results/`**
  Generated experiment outputs. Must not be altered by agents.

---

## Modification Scope (IMPORTANT)

### Global Rule

Agents MAY refactor any part of the codebase
**ONLY IF** the refactor preserves existing behavior for all modes.

### Mode-specific Constraint (HARD RULE)

If a refactor introduces logic differences that affect only some modes
(e.g. BaseModel vs MoE-based models):

- The change MUST be implemented via **explicit branching**
- Example:
  - conditional checks on `mode`
  - subclass overrides
  - model-specific code paths

Agents MUST NOT:
- Silently change shared logic in a way that alters BaseModel behavior
- Break backward compatibility for existing modes

### Explicitly Allowed

Agents are allowed to:
- Add new model variants under `src/models/`
- Refactor shared utilities IF behavior is preserved for all modes
- Modify existing model files under `src/models/` when implementing new logic
- Update `src/models/__init__.py` to register new models
- Add or modify scripts under `scripts/`

---

## Restricted Files & Directories

Agents must NOT modify unless explicitly instructed:
- `dataset/`
- `dataset/raw_atsa/`
- `results/`
- Generated JSON / CSV experiment outputs

Agents must NOT:
- Change dataset formats
- Rename existing result files
- Delete or overwrite experiment outputs

---

## Architectural Invariants (DO NOT VIOLATE)

- MoE is NOT a generic feed-forward or attention layer
- MoE MUST act as a conditional semantic deformation mechanism
- The BERT encoder backbone MUST NOT be structurally altered
- Training loops in `src/core/run/` MUST remain backward compatible
- Public model interfaces (forward signatures, returned keys) MUST remain stable

Any MoE-related implementation MUST follow:
- the conceptual constraints in HAG-MoE-Pipeline-V2.md
- grouped routing semantics if polarity-aware MoE is involved

---

## Coding Style & Conventions

- Python, 4-space indentation
- Follow existing naming conventions in `src/models/`
- Prefer explicit branching over implicit behavior changes
- Avoid introducing unnecessary abstractions

---

## Verification & Validation

There is no automated test suite.

After changes, agents MUST:
- Ensure all modes still import and run
- Provide at least one runnable command for verification, e.g.:
  - bash scripts/run_base_model.sh ce laptop14
  - bash scripts/run_moe_head.sh ce rest14

Agents MUST NOT fabricate or modify experimental results.

---

## Agent Behavior Guidelines

- Make minimal, incremental changes
- Clearly separate shared logic and mode-specific logic
- Prefer adding new files over destabilizing existing baselines
- If architectural intent is ambiguous, STOP and ask for clarification
