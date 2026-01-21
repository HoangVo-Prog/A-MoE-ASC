
# AGENTS.md

## Role

You are a Coding Agent responsible for refactoring the training pipeline of this repository.

The dataset format has changed: **only two splits exist (train and test)**.  
The goal is to **unify all training modes into a single training workflow** that:

- Trains on the train split
- Evaluates on the test split
- Repeats the process across multiple seeds to measure stability
- Saves metrics and artifacts in a **benchmark-like format**

You must work incrementally and preserve backward compatibility where possible.

---

## High-Level Goal

Replace the current fragmented run logic (benchmark, kfold, full-only) with a **single multi-seed train–test pipeline**, while:

- Reusing existing training, evaluation, logging, and artifact utilities
- Keeping model code untouched
- Preserving metric schemas similar to `benchmarks.py`

---

## Non-Goals

You must **NOT**:

- Modify model architectures or forward signatures
- Change `get_model()` logic
- Break batch key contracts expected by `engine.py`
- Remove legacy functions (benchmark or kfold), unless explicitly instructed

---

## Core Design Principles

1. **Single Run Path**  
   The default execution path must be:

train → test (per epoch) → best checkpoint → final test eval
repeated across seeds


2. **Multi-Seed Stability Evaluation**  
All experiments must:
- Run across multiple seeds
- Aggregate mean and std of metrics
- Optionally compute ensemble results by averaging logits

3. **Benchmark-Style Outputs**  
Even without kfold or val:
- Metrics, confusion matrices, calibration, MoE metrics
- Must follow the same structure as benchmark outputs
- Must be saved per seed and aggregated across seeds

4. **Minimal Invasiveness**  
Prefer reuse over rewriting.
Touch only what is necessary to support train–test-only logic.

---

## Phased Implementation Plan

### Phase 1: Configuration Cleanup

- Introduce a unified run mode (e.g. `run_mode="single"` or `single_run=True`)
- Normalize seed handling:
- Always expose `cfg.seed_list`
- If not provided, derive from `cfg.seed` and `cfg.num_seeds`
- Deprecate reliance on validation- or kfold-specific flags in default runs
- Ensure missing `val_path` does not raise errors

**Definition of Done**
- Running `main.py` without extra flags triggers single-run multi-seed training

---

### Phase 2: Dataset and Dataloader Contract

- `get_dataset()` must return only `(train_set, test_set)`
- `get_dataloader()` must allow `val_loader=None`
- `datasets.py` must:
- Support the new JSON format
- Map label fields robustly (sentiment, polarity, label, etc.)
- Provide all required batch keys for:
 - Standard modes
 - HAGMoE (aspect spans, masks, debug fields if available)
- Debug-only fields must be optional and safely ignored if missing

**Definition of Done**
- `engine._forward_step()` never crashes due to missing batch keys
- HAGMoE debug utilities run without errors when enabled

---

### Phase 3: Engine Best-Checkpoint Logic (Test-Based)

Current behavior:
- Best checkpoint is selected using validation metrics

New required behavior:
- If `val_loader is None` and `test_loader is provided`:
- Use **test macro F1** for best checkpoint selection
- Preserve neutral-class constraint if currently enforced
- Allow early stopping based on test metrics

This change must be implemented inside `engine.run_training_loop`.

**Definition of Done**
- `best_state_dict` and `best_epoch` are set even without validation
- Early stopping works correctly in train–test-only mode

---

### Phase 4: Unified Multi-Seed Runner

Implement a new runner (e.g. `run_single_train_eval`) that:

For each seed:
1. Set seed
2. Build model via `get_model()`
3. Call `engine.run_training_loop` with:
- `train_loader=train_loader`
- `val_loader=None`
- `test_loader=test_loader`
4. Load best checkpoint
5. Collect test logits
6. Compute:
- acc, macro F1, per-class F1
- confusion matrix (raw and normalized)
- calibration (ECE, reliability bins)
- MoE metrics and routing summaries if available
7. Save per-seed artifacts using `save_artifacts`

After all seeds:
- Aggregate metrics:
- mean and std
- aggregated confusion matrices
- aggregated calibration and MoE metrics if supported
- Optionally compute ensemble logits across seeds
- Save:
- aggregated artifacts (`seed="avg"`)
- ensemble artifacts if enabled
- a final summary JSON similar to benchmark output

**Definition of Done**
- Output structure mirrors benchmark runs
- Multi-seed stability is visible in saved artifacts

---

### Phase 5: Main Entry Simplification

- Update `main.py` to:
- Default to unified single-run pipeline
- Keep legacy benchmark and kfold paths optional and explicit
- Update `src/core/run/__init__.py` to export the new runner

**Definition of Done**
- Default CLI execution uses the new pipeline
- Legacy code paths remain import-safe

---

## Output and Artifact Requirements

Each run must save:

- Per-seed artifacts:
- acc, f1, f1_per_class
- confusion matrix (raw and normalized)
- calibration
- moe_metrics (if any)
- Aggregated artifacts:
- mean and std across seeds
- aggregated confusion matrices
- Optional ensemble results:
- ensemble metrics
- ensemble confusion matrix

Directory structure must remain consistent with existing artifact conventions:
```

output_dir /
mode /
method /
loss_type /
seed_x /
fold_full /
test /

```

---

## Definition of Done (Global)

- Single-run multi-seed pipeline works end-to-end
- No model code changes required
- Artifacts and summaries are comparable to previous benchmark outputs
- HAGMoE and non-MoE modes both run without special casing
- Codebase remains backward compatible and import-safe

