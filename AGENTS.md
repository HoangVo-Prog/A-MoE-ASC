# Codex Prompt: Add a New ATSC Model (Minimal Integration)

## Goal
Add a new model class to this repo so it can be selected by CLI via `--mode <NewModelName>`, and it trains/evals using the existing pipeline unchanged.

We only want minimal integration:
1) Create a new model file under `src/models/` (e.g., `src/models/new_model.py`)
2) Export the class from `src/models/__init__.py`
No other training/engine logic should be modified unless absolutely necessary to keep compatibility.

## Constraints
- DO NOT change training loop logic in `src/core/run/engine.py`.
- The new model must work with existing `helper.get_model(cfg)` behavior:
  - `cfg.mode` is a class name exported from `src.models`
  - kwargs are built by signature matching from Config flattened dict
- Keep the forward contract consistent with BaseModel for non-HAG models:
  forward(
    input_ids_sent,
    attention_mask_sent,
    input_ids_term,
    attention_mask_term,
    labels=None,
    fusion_method="concat",
  ) -> Dict with keys:
    - "loss": torch.Tensor or None
    - "logits": torch.Tensor [B, num_labels]
  Optional keys are allowed, but must not break the loop.

- Loss behavior must follow existing conventions:
  - support `loss_type in {"ce","weighted_ce","focal"}`
  - use `class_weights` if weighted_ce/focal is selected

## Implementation Steps (DO THESE IN ORDER)

### Step 1: Create new model file
- Create `src/models/new_model.py`.
- Define `class NewModel(nn.Module)` (use the exact class name you plan to pass via `--mode`).
- Follow BaseModel-style init signature so Config can populate args automatically.
  Recommended init signature:
    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        dropout: float,
        head_type: str,
        loss_type: str = "ce",
        class_weights=None,
        focal_gamma: float = 2.0,
        # plus any new params you want to introduce (must also exist in Config or have defaults)
    ) -> None:

- Inside init:
  - load encoder via `AutoModel.from_pretrained(model_name)`
  - build head(s) similarly to BaseModel (reuse build_head pattern or copy minimal)
  - store loss configs and register_buffer("class_weights", ...) like BaseModel
  - validate loss_type and class_weights requirements

- In forward:
  - run encoder on sent + term inputs
  - compute fused representation however you want (NEW logic here)
  - compute logits [B, num_labels]
  - return dict: {"loss": loss or None, "logits": logits}

- Implement a private `_compute_loss(logits, labels)` helper like BaseModel to support ce/weighted_ce/focal.

### Step 2: Export in src/models/__init__.py
- Add:
  from .new_model import NewModel
- Ensure the class name matches CLI usage (cfg.mode must equal "NewModel").

### Step 3: Quick smoke checks (no training code changes)
Add a minimal local import test (no unit tests framework needed):
- Ensure `from src.models import NewModel` works.
- Ensure `helper.get_model(cfg)` can instantiate when cfg.mode="NewModel".
  - Do NOT edit helper.py unless import fails due to naming.

## Notes about Config
- New init parameters must either:
  (a) already exist in Config, OR
  (b) have safe defaults in the model __init__ signature.
- Do not add new Config fields in this task unless absolutely required.

## Deliverables
- `src/models/new_model.py` containing NewModel implementation
- `src/models/__init__.py` updated to export NewModel
- No other files changed.

## Acceptance Criteria
- Running with `--mode NewModel` builds the model successfully.
- Forward pass returns {"loss","logits"} and training loop runs without key errors.
- Loss_type behavior matches existing rules: ce / weighted_ce / focal.
