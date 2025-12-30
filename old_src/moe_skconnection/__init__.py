"""moe_skconnection package.

Scaffold: sequence-level MoE provides delta logits via skip-style residual add.
"""

from .model import build_model, SkBertConcatClassifier  # noqa: F401
