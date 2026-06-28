"""SneakerPriceNet -- a deliberately simple feedforward regression net.

The architecture is not the point of this project; the pipeline around it is.
Input -> two hidden layers (ReLU + BatchNorm) -> scalar price_premium output.

Implemented in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for SneakerPriceNet."""

    hidden_dim: int = 64
    dropout_rate: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 20
