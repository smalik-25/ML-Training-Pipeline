"""SneakerPriceNet -- a deliberately simple feedforward regression net.

The architecture is not the point of this project; the pipeline around it is.
Input -> two hidden blocks (Linear + BatchNorm + ReLU + Dropout) -> a single
scalar predicting price_premium. Kept small and unexotic on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for SneakerPriceNet."""

    hidden_dim: int = 64
    dropout_rate: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 20


class SneakerPriceNet(nn.Module):
    """Two-hidden-layer MLP regressor for price_premium.

    Args:
        input_dim: number of input features.
        config: hyperparameters (hidden width, dropout).
    """

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        h = config.hidden_dim
        p = config.dropout_rate
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a 1-D tensor of predicted premiums (batch,)."""
        return self.net(x).squeeze(-1)
