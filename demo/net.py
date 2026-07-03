"""Self-contained model + inference for the demo Space.

Vendored from the main project (models/net.py + stages/inference.py) so the
Hugging Face Space is a standalone folder that doesn't need the whole package
installed. Kept deliberately small and in sync with the trained model.pt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURE_COLUMNS = [
    "days_since_release",
    "size_us",
    "retail_price",
    "size_premium",
    "release_type_encoded",
    "rolling_7d_avg_premium",
    "search_index_7d_pre_drop",
    "brand_avg_premium",
]


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    dropout_rate: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 20


class SneakerPriceNet(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        h, p = config.hidden_dim, config.dropout_rate
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
        return self.net(x).squeeze(-1)


@dataclass
class ModelBundle:
    model: nn.Module
    feature_columns: list[str]
    impute_values: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    run_date: str | None = None


def load_bundle(model_path: str) -> ModelBundle:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SneakerPriceNet(payload["input_dim"], ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return ModelBundle(
        model=model,
        feature_columns=list(payload["feature_columns"]),
        impute_values=np.asarray(payload["impute_values"], dtype=float),
        feature_mean=np.asarray(payload["feature_mean"], dtype=float),
        feature_std=np.asarray(payload["feature_std"], dtype=float),
        run_date=payload.get("run_date"),
    )


def build_matrix(df: pd.DataFrame, bundle: ModelBundle) -> np.ndarray:
    """Impute (with saved training means) then standardize (saved train stats).

    Same transform the training stage fit, applied from the persisted stats, so
    the demo prediction matches what the model actually learned.
    """
    x = df[bundle.feature_columns].to_numpy(dtype=float, copy=True)
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = np.take(bundle.impute_values, cols)
    return (x - bundle.feature_mean) / bundle.feature_std


def predict(bundle: ModelBundle, records: list[dict]) -> np.ndarray:
    df = pd.DataFrame(records)
    x = build_matrix(df, bundle)
    with torch.no_grad():
        return bundle.model(torch.tensor(x, dtype=torch.float32)).numpy()
