"""Stage 4 -- Training (Ray Train + PyTorch + MLflow).

Reads the validated Parquet, does a *temporal* train/val split, trains
SneakerPriceNet inside Ray Train's ``TorchTrainer``, tracks every run to MLflow,
and saves the model artifact to both MLflow and S3.

Two design choices worth calling out:

  * Temporal split, not random. We train on sales before 2023 and validate on
    2023+. A random split would leak future information into the training set
    (the model could "see" a shoe's later sales while predicting its earlier
    ones), inflating validation metrics in a way that wouldn't survive
    production, where we always predict forward in time.
  * Ray Train even on one machine. We wrap the loop in
    ``ray.train.torch.TorchTrainer`` rather than a raw loop or
    ``DataParallel``. On a single machine this is functionally a normal training
    run, but it exercises the exact distributed pattern (worker processes,
    ``prepare_model``, sharded data loaders, ``report``/checkpoint) so scaling to
    multiple workers/GPUs is a config change, not a rewrite.

The pure data-prep functions (temporal split, imputation, standardization) hold
no torch/ray dependency and are imported by the unit tests directly; the heavy
frameworks are imported lazily inside the training functions.

CLI:
    python -m stages.train --run-date 2025-01-01 \\
        --config config/pipeline_config.yaml --num-workers 2
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stages.config import PipelineConfig, load_config
from stages.io import path_join, read_parquet, write_bytes

logger = logging.getLogger("stages.train")

TARGET = "price_premium"
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
SPLIT_YEAR = 2023
SEED = 13
EXPERIMENT_NAME = "sneaker-price-model"


def resolve_tracking_uri() -> str:
    """Resolve the MLflow tracking URI.

    Honours ``MLFLOW_TRACKING_URI`` if set (e.g. the compose server at
    ``http://localhost:5000``). Otherwise defaults to a local **SQLite** store,
    not the legacy file store: MLflow 3.x put the file backend in maintenance
    mode and, more importantly, it cannot back the Phase 5 model registry. A
    DB-backed store works for both tracking and the registry.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    return f"sqlite:///{os.path.abspath('mlflow.db')}"


@dataclass(frozen=True)
class TrainingArrays:
    """Model-ready arrays plus the fitted preprocessing stats.

    The impute/standardize stats are fit on the training split only (no leakage)
    and saved with the model so inference can reproduce the same preprocessing.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_columns: list[str]
    impute_values: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray


def temporal_split(
    df: pd.DataFrame, split_year: int = SPLIT_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by sale year: < split_year is train, >= split_year is validation."""
    years = pd.to_datetime(df["sale_date"]).dt.year
    return df[years < split_year].copy(), df[years >= split_year].copy()


def _fill_nan(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Replace NaNs column-wise with the supplied per-column values."""
    out = x.copy()
    rows, cols = np.where(np.isnan(out))
    out[rows, cols] = np.take(values, cols)
    return out


def prepare_arrays(
    df: pd.DataFrame, split_year: int = SPLIT_YEAR
) -> TrainingArrays:
    """Temporal split -> impute (train means) -> standardize (train stats).

    Raises:
        ValueError: if either side of the split is empty (the temporal boundary
            must actually divide the data, or the run is meaningless).
    """
    train_df, val_df = temporal_split(df, split_year)
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"temporal split at {split_year} left an empty side "
            f"(train={len(train_df)}, val={len(val_df)})"
        )

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)

    # Impute NaNs (e.g. rolling_7d_avg_premium on insufficient-history rows) with
    # train-set column means -- fit on train only.
    impute_values = np.nanmean(x_train, axis=0)
    impute_values = np.where(np.isnan(impute_values), 0.0, impute_values)
    x_train = _fill_nan(x_train, impute_values)
    x_val = _fill_nan(x_val, impute_values)

    # Standardize with train statistics (guard against zero-variance columns).
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    x_train = (x_train - feature_mean) / feature_std
    x_val = (x_val - feature_mean) / feature_std

    return TrainingArrays(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        feature_columns=list(FEATURE_COLUMNS),
        impute_values=impute_values,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _train_func(loop_config: dict) -> None:
    """Ray Train worker loop. Heavy imports are local to the worker process."""
    import ray.train
    import torch
    from ray.train import Checkpoint
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from models.net import ModelConfig, SneakerPriceNet

    torch.manual_seed(loop_config["seed"])
    mcfg = ModelConfig(**loop_config["model_config"])

    x_train = torch.tensor(np.asarray(loop_config["X_train"]), dtype=torch.float32)
    y_train = torch.tensor(np.asarray(loop_config["y_train"]), dtype=torch.float32)
    x_val = torch.tensor(np.asarray(loop_config["X_val"]), dtype=torch.float32)
    y_val = torch.tensor(np.asarray(loop_config["y_val"]), dtype=torch.float32)
    input_dim = x_train.shape[1]

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=mcfg.batch_size,
        shuffle=True,
    )
    loader = ray.train.torch.prepare_data_loader(loader)

    model = ray.train.torch.prepare_model(SneakerPriceNet(input_dim, mcfg))
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg.learning_rate)
    loss_fn = nn.MSELoss()
    rank = ray.train.get_context().get_world_rank()

    for epoch in range(mcfg.num_epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": float(val_loss**0.5),
        }

        # Persist the trained weights in the final checkpoint (rank 0 only).
        if epoch == mcfg.num_epochs - 1 and rank == 0:
            inner = model.module if hasattr(model, "module") else model
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(
                    {
                        "state_dict": inner.state_dict(),
                        "input_dim": input_dim,
                        "model_config": loop_config["model_config"],
                    },
                    os.path.join(tmp, "model.pt"),
                )
                ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))
        else:
            ray.train.report(metrics)


def run(
    config: PipelineConfig,
    num_workers: int = 1,
    model_config: dict | None = None,
    split_year: int = SPLIT_YEAR,
) -> str:
    """Train the model and log it. Returns the MLflow run_id."""
    import mlflow
    import torch
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from models.net import ModelConfig

    df = read_parquet(config.validated_uri())
    arrays = prepare_arrays(df, split_year)
    mcfg = ModelConfig(**model_config) if model_config else ModelConfig()

    loop_config = {
        "X_train": arrays.X_train,
        "y_train": arrays.y_train,
        "X_val": arrays.X_val,
        "y_val": arrays.y_val,
        "model_config": asdict(mcfg),
        "seed": SEED,
    }

    ray_results = os.path.abspath("./ray_results")
    trainer = TorchTrainer(
        _train_func,
        train_loop_config=loop_config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=False),
        run_config=RunConfig(storage_path=ray_results, name=f"train_{config.run_date}"),
    )

    mlflow.set_tracking_uri(resolve_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as active:
        run_id = active.info.run_id
        mlflow.log_params(
            {
                **asdict(mcfg),
                "num_workers": num_workers,
                "run_date": config.run_date,
                "split_year": split_year,
                "n_features": len(FEATURE_COLUMNS),
                "n_train_rows": len(arrays.y_train),
                "n_val_rows": len(arrays.y_val),
            }
        )
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))

        result = trainer.fit()

        metrics_df = result.metrics_dataframe
        if metrics_df is not None:
            for _, row in metrics_df.iterrows():
                step = int(row["epoch"])
                mlflow.log_metric("train_loss", float(row["train_loss"]), step=step)
                mlflow.log_metric("val_loss", float(row["val_loss"]), step=step)
                mlflow.log_metric("val_rmse", float(row["val_rmse"]), step=step)

        final_val_rmse = float(result.metrics["val_rmse"])
        mlflow.log_metric("final_val_rmse", final_val_rmse)

        # Assemble the self-contained model.pt: weights + the preprocessing
        # stats fit on train, so inference reproduces the exact pipeline.
        with result.checkpoint.as_directory() as ckpt_dir:
            ckpt = torch.load(
                os.path.join(ckpt_dir, "model.pt"), map_location="cpu"
            )
        payload = {
            **ckpt,
            "feature_columns": FEATURE_COLUMNS,
            "impute_values": arrays.impute_values.tolist(),
            "feature_mean": arrays.feature_mean.tolist(),
            "feature_std": arrays.feature_std.tolist(),
            "run_date": config.run_date,
        }

        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, "model.pt")
            torch.save(payload, local_path)
            mlflow.log_artifact(local_path, artifact_path="model")
            with open(local_path, "rb") as fh:
                model_uri = write_bytes(
                    fh.read(), path_join(config.model_uri(run_id), "model.pt")
                )
        mlflow.log_param("model_s3_uri", model_uri)

    logger.info(
        "training complete run_id=%s val_rmse=%.5f model -> %s",
        run_id, final_val_rmse, model_uri,
    )
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with Ray Train + PyTorch.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--num-workers", type=int, default=1, help="Ray workers.")
    parser.add_argument(
        "--split-year",
        type=int,
        default=SPLIT_YEAR,
        help="Temporal train/val boundary: sales < year train, >= year validate.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    run_id = run(config, num_workers=args.num_workers, split_year=args.split_year)
    logger.info("training stage complete, mlflow run_id=%s", run_id)


if __name__ == "__main__":
    main()
