"""Train and score the sequence model for machine-failure early warning.

Input contract
--------------
``data/interim/windows.npz`` contains ``X`` with shape ``(N, 30, 21)`` and
``y`` with shape ``(N,)``.  ``data/interim/window_index.parquet`` has one row
per window, in exactly the same order, including a machine-level ``split``.

The train/test split is deliberately *not* recreated here: it is assigned by
``windowing.py`` at the machine level.  This avoids nearly identical windows
from the same machine leaking into both train and test sets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PATHS, RANDOM_SEED, SENSOR_COLS, WINDOW_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lstm.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
TRAINING_INFO_PATH = PROJECT_ROOT / "models" / "lstm_training_info.json"


def project_path(path: str | Path) -> Path:
    """Resolve ICD paths relative to the project, regardless of shell cwd."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def set_seed(seed: int) -> None:
    """Make the split and TensorFlow initialization repeatable where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_training_data(
    windows_path: str | Path = PATHS["windows"],
    index_path: str | Path = PATHS["index"],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and validate the IF-02 handoff files."""
    windows_path = project_path(windows_path)
    index_path = project_path(index_path)
    if not windows_path.exists():
        raise FileNotFoundError(f"Windows file not found: {windows_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Window index not found: {index_path}")

    with np.load(windows_path) as archive:
        if "X" not in archive or "y" not in archive:
            raise ValueError("windows.npz must contain arrays named X and y")
        X = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.int8)
    index = pd.read_parquet(index_path).reset_index(drop=True)

    required_columns = {"window_id", "machine_id", "split", "label"}
    missing = required_columns.difference(index.columns)
    if missing:
        raise ValueError(f"window_index is missing required columns: {sorted(missing)}")
    if X.ndim != 3 or X.shape[1:] != (WINDOW_SIZE, len(SENSOR_COLS)):
        raise ValueError(
            f"Expected X shape (N, {WINDOW_SIZE}, {len(SENSOR_COLS)}); got {X.shape}"
        )
    if len(X) != len(y) or len(X) != len(index):
        raise ValueError("X, y, and window_index must have the same number of rows")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must be binary (0 or 1)")
    if not np.array_equal(y, index["label"].to_numpy(dtype=np.int8)):
        raise ValueError("y and window_index.label differ; the handoff is inconsistent")
    if index["window_id"].duplicated().any():
        raise ValueError("window_index.window_id must be unique")
    if not set(index["split"].unique()).issubset({"train", "test"}):
        raise ValueError("split must contain only 'train' and 'test'")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity")
    return X, y, index


def machine_validation_split(
    index: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, test masks with validation split by machine."""
    train_machines = np.sort(index.loc[index["split"] == "train", "machine_id"].unique())
    test_machines = np.sort(index.loc[index["split"] == "test", "machine_id"].unique())
    if not len(train_machines) or not len(test_machines):
        raise ValueError("Both train and test splits must contain at least one machine")
    if set(train_machines).intersection(test_machines):
        raise ValueError("A machine appears in both train and test: data leakage")
    if len(train_machines) < 2:
        raise ValueError("At least two training machines are needed for a validation split")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_machines)
    n_validation = max(1, round(len(train_machines) * validation_fraction))
    n_validation = min(n_validation, len(train_machines) - 1)
    validation_machines = shuffled[:n_validation]

    train_mask = (index["split"] == "train") & ~index["machine_id"].isin(validation_machines)
    validation_mask = (index["split"] == "train") & index["machine_id"].isin(validation_machines)
    test_mask = index["split"] == "test"
    return train_mask.to_numpy(), validation_mask.to_numpy(), test_mask.to_numpy()


def fit_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-sensor standardization on train readings only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0, dtype=np.float64)
    scale = X_train.reshape(-1, X_train.shape[-1]).std(axis=0, dtype=np.float64)
    # Constant sensors carry no sequence signal; keep their normalized value at 0.
    scale[scale < 1e-7] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((X - mean[None, None, :]) / scale[None, None, :]).astype(np.float32)


def save_scaler(mean: np.ndarray, scale: np.ndarray) -> None:
    """Save a lightweight, joblib-compatible scaler artifact for ``score.py``."""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("joblib is required; install requirements.txt") from exc

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "mean": mean,
            "scale": scale,
            "sensor_columns": SENSOR_COLS,
            "window_size": WINDOW_SIZE,
            "kind": "per_sensor_standardization",
        },
        SCALER_PATH,
    )


def build_model(timesteps: int, features: int) -> Any:
    """Build the CPU-friendly architecture specified in the ICD."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "TensorFlow is required for LSTM training. Install requirements.txt first."
        ) from exc

    inputs = tf.keras.Input(shape=(timesteps, features), name="sensor_window")
    x = tf.keras.layers.BatchNormalization(name="input_batch_norm")(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_64")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.LSTM(32, name="lstm_32")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_16")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="machine_failure_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balance positive and negative windows without resampling time series."""
    counts = np.bincount(y_train.astype(int), minlength=2)
    if not counts[0] or not counts[1]:
        raise ValueError("Training split must contain both positive and negative windows")
    total = len(y_train)
    return {label: total / (2.0 * count) for label, count in enumerate(counts)}


def write_predictions(index: pd.DataFrame, risks: np.ndarray, output_path: str | Path) -> Path:
    """Add/replace ``risk_lstm`` while preserving any baseline prediction column."""
    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_columns = ["window_id", "machine_id", "window_end", "label", "split"]
    missing = set(base_columns).difference(index.columns)
    if missing:
        raise ValueError(f"window_index cannot form predictions: missing {sorted(missing)}")
    current = index.loc[:, base_columns].copy()
    current["risk_lstm"] = np.asarray(risks, dtype=np.float32)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if "window_id" in existing and existing["window_id"].is_unique:
            preserved = existing.drop(columns=[c for c in current.columns if c in existing], errors="ignore")
            current = current.merge(preserved, on="window_id", how="left", validate="one_to_one")
    current.to_parquet(output_path, index=False)
    return output_path


def train(
    epochs: int = 30,
    batch_size: int = 128,
    validation_fraction: float = 0.2,
    patience: int = 5,
    verbose: int = 2,
) -> dict[str, Any]:
    """Train, save, and score the LSTM. Returns a concise run summary."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    set_seed(RANDOM_SEED)
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required; install requirements.txt first.") from exc
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    X, y, index = load_training_data()
    train_mask, validation_mask, test_mask = machine_validation_split(
        index, validation_fraction, RANDOM_SEED
    )
    mean, scale = fit_normalizer(X[train_mask])
    X_normalized = normalize(X, mean, scale)
    weights = class_weights(y[train_mask])

    model = build_model(X.shape[1], X.shape[2])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_pr_auc", mode="max", save_best_only=True
        ),
    ]
    history = model.fit(
        X_normalized[train_mask],
        y[train_mask],
        validation_data=(X_normalized[validation_mask], y[validation_mask]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=verbose,
    )

    # ModelCheckpoint has already saved the best model. Save the exact matching scaler.
    save_scaler(mean, scale)
    risks = model.predict(X_normalized, batch_size=batch_size, verbose=0).reshape(-1)
    predictions_path = write_predictions(index, risks, PATHS["preds"])

    summary: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "predictions_path": str(predictions_path),
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(validation_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "train_machines": int(index.loc[train_mask, "machine_id"].nunique()),
        "validation_machines": int(index.loc[validation_mask, "machine_id"].nunique()),
        "test_machines": int(index.loc[test_mask, "machine_id"].nunique()),
        "class_weights": {str(key): value for key, value in weights.items()},
        "epochs_completed": len(history.history["loss"]),
        "best_validation_pr_auc": float(max(history.history["val_pr_auc"])),
    }
    TRAINING_INFO_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the machine-failure LSTM sequence model.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--quiet", action="store_true", help="Suppress Keras epoch logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        verbose=0 if args.quiet else 2,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
