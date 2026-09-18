"""Create leakage-safe, machine-level split sliding windows for the LSTM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HORIZON, PATHS, RANDOM_SEED, SENSOR_COLS, STRIDE, TEST_MACHINE_FRAC, WINDOW_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def assign_machine_splits(machine_ids: np.ndarray) -> set[int]:
    """Choose test machines once; no machine can appear in both partitions."""
    machine_ids = np.sort(np.unique(machine_ids))
    if len(machine_ids) < 2:
        raise ValueError("At least two machines are required for a machine-level split")
    count = max(1, round(len(machine_ids) * TEST_MACHINE_FRAC))
    count = min(count, len(machine_ids) - 1)
    return set(np.random.default_rng(RANDOM_SEED).choice(machine_ids, size=count, replace=False).tolist())


def make_windows(raw_path: str | Path = PATHS["raw"]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw_path = project_path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}. Run src.data_load first.")
    raw = pd.read_parquet(raw_path).sort_values(["machine_id", "timestamp"], kind="stable")
    required = {"machine_id", "timestamp", *SENSOR_COLS, "rul"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    if raw.duplicated(["machine_id", "timestamp"]).any():
        raise ValueError("Raw data contains duplicate machine/timestamp pairs")

    test_machines = assign_machine_splits(raw["machine_id"].to_numpy())
    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, int | str]] = []
    for machine_id, group in raw.groupby("machine_id", sort=True):
        group = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        values = group[SENSOR_COLS].to_numpy(dtype=np.float32)
        timestamps = group["timestamp"].to_numpy(dtype=np.int32)
        rul = group["rul"].to_numpy(dtype=np.int32)
        if len(group) < WINDOW_SIZE:
            continue
        split = "test" if machine_id in test_machines else "train"
        for start in range(0, len(group) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE - 1
            rul_at_end = int(rul[end])
            label = int(rul_at_end <= HORIZON)
            windows.append(values[start : end + 1])
            labels.append(label)
            metadata.append(
                {
                    "window_id": len(metadata),
                    "machine_id": int(machine_id),
                    "window_start": int(timestamps[start]),
                    "window_end": int(timestamps[end]),
                    "rul_at_end": rul_at_end,
                    "label": label,
                    "split": split,
                }
            )
    if not windows:
        raise ValueError("No valid windows were generated")
    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int8)
    index = pd.DataFrame(metadata).astype({"label": "int8"})
    validate_handoff(X, y, index)
    return X, y, index


def validate_handoff(X: np.ndarray, y: np.ndarray, index: pd.DataFrame) -> None:
    """Assert every IF-02 invariant before writing artifacts."""
    assert X.dtype == np.float32 and X.ndim == 3 and X.shape[1:] == (WINDOW_SIZE, len(SENSOR_COLS))
    assert y.dtype == np.int8 and X.shape[0] == y.shape[0] == len(index)
    assert np.array_equal(index["window_id"].to_numpy(), np.arange(len(index)))
    assert np.array_equal(y, index["label"].to_numpy(dtype=np.int8))
    assert set(index["split"].unique()) == {"train", "test"}
    assert not set(index.loc[index.split == "train", "machine_id"]).intersection(
        set(index.loc[index.split == "test", "machine_id"])
    )
    assert np.isfinite(X).all()


def save_windows() -> tuple[Path, Path]:
    X, y, index = make_windows()
    windows_path, index_path = project_path(PATHS["windows"]), project_path(PATHS["index"])
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(windows_path, X=X, y=y)
    index.to_parquet(index_path, index=False)
    print(
        f"Saved {len(X):,} windows ({int(y.sum()):,} positive); "
        f"{index.loc[index.split == 'train', 'machine_id'].nunique()} train machines, "
        f"{index.loc[index.split == 'test', 'machine_id'].nunique()} test machines"
    )
    return windows_path, index_path


if __name__ == "__main__":
    argparse.ArgumentParser(description="Create C-MAPSS LSTM windows.").parse_args()
    save_windows()
