"""Create leakage-safe, machine-level split sliding windows for the LSTM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HORIZON, PATHS, RANDOM_SEED, SENSOR_COLS, STRIDE, TEST_MACHINE_FRAC, WINDOW_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def assign_machine_splits(machine_ids: np.ndarray) -> set[int]:
    machine_ids = np.sort(np.unique(machine_ids))
    if len(machine_ids) < 2:
        raise ValueError("At least two machines are required for a machine-level split")
    count = min(max(1, round(len(machine_ids) * TEST_MACHINE_FRAC)), len(machine_ids) - 1)
    return set(np.random.default_rng(RANDOM_SEED).choice(machine_ids, size=count, replace=False).tolist())


def make_windows(raw_path: str | Path = PATHS["raw"]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw = pd.read_parquet(project_path(raw_path)).sort_values(["machine_id", "timestamp"], kind="stable")
    required = {"machine_id", "timestamp", *SENSOR_COLS, "rul"}
    missing = required.difference(raw.columns)
    if missing or raw.duplicated(["machine_id", "timestamp"]).any():
        raise ValueError(f"Invalid raw data; missing columns: {sorted(missing)}")
    test_machines = assign_machine_splits(raw["machine_id"].to_numpy())
    windows, labels, rows = [], [], []
    for machine_id, group in raw.groupby("machine_id", sort=True):
        group = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        values = group[SENSOR_COLS].to_numpy(dtype=np.float32)
        timestamps, rul = group["timestamp"].to_numpy(), group["rul"].to_numpy()
        for start in range(0, len(group) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE - 1
            label = int(rul[end] <= HORIZON)
            windows.append(values[start : end + 1])
            labels.append(label)
            rows.append({"window_id": len(rows), "machine_id": int(machine_id), "window_start": int(timestamps[start]), "window_end": int(timestamps[end]), "rul_at_end": int(rul[end]), "label": label, "split": "test" if machine_id in test_machines else "train"})
    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int8)
    index = pd.DataFrame(rows).astype({"label": "int8"})
    if X.shape != (len(index), WINDOW_SIZE, len(SENSOR_COLS)) or not np.array_equal(y, index["label"].to_numpy(dtype=np.int8)):
        raise ValueError("Invalid window handoff")
    if set(index.loc[index["split"] == "train", "machine_id"]).intersection(index.loc[index["split"] == "test", "machine_id"]):
        raise ValueError("Machine split leakage detected")
    return X, y, index


def save_windows() -> tuple[Path, Path]:
    X, y, index = make_windows()
    windows_path, index_path = project_path(PATHS["windows"]), project_path(PATHS["index"])
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(windows_path, X=X, y=y)
    index.to_parquet(index_path, index=False)
    print(f"Saved {len(X):,} windows ({int(y.sum()):,} positive); {index.loc[index['split'] == 'train', 'machine_id'].nunique()} train machines, {index.loc[index['split'] == 'test', 'machine_id'].nunique()} test machines")
    return windows_path, index_path


if __name__ == "__main__":
    save_windows()
