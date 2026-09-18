"""Convert NASA C-MAPSS FD001 run-to-failure data to the ICD raw contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PATHS, SENSOR_COLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMAPSS_COLUMNS = ["machine_id", "timestamp", "setting_1", "setting_2", "setting_3", *SENSOR_COLS]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cmapss(dataset: str = "FD001") -> pd.DataFrame:
    """Load a C-MAPSS train split and derive RUL/failure-event fields.

    C-MAPSS `train_*.txt` trajectories all run through failure, unlike the
    corresponding test files, so the last cycle for each engine is ground
    truth failure and can safely form the early-warning label downstream.
    """
    dataset = dataset.upper()
    if dataset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("dataset must be one of FD001, FD002, FD003, FD004")
    source = PROJECT_ROOT / "CMaps" / f"train_{dataset}.txt"
    if not source.exists():
        raise FileNotFoundError(f"C-MAPSS input not found: {source}")

    frame = pd.read_csv(source, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(f"Expected {len(CMAPSS_COLUMNS)} C-MAPSS columns, got {frame.shape[1]}")
    frame = frame.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(drop=True)
    last_cycle = frame.groupby("machine_id", sort=False)["timestamp"].transform("max")
    frame["rul"] = (last_cycle - frame["timestamp"]).astype("int32")
    frame["failure_event"] = (frame["rul"] == 0).astype("int8")
    raw = frame[["machine_id", "timestamp", *SENSOR_COLS, "rul", "failure_event"]].copy()
    raw["machine_id"] = raw["machine_id"].astype("int32")
    raw["timestamp"] = raw["timestamp"].astype("int32")
    raw[SENSOR_COLS] = raw[SENSOR_COLS].astype("float32")
    return raw


def save_raw_data(dataset: str = "FD001", output: str | Path = PATHS["raw"]) -> Path:
    raw = load_cmapss(dataset)
    if raw.duplicated(["machine_id", "timestamp"]).any():
        raise ValueError("C-MAPSS source has duplicated machine/cycle rows")
    output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output, index=False)
    print(f"Saved {len(raw):,} rows from {raw.machine_id.nunique()} machines to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create raw ICD data from NASA C-MAPSS.")
    parser.add_argument("--dataset", default="FD001", help="C-MAPSS subset, default FD001.")
    args = parser.parse_args()
    save_raw_data(args.dataset)


if __name__ == "__main__":
    main()
