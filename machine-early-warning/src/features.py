"""Feature engineering for machine failure early warning baseline.

Extracts 147 summary statistics (mean, std, min, max, slope, delta, last)
across 30-cycle windows for all 21 sensors using vectorized NumPy operations.
Adheres strictly to ICD IF-03 data contract.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from src.config import PATHS, SENSOR_COLS, WINDOW_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("features")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_STAT_NAMES = ["mean", "std", "min", "max", "slope", "delta", "last"]


def project_path(path: Union[str, Path]) -> Path:
    """Resolve paths relative to project root if relative."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def validate_no_target_leakage(feature_df: pd.DataFrame) -> None:
    """
    Ensure no target or ground-truth columns leak into feature space.
    Raises ValueError if forbidden keywords (rul, failure, etc.) are present.
    """
    forbidden_tokens = ["rul", "failure", "actual_failure", "future", "target", "label"]
    leaked = []
    for col in feature_df.columns:
        col_lower = col.lower()
        for token in forbidden_tokens:
            if token in col_lower:
                leaked.append(col)
                break

    if leaked:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Features contain forbidden target columns: {leaked}"
        )


def compute_vectorized_slopes(X: np.ndarray) -> np.ndarray:
    """
    Compute linear regression slope for each sensor across the 30-cycle window axis.
    Vectorized formula: slope = sum((t - mean(t)) * (x_t - mean(x))) / sum((t - mean(t))^2)
    
    Parameters
    ----------
    X : np.ndarray, shape (N, W, S)
        Window tensor with N windows, W timesteps (30), S sensors (21).
        
    Returns
    -------
    slopes : np.ndarray, shape (N, S)
    """
    timesteps = X.shape[1]
    t = np.arange(timesteps, dtype=np.float32)
    t_mean = t.mean()
    t_diff = t - t_mean
    denom = np.sum(t_diff ** 2)
    
    if denom == 0:
        return np.zeros((X.shape[0], X.shape[2]), dtype=np.float32)
        
    weights = (t_diff / denom).reshape(1, timesteps, 1)  # shape (1, W, 1)
    slopes = np.sum(X * weights, axis=1)  # shape (N, S)
    return slopes.astype(np.float32)


def extract_features_from_windows(
    X: np.ndarray,
    sensor_names: List[str] = SENSOR_COLS,
) -> pd.DataFrame:
    """
    Extract 7 summary statistics for each sensor across the window dimension.
    Fully vectorized across N windows.

    Parameters
    ----------
    X : np.ndarray, shape (N, 30, S)
    sensor_names : List[str]
        Ordered sensor names matching third dimension of X.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with N rows and S * 7 columns (float32).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N, W, S), got shape {X.shape}")
    
    n_windows, window_len, n_sensors = X.shape
    if n_sensors != len(sensor_names):
        raise ValueError(
            f"Dimension mismatch: X has {n_sensors} sensors, but {len(sensor_names)} names provided"
        )

    # 1. Mean
    means = np.mean(X, axis=1).astype(np.float32)
    
    # 2. Standard Deviation
    stds = np.std(X, axis=1).astype(np.float32)
    
    # 3. Minimum
    mins = np.min(X, axis=1).astype(np.float32)
    
    # 4. Maximum
    maxs = np.max(X, axis=1).astype(np.float32)
    
    # 5. Slope
    slopes = compute_vectorized_slopes(X)
    
    # 6. Delta (last - first)
    deltas = (X[:, -1, :] - X[:, 0, :]).astype(np.float32)
    
    # 7. Last value
    lasts = X[:, -1, :].astype(np.float32)

    # Build dictionary of all 147 feature columns
    feat_dict = {}
    for s_idx, s_name in enumerate(sensor_names):
        feat_dict[f"{s_name}_mean"] = means[:, s_idx]
        feat_dict[f"{s_name}_std"] = stds[:, s_idx]
        feat_dict[f"{s_name}_min"] = mins[:, s_idx]
        feat_dict[f"{s_name}_max"] = maxs[:, s_idx]
        feat_dict[f"{s_name}_slope"] = slopes[:, s_idx]
        feat_dict[f"{s_name}_delta"] = deltas[:, s_idx]
        feat_dict[f"{s_name}_last"] = lasts[:, s_idx]

    df_features = pd.DataFrame(feat_dict)
    
    # Ensure all columns are float32
    df_features = df_features.astype(np.float32)
    
    # Check for NaN / Infs
    if not np.isfinite(df_features.values).all():
        df_features = df_features.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return df_features


def build_features(
    windows_path: Union[str, Path] = PATHS["windows"],
    index_path: Union[str, Path] = PATHS["index"],
    output_path: Union[str, Path] = PATHS["features"],
) -> Tuple[pd.DataFrame, Path]:
    """
    Load IF-02 window handoff files, compute features, validate alignment and leakage,
    and save to Parquet.
    """
    w_path = project_path(windows_path)
    i_path = project_path(index_path)
    out_path = project_path(output_path)

    if not w_path.exists():
        raise FileNotFoundError(f"Windows npz not found: {w_path}")
    if not i_path.exists():
        raise FileNotFoundError(f"Window index not found: {i_path}")

    logger.info(f"Loading windows from {w_path}...")
    with np.load(w_path) as archive:
        if "X" not in archive:
            raise KeyError("Array 'X' missing in windows.npz")
        X = archive["X"].astype(np.float32)

    logger.info(f"Loading window index from {i_path}...")
    index_df = pd.read_parquet(i_path)

    if len(X) != len(index_df):
        raise ValueError(
            f"Length mismatch: X has {len(X)} rows, window_index has {len(index_df)} rows"
        )

    logger.info(f"Extracting features for {len(X):,} windows (shape: {X.shape})...")
    features_df = extract_features_from_windows(X, sensor_names=SENSOR_COLS)

    # Validate no leakage in features
    validate_no_target_leakage(features_df)

    # Prepend window_id as primary key
    features_df.insert(0, "window_id", index_df["window_id"].values.astype(int))

    # Assertions
    assert len(features_df) == len(index_df), "Row count must strictly match window_index"
    assert (features_df["window_id"] == index_df["window_id"]).all(), "Window IDs must strictly align"

    # Save to primary destination
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(features_df):,} feature rows ({features_df.shape[1] - 1} features) to: {out_path}")

    # Also save/copy to data/interim/features.parquet for convenience
    interim_feat_path = PROJECT_ROOT / "data" / "interim" / "features.parquet"
    if interim_feat_path != out_path:
        interim_feat_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(interim_feat_path, index=False)
        logger.info(f"Mirrored features to: {interim_feat_path}")

    return features_df, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="P2: Feature Engineering Pipeline")
    parser.add_argument("--windows", type=str, default=PATHS["windows"], help="Path to windows.npz")
    parser.add_argument("--index", type=str, default=PATHS["index"], help="Path to window_index.parquet")
    parser.add_argument("--output", type=str, default=PATHS["features"], help="Output features parquet")
    args = parser.parse_args()

    build_features(windows_path=args.windows, index_path=args.index, output_path=args.output)


if __name__ == "__main__":
    main()
