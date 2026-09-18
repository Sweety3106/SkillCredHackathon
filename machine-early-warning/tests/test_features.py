"""Unit tests for P2 feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.config import SENSOR_COLS
from src.features import (
    compute_vectorized_slopes,
    extract_features_from_windows,
    validate_no_target_leakage,
)


@pytest.fixture
def sample_windows():
    np.random.seed(42)
    n_windows = 25
    timesteps = 30
    n_sensors = 21
    X = np.random.normal(500, 15, size=(n_windows, timesteps, n_sensors)).astype(np.float32)
    return X


def test_feature_count(sample_windows):
    """Test that exactly 21 sensors * 7 statistics = 147 features are extracted."""
    df = extract_features_from_windows(sample_windows, sensor_names=SENSOR_COLS)
    expected_feature_count = len(SENSOR_COLS) * 7
    assert df.shape == (25, expected_feature_count)
    assert len(df.columns) == 147


def test_feature_names(sample_windows):
    """Test that feature column names follow the standard pattern: {sensor}_{stat}."""
    df = extract_features_from_windows(sample_windows, sensor_names=SENSOR_COLS)
    stats = {"mean", "std", "min", "max", "slope", "delta", "last"}
    for col in df.columns:
        parts = col.split("_")
        assert len(parts) == 3, f"Unexpected column format: {col}"
        assert f"{parts[0]}_{parts[1]}" in SENSOR_COLS
        assert parts[2] in stats


def test_no_rul_features(sample_windows):
    """Test that feature validation rejects any ground-truth or RUL leakage."""
    df = extract_features_from_windows(sample_windows, sensor_names=SENSOR_COLS)
    
    # Clean features must pass
    validate_no_target_leakage(df)

    # Injected RUL column must be caught and raise ValueError
    leaked_df = df.copy()
    leaked_df["sensor_1_rul"] = 100.0
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        validate_no_target_leakage(leaked_df)

    # Injected failure column must also be caught
    leaked_df2 = df.copy()
    leaked_df2["failure_flag"] = 0
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        validate_no_target_leakage(leaked_df2)


def test_slope_calculation():
    """Test mathematical accuracy of vectorized slope calculation."""
    # Construct a window where sensor 1 is strictly linear: y = 2*t + 5
    W = 30
    t = np.arange(W, dtype=np.float32)
    sensor_linear = 2.0 * t + 5.0
    
    X = np.zeros((1, W, 21), dtype=np.float32)
    X[0, :, 0] = sensor_linear
    
    slopes = compute_vectorized_slopes(X)
    assert np.isclose(slopes[0, 0], 2.0, atol=1e-5)


def test_window_alignment(sample_windows):
    """Test that delta equals last minus first, and min <= mean <= max."""
    df = extract_features_from_windows(sample_windows, sensor_names=SENSOR_COLS)
    
    for s in SENSOR_COLS[:3]:
        deltas = df[f"{s}_delta"].values
        lasts = df[f"{s}_last"].values
        firsts = sample_windows[:, 0, SENSOR_COLS.index(s)]
        np.testing.assert_allclose(deltas, lasts - firsts, rtol=1e-5)

        mins = df[f"{s}_min"].values
        means = df[f"{s}_mean"].values
        maxs = df[f"{s}_max"].values
        assert (mins <= means).all()
        assert (means <= maxs).all()
