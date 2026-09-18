"""Unit tests for P2 baseline model training, scoring, and prediction merging."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train_baseline import (
    load_features_and_index,
    merge_predictions,
    split_by_machine,
)
from src.score import load_model, predict_risk


@pytest.fixture
def dummy_data(tmp_path):
    np.random.seed(42)
    n_windows = 60
    n_features = 147
    
    # 3 machines: 1, 2 in train, 3 in test
    rows_idx = []
    for i in range(n_windows):
        m = (i // 20) + 1
        split = "test" if m == 3 else "train"
        label = 1 if (i % 20) >= 15 else 0
        rows_idx.append({
            "window_id": i,
            "machine_id": m,
            "window_start": (i % 20) * 5 + 1,
            "window_end": (i % 20) * 5 + 30,
            "rul_at_end": max(0, 30 - (i % 20)),
            "label": label,
            "split": split,
        })
    df_idx = pd.DataFrame(rows_idx)
    
    # Features
    feat_dict = {"window_id": df_idx["window_id"].values}
    for f in range(n_features):
        feat_dict[f"sensor_{f % 21 + 1}_stat{f // 21}"] = np.random.normal(500, 10, n_windows).astype(np.float32)
    df_feat = pd.DataFrame(feat_dict)
    
    idx_path = tmp_path / "window_index.parquet"
    feat_path = tmp_path / "features.parquet"
    preds_path = tmp_path / "predictions.parquet"
    
    df_idx.to_parquet(idx_path, index=False)
    df_feat.to_parquet(feat_path, index=False)
    
    return {
        "df_idx": df_idx,
        "df_feat": df_feat,
        "idx_path": idx_path,
        "feat_path": feat_path,
        "preds_path": preds_path,
    }


def test_train_test_machine_separation(dummy_data):
    """Verify that machine-level split strictly prevents machine leakage."""
    df_idx = dummy_data["df_idx"]
    train_mask, val_mask, test_mask = split_by_machine(df_idx, validation_frac=0.5, seed=42)
    
    train_machs = set(df_idx.loc[train_mask, "machine_id"])
    val_machs = set(df_idx.loc[val_mask, "machine_id"])
    test_machs = set(df_idx.loc[test_mask, "machine_id"])
    
    assert len(train_machs.intersection(test_machs)) == 0
    assert len(val_machs.intersection(test_machs)) == 0
    assert len(train_machs.intersection(val_machs)) == 0
    assert test_machs == {3}


def test_safe_prediction_merging(dummy_data):
    """Verify that merging risk_baseline preserves existing risk_lstm without row reordering."""
    df_idx = dummy_data["df_idx"]
    preds_path = dummy_data["preds_path"]
    
    # Simulate existing predictions file with P3's risk_lstm
    existing_preds = df_idx[["window_id", "machine_id", "window_end", "label", "split"]].copy()
    np.random.seed(123)
    existing_preds["risk_lstm"] = np.random.uniform(0.1, 0.9, len(df_idx)).astype(np.float32)
    existing_preds.to_parquet(preds_path, index=False)
    
    # Merge new risk_baseline
    new_risk_baseline = np.random.uniform(0.0, 1.0, len(df_idx)).astype(np.float32)
    merge_predictions(df_idx, new_risk_baseline, output_path=preds_path)
    
    merged = pd.read_parquet(preds_path)
    assert "risk_baseline" in merged.columns
    assert "risk_lstm" in merged.columns
    assert (merged["window_id"] == df_idx["window_id"]).all()
    np.testing.assert_allclose(merged["risk_baseline"].values, new_risk_baseline, rtol=1e-5)
    np.testing.assert_allclose(merged["risk_lstm"].values, existing_preds["risk_lstm"].values, rtol=1e-5)


def test_score_interface():
    """Verify load_model and predict_risk functions on baseline artifact."""
    model = load_model("baseline")
    assert hasattr(model, "predict_proba")
    
    # Test batch prediction
    batch = np.random.normal(500, 10, size=(5, 147)).astype(np.float32)
    risks = predict_risk(model, batch)
    
    assert len(risks) == 5
    assert (risks >= 0.0).all() and (risks <= 1.0).all()
    assert np.isfinite(risks).all()
