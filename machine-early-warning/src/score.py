"""Unified model scoring interface for baseline and sequence models.

Adheres to ICD IF-04 contract to unblock P5 (FastAPI API) and unified inference:
- load_model(name): Loads 'baseline' (Joblib) or 'lstm' (Keras).
- predict_risk(model, batch): Returns risk probabilities in [0.0, 1.0].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Union

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
LSTM_MODEL_PATH = PROJECT_ROOT / "models" / "lstm.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_columns.json"


def load_model(name: str) -> Any:
    """
    Load model artifact by name.
    
    Parameters
    ----------
    name : str
        'baseline' or 'lstm'
        
    Returns
    -------
    model : Any
        Loaded model instance.
    """
    name_clean = name.strip().lower()
    if name_clean == "baseline":
        if not BASELINE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Baseline model not found at {BASELINE_MODEL_PATH}")
        return joblib.load(BASELINE_MODEL_PATH)
    elif name_clean == "lstm":
        if not LSTM_MODEL_PATH.exists():
            raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_PATH}")
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError("TensorFlow required to load LSTM model") from e
        return tf.keras.models.load_model(LSTM_MODEL_PATH)
    else:
        raise ValueError(f"Unknown model name '{name}'. Expected 'baseline' or 'lstm'")


def load_feature_columns() -> List[str]:
    """Load ordered feature column list for baseline model."""
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found at {FEATURE_COLS_PATH}")
    with open(FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


def predict_risk(model: Any, batch: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Generate risk prediction probabilities in [0.0, 1.0].

    Parameters
    ----------
    model : Any
        Loaded baseline or LSTM model.
    batch : pd.DataFrame or np.ndarray
        - For baseline: DataFrame containing the 147 feature columns (or 2D numpy array).
        - For LSTM: 3D numpy array of shape (B, 30, 21).

    Returns
    -------
    risk : np.ndarray
        1D float array of shape (B,) with risk probabilities in [0.0, 1.0].
    """
    # Case 1: Keras LSTM model
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        X = np.asarray(batch, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"LSTM expects 3D batch (B, 30, 21), got shape {X.shape}")
        
        # Apply normalization if scaler artifact exists
        if SCALER_PATH.exists():
            scaler_data = joblib.load(SCALER_PATH)
            mean = scaler_data["mean"]
            scale = scaler_data["scale"]
            X = (X - mean[None, None, :]) / scale[None, None, :]
            
        preds = model.predict(X, verbose=0)
        return np.clip(preds.reshape(-1), 0.0, 1.0).astype(np.float32)

    # Case 2: Scikit-learn Baseline (HistGradientBoostingClassifier / LogisticRegression)
    elif hasattr(model, "predict_proba"):
        if isinstance(batch, pd.DataFrame):
            if FEATURE_COLS_PATH.exists():
                expected_cols = load_feature_columns()
                # If window_id is present, ignore it
                available = [c for c in expected_cols if c in batch.columns]
                if len(available) == len(expected_cols):
                    X_input = batch[expected_cols].to_numpy(dtype=np.float32)
                else:
                    X_input = batch.to_numpy(dtype=np.float32)
            else:
                X_input = batch.drop(columns=["window_id"], errors="ignore").to_numpy(dtype=np.float32)
        else:
            X_input = np.asarray(batch, dtype=np.float32)
            
        probs = model.predict_proba(X_input)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.clip(probs[:, 1], 0.0, 1.0).astype(np.float32)
        elif probs.ndim == 1:
            return np.clip(probs, 0.0, 1.0).astype(np.float32)
        else:
            return np.clip(probs.reshape(-1), 0.0, 1.0).astype(np.float32)

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
