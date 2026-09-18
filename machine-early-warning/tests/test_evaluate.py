"""
Unit tests for P4 Evaluation, Metrics, Graphs & Presentation pipeline.
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    validate_predictions,
    filter_test_data,
    calculate_classification_metrics,
    calculate_machine_false_alarm_rate,
    calculate_lead_time,
    calculate_threshold_sweep,
    resolve_failure_cycles,
    run_leakage_audit,
    run_evaluation,
)


@pytest.fixture
def synthetic_predictions():
    """
    Generate small deterministic synthetic predictions dataframe.
    Contains:
    - machine 1: failing machine, caught before failure
    - machine 2: failing machine, missed (never alerted)
    - machine 3: non-failing machine, no false alarm
    - machine 4: non-failing machine, with false alarm
    """
    rows = []
    # Machine 1: test, fails at cycle 50, alert starts at cycle 35 (risk 0.8)
    for c in range(20, 51):
        risk = 0.85 if c >= 35 else 0.15
        label = 1 if (50 - c) <= 15 else 0
        rows.append({
            "window_id": len(rows),
            "machine_id": 1,
            "window_end": c,
            "label": label,
            "risk_baseline": risk,
            "risk_lstm": risk,
            "split": "test",
            "actual_failure_cycle": 50,
        })

    # Machine 2: test, fails at cycle 60, never alerts (risk 0.20) -> missed
    for c in range(30, 61):
        label = 1 if (60 - c) <= 15 else 0
        rows.append({
            "window_id": len(rows),
            "machine_id": 2,
            "window_end": c,
            "label": label,
            "risk_baseline": 0.20,
            "risk_lstm": 0.22,
            "split": "test",
            "actual_failure_cycle": 60,
        })

    # Machine 3: test, non-failing, max risk 0.30 -> normal, no false alarm
    for c in range(20, 51):
        rows.append({
            "window_id": len(rows),
            "machine_id": 3,
            "window_end": c,
            "label": 0,
            "risk_baseline": 0.30,
            "risk_lstm": 0.25,
            "split": "test",
            "actual_failure_cycle": None,
        })

    # Machine 4: test, non-failing, max risk 0.75 -> false alarm
    for c in range(20, 51):
        risk = 0.75 if c == 30 else 0.10
        rows.append({
            "window_id": len(rows),
            "machine_id": 4,
            "window_end": c,
            "label": 0,
            "risk_baseline": risk,
            "risk_lstm": risk,
            "split": "test",
            "actual_failure_cycle": None,
        })

    # Machine 5: train split
    for c in range(20, 40):
        rows.append({
            "window_id": len(rows),
            "machine_id": 5,
            "window_end": c,
            "label": 0,
            "risk_baseline": 0.10,
            "risk_lstm": 0.10,
            "split": "train",
            "actual_failure_cycle": 40,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 1. INPUT VALIDATION TESTS
# ==============================================================================

def test_validate_predictions_success(synthetic_predictions):
    info = validate_predictions(synthetic_predictions)
    assert info["total_rows"] == len(synthetic_predictions)
    assert info["total_machines"] == 5
    assert "test" in info["splits"]


def test_validate_predictions_missing_columns(synthetic_predictions):
    df_missing = synthetic_predictions.drop(columns=["risk_baseline"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_predictions(df_missing)


def test_validate_predictions_invalid_labels(synthetic_predictions):
    df_bad = synthetic_predictions.copy()
    df_bad.loc[0, "label"] = 99
    with pytest.raises(ValueError, match="non-binary values"):
        validate_predictions(df_bad)


def test_validate_predictions_risk_out_of_bounds(synthetic_predictions):
    df_bad = synthetic_predictions.copy()
    df_bad.loc[0, "risk_baseline"] = 1.45
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        validate_predictions(df_bad)


def test_validate_predictions_duplicate_window_id(synthetic_predictions):
    df_bad = synthetic_predictions.copy()
    df_bad.loc[1, "window_id"] = df_bad.loc[0, "window_id"]
    with pytest.raises(ValueError, match="Duplicate 'window_id'"):
        validate_predictions(df_bad)


def test_validate_predictions_missing_test_split(synthetic_predictions):
    df_no_test = synthetic_predictions[synthetic_predictions["split"] == "train"].copy()
    with pytest.raises(ValueError, match="No 'test' split found"):
        validate_predictions(df_no_test)


# ==============================================================================
# 2. TEST-ONLY FILTERING
# ==============================================================================

def test_filter_test_data(synthetic_predictions):
    test_df = filter_test_data(synthetic_predictions)
    assert (test_df["split"] == "test").all()
    assert 5 not in test_df["machine_id"].values  # machine 5 was train


# ==============================================================================
# 3. CLASSIFICATION METRICS
# ==============================================================================

def test_calculate_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.8])
    metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_conversion():
    y_true = np.array([0, 1])
    y_prob = np.array([0.45, 0.45])

    # Threshold 0.40 -> both predicted 1
    m1 = calculate_classification_metrics(y_true, y_prob, threshold=0.40)
    assert m1["confusion_matrix"] == [[0, 1], [0, 1]]

    # Threshold 0.50 -> both predicted 0
    m2 = calculate_classification_metrics(y_true, y_prob, threshold=0.50)
    assert m2["confusion_matrix"] == [[1, 0], [1, 0]]


# ==============================================================================
# 4. MACHINE-LEVEL FALSE ALARM RATE
# ==============================================================================

def test_machine_level_false_alarm_rate(synthetic_predictions):
    test_df = synthetic_predictions[synthetic_predictions["split"] == "test"]
    failure_info = resolve_failure_cycles(test_df)

    # Threshold 0.50:
    # Machine 3: max risk 0.30 -> no false alarm
    # Machine 4: max risk 0.75 -> false alarm
    # Total non-failing machines: 2. False alarms: 1. FAR = 1 / 2 = 0.50
    far_50 = calculate_machine_false_alarm_rate(test_df, "risk_baseline", 0.50, failure_info)
    assert far_50["total_non_failing_machines"] == 2
    assert far_50["false_alarm_machines"] == 1
    assert far_50["false_alarm_rate"] == 0.50

    # Threshold 0.80:
    # Machine 3: max risk 0.30
    # Machine 4: max risk 0.75
    # Neither crosses 0.80. FAR = 0 / 2 = 0.0
    far_80 = calculate_machine_false_alarm_rate(test_df, "risk_baseline", 0.80, failure_info)
    assert far_80["false_alarm_machines"] == 0
    assert far_80["false_alarm_rate"] == 0.0


# ==============================================================================
# 5. LEAD TIME & CAUGHT / MISSED HANDLING
# ==============================================================================

def test_lead_time_calculation(synthetic_predictions):
    test_df = synthetic_predictions[synthetic_predictions["split"] == "test"]
    failure_info = resolve_failure_cycles(test_df)

    summary, per_mach = calculate_lead_time(test_df, "risk_baseline", 0.50, failure_info)

    # Machine 1: fails at 50, alerts at 35 -> lead time = 50 - 35 = 15 cycles (CAUGHT)
    m1 = per_mach[per_mach["machine_id"] == 1].iloc[0]
    assert bool(m1["caught"]) is True
    assert m1["status"] == "CAUGHT"
    assert m1["lead_time"] == 15.0

    # Machine 2: fails at 60, max risk 0.20 -> never crosses 0.50 -> MISSED
    m2 = per_mach[per_mach["machine_id"] == 2].iloc[0]
    assert bool(m2["caught"]) is False
    assert m2["status"] == "MISSED"
    assert pd.isna(m2["lead_time"])

    # Overall summary
    assert summary["machines_caught"] == 1
    assert summary["machines_missed"] == 1
    assert summary["mean_lead_time_cycles"] == 15.0


# ==============================================================================
# 6. THRESHOLD SWEEP
# ==============================================================================

def test_threshold_sweep(synthetic_predictions):
    test_df = synthetic_predictions[synthetic_predictions["split"] == "test"]
    sweep_df = calculate_threshold_sweep(test_df, "risk_baseline", thresholds=[0.20, 0.50, 0.90])

    assert len(sweep_df) == 3
    assert set(sweep_df.columns) == {
        "threshold", "precision", "recall", "f1",
        "false_alarm_rate", "mean_lead_time", "machines_caught", "machines_missed"
    }
    # At 0.90, neither failing machine crosses 0.90 -> 0 caught
    row_90 = sweep_df[sweep_df["threshold"] == 0.90].iloc[0]
    assert row_90["machines_caught"] == 0


# ==============================================================================
# 7. LEAKAGE AUDIT TESTS
# ==============================================================================

def test_leakage_audit_clean(synthetic_predictions, tmp_path):
    test_df = synthetic_predictions[synthetic_predictions["split"] == "test"]
    audit = run_leakage_audit(synthetic_predictions, test_df, project_root=tmp_path, selected_threshold=0.50)

    assert audit["machine_split_leakage"] == "PASS"
    assert audit["test_only_evaluation"] == "PASS"


def test_leakage_audit_machine_overlap(synthetic_predictions, tmp_path):
    # Overlap machine 1 in both train and test
    bad_df = synthetic_predictions.copy()
    bad_df.loc[bad_df["machine_id"] == 1, "split"] = "train"
    bad_df_with_overlap = pd.concat([synthetic_predictions, bad_df.iloc[:5]], ignore_index=True)

    test_df = bad_df_with_overlap[bad_df_with_overlap["split"] == "test"]
    audit = run_leakage_audit(bad_df_with_overlap, test_df, project_root=tmp_path)
    assert audit["machine_split_leakage"] == "FAIL"


# ==============================================================================
# 8. END-TO-END PIPELINE RUN
# ==============================================================================

def test_run_evaluation_end_to_end(synthetic_predictions, tmp_path):
    pred_path = tmp_path / "test_preds.parquet"
    out_dir = tmp_path / "outputs"
    synthetic_predictions.to_parquet(pred_path, index=False)

    results = run_evaluation(
        predictions_path=pred_path,
        output_dir=out_dir,
        threshold=0.50,
        dataset_name="SYNTHETIC_TEST",
    )

    # Check output files exist
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metrics" / "metrics.json").exists()
    assert (out_dir / "metrics" / "per_machine_metrics.csv").exists()
    assert (out_dir / "metrics" / "threshold_sweep.csv").exists()
    assert (out_dir / "metrics" / "model_comparison.csv").exists()
    assert (out_dir / "figures" / "confusion_matrix_baseline.png").exists()
    assert (out_dir / "figures" / "pr_curve.png").exists()
    assert (out_dir / "figures" / "threshold_sweep.png").exists()
    assert (out_dir / "figures" / "risk_trajectory.png").exists()
    assert (out_dir / "figures" / "feature_importance.png").exists()
    assert (out_dir / "figures" / "sensor_traces.png").exists()

    # Verify JSON structure
    assert results["selected_threshold"] == 0.50
    assert "baseline" in results["models"]
    assert results["models"]["baseline"]["machines_caught"] == 1
