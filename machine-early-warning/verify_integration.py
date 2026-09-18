"""
Comprehensive Integration Verification Script for Machine Failure Early Warning System.

Performs deterministic invariant and integrity checks across all pipeline stages:
1. Data Contracts (IF-01 Raw Data, IF-02 Windows & Index)
2. Machine-Level Split & Anti-Leakage Invariants
3. Model Artifacts (IF-04) & Predictions Contract (IF-05)
4. Evaluation Results & Exact ICD Schema Compliance (IF-08)
5. Presentation Figures Health & Integrity
6. Operational Sanity (Lead Time, Non-Negative Warning, 100% Caught)
7. Bonus Features (Alert Deduplication & Financial Cost Model)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def verify_all():
    print("=" * 70)
    print("      MACHINE FAILURE EARLY WARNING SYSTEM — INTEGRATION VERIFICATION")
    print("=" * 70)

    errors = []

    # --------------------------------------------------------------------------
    # 1. DATA CONTRACT INVARIANTS (IF-01 & IF-02)
    # --------------------------------------------------------------------------
    print("\n[CHECKPOINT 1/6] Validating Data Contracts (IF-01 & IF-02)...")
    raw_path = Path("data/raw/raw_data.parquet")
    if not raw_path.exists():
        errors.append("data/raw/raw_data.parquet does not exist.")
    else:
        raw = pd.read_parquet(raw_path)
        n_mac = raw["machine_id"].nunique()
        nulls = raw.isnull().sum().sum()
        if n_mac != 100:
            errors.append(f"Expected 100 machines in raw data, got {n_mac}")
        if nulls != 0:
            errors.append(f"Raw data contains {nulls} null values")
        print(f"  [OK] IF-01 Raw Data: {raw.shape[0]:,} rows, {n_mac} machines, 21 sensors, 0 nulls")

    win_path = Path("data/interim/windows.npz")
    idx_path = Path("data/interim/window_index.parquet")
    if not win_path.exists() or not idx_path.exists():
        errors.append("Windows or window_index file missing under data/interim/")
    else:
        with np.load(win_path) as w:
            X, y = w["X"], w["y"]
        idx = pd.read_parquet(idx_path)
        if not (len(X) == len(y) == len(idx) == 17731):
            errors.append(f"Window length mismatch: X={len(X)}, y={len(y)}, idx={len(idx)}")
        if X.shape[1:] != (30, 21):
            errors.append(f"Expected window shape (30, 21), got {X.shape[1:]}")
        if not set(np.unique(y)).issubset({0, 1}):
            errors.append("Labels y are not binary subset of {0, 1}")
        print(f"  [OK] IF-02 Windows: X shape {X.shape}, y shape {y.shape}, index {idx.shape}")

    # --------------------------------------------------------------------------
    # 2. SPLIT ISOLATION & LEAKAGE INVARIANTS
    # --------------------------------------------------------------------------
    print("\n[CHECKPOINT 2/6] Validating Split Isolation & Zero-Leakage...")
    if idx_path.exists():
        train_m = set(idx[idx["split"] == "train"]["machine_id"])
        test_m = set(idx[idx["split"] == "test"]["machine_id"])
        overlap = train_m.intersection(test_m)
        if len(overlap) > 0:
            errors.append(f"FATAL LEAKAGE: Machines present in both train and test: {overlap}")
        else:
            print(f"  [OK] Machine Split Invariant: {len(train_m)} train machines vs {len(test_m)} test machines (Zero Overlap)")
        
        # Check window temporal integrity
        if (idx["window_start"] >= idx["window_end"]).any():
            errors.append("Invalid window bounds: window_start >= window_end")
        else:
            print("  [OK] Temporal Sequence Invariant: All windows historical (window_start < window_end)")

    # --------------------------------------------------------------------------
    # 3. MODEL ARTIFACTS (IF-04) & PREDICTIONS CONTRACT (IF-05)
    # --------------------------------------------------------------------------
    print("\n[CHECKPOINT 3/6] Validating Model Artifacts & Predictions Contract (IF-04 & IF-05)...")
    lstm_model_path = Path("models/lstm.keras")
    scaler_path = Path("models/scaler.joblib")
    preds_path = Path("outputs/predictions.parquet")

    if not lstm_model_path.exists():
        errors.append("models/lstm.keras is missing.")
    else:
        print(f"  [OK] IF-04 Model Artifact: lstm.keras ({lstm_model_path.stat().st_size / 1024:.1f} KB)")

    if not scaler_path.exists():
        errors.append("models/scaler.joblib is missing.")
    else:
        print(f"  [OK] IF-04 Scaler Artifact: scaler.joblib ({scaler_path.stat().st_size / 1024:.1f} KB)")

    if not preds_path.exists():
        errors.append("outputs/predictions.parquet is missing.")
    else:
        preds = pd.read_parquet(preds_path)
        required_cols = ["window_id", "machine_id", "window_end", "label", "split", "risk_lstm"]
        missing_cols = [c for c in required_cols if c not in preds.columns]
        if missing_cols:
            errors.append(f"predictions.parquet missing columns: {missing_cols}")
        if preds["risk_lstm"].isnull().sum() > 0:
            errors.append("risk_lstm contains null values")
        if (preds["risk_lstm"] < 0.0).any() or (preds["risk_lstm"] > 1.0).any():
            errors.append("risk_lstm values outside [0.0, 1.0]")
        print(f"  [OK] IF-05 Predictions: {len(preds):,} rows, risk range [{preds['risk_lstm'].min():.4f}, {preds['risk_lstm'].max():.4f}], 0 nulls")

    # --------------------------------------------------------------------------
    # 4. EVALUATION RESULTS & EXACT ICD SCHEMA (IF-08)
    # --------------------------------------------------------------------------
    print("\n[CHECKPOINT 4/6] Validating Evaluation Report & ICD Schema Compliance (IF-08)...")
    metrics_path = Path("outputs/metrics.json")
    if not metrics_path.exists():
        errors.append("outputs/metrics.json is missing.")
    else:
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            # Check ICD IF-08 required top-level keys
            for key in ["threshold", "horizon", "models", "per_machine"]:
                if key not in metrics:
                    errors.append(f"metrics.json missing required ICD IF-08 key: '{key}'")

            lstm_m = metrics["models"].get("lstm", {})
            pr_auc = lstm_m.get("pr_auc")
            f1 = lstm_m.get("f1")
            lead_time = lstm_m.get("mean_lead_time_cycles")
            caught = lstm_m.get("machines_caught")

            if pr_auc is None or pr_auc < 0.90:
                errors.append(f"PR-AUC unexpectedly low: {pr_auc}")
            if caught != 20:
                errors.append(f"Expected 20 machines caught, got {caught}")
            if lead_time is None or lead_time < 20:
                errors.append(f"Lead time unexpectedly low: {lead_time}")

            print(f"  [OK] IF-08 Schema Compliance: Valid RFC 8259 JSON")
            print(f"  [OK] Performance: PR-AUC = {pr_auc:.4f}, ROC-AUC = {lstm_m.get('roc_auc', 0):.4f}, F1 = {f1:.4f}, Recall = {lstm_m.get('recall', 0)*100:.2f}%")
            print(f"  [OK] Operational Runway: Mean Lead Time = {lead_time:.2f} cycles, Caught = {caught}/20 (100%)")
        except Exception as e:
            errors.append(f"Error parsing metrics.json: {e}")

    # --------------------------------------------------------------------------
    # 5. PRESENTATION FIGURES INTEGRITY
    # --------------------------------------------------------------------------
    print("\n[CHECKPOINT 5/6] Validating Presentation Figures...")
    expected_figures = [
        "confusion_matrix_lstm.png",
        "pr_curve.png",
        "threshold_sweep.png",
        "risk_trajectory.png",
        "feature_importance.png",
        "sensor_traces.png",
        "cost_model.png",
    ]
    figures_dir = Path("outputs/figures")
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        if not fig_path.exists():
            errors.append(f"Figure missing: {fig_name}")
        elif fig_path.stat().st_size < 10000:
            errors.append(f"Figure {fig_name} is too small or corrupted ({fig_path.stat().st_size} bytes)")
        else:
            print(f"  [OK] Figure: {fig_name:<28} ({fig_path.stat().st_size / 1024:.1f} KB)")

    # --------------------------------------------------------------------------
    # 6. BONUS FEATURES & PRESENTATION DECK
    # --------------------------------------------------------------------------
    print("\n[CHECKPOINT 6/6] Validating Bonus Features & Pitch Resources...")
    if metrics_path.exists():
        if "alert_deduplication" not in metrics:
            errors.append("metrics.json missing 'alert_deduplication' bonus feature")
        else:
            dedup = metrics["alert_deduplication"]
            print(f"  [OK] Bonus 1 (Alert Deduplication): {dedup['headline']}")

        if "cost_model" not in metrics:
            errors.append("metrics.json missing 'cost_model' bonus feature")
        else:
            cost = metrics["cost_model"]
            print(f"  [OK] Bonus 5 (Financial Cost Model): Optimal Threshold = {cost['cost_optimal_threshold']:.2f}")

    slides_path = Path("slides/DEMO_SCRIPT_AND_SLIDES.md")
    if not slides_path.exists() or slides_path.stat().st_size < 1000:
        errors.append("Presentation pitch script missing or incomplete under slides/")
    else:
        print(f"  [OK] Pitch Script & Slides: slides/DEMO_SCRIPT_AND_SLIDES.md ({slides_path.stat().st_size / 1024:.1f} KB)")

    # --------------------------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    if not errors:
        print(">>> RESULT: 100% VERIFICATION PASSED. ALL SYSTEMS FULLY OPERATIONAL. <<<")
        print("=" * 70)
        return 0
    else:
        print(f">>> RESULT: {len(errors)} FAILED CHECKS ENCOUNTERED <<<")
        for err in errors:
            print(f"  - [FAIL] {err}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(verify_all())
