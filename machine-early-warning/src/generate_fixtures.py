import os
import json
import numpy as np
import pandas as pd

def generate_fixtures():
    os.makedirs("data/fixtures", exist_ok=True)
    
    # 1. IF-01 raw_data.parquet
    # 3 machines, 60 cycles each, 21 sensors, one machine failing at the end
    n_machines = 3
    n_cycles = 60
    n_sensors = 21
    
    rows = []
    for m in range(1, n_machines + 1):
        for c in range(1, n_cycles + 1):
            rul = n_cycles - c
            failure_event = 1 if (m == 1 and rul == 0) else 0 # machine 1 fails
            row = {
                "machine_id": m,
                "timestamp": c,
                "rul": rul,
                "failure_event": failure_event
            }
            for s in range(1, n_sensors + 1):
                row[f"sensor_{s}"] = np.random.normal(500, 10)
            rows.append(row)
            
    df_raw = pd.DataFrame(rows)
    df_raw.to_parquet("data/fixtures/raw_data.parquet", index=False)
    
    # 2. IF-02 windows.npz and window_index.parquet
    n_windows = 150
    X = np.random.randn(n_windows, 30, 21).astype(np.float32)
    y = np.random.randint(0, 2, size=n_windows).astype(np.int8)
    np.savez("data/fixtures/windows.npz", X=X, y=y)
    
    idx_rows = []
    for i in range(n_windows):
        machine_id = (i // 50) + 1
        window_start = (i % 50) + 1
        window_end = window_start + 29
        rul_at_end = max(0, 60 - window_end)
        label = 1 if rul_at_end <= 30 else 0
        split = "test" if machine_id == 3 else "train"
        idx_rows.append({
            "window_id": i,
            "machine_id": machine_id,
            "window_start": window_start,
            "window_end": window_end,
            "rul_at_end": rul_at_end,
            "label": label,
            "split": split
        })
    df_idx = pd.DataFrame(idx_rows)
    df_idx["label"] = df_idx["label"].astype(np.int8)
    df_idx.to_parquet("data/fixtures/window_index.parquet", index=False)
    
    # 3. IF-03 features.parquet
    feat_rows = []
    for i in range(n_windows):
        row = {"window_id": i}
        for s in range(1, 22):
            row[f"sensor_{s}_mean"] = np.random.normal(500, 5)
            row[f"sensor_{s}_std"] = np.random.uniform(0, 2)
            row[f"sensor_{s}_min"] = np.random.normal(490, 5)
            row[f"sensor_{s}_max"] = np.random.normal(510, 5)
            row[f"sensor_{s}_slope"] = np.random.normal(0, 0.1)
            row[f"sensor_{s}_delta"] = np.random.normal(0, 2)
            row[f"sensor_{s}_last"] = np.random.normal(500, 5)
        feat_rows.append(row)
    df_feat = pd.DataFrame(feat_rows)
    df_feat.to_parquet("data/fixtures/features.parquet", index=False)
    
    # 4. IF-05 predictions.parquet
    pred_rows = []
    for i in range(n_windows):
        pred_rows.append({
            "window_id": i,
            "machine_id": df_idx.iloc[i]["machine_id"],
            "window_end": df_idx.iloc[i]["window_end"],
            "label": int(df_idx.iloc[i]["label"]),
            "risk_baseline": float(np.random.uniform(0, 1)),
            "risk_lstm": float(np.random.uniform(0, 1)),
            "split": df_idx.iloc[i]["split"]
        })
    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_parquet("data/fixtures/predictions.parquet", index=False)
    
    # 5. IF-06 API JSONs
    health = {"status": "ok", "models_loaded": ["baseline", "lstm"]}
    with open("data/fixtures/health.json", "w") as f: json.dump(health, f)
        
    machines = {"machines": [
        {"machine_id": 47, "current_risk": 0.84, "risk_trend": "rising", "priority_rank": 1, "status": "critical", "cycles_observed": 192, "predicted_rul": 14}
    ]}
    with open("data/fixtures/machines.json", "w") as f: json.dump(machines, f)
        
    timeline = {
        "machine_id": 47, "threshold": 0.5, "alert_at_cycle": 174, "actual_failure_cycle": 192, "lead_time": 18,
        "series": [
            {"cycle": 100, "risk_baseline": 0.11, "risk_lstm": 0.09, "label": 0},
            {"cycle": 101, "risk_baseline": 0.13, "risk_lstm": 0.12, "label": 0}
        ]
    }
    with open("data/fixtures/machine_47_timeline.json", "w") as f: json.dump(timeline, f)
        
    sensors = {
        "machine_id": 47,
        "top_drivers": [
            {"sensor": "sensor_11", "importance": 0.23, "direction": "up", "series": [{"cycle": 100, "value": 47.2}, {"cycle": 101, "value": 47.6}]}
        ]
    }
    with open("data/fixtures/machine_47_sensors.json", "w") as f: json.dump(sensors, f)
        
    metrics = {
        "threshold": 0.5, "horizon": 30,
        "models": {
            "baseline": {
                "precision": 0.81, "recall": 0.74, "f1": 0.77, "roc_auc": 0.93, "pr_auc": 0.79,
                "false_alarm_rate": 0.06, "mean_lead_time_cycles": 21.4,
                "machines_caught": 17, "machines_missed": 3,
                "confusion_matrix": [[820, 40], [55, 160]]
            },
            "lstm": { "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0, "pr_auc": 0, "false_alarm_rate": 0, "mean_lead_time_cycles": 0, "machines_caught": 0, "machines_missed": 0, "confusion_matrix": [[0,0],[0,0]] }
        },
        "per_machine": [
            {"machine_id": 47, "alert_cycle": 174, "failure_cycle": 192, "lead_time": 18, "caught": True}
        ]
    }
    with open("data/fixtures/metrics.json", "w") as f: json.dump(metrics, f)

if __name__ == "__main__":
    generate_fixtures()
    print("Fixtures generated successfully.")
