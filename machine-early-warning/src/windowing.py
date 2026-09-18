import pandas as pd
import numpy as np
import os
from config import PATHS, WINDOW_SIZE, HORIZON, SENSOR_COLS, RANDOM_SEED, TEST_MACHINE_FRAC

def create_windows():
    if not os.path.exists(PATHS["raw"]):
        print(f"Raw data not found at {PATHS['raw']}. Please run data_load.py first.")
        return
        
    df = pd.read_parquet(PATHS["raw"])
    
    # Sort just in case
    df = df.sort_values(by=["machine_id", "timestamp"])
    
    # Train/test split by machine
    machines = df["machine_id"].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(machines)
    n_test = int(len(machines) * TEST_MACHINE_FRAC)
    test_machines = set(machines[:n_test])
    
    X_list = []
    y_list = []
    index_rows = []
    
    window_id = 0
    for machine_id, group in df.groupby("machine_id"):
        group = group.reset_index(drop=True)
        split = "test" if machine_id in test_machines else "train"
        
        for start_idx in range(len(group) - WINDOW_SIZE + 1):
            end_idx = start_idx + WINDOW_SIZE
            window = group.iloc[start_idx:end_idx]
            
            # Extract features (21 sensors)
            x_window = window[SENSOR_COLS].values.astype(np.float32)
            
            # Label generation: rul at the end of the window <= HORIZON
            rul_at_end = window.iloc[-1]["rul"]
            label = 1 if rul_at_end <= HORIZON else 0
            
            X_list.append(x_window)
            y_list.append(np.int8(label))
            
            index_rows.append({
                "window_id": window_id,
                "machine_id": machine_id,
                "window_start": window.iloc[0]["timestamp"],
                "window_end": window.iloc[-1]["timestamp"],
                "rul_at_end": rul_at_end,
                "label": np.int8(label),
                "split": split
            })
            window_id += 1
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    
    df_index = pd.DataFrame(index_rows)
    df_index["label"] = df_index["label"].astype(np.int8)
    
    # Assert invariants
    assert X.shape[0] == y.shape[0] == len(df_index), "Lengths do not match"
    assert X.shape[1] == WINDOW_SIZE, f"Window size is not {WINDOW_SIZE}"
    assert X.shape[2] == len(SENSOR_COLS), "Sensor columns mismatch"
    assert not df_index.duplicated(["machine_id", "window_start"]).any(), "Duplicate windows found"
    
    # Test split by machine only
    train_machines = set(df_index[df_index["split"] == "train"]["machine_id"])
    test_mac_check = set(df_index[df_index["split"] == "test"]["machine_id"])
    assert len(train_machines.intersection(test_mac_check)) == 0, "Data leakage: machines in both train and test"
    
    os.makedirs(os.path.dirname(PATHS["windows"]), exist_ok=True)
    np.savez(PATHS["windows"], X=X, y=y)
    df_index.to_parquet(PATHS["index"], index=False)
    
    print(f"Windowing complete. Total windows: {len(df_index)}")
    print(f"Label distribution: 0: {sum(y==0)}, 1: {sum(y==1)}")
    print(f"Train machines: {len(train_machines)}, Test machines: {len(test_mac_check)}")

if __name__ == "__main__":
    create_windows()
