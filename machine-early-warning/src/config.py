WINDOW_SIZE = 30  # how many readings make one chunk
HORIZON = 30      # "will it fail in the next 30 cycles?"
STRIDE = 1        # move the window forward by 1 each time
RANDOM_SEED = 42
TEST_MACHINE_FRAC = 0.2  # 20% of MACHINES go to test (not 20% of rows!)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ID_COL, TIME_COL, LABEL_COL = "machine_id", "timestamp", "label"

PATHS = {
    "raw": "data/raw/raw_data.parquet",
    "windows": "data/interim/windows.npz",
    "index": "data/interim/window_index.parquet",
    "features": "data/processed/features.parquet",
    "preds": "outputs/predictions.parquet",
    "metrics": "outputs/metrics.json",
}
