# Machine Failure Early-Warning System

## LSTM sequence model

`src/train_lstm.py` trains the ICD-specified sequence classifier directly on
the raw `(windows, 30, 21)` sensor windows.  It expects the window handoff
files produced by `src.windowing`:

- `data/interim/windows.npz`, containing `X` and `y`.
- `data/interim/window_index.parquet`, containing the matching `window_id`,
  `machine_id`, `label`, and machine-level `split` columns.

From the project directory, install dependencies and run:

```bash
pip install -r requirements.txt
python -m src.train_lstm
```

The trainer refuses invalid handoffs, including row-count mismatches, label
mismatches, non-finite sensor values, and machines appearing in both train and
test. It creates a validation set by **machine**, fits per-sensor normalization
only on training readings, uses balanced class weights, and early-stops on
validation PR-AUC.

Artifacts:

- `models/lstm.keras` - best LSTM checkpoint.
- `models/scaler.joblib` - train-only per-sensor mean and scale used at serving.
- `models/lstm_training_info.json` - run metadata and best validation PR-AUC.
- `outputs/predictions.parquet` - `risk_lstm` aligned by `window_id`, while
  preserving a pre-existing `risk_baseline` column.

## Supplied C-MAPSS dataset

The supplied archive is unpacked to `CMaps/`. The default dataset is NASA
C-MAPSS **FD001**: 100 run-to-failure engine trajectories under one operating
condition and one fault mode. `train_FD001.txt` is intentionally used instead
of `test_FD001.txt`, because each training trajectory has a known final failure
cycle. The loader derives `rul` and `failure_event` from that cycle.

Run the complete LSTM data path from the project directory:

```bash
python -m src.data_load --dataset FD001
python -m src.windowing
python -m src.train_lstm
```

This produces an 80/20 train/test split **by engine**, then a machine-level
validation subset within training. No engine occurs in more than one split.
