# Machine Failure Early Warning System

Predictive-maintenance pipeline for NASA C-MAPSS turbofan engines. It converts
run-to-failure sensor trajectories into 30-cycle windows, predicts imminent
failure, and reports operational lead-time metrics.

## LSTM sequence model

`src/train_lstm.py` trains the ICD-specified sequence model directly on raw
`(N, 30, 21)` sensor windows. Train/test separation and the validation split
are both performed **by engine**, preventing overlapping-window leakage.

The model standardizes every sensor using training-only statistics, then uses
BatchNormalization, LSTM(64), dropout, LSTM(32), dropout, Dense(16), and a
sigmoid risk output. It uses balanced class weights and early-stops on
validation PR-AUC.

## Data and training

The supplied C-MAPSS source files are expected in `CMaps/`. FD001 is the
default because its 100 training trajectories run through known failure cycles.

```bash
pip install -r requirements.txt
python -m src.data_load --dataset FD001
python -m src.windowing
python -m src.train_lstm
```

The LSTM path produces raw/window artifacts, `models/lstm.keras`,
`models/scaler.joblib`, `models/lstm_training_info.json`, and
`outputs/predictions.parquet` with `risk_lstm`.

## Evaluation

Evaluation runs on held-out test engines and reports precision, recall, F1,
PR-AUC, ROC-AUC, machine-level false-alarm rate, and early-warning lead time.
A threshold sweep from 0.10 to 0.90 supports practical operating decisions.

```bash
python -m src.evaluate
python -m pytest tests/test_evaluate.py -v
```

Evaluation artifacts include `outputs/metrics.json`, per-machine and threshold
CSV reports, and plots under `outputs/figures/`.
