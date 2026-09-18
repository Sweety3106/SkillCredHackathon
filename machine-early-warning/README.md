# Machine Failure Early Warning System

A production-grade predictive maintenance and early-warning AI system for industrial turbofan engines using the NASA C-MAPSS dataset (FD001). The system continuously monitors multivariate sensor time series within rolling windows to forecast approaching end-of-life machine failures well in advance.

---

## 1. System Architecture & ICD Pipeline

```
[NASA C-MAPSS FD001]
         │
         ▼
  [data_load.py (P1)] ────────► data/raw/raw_data.parquet (IF-01)
         │
         ▼
  [windowing.py (P1)] ────────► data/interim/windows.npz (N, 30, 21) & window_index.parquet (IF-02)
         │                                       │
         ▼                                       ▼
  [features.py (P2)]                   [train_lstm.py (P3)]
         │                                       │
         ▼                                       ▼
  [train_baseline.py (P2)]             models/lstm.keras & scaler.joblib (IF-04)
         │                                       │
         └───────────────┬───────────────────────┘
                         │
                         ▼
             outputs/predictions.parquet (IF-05)
                         │
                         ▼
              [src/evaluate.py (P4)]
                         │
                         ├─► outputs/metrics.json (IF-08)
                         ├─► outputs/metrics/per_machine_metrics.csv
                         ├─► outputs/metrics/threshold_sweep.csv
                         ├─► outputs/metrics/model_comparison.csv
                         ├─► outputs/figures/*.png (7 presentation plots)
                         └─► slides/DEMO_SCRIPT_AND_SLIDES.md
```

---

## 2. Evaluation Methodology & Core Metrics (P4 Contribution)

Predictive maintenance cannot be evaluated purely on classification accuracy. Normal engine cycles vastly outnumber critical pre-failure cycles (~83% vs ~17%). Our evaluation framework is designed around operational realities:

### 1. Test-Split Isolation & Ground Truth
- **Strict Machine-Level Split**: Engines are partitioned by machine ID (80 engines train/validation, 20 engines held-out test). No engine appears in both splits, preventing window overlap leakage.
- **Ground Truth Failure Cycle**: In NASA C-MAPSS FD001, each test engine runs until failure:
  $$\text{Actual Failure Cycle} = \text{window\_end} + \text{RUL}_{\text{end}}$$

### 2. Lead Time Calculation (Advance Warning Runway)
- For each test engine, the evaluator identifies the **first cycle** where predicted failure risk crosses the operational threshold before failure:
  $$\text{Lead Time} = \text{Actual Failure Cycle} - \text{First Alert Cycle}$$
- **Machines Caught**: $\text{First Alert Cycle} < \text{Actual Failure Cycle}$ ($\text{Lead Time} > 0$).
- **Machines Missed**: Zero alerts raised prior to failure.

### 3. Machine-Level False Alarm Rate (FAR)
- Out of all healthy/non-failing machines, the fraction that ever triggered an alarm:
  $$\text{FAR} = \frac{\text{Total Non-Failing Machines with } \ge 1 \text{ False Alert}}{\text{Total Non-Failing Machines}}$$
  Judges care about asset-level false alarms, not raw window counts.

### 4. 17-Point Threshold Sweep
- Evaluates candidate thresholds from $0.10$ to $0.90$ (step $0.05$) to characterize Precision, Recall, F1, Lead Time, and False Alarm trade-offs.

---

## 3. Real Test Evaluation Results (Held-Out Test Engines)

Evaluated on **3,661 held-out test windows across 20 unseen C-MAPSS engines** at operating threshold $\tau = 0.50$:

| Metric | Score / Value | Interpretation |
|---|---|---|
| **PR-AUC** | **0.9770** | Exceptional minority-class ranking power |
| **ROC-AUC** | **0.9946** | Strong separation between healthy and failing states |
| **Precision** | **82.11%** | 8 out of 10 alert windows correspond to genuine pre-failure |
| **Recall** | **95.48%** | Captures over 95% of imminent failure states |
| **F1 Score** | **0.8829** | Harmonic balance between coverage and alert precision |
| **Mean Lead Time** | **35.95 cycles** | Average advance warning runway across caught engines |
| **Median Lead Time** | **32.50 cycles** | Robust operational lead time |
| **Min / Max Lead Time**| **23 / 60 cycles** | Every engine caught with at least 23 cycles warning |
| **Machines Caught** | **20 / 20 (100%)** | Zero missed engine failures across all test units |
| **Machine False Alarms**| **0.0%** | Zero healthy machine shutdowns |

### Confusion Matrix (Test Split @ 0.50)
- **True Normal (TN)**: 2,912 windows
- **False Alert (FP)**: 129 windows (only 4.2% window-level false positive rate)
- **Missed Imminent (FN)**: 28 windows
- **True Warning (TP)**: 592 windows

---

## 4. Head-to-Head Model Comparison (P2 Baseline vs. P3 LSTM)

Evaluated side-by-side on the identical 3,661 test windows across 20 unseen C-MAPSS engines:

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Machine FAR | Mean Lead Time | Engines Caught |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE (HistGradientBoosting)** | **91.28%** | 91.13% | **0.9120** | 0.9929 | 0.9681 | **0.0%** | 31.35 cycles | **20 / 20 (100%)** |
| **LSTM (Deep Sequence)** | 82.11% | **95.48%** | 0.8829 | **0.9946** | **0.9770** | **0.0%** | **35.95 cycles** | **20 / 20 (100%)** |

### Key Takeaway for Judges:
- **Baseline Advantage**: The tree-based baseline achieves higher overall precision (**91.28% vs 82.11%**) and F1 score (**0.9120**), training in seconds on CPU without GPU dependencies.
- **LSTM Advantage**: The deep LSTM sequence model captures complex temporal dynamics earlier, providing **+4.60 additional cycles of early-warning runway** (35.95 vs 31.35 cycles) and higher recall (**95.48% vs 91.13%**).

---

## 5. Classical ML Baseline & Feature Engineering (P2 Contribution)

### Feature Engineering (`src/features.py`)
- **147 Engineered Features**: For each of the 21 sensors across the 30-cycle window, extracts 7 deterministic statistical indicators:
  1. `mean`: Average sensor intensity over 30 cycles
  2. `std`: Sensor noise and volatility
  3. `min`: Extreme dip
  4. `max`: Peak thermal/pressure stress
  5. `slope`: Linear degradation rate $\left(\frac{\sum (t - \bar{t}) x_t}{\sum (t - \bar{t})^2}\right)$
  6. `delta`: Net drift ($x_{29} - x_0$)
  7. `last`: Most recent sensor reading ($x_{29}$)
- **Vectorized Extraction**: Fully vectorized NumPy implementation processes 17,731 windows in under 0.3 seconds.
- **Leakage Elimination**: `validate_no_target_leakage()` strictly verifies zero presence of RUL, failure flags, or future readings.

### Baseline Model Training (`src/train_baseline.py`)
- **Classifier**: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`.
- **Isolation**: Trained strictly on 64 training engines (11,439 windows), tuned on 16 validation engines (2,631 windows), evaluated on 20 held-out test engines (3,661 windows).
- **Explainability**: Real permutation feature importances saved to `outputs/metrics/feature_importance.csv` and `outputs/feature_importance.json`.
- **Safe Predictions Merging**: Merges `risk_baseline` into `outputs/predictions.parquet` on `window_id`, safely preserving P3's `risk_lstm`.

---

## 6. Bonus Features (ICD Section 8)

### Bonus Feature 1: Alert Deduplication
- **Problem**: In raw sliding windows, an engine in degradation triggers alerts on dozens of successive cycles (average **36.0 alerts per engine**), causing technician alarm fatigue.
- **Solution**: We implement an intelligent cooldown filter (15 cycles cooldown unless risk jumps by $\ge 0.15$).
- **Result**: Alerts per engine drop from **36.0 to 4.3 (88.1% alert reduction)**, delivering concise, actionable notifications to maintenance teams.

### Bonus Feature 5: Financial Cost Model & Economic Optimization
- **Operational Costs**:
  - Cost of 1 False Alert (technician inspection): **₹5,000**
  - Cost of 1 Missed Breakdown (catastrophic plant downtime): **₹200,000**
- **Economic Curve**: The pipeline computes the total financial risk across all 17 thresholds, plotting the economic sweet-spot curve in `outputs/figures/cost_model.png` and establishing that thresholds between $0.45$ and $0.70$ yield the lowest expected operating cost.

---

## 7. Generated Artifacts & Visualizations

All artifacts are persisted under `outputs/`:

```
outputs/
├── metrics.json                         # Primary report matching ICD IF-08 exactly
├── feature_importance.json              # Top sensor drivers for dashboard & API
├── metrics/
│   ├── metrics.json                     # Mirror metrics JSON
│   ├── per_machine_metrics.csv          # Machine-by-machine lead times, alert cycles, and status
│   ├── threshold_sweep.csv              # 17-point threshold sensitivity table
│   ├── model_comparison.csv             # Head-to-head model comparison
│   └── feature_importance.csv           # Permutation feature importances
└── figures/
    ├── confusion_matrix_baseline.png    # Baseline confusion matrix
    ├── confusion_matrix_lstm.png        # LSTM confusion matrix
    ├── pr_curve.png                     # Precision-Recall curve with both models
    ├── threshold_sweep.png              # Precision, Recall, FAR vs Lead Time multi-panel plot
    ├── risk_trajectory.png              # Single-asset timeline (ALERT, FAILURE, shaded runway)
    ├── sensor_traces.png                # Top degradation sensor signals (Sensors 11, 9, 12, 14)
    ├── feature_importance.png           # Leakage-audited top 15 model features
    └── cost_model.png                   # Financial cost curve across operating thresholds
```

---

## 8. How to Run the Pipeline

### 1. Prerequisites
```bash
pip install -r requirements.txt
pip install pytest
```

### 2. Run Data Processing & Training
```bash
python -m src.data_load          # P1: Load C-MAPSS data
python -m src.windowing          # P1: Build 30-cycle sliding windows
python -m src.features           # P2: Extract 147 statistical features
python -m src.train_baseline     # P2: Train HistGradientBoosting baseline
python -m src.train_lstm         # P3: Train LSTM sequence model
```
```

### 3. Run Evaluation Pipeline (P4)
```bash
# Evaluates on real test predictions, runs leakage audit, generates all figures and metrics.json
python -m src.evaluate --predictions outputs/predictions.parquet --threshold 0.50
```

### 4. Run Automated Test Suite
```bash
python -m pytest tests/test_evaluate.py -v
```

---

## 7. Demo & Presentation Resources

- **Spoken Demo Script**: [`slides/DEMO_SCRIPT_AND_SLIDES.md`](slides/DEMO_SCRIPT_AND_SLIDES.md) contains the exact 4-minute pitch script with timing markers, slide contents, and a cheat sheet for judge questions.
